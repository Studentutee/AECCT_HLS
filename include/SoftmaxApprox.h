#ifndef SOFTMAX_APPROX_H
#define SOFTMAX_APPROX_H

#include "SoftmaxDesc.h"
#include "SoftmaxApproxLutData.h"

static inline softmax_x_t softmax_clamp_x(softmax_x_t x) {
#pragma hls_inline
  const softmax_x_t neg_t = softmax_x_t(-SoftmaxApproxCfg::SOFTMAX_NEG_T);
  if (x > softmax_x_t(0)) x = softmax_x_t(0);
  if (x < neg_t) x = neg_t;
  return x;
}

static inline softmax_idx_t softmax_exp_idx(softmax_x_t x_clamped) {
#pragma hls_inline
  const softmax_x_t mag = softmax_x_t(0) - x_clamped;
  softmax_x_t idx_f = mag * SOFTMAX_EXP_IDX_SCALE;
  int idx_i = (idx_f + softmax_x_t(0.5)).to_int();
  softmax_idx_t idx = softmax_idx_t(idx_i);
  if (idx < 0) idx = 0;
  if (idx > (SoftmaxApproxCfg::EXP_LUT_SIZE - 1)) idx = (SoftmaxApproxCfg::EXP_LUT_SIZE - 1);
  return idx;
}

static inline softmax_exp_t softmax_exp_lut(softmax_x_t x) {
#pragma hls_inline
  softmax_x_t x_c = softmax_clamp_x(x);
  softmax_idx_t idx = softmax_exp_idx(x_c);
  return g_exp_lut[idx];
}

static inline softmax_sum_t softmax_eps() {
#pragma hls_inline
  const int k = SoftmaxApproxCfg::EPS_POW2_NEG;
  softmax_sum_t eps = softmax_sum_t(1);
  eps = eps >> k;
  return eps;
}

static inline softmax_sum_t softmax_sum_floor(softmax_sum_t sumexp) {
#pragma hls_inline
  softmax_sum_t eps = softmax_eps();
  if (sumexp < eps) sumexp = eps;
  return sumexp;
}

static inline softmax_idx_t softmax_rcp_idx(softmax_sum_t sumexp_safe) {
#pragma hls_inline
  const softmax_sum_t one = softmax_sum_t(1);
  const softmax_sum_t maxv = softmax_sum_t(SOFTMAX_SUMEXP_MAX);
  softmax_sum_t s = sumexp_safe;

  if (s < one) s = one;
  if (s > maxv) s = maxv;

  const softmax_sum_t numer = s - one;
  softmax_sum_t idx_f = numer * SOFTMAX_RCP_IDX_SCALE;
  int idx_i = (idx_f + softmax_sum_t(0.5)).to_int();
  softmax_idx_t idx = softmax_idx_t(idx_i);
  if (idx < 0) idx = 0;
  if (idx > (SoftmaxApproxCfg::RCP_LUT_SIZE - 1)) idx = (SoftmaxApproxCfg::RCP_LUT_SIZE - 1);
  return idx;
}

static inline softmax_inv_t softmax_rcp_lut(softmax_sum_t sumexp) {
#pragma hls_inline
  softmax_sum_t s = softmax_sum_floor(sumexp);
  softmax_idx_t idx = softmax_rcp_idx(s);
  softmax_inv_t inv0 = g_rcp_lut[idx];
  // One-step Newton-Raphson refinement: inv = inv0 * (2 - s * inv0)
  softmax_sum_t prod = softmax_sum_t(inv0) * s;
  softmax_sum_t corr = softmax_sum_t(2.0) - prod;
  softmax_inv_t inv1 = softmax_inv_t(softmax_sum_t(inv0) * corr);
  return inv1;
}

template <unsigned MAX_LEN>
static inline void SoftmaxApprox(
    const softmax_score_t scores_in[MAX_LEN],
    softmax_prob_t probs_out[MAX_LEN],
    unsigned len
) {
#pragma hls_inline
  if (len == 0u) {
    return;
  }
  if (len > MAX_LEN) {
    len = MAX_LEN;
  }

  softmax_score_t max_score = scores_in[0];
  for (unsigned i = 1u; i < len; ++i) {
    if (scores_in[i] > max_score) {
      max_score = scores_in[i];
    }
  }

  softmax_exp_t exp_buf[MAX_LEN];
  softmax_sum_t sumexp = softmax_sum_t(0);
  for (unsigned i = 0u; i < len; ++i) {
    softmax_x_t x = softmax_x_t(scores_in[i] - max_score);
    softmax_exp_t e = softmax_exp_lut(x);
    exp_buf[i] = e;
    sumexp += softmax_sum_t(e);
  }

  softmax_inv_t inv_sumexp = softmax_rcp_lut(sumexp);
  for (unsigned i = 0u; i < len; ++i) {
    probs_out[i] = softmax_prob_t(exp_buf[i] * inv_sumexp);
  }
}

#endif // SOFTMAX_APPROX_H
