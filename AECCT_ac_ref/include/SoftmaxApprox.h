#ifndef AECCT_REF_SOFTMAX_APPROX_H
#define AECCT_REF_SOFTMAX_APPROX_H

#include "ac_std_float.h"
#include "SoftmaxApproxLutData.h"

namespace aecct_ref {

typedef ac_ieee_float<binary32> ref_softmax_fp32_t;

static const int REF_SOFTMAX_NEG_T = 12;
static const int REF_SOFTMAX_EXP_LUT_SIZE = 256;
static const int REF_SOFTMAX_RCP_LUT_SIZE = 256;
static const float REF_SOFTMAX_SUMEXP_MAX = 256.0f;
static const float REF_SOFTMAX_EXP_IDX_SCALE = 21.25f;
static const float REF_SOFTMAX_RCP_IDX_SCALE = 1.0f;
static const float REF_SOFTMAX_EPS = 1.0e-6f;

static inline ref_softmax_fp32_t ref_softmax_clamp_x(ref_softmax_fp32_t x) {
  ref_softmax_fp32_t neg_t = ref_softmax_fp32_t(-12.0f);
  if (x > ref_softmax_fp32_t(0.0f)) x = ref_softmax_fp32_t(0.0f);
  if (x < neg_t) x = neg_t;
  return x;
}

static inline int ref_softmax_exp_idx(ref_softmax_fp32_t x_clamped) {
  ref_softmax_fp32_t mag = ref_softmax_fp32_t(0.0f) - x_clamped;
  ref_softmax_fp32_t idxf = mag * ref_softmax_fp32_t(REF_SOFTMAX_EXP_IDX_SCALE);
  int idx = (idxf + ref_softmax_fp32_t(0.5f)).to_float();
  if (idx < 0) idx = 0;
  if (idx >= REF_SOFTMAX_EXP_LUT_SIZE) idx = REF_SOFTMAX_EXP_LUT_SIZE - 1;
  return idx;
}

static inline ref_softmax_fp32_t ref_softmax_exp_lut(ref_softmax_fp32_t x) {
  ref_softmax_fp32_t xc = ref_softmax_clamp_x(x);
  int idx = ref_softmax_exp_idx(xc);
  return ref_softmax_fp32_t(g_ref_softmax_exp_lut[idx]);
}

static inline int ref_softmax_rcp_idx(ref_softmax_fp32_t s) {
  ref_softmax_fp32_t one = ref_softmax_fp32_t(1.0f);
  ref_softmax_fp32_t maxv = ref_softmax_fp32_t(REF_SOFTMAX_SUMEXP_MAX);
  if (s < one) s = one;
  if (s > maxv) s = maxv;
  ref_softmax_fp32_t idxf = (s - one) * ref_softmax_fp32_t(REF_SOFTMAX_RCP_IDX_SCALE);
  int idx = (idxf + ref_softmax_fp32_t(0.5f)).to_float();
  if (idx < 0) idx = 0;
  if (idx >= REF_SOFTMAX_RCP_LUT_SIZE) idx = REF_SOFTMAX_RCP_LUT_SIZE - 1;
  return idx;
}

static inline ref_softmax_fp32_t ref_softmax_rcp_lut(ref_softmax_fp32_t sumexp) {
  if (sumexp < ref_softmax_fp32_t(REF_SOFTMAX_EPS)) {
    sumexp = ref_softmax_fp32_t(REF_SOFTMAX_EPS);
  }
  int idx = ref_softmax_rcp_idx(sumexp);
  ref_softmax_fp32_t inv0 = ref_softmax_fp32_t(g_ref_softmax_rcp_lut[idx]);
  // One-step Newton refinement: inv = inv0 * (2 - s*inv0)
  ref_softmax_fp32_t inv1 = inv0 * (ref_softmax_fp32_t(2.0f) - sumexp * inv0);
  return inv1;
}

} // namespace aecct_ref

#endif
