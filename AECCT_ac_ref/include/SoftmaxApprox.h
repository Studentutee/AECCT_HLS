#ifndef AECCT_REF_SOFTMAX_APPROX_H
#define AECCT_REF_SOFTMAX_APPROX_H

#include "RefTypes.h"
#include "RefSoftmaxExpMode.h"
#include "SoftmaxApproxLutData.h"

namespace aecct_ref {

static const int REF_SOFTMAX_NEG_T = 12;
static const int REF_SOFTMAX_EXP_LUT_SIZE = 256;
static const int REF_SOFTMAX_RCP_LUT_SIZE = 256;
static const ref_fp16_t REF_SOFTMAX_SUMEXP_MAX = ref_fp16_t(256.0f);
static const ref_fp16_t REF_SOFTMAX_EXP_IDX_SCALE = ref_fp16_t(21.25f);
static const ref_fp16_t REF_SOFTMAX_EPS = ref_fp16_t(1.0e-6f);

static inline ref_fp16_t ref_softmax_clamp_x(ref_fp16_t x) {
  if (x > ref_fp16_t(0.0f)) x = ref_fp16_t(0.0f);
  if (x < ref_fp16_t(-12.0f)) x = ref_fp16_t(-12.0f);
  return x;
}

static inline int ref_softmax_exp_idx(ref_fp16_t x_clamped) {
  ref_fp16_t mag = ref_fp16_t(0.0f) - x_clamped;
  ref_fp16_t idx_f = (mag * REF_SOFTMAX_EXP_IDX_SCALE) + ref_fp16_t(0.5f);
  int idx = idx_f.convert_to_int();
  if (idx < 0) idx = 0;
  if (idx >= REF_SOFTMAX_EXP_LUT_SIZE) idx = REF_SOFTMAX_EXP_LUT_SIZE - 1;
  return idx;
}

static inline ref_fp16_t ref_softmax_exp_lut(ref_fp16_t x) {
  const ref_fp16_t xc = ref_softmax_clamp_x(x);
  const int idx = ref_softmax_exp_idx(xc);
  return g_ref_softmax_exp_lut[idx];
}

static inline ref_fp16_t ref_softmax_exp_lerp_lut(ref_fp16_t x) {
  const int last_idx = REF_SOFTMAX_EXP_LUT_SIZE - 1;
  ref_fp16_t idx_f = (ref_fp16_t(0.0f) - ref_softmax_clamp_x(x)) * REF_SOFTMAX_EXP_IDX_SCALE;
  if (idx_f < ref_fp16_t(0.0f)) idx_f = ref_fp16_t(0.0f);
  if (idx_f > ref_fp16_t(last_idx)) idx_f = ref_fp16_t(last_idx);
  int idx_lo = idx_f.convert_to_int();
  if (idx_lo < 0) idx_lo = 0;
  if (idx_lo > last_idx) idx_lo = last_idx;
  int idx_hi = idx_lo + 1;
  if (idx_hi > last_idx) idx_hi = last_idx;
  const ref_fp16_t y_lo = g_ref_softmax_exp_lut[idx_lo];
  if (idx_hi == idx_lo) return y_lo;
  const ref_fp16_t y_hi = g_ref_softmax_exp_lut[idx_hi];
  const ref_fp16_t frac = idx_f - ref_fp16_t(idx_lo);
  return y_lo + ((y_hi - y_lo) * frac);
}

static inline ref_fp16_t ref_softmax_exp_dispatch(
  ref_fp16_t x,
  RefSoftmaxExpMode mode
) {
  switch (mode) {
    case RefSoftmaxExpMode::BASELINE_NEAREST_LUT:
      return ref_softmax_exp_lut(x);
    case RefSoftmaxExpMode::V2_LERP_LUT:
      return ref_softmax_exp_lerp_lut(x);
    case RefSoftmaxExpMode::V3_BASE2_RESERVED:
      return ref_softmax_exp_lut(x);
    default:
      return ref_softmax_exp_lut(x);
  }
}

static inline int ref_softmax_rcp_idx(ref_fp16_t s) {
  if (s < ref_fp16_t(1.0f)) s = ref_fp16_t(1.0f);
  if (s > REF_SOFTMAX_SUMEXP_MAX) s = REF_SOFTMAX_SUMEXP_MAX;
  ref_fp16_t idx_f = (s - ref_fp16_t(1.0f)) + ref_fp16_t(0.5f);
  int idx = idx_f.convert_to_int();
  if (idx < 0) idx = 0;
  if (idx >= REF_SOFTMAX_RCP_LUT_SIZE) idx = REF_SOFTMAX_RCP_LUT_SIZE - 1;
  return idx;
}

static inline ref_fp16_t ref_softmax_rcp_lut(ref_fp16_t sumexp) {
  ref_fp16_t s = sumexp;
  if (s < REF_SOFTMAX_EPS) s = REF_SOFTMAX_EPS;
  const int idx = ref_softmax_rcp_idx(s);
  const ref_fp16_t inv0 = g_ref_softmax_rcp_lut[idx];
  const ref_fp16_t inv1 = inv0 * (ref_fp16_t(2.0f) - s * inv0);
  return inv1;
}

} // namespace aecct_ref

#endif
