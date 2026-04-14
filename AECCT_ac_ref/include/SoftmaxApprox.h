#ifndef AECCT_REF_SOFTMAX_APPROX_H
#define AECCT_REF_SOFTMAX_APPROX_H

#include "RefTypes.h"
#include <cmath>
#include "RefSoftmaxExpMode.h"
#include "SoftmaxApproxLutData.h"

namespace aecct_ref {

typedef ref_fp32_t ref_softmax_fp32_t;

static const int REF_SOFTMAX_NEG_T = 12;
static const int REF_SOFTMAX_EXP_LUT_SIZE = 256;
static const int REF_SOFTMAX_RCP_LUT_SIZE = 256;
static const float REF_SOFTMAX_SUMEXP_MAX = 256.0f;
static const float REF_SOFTMAX_EXP_IDX_SCALE = 21.25f;
static const float REF_SOFTMAX_RCP_IDX_SCALE = 1.0f;
static const float REF_SOFTMAX_EPS = 1.0e-6f;

template <typename FloatT>
static inline FloatT ref_softmax_clamp_x(FloatT x) {
  float xf = x.to_float();
  if (xf > 0.0f) xf = 0.0f;
  if (xf < -12.0f) xf = -12.0f;
  return FloatT(xf);
}

template <typename FloatT>
static inline int ref_softmax_exp_idx(FloatT x_clamped) {
  const float mag = -x_clamped.to_float();
  int idx = static_cast<int>(std::floor((mag * REF_SOFTMAX_EXP_IDX_SCALE) + 0.5f));
  if (idx < 0) idx = 0;
  if (idx >= REF_SOFTMAX_EXP_LUT_SIZE) idx = REF_SOFTMAX_EXP_LUT_SIZE - 1;
  return idx;
}

template <typename FloatT>
static inline FloatT ref_softmax_exp_lut(FloatT x) {
  const FloatT xc = ref_softmax_clamp_x(x);
  const int idx = ref_softmax_exp_idx(xc);
  return FloatT(g_ref_softmax_exp_lut[idx]);
}

template <typename FloatT>
static inline FloatT ref_softmax_exp_lerp_lut(FloatT x) {
  const int last_idx = REF_SOFTMAX_EXP_LUT_SIZE - 1;
  float xc = ref_softmax_clamp_x(x).to_float();
  float mag = -xc;
  float idxf = mag * REF_SOFTMAX_EXP_IDX_SCALE;
  if (idxf < 0.0f) idxf = 0.0f;
  if (idxf > static_cast<float>(last_idx)) idxf = static_cast<float>(last_idx);
  int idx_lo = static_cast<int>(std::floor(idxf));
  if (idx_lo < 0) idx_lo = 0;
  if (idx_lo > last_idx) idx_lo = last_idx;
  int idx_hi = idx_lo + 1;
  if (idx_hi > last_idx) idx_hi = last_idx;
  const float y_lo = g_ref_softmax_exp_lut[idx_lo];
  if (idx_hi == idx_lo) return FloatT(y_lo);
  const float y_hi = g_ref_softmax_exp_lut[idx_hi];
  const float frac = idxf - static_cast<float>(idx_lo);
  return FloatT(y_lo + ((y_hi - y_lo) * frac));
}

// Leaf-kernel selector only: keep reciprocal LUT, online row-state, and exact path semantics unchanged.
template <typename FloatT>
static inline FloatT ref_softmax_exp_dispatch(
  FloatT x,
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

template <typename FloatT>
static inline int ref_softmax_rcp_idx(FloatT s) {
  float sf = s.to_float();
  if (sf < 1.0f) sf = 1.0f;
  if (sf > REF_SOFTMAX_SUMEXP_MAX) sf = REF_SOFTMAX_SUMEXP_MAX;
  int idx = static_cast<int>(std::floor(((sf - 1.0f) * REF_SOFTMAX_RCP_IDX_SCALE) + 0.5f));
  if (idx < 0) idx = 0;
  if (idx >= REF_SOFTMAX_RCP_LUT_SIZE) idx = REF_SOFTMAX_RCP_LUT_SIZE - 1;
  return idx;
}

template <typename FloatT>
static inline FloatT ref_softmax_rcp_lut(FloatT sumexp) {
  float s = sumexp.to_float();
  if (s < REF_SOFTMAX_EPS) s = REF_SOFTMAX_EPS;
  const int idx = ref_softmax_rcp_idx(FloatT(s));
  const float inv0 = g_ref_softmax_rcp_lut[idx];
  const float inv1 = inv0 * (2.0f - s * inv0);
  return FloatT(inv1);
}

} // namespace aecct_ref

#endif
