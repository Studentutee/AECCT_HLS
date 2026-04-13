#ifndef AECCT_REF_SOFTMAX_APPROX_H
#define AECCT_REF_SOFTMAX_APPROX_H

#include "RefTypes.h"
#include <cmath>
#include "RefSoftmaxExpMode.h"
#include "SoftmaxApproxLutData.h"

namespace aecct_ref {

typedef ref_fp16_t ref_softmax_fp_t;

static const int REF_SOFTMAX_NEG_T = 12;
static const int REF_SOFTMAX_EXP_LUT_SIZE = 256;
static const int REF_SOFTMAX_RCP_LUT_SIZE = 256;
static const double REF_SOFTMAX_SUMEXP_MAX = 256.0;
static const double REF_SOFTMAX_EXP_IDX_SCALE = 21.25;
static const double REF_SOFTMAX_RCP_IDX_SCALE = 1.0;
static const double REF_SOFTMAX_EPS = 1.0e-6;

template <typename FloatT>
static inline FloatT ref_softmax_clamp_x(FloatT x) {
  FloatT xf = x;
  if (xf > FloatT(0.0)) xf = FloatT(0.0);
  if (xf < FloatT(-12.0)) xf = FloatT(-12.0);
  return xf;
}

template <typename FloatT>
static inline int ref_softmax_exp_idx(FloatT x_clamped) {
  const double mag = -x_clamped.to_double();
  int idx = static_cast<int>(std::floor((mag * REF_SOFTMAX_EXP_IDX_SCALE) + 0.5));
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
  const double idxf_raw = (-ref_softmax_clamp_x(x).to_double()) * REF_SOFTMAX_EXP_IDX_SCALE;
  double idxf = idxf_raw;
  if (idxf < 0.0) idxf = 0.0;
  if (idxf > static_cast<double>(last_idx)) idxf = static_cast<double>(last_idx);
  int idx_lo = static_cast<int>(std::floor(idxf));
  if (idx_lo < 0) idx_lo = 0;
  if (idx_lo > last_idx) idx_lo = last_idx;
  int idx_hi = idx_lo + 1;
  if (idx_hi > last_idx) idx_hi = last_idx;
  const FloatT y_lo = FloatT(g_ref_softmax_exp_lut[idx_lo]);
  if (idx_hi == idx_lo) return y_lo;
  const FloatT y_hi = FloatT(g_ref_softmax_exp_lut[idx_hi]);
  const FloatT frac = FloatT(idxf - static_cast<double>(idx_lo));
  return y_lo + ((y_hi - y_lo) * frac);
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
  double sf = s.to_double();
  if (sf < 1.0) sf = 1.0;
  if (sf > REF_SOFTMAX_SUMEXP_MAX) sf = REF_SOFTMAX_SUMEXP_MAX;
  int idx = static_cast<int>(std::floor(((sf - 1.0) * REF_SOFTMAX_RCP_IDX_SCALE) + 0.5));
  if (idx < 0) idx = 0;
  if (idx >= REF_SOFTMAX_RCP_LUT_SIZE) idx = REF_SOFTMAX_RCP_LUT_SIZE - 1;
  return idx;
}

template <typename FloatT>
static inline FloatT ref_softmax_rcp_lut(FloatT sumexp) {
  FloatT s = sumexp;
  if (s < FloatT(REF_SOFTMAX_EPS)) s = FloatT(REF_SOFTMAX_EPS);
  const int idx = ref_softmax_rcp_idx(s);
  const FloatT inv0 = FloatT(g_ref_softmax_rcp_lut[idx]);
  const FloatT inv1 = inv0 * (FloatT(2.0) - s * inv0);
  return inv1;
}

} // namespace aecct_ref

#endif
