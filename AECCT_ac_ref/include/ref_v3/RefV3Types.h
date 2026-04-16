#pragma once

#include "ac_fixed.h"
#include "ac_int.h"
#include "ac_std_float.h"

namespace aecct_ref {
namespace ref_v3 {

typedef ac_ieee_float<binary16> refv3_fp_t;
typedef ac_int<1, false> refv3_bit1_t;
typedef refv3_bit1_t bit1_t;

struct RefV3TernaryLinearParams {
  const refv3_fp_t* weight_fp;
  const refv3_fp_t* bias_fp;
};

template <typename ScalarT>
static inline refv3_fp_t refv3_fp_from_scalar(ScalarT v) {
  return refv3_fp_t(v);
}

static inline RefV3TernaryLinearParams refv3_make_ternary_linear_params(
  const refv3_fp_t* weight,
  const refv3_fp_t* bias) {
  RefV3TernaryLinearParams params;
  params.weight_fp = weight;
  params.bias_fp = bias;
  return params;
}

static inline refv3_fp_t refv3_abs_fp(refv3_fp_t x) {
  const refv3_fp_t zero(0.0f);
  return (x < zero) ? (zero - x) : x;
}

static inline ac_int<8, true> refv3_quantize_int8_local(refv3_fp_t x, refv3_fp_t s_x) {
  const refv3_fp_t scaled = x * s_x;
  const refv3_fp_t sat_hi(127.0f);
  const refv3_fp_t sat_lo(-127.0f);
  const refv3_fp_t zero(0.0f);
  const refv3_fp_t half(0.5f);

  if (scaled >= sat_hi) {
    return ac_int<8, true>(127);
  }
  if (scaled <= sat_lo) {
    return ac_int<8, true>(-127);
  }

  const refv3_fp_t rounded = (scaled >= zero) ? (scaled + half) : (scaled - half);
  return ac_int<8, true>(rounded.convert_to_int());
}

namespace refv3_boundary_detail {

static inline ac_int<2, true> decode_ternary_weight_sign_i2(refv3_fp_t w) {
  const refv3_fp_t one(1.0f);
  const refv3_fp_t neg_one(-1.0f);
  const refv3_fp_t zero(0.0f);
  const refv3_fp_t pos_half(0.5f);
  const refv3_fp_t neg_half(-0.5f);

  if (w == one) return ac_int<2, true>(1);
  if (w == neg_one) return ac_int<2, true>(-1);
  if (w == zero) return ac_int<2, true>(0);
  if (w >= pos_half) return ac_int<2, true>(1);
  if (w <= neg_half) return ac_int<2, true>(-1);
  return ac_int<2, true>(0);
}

} // namespace refv3_boundary_detail

static inline ac_int<2, true> refv3_ternary_weight_sign_at(
  const RefV3TernaryLinearParams& params,
  int idx) {
  return refv3_boundary_detail::decode_ternary_weight_sign_i2(params.weight_fp[idx]);
}

static inline refv3_fp_t refv3_linear_weight_fp_at(
  const RefV3TernaryLinearParams& params,
  int idx) {
  return params.weight_fp[idx];
}

static inline refv3_fp_t refv3_linear_bias_fp_at(
  const RefV3TernaryLinearParams& params,
  int idx) {
  return params.bias_fp[idx];
}

} // namespace ref_v3
} // namespace aecct_ref
