#pragma once

#include "ac_fixed.h"
#include "ac_int.h"
#include "ac_std_float.h"

namespace aecct_ref {
namespace ref_v3 {

typedef ac_ieee_float<binary16> refv3_fp_t;
typedef ac_int<1, false> refv3_bit1_t;
typedef refv3_bit1_t bit1_t;

static inline refv3_fp_t refv3_fp_from_double(double v) {
  return refv3_fp_t(v);
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

static inline ac_int<2, true> refv3_decode_ternary_weight_sign_i2_local(double w) {
  if (w == 1.0) return ac_int<2, true>(1);
  if (w == -1.0) return ac_int<2, true>(-1);
  if (w == 0.0) return ac_int<2, true>(0);
  if (w >= 0.5) return ac_int<2, true>(1);
  if (w <= -0.5) return ac_int<2, true>(-1);
  return ac_int<2, true>(0);
}

} // namespace ref_v3
} // namespace aecct_ref
