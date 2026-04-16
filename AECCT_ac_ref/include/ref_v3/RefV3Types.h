#pragma once

#include "ac_fixed.h"
#include "ac_int.h"
#include "ac_std_float.h"

namespace aecct_ref {
namespace ref_v3 {

typedef ac_ieee_float<binary16> refv3_fp_t;
typedef ac_int<1, false> refv3_bit1_t;
typedef refv3_bit1_t bit1_t;

static inline refv3_fp_t refv3_fp_from_float(float v) {
  return refv3_fp_t(v);
}

static inline float refv3_fp_to_float(refv3_fp_t v) {
  return v.to_float();
}

} // namespace ref_v3
} // namespace aecct_ref
