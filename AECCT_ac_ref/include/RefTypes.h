#pragma once
#include "ac_fixed.h"
#include "ac_int.h"

// Keep ALL quant formats here (single source of truth for the ref harness).
// Edit these typedefs when you sweep bit-widths.

namespace aecct_ref {

// Activation (example): signed, 16-bit total, 4 integer bits (incl sign).
typedef ac_fixed<16, 4, true, AC_RND_CONV, AC_SAT_SYM> act_t;

// Weight storage (example): signed, 32-bit total, 16 integer bits (incl sign).
typedef ac_fixed<32, 16, true, AC_RND_CONV, AC_SAT_SYM> w_t;

// Accumulator (example): wider to reduce overflow in MAC.
typedef ac_fixed<48, 20, true, AC_RND_CONV, AC_SAT_SYM> acc_t;

// Output logits type (example). You may choose acc_t directly.
typedef acc_t logit_t;

// Binary prediction
typedef ac_int<1, false> bit1_t;

static inline act_t act_from_double(double x) {
  // Quantization happens here (destination type defines rounding/overflow).
  return act_t(x);
}
static inline w_t w_from_double(double x) {
  return w_t(x);
}
static inline double to_double(const logit_t &x) {
  return x.to_double();
}

} // namespace aecct_ref
