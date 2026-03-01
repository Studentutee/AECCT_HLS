#pragma once
// Fixed-point datatypes and fp32 conversion helpers.

#include <ac_fixed.h>

#include "AecctTypes.h"
#include "AecctUtil.h"

namespace aecct {

typedef ac_fixed<16, 6, true, AC_RND, AC_SAT> quant_act_t;
typedef ac_fixed<16, 6, true, AC_RND, AC_SAT> quant_w_t;
typedef ac_fixed<18, 2, false, AC_RND, AC_SAT> quant_prob_t;
typedef ac_fixed<32, 12, true, AC_RND, AC_SAT> quant_acc_t;

static inline fp32_t quant_bits_to_fp32(u32_t w) {
    return fp32_from_bits(w);
}

static inline u32_t quant_fp32_to_bits(const fp32_t& f) {
    return bits_from_fp32(f);
}

static inline quant_act_t quant_act_from_bits(u32_t w) {
    fp32_t f = quant_bits_to_fp32(w);
    return f.template convert_to_ac_fixed<16, 6, true, AC_RND, AC_SAT>(false);
}

static inline u32_t quant_bits_from_acc(quant_acc_t v) {
    fp32_t f(v);
    return quant_fp32_to_bits(f);
}

static inline u32_t quant_bits_from_act(quant_act_t v) {
    fp32_t f(v);
    return quant_fp32_to_bits(f);
}

static inline u32_t quant_bits_from_prob(quant_prob_t v) {
    fp32_t f(v);
    return quant_fp32_to_bits(f);
}

} // namespace aecct
