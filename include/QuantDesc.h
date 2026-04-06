#pragma once
// Fixed-point datatypes and fp32 conversion helpers.

#include <cstdint>

#include <ac_fixed.h>

#include "AecctTypes.h"
#include "AecctUtil.h"

namespace aecct {

typedef ac_fixed<16, 6, true, AC_RND, AC_SAT> quant_act_t;
typedef ac_fixed<16, 6, true, AC_RND, AC_SAT> quant_w_t;
typedef ac_fixed<18, 2, false, AC_RND, AC_SAT> quant_prob_t;
typedef ac_fixed<32, 12, true, AC_RND, AC_SAT> quant_acc_t;

// Backup profile quant path:
// - activation: INT8
// - ternary weight: {-1,0,+1}
// - accumulator: INT16 (saturating)
typedef ac_int<8, true> quant_act_i8_t;
typedef ac_int<8, true> quant_w_i8_t;
typedef ac_int<16, true> quant_acc_i16_t;

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

static inline quant_act_i8_t quant_act_i8_from_word(u32_t w) {
    quant_act_i8_t v = 0;
    v.set_slc(0, w.template slc<8>(0));
    return v;
}

static inline u32_t quant_word_from_act_i8(quant_act_i8_t v) {
    u32_t w = 0;
    w.set_slc(0, (ac_int<8, false>)v);
    return w;
}

static inline quant_acc_i16_t quant_acc_i16_from_word(u32_t w) {
    quant_acc_i16_t v = 0;
    v.set_slc(0, w.template slc<16>(0));
    return v;
}

static inline u32_t quant_word_from_acc_i16(quant_acc_i16_t v) {
    u32_t w = 0;
    w.set_slc(0, (ac_int<16, false>)v);
    return w;
}

static inline bool quant_decode_ternary_weight_i8(uint32_t code, quant_w_i8_t& out_w) {
    if (code == 0u) {
        out_w = (quant_w_i8_t)0;
        return true;
    }
    if (code == 1u) {
        out_w = (quant_w_i8_t)1;
        return true;
    }
    if (code == 2u) {
        out_w = (quant_w_i8_t)-1;
        return true;
    }
    return false;
}

static inline quant_acc_i16_t quant_acc_i16_saturating_add(quant_acc_i16_t a, quant_acc_i16_t b) {
    const ac_int<17, true> sum = (ac_int<17, true>)a + (ac_int<17, true>)b;
    if (sum > (ac_int<17, true>)32767) {
        return (quant_acc_i16_t)32767;
    }
    if (sum < (ac_int<17, true>)-32768) {
        return (quant_acc_i16_t)-32768;
    }
    return (quant_acc_i16_t)sum;
}

static inline quant_acc_i16_t quant_acc_i16_saturating_madd(
    quant_acc_i16_t acc,
    quant_act_i8_t act,
    quant_w_i8_t w
) {
    const ac_int<16, true> prod = (ac_int<16, true>)act * (ac_int<16, true>)w;
    return quant_acc_i16_saturating_add(acc, (quant_acc_i16_t)prod);
}

static inline bool quant_no_overflow_for_frozen_shape(uint32_t reduction_len) {
    const int32_t worst_abs = (int32_t)reduction_len * 127;
    return worst_abs <= 32767;
}

} // namespace aecct
