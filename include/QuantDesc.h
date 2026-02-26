#pragma once
// QuantDesc.h
// M15 fixed-point datatypes and conversion helpers (single source of truth)

#include <ac_fixed.h>
#include <cstdint>

#include "AecctTypes.h"

namespace aecct {

    typedef ac_fixed<16, 6, true, AC_RND, AC_SAT> quant_act_t;
    typedef ac_fixed<16, 6, true, AC_RND, AC_SAT> quant_w_t;
    typedef ac_fixed<18, 2, false, AC_RND, AC_SAT> quant_prob_t;
    typedef ac_fixed<32, 12, true, AC_RND, AC_SAT> quant_acc_t;

    static inline float quant_bits_to_f32(u32_t w) {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.u = (uint32_t)w.to_uint();
        return cvt.f;
    }

    static inline u32_t quant_f32_to_bits(float f) {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.f = f;
        return (u32_t)cvt.u;
    }

    static inline quant_act_t quant_act_from_bits(u32_t w) {
        return quant_act_t(quant_bits_to_f32(w));
    }

    static inline u32_t quant_bits_from_acc(quant_acc_t v) {
        return quant_f32_to_bits((float)v.to_double());
    }

    static inline u32_t quant_bits_from_act(quant_act_t v) {
        return quant_f32_to_bits((float)v.to_double());
    }

} // namespace aecct
