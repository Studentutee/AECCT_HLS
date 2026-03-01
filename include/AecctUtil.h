#pragma once
// Shared helper utilities.

#include <cstdint>

#include <ac_int.h>
#include <ac_std_float.h>

#include "AecctTypes.h"

namespace aecct {

typedef ac_ieee_float<binary32> fp32_t;

static inline uint32_t align_up_u32(uint32_t x, uint32_t align) {
    if (align == 0u) { return x; }
    uint32_t r = x % align;
    return (r == 0u) ? x : (x + (align - r));
}

static inline bool in_range_u32(uint32_t x, uint32_t lo, uint32_t hi_inclusive) {
    return (x >= lo) && (x <= hi_inclusive);
}

static inline uint32_t mask_u32(unsigned w) {
    if (w >= 32u) { return 0xFFFFFFFFu; }
    if (w == 0u) { return 0u; }
    return (1u << w) - 1u;
}

static inline fp32_t fp32_from_bits(const u32_t& bits) {
    fp32_t v;
    ac_int<32, true> raw = (ac_int<32, true>)bits;
    v.set_data(raw);
    return v;
}

static inline u32_t bits_from_fp32(const fp32_t& v) {
    ac_int<32, true> raw = v.data_ac_int();
    return (u32_t)raw;
}

static inline u32_t fp32_bits_from_double(double v) {
    fp32_t x(v);
    return bits_from_fp32(x);
}

static inline fp32_t fp32_zero() {
    return fp32_from_bits((u32_t)0u);
}

static inline fp32_t fp32_one() {
    return fp32_from_bits((u32_t)0x3F800000u);
}

} // namespace aecct
