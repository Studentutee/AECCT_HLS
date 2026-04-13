#pragma once
// Shared helper utilities.

#include <cstdint>

#include <ac_int.h>
#include <ac_std_float.h>

#include "AecctTypes.h"

namespace aecct {

typedef ac_ieee_float<binary32> fp32_t;
typedef ac_std_float<16, 5> fp16_t;

static inline fp16_t fp16_from_bits(const u16_t& bits) {
    fp16_t v;
    v.set_data((ac_int<16, true>)(ac_int<16, false>)bits);
    return v;
}

static inline u16_t bits_from_fp16(const fp16_t& v) {
    const ac_int<16, true> raw = v.data_ac_int();
    return (u16_t)((ac_int<16, false>)raw);
}

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


static inline u16_t fp16_lane_from_fp32_bits(const u32_t& bits) {
    const fp32_t x = fp32_from_bits(bits);
    const fp16_t h(x);
    const ac_int<16, true> raw = h.data_ac_int();
    return (u16_t)((ac_int<16, false>)raw);
}

static inline u32_t fp32_bits_from_fp16_lane(const u16_t& lane) {
    fp16_t h;
    h.set_data((ac_int<16, true>)(ac_int<16, false>)lane);
    const fp32_t y(h);
    return bits_from_fp32(y);
}

static inline u32_t pack_fp16_lanes(const u16_t& lo_lane, const u16_t& hi_lane) {
    u32_t word = 0;
    word.set_slc(0, lo_lane);
    word.set_slc(16, hi_lane);
    return word;
}

static inline u16_t unpack_fp16_lane(const u32_t& word, uint32_t lane_idx) {
    return (lane_idx == 0u) ? word.template slc<16>(0) : word.template slc<16>(16);
}

static inline uint32_t x_work_packed_u32_words(uint32_t elems) {
    return (elems + 1u) >> 1;
}

static inline uint32_t x_work_packed_row_words(uint32_t d_model) {
    return x_work_packed_u32_words(d_model);
}

template<typename SramView>
static inline u32_t x_work_load_fp32_bits(const SramView& sram, uint32_t x_base_word, uint32_t elem_idx) {
    const uint32_t packed_word_idx = elem_idx >> 1;
    const uint32_t lane_idx = elem_idx & 1u;
    const u32_t packed = sram[x_base_word + packed_word_idx];
    const u16_t lane = unpack_fp16_lane(packed, lane_idx);
    return fp32_bits_from_fp16_lane(lane);
}

template<typename SramView>
static inline fp32_t x_work_load_fp32(const SramView& sram, uint32_t x_base_word, uint32_t elem_idx) {
    return fp32_from_bits(x_work_load_fp32_bits(sram, x_base_word, elem_idx));
}

template<typename SramView>
static inline void x_work_store_fp32_bits(SramView& sram, uint32_t x_base_word, uint32_t elem_idx, const u32_t& fp32_bits) {
    const uint32_t packed_word_idx = elem_idx >> 1;
    const uint32_t lane_idx = elem_idx & 1u;
    const uint32_t addr = x_base_word + packed_word_idx;
    const u16_t new_lane = fp16_lane_from_fp32_bits(fp32_bits);
    u16_t lo_lane = 0;
    u16_t hi_lane = 0;
    if (lane_idx == 0u) {
        const u32_t prior = sram[addr];
        lo_lane = new_lane;
        hi_lane = unpack_fp16_lane(prior, 1u);
    } else {
        const u32_t prior = sram[addr];
        lo_lane = unpack_fp16_lane(prior, 0u);
        hi_lane = new_lane;
    }
    sram[addr] = pack_fp16_lanes(lo_lane, hi_lane);
}

template<typename SramView>
static inline void x_work_store_fp32(SramView& sram, uint32_t x_base_word, uint32_t elem_idx, const fp32_t& value) {
    x_work_store_fp32_bits(sram, x_base_word, elem_idx, bits_from_fp32(value));
}

template<typename SramView>
static inline u16_t x_work_load_fp16_bits(const SramView& sram, uint32_t x_base_word, uint32_t elem_idx) {
    const uint32_t packed_word_idx = elem_idx >> 1;
    const uint32_t lane_idx = elem_idx & 1u;
    const u32_t packed = sram[x_base_word + packed_word_idx];
    return unpack_fp16_lane(packed, lane_idx);
}

template<typename SramView>
static inline fp16_t x_work_load_fp16(const SramView& sram, uint32_t x_base_word, uint32_t elem_idx) {
    return fp16_from_bits(x_work_load_fp16_bits(sram, x_base_word, elem_idx));
}

template<typename SramView>
static inline void x_work_store_fp16_bits(SramView& sram, uint32_t x_base_word, uint32_t elem_idx, const u16_t& fp16_bits) {
    const uint32_t packed_word_idx = elem_idx >> 1;
    const uint32_t lane_idx = elem_idx & 1u;
    const uint32_t addr = x_base_word + packed_word_idx;
    u16_t lo_lane = 0;
    u16_t hi_lane = 0;
    const u32_t prior = sram[addr];
    if (lane_idx == 0u) {
        lo_lane = fp16_bits;
        hi_lane = unpack_fp16_lane(prior, 1u);
    } else {
        lo_lane = unpack_fp16_lane(prior, 0u);
        hi_lane = fp16_bits;
    }
    sram[addr] = pack_fp16_lanes(lo_lane, hi_lane);
}

template<typename SramView>
static inline void x_work_store_fp16(SramView& sram, uint32_t x_base_word, uint32_t elem_idx, const fp16_t& value) {
    x_work_store_fp16_bits(sram, x_base_word, elem_idx, bits_from_fp16(value));
}

static inline fp16_t fp16_zero() {
    return fp16_from_bits((u16_t)0u);
}

static inline fp16_t fp16_one() {
    return fp16_t(1.0f);
}

static inline fp32_t fp32_zero() {
    return fp32_from_bits((u32_t)0u);
}

static inline fp32_t fp32_one() {
    return fp32_from_bits((u32_t)0x3F800000u);
}

// DUT FP16 first cut keeps the external SRAM/data_out contract in FP32-bit words
// while forcing selected linear-path arithmetic through an FP16 roundtrip.
// This is intentionally a bounded migration step: arithmetic drift becomes
// visible without requiring immediate SramMap/layout changes.
static inline fp32_t fp16_linear_roundtrip(const fp32_t& x) {
    fp16_t h(x);
    fp32_t y(h);
    return y;
}

} // namespace aecct
