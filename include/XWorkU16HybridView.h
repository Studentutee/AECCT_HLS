#pragma once
// Focused hybrid SRAM view for bounded true-u16 X_WORK experiments.
// Non-X_WORK regions continue to use the legacy u32 carrier array.
// X_WORK itself is stored as one fp16 lane per u16 word.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctUtil.h"

namespace aecct {

template<uint32_t MAIN_WORDS, uint32_t X_WORDS>
struct XWorkU16HybridView {
    u32_t (&main_sram)[MAIN_WORDS];
    u16_t (&x_work_words)[X_WORDS];

    inline u32_t& operator[](uint32_t idx) { return main_sram[idx]; }
    inline const u32_t& operator[](uint32_t idx) const { return main_sram[idx]; }
};

template<uint32_t MAIN_WORDS, uint32_t X_WORDS>
static inline u32_t x_work_load_fp32_bits(
    const XWorkU16HybridView<MAIN_WORDS, X_WORDS>& view,
    uint32_t x_base_word,
    uint32_t elem_idx) {
    const uint32_t addr = x_base_word + elem_idx;
    if (addr >= X_WORDS) {
        return (u32_t)0u;
    }
    return fp32_bits_from_fp16_lane(view.x_work_words[addr]);
}

template<uint32_t MAIN_WORDS, uint32_t X_WORDS>
static inline void x_work_store_fp32_bits(
    XWorkU16HybridView<MAIN_WORDS, X_WORDS>& view,
    uint32_t x_base_word,
    uint32_t elem_idx,
    const u32_t& fp32_bits) {
    const uint32_t addr = x_base_word + elem_idx;
    if (addr >= X_WORDS) {
        return;
    }
    view.x_work_words[addr] = fp16_lane_from_fp32_bits(fp32_bits);
}

template<uint32_t MAIN_WORDS, uint32_t X_WORDS>
static inline void x_work_store_fp32(
    XWorkU16HybridView<MAIN_WORDS, X_WORDS>& view,
    uint32_t x_base_word,
    uint32_t elem_idx,
    const fp32_t& value) {
    x_work_store_fp32_bits(view, x_base_word, elem_idx, bits_from_fp32(value));
}

} // namespace aecct
