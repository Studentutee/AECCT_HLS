#pragma once
// Preprocess block used by bring-up flow.

#include <cstdint>

#include "AecctTypes.h"
#include "PreprocDescBringup.h"

namespace aecct {

struct PreprocCfg {
    u32_t infer_in_words;
    u32_t x_out_words;
};

static inline void PreprocEmbedSPE(
    u32_t* sram,
    const PreprocCfg& cfg,
    u32_t in_base_word,
    u32_t x_out_base_word
) {
    uint32_t in_base = (uint32_t)in_base_word.to_uint();
    uint32_t x_base = (uint32_t)x_out_base_word.to_uint();
    uint32_t infer_in_words = (uint32_t)cfg.infer_in_words.to_uint();
    uint32_t x_out_words = (uint32_t)cfg.x_out_words.to_uint();

    for (uint32_t i = 0; i < x_out_words; ++i) {
        if (i < infer_in_words) {
            sram[x_base + i] = sram[in_base + i];
        }
        else {
            sram[x_base + i] = (u32_t)0u;
        }
    }
}

} // namespace aecct
