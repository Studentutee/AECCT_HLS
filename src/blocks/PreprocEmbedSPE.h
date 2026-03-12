#pragma once
// Preprocess block used by bring-up flow.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "AecctRanges.h"
#include "PreprocDescBringup.h"

namespace aecct {

struct PreprocBlockContract {
    bool start;
    bool done;
    bool err_valid;
    u16_t err_code;
    TokenRange token_range;
    TileRange tile_range;
    PhaseId phase_id;
    u32_t x_work_base_word;
    u32_t scratch_base_word;
    u32_t w_base_word;
};

static inline void clear_preproc_contract(PreprocBlockContract& c) {
    c.start = false;
    c.done = false;
    c.err_valid = false;
    c.err_code = 0;
    c.token_range = make_token_range(0, 0);
    c.tile_range = make_tile_range(0, 0);
    c.phase_id = PHASE_PREPROC;
    c.x_work_base_word = 0;
    c.scratch_base_word = 0;
    c.w_base_word = 0;
}

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
