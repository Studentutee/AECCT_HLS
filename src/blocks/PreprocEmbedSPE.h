#pragma once
// Preprocess block used by bring-up flow.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "AecctRanges.h"
#include "AttnTopManagedPackets.h"
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

struct PreprocTopManagedWindowMeta {
    u16_t phase_id;
    u16_t subphase_id;
    u16_t token_begin;
    u16_t token_end;
    u16_t token_idx;
    u16_t tile_begin;
    u16_t tile_end;
    u16_t tile_idx;
    u16_t tile_valid_words;
};

static inline bool preproc_top_managed_window_meta_ok(
    const PreprocTopManagedWindowMeta& m,
    uint32_t expect_phase_id,
    uint32_t expect_token_idx,
    uint32_t expect_tile_idx
) {
    if ((uint32_t)m.phase_id.to_uint() != expect_phase_id) { return false; }
    if ((uint32_t)m.token_idx.to_uint() != expect_token_idx) { return false; }
    if ((uint32_t)m.tile_idx.to_uint() != expect_tile_idx) { return false; }

    const uint32_t t_begin = (uint32_t)m.token_begin.to_uint();
    const uint32_t t_end = (uint32_t)m.token_end.to_uint();
    const uint32_t dt_begin = (uint32_t)m.tile_begin.to_uint();
    const uint32_t dt_end = (uint32_t)m.tile_end.to_uint();
    const uint32_t valid = (uint32_t)m.tile_valid_words.to_uint();
    if (t_end <= t_begin) { return false; }
    if (dt_end <= dt_begin) { return false; }
    if (valid == 0u || valid > (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS) { return false; }
    return true;
}

template<typename SramView>
static inline void PreprocEmbedSPECoreWindow(
    SramView& sram,
    const PreprocCfg& cfg,
    u32_t in_base_word,
    u32_t x_out_base_word,
    const PreprocBlockContract& contract
) {
    uint32_t infer_in_words = (uint32_t)cfg.infer_in_words.to_uint();
    uint32_t x_out_words = (uint32_t)cfg.x_out_words.to_uint();
    if (infer_in_words == 0u) { infer_in_words = (uint32_t)PREPROC_IN_WORDS_EXPECTED; }
    if (x_out_words == 0u) { x_out_words = (uint32_t)PREPROC_X_OUT_WORDS_EXPECTED; }
    if (x_out_words == 0u) {
        return;
    }

    const uint32_t in_base = (uint32_t)in_base_word.to_uint();
    const uint32_t x_base = (uint32_t)x_out_base_word.to_uint();
    const uint32_t token_stride = (uint32_t)PREPROC_X_TOKEN_STRIDE_WORDS;
    if (token_stride == 0u) {
        return;
    }

    const uint32_t token_count = (x_out_words + token_stride - 1u) / token_stride;
    const uint32_t tile_words = (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS;
    const uint32_t tile_count = attn_top_managed_tile_count(token_stride, tile_words);
    if (token_count == 0u || tile_count == 0u) {
        return;
    }

    uint32_t token_begin = (uint32_t)contract.token_range.begin.to_uint();
    uint32_t token_end = (uint32_t)contract.token_range.end.to_uint();
    if (token_begin > token_count) { token_begin = token_count; }
    if (token_end > token_count) { token_end = token_count; }
    if (token_end <= token_begin) {
        return;
    }

    uint32_t tile_begin = (uint32_t)contract.tile_range.begin.to_uint();
    uint32_t tile_end = (uint32_t)contract.tile_range.end.to_uint();
    if (tile_begin > tile_count) { tile_begin = tile_count; }
    if (tile_end > tile_count) { tile_end = tile_count; }
    if (tile_end <= tile_begin) {
        return;
    }

    const uint32_t phase_id_u32 = (uint32_t)contract.phase_id;
    const uint32_t subphase_id_u32 = (uint32_t)ATTN_SUBPHASE_QSRC;
    PREPROC_TOP_MANAGED_TOKEN_LOOP: for (uint32_t t = token_begin; t < token_end; ++t) {
        const uint32_t token_base = t * token_stride;
        PREPROC_TOP_MANAGED_TILE_LOOP: for (uint32_t dt = tile_begin; dt < tile_end; ++dt) {
            const uint32_t tile_offset = dt * tile_words;
            const uint32_t valid = attn_top_managed_tile_valid_words(token_stride, tile_words, dt);

            PreprocTopManagedWindowMeta meta;
            meta.phase_id = (u16_t)phase_id_u32;
            meta.subphase_id = (u16_t)subphase_id_u32;
            meta.token_begin = (u16_t)token_begin;
            meta.token_end = (u16_t)token_end;
            meta.token_idx = (u16_t)t;
            meta.tile_begin = (u16_t)tile_begin;
            meta.tile_end = (u16_t)tile_end;
            meta.tile_idx = (u16_t)dt;
            meta.tile_valid_words = (u16_t)valid;
            if (!preproc_top_managed_window_meta_ok(meta, phase_id_u32, t, dt)) {
                continue;
            }

            PREPROC_TOP_MANAGED_TILE_STORE_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                const uint32_t linear_idx = token_base + tile_offset + i;
                if (linear_idx >= x_out_words) {
                    continue;
                }
                if (linear_idx < infer_in_words) {
                    sram[x_base + linear_idx] = sram[in_base + linear_idx];
                } else {
                    sram[x_base + linear_idx] = (u32_t)0u;
                }
            }
        }
    }
}

template<typename SramView>
static inline void PreprocEmbedSPECoreWindowDirect(
    SramView& sram,
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
        } else {
            sram[x_base + i] = (u32_t)0u;
        }
    }
}

static inline void PreprocEmbedSPE(
    u32_t* sram,
    const PreprocCfg& cfg,
    u32_t in_base_word,
    u32_t x_out_base_word
) {
    PreprocBlockContract contract;
    clear_preproc_contract(contract);
    contract.start = true;
    contract.phase_id = PHASE_PREPROC;
    contract.x_work_base_word = x_out_base_word;

    uint32_t x_out_words = (uint32_t)cfg.x_out_words.to_uint();
    if (x_out_words == 0u) { x_out_words = (uint32_t)PREPROC_X_OUT_WORDS_EXPECTED; }
    const uint32_t token_stride = (uint32_t)PREPROC_X_TOKEN_STRIDE_WORDS;
    const uint32_t token_count =
        (token_stride == 0u) ? 0u : ((x_out_words + token_stride - 1u) / token_stride);
    const uint32_t tile_count =
        (token_stride == 0u) ? 0u : attn_top_managed_tile_count(token_stride, (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS);
    contract.token_range = make_token_range((u32_t)0u, (u32_t)token_count);
    contract.tile_range = make_tile_range((u32_t)0u, (u32_t)tile_count);

    // Mainline migration: default Preproc entry now consumes Top-managed token/tile windows.
    PreprocEmbedSPECoreWindow<u32_t*>(
        sram,
        cfg,
        in_base_word,
        x_out_base_word,
        contract
    );
    contract.done = true;
}

template<uint32_t SRAM_WORDS>
static inline void PreprocEmbedSPETopManagedWindowBridge(
    u32_t (&sram_window)[SRAM_WORDS],
    const PreprocCfg& cfg,
    u32_t in_base_word,
    u32_t x_out_base_word,
    const PreprocBlockContract& contract
) {
    PreprocEmbedSPECoreWindow<u32_t (&)[SRAM_WORDS]>(
        sram_window,
        cfg,
        in_base_word,
        x_out_base_word,
        contract
    );
}

} // namespace aecct
