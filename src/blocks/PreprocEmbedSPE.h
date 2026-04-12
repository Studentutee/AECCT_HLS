#pragma once
// Preprocess block used by bring-up flow.
// Input: IO/ingest words staged by Top.
// Intermediate: token-wise embed/SPE accumulation in the caller-owned working window.
// Output: X_WORK tensor for downstream TransformerLayer consumption.
// Ownership boundary: Top owns external sequencing and shared-SRAM policy.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "AecctRanges.h"
#include "AecctUtil.h"
#include "AttnTopManagedPackets.h"
#include "PreprocDescBringup.h"
#include "gen/WeightStreamOrder.h"

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
    const PreprocBlockContract& contract,
    const u32_t* topfed_in_words = 0
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

    // Clip to Top-selected token window before any tile walk.
    uint32_t token_begin = (uint32_t)contract.token_range.begin.to_uint();
    uint32_t token_end = (uint32_t)contract.token_range.end.to_uint();
    if (token_begin > token_count) { token_begin = token_count; }
    if (token_end > token_count) { token_end = token_count; }
    if (token_end <= token_begin) {
        return;
    }

    // Clip to Top-selected tile window before touching SRAM.
    uint32_t tile_begin = (uint32_t)contract.tile_range.begin.to_uint();
    uint32_t tile_end = (uint32_t)contract.tile_range.end.to_uint();
    if (tile_begin > tile_count) { tile_begin = tile_count; }
    if (tile_end > tile_count) { tile_end = tile_count; }
    if (tile_end <= tile_begin) {
        return;
    }

    const uint32_t phase_id_u32 = (uint32_t)contract.phase_id;
    const uint32_t subphase_id_u32 = (uint32_t)ATTN_SUBPHASE_QSRC;
    const uint32_t param_base = (uint32_t)contract.w_base_word.to_uint();
    const uint32_t h_base = param_base + kParamMeta[20u].offset_w;   // BCH_H_BITPACK
    const uint32_t src_embed_base = param_base + kParamMeta[21u].offset_w; // src_embed
    const uint32_t lpe_base = param_base + kParamMeta[68u].offset_w; // lpe_token
    const uint32_t src_embed_dim = (uint32_t)kParamMeta[21u].d1;
    const uint32_t lpe_dim = (uint32_t)kParamMeta[68u].d1;
    const uint32_t d_model = src_embed_dim + lpe_dim;

    fp32_t var_feature[CODE_N];
    uint32_t hard_bit[CODE_N];
    fp32_t check_feature[CODE_C];
    fp32_t node_feature[N_NODES];

    PREPROC_VAR_FEATURE_INIT_LOOP: for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
        const u32_t y_bits =
            (v < infer_in_words) ?
                ((topfed_in_words != 0) ? topfed_in_words[v] : sram[in_base + v]) :
                bits_from_fp32(fp32_zero());
        const fp32_t y = fp32_from_bits(y_bits);
        var_feature[v] = (y < fp32_zero()) ? (fp32_zero() - y) : y;
        hard_bit[v] = (y < fp32_zero()) ? 1u : 0u;
        node_feature[v] = var_feature[v];
    }

    PREPROC_CHECK_FEATURE_INIT_LOOP: for (uint32_t c = 0u; c < (uint32_t)CODE_C; ++c) {
        uint32_t parity = 0u;
        PREPROC_CHECK_PARITY_LOOP: for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
            const uint32_t bit_index = c * (uint32_t)CODE_N + v;
            const uint32_t word_index = bit_index >> 5;
            const uint32_t bit_in_word = bit_index & 31u;
            const uint32_t h_word = (uint32_t)sram[h_base + word_index].to_uint();
            const uint32_t h_bit = (h_word >> bit_in_word) & 1u;
            if (h_bit != 0u) {
                parity ^= hard_bit[v];
            }
        }
        check_feature[c] = (parity == 0u) ? fp32_one() : (fp32_zero() - fp32_one());
        node_feature[(uint32_t)CODE_N + c] = check_feature[c];
    }

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
            // Metadata gate is the compatibility seam for token/tile ownership visibility.
            if (!preproc_top_managed_window_meta_ok(meta, phase_id_u32, t, dt)) {
                continue;
            }

            PREPROC_TOP_MANAGED_TILE_STORE_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                const uint32_t linear_word_idx = token_base + tile_offset + i;
                if (linear_word_idx >= x_out_words) {
                    continue;
                }
                const uint32_t d_word = linear_word_idx - token_base;
                const uint32_t d0 = d_word * 2u;
                const uint32_t elem_base = t * d_model;
                PREPROC_TOP_MANAGED_LANE_STORE_LOOP: for (uint32_t lane = 0u; lane < 2u; ++lane) {
                    const uint32_t d = d0 + lane;
                    if (d >= d_model) {
                        continue;
                    }
                    if (d < src_embed_dim) {
                        const u32_t embed_bits = sram[src_embed_base + t * src_embed_dim + d];
                        const fp32_t embed_v = fp32_from_bits(embed_bits);
                        const fp32_t x = node_feature[t] * embed_v;
                        x_work_store_fp32(sram, x_base, elem_base + d, x);
                    } else if (d < (src_embed_dim + lpe_dim)) {
                        const uint32_t lpe_d = d - src_embed_dim;
                        x_work_store_fp32_bits(sram, x_base, elem_base + d, sram[lpe_base + t * lpe_dim + lpe_d]);
                    } else {
                        // Tail beyond infer payload is explicitly zero-filled into X_WORK.
                        x_work_store_fp32_bits(sram, x_base, elem_base + d, (u32_t)0u);
                    }
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

    const uint32_t out_elems = infer_in_words;
    for (uint32_t i = 0; i < out_elems; ++i) {
        if (i < infer_in_words) {
            x_work_store_fp32_bits(sram, x_base, i, sram[in_base + i]);
        } else {
            x_work_store_fp32_bits(sram, x_base, i, (u32_t)0u);
        }
    }
}

// Public Preproc entry.
// Default mainline is the Top-managed token/tile window core; the direct variant is
// kept only as a compatibility helper under the same caller-owned base words.
static inline void PreprocEmbedSPE(
    u32_t* sram,
    const PreprocCfg& cfg,
    u32_t in_base_word,
    u32_t x_out_base_word
) {
    // Public wrapper builds contract metadata and forwards Top-owned windows to the core worker.
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
