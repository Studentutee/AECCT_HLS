#pragma once
// Final head: logits and x_pred.
// Pass A produces the token-wise scalar buffer.
// Pass B consumes that scalar buffer for OUT_FC reduction and out_mode-dependent streaming.
// Ownership boundary: Top selects SRAM bases and output mode; FinalHead does not own
// external sequencing policy.

#include <cstdint>
#include <cstdio>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "AecctRanges.h"
#include "AecctUtil.h"
#include "AttnTopManagedPackets.h"
#include "TransformerLayer.h"
#include "gen/SramMap.h"
#include "gen/ModelShapes.h"
#include "gen/WeightStreamOrder.h"

namespace aecct {

struct FinalHeadContract {
    bool start;
    bool done;
    bool err_valid;
    u16_t err_code;
    TokenRange token_range;
    TileRange tile_range;
    PhaseId phase_id;
    u32_t x_work_base_word;
    u32_t scratch_base_word;
    u32_t final_scalar_base_word;
    u32_t w_base_word;
};

static inline void clear_final_head_contract(FinalHeadContract& c) {
    c.start = false;
    c.done = false;
    c.err_valid = false;
    c.err_code = 0;
    c.token_range = make_token_range(0, 0);
    c.tile_range = make_tile_range(0, 0);
    c.phase_id = PHASE_FINAL_HEAD;
    c.x_work_base_word = 0;
    c.scratch_base_word = 0;
    c.final_scalar_base_word = 0;
    c.w_base_word = 0;
}

struct HeadParamBase {
    u32_t param_base_word;
    u32_t ffn1_w_base_word;
    u32_t ffn1_b_base_word;
    u32_t out_fc_w_base_word;
    u32_t out_fc_b_base_word;
};

static inline HeadParamBase make_head_param_base(u32_t w_base_word) {
    HeadParamBase hp;
    hp.param_base_word = w_base_word;
    hp.ffn1_w_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[66u].offset_w);
    hp.ffn1_b_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[18u].offset_w);
    hp.out_fc_w_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[67u].offset_w);
    hp.out_fc_b_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[19u].offset_w);
    return hp;
}

static const uint32_t FINAL_HEAD_OUTMODE_XPRED = 0u;
static const uint32_t FINAL_HEAD_OUTMODE_LOGITS = 1u;
static const uint32_t FINAL_HEAD_OUTMODE_NONE = 2u;

struct FinalHeadTopManagedTileMeta {
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

static inline bool final_head_top_managed_tile_meta_ok(
    const FinalHeadTopManagedTileMeta& m,
    uint32_t expect_phase_id,
    uint32_t expect_tile_idx
) {
    if ((uint32_t)m.phase_id.to_uint() != expect_phase_id) { return false; }
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
static inline bool FinalHeadCorePassABTopManaged(
    SramView& sram,
    const CfgRegs& cfg,
    u32_t x_end_base_word,
    const u32_t* y_words,
    u32_t logits_base_word,
    u32_t xpred_base_word,
    const HeadParamBase& hp,
    const FinalHeadContract& contract,
    data_ch_t* data_out,
    u32_t outmode_word,
    const u32_t* topfed_final_scalar_words = 0
) {
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    if (d_model == 0u) { d_model = (uint32_t)D_MODEL; }
    if (d_model == 0u) {
        return false;
    }

    const uint32_t token_count = (uint32_t)N_NODES;
    const uint32_t logits_words = (uint32_t)EXP_LEN_OUT_LOGITS_WORDS;
    const uint32_t xpred_words = (uint32_t)EXP_LEN_OUT_XPRED_WORDS;
    const uint32_t x_end_base = (uint32_t)x_end_base_word.to_uint();
    const uint32_t logits_base = (uint32_t)logits_base_word.to_uint();
    const uint32_t xpred_base = (uint32_t)xpred_base_word.to_uint();
    const uint32_t final_scalar_base = (uint32_t)contract.final_scalar_base_word.to_uint();
    const uint32_t out_fc_w_base = (uint32_t)hp.out_fc_w_base_word.to_uint();
    const uint32_t out_fc_b_base = (uint32_t)hp.out_fc_b_base_word.to_uint();

    uint32_t token_begin = (uint32_t)contract.token_range.begin.to_uint();
    uint32_t token_end = (uint32_t)contract.token_range.end.to_uint();
    if (token_begin > token_count) { token_begin = token_count; }
    if (token_end > token_count) { token_end = token_count; }
    if (token_end <= token_begin) {
        return false;
    }

    const uint32_t tile_words = (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS;
    const uint32_t class_tile_count = attn_top_managed_tile_count(logits_words, tile_words);
    uint32_t tile_begin = (uint32_t)contract.tile_range.begin.to_uint();
    uint32_t tile_end = (uint32_t)contract.tile_range.end.to_uint();
    if (tile_begin > class_tile_count) { tile_begin = class_tile_count; }
    if (tile_end > class_tile_count) { tile_end = class_tile_count; }
    if (tile_end <= tile_begin) {
        return false;
    }

    const uint32_t phase_id_u32 = (uint32_t)contract.phase_id;

    // Pass A: produce token-wise scalar and write FINAL_SCALAR_BUF.
    FINAL_HEAD_PASSA_TOKEN_LOOP: for (uint32_t t = token_begin; t < token_end; ++t) {
        u32_t st_bits = 0;
        if (topfed_final_scalar_words != 0) {
            // Top-fed scalar payload path: Top provides token scalars, FinalHead consumes.
            st_bits = topfed_final_scalar_words[t];
        } else {
            const uint32_t x_word = x_end_base + t * d_model;
            st_bits = sram[x_word];
        }
        sram[final_scalar_base + t] = st_bits;
    }

    const uint32_t outmode = (uint32_t)outmode_word.to_uint();
    const bool stream_logits = (outmode == FINAL_HEAD_OUTMODE_LOGITS);
    const bool stream_xpred = (outmode == FINAL_HEAD_OUTMODE_XPRED);
    const bool stream_enabled = (data_out != 0) && (stream_logits || stream_xpred);
#ifndef __SYNTHESIS__
    static bool final_head_mainline_logged = false;
    if (!final_head_mainline_logged) {
        final_head_mainline_logged = true;
        std::printf(
            "[p11ba][FINALHEAD_MAINLINE] passA_passB_top_managed token_begin=%u token_end=%u tile_begin=%u tile_end=%u outmode=%u stream_enabled=%u\n",
            (unsigned)token_begin,
            (unsigned)token_end,
            (unsigned)tile_begin,
            (unsigned)tile_end,
            (unsigned)outmode,
            stream_enabled ? 1u : 0u
        );
    }
#endif

    // Pass B: OUT_FC reduction over FINAL_SCALAR_BUF, then outmode-dependent stream.
    FINAL_HEAD_PASSB_CLASS_TILE_LOOP: for (uint32_t dt = tile_begin; dt < tile_end; ++dt) {
        const uint32_t tile_offset = dt * tile_words;
        const uint32_t valid = attn_top_managed_tile_valid_words(logits_words, tile_words, dt);

        FinalHeadTopManagedTileMeta meta;
        meta.phase_id = (u16_t)phase_id_u32;
        meta.subphase_id = (u16_t)ATTN_SUBPHASE_OUT;
        meta.token_begin = (u16_t)token_begin;
        meta.token_end = (u16_t)token_end;
        meta.token_idx = (u16_t)token_begin;
        meta.tile_begin = (u16_t)tile_begin;
        meta.tile_end = (u16_t)tile_end;
        meta.tile_idx = (u16_t)dt;
        meta.tile_valid_words = (u16_t)valid;
        if (!final_head_top_managed_tile_meta_ok(meta, phase_id_u32, dt)) {
            continue;
        }

        FINAL_HEAD_PASSB_CLASS_REDUCE_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
            const uint32_t c = tile_offset + i;
            if (c >= logits_words) {
                continue;
            }

            fp32_t acc = fp32_from_bits(sram[out_fc_b_base + c]);
            FINAL_HEAD_PASSB_TOKEN_REDUCE_LOOP: for (uint32_t t = token_begin; t < token_end; ++t) {
                const u32_t st_bits =
                    (topfed_final_scalar_words != 0) ?
                    topfed_final_scalar_words[t] :
                    sram[final_scalar_base + t];
                const fp32_t st = fp32_from_bits(st_bits);
                const fp32_t w = fp32_from_bits(sram[out_fc_w_base + c * token_count + t]);
                acc += (w * st);
            }

            const u32_t logits_bits = bits_from_fp32(acc);
            sram[logits_base + c] = logits_bits;
            if (stream_enabled && stream_logits) {
                data_out->write(logits_bits);
            }

            if (c < xpred_words) {
                const fp32_t y = (y_words != 0) ? fp32_from_bits(y_words[c]) : fp32_one();
                const fp32_t prod = acc * y;
                const bool mismatch = (prod < fp32_zero());
                const u32_t xpred_bits = mismatch ? bits_from_fp32(fp32_one()) : bits_from_fp32(fp32_zero());
                sram[xpred_base + c] = xpred_bits;
                if (stream_enabled && stream_xpred) {
                    data_out->write(xpred_bits);
                }
            }
        }
    }

    return stream_enabled;
}

// Public FinalHead entry.
// Read in two chunks: Pass A writes FINAL_SCALAR_BUF, then Pass B reduces OUT_FC and
// emits either logits or x_pred according to Top-owned out_mode.
static inline void FinalHead(
    u32_t* sram,
    const CfgRegs& cfg,
    u32_t x_end_base_word,
    const u32_t* y_words,
    u32_t logits_base_word,
    u32_t xpred_base_word,
    const HeadParamBase& hp,
    data_ch_t* data_out = 0,
    u32_t outmode_word = (u32_t)FINAL_HEAD_OUTMODE_NONE
) {
    FinalHeadContract contract;
    clear_final_head_contract(contract);
    contract.start = true;
    contract.phase_id = PHASE_FINAL_HEAD;
    contract.x_work_base_word = x_end_base_word;
    contract.final_scalar_base_word = (u32_t)sram_map::SCR_FINAL_SCALAR_BASE_W;
    contract.w_base_word = hp.param_base_word;
    contract.token_range = make_token_range((u32_t)0u, (u32_t)N_NODES);
    const uint32_t class_tile_count = attn_top_managed_tile_count(
        (uint32_t)EXP_LEN_OUT_LOGITS_WORDS,
        (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS);
    contract.tile_range = make_tile_range((u32_t)0u, (u32_t)class_tile_count);

    (void)FinalHeadCorePassABTopManaged<u32_t*>(
        sram,
        cfg,
        x_end_base_word,
        y_words,
        logits_base_word,
        xpred_base_word,
        hp,
        contract,
        data_out,
        outmode_word
    );
    contract.done = true;
}

} // namespace aecct
