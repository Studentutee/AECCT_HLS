#pragma once
// AD bring-up helper: Top-managed Phase-A Q staging over in_ch/out_ch.
// This helper keeps packet semantics minimal and local-only.

#include <ac_channel.h>

#include "AecctTypes.h"
#include "AttnDescBringup.h"
#include "AttnTopManagedPackets.h"
#include "TernaryLinearLive.h"
#include "TernaryLiveQkvLeafKernelTop.h"

namespace aecct {

typedef ac_channel<AttnTopManagedPacket> attn_q_pkt_ch_t;

static_assert(ATTN_TOP_MANAGED_TILE_WORDS == (unsigned)kTernaryLiveL0WqCols, "Q tile words must match WQ cols");
static_assert(ATTN_TOP_MANAGED_TILE_WORDS == (unsigned)kTernaryLiveL0WqRows, "Q tile words must match WQ rows");

static inline bool attn_top_emit_phasea_q_work_unit(
    const u32_t* sram,
    u32_t x_row_base_word,
    u32_t token_idx,
    u32_t d_tile_idx,
    attn_q_pkt_ch_t& in_ch
) {
    AttnTopManagedPacket x_pkt;
    attn_packet_clear(x_pkt);
    x_pkt.kind = (u16_t)ATTN_PKT_X;
    x_pkt.token_idx = (u16_t)token_idx;
    x_pkt.d_tile_idx = (u16_t)d_tile_idx;
    for (unsigned i = 0; i < ATTN_TOP_MANAGED_TILE_WORDS; ++i) {
        x_pkt.data[i] = sram[(uint32_t)x_row_base_word.to_uint() + i];
    }
    in_ch.write(x_pkt);

    AttnTopManagedPacket wq_pkt;
    attn_packet_clear(wq_pkt);
    wq_pkt.kind = (u16_t)ATTN_PKT_WQ;
    wq_pkt.token_idx = (u16_t)token_idx;
    wq_pkt.d_tile_idx = (u16_t)d_tile_idx;
    in_ch.write(wq_pkt);
    return true;
}

static inline bool attn_block_phasea_q_consume_emit(
    attn_q_pkt_ch_t& in_ch,
    attn_q_pkt_ch_t& out_ch,
    const u32_t wq_payload_words[kTernaryLiveL0WqPayloadWords],
    u32_t wq_inv_sw_bits
) {
    AttnTopManagedPacket x_pkt;
    AttnTopManagedPacket wq_pkt;
    if (!in_ch.nb_read(x_pkt) || !in_ch.nb_read(wq_pkt)) {
        return false;
    }
    if ((uint32_t)x_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_X) { return false; }
    if ((uint32_t)wq_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_WQ) { return false; }
    if (x_pkt.token_idx != wq_pkt.token_idx) { return false; }
    if (x_pkt.d_tile_idx != wq_pkt.d_tile_idx) { return false; }

    u32_t q_out[kTernaryLiveL0WqRows];
    u32_t q_out_act_q[kTernaryLiveL0WqRows];
    u32_t q_out_inv_sw_bits = (u32_t)0;
    TernaryLiveL0WqRowTop wq_top;
    if (!wq_top.run(
            x_pkt.data,
            wq_payload_words,
            wq_inv_sw_bits,
            q_out,
            q_out_act_q,
            q_out_inv_sw_bits)) {
        return false;
    }

    AttnTopManagedPacket q_pkt;
    attn_packet_clear(q_pkt);
    q_pkt.kind = (u16_t)ATTN_PKT_Q;
    q_pkt.token_idx = x_pkt.token_idx;
    q_pkt.d_tile_idx = x_pkt.d_tile_idx;
    q_pkt.inv_sw_bits = q_out_inv_sw_bits;
    for (unsigned i = 0; i < ATTN_TOP_MANAGED_TILE_WORDS; ++i) {
        q_pkt.data[i] = q_out[i];
    }
    out_ch.write(q_pkt);
    return true;
}

static inline bool attn_top_writeback_phasea_q_work_unit(
    u32_t* sram,
    u32_t scr_q_row_base_word,
    u32_t scr_q_act_q_row_base_word,
    u32_t scr_q_sx_word_addr,
    u32_t token_idx,
    u32_t d_tile_idx,
    attn_q_pkt_ch_t& out_ch
) {
    AttnTopManagedPacket q_pkt;
    if (!out_ch.nb_read(q_pkt)) {
        return false;
    }
    if ((uint32_t)q_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_Q) { return false; }
    if ((uint32_t)q_pkt.token_idx.to_uint() != (uint32_t)token_idx.to_uint()) { return false; }
    if ((uint32_t)q_pkt.d_tile_idx.to_uint() != (uint32_t)d_tile_idx.to_uint()) { return false; }

    const uint32_t q_base = (uint32_t)scr_q_row_base_word.to_uint();
    const uint32_t q_act_q_base = (uint32_t)scr_q_act_q_row_base_word.to_uint();
    for (unsigned i = 0; i < ATTN_TOP_MANAGED_TILE_WORDS; ++i) {
        sram[q_base + i] = q_pkt.data[i];
        sram[q_act_q_base + i] = q_pkt.data[i];
    }
    sram[(uint32_t)scr_q_sx_word_addr.to_uint()] = q_pkt.inv_sw_bits;
    return true;
}

static inline bool attn_phasea_top_managed_q_meta_ok(
    const QuantLinearMeta& meta,
    QuantLinearMatrixId expected_matrix_id,
    uint32_t d_model
) {
    if (meta.matrix_id != (uint32_t)expected_matrix_id) {
        return false;
    }
    if (meta.rows != d_model || meta.cols != d_model) {
        return false;
    }
    if (meta.num_weights != (meta.rows * meta.cols)) {
        return false;
    }
    if (meta.payload_words_2b == 0u) {
        return false;
    }
    if (meta.payload_words_2b != ternary_payload_words_2b(meta.num_weights)) {
        return false;
    }
    if (meta.last_word_valid_count == 0u || meta.last_word_valid_count > kQkvCtPackedWordElems) {
        return false;
    }
    if (meta.last_word_valid_count != ternary_last_word_valid_count(meta.num_weights)) {
        return false;
    }
    return true;
}

// P11AD_MAINLINE_HELPER_ENTRYPOINT
// Real design-side Top-managed Q entrypoint used by Top mainline wiring.
// Returns true only when the Top-managed path is executed end-to-end.
static inline bool attn_phasea_top_managed_q_mainline(
    u32_t* sram,
    u32_t param_base_word,
    u32_t x_in_base_word,
    const AttnCfg& cfg,
    const AttnScratch& sc,
    bool& fallback_taken
) {
    fallback_taken = true;
    if (sram == (u32_t*)0) {
        return false;
    }

    uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    if (token_count == 0u) {
        token_count = (uint32_t)ATTN_TOKEN_COUNT;
    }
    if (d_model == 0u) {
        d_model = (uint32_t)ATTN_D_MODEL;
    }
    if (token_count == 0u || d_model == 0u) {
        return false;
    }
    if ((uint32_t)param_base_word.to_uint() == 0u) {
        return false;
    }

    const uint32_t tile_words = (uint32_t)ATTN_TOP_MANAGED_TILE_WORDS;
    if (tile_words == 0u || (d_model % tile_words) != 0u) {
        return false;
    }
    const uint32_t d_tile_count = d_model / tile_words;

    const QuantLinearMeta wq_meta = ternary_linear_live_l0_wq_meta();
    if (!attn_phasea_top_managed_q_meta_ok(wq_meta, QLM_L0_WQ, d_model)) {
        return false;
    }

    const ParamMeta wq_payload_meta = kParamMeta[wq_meta.weight_param_id];
    const ParamMeta wq_inv_meta = kParamMeta[wq_meta.inv_sw_param_id];
    if (wq_meta.payload_words_2b > (uint32_t)kTernaryLiveL0WqPayloadWords) {
        return false;
    }
    if (wq_payload_meta.len_w < wq_meta.payload_words_2b || wq_inv_meta.len_w == 0u) {
        return false;
    }

    const uint32_t param_base = (uint32_t)param_base_word.to_uint();
    const uint32_t wq_payload_base = param_base + wq_payload_meta.offset_w;
    const uint32_t wq_inv_addr = param_base + wq_inv_meta.offset_w;

    u32_t wq_payload_words[kTernaryLiveL0WqPayloadWords];
    for (uint32_t i = 0u; i < wq_meta.payload_words_2b; ++i) {
        wq_payload_words[i] = sram[wq_payload_base + i];
    }
    const u32_t wq_inv_sw_bits = sram[wq_inv_addr];

    attn_q_pkt_ch_t in_ch;
    attn_q_pkt_ch_t out_ch;

    const uint32_t x_base = (uint32_t)x_in_base_word.to_uint();
    const uint32_t q_base = (uint32_t)sc.q_base_word.to_uint();
    const uint32_t q_act_q_base = (uint32_t)sc.q_act_q_base_word.to_uint();
    const uint32_t q_sx_base = (uint32_t)sc.q_sx_base_word.to_uint();

    for (uint32_t t = 0u; t < token_count; ++t) {
        const uint32_t row_x_base = x_base + t * d_model;
        const uint32_t row_q_base = q_base + t * d_model;
        const uint32_t row_q_act_q_base = q_act_q_base + t * d_model;
        for (uint32_t dt = 0u; dt < d_tile_count; ++dt) {
            const uint32_t tile_offset = dt * tile_words;
            if (!attn_top_emit_phasea_q_work_unit(
                    sram,
                    (u32_t)(row_x_base + tile_offset),
                    (u32_t)t,
                    (u32_t)dt,
                    in_ch)) {
                return false;
            }
            if (!attn_block_phasea_q_consume_emit(
                    in_ch,
                    out_ch,
                    wq_payload_words,
                    wq_inv_sw_bits)) {
                return false;
            }
            if (!attn_top_writeback_phasea_q_work_unit(
                    sram,
                    (u32_t)(row_q_base + tile_offset),
                    (u32_t)(row_q_act_q_base + tile_offset),
                    (u32_t)q_sx_base,
                    (u32_t)t,
                    (u32_t)dt,
                    out_ch)) {
                return false;
            }
        }
    }

    fallback_taken = false;
    return true;
}

} // namespace aecct
