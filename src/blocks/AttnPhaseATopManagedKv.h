#pragma once
// AC bring-up helper: Top-managed Phase-A KV staging over in_ch/out_ch.
// This helper keeps packet semantics minimal and provisional pre G-AC.

#include <ac_channel.h>

#include "AecctTypes.h"
#include "AttnDescBringup.h"
#include "AttnTopManagedPackets.h"
#include "TernaryLinearLive.h"
#include "TernaryLiveQkvLeafKernelTop.h"

namespace aecct {

typedef ac_channel<AttnTopManagedPacket> attn_pkt_ch_t;

static inline bool attn_top_emit_phasea_kv_work_unit(
    const u32_t* sram,
    u32_t x_row_base_word,
    u32_t token_idx,
    u32_t d_tile_idx,
    attn_pkt_ch_t& in_ch
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

    AttnTopManagedPacket wk_pkt;
    attn_packet_clear(wk_pkt);
    wk_pkt.kind = (u16_t)ATTN_PKT_WK;
    wk_pkt.token_idx = (u16_t)token_idx;
    wk_pkt.d_tile_idx = (u16_t)d_tile_idx;
    in_ch.write(wk_pkt);

    AttnTopManagedPacket wv_pkt;
    attn_packet_clear(wv_pkt);
    wv_pkt.kind = (u16_t)ATTN_PKT_WV;
    wv_pkt.token_idx = (u16_t)token_idx;
    wv_pkt.d_tile_idx = (u16_t)d_tile_idx;
    in_ch.write(wv_pkt);

    return true;
}

static inline bool attn_block_phasea_kv_consume_emit(
    attn_pkt_ch_t& in_ch,
    attn_pkt_ch_t& out_ch,
    const u32_t wk_payload_words[kTernaryLiveL0WkPayloadWords],
    u32_t wk_inv_sw_bits,
    const u32_t wv_payload_words[kTernaryLiveL0WvPayloadWords],
    u32_t wv_inv_sw_bits
) {
    AttnTopManagedPacket x_pkt;
    AttnTopManagedPacket wk_pkt;
    AttnTopManagedPacket wv_pkt;
    if (!in_ch.nb_read(x_pkt) || !in_ch.nb_read(wk_pkt) || !in_ch.nb_read(wv_pkt)) {
        return false;
    }
    if ((uint32_t)x_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_X) { return false; }
    if ((uint32_t)wk_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_WK) { return false; }
    if ((uint32_t)wv_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_WV) { return false; }
    if (x_pkt.token_idx != wk_pkt.token_idx || x_pkt.token_idx != wv_pkt.token_idx) { return false; }
    if (x_pkt.d_tile_idx != wk_pkt.d_tile_idx || x_pkt.d_tile_idx != wv_pkt.d_tile_idx) { return false; }

    u32_t k_out[kTernaryLiveL0WkRows];
    u32_t k_out_act_q[kTernaryLiveL0WkRows];
    u32_t k_out_inv_sw_bits = (u32_t)0;
    TernaryLiveL0WkRowTop wk_top;
    if (!wk_top.run(
            x_pkt.data,
            wk_payload_words,
            wk_inv_sw_bits,
            k_out,
            k_out_act_q,
            k_out_inv_sw_bits)) {
        return false;
    }

    u32_t v_out[kTernaryLiveL0WvRows];
    u32_t v_out_act_q[kTernaryLiveL0WvRows];
    u32_t v_out_inv_sw_bits = (u32_t)0;
    TernaryLiveL0WvRowTop wv_top;
    if (!wv_top.run(
            x_pkt.data,
            wv_payload_words,
            wv_inv_sw_bits,
            v_out,
            v_out_act_q,
            v_out_inv_sw_bits)) {
        return false;
    }

    AttnTopManagedPacket k_pkt;
    attn_packet_clear(k_pkt);
    k_pkt.kind = (u16_t)ATTN_PKT_K;
    k_pkt.token_idx = x_pkt.token_idx;
    k_pkt.d_tile_idx = x_pkt.d_tile_idx;
    k_pkt.inv_sw_bits = k_out_inv_sw_bits;
    for (unsigned i = 0; i < ATTN_TOP_MANAGED_TILE_WORDS; ++i) {
        k_pkt.data[i] = k_out[i];
    }
    out_ch.write(k_pkt);

    AttnTopManagedPacket v_pkt;
    attn_packet_clear(v_pkt);
    v_pkt.kind = (u16_t)ATTN_PKT_V;
    v_pkt.token_idx = x_pkt.token_idx;
    v_pkt.d_tile_idx = x_pkt.d_tile_idx;
    v_pkt.inv_sw_bits = v_out_inv_sw_bits;
    for (unsigned i = 0; i < ATTN_TOP_MANAGED_TILE_WORDS; ++i) {
        v_pkt.data[i] = v_out[i];
    }
    out_ch.write(v_pkt);

    return true;
}

static inline bool attn_top_writeback_phasea_kv_work_unit(
    u32_t* sram,
    u32_t scr_k_row_base_word,
    u32_t scr_v_row_base_word,
    u32_t token_idx,
    u32_t d_tile_idx,
    attn_pkt_ch_t& out_ch
) {
    AttnTopManagedPacket k_pkt;
    AttnTopManagedPacket v_pkt;
    if (!out_ch.nb_read(k_pkt) || !out_ch.nb_read(v_pkt)) {
        return false;
    }
    if ((uint32_t)k_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_K) { return false; }
    if ((uint32_t)v_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_V) { return false; }
    if ((uint32_t)k_pkt.token_idx.to_uint() != (uint32_t)token_idx.to_uint() ||
        (uint32_t)v_pkt.token_idx.to_uint() != (uint32_t)token_idx.to_uint()) {
        return false;
    }
    if ((uint32_t)k_pkt.d_tile_idx.to_uint() != (uint32_t)d_tile_idx.to_uint() ||
        (uint32_t)v_pkt.d_tile_idx.to_uint() != (uint32_t)d_tile_idx.to_uint()) {
        return false;
    }

    const uint32_t k_base = (uint32_t)scr_k_row_base_word.to_uint();
    const uint32_t v_base = (uint32_t)scr_v_row_base_word.to_uint();
    for (unsigned i = 0; i < ATTN_TOP_MANAGED_TILE_WORDS; ++i) {
        sram[k_base + i] = k_pkt.data[i];
        sram[v_base + i] = v_pkt.data[i];
    }
    return true;
}

static inline bool attn_phasea_top_managed_meta_ok(
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

// P11AC_MAINLINE_HELPER_ENTRYPOINT
// Real design-side Phase-A Top-managed KV entrypoint used by Top mainline wiring.
// Returns true only when the Top-managed path is executed end-to-end.
static inline bool attn_phasea_top_managed_kv_mainline(
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

    const QuantLinearMeta wk_meta = ternary_linear_live_l0_wk_meta();
    const QuantLinearMeta wv_meta = ternary_linear_live_l0_wv_meta();
    if (!attn_phasea_top_managed_meta_ok(wk_meta, QLM_L0_WK, d_model)) {
        return false;
    }
    if (!attn_phasea_top_managed_meta_ok(wv_meta, QLM_L0_WV, d_model)) {
        return false;
    }

    const ParamMeta wk_payload_meta = kParamMeta[wk_meta.weight_param_id];
    const ParamMeta wk_inv_meta = kParamMeta[wk_meta.inv_sw_param_id];
    const ParamMeta wv_payload_meta = kParamMeta[wv_meta.weight_param_id];
    const ParamMeta wv_inv_meta = kParamMeta[wv_meta.inv_sw_param_id];
    if (wk_payload_meta.len_w < wk_meta.payload_words_2b || wk_inv_meta.len_w == 0u) {
        return false;
    }
    if (wv_payload_meta.len_w < wv_meta.payload_words_2b || wv_inv_meta.len_w == 0u) {
        return false;
    }

    const uint32_t param_base = (uint32_t)param_base_word.to_uint();
    const uint32_t wk_payload_base = param_base + wk_payload_meta.offset_w;
    const uint32_t wv_payload_base = param_base + wv_payload_meta.offset_w;
    const uint32_t wk_inv_addr = param_base + wk_inv_meta.offset_w;
    const uint32_t wv_inv_addr = param_base + wv_inv_meta.offset_w;

    u32_t wk_payload_words[kTernaryLiveL0WkPayloadWords];
    u32_t wv_payload_words[kTernaryLiveL0WvPayloadWords];
    for (uint32_t i = 0u; i < wk_meta.payload_words_2b; ++i) {
        wk_payload_words[i] = sram[wk_payload_base + i];
    }
    for (uint32_t i = 0u; i < wv_meta.payload_words_2b; ++i) {
        wv_payload_words[i] = sram[wv_payload_base + i];
    }
    const u32_t wk_inv_sw_bits = sram[wk_inv_addr];
    const u32_t wv_inv_sw_bits = sram[wv_inv_addr];

    attn_pkt_ch_t in_ch;
    attn_pkt_ch_t out_ch;

    const uint32_t x_base = (uint32_t)x_in_base_word.to_uint();
    const uint32_t k_base = (uint32_t)sc.k_base_word.to_uint();
    const uint32_t v_base = (uint32_t)sc.v_base_word.to_uint();
    const uint32_t k_act_q_base = (uint32_t)sc.k_act_q_base_word.to_uint();
    const uint32_t v_act_q_base = (uint32_t)sc.v_act_q_base_word.to_uint();

    for (uint32_t t = 0u; t < token_count; ++t) {
        const uint32_t row_x_base = x_base + t * d_model;
        const uint32_t row_k_base = k_base + t * d_model;
        const uint32_t row_v_base = v_base + t * d_model;
        const uint32_t row_k_act_q_base = k_act_q_base + t * d_model;
        const uint32_t row_v_act_q_base = v_act_q_base + t * d_model;
        for (uint32_t dt = 0u; dt < d_tile_count; ++dt) {
            const uint32_t tile_offset = dt * tile_words;
            if (!attn_top_emit_phasea_kv_work_unit(
                    sram,
                    (u32_t)(row_x_base + tile_offset),
                    (u32_t)t,
                    (u32_t)dt,
                    in_ch)) {
                return false;
            }
            if (!attn_block_phasea_kv_consume_emit(
                    in_ch,
                    out_ch,
                    wk_payload_words,
                    wk_inv_sw_bits,
                    wv_payload_words,
                    wv_inv_sw_bits)) {
                return false;
            }
            if (!attn_top_writeback_phasea_kv_work_unit(
                    sram,
                    (u32_t)(row_k_base + tile_offset),
                    (u32_t)(row_v_base + tile_offset),
                    (u32_t)t,
                    (u32_t)dt,
                    out_ch)) {
                return false;
            }
            for (uint32_t i = 0u; i < tile_words; ++i) {
                sram[row_k_act_q_base + tile_offset + i] = sram[row_k_base + tile_offset + i];
                sram[row_v_act_q_base + tile_offset + i] = sram[row_v_base + tile_offset + i];
            }
        }
    }

    fallback_taken = false;
    return true;
}

} // namespace aecct
