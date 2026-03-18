#pragma once
// AC bring-up helper: Top-managed Phase-A KV staging over in_ch/out_ch.
// This helper keeps packet semantics minimal and provisional pre G-AC.

#include <ac_channel.h>

#include "AecctTypes.h"
#include "AttnTopManagedPackets.h"
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

} // namespace aecct
