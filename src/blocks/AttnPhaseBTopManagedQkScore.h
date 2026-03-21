#pragma once
// AE bring-up helper: Top-managed Phase-B QK/score staging (local-only).
// This helper is additive to landed AC/AD surfaces and does not alter them.

#include <ac_channel.h>
#include <cstdint>

#include "AecctTypes.h"
#include "AttnDescBringup.h"
#include "AttnTopManagedPackets.h"
#include "QuantDesc.h"

namespace aecct {

typedef ac_channel<AttnTopManagedWorkPacket> attn_phaseb_qk_pkt_ch_t;

template<typename SramView>
static inline bool attn_phaseb_sram_view_ok(SramView&) {
    return true;
}

static inline bool attn_phaseb_sram_view_ok(const u32_t* sram) {
    return sram != (const u32_t*)0;
}

static inline bool attn_phaseb_sram_view_ok(u32_t* sram) {
    return sram != (u32_t*)0;
}

static inline quant_acc_t attn_phaseb_inv_sqrt_d_head(uint32_t d_head) {
    u32_t bits = (u32_t)0x3F800000u;
    if (d_head == 2u) { bits = (u32_t)0x3F3504F3u; }
    else if (d_head == 4u) { bits = (u32_t)0x3F000000u; }
    else if (d_head == 8u) { bits = (u32_t)0x3EB504F3u; }
    else if (d_head == 16u) { bits = (u32_t)0x3E800000u; }
    else if (d_head == 32u) { bits = (u32_t)0x3E3504F3u; }
    else if (d_head == 64u) { bits = (u32_t)0x3E000000u; }
    fp32_t fp = fp32_from_bits(bits);
    return fp.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
}

template<typename SramView>
static inline bool attn_phaseb_emit_qsrc_tile(
    const SramView& sram,
    u32_t q_tile_base_word,
    u32_t token_idx,
    u32_t key_token_idx,
    u32_t head_group_id,
    u32_t d_tile_idx,
    u32_t tile_begin,
    u32_t tile_end,
    u32_t tile_valid_words,
    attn_phaseb_qk_pkt_ch_t& in_ch
) {
    const uint32_t valid = (uint32_t)tile_valid_words.to_uint();
    if (valid == 0u || valid > (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS) {
        return false;
    }

    AttnTopManagedWorkPacket pkt;
    attn_work_packet_clear(pkt);
    pkt.kind = (u16_t)ATTN_PKT_Q;
    pkt.phase_id = (u16_t)ATTN_PHASE_B;
    pkt.subphase_id = (u16_t)ATTN_SUBPHASE_QSRC;
    pkt.head_group_id = (u16_t)head_group_id;
    pkt.token_idx = (u16_t)token_idx;
    pkt.token_begin = (u16_t)token_idx;
    pkt.token_end = (u16_t)(token_idx + 1u);
    pkt.d_tile_idx = (u16_t)d_tile_idx;
    pkt.tile_begin = (u16_t)tile_begin;
    pkt.tile_end = (u16_t)tile_end;
    pkt.tile_valid_words = (u16_t)tile_valid_words;
    pkt.flags = (u16_t)key_token_idx;
    for (uint32_t i = 0u; i < valid; ++i) {
        pkt.data[i] = sram[(uint32_t)q_tile_base_word.to_uint() + i];
    }
    in_ch.write(pkt);
    return true;
}

template<typename SramView>
static inline bool attn_phaseb_emit_kvscan_tile(
    const SramView& sram,
    u32_t k_tile_base_word,
    u32_t token_idx,
    u32_t key_token_idx,
    u32_t head_group_id,
    u32_t d_tile_idx,
    u32_t tile_begin,
    u32_t tile_end,
    u32_t tile_valid_words,
    attn_phaseb_qk_pkt_ch_t& in_ch
) {
    const uint32_t valid = (uint32_t)tile_valid_words.to_uint();
    if (valid == 0u || valid > (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS) {
        return false;
    }

    AttnTopManagedWorkPacket pkt;
    attn_work_packet_clear(pkt);
    pkt.kind = (u16_t)ATTN_PKT_K;
    pkt.phase_id = (u16_t)ATTN_PHASE_B;
    pkt.subphase_id = (u16_t)ATTN_SUBPHASE_KVSCAN;
    pkt.head_group_id = (u16_t)head_group_id;
    pkt.token_idx = (u16_t)token_idx;
    pkt.token_begin = (u16_t)key_token_idx;
    pkt.token_end = (u16_t)(key_token_idx + 1u);
    pkt.d_tile_idx = (u16_t)d_tile_idx;
    pkt.tile_begin = (u16_t)tile_begin;
    pkt.tile_end = (u16_t)tile_end;
    pkt.tile_valid_words = (u16_t)tile_valid_words;
    pkt.flags = (u16_t)key_token_idx;
    for (uint32_t i = 0u; i < valid; ++i) {
        pkt.data[i] = sram[(uint32_t)k_tile_base_word.to_uint() + i];
    }
    in_ch.write(pkt);
    return true;
}

static inline bool attn_phaseb_block_qk_dot_consume_emit(
    attn_phaseb_qk_pkt_ch_t& in_ch,
    attn_phaseb_qk_pkt_ch_t& out_ch,
    u32_t token_idx,
    u32_t key_token_idx,
    u32_t head_group_id,
    u32_t tile_begin,
    u32_t tile_end,
    quant_acc_t inv_sqrt_d_head
) {
    const uint32_t t_idx = (uint32_t)token_idx.to_uint();
    const uint32_t k_idx = (uint32_t)key_token_idx.to_uint();
    const uint32_t group_id = (uint32_t)head_group_id.to_uint();
    const uint32_t dt_begin = (uint32_t)tile_begin.to_uint();
    const uint32_t dt_end = (uint32_t)tile_end.to_uint();

    if (dt_end <= dt_begin) {
        return false;
    }

    quant_acc_t dot = quant_acc_t(0);
    ATTN_P11AE_DOT_TILE_LOOP: for (uint32_t dt = dt_begin; dt < dt_end; ++dt) {
        AttnTopManagedWorkPacket q_pkt;
        AttnTopManagedWorkPacket k_pkt;
        if (!in_ch.nb_read(q_pkt) || !in_ch.nb_read(k_pkt)) {
            return false;
        }
        if ((uint32_t)q_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_Q) { return false; }
        if ((uint32_t)k_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_K) { return false; }
        if ((uint32_t)q_pkt.phase_id.to_uint() != (uint32_t)ATTN_PHASE_B) { return false; }
        if ((uint32_t)k_pkt.phase_id.to_uint() != (uint32_t)ATTN_PHASE_B) { return false; }
        if ((uint32_t)q_pkt.subphase_id.to_uint() != (uint32_t)ATTN_SUBPHASE_QSRC) { return false; }
        if ((uint32_t)k_pkt.subphase_id.to_uint() != (uint32_t)ATTN_SUBPHASE_KVSCAN) { return false; }
        if ((uint32_t)q_pkt.head_group_id.to_uint() != group_id) { return false; }
        if ((uint32_t)k_pkt.head_group_id.to_uint() != group_id) { return false; }
        if ((uint32_t)q_pkt.token_idx.to_uint() != t_idx || (uint32_t)k_pkt.token_idx.to_uint() != t_idx) { return false; }
        if ((uint32_t)q_pkt.flags.to_uint() != k_idx || (uint32_t)k_pkt.flags.to_uint() != k_idx) { return false; }
        if ((uint32_t)q_pkt.d_tile_idx.to_uint() != dt || (uint32_t)k_pkt.d_tile_idx.to_uint() != dt) { return false; }
        if ((uint32_t)q_pkt.tile_begin.to_uint() != dt_begin || (uint32_t)q_pkt.tile_end.to_uint() != dt_end) { return false; }
        if ((uint32_t)k_pkt.tile_begin.to_uint() != dt_begin || (uint32_t)k_pkt.tile_end.to_uint() != dt_end) { return false; }

        const uint32_t valid_q = (uint32_t)q_pkt.tile_valid_words.to_uint();
        const uint32_t valid_k = (uint32_t)k_pkt.tile_valid_words.to_uint();
        if (valid_q == 0u || valid_q > (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS) { return false; }
        if (valid_q != valid_k) { return false; }
        ATTN_P11AE_DOT_COL_LOOP: for (uint32_t i = 0u; i < valid_q; ++i) {
            const quant_act_t qv = quant_act_from_bits(q_pkt.data[i]);
            const quant_act_t kv = quant_act_from_bits(k_pkt.data[i]);
            dot += quant_acc_t(qv) * quant_acc_t(kv);
        }
    }

    const quant_acc_t scaled = dot * inv_sqrt_d_head;
    AttnTopManagedWorkPacket score_pkt;
    attn_work_packet_clear(score_pkt);
    score_pkt.kind = (u16_t)ATTN_PKT_SCORE;
    score_pkt.phase_id = (u16_t)ATTN_PHASE_B;
    score_pkt.subphase_id = (u16_t)ATTN_SUBPHASE_MASK;
    score_pkt.head_group_id = (u16_t)head_group_id;
    score_pkt.token_idx = (u16_t)token_idx;
    score_pkt.token_begin = (u16_t)key_token_idx;
    score_pkt.token_end = (u16_t)(key_token_idx + 1u);
    score_pkt.d_tile_idx = (u16_t)0u;
    score_pkt.tile_begin = (u16_t)0u;
    score_pkt.tile_end = (u16_t)1u;
    score_pkt.tile_valid_words = (u16_t)1u;
    score_pkt.flags = (u16_t)key_token_idx;
    score_pkt.data[0] = quant_bits_from_acc(scaled);
    out_ch.write(score_pkt);
    return true;
}

template<typename SramView>
static inline bool attn_phaseb_top_writeback_score(
    SramView& sram,
    u32_t score_word_addr,
    u32_t token_idx,
    u32_t key_token_idx,
    u32_t head_group_id,
    attn_phaseb_qk_pkt_ch_t& out_ch
) {
    AttnTopManagedWorkPacket score_pkt;
    if (!out_ch.nb_read(score_pkt)) {
        return false;
    }
    if ((uint32_t)score_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_SCORE) { return false; }
    if ((uint32_t)score_pkt.phase_id.to_uint() != (uint32_t)ATTN_PHASE_B) { return false; }
    if ((uint32_t)score_pkt.subphase_id.to_uint() != (uint32_t)ATTN_SUBPHASE_MASK) { return false; }
    if ((uint32_t)score_pkt.head_group_id.to_uint() != (uint32_t)head_group_id.to_uint()) { return false; }
    if ((uint32_t)score_pkt.token_idx.to_uint() != (uint32_t)token_idx.to_uint()) { return false; }
    if ((uint32_t)score_pkt.flags.to_uint() != (uint32_t)key_token_idx.to_uint()) { return false; }
    if ((uint32_t)score_pkt.tile_valid_words.to_uint() != 1u) { return false; }
    sram[(uint32_t)score_word_addr.to_uint()] = score_pkt.data[0];
    return true;
}

// P11AE_MAINLINE_HELPER_ENTRYPOINT
// Real design-side QK/score entrypoint used by Top mainline wiring.
// Writes score rows into the current score span consumed by AF.
template<typename SramView>
static inline bool attn_phaseb_top_managed_qk_score_mainline(
    SramView& sram,
    const AttnCfg& cfg,
    const AttnScratch& sc,
    u32_t token_idx,
    bool& fallback_taken
) {
    fallback_taken = true;
    if (!attn_phaseb_sram_view_ok(sram)) {
        return false;
    }

    uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t n_heads = (uint32_t)cfg.n_heads.to_uint();
    uint32_t d_head = (uint32_t)cfg.d_head.to_uint();
    uint32_t token = (uint32_t)token_idx.to_uint();

    if (token_count == 0u) { token_count = (uint32_t)ATTN_TOKEN_COUNT; }
    if (d_model == 0u) { d_model = (uint32_t)ATTN_D_MODEL; }
    if (n_heads == 0u) { n_heads = (uint32_t)ATTN_N_HEADS; }
    if (n_heads == 0u) { n_heads = 1u; }
    if (d_head == 0u) { d_head = d_model / n_heads; }

    if (token >= token_count || d_model == 0u || n_heads == 0u || d_head == 0u) {
        return false;
    }
    if ((n_heads * d_head) != d_model) {
        return false;
    }

    const uint32_t q_base = (uint32_t)sc.q_base_word.to_uint();
    const uint32_t k_base = (uint32_t)sc.k_base_word.to_uint();
    const uint32_t score_base = (uint32_t)sc.score_base_word.to_uint();
    const uint32_t q_row_base = q_base + token * d_model;
    const uint32_t tile_words = (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS;
    const uint32_t d_tile_count = attn_top_managed_tile_count(d_head, tile_words);
    if (d_tile_count == 0u) {
        return false;
    }
    const quant_acc_t inv_sqrt_d_head = attn_phaseb_inv_sqrt_d_head(d_head);

    attn_phaseb_qk_pkt_ch_t in_ch;
    attn_phaseb_qk_pkt_ch_t out_ch;

    ATTN_P11AE_HEAD_LOOP: for (uint32_t h = 0u; h < n_heads; ++h) {
        const uint32_t head_col_base = h * d_head;
        const uint32_t score_head_base = score_base + h * token_count;
        const u16_t head_group_id = attn_phaseb_head_group_id_from_head_idx(h);
        (void)attn_phaseb_rule_id_from_head_group(head_group_id);

        ATTN_P11AE_KEY_TOKEN_LOOP: for (uint32_t j = 0u; j < token_count; ++j) {
            const uint32_t k_row_base = k_base + j * d_model + head_col_base;
            ATTN_P11AE_TILE_EMIT_LOOP: for (uint32_t dt = 0u; dt < d_tile_count; ++dt) {
                const uint32_t tile_offset = dt * tile_words;
                const uint32_t tile_valid_words =
                    attn_top_managed_tile_valid_words(d_head, tile_words, dt);
                if (!attn_phaseb_emit_qsrc_tile(
                        sram,
                        (u32_t)(q_row_base + head_col_base + tile_offset),
                        (u32_t)token,
                        (u32_t)j,
                        (u32_t)head_group_id.to_uint(),
                        (u32_t)dt,
                        (u32_t)0u,
                        (u32_t)d_tile_count,
                        (u32_t)tile_valid_words,
                        in_ch)) {
                    return false;
                }
                if (!attn_phaseb_emit_kvscan_tile(
                        sram,
                        (u32_t)(k_row_base + tile_offset),
                        (u32_t)token,
                        (u32_t)j,
                        (u32_t)head_group_id.to_uint(),
                        (u32_t)dt,
                        (u32_t)0u,
                        (u32_t)d_tile_count,
                        (u32_t)tile_valid_words,
                        in_ch)) {
                    return false;
                }
            }

            if (!attn_phaseb_block_qk_dot_consume_emit(
                    in_ch,
                    out_ch,
                    (u32_t)token,
                    (u32_t)j,
                    (u32_t)head_group_id.to_uint(),
                    (u32_t)0u,
                    (u32_t)d_tile_count,
                    inv_sqrt_d_head)) {
                return false;
            }
            if (!attn_phaseb_top_writeback_score(
                    sram,
                    (u32_t)(score_head_base + j),
                    (u32_t)token,
                    (u32_t)j,
                    (u32_t)head_group_id.to_uint(),
                    out_ch)) {
                return false;
            }
        }
    }

    fallback_taken = false;
    return true;
}

} // namespace aecct
