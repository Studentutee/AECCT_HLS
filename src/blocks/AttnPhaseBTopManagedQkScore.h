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

typedef ac_channel<AttnTopManagedWorkPacket> attn_phaseb_q_pkt_ch_t;
typedef ac_channel<AttnTopManagedWorkPacket> attn_phaseb_k_pkt_ch_t;
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
    attn_phaseb_q_pkt_ch_t& q_ch
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
    q_ch.write(pkt);
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
    attn_phaseb_k_pkt_ch_t& k_ch
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
    k_ch.write(pkt);
    return true;
}

static inline bool attn_phaseb_block_qk_dot_consume_emit(
    attn_phaseb_q_pkt_ch_t& q_ch,
    attn_phaseb_k_pkt_ch_t& k_ch,
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
        if (!q_ch.nb_read(q_pkt) || !k_ch.nb_read(k_pkt)) {
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
    bool& fallback_taken,
    u32_t phase_entry_probe_q_base_word = (u32_t)0u,
    u32_t phase_entry_probe_k_base_word = (u32_t)0u,
    const u32_t* phase_entry_probe_q_words = 0,
    const u32_t* phase_entry_probe_k_words = 0,
    u32_t phase_entry_probe_words_valid = (u32_t)0u,
    u32_t* phase_entry_probe_visible = 0,
    u32_t* phase_entry_probe_owner_ok = 0,
    u32_t* phase_entry_probe_compare_ok = 0,
    u32_t score_tile_bridge_base_word = (u32_t)0u,
    const u32_t* score_tile_bridge_words = 0,
    u32_t score_tile_bridge_words_valid = (u32_t)0u,
    u32_t score_tile_bridge_key_begin = (u32_t)0u,
    u32_t* score_tile_bridge_visible = 0,
    u32_t* score_tile_bridge_owner_ok = 0,
    u32_t* score_tile_bridge_consumed = 0,
    u32_t* score_tile_bridge_compare_ok = 0,
    u32_t score_tile_bridge_head_idx = (u32_t)0u,
    u32_t score_tile_bridge_family_case_count = (u32_t)0u,
    const u32_t* score_tile_bridge_family_base_words = 0,
    const u32_t* score_tile_bridge_family_words = 0,
    const u32_t* score_tile_bridge_family_words_valid = 0,
    const u32_t* score_tile_bridge_family_key_begin = 0,
    const u32_t* score_tile_bridge_family_head_idx = 0,
    u32_t* score_tile_bridge_family_visible_count = 0,
    u32_t* score_tile_bridge_family_owner_ok = 0,
    u32_t* score_tile_bridge_family_consumed_count = 0,
    u32_t* score_tile_bridge_family_compare_ok = 0,
    u32_t* score_tile_bridge_family_case_mask = 0
) {
    static const uint32_t kScoreTileBridgeFamilyMaxCases = 3u;
    static const uint32_t kScoreTileBridgeFamilyStrideWords =
        (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS;
    fallback_taken = true;
    if (phase_entry_probe_visible != 0) { *phase_entry_probe_visible = (u32_t)0u; }
    if (phase_entry_probe_owner_ok != 0) { *phase_entry_probe_owner_ok = (u32_t)0u; }
    if (phase_entry_probe_compare_ok != 0) { *phase_entry_probe_compare_ok = (u32_t)0u; }
    if (score_tile_bridge_visible != 0) { *score_tile_bridge_visible = (u32_t)0u; }
    if (score_tile_bridge_owner_ok != 0) { *score_tile_bridge_owner_ok = (u32_t)0u; }
    if (score_tile_bridge_consumed != 0) { *score_tile_bridge_consumed = (u32_t)0u; }
    if (score_tile_bridge_compare_ok != 0) { *score_tile_bridge_compare_ok = (u32_t)0u; }
    if (score_tile_bridge_family_visible_count != 0) { *score_tile_bridge_family_visible_count = (u32_t)0u; }
    if (score_tile_bridge_family_owner_ok != 0) { *score_tile_bridge_family_owner_ok = (u32_t)1u; }
    if (score_tile_bridge_family_consumed_count != 0) { *score_tile_bridge_family_consumed_count = (u32_t)0u; }
    if (score_tile_bridge_family_compare_ok != 0) { *score_tile_bridge_family_compare_ok = (u32_t)1u; }
    if (score_tile_bridge_family_case_mask != 0) { *score_tile_bridge_family_case_mask = (u32_t)0u; }
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
    const uint32_t phase_entry_probe_valid_words = (uint32_t)phase_entry_probe_words_valid.to_uint();
    const bool phase_entry_probe_enabled =
        (phase_entry_probe_q_words != 0) &&
        (phase_entry_probe_k_words != 0) &&
        (phase_entry_probe_valid_words > 0u);
    const uint32_t score_tile_bridge_valid_words = (uint32_t)score_tile_bridge_words_valid.to_uint();
    const bool score_tile_bridge_enabled =
        (score_tile_bridge_words != 0) &&
        (score_tile_bridge_valid_words > 0u);
    const uint32_t score_tile_bridge_key_begin_u32 = (uint32_t)score_tile_bridge_key_begin.to_uint();
    const uint32_t score_tile_bridge_head_idx_u32 = (uint32_t)score_tile_bridge_head_idx.to_uint();
    const uint32_t score_tile_bridge_family_case_count_u32 =
        (uint32_t)score_tile_bridge_family_case_count.to_uint();
    const bool score_tile_bridge_family_enabled =
        (score_tile_bridge_family_case_count_u32 > 0u);
    uint32_t score_tile_bridge_family_key_begin_u32[kScoreTileBridgeFamilyMaxCases];
    uint32_t score_tile_bridge_family_words_valid_u32[kScoreTileBridgeFamilyMaxCases];
    uint32_t score_tile_bridge_family_head_idx_u32[kScoreTileBridgeFamilyMaxCases];
    uint32_t score_tile_bridge_family_base_word_u32[kScoreTileBridgeFamilyMaxCases];
    uint32_t score_tile_bridge_family_seen_words[kScoreTileBridgeFamilyMaxCases];
    uint32_t score_tile_bridge_family_visible_count_u32 = 0u;
    uint32_t score_tile_bridge_family_consumed_count_u32 = 0u;
    uint32_t score_tile_bridge_family_case_mask_u32 = 0u;
    bool score_tile_bridge_seen = false;
    if (score_tile_bridge_enabled && score_tile_bridge_family_enabled) {
        return false;
    }
    if (score_tile_bridge_enabled) {
        if (score_tile_bridge_valid_words > token_count) {
            return false;
        }
        if (score_tile_bridge_valid_words > tile_words) {
            return false;
        }
        if (score_tile_bridge_key_begin_u32 >= token_count) {
            return false;
        }
        if ((score_tile_bridge_key_begin_u32 + score_tile_bridge_valid_words) > token_count) {
            return false;
        }
        if (score_tile_bridge_head_idx_u32 >= n_heads) {
            return false;
        }
    }
    if (score_tile_bridge_family_enabled) {
        if (score_tile_bridge_family_case_count_u32 > kScoreTileBridgeFamilyMaxCases) {
            return false;
        }
        if (score_tile_bridge_family_base_words == 0 ||
            score_tile_bridge_family_words == 0 ||
            score_tile_bridge_family_words_valid == 0 ||
            score_tile_bridge_family_key_begin == 0 ||
            score_tile_bridge_family_head_idx == 0) {
            return false;
        }

        ATTN_P11AE_SCORE_TILE_FAMILY_PRECHECK_LOOP: for (uint32_t c = 0u;
             c < score_tile_bridge_family_case_count_u32; ++c) {
            const uint32_t valid =
                (uint32_t)score_tile_bridge_family_words_valid[c].to_uint();
            const uint32_t key_begin =
                (uint32_t)score_tile_bridge_family_key_begin[c].to_uint();
            const uint32_t head_idx =
                (uint32_t)score_tile_bridge_family_head_idx[c].to_uint();
            if (valid == 0u || valid > token_count || valid > tile_words) {
                return false;
            }
            if (key_begin >= token_count || (key_begin + valid) > token_count) {
                return false;
            }
            if (head_idx >= n_heads) {
                return false;
            }

            ATTN_P11AE_SCORE_TILE_FAMILY_OVERLAP_LOOP: for (uint32_t p = 0u; p < c; ++p) {
                if (score_tile_bridge_family_head_idx_u32[p] != head_idx) {
                    continue;
                }
                const uint32_t prev_begin = score_tile_bridge_family_key_begin_u32[p];
                const uint32_t prev_end =
                    prev_begin + score_tile_bridge_family_words_valid_u32[p];
                const uint32_t this_end = key_begin + valid;
                if (!(this_end <= prev_begin || key_begin >= prev_end)) {
                    return false;
                }
            }

            score_tile_bridge_family_words_valid_u32[c] = valid;
            score_tile_bridge_family_key_begin_u32[c] = key_begin;
            score_tile_bridge_family_head_idx_u32[c] = head_idx;
            score_tile_bridge_family_base_word_u32[c] =
                (uint32_t)score_tile_bridge_family_base_words[c].to_uint();
            score_tile_bridge_family_seen_words[c] = 0u;
        }
    }

    ATTN_P11AE_HEAD_LOOP: for (uint32_t h = 0u; h < n_heads; ++h) {
        const uint32_t head_col_base = h * d_head;
        const uint32_t score_head_base = score_base + h * token_count;
        const u16_t head_group_id = attn_phaseb_head_group_id_from_head_idx(h);
        (void)attn_phaseb_rule_id_from_head_group(head_group_id);

        // Local-only W4-M1 probe:
        // caller/Top can provide one narrow phase-entry descriptor for ownership visibility.
        if (phase_entry_probe_enabled && h == 0u) {
            const uint32_t expected_q_probe_base = q_row_base;
            const uint32_t expected_k_probe_base = k_base;
            if (phase_entry_probe_visible != 0) { *phase_entry_probe_visible = (u32_t)1u; }
            const bool owner_ok =
                ((uint32_t)phase_entry_probe_q_base_word.to_uint() == expected_q_probe_base) &&
                ((uint32_t)phase_entry_probe_k_base_word.to_uint() == expected_k_probe_base);
            if (phase_entry_probe_owner_ok != 0) {
                *phase_entry_probe_owner_ok = (u32_t)(owner_ok ? 1u : 0u);
            }
            if (!owner_ok) {
                return false;
            }
            if (phase_entry_probe_valid_words > tile_words || phase_entry_probe_valid_words > d_head) {
                return false;
            }
            bool probe_compare_ok = true;
            ATTN_P11AE_PHASE_ENTRY_PROBE_COL_LOOP: for (uint32_t i = 0u; i < phase_entry_probe_valid_words; ++i) {
                const uint32_t probe_q_word = (uint32_t)phase_entry_probe_q_words[i].to_uint();
                const uint32_t probe_k_word = (uint32_t)phase_entry_probe_k_words[i].to_uint();
                const uint32_t sram_q_word = (uint32_t)sram[expected_q_probe_base + i].to_uint();
                const uint32_t sram_k_word = (uint32_t)sram[expected_k_probe_base + i].to_uint();
                if (probe_q_word != sram_q_word || probe_k_word != sram_k_word) {
                    probe_compare_ok = false;
                }
            }
            if (phase_entry_probe_compare_ok != 0) {
                *phase_entry_probe_compare_ok = (u32_t)(probe_compare_ok ? 1u : 0u);
            }
            if (!probe_compare_ok) {
                return false;
            }
        }

        ATTN_P11AE_KEY_TOKEN_LOOP: for (uint32_t j = 0u; j < token_count; ++j) {
            const uint32_t k_row_base = k_base + j * d_model + head_col_base;
            quant_acc_t dot = quant_acc_t(0);
            ATTN_P11AE_TILE_DOT_LOOP: for (uint32_t dt = 0u; dt < d_tile_count; ++dt) {
                const uint32_t tile_offset = dt * tile_words;
                const uint32_t tile_valid_words =
                    attn_top_managed_tile_valid_words(d_head, tile_words, dt);
                if (tile_valid_words == 0u || tile_valid_words > tile_words) {
                    return false;
                }
                if ((tile_offset + tile_valid_words) > d_head) {
                    return false;
                }
                ATTN_P11AE_DOT_COL_LOOP: for (uint32_t i = 0u; i < tile_valid_words; ++i) {
                    const quant_act_t qv =
                        quant_act_from_bits(sram[q_row_base + head_col_base + tile_offset + i]);
                    const quant_act_t kv =
                        quant_act_from_bits(sram[k_row_base + tile_offset + i]);
                    dot += quant_acc_t(qv) * quant_acc_t(kv);
                }
            }
            const quant_acc_t scaled = dot * inv_sqrt_d_head;
            const uint32_t scaled_bits = (uint32_t)quant_bits_from_acc(scaled).to_uint();
            int32_t score_tile_bridge_family_case_idx = -1;
            if (score_tile_bridge_family_enabled) {
                ATTN_P11AE_SCORE_TILE_FAMILY_CASE_LOOP: for (uint32_t c = 0u;
                     c < score_tile_bridge_family_case_count_u32; ++c) {
                    const bool family_selected =
                        (h == score_tile_bridge_family_head_idx_u32[c]) &&
                        (j >= score_tile_bridge_family_key_begin_u32[c]) &&
                        (j < (score_tile_bridge_family_key_begin_u32[c] +
                              score_tile_bridge_family_words_valid_u32[c]));
                    if (family_selected) {
                        if (score_tile_bridge_family_case_idx >= 0) {
                            return false;
                        }
                        score_tile_bridge_family_case_idx = (int32_t)c;
                    }
                }
            }
            const bool score_tile_bridge_selected =
                score_tile_bridge_enabled &&
                (h == score_tile_bridge_head_idx_u32) &&
                (j >= score_tile_bridge_key_begin_u32) &&
                (j < (score_tile_bridge_key_begin_u32 + score_tile_bridge_valid_words));
            if (score_tile_bridge_family_case_idx >= 0) {
                const uint32_t case_idx = (uint32_t)score_tile_bridge_family_case_idx;
                if (score_tile_bridge_family_seen_words[case_idx] == 0u) {
                    score_tile_bridge_family_visible_count_u32 += 1u;
                    score_tile_bridge_family_case_mask_u32 |= (1u << case_idx);
                    if (score_tile_bridge_family_visible_count != 0) {
                        *score_tile_bridge_family_visible_count =
                            (u32_t)score_tile_bridge_family_visible_count_u32;
                    }
                    if (score_tile_bridge_family_case_mask != 0) {
                        *score_tile_bridge_family_case_mask =
                            (u32_t)score_tile_bridge_family_case_mask_u32;
                    }
                }
                const uint32_t expected_bridge_base =
                    score_head_base + score_tile_bridge_family_key_begin_u32[case_idx];
                const bool owner_ok =
                    (score_tile_bridge_family_base_word_u32[case_idx] ==
                     expected_bridge_base);
                if (score_tile_bridge_family_owner_ok != 0) {
                    *score_tile_bridge_family_owner_ok =
                        (u32_t)(owner_ok ? 1u : 0u);
                }
                if (!owner_ok) {
                    return false;
                }
                const uint32_t bridge_idx =
                    j - score_tile_bridge_family_key_begin_u32[case_idx];
                const uint32_t family_word_idx =
                    case_idx * kScoreTileBridgeFamilyStrideWords + bridge_idx;
                const uint32_t bridge_bits =
                    (uint32_t)score_tile_bridge_family_words[family_word_idx].to_uint();
                const bool bridge_compare_ok = (bridge_bits == scaled_bits);
                if (score_tile_bridge_family_compare_ok != 0) {
                    *score_tile_bridge_family_compare_ok =
                        (u32_t)(bridge_compare_ok ? 1u : 0u);
                }
                if (!bridge_compare_ok) {
                    return false;
                }
                sram[score_head_base + j] = score_tile_bridge_family_words[family_word_idx];
                score_tile_bridge_family_seen_words[case_idx] += 1u;
                score_tile_bridge_family_consumed_count_u32 += 1u;
                if (score_tile_bridge_family_consumed_count != 0) {
                    *score_tile_bridge_family_consumed_count =
                        (u32_t)score_tile_bridge_family_consumed_count_u32;
                }
            } else if (score_tile_bridge_selected) {
                if (score_tile_bridge_visible != 0) {
                    *score_tile_bridge_visible = (u32_t)1u;
                }
                const uint32_t expected_bridge_base = score_head_base + score_tile_bridge_key_begin_u32;
                const bool owner_ok =
                    ((uint32_t)score_tile_bridge_base_word.to_uint() == expected_bridge_base);
                if (score_tile_bridge_owner_ok != 0) {
                    *score_tile_bridge_owner_ok = (u32_t)(owner_ok ? 1u : 0u);
                }
                if (!owner_ok) {
                    return false;
                }
                const uint32_t bridge_idx = j - score_tile_bridge_key_begin_u32;
                const uint32_t bridge_bits = (uint32_t)score_tile_bridge_words[bridge_idx].to_uint();
                const bool bridge_compare_ok = (bridge_bits == scaled_bits);
                if (score_tile_bridge_compare_ok != 0) {
                    *score_tile_bridge_compare_ok = (u32_t)(bridge_compare_ok ? 1u : 0u);
                }
                if (!bridge_compare_ok) {
                    return false;
                }
                sram[score_head_base + j] = score_tile_bridge_words[bridge_idx];
                if (score_tile_bridge_consumed != 0) {
                    *score_tile_bridge_consumed = (u32_t)1u;
                }
                score_tile_bridge_seen = true;
            } else {
                sram[score_head_base + j] = (u32_t)scaled_bits;
            }
        }
    }

    if (score_tile_bridge_enabled && !score_tile_bridge_seen) {
        return false;
    }
    if (score_tile_bridge_family_enabled) {
        ATTN_P11AE_SCORE_TILE_FAMILY_FINALIZE_LOOP: for (uint32_t c = 0u;
             c < score_tile_bridge_family_case_count_u32; ++c) {
            if (score_tile_bridge_family_seen_words[c] !=
                score_tile_bridge_family_words_valid_u32[c]) {
                return false;
            }
        }
        if (score_tile_bridge_family_visible_count != 0) {
            *score_tile_bridge_family_visible_count =
                (u32_t)score_tile_bridge_family_visible_count_u32;
        }
        if (score_tile_bridge_family_consumed_count != 0) {
            *score_tile_bridge_family_consumed_count =
                (u32_t)score_tile_bridge_family_consumed_count_u32;
        }
        if (score_tile_bridge_family_case_mask != 0) {
            *score_tile_bridge_family_case_mask =
                (u32_t)score_tile_bridge_family_case_mask_u32;
        }
    }
    fallback_taken = false;
    return true;
}

} // namespace aecct
