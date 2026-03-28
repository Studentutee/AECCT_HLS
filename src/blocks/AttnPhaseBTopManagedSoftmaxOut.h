#pragma once
// AF bring-up helper: Top-managed Phase-B single-pass online softmax/output (local-only).
// This helper is additive to landed AC/AD/AE surfaces and does not alter them.

#include <ac_channel.h>
#include <cstdint>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "AttnDescBringup.h"
#include "AttnTopManagedPackets.h"
#include "QuantDesc.h"
#include "SoftmaxApprox.h"

namespace aecct {

typedef ac_channel<AttnTopManagedWorkPacket> attn_phaseb_softmax_score_ch_t;
typedef ac_channel<AttnTopManagedWorkPacket> attn_phaseb_softmax_v_ch_t;
typedef ac_channel<AttnTopManagedWorkPacket> attn_phaseb_softmax_pkt_ch_t;

template<typename SramView>
static inline bool attn_phaseb_softmax_sram_view_ok(SramView&) {
    return true;
}

static inline bool attn_phaseb_softmax_sram_view_ok(const u32_t* sram) {
    return sram != (const u32_t*)0;
}

static inline bool attn_phaseb_softmax_sram_view_ok(u32_t* sram) {
    return sram != (u32_t*)0;
}

template<typename SramView>
static inline bool attn_phaseb_emit_mask_score_word(
    const SramView& sram,
    u32_t score_word_addr,
    u32_t token_idx,
    u32_t key_token_idx,
    u32_t head_group_id,
    attn_phaseb_softmax_score_ch_t& score_ch
) {
    AttnTopManagedWorkPacket pkt;
    attn_work_packet_clear(pkt);
    pkt.kind = (u16_t)ATTN_PKT_SCORE;
    pkt.phase_id = (u16_t)ATTN_PHASE_B;
    pkt.subphase_id = (u16_t)ATTN_SUBPHASE_MASK;
    pkt.head_group_id = (u16_t)head_group_id;
    pkt.token_idx = (u16_t)token_idx;
    pkt.token_begin = (u16_t)key_token_idx;
    pkt.token_end = (u16_t)(key_token_idx + 1u);
    pkt.d_tile_idx = (u16_t)0u;
    pkt.tile_begin = (u16_t)0u;
    pkt.tile_end = (u16_t)1u;
    pkt.tile_valid_words = (u16_t)1u;
    pkt.flags = (u16_t)key_token_idx;
    pkt.data[0] = sram[(uint32_t)score_word_addr.to_uint()];
    score_ch.write(pkt);
    return true;
}

template<typename SramView>
static inline bool attn_phaseb_emit_v_tile(
    const SramView& sram,
    u32_t v_tile_base_word,
    u32_t token_idx,
    u32_t key_token_idx,
    u32_t head_group_id,
    u32_t d_tile_idx,
    u32_t tile_begin,
    u32_t tile_end,
    u32_t tile_valid_words,
    attn_phaseb_softmax_v_ch_t& v_ch
) {
    const uint32_t valid = (uint32_t)tile_valid_words.to_uint();
    if (valid == 0u || valid > (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS) {
        return false;
    }

    AttnTopManagedWorkPacket pkt;
    attn_work_packet_clear(pkt);
    pkt.kind = (u16_t)ATTN_PKT_V;
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
        pkt.data[i] = sram[(uint32_t)v_tile_base_word.to_uint() + i];
    }
    v_ch.write(pkt);
    return true;
}

static inline bool attn_phaseb_block_softmax_out_consume_emit(
    attn_phaseb_softmax_score_ch_t& score_ch,
    attn_phaseb_softmax_v_ch_t& v_ch,
    attn_phaseb_softmax_pkt_ch_t& out_ch,
    u32_t token_idx,
    u32_t head_group_id,
    u32_t token_count,
    u32_t d_head,
    u32_t tile_begin,
    u32_t tile_end
) {
    const uint32_t t_idx = (uint32_t)token_idx.to_uint();
    const uint32_t group_id = (uint32_t)head_group_id.to_uint();
    const uint32_t n_tokens = (uint32_t)token_count.to_uint();
    const uint32_t d_words = (uint32_t)d_head.to_uint();
    const uint32_t dt_begin = (uint32_t)tile_begin.to_uint();
    const uint32_t dt_end = (uint32_t)tile_end.to_uint();
    const uint32_t tile_words = (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS;

    if (n_tokens == 0u || d_words == 0u || dt_end <= dt_begin) {
        return false;
    }

    softmax_score_t running_max = softmax_score_t(0);
    softmax_sum_t running_l = softmax_sum_t(0);
    quant_acc_t running_acc[ATTN_D_MODEL];
    ATTN_P11AF_ACC_CLEAR_LOOP: for (uint32_t i = 0u; i < (uint32_t)ATTN_D_MODEL; ++i) {
        running_acc[i] = quant_acc_t(0);
    }

    bool have_state = false;
    ATTN_P11AF_KEY_TOKEN_LOOP: for (uint32_t j = 0u; j < n_tokens; ++j) {
        AttnTopManagedWorkPacket score_pkt;
        if (!score_ch.nb_read(score_pkt)) {
            return false;
        }
        if ((uint32_t)score_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_SCORE) { return false; }
        if ((uint32_t)score_pkt.phase_id.to_uint() != (uint32_t)ATTN_PHASE_B) { return false; }
        if ((uint32_t)score_pkt.subphase_id.to_uint() != (uint32_t)ATTN_SUBPHASE_MASK) { return false; }
        if ((uint32_t)score_pkt.head_group_id.to_uint() != group_id) { return false; }
        if ((uint32_t)score_pkt.token_idx.to_uint() != t_idx) { return false; }
        if ((uint32_t)score_pkt.flags.to_uint() != j) { return false; }
        if ((uint32_t)score_pkt.tile_valid_words.to_uint() != 1u) { return false; }

        const fp32_t score_fp = fp32_from_bits(score_pkt.data[0]);
        const softmax_score_t score =
            score_fp.template convert_to_ac_fixed<18, 6, true, AC_RND, AC_SAT>(false);

        if (!have_state) {
            running_max = score;
            running_l = softmax_sum_t(1);
            ATTN_P11AF_INIT_ACC_TILE_LOOP: for (uint32_t dt = dt_begin; dt < dt_end; ++dt) {
                AttnTopManagedWorkPacket v_pkt;
                if (!v_ch.nb_read(v_pkt)) {
                    return false;
                }
                if ((uint32_t)v_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_V) { return false; }
                if ((uint32_t)v_pkt.phase_id.to_uint() != (uint32_t)ATTN_PHASE_B) { return false; }
                if ((uint32_t)v_pkt.subphase_id.to_uint() != (uint32_t)ATTN_SUBPHASE_KVSCAN) { return false; }
                if ((uint32_t)v_pkt.head_group_id.to_uint() != group_id) { return false; }
                if ((uint32_t)v_pkt.token_idx.to_uint() != t_idx) { return false; }
                if ((uint32_t)v_pkt.flags.to_uint() != j) { return false; }
                if ((uint32_t)v_pkt.d_tile_idx.to_uint() != dt) { return false; }
                if ((uint32_t)v_pkt.tile_begin.to_uint() != dt_begin || (uint32_t)v_pkt.tile_end.to_uint() != dt_end) { return false; }

                const uint32_t valid = (uint32_t)v_pkt.tile_valid_words.to_uint();
                if (valid == 0u || valid > tile_words) { return false; }
                const uint32_t tile_offset = dt * tile_words;
                if ((tile_offset + valid) > d_words) { return false; }
                ATTN_P11AF_INIT_ACC_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                    const quant_act_t vv = quant_act_from_bits(v_pkt.data[i]);
                    running_acc[tile_offset + i] = quant_acc_t(vv);
                }
            }
            have_state = true;
            continue;
        }

        if (score > running_max) {
            const softmax_x_t old_minus_new = softmax_x_t(running_max - score);
            const softmax_exp_t alpha = softmax_exp_lut(old_minus_new);
            running_l = softmax_sum_t(running_l * softmax_sum_t(alpha)) + softmax_sum_t(1);
            ATTN_P11AF_RENORM_ACC_TILE_LOOP: for (uint32_t dt = dt_begin; dt < dt_end; ++dt) {
                AttnTopManagedWorkPacket v_pkt;
                if (!v_ch.nb_read(v_pkt)) {
                    return false;
                }
                if ((uint32_t)v_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_V) { return false; }
                if ((uint32_t)v_pkt.phase_id.to_uint() != (uint32_t)ATTN_PHASE_B) { return false; }
                if ((uint32_t)v_pkt.subphase_id.to_uint() != (uint32_t)ATTN_SUBPHASE_KVSCAN) { return false; }
                if ((uint32_t)v_pkt.head_group_id.to_uint() != group_id) { return false; }
                if ((uint32_t)v_pkt.token_idx.to_uint() != t_idx) { return false; }
                if ((uint32_t)v_pkt.flags.to_uint() != j) { return false; }
                if ((uint32_t)v_pkt.d_tile_idx.to_uint() != dt) { return false; }
                if ((uint32_t)v_pkt.tile_begin.to_uint() != dt_begin || (uint32_t)v_pkt.tile_end.to_uint() != dt_end) { return false; }

                const uint32_t valid = (uint32_t)v_pkt.tile_valid_words.to_uint();
                if (valid == 0u || valid > tile_words) { return false; }
                const uint32_t tile_offset = dt * tile_words;
                if ((tile_offset + valid) > d_words) { return false; }
                ATTN_P11AF_RENORM_ACC_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                    const quant_act_t vv = quant_act_from_bits(v_pkt.data[i]);
                    running_acc[tile_offset + i] =
                        quant_acc_t(running_acc[tile_offset + i] * quant_acc_t(alpha)) + quant_acc_t(vv);
                }
            }
            running_max = score;
        } else {
            const softmax_x_t score_minus_old = softmax_x_t(score - running_max);
            const softmax_exp_t beta = softmax_exp_lut(score_minus_old);
            running_l += softmax_sum_t(beta);
            ATTN_P11AF_ACC_TILE_LOOP: for (uint32_t dt = dt_begin; dt < dt_end; ++dt) {
                AttnTopManagedWorkPacket v_pkt;
                if (!v_ch.nb_read(v_pkt)) {
                    return false;
                }
                if ((uint32_t)v_pkt.kind.to_uint() != (uint32_t)ATTN_PKT_V) { return false; }
                if ((uint32_t)v_pkt.phase_id.to_uint() != (uint32_t)ATTN_PHASE_B) { return false; }
                if ((uint32_t)v_pkt.subphase_id.to_uint() != (uint32_t)ATTN_SUBPHASE_KVSCAN) { return false; }
                if ((uint32_t)v_pkt.head_group_id.to_uint() != group_id) { return false; }
                if ((uint32_t)v_pkt.token_idx.to_uint() != t_idx) { return false; }
                if ((uint32_t)v_pkt.flags.to_uint() != j) { return false; }
                if ((uint32_t)v_pkt.d_tile_idx.to_uint() != dt) { return false; }
                if ((uint32_t)v_pkt.tile_begin.to_uint() != dt_begin || (uint32_t)v_pkt.tile_end.to_uint() != dt_end) { return false; }

                const uint32_t valid = (uint32_t)v_pkt.tile_valid_words.to_uint();
                if (valid == 0u || valid > tile_words) { return false; }
                const uint32_t tile_offset = dt * tile_words;
                if ((tile_offset + valid) > d_words) { return false; }
                ATTN_P11AF_ACC_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                    const quant_act_t vv = quant_act_from_bits(v_pkt.data[i]);
                    running_acc[tile_offset + i] += quant_acc_t(beta) * quant_acc_t(vv);
                }
            }
        }
    }

    if (!have_state) {
        return false;
    }
    const softmax_inv_t inv_l = softmax_rcp_lut(running_l);
    ATTN_P11AF_OUTPUT_TILE_LOOP: for (uint32_t dt = dt_begin; dt < dt_end; ++dt) {
        const uint32_t valid = attn_top_managed_tile_valid_words(d_words, tile_words, dt);
        const uint32_t tile_offset = dt * tile_words;
        AttnTopManagedWorkPacket out_pkt;
        attn_work_packet_clear(out_pkt);
        out_pkt.kind = (u16_t)ATTN_PKT_OUT;
        out_pkt.phase_id = (u16_t)ATTN_PHASE_B;
        out_pkt.subphase_id = (u16_t)ATTN_SUBPHASE_OUT;
        out_pkt.head_group_id = (u16_t)head_group_id;
        out_pkt.token_idx = (u16_t)token_idx;
        out_pkt.token_begin = (u16_t)token_idx;
        out_pkt.token_end = (u16_t)(token_idx + 1u);
        out_pkt.d_tile_idx = (u16_t)dt;
        out_pkt.tile_begin = (u16_t)dt_begin;
        out_pkt.tile_end = (u16_t)dt_end;
        out_pkt.tile_valid_words = (u16_t)valid;
        out_pkt.flags = (u16_t)0u;
        ATTN_P11AF_OUTPUT_COL_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
            const quant_acc_t out_val = running_acc[tile_offset + i] * quant_acc_t(inv_l);
            out_pkt.data[i] = quant_bits_from_acc(out_val);
        }
        out_ch.write(out_pkt);
    }

    return true;
}

template<typename SramView>
static inline bool attn_phaseb_top_writeback_out_tile(
    SramView& sram,
    u32_t pre_tile_base_word,
    u32_t post_tile_base_word,
    u32_t out_tile_base_word,
    u32_t token_idx,
    u32_t head_group_id,
    u32_t d_tile_idx,
    u32_t tile_begin,
    u32_t tile_end,
    attn_phaseb_softmax_pkt_ch_t& out_ch
) {
    AttnTopManagedWorkPacket pkt;
    if (!out_ch.nb_read(pkt)) {
        return false;
    }
    if ((uint32_t)pkt.kind.to_uint() != (uint32_t)ATTN_PKT_OUT) { return false; }
    if ((uint32_t)pkt.phase_id.to_uint() != (uint32_t)ATTN_PHASE_B) { return false; }
    if ((uint32_t)pkt.subphase_id.to_uint() != (uint32_t)ATTN_SUBPHASE_OUT) { return false; }
    if ((uint32_t)pkt.head_group_id.to_uint() != (uint32_t)head_group_id.to_uint()) { return false; }
    if ((uint32_t)pkt.token_idx.to_uint() != (uint32_t)token_idx.to_uint()) { return false; }
    if ((uint32_t)pkt.d_tile_idx.to_uint() != (uint32_t)d_tile_idx.to_uint()) { return false; }
    if ((uint32_t)pkt.tile_begin.to_uint() != (uint32_t)tile_begin.to_uint() ||
        (uint32_t)pkt.tile_end.to_uint() != (uint32_t)tile_end.to_uint()) { return false; }

    const uint32_t valid = (uint32_t)pkt.tile_valid_words.to_uint();
    if (valid == 0u || valid > (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS) {
        return false;
    }

    const uint32_t pre_base = (uint32_t)pre_tile_base_word.to_uint();
    const uint32_t post_base = (uint32_t)post_tile_base_word.to_uint();
    const uint32_t out_base = (uint32_t)out_tile_base_word.to_uint();
    ATTN_P11AF_WRITEBACK_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
        sram[pre_base + i] = pkt.data[i];
        sram[post_base + i] = pkt.data[i];
        sram[out_base + i] = pkt.data[i];
    }
    return true;
}

// P11AF_MAINLINE_HELPER_ENTRYPOINT
// Real design-side single-pass online softmax/output entrypoint used by Top mainline wiring.
// Consumes the score span produced by AE and writes pre/post/out for the target token row.
template<typename SramView>
static inline bool attn_phaseb_top_managed_softmax_out_mainline(
    SramView& sram,
    const AttnCfg& cfg,
    const AttnScratch& sc,
    u32_t token_idx,
    u32_t attn_out_base_word,
    bool& fallback_taken
) {
    fallback_taken = true;
    if (!attn_phaseb_softmax_sram_view_ok(sram)) {
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

    const uint32_t score_base = (uint32_t)sc.score_base_word.to_uint();
    const uint32_t v_base = (uint32_t)sc.v_base_word.to_uint();
    const uint32_t pre_base = (uint32_t)sc.pre_concat_base_word.to_uint();
    const uint32_t post_base = (uint32_t)sc.post_concat_base_word.to_uint();
    const uint32_t out_base = (uint32_t)attn_out_base_word.to_uint();
    const uint32_t tile_words = (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS;
    const uint32_t d_tile_count = attn_top_managed_tile_count(d_head, tile_words);
    if (d_tile_count == 0u) {
        return false;
    }
    if (d_head > (uint32_t)ATTN_D_MODEL) {
        return false;
    }

    const uint32_t pre_row_base = pre_base + token * d_model;
    const uint32_t post_row_base = post_base + token * d_model;
    const uint32_t out_row_base = out_base + token * d_model;

    ATTN_P11AF_HEAD_LOOP: for (uint32_t h = 0u; h < n_heads; ++h) {
        const uint32_t head_col_base = h * d_head;
        const uint32_t score_head_base = score_base + h * token_count;
        const u16_t head_group_id = attn_phaseb_head_group_id_from_head_idx(h);
        (void)attn_phaseb_rule_id_from_head_group(head_group_id);

        softmax_score_t running_max = softmax_score_t(0);
        softmax_sum_t running_l = softmax_sum_t(0);
        quant_acc_t running_acc[ATTN_D_MODEL];
        ATTN_P11AF_MAINLINE_ACC_CLEAR_LOOP: for (uint32_t i = 0u; i < (uint32_t)ATTN_D_MODEL; ++i) {
            running_acc[i] = quant_acc_t(0);
        }
        bool have_state = false;

        ATTN_P11AF_MAINLINE_KEY_TOKEN_LOOP: for (uint32_t j = 0u; j < token_count; ++j) {
            const fp32_t score_fp = fp32_from_bits(sram[score_head_base + j]);
            const softmax_score_t score =
                score_fp.template convert_to_ac_fixed<18, 6, true, AC_RND, AC_SAT>(false);
            const uint32_t v_row_base = v_base + j * d_model + head_col_base;

            if (!have_state) {
                running_max = score;
                running_l = softmax_sum_t(1);
                ATTN_P11AF_MAINLINE_INIT_ACC_TILE_LOOP: for (uint32_t dt = 0u; dt < d_tile_count; ++dt) {
                    const uint32_t tile_offset = dt * tile_words;
                    const uint32_t valid =
                        attn_top_managed_tile_valid_words(d_head, tile_words, dt);
                    if (valid == 0u || valid > tile_words) { return false; }
                    if ((tile_offset + valid) > d_head) { return false; }
                    ATTN_P11AF_MAINLINE_INIT_ACC_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                        const quant_act_t vv = quant_act_from_bits(sram[v_row_base + tile_offset + i]);
                        running_acc[tile_offset + i] = quant_acc_t(vv);
                    }
                }
                have_state = true;
                continue;
            }

            if (score > running_max) {
                const softmax_x_t old_minus_new = softmax_x_t(running_max - score);
                const softmax_exp_t alpha = softmax_exp_lut(old_minus_new);
                running_l = softmax_sum_t(running_l * softmax_sum_t(alpha)) + softmax_sum_t(1);
                ATTN_P11AF_MAINLINE_RENORM_TILE_LOOP: for (uint32_t dt = 0u; dt < d_tile_count; ++dt) {
                    const uint32_t tile_offset = dt * tile_words;
                    const uint32_t valid =
                        attn_top_managed_tile_valid_words(d_head, tile_words, dt);
                    if (valid == 0u || valid > tile_words) { return false; }
                    if ((tile_offset + valid) > d_head) { return false; }
                    ATTN_P11AF_MAINLINE_RENORM_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                        const quant_act_t vv = quant_act_from_bits(sram[v_row_base + tile_offset + i]);
                        running_acc[tile_offset + i] =
                            quant_acc_t(running_acc[tile_offset + i] * quant_acc_t(alpha)) + quant_acc_t(vv);
                    }
                }
                running_max = score;
            } else {
                const softmax_x_t score_minus_old = softmax_x_t(score - running_max);
                const softmax_exp_t beta = softmax_exp_lut(score_minus_old);
                running_l += softmax_sum_t(beta);
                ATTN_P11AF_MAINLINE_ACC_TILE_LOOP: for (uint32_t dt = 0u; dt < d_tile_count; ++dt) {
                    const uint32_t tile_offset = dt * tile_words;
                    const uint32_t valid =
                        attn_top_managed_tile_valid_words(d_head, tile_words, dt);
                    if (valid == 0u || valid > tile_words) { return false; }
                    if ((tile_offset + valid) > d_head) { return false; }
                    ATTN_P11AF_MAINLINE_ACC_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                        const quant_act_t vv = quant_act_from_bits(sram[v_row_base + tile_offset + i]);
                        running_acc[tile_offset + i] += quant_acc_t(beta) * quant_acc_t(vv);
                    }
                }
            }
        }

        if (!have_state) {
            return false;
        }
        const softmax_inv_t inv_l = softmax_rcp_lut(running_l);
        ATTN_P11AF_MAINLINE_WRITEBACK_TILE_LOOP: for (uint32_t dt = 0u; dt < d_tile_count; ++dt) {
            const uint32_t tile_offset = dt * tile_words;
            const uint32_t valid = attn_top_managed_tile_valid_words(d_head, tile_words, dt);
            if (valid == 0u || valid > tile_words) { return false; }
            if ((tile_offset + valid) > d_head) { return false; }
            ATTN_P11AF_MAINLINE_WRITEBACK_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                const quant_acc_t out_val = running_acc[tile_offset + i] * quant_acc_t(inv_l);
                const u32_t out_bits = quant_bits_from_acc(out_val);
                sram[pre_row_base + head_col_base + tile_offset + i] = out_bits;
                sram[post_row_base + head_col_base + tile_offset + i] = out_bits;
                sram[out_row_base + head_col_base + tile_offset + i] = out_bits;
            }
        }
    }

    fallback_taken = false;
    return true;
}

} // namespace aecct
