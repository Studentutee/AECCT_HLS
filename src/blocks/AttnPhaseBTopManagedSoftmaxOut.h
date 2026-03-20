#pragma once
// AF bring-up helper: Top-managed Phase-B single-pass online softmax/output (local-only).
// This helper is additive to landed AC/AD/AE surfaces and does not alter them.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "AttnDescBringup.h"
#include "QuantDesc.h"
#include "SoftmaxApprox.h"

namespace aecct {

// P11AF_MAINLINE_HELPER_ENTRYPOINT
// Real design-side single-pass online softmax/output entrypoint used by Top mainline wiring.
// Consumes the score span produced by AE and writes pre/post/out for the target token row.
static inline bool attn_phaseb_top_managed_softmax_out_mainline(
    u32_t* sram,
    const AttnCfg& cfg,
    const AttnScratch& sc,
    u32_t token_idx,
    u32_t attn_out_base_word,
    bool& fallback_taken
) {
    fallback_taken = true;
    if (sram == (u32_t*)0) {
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

    const uint32_t pre_row_base = pre_base + token * d_model;
    const uint32_t post_row_base = post_base + token * d_model;
    const uint32_t out_row_base = out_base + token * d_model;

    ATTN_P11AF_HEAD_LOOP: for (uint32_t h = 0u; h < n_heads; ++h) {
        const uint32_t head_col_base = h * d_head;
        const uint32_t score_head_base = score_base + h * token_count;

        softmax_score_t running_max = softmax_score_t(0);
        softmax_sum_t running_l = softmax_sum_t(0);
        quant_acc_t running_acc[ATTN_D_MODEL];
        for (uint32_t d = 0u; d < d_head; ++d) {
            running_acc[d] = quant_acc_t(0);
        }

        bool have_state = false;
        ATTN_P11AF_KEY_TOKEN_LOOP: for (uint32_t j = 0u; j < token_count; ++j) {
            const fp32_t score_fp = fp32_from_bits(sram[score_head_base + j]);
            const softmax_score_t score =
                score_fp.template convert_to_ac_fixed<18, 6, true, AC_RND, AC_SAT>(false);

            const uint32_t v_row_base = v_base + j * d_model + head_col_base;
            if (!have_state) {
                running_max = score;
                running_l = softmax_sum_t(1);
                ATTN_P11AF_INIT_ACC_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
                    const quant_act_t vv = quant_act_from_bits(sram[v_row_base + d]);
                    running_acc[d] = quant_acc_t(vv);
                }
                have_state = true;
                continue;
            }

            if (score > running_max) {
                const softmax_x_t old_minus_new = softmax_x_t(running_max - score);
                const softmax_exp_t alpha = softmax_exp_lut(old_minus_new);
                running_l = softmax_sum_t(running_l * softmax_sum_t(alpha)) + softmax_sum_t(1);
                ATTN_P11AF_RENORM_ACC_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
                    const quant_act_t vv = quant_act_from_bits(sram[v_row_base + d]);
                    running_acc[d] =
                        quant_acc_t(running_acc[d] * quant_acc_t(alpha)) + quant_acc_t(vv);
                }
                running_max = score;
            } else {
                const softmax_x_t score_minus_old = softmax_x_t(score - running_max);
                const softmax_exp_t beta = softmax_exp_lut(score_minus_old);
                running_l += softmax_sum_t(beta);
                ATTN_P11AF_ACC_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
                    const quant_act_t vv = quant_act_from_bits(sram[v_row_base + d]);
                    running_acc[d] += quant_acc_t(beta) * quant_acc_t(vv);
                }
            }
        }

        if (!have_state) {
            return false;
        }
        const softmax_inv_t inv_l = softmax_rcp_lut(running_l);
        ATTN_P11AF_OUTPUT_COL_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
            const quant_acc_t out_val = running_acc[d] * quant_acc_t(inv_l);
            sram[pre_row_base + head_col_base + d] = quant_bits_from_acc(out_val);
        }

    }

    ATTN_P11AF_POST_ROW_COPY_LOOP: for (uint32_t i = 0u; i < d_model; ++i) {
        sram[post_row_base + i] = sram[pre_row_base + i];
    }
    ATTN_P11AF_OUT_ROW_WRITEBACK_LOOP: for (uint32_t i = 0u; i < d_model; ++i) {
        sram[out_row_base + i] = sram[post_row_base + i];
    }

    fallback_taken = false;
    return true;
}

} // namespace aecct
