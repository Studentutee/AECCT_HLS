#pragma once
// Layer0 attention block.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "AttnDescBringup.h"
#include "QuantDesc.h"
#include "SoftmaxApprox.h"

namespace aecct {

static inline quant_acc_t attn_inv_sqrt_d_head(uint32_t d_head) {
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

template<unsigned STAGE_MODE>
static inline void AttnLayer0(
    u32_t* sram,
    const AttnCfg& cfg,
    u32_t x_in_base_word,
    u32_t attn_out_base_word,
    const AttnScratch& sc
) {
    uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t tensor_words = token_count * d_model;

    uint32_t x_in_base = (uint32_t)x_in_base_word.to_uint();
    uint32_t q_base = (uint32_t)sc.q_base_word.to_uint();
    uint32_t k_base = (uint32_t)sc.k_base_word.to_uint();
    uint32_t v_base = (uint32_t)sc.v_base_word.to_uint();
    uint32_t score_base = (uint32_t)sc.score_base_word.to_uint();
    uint32_t softmax_base = (uint32_t)sc.softmax_base_word.to_uint();
    uint32_t pre_base = (uint32_t)sc.pre_concat_base_word.to_uint();
    uint32_t post_base = (uint32_t)sc.post_concat_base_word.to_uint();
    uint32_t out_base = (uint32_t)attn_out_base_word.to_uint();
    uint32_t n_heads = (uint32_t)cfg.n_heads.to_uint();
    uint32_t d_head = (uint32_t)cfg.d_head.to_uint();
    if (n_heads == 0u) { n_heads = (uint32_t)ATTN_N_HEADS; }
    if (d_head == 0u) { d_head = d_model / n_heads; }
    quant_acc_t inv_sqrt_d_head = attn_inv_sqrt_d_head(d_head);

    if constexpr (STAGE_MODE == ATTN_STAGE_QKV || STAGE_MODE == ATTN_STAGE_FULL) {
        for (uint32_t i = 0; i < tensor_words; ++i) {
            u32_t x = sram[x_in_base + i];
            sram[q_base + i] = x;
            sram[k_base + i] = x;
            sram[v_base + i] = x;
            sram[(uint32_t)sc.q_act_q_base_word.to_uint() + i] = x;
            sram[(uint32_t)sc.k_act_q_base_word.to_uint() + i] = x;
            sram[(uint32_t)sc.v_act_q_base_word.to_uint() + i] = x;
        }
        sram[(uint32_t)sc.q_sx_base_word.to_uint()] = bits_from_fp32(fp32_one());
    }

    if constexpr (STAGE_MODE == ATTN_STAGE_SCORES || STAGE_MODE == ATTN_STAGE_FULL) {
        softmax_score_t score_row[N_NODES];
        softmax_prob_t prob_row[N_NODES];
        for (uint32_t t = 0; t < token_count; ++t) {
            for (uint32_t h = 0; h < n_heads; ++h) {
                uint32_t head_col_base = h * d_head;

                for (uint32_t j = 0; j < token_count; ++j) {
                    quant_acc_t dot = 0;
                    uint32_t q_row = q_base + t * d_model + head_col_base;
                    uint32_t k_row = k_base + j * d_model + head_col_base;
                    for (uint32_t d = 0; d < d_head; ++d) {
                        quant_act_t qv = quant_act_from_bits(sram[q_row + d]);
                        quant_act_t kv = quant_act_from_bits(sram[k_row + d]);
                        dot += quant_acc_t(qv) * quant_acc_t(kv);
                    }
                    score_row[j] = softmax_score_t(dot * inv_sqrt_d_head);
                }

                SoftmaxApprox<N_NODES>(score_row, prob_row, token_count);

                if (t == 0u && h == 0u) {
                    for (uint32_t j = 0; j < token_count; ++j) {
                        fp32_t score_fp(score_row[j]);
                        fp32_t prob_fp(prob_row[j]);
                        sram[score_base + j] = bits_from_fp32(score_fp);
                        sram[softmax_base + j] = bits_from_fp32(prob_fp);
                    }
                }

                for (uint32_t d = 0; d < d_head; ++d) {
                    quant_acc_t acc = 0;
                    for (uint32_t j = 0; j < token_count; ++j) {
                        uint32_t v_idx = v_base + j * d_model + head_col_base + d;
                        quant_act_t vv = quant_act_from_bits(sram[v_idx]);
                        acc += quant_acc_t(prob_row[j]) * quant_acc_t(vv);
                    }
                    uint32_t out_idx = pre_base + t * d_model + head_col_base + d;
                    sram[out_idx] = quant_bits_from_acc(acc);
                }
            }
        }

        for (uint32_t i = 0; i < tensor_words; ++i) {
            sram[post_base + i] = sram[pre_base + i];
        }
    }

    if constexpr (STAGE_MODE == ATTN_STAGE_OUT || STAGE_MODE == ATTN_STAGE_FULL) {
        for (uint32_t i = 0; i < tensor_words; ++i) {
            sram[out_base + i] = sram[post_base + i];
        }
    }
}

} // namespace aecct
