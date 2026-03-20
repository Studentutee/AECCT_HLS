#pragma once
// AE bring-up helper: Top-managed Phase-B QK/score staging (local-only).
// This helper is additive to landed AC/AD surfaces and does not alter them.

#include <cstdint>

#include "AecctTypes.h"
#include "AttnDescBringup.h"
#include "QuantDesc.h"

namespace aecct {

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

// P11AE_MAINLINE_HELPER_ENTRYPOINT
// Real design-side QK/score entrypoint used by Top mainline wiring.
// Writes score rows into the current score span consumed by AF.
static inline bool attn_phaseb_top_managed_qk_score_mainline(
    u32_t* sram,
    const AttnCfg& cfg,
    const AttnScratch& sc,
    u32_t token_idx,
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

    const uint32_t q_base = (uint32_t)sc.q_base_word.to_uint();
    const uint32_t k_base = (uint32_t)sc.k_base_word.to_uint();
    const uint32_t score_base = (uint32_t)sc.score_base_word.to_uint();
    const uint32_t q_row_base = q_base + token * d_model;
    const quant_acc_t inv_sqrt_d_head = attn_phaseb_inv_sqrt_d_head(d_head);

    ATTN_P11AE_HEAD_LOOP: for (uint32_t h = 0u; h < n_heads; ++h) {
        const uint32_t head_col_base = h * d_head;
        const uint32_t score_head_base = score_base + h * token_count;
        ATTN_P11AE_KEY_TOKEN_LOOP: for (uint32_t j = 0u; j < token_count; ++j) {
            const uint32_t k_row_base = k_base + j * d_model + head_col_base;
            quant_acc_t dot = 0;
            ATTN_P11AE_DOT_COL_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
                const quant_act_t qv = quant_act_from_bits(sram[q_row_base + head_col_base + d]);
                const quant_act_t kv = quant_act_from_bits(sram[k_row_base + d]);
                dot += quant_acc_t(qv) * quant_acc_t(kv);
            }
            const quant_acc_t scaled = dot * inv_sqrt_d_head;
            sram[score_head_base + j] = quant_bits_from_acc(scaled);
        }
    }

    fallback_taken = false;
    return true;
}

} // namespace aecct

