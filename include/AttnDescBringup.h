#pragma once
// AttnDescBringup.h
// M9: Layer0 Attention shape / SRAM layout / stage switch（single source of truth）
#include "AecctTypes.h"
#include "LayerNormDesc.h"
#include "gen/SramMap.h"

namespace aecct {

    // Layer0 Attention shape（與 ModelShapes/trace 一致）
    static const unsigned ATTN_TOKEN_COUNT = (unsigned)LN_TOKEN_COUNT;
    static const unsigned ATTN_D_MODEL = (unsigned)LN_D_MODEL;
    static const unsigned ATTN_N_HEADS = (unsigned)N_HEAD;
    static const unsigned ATTN_D_HEAD = (unsigned)(ATTN_D_MODEL / ATTN_N_HEADS);
    static const unsigned ATTN_TENSOR_WORDS = (unsigned)(ATTN_TOKEN_COUNT * ATTN_D_MODEL);

    static_assert((ATTN_D_MODEL % ATTN_N_HEADS) == 0u, "ATTN_D_MODEL must be divisible by ATTN_N_HEADS");

    enum AttnStageMode : unsigned {
        ATTN_STAGE_QKV = 1u,     // M9a: Q/K/V (+ quant checkpoints)
        ATTN_STAGE_SCORES = 2u,  // M9b: score/softmax + pre/post concat
        ATTN_STAGE_OUT = 3u,     // M9c: out projection => attn_out
        ATTN_STAGE_FULL = 4u     // M9 full path
    };

    struct AttnCfg {
        u32_t token_count;
        u32_t d_model;
        u32_t n_heads;
        u32_t d_head;
    };

    struct AttnScratch {
        u32_t q_base_word;
        u32_t k_base_word;
        u32_t v_base_word;
        u32_t score_base_word;
        u32_t softmax_base_word;
        u32_t pre_concat_base_word;
        u32_t post_concat_base_word;
        u32_t q_act_q_base_word;
        u32_t k_act_q_base_word;
        u32_t v_act_q_base_word;
        u32_t q_sx_base_word;
    };

    // x_in 來自 LN 輸出，attn_out 寫回 X_PAGE0
    static const unsigned ATTN_X_IN_BASE_WORD_DEFAULT = (unsigned)LN_X_OUT_BASE_WORD_DEFAULT;
    static const unsigned ATTN_OUT_BASE_WORD_DEFAULT = (unsigned)sram_map::X_PAGE0_BASE_W;

    // Q/K 優先放 SCRATCH，其他暫存放 W_REGION（M9 bring-up 先對齊 checkpoint）
    static const unsigned ATTN_Q_BASE_WORD_DEFAULT = (unsigned)sram_map::BASE_SCR_K_W;
    static const unsigned ATTN_K_BASE_WORD_DEFAULT = (unsigned)sram_map::BASE_SCR_V_W;
    static const unsigned ATTN_V_BASE_WORD_DEFAULT = (unsigned)sram_map::W_REGION_BASE;

    static const unsigned ATTN_SCORE_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + ATTN_TENSOR_WORDS * 1u);
    static const unsigned ATTN_SOFTMAX_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + ATTN_TENSOR_WORDS * 2u);
    static const unsigned ATTN_PRE_CONCAT_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + ATTN_TENSOR_WORDS * 3u);
    static const unsigned ATTN_POST_CONCAT_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + ATTN_TENSOR_WORDS * 4u);

    static const unsigned ATTN_Q_ACT_Q_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + ATTN_TENSOR_WORDS * 5u);
    static const unsigned ATTN_K_ACT_Q_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + ATTN_TENSOR_WORDS * 6u);
    static const unsigned ATTN_V_ACT_Q_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + ATTN_TENSOR_WORDS * 7u);
    static const unsigned ATTN_Q_SX_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + ATTN_TENSOR_WORDS * 8u);

    static inline AttnScratch default_attn_scratch() {
        AttnScratch sc;
        sc.q_base_word = (u32_t)ATTN_Q_BASE_WORD_DEFAULT;
        sc.k_base_word = (u32_t)ATTN_K_BASE_WORD_DEFAULT;
        sc.v_base_word = (u32_t)ATTN_V_BASE_WORD_DEFAULT;
        sc.score_base_word = (u32_t)ATTN_SCORE_BASE_WORD_DEFAULT;
        sc.softmax_base_word = (u32_t)ATTN_SOFTMAX_BASE_WORD_DEFAULT;
        sc.pre_concat_base_word = (u32_t)ATTN_PRE_CONCAT_BASE_WORD_DEFAULT;
        sc.post_concat_base_word = (u32_t)ATTN_POST_CONCAT_BASE_WORD_DEFAULT;
        sc.q_act_q_base_word = (u32_t)ATTN_Q_ACT_Q_BASE_WORD_DEFAULT;
        sc.k_act_q_base_word = (u32_t)ATTN_K_ACT_Q_BASE_WORD_DEFAULT;
        sc.v_act_q_base_word = (u32_t)ATTN_V_ACT_Q_BASE_WORD_DEFAULT;
        sc.q_sx_base_word = (u32_t)ATTN_Q_SX_BASE_WORD_DEFAULT;
        return sc;
    }

} // namespace aecct

