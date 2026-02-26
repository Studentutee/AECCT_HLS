#pragma once
// FfnDescBringup.h
// M10 Layer0 FFN shape and SRAM layout (single source of truth)

#include "AecctTypes.h"
#include "AttnDescBringup.h"
#include "LayerNormDesc.h"
#include "ModelShapes.h"
#include "SramMap.h"

namespace aecct {

    static const unsigned FFN_TOKEN_COUNT = (unsigned)ATTN_TOKEN_COUNT;
    static const unsigned FFN_D_MODEL = (unsigned)ATTN_D_MODEL;
    static const unsigned FFN_D_FFN = (unsigned)D_FFN;

    static const unsigned FFN_X_WORDS = (unsigned)(FFN_TOKEN_COUNT * FFN_D_MODEL);
    static const unsigned FFN_W1_OUT_WORDS = (unsigned)(FFN_TOKEN_COUNT * FFN_D_FFN);
    static const unsigned FFN_W2_OUT_WORDS = (unsigned)(FFN_TOKEN_COUNT * FFN_D_MODEL);

    enum FfnStageMode : unsigned {
        FFN_STAGE_W1 = 1u,
        FFN_STAGE_RELU = 2u,
        FFN_STAGE_W2 = 3u,
        FFN_STAGE_FULL = 4u
    };

    struct FfnCfg {
        u32_t token_count;
        u32_t d_model;
        u32_t d_ffn;
    };

    struct FfnScratch {
        u32_t w1_out_base_word;
        u32_t relu_out_base_word;
        u32_t w2_out_base_word;
        u32_t add2_base_word;
        u32_t ln_out_base_word;
        u32_t ln_gamma_base_word;
        u32_t ln_beta_base_word;
    };

    static const unsigned FFN_X_IN_BASE_WORD_DEFAULT = (unsigned)ATTN_OUT_BASE_WORD_DEFAULT;

    // W1/relu 需要 9600 words，放在 W_REGION；w2/add2 放在 SCRATCH；LN output 放 X_PAGE1
    static const unsigned FFN_W1_OUT_BASE_WORD_DEFAULT = (unsigned)sram_map::W_REGION_BASE;
    static const unsigned FFN_RELU_OUT_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + FFN_W1_OUT_WORDS);
    static const unsigned FFN_W2_OUT_BASE_WORD_DEFAULT = (unsigned)sram_map::BASE_SCR_K_W;
    static const unsigned FFN_ADD2_BASE_WORD_DEFAULT = (unsigned)sram_map::BASE_SCR_V_W;
    static const unsigned FFN_LN_OUT_BASE_WORD_DEFAULT = (unsigned)sram_map::X_PAGE1_BASE_W;

    static const unsigned FFN_LN_GAMMA_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + FFN_W1_OUT_WORDS + FFN_W1_OUT_WORDS);
    static const unsigned FFN_LN_BETA_BASE_WORD_DEFAULT = (unsigned)(FFN_LN_GAMMA_BASE_WORD_DEFAULT + FFN_D_MODEL);

    static inline FfnScratch default_ffn_scratch() {
        FfnScratch sc;
        sc.w1_out_base_word = (u32_t)FFN_W1_OUT_BASE_WORD_DEFAULT;
        sc.relu_out_base_word = (u32_t)FFN_RELU_OUT_BASE_WORD_DEFAULT;
        sc.w2_out_base_word = (u32_t)FFN_W2_OUT_BASE_WORD_DEFAULT;
        sc.add2_base_word = (u32_t)FFN_ADD2_BASE_WORD_DEFAULT;
        sc.ln_out_base_word = (u32_t)FFN_LN_OUT_BASE_WORD_DEFAULT;
        sc.ln_gamma_base_word = (u32_t)FFN_LN_GAMMA_BASE_WORD_DEFAULT;
        sc.ln_beta_base_word = (u32_t)FFN_LN_BETA_BASE_WORD_DEFAULT;
        return sc;
    }

} // namespace aecct
