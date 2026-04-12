#pragma once
// FfnDescBringup.h
// M10 Layer0 FFN shape and SRAM layout (single source of truth)

#include "AecctTypes.h"
#include "AttnDescBringup.h"
#include "LayerNormDesc.h"
#include "gen/ModelShapes.h"
#include "gen/SramMap.h"

namespace aecct {

    static const unsigned FFN_TOKEN_COUNT = (unsigned)ATTN_TOKEN_COUNT;
    static const unsigned FFN_D_MODEL = (unsigned)ATTN_D_MODEL;
    static const unsigned FFN_D_FFN = (unsigned)D_FFN;

    static const unsigned FFN_X_WORDS = (unsigned)(FFN_TOKEN_COUNT * FFN_D_MODEL);
    static const unsigned FFN_W2_INPUT_WORDS = (unsigned)(FFN_TOKEN_COUNT * FFN_D_FFN);
    static const unsigned FFN_W1_BIAS_WORDS = (unsigned)FFN_D_FFN;
    static const unsigned FFN_W1_WEIGHT_WORDS = (unsigned)(FFN_D_FFN * FFN_D_MODEL);
    static const unsigned FFN_W2_WEIGHT_WORDS = (unsigned)(FFN_D_MODEL * FFN_D_FFN);
    static const unsigned FFN_W2_BIAS_WORDS = (unsigned)FFN_D_MODEL;
    static const unsigned FFN_W1_OUT_WORDS = (unsigned)(FFN_TOKEN_COUNT * FFN_D_FFN);
    static const unsigned FFN_W2_OUT_WORDS = (unsigned)(FFN_TOKEN_COUNT * FFN_D_MODEL);

    enum FfnStageMode : unsigned {
        FFN_STAGE_W1 = 1u,
        FFN_STAGE_RELU = 2u,
        FFN_STAGE_W2 = 3u,
        FFN_STAGE_FULL = 4u
    };

    enum FfnFallbackPolicyFlags : unsigned {
        FFN_POLICY_NONE = 0u,
        FFN_POLICY_REQUIRE_W2_TOPFED = 1u,
        FFN_POLICY_REQUIRE_W1_TOPFED = 2u
    };

    enum FfnFallbackRejectStage : unsigned {
        FFN_REJECT_STAGE_NONE = 0u,
        FFN_REJECT_STAGE_W1 = 1u,
        FFN_REJECT_STAGE_W2 = 2u
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

    // W1/relu stay in runtime scratch; final write-back still targets the compatibility X_WORK slice in the transitional path
    static const unsigned FFN_RUNTIME_BASE_WORD_DEFAULT =
        (unsigned)align_up_words((uint32_t)ATTN_RUNTIME_END_EXCL_WORD_DEFAULT, (uint32_t)ALIGN_WORDS);
    static const unsigned FFN_W1_OUT_BASE_WORD_DEFAULT = (unsigned)FFN_RUNTIME_BASE_WORD_DEFAULT;
    static const unsigned FFN_RELU_OUT_BASE_WORD_DEFAULT = (unsigned)(FFN_W1_OUT_BASE_WORD_DEFAULT + FFN_W1_OUT_WORDS);
    static const unsigned FFN_W2_OUT_BASE_WORD_DEFAULT = (unsigned)(FFN_RELU_OUT_BASE_WORD_DEFAULT + FFN_W1_OUT_WORDS);
    static const unsigned FFN_ADD2_BASE_WORD_DEFAULT = (unsigned)(FFN_W2_OUT_BASE_WORD_DEFAULT + FFN_W2_OUT_WORDS);
    static const unsigned FFN_LN_OUT_BASE_WORD_DEFAULT = (unsigned)FFN_ADD2_BASE_WORD_DEFAULT;

    static const unsigned FFN_LN_GAMMA_BASE_WORD_DEFAULT = (unsigned)(FFN_ADD2_BASE_WORD_DEFAULT + FFN_W2_OUT_WORDS);
    static const unsigned FFN_LN_BETA_BASE_WORD_DEFAULT = (unsigned)(FFN_LN_GAMMA_BASE_WORD_DEFAULT + FFN_D_MODEL);
    static const unsigned FFN_RUNTIME_END_EXCL_WORD_DEFAULT = (unsigned)(FFN_LN_BETA_BASE_WORD_DEFAULT + FFN_D_MODEL);

    static_assert(
        (uint32_t)FFN_RUNTIME_BASE_WORD_DEFAULT >=
            ((uint32_t)sram_map::W_REGION_BASE + (uint32_t)sram_map::W_REGION_WORDS),
        "FFN runtime scratch base must not overlap W_REGION/PARAM");
    static_assert(
        (uint32_t)FFN_RUNTIME_END_EXCL_WORD_DEFAULT <=
            ((uint32_t)sram_map::BACKUP_RUNTIME_SCRATCH_BASE_W + (uint32_t)sram_map::BACKUP_RUNTIME_SCRATCH_WORDS),
        "FFN runtime scratch window exceeds backup runtime scratch region");

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

