#pragma once
// One-layer integration wrapper: attention, FFN, residual add, LayerNorm.
// Boundary notes:
// - Inputs are fully owned by Top (base words, scratch layout, and prebuilt flags).
// - This block delegates attention internals to AttnLayer0 and does local composition.
// - Shared SRAM lifetime/arbitration ownership stays in Top.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "AecctRanges.h"
#include "AecctUtil.h"
#include "AttnLayer0.h"
#include "FFNLayer0.h"
#include "LayerNormBlock.h"
#include "LayerNormDesc.h"
#include "LayerParamBringup.h"
#include "LayerScratchDesc.h"
#include "gen/WeightStreamOrder.h"

namespace aecct {

struct TransformerLayerContract {
    bool start;
    bool done;
    bool err_valid;
    u16_t err_code;
    TokenRange token_range;
    TileRange tile_range;
    PhaseId phase_id;
    u32_t x_work_base_word;
    u32_t scratch_base_word;
    u32_t w_base_word;
};

// Contract helper for Top-owned orchestration metadata.
static inline void clear_transformer_layer_contract(TransformerLayerContract& c) {
    c.start = false;
    c.done = false;
    c.err_valid = false;
    c.err_code = 0;
    c.token_range = make_token_range(0, 0);
    c.tile_range = make_tile_range(0, 0);
    c.phase_id = PHASE_LAYER0;
    c.x_work_base_word = 0;
    c.scratch_base_word = 0;
    c.w_base_word = 0;
}

struct CfgRegs {
    u32_t d_model;
    u32_t n_heads;
    u32_t d_ffn;
    u32_t n_layers;
};

static inline void load_layer_sublayer1_norm_params(
    u32_t* sram,
    uint32_t param_base_word,
    uint32_t layer_id,
    uint32_t gamma_base,
    uint32_t beta_base,
    uint32_t d_model
) {
    const uint32_t norm_w_id = (layer_id == 0u) ? 43u : 63u;
    const uint32_t norm_b_id = (layer_id == 0u) ? 7u : 15u;
    const uint32_t norm_w_base = param_base_word + kParamMeta[norm_w_id].offset_w;
    const uint32_t norm_b_base = param_base_word + kParamMeta[norm_b_id].offset_w;

    TRANSFORMER_LAYER_SUBLAYER1_NORM_PARAM_COPY_LOOP: for (uint32_t c = 0; c < d_model; ++c) {
        sram[gamma_base + c] = sram[norm_w_base + c];
        sram[beta_base + c] = sram[norm_b_base + c];
    }
}

// P00-011AN/P00-011AO: first deep Attn+FFN boundary bridges for Catapult-facing progress.
// This variant keeps Attn/FFN first deep entries array-shaped while preserving
// accepted core compute semantics.
template<uint32_t SRAM_WORDS>
static inline void TransformerLayerTopManagedAttnBridge(
    u32_t (&sram_window)[SRAM_WORDS],
    const CfgRegs& cfg,
    u32_t layer_id,
    u32_t x_in_base_word,
    u32_t x_out_base_word,
    const LayerScratch& sc,
    const LayerParamBase& pb,
    bool kv_prebuilt_from_top_managed = false,
    bool q_prebuilt_from_top_managed = false,
    bool score_prebuilt_from_top_managed = false,
    bool out_prebuilt_from_top_managed = false,
    bool sublayer1_norm_preloaded_by_top = false
) {
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t n_heads = (uint32_t)cfg.n_heads.to_uint();
    uint32_t d_ffn = (uint32_t)cfg.d_ffn.to_uint();
    if (d_model == 0u) { d_model = (uint32_t)ATTN_D_MODEL; }
    if (n_heads == 0u) { n_heads = (uint32_t)ATTN_N_HEADS; }
    if (d_ffn == 0u) { d_ffn = (uint32_t)FFN_D_FFN; }

    AttnCfg attn_cfg;
    attn_cfg.token_count = (u32_t)ATTN_TOKEN_COUNT;
    attn_cfg.d_model = (u32_t)d_model;
    attn_cfg.n_heads = (u32_t)n_heads;
    attn_cfg.d_head = (u32_t)(d_model / n_heads);

    AttnLayer0TopManagedWindowBridge<ATTN_STAGE_FULL>(
        sram_window,
        attn_cfg,
        x_in_base_word,
        sc.attn_out_base_word,
        sc.attn,
        (u32_t)0,
        kv_prebuilt_from_top_managed,
        q_prebuilt_from_top_managed,
        score_prebuilt_from_top_managed,
        out_prebuilt_from_top_managed
    );

    FfnCfg ffn_cfg;
    ffn_cfg.token_count = (u32_t)FFN_TOKEN_COUNT;
    ffn_cfg.d_model = (u32_t)d_model;
    ffn_cfg.d_ffn = (u32_t)d_ffn;

    u32_t topfed_ffn_x_words[FFN_X_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_X_INIT_BRIDGE_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_X_WORDS; ++i) {
        topfed_ffn_x_words[i] = 0;
    }
    uint32_t ffn_token_count = (uint32_t)ffn_cfg.token_count.to_uint();
    uint32_t ffn_d_model = (uint32_t)ffn_cfg.d_model.to_uint();
    if (ffn_token_count == 0u) { ffn_token_count = (uint32_t)FFN_TOKEN_COUNT; }
    if (ffn_d_model == 0u) { ffn_d_model = (uint32_t)FFN_D_MODEL; }
    uint32_t ffn_x_words = ffn_token_count * ffn_d_model;
    if (ffn_x_words > (uint32_t)FFN_X_WORDS) {
        ffn_x_words = (uint32_t)FFN_X_WORDS;
    }
    const uint32_t ffn_x_base = (uint32_t)sc.attn_out_base_word.to_uint();
    TRANSFORMER_LAYER_FFN_TOPFED_X_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < ffn_x_words; ++i) {
        topfed_ffn_x_words[i] = sram_window[ffn_x_base + i];
    }
    const bool use_layer1 = ((uint32_t)layer_id.to_uint() == 1u);
    const uint32_t w1_bias_id = use_layer1 ? 12u : 4u;
    const uint32_t w1_weight_id = use_layer1 ? 56u : 36u;
    const uint32_t param_base = (uint32_t)pb.param_base_word.to_uint();
    const uint32_t w1_bias_base = param_base + kParamMeta[w1_bias_id].offset_w;
    const uint32_t w1_weight_base = param_base + kParamMeta[w1_weight_id].offset_w;
    u32_t topfed_ffn_w1_words[FFN_W1_WEIGHT_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W1_INIT_BRIDGE_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W1_WEIGHT_WORDS; ++i) {
        topfed_ffn_w1_words[i] = 0;
    }
    uint32_t w1_weight_words = ffn_d_model * (uint32_t)ffn_cfg.d_ffn.to_uint();
    if (w1_weight_words == 0u) {
        w1_weight_words = (uint32_t)FFN_W1_WEIGHT_WORDS;
    }
    if (w1_weight_words > (uint32_t)FFN_W1_WEIGHT_WORDS) {
        w1_weight_words = (uint32_t)FFN_W1_WEIGHT_WORDS;
    }
    TRANSFORMER_LAYER_FFN_TOPFED_W1_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < w1_weight_words; ++i) {
        topfed_ffn_w1_words[i] = sram_window[w1_weight_base + i];
    }
    u32_t topfed_ffn_w1_bias_words[FFN_W1_BIAS_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W1_BIAS_INIT_BRIDGE_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W1_BIAS_WORDS; ++i) {
        topfed_ffn_w1_bias_words[i] = 0;
    }
    uint32_t w1_bias_words = (uint32_t)ffn_cfg.d_ffn.to_uint();
    if (w1_bias_words == 0u) {
        w1_bias_words = (uint32_t)FFN_W1_BIAS_WORDS;
    }
    if (w1_bias_words > (uint32_t)FFN_W1_BIAS_WORDS) {
        w1_bias_words = (uint32_t)FFN_W1_BIAS_WORDS;
    }
    TRANSFORMER_LAYER_FFN_TOPFED_W1_BIAS_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < w1_bias_words; ++i) {
        topfed_ffn_w1_bias_words[i] = sram_window[w1_bias_base + i];
    }

    // Stage-split FFN dispatch keeps caller ownership explicit for payload descriptors.
    FFNLayer0TopManagedWindowBridge<FFN_STAGE_W1>(
        sram_window,
        ffn_cfg,
        sc.attn_out_base_word,
        sc.ffn,
        pb.param_base_word,
        layer_id,
        topfed_ffn_x_words,
        topfed_ffn_w1_words,
        (u32_t)w1_weight_words,
        0,
        (u32_t)0u,
        0,
        (u32_t)0u,
        0,
        (u32_t)0u,
        (u32_t)FFN_POLICY_REQUIRE_W1_TOPFED,
        0,
        0,
        (u32_t)ffn_x_words,
        topfed_ffn_w1_bias_words,
        (u32_t)w1_bias_words
    );
    FFNLayer0TopManagedWindowBridge<FFN_STAGE_RELU>(
        sram_window,
        ffn_cfg,
        sc.attn_out_base_word,
        sc.ffn,
        pb.param_base_word,
        layer_id
    );

    uint32_t ffn_d_ffn = (uint32_t)ffn_cfg.d_ffn.to_uint();
    if (ffn_d_ffn == 0u) { ffn_d_ffn = (uint32_t)FFN_D_FFN; }
    const uint32_t relu_base = (uint32_t)sc.ffn.relu_out_base_word.to_uint();
    u32_t topfed_ffn_w2_input_words[FFN_W2_INPUT_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W2_INPUT_INIT_BRIDGE_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W2_INPUT_WORDS; ++i) {
        topfed_ffn_w2_input_words[i] = 0;
    }
    uint32_t w2_input_words = ffn_token_count * ffn_d_ffn;
    if (w2_input_words > (uint32_t)FFN_W2_INPUT_WORDS) {
        w2_input_words = (uint32_t)FFN_W2_INPUT_WORDS;
    }
    TRANSFORMER_LAYER_FFN_TOPFED_W2_INPUT_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < w2_input_words; ++i) {
        topfed_ffn_w2_input_words[i] = sram_window[relu_base + i];
    }
    const uint32_t w2_weight_id = use_layer1 ? 59u : 39u;
    const uint32_t w2_bias_id = use_layer1 ? 13u : 5u;
    const uint32_t w2_weight_base = param_base + kParamMeta[w2_weight_id].offset_w;
    const uint32_t w2_bias_base = param_base + kParamMeta[w2_bias_id].offset_w;
    u32_t topfed_ffn_w2_words[FFN_W2_WEIGHT_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W2_WEIGHT_INIT_BRIDGE_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W2_WEIGHT_WORDS; ++i) {
        topfed_ffn_w2_words[i] = 0;
    }
    uint32_t w2_weight_words = ffn_d_model * ffn_d_ffn;
    if (w2_weight_words > (uint32_t)FFN_W2_WEIGHT_WORDS) {
        w2_weight_words = (uint32_t)FFN_W2_WEIGHT_WORDS;
    }
    TRANSFORMER_LAYER_FFN_TOPFED_W2_WEIGHT_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < w2_weight_words; ++i) {
        topfed_ffn_w2_words[i] = sram_window[w2_weight_base + i];
    }
    u32_t topfed_ffn_w2_bias_words[FFN_W2_BIAS_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_INIT_BRIDGE_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W2_BIAS_WORDS; ++i) {
        topfed_ffn_w2_bias_words[i] = 0;
    }
    uint32_t w2_bias_words = ffn_d_model;
    if (w2_bias_words > (uint32_t)FFN_W2_BIAS_WORDS) {
        w2_bias_words = (uint32_t)FFN_W2_BIAS_WORDS;
    }
    TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < w2_bias_words; ++i) {
        topfed_ffn_w2_bias_words[i] = sram_window[w2_bias_base + i];
    }

    FFNLayer0TopManagedWindowBridge<FFN_STAGE_W2>(
        sram_window,
        ffn_cfg,
        sc.attn_out_base_word,
        sc.ffn,
        pb.param_base_word,
        layer_id,
        0,
        0,
        (u32_t)0u,
        topfed_ffn_w2_input_words,
        (u32_t)w2_input_words,
        topfed_ffn_w2_words,
        (u32_t)w2_weight_words,
        topfed_ffn_w2_bias_words,
        (u32_t)w2_bias_words,
        (u32_t)FFN_POLICY_REQUIRE_W2_TOPFED
    );

    uint32_t residual_base = (uint32_t)sc.attn_out_base_word.to_uint();
    uint32_t w2_base = (uint32_t)sc.ffn.w2_out_base_word.to_uint();
    uint32_t add2_base = (uint32_t)sc.ffn.add2_base_word.to_uint();
    uint32_t words = (uint32_t)FFN_X_WORDS;
    TRANSFORMER_LAYER_FFN_RESIDUAL_ADD_BRIDGE_LOOP: for (uint32_t i = 0; i < words; ++i) {
        fp32_t x = fp32_from_bits(sram_window[residual_base + i]);
        fp32_t y = fp32_from_bits(sram_window[w2_base + i]);
        sram_window[add2_base + i] = bits_from_fp32(x + y);
    }

    uint32_t gamma_base = (uint32_t)sc.ffn.ln_gamma_base_word.to_uint();
    uint32_t beta_base = (uint32_t)sc.ffn.ln_beta_base_word.to_uint();
    if (!sublayer1_norm_preloaded_by_top) {
        // Legacy fallback: keep in-block param fetch only when Top did not preload.
        load_layer_sublayer1_norm_params(
            sram_window,
            (uint32_t)pb.param_base_word.to_uint(),
            (uint32_t)layer_id.to_uint(),
            gamma_base,
            beta_base,
            d_model
        );
    }

    LayerNormCfg ln_cfg;
    ln_cfg.token_count = (u32_t)FFN_TOKEN_COUNT;
    ln_cfg.d_model = (u32_t)d_model;
    ln_cfg.eps_bits = LN_EPS_BITS;

    LayerNormBlockTopManagedWindowBridge(
        sram_window,
        ln_cfg,
        (u32_t)add2_base,
        x_out_base_word,
        (u32_t)gamma_base,
        (u32_t)beta_base,
        PHASE_LAYER0
    );
}

// Integration boundary for one logical layer.
// Accepts Top-managed X_WORK/SCRATCH/W_REGION base words and optional prebuilt Q/KV flags.
// Delegates attention compute to AttnLayer0, then completes FFN + residual + LN handoff.
// Does not own Top FSM dispatch, global SRAM policy, or fallback-policy definition.
static inline void TransformerLayer(
    u32_t* sram,
    const CfgRegs& cfg,
    u32_t layer_id,
    u32_t x_in_base_word,
    u32_t x_out_base_word,
    const LayerScratch& sc,
    const LayerParamBase& pb,
    bool kv_prebuilt_from_top_managed = false,
    bool q_prebuilt_from_top_managed = false,
    bool score_prebuilt_from_top_managed = false,
    bool out_prebuilt_from_top_managed = false,
    bool sublayer1_norm_preloaded_by_top = false
) {
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t n_heads = (uint32_t)cfg.n_heads.to_uint();
    uint32_t d_ffn = (uint32_t)cfg.d_ffn.to_uint();
    if (d_model == 0u) { d_model = (uint32_t)ATTN_D_MODEL; }
    if (n_heads == 0u) { n_heads = (uint32_t)ATTN_N_HEADS; }
    if (d_ffn == 0u) { d_ffn = (uint32_t)FFN_D_FFN; }

    AttnCfg attn_cfg;
    attn_cfg.token_count = (u32_t)ATTN_TOKEN_COUNT;
    attn_cfg.d_model = (u32_t)d_model;
    attn_cfg.n_heads = (u32_t)n_heads;
    attn_cfg.d_head = (u32_t)(d_model / n_heads);

    // AttnLayer0 consumes Top-selected boundaries and prebuilt-flag handoff from Top.
    AttnLayer0<ATTN_STAGE_FULL>(
        sram,
        attn_cfg,
        x_in_base_word,
        sc.attn_out_base_word,
        sc.attn,
        (u32_t)0,
        kv_prebuilt_from_top_managed,
        q_prebuilt_from_top_managed,
        score_prebuilt_from_top_managed,
        out_prebuilt_from_top_managed
    );

    FfnCfg ffn_cfg;
    ffn_cfg.token_count = (u32_t)FFN_TOKEN_COUNT;
    ffn_cfg.d_model = (u32_t)d_model;
    ffn_cfg.d_ffn = (u32_t)d_ffn;

    u32_t topfed_ffn_x_words[FFN_X_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_X_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_X_WORDS; ++i) {
        topfed_ffn_x_words[i] = 0;
    }
    uint32_t ffn_token_count = (uint32_t)ffn_cfg.token_count.to_uint();
    uint32_t ffn_d_model = (uint32_t)ffn_cfg.d_model.to_uint();
    if (ffn_token_count == 0u) { ffn_token_count = (uint32_t)FFN_TOKEN_COUNT; }
    if (ffn_d_model == 0u) { ffn_d_model = (uint32_t)FFN_D_MODEL; }
    uint32_t ffn_x_words = ffn_token_count * ffn_d_model;
    if (ffn_x_words > (uint32_t)FFN_X_WORDS) {
        ffn_x_words = (uint32_t)FFN_X_WORDS;
    }
    const uint32_t ffn_x_base = (uint32_t)sc.attn_out_base_word.to_uint();
    TRANSFORMER_LAYER_FFN_TOPFED_X_PRELOAD_LOOP: for (uint32_t i = 0u; i < ffn_x_words; ++i) {
        topfed_ffn_x_words[i] = sram[ffn_x_base + i];
    }
    const bool use_layer1 = ((uint32_t)layer_id.to_uint() == 1u);
    const uint32_t w1_bias_id = use_layer1 ? 12u : 4u;
    const uint32_t w1_weight_id = use_layer1 ? 56u : 36u;
    const uint32_t param_base = (uint32_t)pb.param_base_word.to_uint();
    const uint32_t w1_bias_base = param_base + kParamMeta[w1_bias_id].offset_w;
    const uint32_t w1_weight_base = param_base + kParamMeta[w1_weight_id].offset_w;
    u32_t topfed_ffn_w1_words[FFN_W1_WEIGHT_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W1_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W1_WEIGHT_WORDS; ++i) {
        topfed_ffn_w1_words[i] = 0;
    }
    uint32_t w1_weight_words = ffn_d_model * (uint32_t)ffn_cfg.d_ffn.to_uint();
    if (w1_weight_words == 0u) {
        w1_weight_words = (uint32_t)FFN_W1_WEIGHT_WORDS;
    }
    if (w1_weight_words > (uint32_t)FFN_W1_WEIGHT_WORDS) {
        w1_weight_words = (uint32_t)FFN_W1_WEIGHT_WORDS;
    }
    TRANSFORMER_LAYER_FFN_TOPFED_W1_PRELOAD_LOOP: for (uint32_t i = 0u; i < w1_weight_words; ++i) {
        topfed_ffn_w1_words[i] = sram[w1_weight_base + i];
    }
    u32_t topfed_ffn_w1_bias_words[FFN_W1_BIAS_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W1_BIAS_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W1_BIAS_WORDS; ++i) {
        topfed_ffn_w1_bias_words[i] = 0;
    }
    uint32_t w1_bias_words = (uint32_t)ffn_cfg.d_ffn.to_uint();
    if (w1_bias_words == 0u) {
        w1_bias_words = (uint32_t)FFN_W1_BIAS_WORDS;
    }
    if (w1_bias_words > (uint32_t)FFN_W1_BIAS_WORDS) {
        w1_bias_words = (uint32_t)FFN_W1_BIAS_WORDS;
    }
    TRANSFORMER_LAYER_FFN_TOPFED_W1_BIAS_PRELOAD_LOOP: for (uint32_t i = 0u; i < w1_bias_words; ++i) {
        topfed_ffn_w1_bias_words[i] = sram[w1_bias_base + i];
    }

    // Stage-split FFN dispatch keeps caller ownership explicit for payload descriptors.
    FFNLayer0<FFN_STAGE_W1>(
        sram,
        ffn_cfg,
        sc.attn_out_base_word,
        sc.ffn,
        pb.param_base_word,
        layer_id,
        topfed_ffn_x_words,
        topfed_ffn_w1_words,
        (u32_t)w1_weight_words,
        0,
        (u32_t)0u,
        0,
        (u32_t)0u,
        0,
        (u32_t)0u,
        (u32_t)FFN_POLICY_REQUIRE_W1_TOPFED,
        0,
        0,
        (u32_t)ffn_x_words,
        topfed_ffn_w1_bias_words,
        (u32_t)w1_bias_words
    );
    FFNLayer0<FFN_STAGE_RELU>(
        sram,
        ffn_cfg,
        sc.attn_out_base_word,
        sc.ffn,
        pb.param_base_word,
        layer_id
    );

    uint32_t ffn_d_ffn = (uint32_t)ffn_cfg.d_ffn.to_uint();
    if (ffn_d_ffn == 0u) { ffn_d_ffn = (uint32_t)FFN_D_FFN; }
    const uint32_t relu_base = (uint32_t)sc.ffn.relu_out_base_word.to_uint();
    u32_t topfed_ffn_w2_input_words[FFN_W2_INPUT_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W2_INPUT_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W2_INPUT_WORDS; ++i) {
        topfed_ffn_w2_input_words[i] = 0;
    }
    uint32_t w2_input_words = ffn_token_count * ffn_d_ffn;
    if (w2_input_words > (uint32_t)FFN_W2_INPUT_WORDS) {
        w2_input_words = (uint32_t)FFN_W2_INPUT_WORDS;
    }
    TRANSFORMER_LAYER_FFN_TOPFED_W2_INPUT_PRELOAD_LOOP: for (uint32_t i = 0u; i < w2_input_words; ++i) {
        topfed_ffn_w2_input_words[i] = sram[relu_base + i];
    }
    const uint32_t w2_weight_id = use_layer1 ? 59u : 39u;
    const uint32_t w2_bias_id = use_layer1 ? 13u : 5u;
    const uint32_t w2_weight_base = param_base + kParamMeta[w2_weight_id].offset_w;
    const uint32_t w2_bias_base = param_base + kParamMeta[w2_bias_id].offset_w;
    u32_t topfed_ffn_w2_words[FFN_W2_WEIGHT_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W2_WEIGHT_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W2_WEIGHT_WORDS; ++i) {
        topfed_ffn_w2_words[i] = 0;
    }
    uint32_t w2_weight_words = ffn_d_model * ffn_d_ffn;
    if (w2_weight_words > (uint32_t)FFN_W2_WEIGHT_WORDS) {
        w2_weight_words = (uint32_t)FFN_W2_WEIGHT_WORDS;
    }
    TRANSFORMER_LAYER_FFN_TOPFED_W2_WEIGHT_PRELOAD_LOOP: for (uint32_t i = 0u; i < w2_weight_words; ++i) {
        topfed_ffn_w2_words[i] = sram[w2_weight_base + i];
    }
    u32_t topfed_ffn_w2_bias_words[FFN_W2_BIAS_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W2_BIAS_WORDS; ++i) {
        topfed_ffn_w2_bias_words[i] = 0;
    }
    uint32_t w2_bias_words = ffn_d_model;
    if (w2_bias_words > (uint32_t)FFN_W2_BIAS_WORDS) {
        w2_bias_words = (uint32_t)FFN_W2_BIAS_WORDS;
    }
    TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_PRELOAD_LOOP: for (uint32_t i = 0u; i < w2_bias_words; ++i) {
        topfed_ffn_w2_bias_words[i] = sram[w2_bias_base + i];
    }

    FFNLayer0<FFN_STAGE_W2>(
        sram,
        ffn_cfg,
        sc.attn_out_base_word,
        sc.ffn,
        pb.param_base_word,
        layer_id,
        0,
        0,
        (u32_t)0u,
        topfed_ffn_w2_input_words,
        (u32_t)w2_input_words,
        topfed_ffn_w2_words,
        (u32_t)w2_weight_words,
        topfed_ffn_w2_bias_words,
        (u32_t)w2_bias_words,
        (u32_t)FFN_POLICY_REQUIRE_W2_TOPFED
    );

    uint32_t residual_base = (uint32_t)sc.attn_out_base_word.to_uint();
    uint32_t w2_base = (uint32_t)sc.ffn.w2_out_base_word.to_uint();
    uint32_t add2_base = (uint32_t)sc.ffn.add2_base_word.to_uint();
    uint32_t words = (uint32_t)FFN_X_WORDS;
    TRANSFORMER_LAYER_FFN_RESIDUAL_ADD_LOOP: for (uint32_t i = 0; i < words; ++i) {
        fp32_t x = fp32_from_bits(sram[residual_base + i]);
        fp32_t y = fp32_from_bits(sram[w2_base + i]);
        sram[add2_base + i] = bits_from_fp32(x + y);
    }

    uint32_t gamma_base = (uint32_t)sc.ffn.ln_gamma_base_word.to_uint();
    uint32_t beta_base = (uint32_t)sc.ffn.ln_beta_base_word.to_uint();
    if (!sublayer1_norm_preloaded_by_top) {
        // Legacy fallback: keep in-block param fetch only when Top did not preload.
        load_layer_sublayer1_norm_params(
            sram,
            (uint32_t)pb.param_base_word.to_uint(),
            (uint32_t)layer_id.to_uint(),
            gamma_base,
            beta_base,
            d_model
        );
    }

    LayerNormCfg ln_cfg;
    ln_cfg.token_count = (u32_t)FFN_TOKEN_COUNT;
    ln_cfg.d_model = (u32_t)d_model;
    ln_cfg.eps_bits = LN_EPS_BITS;

    LayerNormBlock(
        sram,
        ln_cfg,
        (u32_t)add2_base,
        x_out_base_word,
        (u32_t)gamma_base,
        (u32_t)beta_base
    );
}

} // namespace aecct
