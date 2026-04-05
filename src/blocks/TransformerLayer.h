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

// local-only probe surface for W1/W2 seam observability.
// This does not alter functional outputs; it only tracks branch selection.
struct TransformerLayerW2SeamProbe {
    u32_t w1_input_mainline_taken_count;
    u32_t w1_input_fallback_preload_count;
    u32_t w1_weight_mainline_taken_count;
    u32_t w1_weight_fallback_preload_count;
    u32_t w1_bias_mainline_taken_count;
    u32_t w1_bias_fallback_preload_count;
    u32_t w2_weight_mainline_taken_count;
    u32_t w2_weight_fallback_preload_count;
    u32_t w2_bias_mainline_taken_count;
    u32_t w2_bias_fallback_preload_count;
};

static inline void clear_transformer_layer_w2_seam_probe(TransformerLayerW2SeamProbe& p) {
    p.w1_input_mainline_taken_count = (u32_t)0u;
    p.w1_input_fallback_preload_count = (u32_t)0u;
    p.w1_weight_mainline_taken_count = (u32_t)0u;
    p.w1_weight_fallback_preload_count = (u32_t)0u;
    p.w1_bias_mainline_taken_count = (u32_t)0u;
    p.w1_bias_fallback_preload_count = (u32_t)0u;
    p.w2_weight_mainline_taken_count = (u32_t)0u;
    p.w2_weight_fallback_preload_count = (u32_t)0u;
    p.w2_bias_mainline_taken_count = (u32_t)0u;
    p.w2_bias_fallback_preload_count = (u32_t)0u;
}

static inline void transformer_layer_probe_inc(u32_t* counter) {
    if (counter != 0) {
        *counter = (u32_t)((uint32_t)counter->to_uint() + 1u);
    }
}

struct CfgRegs {
    u32_t d_model;
    u32_t n_heads;
    u32_t d_ffn;
    u32_t n_layers;
};

enum TransformerAttnCompatShellStage {
    TRANSFORMER_ATTN_COMPAT_SHELL_DISABLED = 0,
    TRANSFORMER_ATTN_COMPAT_SHELL_FULL = 1,
    TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY = 2,
    TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY = 3,
    TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY = 4
};

static inline TransformerAttnCompatShellStage transformer_layer_select_attn_compat_shell_stage(
    bool attn_compat_shell_enable,
    bool kv_prebuilt_from_top_managed,
    bool q_prebuilt_from_top_managed,
    bool score_prebuilt_from_top_managed,
    bool out_prebuilt_from_top_managed,
    bool attn_out_topfed_payload_enable
) {
    if (!attn_compat_shell_enable) {
        return TRANSFORMER_ATTN_COMPAT_SHELL_DISABLED;
    }
    const bool attn_fully_prebuilt_from_top_managed =
        kv_prebuilt_from_top_managed &&
        q_prebuilt_from_top_managed &&
        score_prebuilt_from_top_managed &&
        out_prebuilt_from_top_managed;
    if (attn_fully_prebuilt_from_top_managed) {
        // Ownership seam: when all upstream stages are prebuilt, keep shell to OUT consume only.
        if (attn_out_topfed_payload_enable) {
            return TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY;
        }
        return TRANSFORMER_ATTN_COMPAT_SHELL_DISABLED;
    }
    const bool attn_score_ready_partial_out_stage_shell_safe =
        kv_prebuilt_from_top_managed &&
        q_prebuilt_from_top_managed &&
        score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        !attn_out_topfed_payload_enable;
    if (attn_score_ready_partial_out_stage_shell_safe) {
        // local-only bounded cut: score-ready partial-prebuild can shrink to OUT stage shell.
        return TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY;
    }
    const bool attn_score_ready_partial_out_stage_shell_payload_enabled_safe =
        kv_prebuilt_from_top_managed &&
        q_prebuilt_from_top_managed &&
        score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        attn_out_topfed_payload_enable;
    if (attn_score_ready_partial_out_stage_shell_payload_enabled_safe) {
        // Stage boundary: payload-enabled score-ready bucket reuses existing OUT consume/fallback seam.
        // Ownership seam: no new SRAM owner/arbitration semantics; selector only narrows to OUT stage.
        return TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY;
    }
    const bool attn_q_ready_kv_not_prebuilt_score_ready_partial_out_stage_shell_safe =
        !kv_prebuilt_from_top_managed &&
        q_prebuilt_from_top_managed &&
        score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        !attn_out_topfed_payload_enable;
    if (attn_q_ready_kv_not_prebuilt_score_ready_partial_out_stage_shell_safe) {
        // Stage boundary: score-ready bucket can consume OUT without requiring local KV materialization.
        // Ownership seam: this shell only consumes already committed post-concat payload for attn_out writeback.
        return TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY;
    }
    const bool attn_q_ready_kv_not_prebuilt_score_ready_partial_out_stage_shell_payload_enabled_safe =
        !kv_prebuilt_from_top_managed &&
        q_prebuilt_from_top_managed &&
        score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        attn_out_topfed_payload_enable;
    if (attn_q_ready_kv_not_prebuilt_score_ready_partial_out_stage_shell_payload_enabled_safe) {
        // Stage boundary: payload-enabled score-ready bucket reuses existing OUT consume/fallback seam.
        // Fallback boundary: non-selected buckets keep legacy FULL policy.
        return TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY;
    }
    const bool attn_kv_ready_q_not_prebuilt_score_ready_partial_out_stage_shell_safe =
        kv_prebuilt_from_top_managed &&
        !q_prebuilt_from_top_managed &&
        score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        !attn_out_topfed_payload_enable;
    if (attn_kv_ready_q_not_prebuilt_score_ready_partial_out_stage_shell_safe) {
        // Stage boundary: score-ready bucket can consume OUT without requiring local Q materialization.
        // Fallback boundary: this keeps direct SRAM fallback policy unchanged outside this exact bucket.
        return TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY;
    }
    const bool attn_kv_ready_q_not_prebuilt_score_ready_partial_out_stage_shell_payload_enabled_safe =
        kv_prebuilt_from_top_managed &&
        !q_prebuilt_from_top_managed &&
        score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        attn_out_topfed_payload_enable;
    if (attn_kv_ready_q_not_prebuilt_score_ready_partial_out_stage_shell_payload_enabled_safe) {
        // Stage boundary: payload-enabled score-ready bucket still reuses existing OUT consume/fallback seam.
        // Ownership seam: selector narrowing only; shared SRAM ownership/arbitration remains Top-managed.
        return TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY;
    }
    const bool attn_qkv_not_prebuilt_score_ready_partial_out_stage_shell_safe =
        !kv_prebuilt_from_top_managed &&
        !q_prebuilt_from_top_managed &&
        score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        !attn_out_topfed_payload_enable;
    if (attn_qkv_not_prebuilt_score_ready_partial_out_stage_shell_safe) {
        // Stage boundary: score-ready bucket can consume OUT when both Q and KV are already committed upstream.
        // Ownership seam: this path is stage-dispatch narrowing only and does not alter shared-SRAM ownership.
        return TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY;
    }
    const bool attn_qkv_not_prebuilt_score_ready_partial_out_stage_shell_payload_enabled_safe =
        !kv_prebuilt_from_top_managed &&
        !q_prebuilt_from_top_managed &&
        score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        attn_out_topfed_payload_enable;
    if (attn_qkv_not_prebuilt_score_ready_partial_out_stage_shell_payload_enabled_safe) {
        // Stage boundary: payload-enabled non-prebuilt QKV score-ready bucket remains OUT-stage compatible.
        // Ownership seam: selector narrowing only; shared SRAM ownership remains Top-managed.
        return TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY;
    }
    const bool attn_qkv_ready_partial_score_stage_shell_safe =
        kv_prebuilt_from_top_managed &&
        q_prebuilt_from_top_managed &&
        !score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        !attn_out_topfed_payload_enable;
    if (attn_qkv_ready_partial_score_stage_shell_safe) {
        // local-only bounded cut: q/kv-ready partial-prebuild can shrink to SCORES stage shell.
        return TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY;
    }
    const bool attn_qkv_ready_partial_score_stage_shell_payload_enabled_safe =
        kv_prebuilt_from_top_managed &&
        q_prebuilt_from_top_managed &&
        !score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        attn_out_topfed_payload_enable;
    if (attn_qkv_ready_partial_score_stage_shell_payload_enabled_safe) {
        // Stage boundary: payload-enabled q/kv-ready score-missing bucket reuses existing SCORES+writeback seam.
        // Ownership seam: no new shared SRAM arbitration semantics are introduced.
        return TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY;
    }
    const bool attn_q_ready_kv_not_prebuilt_qkv_scores_stage_shell_safe =
        !kv_prebuilt_from_top_managed &&
        q_prebuilt_from_top_managed &&
        !score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        !attn_out_topfed_payload_enable;
    if (attn_q_ready_kv_not_prebuilt_qkv_scores_stage_shell_safe) {
        // Stage boundary: q-ready/kv-missing bucket composes QKV + SCORES stages to avoid FULL shell.
        // Ownership seam: this branch keeps Top-owned SRAM policy unchanged and only narrows stage dispatch.
        return TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY;
    }
    const bool attn_kv_ready_q_not_prebuilt_qkv_scores_stage_shell_safe =
        kv_prebuilt_from_top_managed &&
        !q_prebuilt_from_top_managed &&
        !score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        !attn_out_topfed_payload_enable;
    if (attn_kv_ready_q_not_prebuilt_qkv_scores_stage_shell_safe) {
        // Stage boundary: kv-ready/q-missing bucket composes QKV + SCORES stages to avoid FULL shell.
        // Ownership seam: reuse existing stage surface only; no new shared-SRAM arbitration semantics.
        return TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY;
    }
    const bool attn_qkv_not_prebuilt_qkv_scores_stage_shell_safe =
        !kv_prebuilt_from_top_managed &&
        !q_prebuilt_from_top_managed &&
        !score_prebuilt_from_top_managed &&
        !out_prebuilt_from_top_managed &&
        !attn_out_topfed_payload_enable;
    if (attn_qkv_not_prebuilt_qkv_scores_stage_shell_safe) {
        // Stage boundary: non-prebuilt bucket composes QKV + SCORES stages to avoid FULL shell.
        // Fallback boundary: this keeps all non-selected buckets on legacy FULL behavior.
        return TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY;
    }
    // Fallback boundary: other partial-prebuild buckets stay on legacy full-shell behavior.
    return TRANSFORMER_ATTN_COMPAT_SHELL_FULL;
}

// local-only higher-level ownership seam for FFN W1/W2 payload handoff.
// Top can optionally provide preloaded FFN payload descriptors.
struct TransformerLayerFfnTopfedHandoffDesc {
    const u32_t* topfed_w1_x_words;
    u32_t topfed_w1_x_words_valid;
    const u32_t* topfed_w1_weight_words;
    u32_t topfed_w1_weight_words_valid;
    const u32_t* topfed_w1_bias_words;
    u32_t topfed_w1_bias_words_valid;
    const u32_t* topfed_w2_input_words;
    u32_t topfed_w2_input_words_valid;
    const u32_t* topfed_w2_weight_words;
    u32_t topfed_w2_weight_words_valid;
    const u32_t* topfed_w2_bias_words;
    u32_t topfed_w2_bias_words_valid;
};

static inline TransformerLayerFfnTopfedHandoffDesc make_transformer_layer_ffn_topfed_handoff_desc() {
    TransformerLayerFfnTopfedHandoffDesc desc;
    desc.topfed_w1_x_words = 0;
    desc.topfed_w1_x_words_valid = (u32_t)0u;
    desc.topfed_w1_weight_words = 0;
    desc.topfed_w1_weight_words_valid = (u32_t)0u;
    desc.topfed_w1_bias_words = 0;
    desc.topfed_w1_bias_words_valid = (u32_t)0u;
    desc.topfed_w2_input_words = 0;
    desc.topfed_w2_input_words_valid = (u32_t)0u;
    desc.topfed_w2_weight_words = 0;
    desc.topfed_w2_weight_words_valid = (u32_t)0u;
    desc.topfed_w2_bias_words = 0;
    desc.topfed_w2_bias_words_valid = (u32_t)0u;
    return desc;
}

static inline TransformerLayerFfnTopfedHandoffDesc make_transformer_layer_ffn_topfed_handoff_desc(
    const u32_t* topfed_w1_bias_words,
    u32_t topfed_w1_bias_words_valid
) {
    TransformerLayerFfnTopfedHandoffDesc desc = make_transformer_layer_ffn_topfed_handoff_desc();
    desc.topfed_w1_bias_words = topfed_w1_bias_words;
    desc.topfed_w1_bias_words_valid = topfed_w1_bias_words_valid;
    return desc;
}

static inline TransformerLayerFfnTopfedHandoffDesc make_transformer_layer_ffn_topfed_handoff_desc(
    const u32_t* topfed_w1_x_words,
    u32_t topfed_w1_x_words_valid,
    const u32_t* topfed_w1_weight_words,
    u32_t topfed_w1_weight_words_valid,
    const u32_t* topfed_w1_bias_words,
    u32_t topfed_w1_bias_words_valid
) {
    TransformerLayerFfnTopfedHandoffDesc desc = make_transformer_layer_ffn_topfed_handoff_desc();
    desc.topfed_w1_x_words = topfed_w1_x_words;
    desc.topfed_w1_x_words_valid = topfed_w1_x_words_valid;
    desc.topfed_w1_weight_words = topfed_w1_weight_words;
    desc.topfed_w1_weight_words_valid = topfed_w1_weight_words_valid;
    desc.topfed_w1_bias_words = topfed_w1_bias_words;
    desc.topfed_w1_bias_words_valid = topfed_w1_bias_words_valid;
    return desc;
}

static inline TransformerLayerFfnTopfedHandoffDesc make_transformer_layer_ffn_topfed_handoff_desc(
    const u32_t* topfed_w1_x_words,
    u32_t topfed_w1_x_words_valid,
    const u32_t* topfed_w1_weight_words,
    u32_t topfed_w1_weight_words_valid,
    const u32_t* topfed_w1_bias_words,
    u32_t topfed_w1_bias_words_valid,
    const u32_t* topfed_w2_input_words,
    u32_t topfed_w2_input_words_valid,
    const u32_t* topfed_w2_weight_words,
    u32_t topfed_w2_weight_words_valid,
    const u32_t* topfed_w2_bias_words,
    u32_t topfed_w2_bias_words_valid
) {
    TransformerLayerFfnTopfedHandoffDesc desc = make_transformer_layer_ffn_topfed_handoff_desc(
        topfed_w1_x_words,
        topfed_w1_x_words_valid,
        topfed_w1_weight_words,
        topfed_w1_weight_words_valid,
        topfed_w1_bias_words,
        topfed_w1_bias_words_valid
    );
    desc.topfed_w2_input_words = topfed_w2_input_words;
    desc.topfed_w2_input_words_valid = topfed_w2_input_words_valid;
    desc.topfed_w2_weight_words = topfed_w2_weight_words;
    desc.topfed_w2_weight_words_valid = topfed_w2_weight_words_valid;
    desc.topfed_w2_bias_words = topfed_w2_bias_words;
    desc.topfed_w2_bias_words_valid = topfed_w2_bias_words_valid;
    return desc;
}

static inline void transformer_layer_select_topfed_words(
    const u32_t* handoff_words,
    uint32_t handoff_words_valid,
    uint32_t handoff_words_max,
    const u32_t* local_words,
    uint32_t local_words_valid,
    const u32_t*& selected_words,
    u32_t& selected_words_valid
) {
    // Selection seam: prefer Top-fed payload when descriptor is valid, else keep local materialization.
    selected_words = local_words;
    selected_words_valid = (u32_t)local_words_valid;
    if (handoff_words == 0) {
        return;
    }
    uint32_t from_top_valid = handoff_words_valid;
    if (from_top_valid == 0u) {
        return;
    }
    if (from_top_valid > handoff_words_max) {
        from_top_valid = handoff_words_max;
    }
    selected_words = handoff_words;
    selected_words_valid = (u32_t)from_top_valid;
}

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
// Reading rule: this bridge is not a new algorithm. It is a caller-side ownership
// seam that stages prebuilt/top-fed descriptors before entering the accepted
// attention + FFN compute blocks.
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
    bool sublayer1_norm_preloaded_by_top = false,
    TransformerLayerFfnTopfedHandoffDesc ffn_topfed_handoff_desc =
        make_transformer_layer_ffn_topfed_handoff_desc(),
    bool attn_out_topfed_payload_enable = false,
    const u32_t* attn_out_topfed_payload_words = 0,
    u32_t attn_out_topfed_payload_words_valid = (u32_t)0u,
    bool attn_compat_shell_enable = true,
    TransformerLayerW2SeamProbe* w2_seam_probe = 0
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
    // Top passes compat-shell policy into this block; this block consumes it only.
    const AttnLayer0PrebuiltHandoffDesc attn_prebuilt_handoff =
        make_attn_layer0_prebuilt_handoff_desc(
            kv_prebuilt_from_top_managed,
            q_prebuilt_from_top_managed,
            score_prebuilt_from_top_managed,
            out_prebuilt_from_top_managed,
            attn_out_topfed_payload_enable,
            attn_out_topfed_payload_words,
            attn_out_topfed_payload_words_valid);
    const TransformerAttnCompatShellStage attn_shell_stage =
        transformer_layer_select_attn_compat_shell_stage(
            attn_compat_shell_enable,
            kv_prebuilt_from_top_managed,
            q_prebuilt_from_top_managed,
            score_prebuilt_from_top_managed,
            out_prebuilt_from_top_managed,
            attn_out_topfed_payload_enable);
    if (attn_shell_stage == TRANSFORMER_ATTN_COMPAT_SHELL_FULL) {
        // Stage boundary: non-selected partial-prebuild buckets keep full-shell execution.
        AttnLayer0TopManagedWindowBridge<ATTN_STAGE_FULL>(
            sram_window,
            attn_cfg,
            x_in_base_word,
            sc.attn_out_base_word,
            sc.attn,
            (u32_t)0,
            attn_prebuilt_handoff
        );
    } else if (attn_shell_stage == TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
        // Stage boundary: OUT-stage shell for fully-prebuilt payload consume and selected score-ready partial bucket.
        AttnLayer0TopManagedWindowBridge<ATTN_STAGE_OUT>(
            sram_window,
            attn_cfg,
            x_in_base_word,
            sc.attn_out_base_word,
            sc.attn,
            (u32_t)0,
            attn_prebuilt_handoff
        );
    } else if (attn_shell_stage == TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY) {
        // Stage boundary: SCORES-stage shell for q/kv-ready, score-not-prebuilt bucket.
        AttnLayer0TopManagedWindowBridge<ATTN_STAGE_SCORES>(
            sram_window,
            attn_cfg,
            x_in_base_word,
            sc.attn_out_base_word,
            sc.attn,
            (u32_t)0,
            attn_prebuilt_handoff
        );
        // Ownership seam: SCORES-only stage still commits post->attn_out writeback for downstream FFN input.
        const uint32_t token_count = (uint32_t)attn_cfg.token_count.to_uint();
        const uint32_t attn_d_model = (uint32_t)attn_cfg.d_model.to_uint();
        uint32_t attn_tensor_words = token_count * attn_d_model;
        if (attn_tensor_words == 0u) {
            attn_tensor_words = (uint32_t)ATTN_TENSOR_WORDS;
        }
        const uint32_t post_base = (uint32_t)sc.attn.post_concat_base_word.to_uint();
        const uint32_t out_base = (uint32_t)sc.attn_out_base_word.to_uint();
        TRANSFORMER_ATTN_SCORES_ONLY_OUT_WRITEBACK_BRIDGE_LOOP: for (uint32_t i = 0u; i < attn_tensor_words; ++i) {
            sram_window[out_base + i] = sram_window[post_base + i];
        }
    } else if (attn_shell_stage == TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY) {
        // Stage boundary: q-ready/kv-not-prebuilt bucket composes QKV then SCORES stages.
        AttnLayer0TopManagedWindowBridge<ATTN_STAGE_QKV>(
            sram_window,
            attn_cfg,
            x_in_base_word,
            sc.attn_out_base_word,
            sc.attn,
            (u32_t)0,
            attn_prebuilt_handoff
        );
        AttnLayer0TopManagedWindowBridge<ATTN_STAGE_SCORES>(
            sram_window,
            attn_cfg,
            x_in_base_word,
            sc.attn_out_base_word,
            sc.attn,
            (u32_t)0,
            attn_prebuilt_handoff
        );
        // Ownership seam: composed stages still commit explicit post->attn_out writeback for downstream FFN input.
        const uint32_t token_count = (uint32_t)attn_cfg.token_count.to_uint();
        const uint32_t attn_d_model = (uint32_t)attn_cfg.d_model.to_uint();
        uint32_t attn_tensor_words = token_count * attn_d_model;
        if (attn_tensor_words == 0u) {
            attn_tensor_words = (uint32_t)ATTN_TENSOR_WORDS;
        }
        const uint32_t post_base = (uint32_t)sc.attn.post_concat_base_word.to_uint();
        const uint32_t out_base = (uint32_t)sc.attn_out_base_word.to_uint();
        TRANSFORMER_ATTN_QKV_SCORES_ONLY_OUT_WRITEBACK_BRIDGE_LOOP: for (uint32_t i = 0u; i < attn_tensor_words; ++i) {
            sram_window[out_base + i] = sram_window[post_base + i];
        }
    }

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
    const uint32_t topfed_w1_x_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w1_x_words_valid.to_uint();
    // Top decides descriptor readiness here; this block only consumes it.
    const bool w1_input_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w1_x_words != 0) &&
        (topfed_w1_x_words_valid_raw >= ffn_x_words);
    const uint32_t ffn_x_base = (uint32_t)sc.attn_out_base_word.to_uint();
    if (!w1_input_topfed_ready) {
        // Probe this branch to prove W1 input compatibility preload fallback.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_input_fallback_preload_count);
        }
        // This fallback keeps the legacy preload path alive when descriptor data is absent.
        TRANSFORMER_LAYER_FFN_TOPFED_X_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < ffn_x_words; ++i) {
            topfed_ffn_x_words[i] = sram_window[ffn_x_base + i];
        }
    }
    const u32_t* selected_topfed_ffn_x_words = 0;
    u32_t selected_topfed_ffn_x_words_valid = (u32_t)0u;
    if (w1_input_topfed_ready) {
        // Probe this branch to prove W1 input mainline descriptor consumption.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_input_mainline_taken_count);
        }
        // This mainline consumes W1 data directly from the top-fed descriptor.
        selected_topfed_ffn_x_words = ffn_topfed_handoff_desc.topfed_w1_x_words;
        uint32_t selected_valid = topfed_w1_x_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_X_WORDS) {
            selected_valid = (uint32_t)FFN_X_WORDS;
        }
        selected_topfed_ffn_x_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_x_words = topfed_ffn_x_words;
        selected_topfed_ffn_x_words_valid = (u32_t)ffn_x_words;
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
    const uint32_t topfed_w1_weight_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w1_weight_words_valid.to_uint();
    // Top decides descriptor readiness here; this block only consumes it.
    const bool w1_weight_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w1_weight_words != 0) &&
        (topfed_w1_weight_words_valid_raw >= w1_weight_words);
    if (!w1_weight_topfed_ready) {
        // Probe this branch to prove W1 weight compatibility preload fallback.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_weight_fallback_preload_count);
        }
        // This fallback keeps the legacy preload path alive when descriptor data is absent.
        TRANSFORMER_LAYER_FFN_TOPFED_W1_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < w1_weight_words; ++i) {
            topfed_ffn_w1_words[i] = sram_window[w1_weight_base + i];
        }
    }
    const u32_t* selected_topfed_ffn_w1_words = 0;
    u32_t selected_topfed_ffn_w1_words_valid = (u32_t)0u;
    if (w1_weight_topfed_ready) {
        // Probe this branch to prove W1 weight mainline descriptor consumption.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_weight_mainline_taken_count);
        }
        // This mainline consumes W1 weight directly from the top-fed descriptor.
        selected_topfed_ffn_w1_words = ffn_topfed_handoff_desc.topfed_w1_weight_words;
        uint32_t selected_valid = topfed_w1_weight_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_W1_WEIGHT_WORDS) {
            selected_valid = (uint32_t)FFN_W1_WEIGHT_WORDS;
        }
        selected_topfed_ffn_w1_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_w1_words = topfed_ffn_w1_words;
        selected_topfed_ffn_w1_words_valid = (u32_t)w1_weight_words;
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
    const uint32_t topfed_w1_bias_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w1_bias_words_valid.to_uint();
    // Top decides descriptor readiness here; this block only consumes it.
    const bool w1_bias_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w1_bias_words != 0) &&
        (topfed_w1_bias_words_valid_raw >= w1_bias_words);
    if (!w1_bias_topfed_ready) {
        // Probe this branch to prove W1 bias compatibility preload fallback.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_bias_fallback_preload_count);
        }
        // This local materialization remains a compatibility path, not a new ownership model.
        TRANSFORMER_LAYER_FFN_TOPFED_W1_BIAS_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < w1_bias_words; ++i) {
            topfed_ffn_w1_bias_words[i] = sram_window[w1_bias_base + i];
        }
    }
    const u32_t* selected_topfed_ffn_w1_bias_words = 0;
    u32_t selected_topfed_ffn_w1_bias_words_valid = (u32_t)0u;
    if (w1_bias_topfed_ready) {
        // Probe this branch to prove W1 bias mainline descriptor consumption.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_bias_mainline_taken_count);
        }
        // This mainline consumes W1 bias directly from the top-fed descriptor.
        selected_topfed_ffn_w1_bias_words = ffn_topfed_handoff_desc.topfed_w1_bias_words;
        uint32_t selected_valid = topfed_w1_bias_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_W1_BIAS_WORDS) {
            selected_valid = (uint32_t)FFN_W1_BIAS_WORDS;
        }
        selected_topfed_ffn_w1_bias_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_w1_bias_words = topfed_ffn_w1_bias_words;
        selected_topfed_ffn_w1_bias_words_valid = (u32_t)w1_bias_words;
    }

    // Stage-split FFN dispatch keeps caller ownership explicit for payload descriptors.
    // FFN stage transition: W1 consumes selected input/weight/bias payload descriptors.
    FFNLayer0TopManagedWindowBridge<FFN_STAGE_W1>(
        sram_window,
        ffn_cfg,
        sc.attn_out_base_word,
        sc.ffn,
        pb.param_base_word,
        layer_id,
        selected_topfed_ffn_x_words,
        selected_topfed_ffn_w1_words,
        selected_topfed_ffn_w1_words_valid,
        0,
        (u32_t)0u,
        0,
        (u32_t)0u,
        0,
        (u32_t)0u,
        (u32_t)FFN_POLICY_REQUIRE_W1_TOPFED,
        0,
        0,
        selected_topfed_ffn_x_words_valid,
        selected_topfed_ffn_w1_bias_words,
        selected_topfed_ffn_w1_bias_words_valid
    );
    // FFN stage transition: ReLU consumes W1 output scratch.
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
    const uint32_t topfed_w2_input_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w2_input_words_valid.to_uint();
    // Top decides descriptor span at dispatch; this block only consumes that policy.
    const bool w2_input_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w2_input_words != 0) &&
        (topfed_w2_input_words_valid_raw >= w2_input_words);
    if (!w2_input_topfed_ready) {
        // Compatibility fallback: materialize from ReLU scratch only when Top-fed payload is absent.
        TRANSFORMER_LAYER_FFN_TOPFED_W2_INPUT_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < w2_input_words; ++i) {
            topfed_ffn_w2_input_words[i] = sram_window[relu_base + i];
        }
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
    const uint32_t topfed_w2_weight_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w2_weight_words_valid.to_uint();
    // Top decides descriptor readiness; this block only consumes it.
    const bool w2_weight_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w2_weight_words != 0) &&
        (topfed_w2_weight_words_valid_raw >= w2_weight_words);
    if (!w2_weight_topfed_ready) {
        // Probe this branch to prove compatibility preload fallback for W2 weight.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w2_weight_fallback_preload_count);
        }
        // This fallback keeps the legacy preload path alive when descriptor data is absent.
        TRANSFORMER_LAYER_FFN_TOPFED_W2_WEIGHT_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < w2_weight_words; ++i) {
            topfed_ffn_w2_words[i] = sram_window[w2_weight_base + i];
        }
    }
    u32_t topfed_ffn_w2_bias_words[FFN_W2_BIAS_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_INIT_BRIDGE_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W2_BIAS_WORDS; ++i) {
        topfed_ffn_w2_bias_words[i] = 0;
    }
    uint32_t w2_bias_words = ffn_d_model;
    if (w2_bias_words > (uint32_t)FFN_W2_BIAS_WORDS) {
        w2_bias_words = (uint32_t)FFN_W2_BIAS_WORDS;
    }
    const uint32_t topfed_w2_bias_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w2_bias_words_valid.to_uint();
    // Top decides descriptor readiness; this block only consumes it.
    const bool w2_bias_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w2_bias_words != 0) &&
        (topfed_w2_bias_words_valid_raw >= w2_bias_words);
    if (!w2_bias_topfed_ready) {
        // Probe this branch to prove compatibility preload fallback for W2 bias.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w2_bias_fallback_preload_count);
        }
        // This local materialization remains a compatibility path, not a new ownership model.
        TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_PRELOAD_BRIDGE_LOOP: for (uint32_t i = 0u; i < w2_bias_words; ++i) {
            topfed_ffn_w2_bias_words[i] = sram_window[w2_bias_base + i];
        }
    }
    const u32_t* selected_topfed_ffn_w2_input_words = 0;
    u32_t selected_topfed_ffn_w2_input_words_valid = (u32_t)0u;
    if (w2_input_topfed_ready) {
        // Mainline consumes prebuilt W2 input and skips local materialization.
        selected_topfed_ffn_w2_input_words = ffn_topfed_handoff_desc.topfed_w2_input_words;
        uint32_t selected_valid = topfed_w2_input_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_W2_INPUT_WORDS) {
            selected_valid = (uint32_t)FFN_W2_INPUT_WORDS;
        }
        selected_topfed_ffn_w2_input_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_w2_input_words = topfed_ffn_w2_input_words;
        selected_topfed_ffn_w2_input_words_valid = (u32_t)w2_input_words;
    }
    const u32_t* selected_topfed_ffn_w2_words = 0;
    u32_t selected_topfed_ffn_w2_words_valid = (u32_t)0u;
    if (w2_weight_topfed_ready) {
        // Probe this branch to prove W2 weight mainline descriptor consumption.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w2_weight_mainline_taken_count);
        }
        // This mainline consumes W2 weight from the top-fed descriptor.
        selected_topfed_ffn_w2_words = ffn_topfed_handoff_desc.topfed_w2_weight_words;
        uint32_t selected_valid = topfed_w2_weight_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_W2_WEIGHT_WORDS) {
            selected_valid = (uint32_t)FFN_W2_WEIGHT_WORDS;
        }
        selected_topfed_ffn_w2_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_w2_words = topfed_ffn_w2_words;
        selected_topfed_ffn_w2_words_valid = (u32_t)w2_weight_words;
    }
    const u32_t* selected_topfed_ffn_w2_bias_words = 0;
    u32_t selected_topfed_ffn_w2_bias_words_valid = (u32_t)0u;
    if (w2_bias_topfed_ready) {
        // Probe this branch to prove W2 bias mainline descriptor consumption.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w2_bias_mainline_taken_count);
        }
        // This mainline consumes W2 bias from the top-fed descriptor.
        selected_topfed_ffn_w2_bias_words = ffn_topfed_handoff_desc.topfed_w2_bias_words;
        uint32_t selected_valid = topfed_w2_bias_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_W2_BIAS_WORDS) {
            selected_valid = (uint32_t)FFN_W2_BIAS_WORDS;
        }
        selected_topfed_ffn_w2_bias_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_w2_bias_words = topfed_ffn_w2_bias_words;
        selected_topfed_ffn_w2_bias_words_valid = (u32_t)w2_bias_words;
    }

    // FFN stage transition: W2 consumes ReLU output and writes FFN output scratch.
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
        selected_topfed_ffn_w2_input_words,
        selected_topfed_ffn_w2_input_words_valid,
        selected_topfed_ffn_w2_words,
        selected_topfed_ffn_w2_words_valid,
        selected_topfed_ffn_w2_bias_words,
        selected_topfed_ffn_w2_bias_words_valid,
        (u32_t)FFN_POLICY_REQUIRE_W2_TOPFED
    );

    uint32_t residual_base = (uint32_t)sc.attn_out_base_word.to_uint();
    uint32_t w2_base = (uint32_t)sc.ffn.w2_out_base_word.to_uint();
    uint32_t add2_base = (uint32_t)sc.ffn.add2_base_word.to_uint();
    uint32_t words = (uint32_t)FFN_X_WORDS;
    // Write-back boundary: residual add commits to add2 scratch before LayerNorm.
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
// Read this function in four chunks:
// 1) optional attention shell call
// 2) FFN top-fed descriptor selection / preload fallback
// 3) residual add write-back into x_out_base_word
// 4) sublayer-1 LayerNorm handoff
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
    bool sublayer1_norm_preloaded_by_top = false,
    TransformerLayerFfnTopfedHandoffDesc ffn_topfed_handoff_desc =
        make_transformer_layer_ffn_topfed_handoff_desc(),
    bool attn_out_topfed_payload_enable = false,
    const u32_t* attn_out_topfed_payload_words = 0,
    u32_t attn_out_topfed_payload_words_valid = (u32_t)0u,
    bool attn_compat_shell_enable = true,
    TransformerLayerW2SeamProbe* w2_seam_probe = 0
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
    // Top-owned compat-shell policy is consumed here in the pointer entry as well.
    const AttnLayer0PrebuiltHandoffDesc attn_prebuilt_handoff =
        make_attn_layer0_prebuilt_handoff_desc(
            kv_prebuilt_from_top_managed,
            q_prebuilt_from_top_managed,
            score_prebuilt_from_top_managed,
            out_prebuilt_from_top_managed,
            attn_out_topfed_payload_enable,
            attn_out_topfed_payload_words,
            attn_out_topfed_payload_words_valid);
    const TransformerAttnCompatShellStage attn_shell_stage =
        transformer_layer_select_attn_compat_shell_stage(
            attn_compat_shell_enable,
            kv_prebuilt_from_top_managed,
            q_prebuilt_from_top_managed,
            score_prebuilt_from_top_managed,
            out_prebuilt_from_top_managed,
            attn_out_topfed_payload_enable);
    if (attn_shell_stage == TRANSFORMER_ATTN_COMPAT_SHELL_FULL) {
        // Stage boundary: non-selected partial-prebuild buckets keep full-shell execution.
        AttnLayer0<ATTN_STAGE_FULL>(
            sram,
            attn_cfg,
            x_in_base_word,
            sc.attn_out_base_word,
            sc.attn,
            (u32_t)0,
            attn_prebuilt_handoff
        );
    } else if (attn_shell_stage == TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
        // Stage boundary: OUT-stage shell for fully-prebuilt payload consume and selected score-ready partial bucket.
        AttnLayer0<ATTN_STAGE_OUT>(
            sram,
            attn_cfg,
            x_in_base_word,
            sc.attn_out_base_word,
            sc.attn,
            (u32_t)0,
            attn_prebuilt_handoff
        );
    } else if (attn_shell_stage == TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY) {
        // Stage boundary: SCORES-stage shell for q/kv-ready, score-not-prebuilt bucket.
        AttnLayer0<ATTN_STAGE_SCORES>(
            sram,
            attn_cfg,
            x_in_base_word,
            sc.attn_out_base_word,
            sc.attn,
            (u32_t)0,
            attn_prebuilt_handoff
        );
        // Ownership seam: SCORES-only stage still commits post->attn_out writeback for downstream FFN input.
        const uint32_t token_count = (uint32_t)attn_cfg.token_count.to_uint();
        const uint32_t attn_d_model = (uint32_t)attn_cfg.d_model.to_uint();
        uint32_t attn_tensor_words = token_count * attn_d_model;
        if (attn_tensor_words == 0u) {
            attn_tensor_words = (uint32_t)ATTN_TENSOR_WORDS;
        }
        const uint32_t post_base = (uint32_t)sc.attn.post_concat_base_word.to_uint();
        const uint32_t out_base = (uint32_t)sc.attn_out_base_word.to_uint();
        TRANSFORMER_ATTN_SCORES_ONLY_OUT_WRITEBACK_LOOP: for (uint32_t i = 0u; i < attn_tensor_words; ++i) {
            sram[out_base + i] = sram[post_base + i];
        }
    } else if (attn_shell_stage == TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY) {
        // Stage boundary: q-ready/kv-not-prebuilt bucket composes QKV then SCORES stages.
        AttnLayer0<ATTN_STAGE_QKV>(
            sram,
            attn_cfg,
            x_in_base_word,
            sc.attn_out_base_word,
            sc.attn,
            (u32_t)0,
            attn_prebuilt_handoff
        );
        AttnLayer0<ATTN_STAGE_SCORES>(
            sram,
            attn_cfg,
            x_in_base_word,
            sc.attn_out_base_word,
            sc.attn,
            (u32_t)0,
            attn_prebuilt_handoff
        );
        // Ownership seam: composed stages still commit explicit post->attn_out writeback for downstream FFN input.
        const uint32_t token_count = (uint32_t)attn_cfg.token_count.to_uint();
        const uint32_t attn_d_model = (uint32_t)attn_cfg.d_model.to_uint();
        uint32_t attn_tensor_words = token_count * attn_d_model;
        if (attn_tensor_words == 0u) {
            attn_tensor_words = (uint32_t)ATTN_TENSOR_WORDS;
        }
        const uint32_t post_base = (uint32_t)sc.attn.post_concat_base_word.to_uint();
        const uint32_t out_base = (uint32_t)sc.attn_out_base_word.to_uint();
        TRANSFORMER_ATTN_QKV_SCORES_ONLY_OUT_WRITEBACK_LOOP: for (uint32_t i = 0u; i < attn_tensor_words; ++i) {
            sram[out_base + i] = sram[post_base + i];
        }
    }

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
    const uint32_t topfed_w1_x_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w1_x_words_valid.to_uint();
    // Top decides descriptor readiness here; this block only consumes it.
    const bool w1_input_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w1_x_words != 0) &&
        (topfed_w1_x_words_valid_raw >= ffn_x_words);
    const uint32_t ffn_x_base = (uint32_t)sc.attn_out_base_word.to_uint();
    if (!w1_input_topfed_ready) {
        // Probe this branch to prove W1 input compatibility preload fallback.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_input_fallback_preload_count);
        }
        // This fallback keeps the legacy preload path alive when descriptor data is absent.
        TRANSFORMER_LAYER_FFN_TOPFED_X_PRELOAD_LOOP: for (uint32_t i = 0u; i < ffn_x_words; ++i) {
            topfed_ffn_x_words[i] = sram[ffn_x_base + i];
        }
    }
    const u32_t* selected_topfed_ffn_x_words = 0;
    u32_t selected_topfed_ffn_x_words_valid = (u32_t)0u;
    if (w1_input_topfed_ready) {
        // Probe this branch to prove W1 input mainline descriptor consumption.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_input_mainline_taken_count);
        }
        // This mainline consumes W1 data directly from the top-fed descriptor.
        selected_topfed_ffn_x_words = ffn_topfed_handoff_desc.topfed_w1_x_words;
        uint32_t selected_valid = topfed_w1_x_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_X_WORDS) {
            selected_valid = (uint32_t)FFN_X_WORDS;
        }
        selected_topfed_ffn_x_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_x_words = topfed_ffn_x_words;
        selected_topfed_ffn_x_words_valid = (u32_t)ffn_x_words;
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
    const uint32_t topfed_w1_weight_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w1_weight_words_valid.to_uint();
    // Top decides descriptor readiness here; this block only consumes it.
    const bool w1_weight_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w1_weight_words != 0) &&
        (topfed_w1_weight_words_valid_raw >= w1_weight_words);
    if (!w1_weight_topfed_ready) {
        // Probe this branch to prove W1 weight compatibility preload fallback.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_weight_fallback_preload_count);
        }
        // This fallback keeps the legacy preload path alive when descriptor data is absent.
        TRANSFORMER_LAYER_FFN_TOPFED_W1_PRELOAD_LOOP: for (uint32_t i = 0u; i < w1_weight_words; ++i) {
            topfed_ffn_w1_words[i] = sram[w1_weight_base + i];
        }
    }
    const u32_t* selected_topfed_ffn_w1_words = 0;
    u32_t selected_topfed_ffn_w1_words_valid = (u32_t)0u;
    if (w1_weight_topfed_ready) {
        // Probe this branch to prove W1 weight mainline descriptor consumption.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_weight_mainline_taken_count);
        }
        // This mainline consumes W1 weight directly from the top-fed descriptor.
        selected_topfed_ffn_w1_words = ffn_topfed_handoff_desc.topfed_w1_weight_words;
        uint32_t selected_valid = topfed_w1_weight_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_W1_WEIGHT_WORDS) {
            selected_valid = (uint32_t)FFN_W1_WEIGHT_WORDS;
        }
        selected_topfed_ffn_w1_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_w1_words = topfed_ffn_w1_words;
        selected_topfed_ffn_w1_words_valid = (u32_t)w1_weight_words;
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
    const uint32_t topfed_w1_bias_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w1_bias_words_valid.to_uint();
    // Top decides descriptor readiness here; this block only consumes it.
    const bool w1_bias_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w1_bias_words != 0) &&
        (topfed_w1_bias_words_valid_raw >= w1_bias_words);
    if (!w1_bias_topfed_ready) {
        // Probe this branch to prove W1 bias compatibility preload fallback.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_bias_fallback_preload_count);
        }
        // This local materialization remains a compatibility path, not a new ownership model.
        TRANSFORMER_LAYER_FFN_TOPFED_W1_BIAS_PRELOAD_LOOP: for (uint32_t i = 0u; i < w1_bias_words; ++i) {
            topfed_ffn_w1_bias_words[i] = sram[w1_bias_base + i];
        }
    }
    const u32_t* selected_topfed_ffn_w1_bias_words = 0;
    u32_t selected_topfed_ffn_w1_bias_words_valid = (u32_t)0u;
    if (w1_bias_topfed_ready) {
        // Probe this branch to prove W1 bias mainline descriptor consumption.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w1_bias_mainline_taken_count);
        }
        // This mainline consumes W1 bias directly from the top-fed descriptor.
        selected_topfed_ffn_w1_bias_words = ffn_topfed_handoff_desc.topfed_w1_bias_words;
        uint32_t selected_valid = topfed_w1_bias_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_W1_BIAS_WORDS) {
            selected_valid = (uint32_t)FFN_W1_BIAS_WORDS;
        }
        selected_topfed_ffn_w1_bias_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_w1_bias_words = topfed_ffn_w1_bias_words;
        selected_topfed_ffn_w1_bias_words_valid = (u32_t)w1_bias_words;
    }

    // Stage-split FFN dispatch keeps caller ownership explicit for payload descriptors.
    // FFN stage transition: W1.
    FFNLayer0<FFN_STAGE_W1>(
        sram,
        ffn_cfg,
        sc.attn_out_base_word,
        sc.ffn,
        pb.param_base_word,
        layer_id,
        selected_topfed_ffn_x_words,
        selected_topfed_ffn_w1_words,
        selected_topfed_ffn_w1_words_valid,
        0,
        (u32_t)0u,
        0,
        (u32_t)0u,
        0,
        (u32_t)0u,
        (u32_t)FFN_POLICY_REQUIRE_W1_TOPFED,
        0,
        0,
        selected_topfed_ffn_x_words_valid,
        selected_topfed_ffn_w1_bias_words,
        selected_topfed_ffn_w1_bias_words_valid
    );
    // FFN stage transition: ReLU.
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
    const uint32_t topfed_w2_input_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w2_input_words_valid.to_uint();
    // Top decides descriptor span at dispatch; this block only consumes that policy.
    const bool w2_input_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w2_input_words != 0) &&
        (topfed_w2_input_words_valid_raw >= w2_input_words);
    if (!w2_input_topfed_ready) {
        // This fallback keeps the legacy path alive when top-fed payload is absent.
        TRANSFORMER_LAYER_FFN_TOPFED_W2_INPUT_PRELOAD_LOOP: for (uint32_t i = 0u; i < w2_input_words; ++i) {
            topfed_ffn_w2_input_words[i] = sram[relu_base + i];
        }
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
    const uint32_t topfed_w2_weight_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w2_weight_words_valid.to_uint();
    // Top decides descriptor readiness; this block only consumes it.
    const bool w2_weight_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w2_weight_words != 0) &&
        (topfed_w2_weight_words_valid_raw >= w2_weight_words);
    if (!w2_weight_topfed_ready) {
        // Probe this branch to prove compatibility preload fallback for W2 weight.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w2_weight_fallback_preload_count);
        }
        // This fallback keeps the legacy preload path alive when descriptor data is absent.
        TRANSFORMER_LAYER_FFN_TOPFED_W2_WEIGHT_PRELOAD_LOOP: for (uint32_t i = 0u; i < w2_weight_words; ++i) {
            topfed_ffn_w2_words[i] = sram[w2_weight_base + i];
        }
    }
    u32_t topfed_ffn_w2_bias_words[FFN_W2_BIAS_WORDS];
    TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)FFN_W2_BIAS_WORDS; ++i) {
        topfed_ffn_w2_bias_words[i] = 0;
    }
    uint32_t w2_bias_words = ffn_d_model;
    if (w2_bias_words > (uint32_t)FFN_W2_BIAS_WORDS) {
        w2_bias_words = (uint32_t)FFN_W2_BIAS_WORDS;
    }
    const uint32_t topfed_w2_bias_words_valid_raw =
        (uint32_t)ffn_topfed_handoff_desc.topfed_w2_bias_words_valid.to_uint();
    // Top decides descriptor readiness; this block only consumes it.
    const bool w2_bias_topfed_ready =
        (ffn_topfed_handoff_desc.topfed_w2_bias_words != 0) &&
        (topfed_w2_bias_words_valid_raw >= w2_bias_words);
    if (!w2_bias_topfed_ready) {
        // Probe this branch to prove compatibility preload fallback for W2 bias.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w2_bias_fallback_preload_count);
        }
        // This local materialization remains a compatibility path, not a new ownership model.
        TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_PRELOAD_LOOP: for (uint32_t i = 0u; i < w2_bias_words; ++i) {
            topfed_ffn_w2_bias_words[i] = sram[w2_bias_base + i];
        }
    }
    const u32_t* selected_topfed_ffn_w2_input_words = 0;
    u32_t selected_topfed_ffn_w2_input_words_valid = (u32_t)0u;
    if (w2_input_topfed_ready) {
        // Mainline consumes prebuilt W2 input and skips local materialization.
        selected_topfed_ffn_w2_input_words = ffn_topfed_handoff_desc.topfed_w2_input_words;
        uint32_t selected_valid = topfed_w2_input_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_W2_INPUT_WORDS) {
            selected_valid = (uint32_t)FFN_W2_INPUT_WORDS;
        }
        selected_topfed_ffn_w2_input_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_w2_input_words = topfed_ffn_w2_input_words;
        selected_topfed_ffn_w2_input_words_valid = (u32_t)w2_input_words;
    }
    const u32_t* selected_topfed_ffn_w2_words = 0;
    u32_t selected_topfed_ffn_w2_words_valid = (u32_t)0u;
    if (w2_weight_topfed_ready) {
        // Probe this branch to prove W2 weight mainline descriptor consumption.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w2_weight_mainline_taken_count);
        }
        // This mainline consumes W2 weight from the top-fed descriptor.
        selected_topfed_ffn_w2_words = ffn_topfed_handoff_desc.topfed_w2_weight_words;
        uint32_t selected_valid = topfed_w2_weight_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_W2_WEIGHT_WORDS) {
            selected_valid = (uint32_t)FFN_W2_WEIGHT_WORDS;
        }
        selected_topfed_ffn_w2_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_w2_words = topfed_ffn_w2_words;
        selected_topfed_ffn_w2_words_valid = (u32_t)w2_weight_words;
    }
    const u32_t* selected_topfed_ffn_w2_bias_words = 0;
    u32_t selected_topfed_ffn_w2_bias_words_valid = (u32_t)0u;
    if (w2_bias_topfed_ready) {
        // Probe this branch to prove W2 bias mainline descriptor consumption.
        if (w2_seam_probe != 0) {
            transformer_layer_probe_inc(&w2_seam_probe->w2_bias_mainline_taken_count);
        }
        // This mainline consumes W2 bias from the top-fed descriptor.
        selected_topfed_ffn_w2_bias_words = ffn_topfed_handoff_desc.topfed_w2_bias_words;
        uint32_t selected_valid = topfed_w2_bias_words_valid_raw;
        if (selected_valid > (uint32_t)FFN_W2_BIAS_WORDS) {
            selected_valid = (uint32_t)FFN_W2_BIAS_WORDS;
        }
        selected_topfed_ffn_w2_bias_words_valid = (u32_t)selected_valid;
    } else {
        selected_topfed_ffn_w2_bias_words = topfed_ffn_w2_bias_words;
        selected_topfed_ffn_w2_bias_words_valid = (u32_t)w2_bias_words;
    }

    // FFN stage transition: W2.
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
        selected_topfed_ffn_w2_input_words,
        selected_topfed_ffn_w2_input_words_valid,
        selected_topfed_ffn_w2_words,
        selected_topfed_ffn_w2_words_valid,
        selected_topfed_ffn_w2_bias_words,
        selected_topfed_ffn_w2_bias_words_valid,
        (u32_t)FFN_POLICY_REQUIRE_W2_TOPFED
    );

    uint32_t residual_base = (uint32_t)sc.attn_out_base_word.to_uint();
    uint32_t w2_base = (uint32_t)sc.ffn.w2_out_base_word.to_uint();
    uint32_t add2_base = (uint32_t)sc.ffn.add2_base_word.to_uint();
    uint32_t words = (uint32_t)FFN_X_WORDS;
    // Write-back boundary: residual output becomes LayerNorm input.
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
