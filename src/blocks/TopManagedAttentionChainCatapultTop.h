#pragma once
// P00-011AL: Catapult-facing Top wrapper for attention-chain/full-loop slice.
// Boundary intent:
// - External interface uses fixed-size arrays/scalars only (no raw whole-SRAM pointer).
// - Wrapper remains the sole SRAM owner for this harness-level slice.
// - Existing local-validated Top orchestration helpers are reused internally.

#include <cstdint>

#include "AecctTypes.h"
#include "AttnDescBringup.h"
#include "LayerNormDesc.h"
#include "Top.h"
#include "TernaryLinearLive.h"
#include "TernaryLiveQkvLeafKernelShapeConfig.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"

#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#endif

#ifndef CCS_BLOCK
#define CCS_BLOCK(name) name
#endif

namespace aecct {

#pragma hls_design top
class TopManagedAttentionChainCatapultTop {
public:
    TopManagedAttentionChainCatapultTop() {}

#pragma hls_design interface
    bool CCS_BLOCK(run)(
        const u32_t x_in_words[ATTN_TENSOR_WORDS],
        const u32_t wq_payload_words[kQkvCtExpectedL0WqPayloadWords],
        u32_t wq_inv_sw_bits,
        const u32_t wk_payload_words[kQkvCtExpectedL0WkPayloadWords],
        u32_t wk_inv_sw_bits,
        const u32_t wv_payload_words[kQkvCtExpectedL0WvPayloadWords],
        u32_t wv_inv_sw_bits,
        u32_t attn_out_words[ATTN_TENSOR_WORDS],
        u32_t final_x_words[LN_X_TOTAL_WORDS],
        u32_t& out_mainline_all_taken,
        u32_t& out_fallback_taken
    ) {
        out_mainline_all_taken = (u32_t)0u;
        out_fallback_taken = (u32_t)1u;

        reset_local_sram();
        load_x_input_window(x_in_words);
        if (!load_qkv_payload_windows(
                wq_payload_words, wq_inv_sw_bits,
                wk_payload_words, wk_inv_sw_bits,
                wv_payload_words, wv_inv_sw_bits)) {
            return false;
        }

        regs_.clear();
        regs_.w_base_set = true;
        regs_.w_base_word = (u32_t)sram_map::W_REGION_BASE;
        regs_.cfg_d_model = (u32_t)ATTN_D_MODEL;
        regs_.cfg_n_heads = (u32_t)ATTN_N_HEADS;
        regs_.cfg_d_ffn = (u32_t)D_FFN;
        regs_.cfg_n_layers = (u32_t)1u;
        regs_.cfg_ready = true;

        run_transformer_layer_loop_top_managed_attn_bridge(regs_, sram_);

        const LayerScratch sc = make_layer_scratch((u32_t)LN_X_OUT_BASE_WORD);
        const uint32_t attn_out_base = (uint32_t)sc.attn_out_base_word.to_uint();
        COPY_ATTN_OUT_WINDOW_LOOP: for (uint32_t i = 0u; i < (uint32_t)ATTN_TENSOR_WORDS; ++i) {
            attn_out_words[i] = sram_[attn_out_base + i];
        }

        const uint32_t final_x_base = (uint32_t)regs_.infer_final_x_base_word.to_uint();
        COPY_FINAL_X_WINDOW_LOOP: for (uint32_t i = 0u; i < (uint32_t)LN_X_TOTAL_WORDS; ++i) {
            final_x_words[i] = sram_[final_x_base + i];
        }

        const bool mainline_all_taken =
            regs_.p11ac_mainline_path_taken &&
            regs_.p11ad_mainline_q_path_taken &&
            regs_.p11ae_mainline_score_path_taken &&
            regs_.p11af_mainline_softmax_output_path_taken;
        const bool fallback_taken =
            regs_.p11ac_fallback_taken ||
            regs_.p11ad_q_fallback_taken ||
            regs_.p11ae_score_fallback_taken ||
            regs_.p11af_softmax_output_fallback_taken;

        out_mainline_all_taken = mainline_all_taken ? (u32_t)1u : (u32_t)0u;
        out_fallback_taken = fallback_taken ? (u32_t)1u : (u32_t)0u;
        return (mainline_all_taken && !fallback_taken);
    }

private:
    u32_t sram_[sram_map::SRAM_WORDS_TOTAL];
    TopRegs regs_;

    void reset_local_sram() {
        RESET_LOCAL_SRAM_LOOP: for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
            sram_[i] = (u32_t)0u;
        }
    }

    void load_x_input_window(const u32_t x_in_words[ATTN_TENSOR_WORDS]) {
        const uint32_t x_base = (uint32_t)LN_X_OUT_BASE_WORD;
        LOAD_X_INPUT_WINDOW_LOOP: for (uint32_t i = 0u; i < (uint32_t)ATTN_TENSOR_WORDS; ++i) {
            sram_[x_base + i] = x_in_words[i];
        }
    }

    bool load_qkv_payload_windows(
        const u32_t wq_payload_words[kQkvCtExpectedL0WqPayloadWords],
        u32_t wq_inv_sw_bits,
        const u32_t wk_payload_words[kQkvCtExpectedL0WkPayloadWords],
        u32_t wk_inv_sw_bits,
        const u32_t wv_payload_words[kQkvCtExpectedL0WvPayloadWords],
        u32_t wv_inv_sw_bits
    ) {
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;

        const QuantLinearMeta wq_meta = ternary_linear_live_l0_wq_meta();
        const QuantLinearMeta wk_meta = ternary_linear_live_l0_wk_meta();
        const QuantLinearMeta wv_meta = ternary_linear_live_l0_wv_meta();

        if (wq_meta.payload_words_2b != kQkvCtExpectedL0WqPayloadWords) { return false; }
        if (wk_meta.payload_words_2b != kQkvCtExpectedL0WkPayloadWords) { return false; }
        if (wv_meta.payload_words_2b != kQkvCtExpectedL0WvPayloadWords) { return false; }

        const ParamMeta wq_payload_meta = kParamMeta[wq_meta.weight_param_id];
        const ParamMeta wk_payload_meta = kParamMeta[wk_meta.weight_param_id];
        const ParamMeta wv_payload_meta = kParamMeta[wv_meta.weight_param_id];
        const ParamMeta wq_inv_meta = kParamMeta[wq_meta.inv_sw_param_id];
        const ParamMeta wk_inv_meta = kParamMeta[wk_meta.inv_sw_param_id];
        const ParamMeta wv_inv_meta = kParamMeta[wv_meta.inv_sw_param_id];

        LOAD_WQ_PAYLOAD_LOOP: for (uint32_t i = 0u; i < kQkvCtExpectedL0WqPayloadWords; ++i) {
            sram_[param_base + wq_payload_meta.offset_w + i] = wq_payload_words[i];
        }
        LOAD_WK_PAYLOAD_LOOP: for (uint32_t i = 0u; i < kQkvCtExpectedL0WkPayloadWords; ++i) {
            sram_[param_base + wk_payload_meta.offset_w + i] = wk_payload_words[i];
        }
        LOAD_WV_PAYLOAD_LOOP: for (uint32_t i = 0u; i < kQkvCtExpectedL0WvPayloadWords; ++i) {
            sram_[param_base + wv_payload_meta.offset_w + i] = wv_payload_words[i];
        }

        sram_[param_base + wq_inv_meta.offset_w] = wq_inv_sw_bits;
        sram_[param_base + wk_inv_meta.offset_w] = wk_inv_sw_bits;
        sram_[param_base + wv_inv_meta.offset_w] = wv_inv_sw_bits;
        return true;
    }
};

} // namespace aecct
