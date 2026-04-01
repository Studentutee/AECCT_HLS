#pragma once
// Layer-0 attention datapath for the accepted local-only ternary/QKV mainline.
// Input: X_WORK row tiles and parameter-backed Q/K/V metadata.
// Intermediate: Q/K/V tensors, score/probability scratch, pre/post-concat buffers.
// Output: attention output tensor write-back to attn_out_base_word.
// Ownership boundary: Top owns global SRAM policy; this block consumes caller-provided windows.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "AttnDescBringup.h"
#include "QuantDesc.h"
#include "SoftmaxApprox.h"
#include "TernaryLinearLive.h"
#include "TernaryLiveQkvLeafKernelShapeConfig.h"
#include "TernaryLiveQkvLeafKernel.h"

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

static inline bool attn_live_qkv_gate_ok(
    const QuantLinearMeta& meta,
    QuantLinearMatrixId expected_matrix_id,
    uint32_t param_base_word,
    uint32_t token_count,
    uint32_t d_model
) {
    if (param_base_word == 0u || token_count == 0u) {
        return false;
    }
    if (meta.matrix_id != (uint32_t)expected_matrix_id) {
        return false;
    }
    if (meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
        return false;
    }
    if (meta.rows != d_model || meta.cols != d_model) {
        return false;
    }
    if (meta.num_weights != (meta.rows * meta.cols)) {
        return false;
    }
    if (meta.payload_words_2b == 0u) {
        return false;
    }
    if (meta.payload_words_2b != ternary_payload_words_2b(meta.num_weights)) {
        return false;
    }
    if (meta.last_word_valid_count == 0u || meta.last_word_valid_count > kQkvCtPackedWordElems) {
        return false;
    }
    if (meta.last_word_valid_count != ternary_last_word_valid_count(meta.num_weights)) {
        return false;
    }
    return true;
}

// local-only boundary descriptor for Top-fed prebuilt handoff flags.
// This keeps the AttnLayer0 seam contractized without changing Top ownership policy.
struct AttnLayer0PrebuiltHandoffDesc {
    bool kv_prebuilt_from_top_managed;
    bool q_prebuilt_from_top_managed;
    bool score_prebuilt_from_top_managed;
    bool out_prebuilt_from_top_managed;
};

static inline AttnLayer0PrebuiltHandoffDesc make_attn_layer0_prebuilt_handoff_desc(
    bool kv_prebuilt_from_top_managed = false,
    bool q_prebuilt_from_top_managed = false,
    bool score_prebuilt_from_top_managed = false,
    bool out_prebuilt_from_top_managed = false
) {
    AttnLayer0PrebuiltHandoffDesc desc;
    desc.kv_prebuilt_from_top_managed = kv_prebuilt_from_top_managed;
    desc.q_prebuilt_from_top_managed = q_prebuilt_from_top_managed;
    desc.score_prebuilt_from_top_managed = score_prebuilt_from_top_managed;
    desc.out_prebuilt_from_top_managed = out_prebuilt_from_top_managed;
    return desc;
}

template<unsigned STAGE_MODE, typename SramView>
// AttnLayer0 executes stage-scoped attention work in SRAM windows owned by Top.
// Ownership boundary: this block only consumes the base offsets provided by Top.
// TransformerLayer handoff boundary:
// - x_in_base_word is the layer input slice selected by TransformerLayer/Top.
// - attn_out_base_word is the output slice consumed by the downstream FFN stage.
// Bypass/fallback boundary:
// - q_prebuilt_from_top_managed skips Q materialization only.
// - kv_prebuilt_from_top_managed skips K/V materialization only.
// Write-back boundary:
// - score/softmax debug dumps write to score_base/softmax_base for (t=0,h=0).
// - stage OUT writes final tensor to attn_out_base_word.
static inline void AttnLayer0CoreWindow(
    SramView& sram,
    const AttnCfg& cfg,
    u32_t x_in_base_word,
    u32_t attn_out_base_word,
    const AttnScratch& sc,
    u32_t param_base_word = (u32_t)0,
    AttnLayer0PrebuiltHandoffDesc prebuilt_handoff = make_attn_layer0_prebuilt_handoff_desc()
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

    // QKV stage:
    // X_WORK + W_REGION metadata/payload -> Q/K/V (plus act_q mirrors) in attention scratch windows.
    if constexpr (STAGE_MODE == ATTN_STAGE_QKV || STAGE_MODE == ATTN_STAGE_FULL) {
        const uint32_t param_base = (uint32_t)param_base_word.to_uint();
        const QuantLinearMeta live_q_meta = ternary_linear_live_l0_wq_meta();
        const QuantLinearMeta live_k_meta = ternary_linear_live_l0_wk_meta();
        const QuantLinearMeta live_v_meta = ternary_linear_live_l0_wv_meta();
        const bool live_q_enabled =
            attn_live_qkv_gate_ok(live_q_meta, QLM_L0_WQ, param_base, token_count, d_model);
        const bool live_k_enabled =
            attn_live_qkv_gate_ok(live_k_meta, QLM_L0_WK, param_base, token_count, d_model);
        const bool live_v_enabled =
            attn_live_qkv_gate_ok(live_v_meta, QLM_L0_WV, param_base, token_count, d_model);
        bool live_q_ok = live_q_enabled;
        bool live_k_ok = live_k_enabled;
        bool live_v_ok = live_v_enabled;
        u32_t live_q_inv_sw_bits = (u32_t)0u;
        // P11AD mainline hook: Top-owned Q prebuild bypasses only Q generation.
        const bool skip_q_materialization = prebuilt_handoff.q_prebuilt_from_top_managed;
        // P11AC mainline hook: Top-owned K/V prebuild bypasses only K/V generation.
        const bool skip_kv_materialization = prebuilt_handoff.kv_prebuilt_from_top_managed;

        if (!skip_kv_materialization) {
            // Baseline priming path: copy X into K/V mirrors before live path overrides.
            ATTN_QKV_KV_PRIME_COPY_LOOP: for (uint32_t i = 0; i < tensor_words; ++i) {
                u32_t x = sram[x_in_base + i];
                sram[k_base + i] = x;
                sram[v_base + i] = x;
                sram[(uint32_t)sc.k_act_q_base_word.to_uint() + i] = x;
                sram[(uint32_t)sc.v_act_q_base_word.to_uint() + i] = x;
            }
        }

        if (!skip_q_materialization && live_q_enabled) {
            const uint32_t q_act_q_base = (uint32_t)sc.q_act_q_base_word.to_uint();
#if defined(AECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE)
            // Preferred Q path: fixed-shape split-kernel call over ternary leaf kernel.
            const ParamMeta live_q_payload_meta = kParamMeta[live_q_meta.weight_param_id];
            const ParamMeta live_q_inv_meta = kParamMeta[live_q_meta.inv_sw_param_id];
            if (live_q_meta.rows != kQkvCtSupportedL0WqRows ||
                live_q_meta.cols != kQkvCtSupportedL0WqCols ||
                live_q_meta.payload_words_2b != kQkvCtExpectedL0WqPayloadWords ||
                live_q_payload_meta.len_w < kQkvCtExpectedL0WqPayloadWords ||
                live_q_inv_meta.len_w == 0u) {
                live_q_ok = false;
            }
            u32_t payload_words[kTernaryLiveL0WqPayloadWords];
            u32_t x_row[kTernaryLiveL0WqCols];
            u32_t out_row[kTernaryLiveL0WqRows];
            u32_t out_act_q_row[kTernaryLiveL0WqRows];
            if (live_q_ok) {
                const uint32_t payload_base = param_base + live_q_payload_meta.offset_w;
                ATTN_Q_SPLIT_PAYLOAD_LOAD_LOOP: for (uint32_t i = 0u; i < kTernaryLiveL0WqPayloadWords; ++i) {
                    payload_words[i] = sram[payload_base + i];
                }
                const uint32_t inv_sw_addr = param_base + live_q_inv_meta.offset_w;
                const u32_t live_q_inv_sw_input = sram[inv_sw_addr];
                ATTN_Q_SPLIT_TOKEN_LOOP: for (uint32_t t = 0; t < token_count && live_q_ok; ++t) {
                    const uint32_t x_row_base = x_in_base + t * d_model;
                    const uint32_t q_row_base = q_base + t * d_model;
                    const uint32_t q_act_q_row_base = q_act_q_base + t * d_model;
                    ATTN_Q_SPLIT_INPUT_COL_LOOP: for (uint32_t in = 0u; in < kTernaryLiveL0WqCols; ++in) {
                        x_row[in] = sram[x_row_base + in];
                    }
                    u32_t out_inv_sw_bits = (u32_t)0u;
                    if (!ternary_live_l0_wq_materialize_row_kernel_split(
                            x_row,
                            payload_words,
                            live_q_inv_sw_input,
                            out_row,
                            out_act_q_row,
                            out_inv_sw_bits)) {
                        live_q_ok = false;
                        break;
                    }
                    if ((uint32_t)out_inv_sw_bits.to_uint() != (uint32_t)live_q_inv_sw_input.to_uint()) {
                        live_q_ok = false;
                        break;
                    }
                    ATTN_Q_SPLIT_OUTPUT_COL_LOOP: for (uint32_t out = 0u; out < kTernaryLiveL0WqRows; ++out) {
                        sram[q_row_base + out] = out_row[out];
                        sram[q_act_q_row_base + out] = out_act_q_row[out];
                    }
                    live_q_inv_sw_bits = out_inv_sw_bits;
                }
            }
#else
            ATTN_Q_FALLBACK_TOKEN_LOOP: for (uint32_t t = 0; t < token_count && live_q_ok; ++t) {
                const uint32_t x_row_base = x_in_base + t * d_model;
                const uint32_t q_row_base = q_base + t * d_model;
                const uint32_t q_act_q_row_base = q_act_q_base + t * d_model;
                ATTN_Q_FALLBACK_OUTPUT_COL_LOOP: for (uint32_t out = 0; out < live_q_meta.rows; ++out) {
                    u32_t q_bits = 0;
                    u32_t inv_sw_bits = 0;
                    if (!ternary_linear_live_l0_wq_compute_q_elem(
                            sram,
                            param_base_word,
                            (u32_t)x_row_base,
                            out,
                            q_bits,
                            inv_sw_bits)) {
                        live_q_ok = false;
                        break;
                    }
                    sram[q_row_base + out] = q_bits;
                    sram[q_act_q_row_base + out] = q_bits;
                    live_q_inv_sw_bits = inv_sw_bits;
                }
            }
#endif
        }

        if (!skip_q_materialization) {
            if (!live_q_ok) {
                const uint32_t q_act_q_base = (uint32_t)sc.q_act_q_base_word.to_uint();
                // Q fallback path is an explicit bypass copy from X into Q windows.
                ATTN_Q_BYPASS_COPY_LOOP: for (uint32_t i = 0; i < tensor_words; ++i) {
                    u32_t x = sram[x_in_base + i];
                    sram[q_base + i] = x;
                    sram[q_act_q_base + i] = x;
                }
                sram[(uint32_t)sc.q_sx_base_word.to_uint()] = bits_from_fp32(fp32_one());
            } else {
                sram[(uint32_t)sc.q_sx_base_word.to_uint()] = live_q_inv_sw_bits;
            }
        }

        if (!skip_kv_materialization && live_k_enabled) {
            const uint32_t k_act_q_base = (uint32_t)sc.k_act_q_base_word.to_uint();
#if defined(AECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE)
            // Preferred K path: fixed-shape split-kernel call over ternary leaf kernel.
            bool use_k_split_top = true;
            const ParamMeta live_k_payload_meta = kParamMeta[live_k_meta.weight_param_id];
            const ParamMeta live_k_inv_meta = kParamMeta[live_k_meta.inv_sw_param_id];
            if (live_k_meta.rows != kQkvCtSupportedL0WkRows ||
                live_k_meta.cols != kQkvCtSupportedL0WkCols ||
                live_k_meta.payload_words_2b != kQkvCtExpectedL0WkPayloadWords ||
                live_k_payload_meta.len_w < kQkvCtExpectedL0WkPayloadWords ||
                live_k_inv_meta.len_w == 0u) {
                use_k_split_top = false;
            }

            if (use_k_split_top) {
                u32_t payload_words[kTernaryLiveL0WkPayloadWords];
                u32_t x_row[kTernaryLiveL0WkCols];
                u32_t out_row[kTernaryLiveL0WkRows];
                u32_t out_act_q_row[kTernaryLiveL0WkRows];
                const uint32_t payload_base = param_base + live_k_payload_meta.offset_w;
                ATTN_K_SPLIT_PAYLOAD_LOAD_LOOP: for (uint32_t i = 0u; i < kTernaryLiveL0WkPayloadWords; ++i) {
                    payload_words[i] = sram[payload_base + i];
                }
                const uint32_t inv_sw_addr = param_base + live_k_inv_meta.offset_w;
                const u32_t live_k_inv_sw_input = sram[inv_sw_addr];
                ATTN_K_SPLIT_TOKEN_LOOP: for (uint32_t t = 0; t < token_count; ++t) {
                    const uint32_t x_row_base = x_in_base + t * d_model;
                    const uint32_t k_row_base = k_base + t * d_model;
                    const uint32_t k_act_q_row_base = k_act_q_base + t * d_model;
                    ATTN_K_SPLIT_INPUT_COL_LOOP: for (uint32_t in = 0u; in < kTernaryLiveL0WkCols; ++in) {
                        x_row[in] = sram[x_row_base + in];
                    }
                    u32_t out_inv_sw_bits = (u32_t)0u;
                    if (!ternary_live_l0_wk_materialize_row_kernel_split(
                            x_row,
                            payload_words,
                            live_k_inv_sw_input,
                            out_row,
                            out_act_q_row,
                            out_inv_sw_bits)) {
                        use_k_split_top = false;
                        break;
                    }
                    if ((uint32_t)out_inv_sw_bits.to_uint() != (uint32_t)live_k_inv_sw_input.to_uint()) {
                        use_k_split_top = false;
                        break;
                    }
                    ATTN_K_SPLIT_OUTPUT_COL_LOOP: for (uint32_t out = 0u; out < kTernaryLiveL0WkRows; ++out) {
                        sram[k_row_base + out] = out_row[out];
                        sram[k_act_q_row_base + out] = out_act_q_row[out];
                    }
                }
            }

            if (!use_k_split_top) {
                ATTN_K_FALLBACK_TOKEN_LOOP: for (uint32_t t = 0; t < token_count && live_k_ok; ++t) {
                    const uint32_t x_row_base = x_in_base + t * d_model;
                    const uint32_t k_row_base = k_base + t * d_model;
                    const uint32_t k_act_q_row_base = k_act_q_base + t * d_model;
                    ATTN_K_FALLBACK_OUTPUT_COL_LOOP: for (uint32_t out = 0; out < live_k_meta.rows; ++out) {
                        u32_t k_bits = 0;
                        u32_t k_inv_sw_bits = 0;
                        if (!ternary_linear_live_l0_wk_compute_q_elem(
                                sram,
                                param_base_word,
                                (u32_t)x_row_base,
                                out,
                                k_bits,
                                k_inv_sw_bits)) {
                            live_k_ok = false;
                            break;
                        }
                        sram[k_row_base + out] = k_bits;
                        sram[k_act_q_row_base + out] = k_bits;
                    }
                }
            }
#else
            ATTN_K_DIRECT_FALLBACK_TOKEN_LOOP: for (uint32_t t = 0; t < token_count && live_k_ok; ++t) {
                const uint32_t x_row_base = x_in_base + t * d_model;
                const uint32_t k_row_base = k_base + t * d_model;
                const uint32_t k_act_q_row_base = k_act_q_base + t * d_model;
                ATTN_K_DIRECT_FALLBACK_OUTPUT_COL_LOOP: for (uint32_t out = 0; out < live_k_meta.rows; ++out) {
                    u32_t k_bits = 0;
                    u32_t k_inv_sw_bits = 0;
                    if (!ternary_linear_live_l0_wk_compute_q_elem(
                            sram,
                            param_base_word,
                            (u32_t)x_row_base,
                            out,
                            k_bits,
                            k_inv_sw_bits)) {
                        live_k_ok = false;
                        break;
                    }
                    sram[k_row_base + out] = k_bits;
                    sram[k_act_q_row_base + out] = k_bits;
                }
            }
#endif
            if (!live_k_ok) {
                const uint32_t k_act_q_base = (uint32_t)sc.k_act_q_base_word.to_uint();
                // K fallback path is an explicit bypass copy from X into K windows.
                ATTN_K_BYPASS_COPY_LOOP: for (uint32_t i = 0; i < tensor_words; ++i) {
                    u32_t x = sram[x_in_base + i];
                    sram[k_base + i] = x;
                    sram[k_act_q_base + i] = x;
                }
            }
        }

        if (!skip_kv_materialization && live_v_enabled) {
            const uint32_t v_act_q_base = (uint32_t)sc.v_act_q_base_word.to_uint();
#if defined(AECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE)
            // Preferred V path: fixed-shape split-kernel call over ternary leaf kernel.
            bool use_v_split_top = true;
            const ParamMeta live_v_payload_meta = kParamMeta[live_v_meta.weight_param_id];
            const ParamMeta live_v_inv_meta = kParamMeta[live_v_meta.inv_sw_param_id];
            if (live_v_meta.rows != kQkvCtSupportedL0WvRows ||
                live_v_meta.cols != kQkvCtSupportedL0WvCols ||
                live_v_meta.payload_words_2b != kQkvCtExpectedL0WvPayloadWords ||
                live_v_payload_meta.len_w < kQkvCtExpectedL0WvPayloadWords ||
                live_v_inv_meta.len_w == 0u) {
                use_v_split_top = false;
            }

            if (use_v_split_top) {
                u32_t payload_words[kTernaryLiveL0WvPayloadWords];
                u32_t x_row[kTernaryLiveL0WvCols];
                u32_t out_row[kTernaryLiveL0WvRows];
                u32_t out_act_q_row[kTernaryLiveL0WvRows];
                const uint32_t payload_base = param_base + live_v_payload_meta.offset_w;
                ATTN_V_SPLIT_PAYLOAD_LOAD_LOOP: for (uint32_t i = 0u; i < kTernaryLiveL0WvPayloadWords; ++i) {
                    payload_words[i] = sram[payload_base + i];
                }
                const uint32_t inv_sw_addr = param_base + live_v_inv_meta.offset_w;
                const u32_t live_v_inv_sw_input = sram[inv_sw_addr];
                ATTN_V_SPLIT_TOKEN_LOOP: for (uint32_t t = 0; t < token_count; ++t) {
                    const uint32_t x_row_base = x_in_base + t * d_model;
                    const uint32_t v_row_base = v_base + t * d_model;
                    const uint32_t v_act_q_row_base = v_act_q_base + t * d_model;
                    ATTN_V_SPLIT_INPUT_COL_LOOP: for (uint32_t in = 0u; in < kTernaryLiveL0WvCols; ++in) {
                        x_row[in] = sram[x_row_base + in];
                    }
                    u32_t out_inv_sw_bits = (u32_t)0u;
                    if (!ternary_live_l0_wv_materialize_row_kernel_split(
                            x_row,
                            payload_words,
                            live_v_inv_sw_input,
                            out_row,
                            out_act_q_row,
                            out_inv_sw_bits)) {
                        use_v_split_top = false;
                        break;
                    }
                    if ((uint32_t)out_inv_sw_bits.to_uint() != (uint32_t)live_v_inv_sw_input.to_uint()) {
                        use_v_split_top = false;
                        break;
                    }
                    ATTN_V_SPLIT_OUTPUT_COL_LOOP: for (uint32_t out = 0u; out < kTernaryLiveL0WvRows; ++out) {
                        sram[v_row_base + out] = out_row[out];
                        sram[v_act_q_row_base + out] = out_act_q_row[out];
                    }
                }
            }

            if (!use_v_split_top) {
                ATTN_V_FALLBACK_TOKEN_LOOP: for (uint32_t t = 0; t < token_count && live_v_ok; ++t) {
                    const uint32_t x_row_base = x_in_base + t * d_model;
                    const uint32_t v_row_base = v_base + t * d_model;
                    const uint32_t v_act_q_row_base = v_act_q_base + t * d_model;
                    ATTN_V_FALLBACK_OUTPUT_COL_LOOP: for (uint32_t out = 0; out < live_v_meta.rows; ++out) {
                        u32_t v_bits = 0;
                        u32_t v_inv_sw_bits = 0;
                        if (!ternary_linear_live_l0_wv_compute_q_elem(
                                sram,
                                param_base_word,
                                (u32_t)x_row_base,
                                out,
                                v_bits,
                                v_inv_sw_bits)) {
                            live_v_ok = false;
                            break;
                        }
                        sram[v_row_base + out] = v_bits;
                        sram[v_act_q_row_base + out] = v_bits;
                    }
                }
            }
#else
            ATTN_V_DIRECT_FALLBACK_TOKEN_LOOP: for (uint32_t t = 0; t < token_count && live_v_ok; ++t) {
                const uint32_t x_row_base = x_in_base + t * d_model;
                const uint32_t v_row_base = v_base + t * d_model;
                const uint32_t v_act_q_row_base = v_act_q_base + t * d_model;
                ATTN_V_DIRECT_FALLBACK_OUTPUT_COL_LOOP: for (uint32_t out = 0; out < live_v_meta.rows; ++out) {
                    u32_t v_bits = 0;
                    u32_t v_inv_sw_bits = 0;
                    if (!ternary_linear_live_l0_wv_compute_q_elem(
                            sram,
                            param_base_word,
                            (u32_t)x_row_base,
                            out,
                            v_bits,
                            v_inv_sw_bits)) {
                        live_v_ok = false;
                        break;
                    }
                    sram[v_row_base + out] = v_bits;
                    sram[v_act_q_row_base + out] = v_bits;
                }
            }
#endif
            if (!live_v_ok) {
                const uint32_t v_act_q_base = (uint32_t)sc.v_act_q_base_word.to_uint();
                // V fallback path is an explicit bypass copy from X into V windows.
                ATTN_V_BYPASS_COPY_LOOP: for (uint32_t i = 0; i < tensor_words; ++i) {
                    u32_t x = sram[x_in_base + i];
                    sram[v_base + i] = x;
                    sram[v_act_q_base + i] = x;
                }
            }
        }
    }

    if constexpr (STAGE_MODE == ATTN_STAGE_SCORES || STAGE_MODE == ATTN_STAGE_FULL) {
        if (prebuilt_handoff.score_prebuilt_from_top_managed) {
            // Top-managed AE/AF path already produced score/pre/post for this layer invocation.
            // Keep existing AC/AD hooks untouched and skip duplicate score/softmax execution only.
        } else {
        // Scores stage:
        // Q/K/V -> scaled dot products -> softmax probabilities -> pre-concat accumulation.
        // Optional debug dumps are emitted only for (t=0, h=0) into score/softmax windows.
        softmax_score_t score_row[N_NODES];
        softmax_prob_t prob_row[N_NODES];
        ATTN_SCORE_TOKEN_LOOP: for (uint32_t t = 0; t < token_count; ++t) {
            ATTN_SCORE_HEAD_LOOP: for (uint32_t h = 0; h < n_heads; ++h) {
                uint32_t head_col_base = h * d_head;

                ATTN_SCORE_KEY_TOKEN_LOOP: for (uint32_t j = 0; j < token_count; ++j) {
                    quant_acc_t dot = 0;
                    uint32_t q_row = q_base + t * d_model + head_col_base;
                    uint32_t k_row = k_base + j * d_model + head_col_base;
                    ATTN_SCORE_DOT_COL_LOOP: for (uint32_t d = 0; d < d_head; ++d) {
                        quant_act_t qv = quant_act_from_bits(sram[q_row + d]);
                        quant_act_t kv = quant_act_from_bits(sram[k_row + d]);
                        dot += quant_acc_t(qv) * quant_acc_t(kv);
                    }
                    score_row[j] = softmax_score_t(dot * inv_sqrt_d_head);
                }

                SoftmaxApprox<N_NODES>(score_row, prob_row, token_count);

                if (t == 0u && h == 0u) {
                    ATTN_SCORE_DEBUG_DUMP_LOOP: for (uint32_t j = 0; j < token_count; ++j) {
                        fp32_t score_fp(score_row[j]);
                        fp32_t prob_fp(prob_row[j]);
                        sram[score_base + j] = bits_from_fp32(score_fp);
                        sram[softmax_base + j] = bits_from_fp32(prob_fp);
                    }
                }

                ATTN_PRECONCAT_HEAD_COL_LOOP: for (uint32_t d = 0; d < d_head; ++d) {
                    quant_acc_t acc = 0;
                    ATTN_PRECONCAT_KEY_TOKEN_ACC_LOOP: for (uint32_t j = 0; j < token_count; ++j) {
                        uint32_t v_idx = v_base + j * d_model + head_col_base + d;
                        quant_act_t vv = quant_act_from_bits(sram[v_idx]);
                        acc += quant_acc_t(prob_row[j]) * quant_acc_t(vv);
                    }
                    uint32_t out_idx = pre_base + t * d_model + head_col_base + d;
                    sram[out_idx] = quant_bits_from_acc(acc);
                }
            }
        }

        // Write-back boundary between pre-concat scratch and post-concat tensor.
        ATTN_POSTCONCAT_COPY_LOOP: for (uint32_t i = 0; i < tensor_words; ++i) {
            sram[post_base + i] = sram[pre_base + i];
        }
        }
    }

    if constexpr (STAGE_MODE == ATTN_STAGE_OUT || STAGE_MODE == ATTN_STAGE_FULL) {
        if (prebuilt_handoff.out_prebuilt_from_top_managed) {
            // Top-managed AF path already wrote final output for this layer invocation.
            return;
        }
        // OUT stage write-back boundary:
        // post-concat tensor -> caller-provided attn_out window for downstream layer glue.
        ATTN_OUT_WRITEBACK_LOOP: for (uint32_t i = 0; i < tensor_words; ++i) {
            sram[out_base + i] = sram[post_base + i];
        }
    }
}

template<unsigned STAGE_MODE>
static inline void AttnLayer0(
    u32_t* sram,
    const AttnCfg& cfg,
    u32_t x_in_base_word,
    u32_t attn_out_base_word,
    const AttnScratch& sc,
    u32_t param_base_word,
    AttnLayer0PrebuiltHandoffDesc prebuilt_handoff
) {
    AttnLayer0CoreWindow<STAGE_MODE, u32_t*>(
        sram,
        cfg,
        x_in_base_word,
        attn_out_base_word,
        sc,
        param_base_word,
        prebuilt_handoff
    );
}

template<unsigned STAGE_MODE>
static inline void AttnLayer0(
    u32_t* sram,
    const AttnCfg& cfg,
    u32_t x_in_base_word,
    u32_t attn_out_base_word,
    const AttnScratch& sc,
    u32_t param_base_word = (u32_t)0,
    bool kv_prebuilt_from_top_managed = false,
    bool q_prebuilt_from_top_managed = false,
    bool score_prebuilt_from_top_managed = false,
    bool out_prebuilt_from_top_managed = false
) {
    AttnLayer0CoreWindow<STAGE_MODE, u32_t*>(
        sram,
        cfg,
        x_in_base_word,
        attn_out_base_word,
        sc,
        param_base_word,
        make_attn_layer0_prebuilt_handoff_desc(
            kv_prebuilt_from_top_managed,
            q_prebuilt_from_top_managed,
            score_prebuilt_from_top_managed,
            out_prebuilt_from_top_managed)
    );
}

// P00-011AN: first deep Attn boundary bridge.
// This keeps the first deep call-site boundary array-shaped for Catapult-facing paths.
// Internal compute semantics remain the same by forwarding to the accepted AttnLayer0 core.
template<unsigned STAGE_MODE, uint32_t SRAM_WORDS>
static inline void AttnLayer0TopManagedWindowBridge(
    u32_t (&sram_window)[SRAM_WORDS],
    const AttnCfg& cfg,
    u32_t x_in_base_word,
    u32_t attn_out_base_word,
    const AttnScratch& sc,
    u32_t param_base_word,
    AttnLayer0PrebuiltHandoffDesc prebuilt_handoff
) {
    AttnLayer0CoreWindow<STAGE_MODE, u32_t (&)[SRAM_WORDS]>(
        sram_window,
        cfg,
        x_in_base_word,
        attn_out_base_word,
        sc,
        param_base_word,
        prebuilt_handoff
    );
}

template<unsigned STAGE_MODE, uint32_t SRAM_WORDS>
static inline void AttnLayer0TopManagedWindowBridge(
    u32_t (&sram_window)[SRAM_WORDS],
    const AttnCfg& cfg,
    u32_t x_in_base_word,
    u32_t attn_out_base_word,
    const AttnScratch& sc,
    u32_t param_base_word = (u32_t)0,
    bool kv_prebuilt_from_top_managed = false,
    bool q_prebuilt_from_top_managed = false,
    bool score_prebuilt_from_top_managed = false,
    bool out_prebuilt_from_top_managed = false
) {
    AttnLayer0CoreWindow<STAGE_MODE, u32_t (&)[SRAM_WORDS]>(
        sram_window,
        cfg,
        x_in_base_word,
        attn_out_base_word,
        sc,
        param_base_word,
        make_attn_layer0_prebuilt_handoff_desc(
            kv_prebuilt_from_top_managed,
            q_prebuilt_from_top_managed,
            score_prebuilt_from_top_managed,
            out_prebuilt_from_top_managed)
    );
}

} // namespace aecct
