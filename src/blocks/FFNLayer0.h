#pragma once
// Layer0 FFN block for the Transformer feed-forward stage.
// Input: attention output tensor or caller-fed FFN input payload.
// Intermediate: W1 MAC -> bias -> ReLU scratch -> W2 MAC -> bias.
// Output: FFN output tensor write-back to the caller-selected SRAM window.
// Ownership boundary: caller/Top owns shared-SRAM policy; this block only
// consumes the payload windows and descriptors passed in at the call boundary.

#include <cstdint>

#include "AecctTypes.h"
#include "AttnTopManagedPackets.h"
#include "FfnDescBringup.h"
#include "QuantDesc.h"
#include "gen/WeightStreamOrder.h"

namespace aecct {

static inline uint32_t ffn_param_addr_word(uint32_t param_base_word, uint32_t param_id, uint32_t elem_idx) {
    return param_base_word + kParamMeta[param_id].offset_w + elem_idx;
}

static inline quant_acc_t ffn_bias_from_sram(const u32_t* sram, uint32_t param_base_word, uint32_t param_id, uint32_t elem_idx) {
    fp32_t f = quant_bits_to_fp32(sram[ffn_param_addr_word(param_base_word, param_id, elem_idx)]);
    return f.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
}

static inline quant_acc_t ffn_bias_from_word(u32_t bits) {
    fp32_t f = quant_bits_to_fp32(bits);
    return f.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
}

static inline quant_w_t ffn_weight_from_sram(const u32_t* sram, uint32_t param_base_word, uint32_t param_id, uint32_t elem_idx) {
    return quant_w_t(quant_act_from_bits(sram[ffn_param_addr_word(param_base_word, param_id, elem_idx)]));
}

static inline fp32_t ffn_quantize_int8_symmetric(fp32_t x, fp32_t s_x) {
    fp32_t q = (x * s_x).round();
    if (q > fp32_t(127.0f)) { q = fp32_t(127.0f); }
    if (q < fp32_t(-127.0f)) { q = fp32_t(-127.0f); }
    return q;
}

static inline QuantLinearMatrixId ffn_w1_matrix_id_for_layer(bool use_layer1) {
    return use_layer1 ? QLM_L1_WFF1 : QLM_L0_WFF1;
}

static inline QuantLinearMatrixId ffn_w2_matrix_id_for_layer(bool use_layer1) {
    return use_layer1 ? QLM_L1_WFF2 : QLM_L0_WFF2;
}

static inline uint32_t ffn_w1_sx_slot_for_layer(bool use_layer1) {
    return use_layer1 ? 6u : 2u;
}

static inline uint32_t ffn_w2_sx_slot_for_layer(bool use_layer1) {
    return use_layer1 ? 7u : 3u;
}

template<typename SramView>
static inline bool ffn_quant_linear_contract_ok(
    SramView& sram,
    uint32_t param_base_word,
    const QuantLinearMeta& meta,
    QuantLinearMatrixId expected_matrix_id,
    uint32_t expect_weight_param_id,
    uint32_t expect_bias_param_id,
    uint32_t expect_rows,
    uint32_t expect_cols,
    uint32_t sx_slot,
    fp32_t& out_s_x,
    fp32_t& out_inv_scale
) {
    if (meta.matrix_id != (uint32_t)expected_matrix_id) {
        return false;
    }
    if (meta.weight_param_id != expect_weight_param_id) {
        return false;
    }
    if (meta.rows != expect_rows || meta.cols != expect_cols) {
        return false;
    }
    if (meta.num_weights != (meta.rows * meta.cols)) {
        return false;
    }
    if (kParamMeta[meta.weight_param_id].len_w < meta.num_weights) {
        return false;
    }
    if (kParamMeta[expect_bias_param_id].len_w < meta.rows) {
        return false;
    }
    if (kParamMeta[meta.inv_sw_param_id].len_w == 0u) {
        return false;
    }
    uint32_t sx_param_id = 0u;
    if (!weight_id_to_param_id(QUANT_SX_8, sx_param_id)) {
        return false;
    }
    if (kParamMeta[sx_param_id].len_w <= sx_slot) {
        return false;
    }

    const uint32_t sx_addr = param_base_word + kParamMeta[sx_param_id].offset_w + sx_slot;
    const uint32_t inv_sw_addr = param_base_word + kParamMeta[meta.inv_sw_param_id].offset_w;
    const u32_t sx_bits = sram[sx_addr];
    const u32_t inv_sw_bits = sram[inv_sw_addr];
    if ((uint32_t)sx_bits.to_uint() == 0u || (uint32_t)inv_sw_bits.to_uint() == 0u) {
        return false;
    }

    const fp32_t s_x = fp32_from_bits(sx_bits);
    const fp32_t inv_sw = fp32_from_bits(inv_sw_bits);
    out_s_x = s_x;
    out_inv_scale = inv_sw / s_x;
    return true;
}

static inline void ffn_count_legacy_fallback_touch(u32_t* fallback_touch_counter) {
    if (fallback_touch_counter != 0) {
        const uint32_t v = (uint32_t)fallback_touch_counter->to_uint();
        *fallback_touch_counter = (u32_t)(v + 1u);
    }
}

struct FfnTopManagedTileMeta {
    u16_t phase_id;
    u16_t subphase_id;
    u16_t token_begin;
    u16_t token_end;
    u16_t token_idx;
    u16_t tile_begin;
    u16_t tile_end;
    u16_t tile_idx;
    u16_t tile_valid_words;
};

static inline bool ffn_top_managed_tile_meta_ok(
    const FfnTopManagedTileMeta& m,
    uint32_t expect_token_idx,
    uint32_t expect_tile_idx
) {
    if ((uint32_t)m.phase_id.to_uint() != (uint32_t)ATTN_PHASE_C) {
        return false;
    }
    if ((uint32_t)m.token_idx.to_uint() != expect_token_idx) {
        return false;
    }
    if ((uint32_t)m.tile_idx.to_uint() != expect_tile_idx) {
        return false;
    }
    const uint32_t t_begin = (uint32_t)m.token_begin.to_uint();
    const uint32_t t_end = (uint32_t)m.token_end.to_uint();
    const uint32_t dt_begin = (uint32_t)m.tile_begin.to_uint();
    const uint32_t dt_end = (uint32_t)m.tile_end.to_uint();
    const uint32_t valid = (uint32_t)m.tile_valid_words.to_uint();
    if (t_end <= t_begin) { return false; }
    if (dt_end <= dt_begin) { return false; }
    if (valid == 0u || valid > (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS) {
        return false;
    }
    return true;
}

static inline quant_acc_t ffn_block_mac_tile(
    const FfnTopManagedTileMeta& meta,
    const u32_t x_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS],
    const u32_t w_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS],
    quant_acc_t acc
) {
    const uint32_t valid = (uint32_t)meta.tile_valid_words.to_uint();
    FFN_BLOCK_MAC_TILE_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
        const quant_act_t xv = quant_act_from_bits(x_tile[i]);
        const quant_w_t wv = quant_act_from_bits(w_tile[i]);
        acc += quant_acc_t(xv) * quant_acc_t(wv);
    }
    return acc;
}

static inline fp32_t ffn_block_mac_tile_quant_fp32(
    const FfnTopManagedTileMeta& meta,
    const u32_t x_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS],
    const u32_t w_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS],
    fp32_t s_x,
    fp32_t inv_scale,
    fp32_t acc
) {
    const uint32_t valid = (uint32_t)meta.tile_valid_words.to_uint();
    FFN_BLOCK_MAC_TILE_QUANT_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
        const fp32_t x_fp = fp32_from_bits(x_tile[i]);
        const fp32_t q_fp = ffn_quantize_int8_symmetric(x_fp, s_x);
        const fp32_t w_fp = fp32_from_bits(w_tile[i]);
        acc += q_fp * (w_fp * inv_scale);
    }
    return acc;
}

static inline void ffn_block_relu_tile(
    const FfnTopManagedTileMeta& meta,
    const u32_t in_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS],
    u32_t out_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS]
) {
    const uint32_t valid = (uint32_t)meta.tile_valid_words.to_uint();
    FFN_BLOCK_RELU_TILE_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
        const quant_act_t v = quant_act_from_bits(in_tile[i]);
        const quant_act_t y = (v > quant_act_t(0)) ? v : quant_act_t(0);
        out_tile[i] = quant_bits_from_act(y);
    }
}

// Core FFN worker.
// Read in this order:
// 1) W1 path (input + W1 + optional caller-fed bias)
// 2) ReLU staging into FFN scratch
// 3) W2 path (ReLU output + W2 + optional caller-fed bias)
// Fallback boundary: when caller-fed descriptors are absent or invalid,
// this worker falls back to local SRAM reads inside the same caller-owned window.
template<unsigned STAGE_MODE, typename SramView>
static inline void FFNLayer0CoreWindow(
    SramView& sram,
    const FfnCfg& cfg,
    u32_t x_in_base_word,
    const FfnScratch& sc,
    u32_t param_base_word,
    u32_t layer_id = (u32_t)0,
    const u32_t* topfed_x_words = 0,
    const u32_t* topfed_w1_weight_words = 0,
    u32_t topfed_w1_weight_words_valid = 0,
    const u32_t* topfed_w2_input_words = 0,
    u32_t topfed_w2_input_words_valid = 0,
    const u32_t* topfed_w2_weight_words = 0,
    u32_t topfed_w2_weight_words_valid = 0,
    const u32_t* topfed_w2_bias_words = 0,
    u32_t topfed_w2_bias_words_valid = 0,
    u32_t fallback_policy_flags = (u32_t)FFN_POLICY_NONE,
    u32_t* fallback_policy_reject_flag = 0,
    u32_t* fallback_legacy_touch_counter = 0,
    u32_t topfed_x_words_valid_override = 0,
    const u32_t* topfed_w1_bias_words = 0,
    u32_t topfed_w1_bias_words_valid = 0,
    u32_t* fallback_policy_reject_stage = 0
) {
    uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t d_ffn = (uint32_t)cfg.d_ffn.to_uint();
    if (token_count == 0u) { token_count = (uint32_t)FFN_TOKEN_COUNT; }
    if (d_model == 0u) { d_model = (uint32_t)FFN_D_MODEL; }
    if (d_ffn == 0u) { d_ffn = (uint32_t)FFN_D_FFN; }
    if (token_count == 0u || d_model == 0u || d_ffn == 0u) {
        return;
    }
    if (fallback_policy_reject_flag != 0) {
        *fallback_policy_reject_flag = (u32_t)0u;
    }
    if (fallback_legacy_touch_counter != 0) {
        *fallback_legacy_touch_counter = (u32_t)0u;
    }
    if (fallback_policy_reject_stage != 0) {
        *fallback_policy_reject_stage = (u32_t)FFN_REJECT_STAGE_NONE;
    }

    const bool use_layer1 = ((uint32_t)layer_id.to_uint() == 1u);
    const uint32_t x_in_base = (uint32_t)x_in_base_word.to_uint();
    const uint32_t w1_base = (uint32_t)sc.w1_out_base_word.to_uint();
    const uint32_t relu_base = (uint32_t)sc.relu_out_base_word.to_uint();
    const uint32_t w2_base = (uint32_t)sc.w2_out_base_word.to_uint();
    const uint32_t param_base = (uint32_t)param_base_word.to_uint();

    const uint32_t w1_bias_id = use_layer1 ? 12u : 4u;
    const uint32_t w1_weight_id = use_layer1 ? 56u : 36u;
    const uint32_t w2_bias_id = use_layer1 ? 13u : 5u;
    const uint32_t w2_weight_id = use_layer1 ? 59u : 39u;
    const uint32_t topfed_x_raw_valid = (uint32_t)topfed_x_words_valid_override.to_uint();
    uint32_t topfed_x_valid = topfed_x_raw_valid;
    if (topfed_x_valid == 0u) {
        topfed_x_valid = token_count * d_model;
    }
    if (topfed_x_valid > (uint32_t)FFN_X_WORDS) {
        topfed_x_valid = (uint32_t)FFN_X_WORDS;
    }
    const uint32_t topfed_w1_raw_valid = (uint32_t)topfed_w1_weight_words_valid.to_uint();
    uint32_t topfed_w1_valid = (uint32_t)topfed_w1_weight_words_valid.to_uint();
    if (topfed_w1_valid == 0u) {
        topfed_w1_valid = d_ffn * d_model;
    }
    if (topfed_w1_valid > (uint32_t)FFN_W1_WEIGHT_WORDS) {
        topfed_w1_valid = (uint32_t)FFN_W1_WEIGHT_WORDS;
    }
    const uint32_t topfed_w1_bias_raw_valid = (uint32_t)topfed_w1_bias_words_valid.to_uint();
    uint32_t topfed_w1_bias_valid = topfed_w1_bias_raw_valid;
    if (topfed_w1_bias_valid == 0u) {
        topfed_w1_bias_valid = d_ffn;
    }
    if (topfed_w1_bias_valid > (uint32_t)FFN_W1_BIAS_WORDS) {
        topfed_w1_bias_valid = (uint32_t)FFN_W1_BIAS_WORDS;
    }
    const uint32_t topfed_w2_input_raw_valid = (uint32_t)topfed_w2_input_words_valid.to_uint();
    const uint32_t topfed_w2_weight_raw_valid = (uint32_t)topfed_w2_weight_words_valid.to_uint();
    const uint32_t topfed_w2_bias_raw_valid = (uint32_t)topfed_w2_bias_words_valid.to_uint();
    uint32_t topfed_w2_input_valid = topfed_w2_input_raw_valid;
    if (topfed_w2_input_valid > (uint32_t)FFN_W2_INPUT_WORDS) {
        topfed_w2_input_valid = (uint32_t)FFN_W2_INPUT_WORDS;
    }
    uint32_t topfed_w2_weight_valid = topfed_w2_weight_raw_valid;
    if (topfed_w2_weight_valid > (uint32_t)FFN_W2_WEIGHT_WORDS) {
        topfed_w2_weight_valid = (uint32_t)FFN_W2_WEIGHT_WORDS;
    }
    uint32_t topfed_w2_bias_valid = topfed_w2_bias_raw_valid;
    if (topfed_w2_bias_valid > (uint32_t)FFN_W2_BIAS_WORDS) {
        topfed_w2_bias_valid = (uint32_t)FFN_W2_BIAS_WORDS;
    }
    const uint32_t expected_w2_input_words = token_count * d_ffn;
    const uint32_t expected_w2_weight_words = d_model * d_ffn;
    const uint32_t expected_w2_bias_words = d_model;
    const uint32_t expected_w1_x_words = token_count * d_model;
    const uint32_t expected_w1_weight_words = d_ffn * d_model;
    const uint32_t expected_w1_bias_words = d_ffn;
    // Descriptor readiness checks are Top-to-block handoff gates, not ownership decisions.
    const bool w1_input_descriptor_ready =
        (topfed_x_words != 0) && (topfed_x_raw_valid >= expected_w1_x_words);
    const bool w1_weight_descriptor_ready =
        (topfed_w1_weight_words != 0) && (topfed_w1_raw_valid >= expected_w1_weight_words);
    const bool w1_bias_descriptor_ready =
        (topfed_w1_bias_words != 0) && (topfed_w1_bias_raw_valid >= expected_w1_bias_words);
    const bool w2_input_descriptor_ready =
        (topfed_w2_input_words != 0) && (topfed_w2_input_raw_valid >= expected_w2_input_words);
    const bool w2_weight_descriptor_ready =
        (topfed_w2_weight_words != 0) && (topfed_w2_weight_raw_valid >= expected_w2_weight_words);
    const bool w2_bias_descriptor_ready =
        (topfed_w2_bias_words != 0) && (topfed_w2_bias_raw_valid >= expected_w2_bias_words);
    // Policy bits can require full Top-fed coverage and reject legacy fallback for this stage.
    const bool require_w1_topfed =
        (((uint32_t)fallback_policy_flags.to_uint() & (uint32_t)FFN_POLICY_REQUIRE_W1_TOPFED) != 0u);
    // Same policy gate for W2 descriptors.
    const bool require_w2_topfed =
        (((uint32_t)fallback_policy_flags.to_uint() & (uint32_t)FFN_POLICY_REQUIRE_W2_TOPFED) != 0u);

    const uint32_t tile_words = (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS;
    const uint32_t d_model_tile_count = attn_top_managed_tile_count(d_model, tile_words);
    const uint32_t d_ffn_tile_count = attn_top_managed_tile_count(d_ffn, tile_words);
    if (d_model_tile_count == 0u || d_ffn_tile_count == 0u) {
        return;
    }

    const QuantLinearMatrixId w1_matrix_id = ffn_w1_matrix_id_for_layer(use_layer1);
    const QuantLinearMeta& w1_meta = kQuantLinearMeta[(uint32_t)w1_matrix_id];
    fp32_t w1_s_x = fp32_t(0.0f);
    fp32_t w1_inv_scale = fp32_t(0.0f);
    const bool w1_quant_contract_ok = ffn_quant_linear_contract_ok(
        sram,
        param_base,
        w1_meta,
        w1_matrix_id,
        w1_weight_id,
        w1_bias_id,
        d_ffn,
        d_model,
        ffn_w1_sx_slot_for_layer(use_layer1),
        w1_s_x,
        w1_inv_scale);
    const QuantLinearMatrixId w2_matrix_id = ffn_w2_matrix_id_for_layer(use_layer1);
    const QuantLinearMeta& w2_meta = kQuantLinearMeta[(uint32_t)w2_matrix_id];
    fp32_t w2_s_x = fp32_t(0.0f);
    fp32_t w2_inv_scale = fp32_t(0.0f);
    const bool w2_quant_contract_ok = ffn_quant_linear_contract_ok(
        sram,
        param_base,
        w2_meta,
        w2_matrix_id,
        w2_weight_id,
        w2_bias_id,
        d_model,
        d_ffn,
        ffn_w2_sx_slot_for_layer(use_layer1),
        w2_s_x,
        w2_inv_scale);

    if (STAGE_MODE == FFN_STAGE_W1 || STAGE_MODE == FFN_STAGE_FULL) {
        // Tightened fallback policy: caller can require fully ready W1 descriptors.
        if (require_w1_topfed &&
            !(w1_input_descriptor_ready && w1_weight_descriptor_ready && w1_bias_descriptor_ready)) {
            if (fallback_policy_reject_flag != 0) {
                *fallback_policy_reject_flag = (u32_t)1u;
            }
            if (fallback_policy_reject_stage != 0) {
                *fallback_policy_reject_stage = (u32_t)FFN_REJECT_STAGE_W1;
            }
            return;
        }
        // Stage W1: MAC over input tile + W1 tile + bias.
        FFN_TOP_MANAGED_W1_TOKEN_LOOP: for (uint32_t t = 0u; t < token_count; ++t) {
            const uint32_t x_row = x_in_base + t * d_model;
            const uint32_t h_row = w1_base + t * d_ffn;
            FFN_TOP_MANAGED_W1_OUT_LOOP: for (uint32_t j = 0u; j < d_ffn; ++j) {
                u32_t bias_bits = sram[ffn_param_addr_word(param_base, w1_bias_id, j)];
                // Caller-fed W1 bias payload path: use preloaded bias words when provided.
                if (topfed_w1_bias_words != 0 && j < topfed_w1_bias_valid) {
                    bias_bits = topfed_w1_bias_words[j];
                } else {
                    ffn_count_legacy_fallback_touch(fallback_legacy_touch_counter);
                }
                quant_acc_t acc = ffn_bias_from_word(bias_bits);
                fp32_t acc_fp = fp32_from_bits(bias_bits);
                const uint32_t w_row = j * d_model;
                FFN_TOP_MANAGED_W1_TILE_LOOP: for (uint32_t dt = 0u; dt < d_model_tile_count; ++dt) {
                    const uint32_t tile_offset = dt * tile_words;
                    const uint32_t valid =
                        attn_top_managed_tile_valid_words(d_model, tile_words, dt);
                    FfnTopManagedTileMeta meta;
                    meta.phase_id = (u16_t)ATTN_PHASE_C;
                    meta.subphase_id = (u16_t)ATTN_SUBPHASE_QSRC;
                    meta.token_begin = (u16_t)0u;
                    meta.token_end = (u16_t)token_count;
                    meta.token_idx = (u16_t)t;
                    meta.tile_begin = (u16_t)0u;
                    meta.tile_end = (u16_t)d_model_tile_count;
                    meta.tile_idx = (u16_t)dt;
                    meta.tile_valid_words = (u16_t)valid;
                    if (!ffn_top_managed_tile_meta_ok(meta, t, dt)) {
                        continue;
                    }

                    u32_t x_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS];
                    u32_t w_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS];
                    FFN_TOP_MANAGED_W1_TILE_LOAD_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                        const uint32_t x_idx = t * d_model + tile_offset + i;
                        const uint32_t w1_idx = w_row + tile_offset + i;
                        // Top-fed FFN input payload path: caller can preload and dispatch x words.
                        if (topfed_x_words != 0 && x_idx < topfed_x_valid) {
                            x_tile[i] = topfed_x_words[x_idx];
                        } else {
                            ffn_count_legacy_fallback_touch(fallback_legacy_touch_counter);
                            x_tile[i] = sram[x_row + tile_offset + i];
                        }
                        // Caller-fed FFN W1 weight path: consume preloaded tiles when provided.
                        if (topfed_w1_weight_words != 0 && w1_idx < topfed_w1_valid) {
                            w_tile[i] = topfed_w1_weight_words[w1_idx];
                        } else {
                            ffn_count_legacy_fallback_touch(fallback_legacy_touch_counter);
                            w_tile[i] = sram[ffn_param_addr_word(param_base, w1_weight_id, w1_idx)];
                        }
                    }
                    if (w1_quant_contract_ok) {
                        acc_fp = ffn_block_mac_tile_quant_fp32(meta, x_tile, w_tile, w1_s_x, w1_inv_scale, acc_fp);
                    } else {
                        acc = ffn_block_mac_tile(meta, x_tile, w_tile, acc);
                    }
                }
                if (w1_quant_contract_ok) {
                    sram[h_row + j] = bits_from_fp32(acc_fp);
                } else {
                    sram[h_row + j] = quant_bits_from_acc(acc);
                }
            }
        }
    }

    if (STAGE_MODE == FFN_STAGE_RELU || STAGE_MODE == FFN_STAGE_FULL) {
        // Stage ReLU: apply activation on W1 output scratch.
        FFN_TOP_MANAGED_RELU_TOKEN_LOOP: for (uint32_t t = 0u; t < token_count; ++t) {
            const uint32_t h_row = w1_base + t * d_ffn;
            const uint32_t a_row = relu_base + t * d_ffn;
            FFN_TOP_MANAGED_RELU_TILE_LOOP: for (uint32_t dt = 0u; dt < d_ffn_tile_count; ++dt) {
                const uint32_t tile_offset = dt * tile_words;
                const uint32_t valid =
                    attn_top_managed_tile_valid_words(d_ffn, tile_words, dt);
                FfnTopManagedTileMeta meta;
                meta.phase_id = (u16_t)ATTN_PHASE_C;
                meta.subphase_id = (u16_t)ATTN_SUBPHASE_MASK;
                meta.token_begin = (u16_t)0u;
                meta.token_end = (u16_t)token_count;
                meta.token_idx = (u16_t)t;
                meta.tile_begin = (u16_t)0u;
                meta.tile_end = (u16_t)d_ffn_tile_count;
                meta.tile_idx = (u16_t)dt;
                meta.tile_valid_words = (u16_t)valid;
                if (!ffn_top_managed_tile_meta_ok(meta, t, dt)) {
                    continue;
                }

                u32_t in_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS];
                u32_t out_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS];
                FFN_TOP_MANAGED_RELU_TILE_LOAD_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                    in_tile[i] = sram[h_row + tile_offset + i];
                }
                ffn_block_relu_tile(meta, in_tile, out_tile);
                FFN_TOP_MANAGED_RELU_TILE_WRITEBACK_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                    sram[a_row + tile_offset + i] = out_tile[i];
                }
            }
        }
    }

    if (STAGE_MODE == FFN_STAGE_W2 || STAGE_MODE == FFN_STAGE_FULL) {
        // Tightened fallback policy: caller can require fully ready W2 descriptors.
        if (require_w2_topfed && !(w2_input_descriptor_ready && w2_weight_descriptor_ready && w2_bias_descriptor_ready)) {
            if (fallback_policy_reject_flag != 0) {
                *fallback_policy_reject_flag = (u32_t)1u;
            }
            if (fallback_policy_reject_stage != 0) {
                *fallback_policy_reject_stage = (u32_t)FFN_REJECT_STAGE_W2;
            }
            return;
        }
        // Stage W2: MAC over ReLU scratch + W2 tile + bias.
        FFN_TOP_MANAGED_W2_TOKEN_LOOP: for (uint32_t t = 0u; t < token_count; ++t) {
            const uint32_t a_row = relu_base + t * d_ffn;
            const uint32_t y_row = w2_base + t * d_model;
            FFN_TOP_MANAGED_W2_OUT_LOOP: for (uint32_t i = 0u; i < d_model; ++i) {
                u32_t bias_bits = sram[ffn_param_addr_word(param_base, w2_bias_id, i)];
                // Caller-fed W2 bias payload path: use preloaded bias words when provided.
                if (topfed_w2_bias_words != 0 && i < topfed_w2_bias_valid) {
                    bias_bits = topfed_w2_bias_words[i];
                } else {
                    ffn_count_legacy_fallback_touch(fallback_legacy_touch_counter);
                }
                quant_acc_t acc = ffn_bias_from_word(bias_bits);
                fp32_t acc_fp = fp32_from_bits(bias_bits);
                const uint32_t w_row = i * d_ffn;
                FFN_TOP_MANAGED_W2_TILE_LOOP: for (uint32_t dt = 0u; dt < d_ffn_tile_count; ++dt) {
                    const uint32_t tile_offset = dt * tile_words;
                    const uint32_t valid =
                        attn_top_managed_tile_valid_words(d_ffn, tile_words, dt);
                    FfnTopManagedTileMeta meta;
                    meta.phase_id = (u16_t)ATTN_PHASE_C;
                    meta.subphase_id = (u16_t)ATTN_SUBPHASE_WO;
                    meta.token_begin = (u16_t)0u;
                    meta.token_end = (u16_t)token_count;
                    meta.token_idx = (u16_t)t;
                    meta.tile_begin = (u16_t)0u;
                    meta.tile_end = (u16_t)d_ffn_tile_count;
                    meta.tile_idx = (u16_t)dt;
                    meta.tile_valid_words = (u16_t)valid;
                    if (!ffn_top_managed_tile_meta_ok(meta, t, dt)) {
                        continue;
                    }

                    u32_t a_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS];
                    u32_t w_tile[ATTN_TOP_MANAGED_WORK_TILE_WORDS];
                    FFN_TOP_MANAGED_W2_TILE_LOAD_LOOP: for (uint32_t k = 0u; k < valid; ++k) {
                        const uint32_t a_idx = t * d_ffn + tile_offset + k;
                        const uint32_t w2_idx = w_row + tile_offset + k;
                        // Caller-fed W2 input payload path: consume preloaded ReLU output tiles.
                        if (topfed_w2_input_words != 0 && a_idx < topfed_w2_input_valid) {
                            a_tile[k] = topfed_w2_input_words[a_idx];
                        } else {
                            ffn_count_legacy_fallback_touch(fallback_legacy_touch_counter);
                            a_tile[k] = sram[a_row + tile_offset + k];
                        }
                        // Caller-fed W2 weight payload path: consume preloaded W2 weights.
                        if (topfed_w2_weight_words != 0 && w2_idx < topfed_w2_weight_valid) {
                            w_tile[k] = topfed_w2_weight_words[w2_idx];
                        } else {
                            ffn_count_legacy_fallback_touch(fallback_legacy_touch_counter);
                            w_tile[k] = sram[ffn_param_addr_word(param_base, w2_weight_id, w2_idx)];
                        }
                    }
                    if (w2_quant_contract_ok) {
                        acc_fp = ffn_block_mac_tile_quant_fp32(meta, a_tile, w_tile, w2_s_x, w2_inv_scale, acc_fp);
                    } else {
                        acc = ffn_block_mac_tile(meta, a_tile, w_tile, acc);
                    }
                }
                if (w2_quant_contract_ok) {
                    sram[y_row + i] = bits_from_fp32(acc_fp);
                } else {
                    sram[y_row + i] = quant_bits_from_acc(acc);
                }
            }
        }
    }
}

template<unsigned STAGE_MODE, typename SramView>
static inline void FFNLayer0CoreWindowDirect(
    SramView& sram,
    const FfnCfg& cfg,
    u32_t x_in_base_word,
    const FfnScratch& sc,
    u32_t param_base_word,
    u32_t layer_id = (u32_t)0
) {
    uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t d_ffn = (uint32_t)cfg.d_ffn.to_uint();
    bool use_layer1 = ((uint32_t)layer_id.to_uint() == 1u);

    uint32_t x_in_base = (uint32_t)x_in_base_word.to_uint();
    uint32_t w1_base = (uint32_t)sc.w1_out_base_word.to_uint();
    uint32_t relu_base = (uint32_t)sc.relu_out_base_word.to_uint();
    uint32_t w2_base = (uint32_t)sc.w2_out_base_word.to_uint();
    uint32_t param_base = (uint32_t)param_base_word.to_uint();

    const uint32_t w1_bias_id = use_layer1 ? 12u : 4u;
    const uint32_t w1_weight_id = use_layer1 ? 56u : 36u;
    const uint32_t w2_bias_id = use_layer1 ? 13u : 5u;
    const uint32_t w2_weight_id = use_layer1 ? 59u : 39u;

    if (STAGE_MODE == FFN_STAGE_W1 || STAGE_MODE == FFN_STAGE_FULL) {
        for (uint32_t t = 0; t < token_count; ++t) {
            uint32_t x_row = x_in_base + t * d_model;
            uint32_t h_row = w1_base + t * d_ffn;
            for (uint32_t j = 0; j < d_ffn; ++j) {
                quant_acc_t acc = ffn_bias_from_sram(sram, param_base, w1_bias_id, j);
                uint32_t w_row = j * d_model;
                for (uint32_t i = 0; i < d_model; ++i) {
                    quant_act_t x = quant_act_from_bits(sram[x_row + i]);
                    quant_w_t w = ffn_weight_from_sram(sram, param_base, w1_weight_id, w_row + i);
                    acc += quant_acc_t(x) * quant_acc_t(w);
                }
                sram[h_row + j] = quant_bits_from_acc(acc);
            }
        }
    }

    if (STAGE_MODE == FFN_STAGE_RELU || STAGE_MODE == FFN_STAGE_FULL) {
        for (uint32_t t = 0; t < token_count; ++t) {
            uint32_t h_row = w1_base + t * d_ffn;
            uint32_t a_row = relu_base + t * d_ffn;
            for (uint32_t j = 0; j < d_ffn; ++j) {
                quant_act_t v = quant_act_from_bits(sram[h_row + j]);
                quant_act_t y = (v > quant_act_t(0)) ? v : quant_act_t(0);
                sram[a_row + j] = quant_bits_from_act(y);
            }
        }
    }

    if (STAGE_MODE == FFN_STAGE_W2 || STAGE_MODE == FFN_STAGE_FULL) {
        for (uint32_t t = 0; t < token_count; ++t) {
            uint32_t a_row = relu_base + t * d_ffn;
            uint32_t y_row = w2_base + t * d_model;
            for (uint32_t i = 0; i < d_model; ++i) {
                quant_acc_t acc = ffn_bias_from_sram(sram, param_base, w2_bias_id, i);
                uint32_t w_row = i * d_ffn;
                for (uint32_t j = 0; j < d_ffn; ++j) {
                    quant_act_t a = quant_act_from_bits(sram[a_row + j]);
                    quant_w_t w = ffn_weight_from_sram(sram, param_base, w2_weight_id, w_row + j);
                    acc += quant_acc_t(a) * quant_acc_t(w);
                }
                sram[y_row + i] = quant_bits_from_acc(acc);
            }
        }
    }
}

template<unsigned STAGE_MODE>
// Public FFN entry used by TransformerLayer.
// Default mainline is still the tile/window core; direct SRAM reads remain only
// as a compatibility fallback inside the worker when caller-fed payloads are not ready.
static inline void FFNLayer0(
    u32_t* sram,
    const FfnCfg& cfg,
    u32_t x_in_base_word,
    const FfnScratch& sc,
    u32_t param_base_word,
    u32_t layer_id = (u32_t)0,
    const u32_t* topfed_x_words = 0,
    const u32_t* topfed_w1_weight_words = 0,
    u32_t topfed_w1_weight_words_valid = 0,
    const u32_t* topfed_w2_input_words = 0,
    u32_t topfed_w2_input_words_valid = 0,
    const u32_t* topfed_w2_weight_words = 0,
    u32_t topfed_w2_weight_words_valid = 0,
    const u32_t* topfed_w2_bias_words = 0,
    u32_t topfed_w2_bias_words_valid = 0,
    u32_t fallback_policy_flags = (u32_t)FFN_POLICY_NONE,
    u32_t* fallback_policy_reject_flag = 0,
    u32_t* fallback_legacy_touch_counter = 0,
    u32_t topfed_x_words_valid_override = 0,
    const u32_t* topfed_w1_bias_words = 0,
    u32_t topfed_w1_bias_words_valid = 0,
    u32_t* fallback_policy_reject_stage = 0
) {
    // Mainline migration note:
    // Default FFN entry now runs through the Top-managed tile/window core.
    // Direct core remains as legacy helper only and is no longer the main path.
    FFNLayer0CoreWindow<STAGE_MODE, u32_t*>(
        sram,
        cfg,
        x_in_base_word,
        sc,
        param_base_word,
        layer_id,
        topfed_x_words,
        topfed_w1_weight_words,
        topfed_w1_weight_words_valid,
        topfed_w2_input_words,
        topfed_w2_input_words_valid,
        topfed_w2_weight_words,
        topfed_w2_weight_words_valid,
        topfed_w2_bias_words,
        topfed_w2_bias_words_valid,
        fallback_policy_flags,
        fallback_policy_reject_flag,
        fallback_legacy_touch_counter,
        topfed_x_words_valid_override,
        topfed_w1_bias_words,
        topfed_w1_bias_words_valid,
        fallback_policy_reject_stage
    );
}

// P00-011AO: first deep FFN boundary bridge.
// Active chain uses this array-shaped entry as the FFN boundary call edge.
// This bridge keeps the caller-visible ownership seam explicit for Catapult-facing
// compile-prep while reusing the same accepted FFN core semantics.
template<unsigned STAGE_MODE, uint32_t SRAM_WORDS>
static inline void FFNLayer0TopManagedWindowBridge(
    u32_t (&sram_window)[SRAM_WORDS],
    const FfnCfg& cfg,
    u32_t x_in_base_word,
    const FfnScratch& sc,
    u32_t param_base_word,
    u32_t layer_id = (u32_t)0,
    const u32_t* topfed_x_words = 0,
    const u32_t* topfed_w1_weight_words = 0,
    u32_t topfed_w1_weight_words_valid = 0,
    const u32_t* topfed_w2_input_words = 0,
    u32_t topfed_w2_input_words_valid = 0,
    const u32_t* topfed_w2_weight_words = 0,
    u32_t topfed_w2_weight_words_valid = 0,
    const u32_t* topfed_w2_bias_words = 0,
    u32_t topfed_w2_bias_words_valid = 0,
    u32_t fallback_policy_flags = (u32_t)FFN_POLICY_NONE,
    u32_t* fallback_policy_reject_flag = 0,
    u32_t* fallback_legacy_touch_counter = 0,
    u32_t topfed_x_words_valid_override = 0,
    const u32_t* topfed_w1_bias_words = 0,
    u32_t topfed_w1_bias_words_valid = 0,
    u32_t* fallback_policy_reject_stage = 0
) {
    FFNLayer0CoreWindow<STAGE_MODE, u32_t (&)[SRAM_WORDS]>(
        sram_window,
        cfg,
        x_in_base_word,
        sc,
        param_base_word,
        layer_id,
        topfed_x_words,
        topfed_w1_weight_words,
        topfed_w1_weight_words_valid,
        topfed_w2_input_words,
        topfed_w2_input_words_valid,
        topfed_w2_weight_words,
        topfed_w2_weight_words_valid,
        topfed_w2_bias_words,
        topfed_w2_bias_words_valid,
        fallback_policy_flags,
        fallback_policy_reject_flag,
        fallback_legacy_touch_counter,
        topfed_x_words_valid_override,
        topfed_w1_bias_words,
        topfed_w1_bias_words_valid,
        fallback_policy_reject_stage
    );
}

} // namespace aecct
