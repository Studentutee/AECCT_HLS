#pragma once
// Tiny HLS-oriented leaf kernel for live ternary QKV row materialization.

#include <cstdint>

#include "AecctTypes.h"
#include "TernaryLinearLive.h"
#include "gen/WeightStreamOrder.h"

namespace aecct {

static inline bool ternary_live_qkv_materialize_row_kernel_impl(
    u32_t* sram,
    u32_t param_base_word,
    QuantLinearMatrixId matrix_id,
    u32_t x_row_base_word,
    u32_t out_row_base_word,
    u32_t out_act_q_row_base_word,
    u32_t& out_inv_sw_bits
) {
    const QuantLinearMeta meta = ternary_linear_live_meta(matrix_id);
    if (meta.matrix_id != (uint32_t)matrix_id) {
        return false;
    }
    if (meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
        return false;
    }
    if (meta.rows == 0u || meta.cols == 0u) {
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
    if (meta.last_word_valid_count == 0u || meta.last_word_valid_count > 16u) {
        return false;
    }
    if (meta.last_word_valid_count != ternary_last_word_valid_count(meta.num_weights)) {
        return false;
    }

    if (!ternary_linear_live_read_inv_sw_bits(sram, param_base_word, matrix_id, out_inv_sw_bits)) {
        return false;
    }

    const uint32_t out_base = (uint32_t)out_row_base_word.to_uint();
    const uint32_t out_act_q_base = (uint32_t)out_act_q_row_base_word.to_uint();
    for (uint32_t out = 0u; out < meta.rows; ++out) {
        u32_t q_bits = (u32_t)0u;
        u32_t inv_sw_bits = (u32_t)0u;
        if (!ternary_linear_live_compute_q_elem(
                sram,
                param_base_word,
                matrix_id,
                x_row_base_word,
                out,
                q_bits,
                inv_sw_bits)) {
            return false;
        }
        if (inv_sw_bits != out_inv_sw_bits) {
            return false;
        }
        sram[out_base + out] = q_bits;
        sram[out_act_q_base + out] = q_bits;
    }
    return true;
}

static inline bool ternary_live_l0_wq_materialize_row_kernel(
    u32_t* sram,
    u32_t param_base_word,
    u32_t x_row_base_word,
    u32_t out_row_base_word,
    u32_t out_act_q_row_base_word,
    u32_t& out_inv_sw_bits
) {
    return ternary_live_qkv_materialize_row_kernel_impl(
        sram,
        param_base_word,
        QLM_L0_WQ,
        x_row_base_word,
        out_row_base_word,
        out_act_q_row_base_word,
        out_inv_sw_bits);
}

static inline bool ternary_live_l0_wk_materialize_row_kernel(
    u32_t* sram,
    u32_t param_base_word,
    u32_t x_row_base_word,
    u32_t out_row_base_word,
    u32_t out_act_q_row_base_word,
    u32_t& out_inv_sw_bits
) {
    return ternary_live_qkv_materialize_row_kernel_impl(
        sram,
        param_base_word,
        QLM_L0_WK,
        x_row_base_word,
        out_row_base_word,
        out_act_q_row_base_word,
        out_inv_sw_bits);
}

static inline bool ternary_live_l0_wv_materialize_row_kernel(
    u32_t* sram,
    u32_t param_base_word,
    u32_t x_row_base_word,
    u32_t out_row_base_word,
    u32_t out_act_q_row_base_word,
    u32_t& out_inv_sw_bits
) {
    return ternary_live_qkv_materialize_row_kernel_impl(
        sram,
        param_base_word,
        QLM_L0_WV,
        x_row_base_word,
        out_row_base_word,
        out_act_q_row_base_word,
        out_inv_sw_bits);
}

} // namespace aecct
