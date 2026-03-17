#pragma once
// Tiny HLS-oriented leaf kernel for live ternary QKV row materialization.

#include <cstdint>

#include "AecctTypes.h"
#include "TernaryLiveQkvLeafKernelShapeConfig.h"
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
    uint32_t expected_num_weights = 0u;
    uint32_t expected_payload_words = 0u;
    uint32_t expected_last_word_valid_count = 0u;
    if (matrix_id == QLM_L0_WQ) {
        expected_num_weights = kQkvCtExpectedL0WqNumWeights;
        expected_payload_words = kQkvCtExpectedL0WqPayloadWords;
        expected_last_word_valid_count = kQkvCtExpectedL0WqLastWordValidCount;
    } else if (matrix_id == QLM_L0_WK) {
        expected_num_weights = kQkvCtExpectedL0WkNumWeights;
        expected_payload_words = kQkvCtExpectedL0WkPayloadWords;
        expected_last_word_valid_count = kQkvCtExpectedL0WkLastWordValidCount;
    } else if (matrix_id == QLM_L0_WV) {
        expected_num_weights = kQkvCtExpectedL0WvNumWeights;
        expected_payload_words = kQkvCtExpectedL0WvPayloadWords;
        expected_last_word_valid_count = kQkvCtExpectedL0WvLastWordValidCount;
    } else {
        return false;
    }
    if (meta.num_weights != expected_num_weights) {
        return false;
    }
    if (meta.payload_words_2b == 0u) {
        return false;
    }
    if (meta.payload_words_2b != expected_payload_words) {
        return false;
    }
    if (meta.payload_words_2b != ternary_payload_words_2b(meta.num_weights)) {
        return false;
    }
    if (meta.last_word_valid_count == 0u || meta.last_word_valid_count > kQkvCtPackedWordElems) {
        return false;
    }
    if (meta.last_word_valid_count != expected_last_word_valid_count) {
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

static inline bool ternary_live_l0_wq_materialize_row_kernel_split(
    const u32_t x_row[kTernaryLiveL0WqCols],
    const u32_t payload_words[kTernaryLiveL0WqPayloadWords],
    u32_t inv_sw_bits,
    u32_t out_row[kTernaryLiveL0WqRows],
    u32_t out_act_q_row[kTernaryLiveL0WqRows],
    u32_t& out_inv_sw_bits
) {
    const QuantLinearMeta meta = ternary_linear_live_l0_wq_meta();
    // Runtime metadata validates against compile-time SSOT for this supported build.
    if (meta.matrix_id != (uint32_t)QLM_L0_WQ) {
        return false;
    }
    if (meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
        return false;
    }
    if (meta.rows != kQkvCtSupportedL0WqRows || meta.cols != kQkvCtSupportedL0WqCols) {
        return false;
    }
    if (meta.num_weights != (meta.rows * meta.cols)) {
        return false;
    }
    if (meta.num_weights != kQkvCtExpectedL0WqNumWeights) {
        return false;
    }
    if (meta.payload_words_2b != kQkvCtExpectedL0WqPayloadWords) {
        return false;
    }
    if (meta.payload_words_2b != ternary_payload_words_2b(meta.num_weights)) {
        return false;
    }
    if (meta.last_word_valid_count == 0u || meta.last_word_valid_count > kQkvCtPackedWordElems) {
        return false;
    }
    if (meta.last_word_valid_count != kQkvCtExpectedL0WqLastWordValidCount) {
        return false;
    }
    if (meta.last_word_valid_count != ternary_last_word_valid_count(meta.num_weights)) {
        return false;
    }

    const fp32_t inv_sw_fp = fp32_from_bits(inv_sw_bits);
    const quant_acc_t inv_sw = inv_sw_fp.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
    if (inv_sw == quant_acc_t(0)) {
        return false;
    }

    out_inv_sw_bits = inv_sw_bits;
    for (uint32_t out = 0u; out < kTernaryLiveL0WqRows; ++out) {
        quant_acc_t acc = 0;
        const uint32_t row_base = out * kTernaryLiveL0WqCols;
        for (uint32_t in = 0u; in < kTernaryLiveL0WqCols; ++in) {
            const uint32_t elem_idx = row_base + in;
            const uint32_t word_idx = (elem_idx >> 4);
            const uint32_t slot = (elem_idx & 15u);
            if (word_idx >= kTernaryLiveL0WqPayloadWords) {
                return false;
            }
            const uint32_t packed = (uint32_t)payload_words[word_idx].to_uint();
            const uint32_t code = (packed >> (slot * 2u)) & 0x3u;

            quant_w_t w = 0;
            if (code == (uint32_t)TERNARY_CODE_ZERO) {
                w = quant_w_t(0);
            } else if (code == (uint32_t)TERNARY_CODE_POS) {
                w = quant_w_t(1);
            } else if (code == (uint32_t)TERNARY_CODE_NEG) {
                w = quant_w_t(-1);
            } else {
                return false;
            }

            const quant_act_t x = quant_act_from_bits(x_row[in]);
            acc += quant_acc_t(x) * quant_acc_t(w);
        }
        const u32_t q_bits = quant_bits_from_acc(acc / inv_sw);
        out_row[out] = q_bits;
        out_act_q_row[out] = q_bits;
    }
    return true;
}

static inline bool ternary_live_l0_wk_materialize_row_kernel_split(
    const u32_t x_row[kTernaryLiveL0WkCols],
    const u32_t payload_words[kTernaryLiveL0WkPayloadWords],
    u32_t inv_sw_bits,
    u32_t out_row[kTernaryLiveL0WkRows],
    u32_t out_act_q_row[kTernaryLiveL0WkRows],
    u32_t& out_inv_sw_bits
) {
    const QuantLinearMeta meta = ternary_linear_live_l0_wk_meta();
    // Runtime metadata validates against compile-time SSOT for this supported build.
    if (meta.matrix_id != (uint32_t)QLM_L0_WK) {
        return false;
    }
    if (meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
        return false;
    }
    if (meta.rows != kQkvCtSupportedL0WkRows || meta.cols != kQkvCtSupportedL0WkCols) {
        return false;
    }
    if (meta.num_weights != (meta.rows * meta.cols)) {
        return false;
    }
    if (meta.num_weights != kQkvCtExpectedL0WkNumWeights) {
        return false;
    }
    if (meta.payload_words_2b != kQkvCtExpectedL0WkPayloadWords) {
        return false;
    }
    if (meta.payload_words_2b != ternary_payload_words_2b(meta.num_weights)) {
        return false;
    }
    if (meta.last_word_valid_count == 0u || meta.last_word_valid_count > kQkvCtPackedWordElems) {
        return false;
    }
    if (meta.last_word_valid_count != kQkvCtExpectedL0WkLastWordValidCount) {
        return false;
    }
    if (meta.last_word_valid_count != ternary_last_word_valid_count(meta.num_weights)) {
        return false;
    }

    const fp32_t inv_sw_fp = fp32_from_bits(inv_sw_bits);
    const quant_acc_t inv_sw = inv_sw_fp.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
    if (inv_sw == quant_acc_t(0)) {
        return false;
    }

    out_inv_sw_bits = inv_sw_bits;
    for (uint32_t out = 0u; out < kTernaryLiveL0WkRows; ++out) {
        quant_acc_t acc = 0;
        const uint32_t row_base = out * kTernaryLiveL0WkCols;
        for (uint32_t in = 0u; in < kTernaryLiveL0WkCols; ++in) {
            const uint32_t elem_idx = row_base + in;
            const uint32_t word_idx = (elem_idx >> 4);
            const uint32_t slot = (elem_idx & 15u);
            if (word_idx >= kTernaryLiveL0WkPayloadWords) {
                return false;
            }
            const uint32_t packed = (uint32_t)payload_words[word_idx].to_uint();
            const uint32_t code = (packed >> (slot * 2u)) & 0x3u;

            quant_w_t w = 0;
            if (code == (uint32_t)TERNARY_CODE_ZERO) {
                w = quant_w_t(0);
            } else if (code == (uint32_t)TERNARY_CODE_POS) {
                w = quant_w_t(1);
            } else if (code == (uint32_t)TERNARY_CODE_NEG) {
                w = quant_w_t(-1);
            } else {
                return false;
            }

            const quant_act_t x = quant_act_from_bits(x_row[in]);
            acc += quant_acc_t(x) * quant_acc_t(w);
        }
        const u32_t q_bits = quant_bits_from_acc(acc / inv_sw);
        out_row[out] = q_bits;
        out_act_q_row[out] = q_bits;
    }
    return true;
}

static inline bool ternary_live_l0_wv_materialize_row_kernel_split(
    const u32_t x_row[kTernaryLiveL0WvCols],
    const u32_t payload_words[kTernaryLiveL0WvPayloadWords],
    u32_t inv_sw_bits,
    u32_t out_row[kTernaryLiveL0WvRows],
    u32_t out_act_q_row[kTernaryLiveL0WvRows],
    u32_t& out_inv_sw_bits
) {
    const QuantLinearMeta meta = ternary_linear_live_l0_wv_meta();
    // Runtime metadata validates against compile-time SSOT for this supported build.
    if (meta.matrix_id != (uint32_t)QLM_L0_WV) {
        return false;
    }
    if (meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
        return false;
    }
    if (meta.rows != kQkvCtSupportedL0WvRows || meta.cols != kQkvCtSupportedL0WvCols) {
        return false;
    }
    if (meta.num_weights != (meta.rows * meta.cols)) {
        return false;
    }
    if (meta.num_weights != kQkvCtExpectedL0WvNumWeights) {
        return false;
    }
    if (meta.payload_words_2b != kQkvCtExpectedL0WvPayloadWords) {
        return false;
    }
    if (meta.payload_words_2b != ternary_payload_words_2b(meta.num_weights)) {
        return false;
    }
    if (meta.last_word_valid_count == 0u || meta.last_word_valid_count > kQkvCtPackedWordElems) {
        return false;
    }
    if (meta.last_word_valid_count != kQkvCtExpectedL0WvLastWordValidCount) {
        return false;
    }
    if (meta.last_word_valid_count != ternary_last_word_valid_count(meta.num_weights)) {
        return false;
    }

    const fp32_t inv_sw_fp = fp32_from_bits(inv_sw_bits);
    const quant_acc_t inv_sw = inv_sw_fp.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
    if (inv_sw == quant_acc_t(0)) {
        return false;
    }

    out_inv_sw_bits = inv_sw_bits;
    for (uint32_t out = 0u; out < kTernaryLiveL0WvRows; ++out) {
        quant_acc_t acc = 0;
        const uint32_t row_base = out * kTernaryLiveL0WvCols;
        for (uint32_t in = 0u; in < kTernaryLiveL0WvCols; ++in) {
            const uint32_t elem_idx = row_base + in;
            const uint32_t word_idx = (elem_idx >> 4);
            const uint32_t slot = (elem_idx & 15u);
            if (word_idx >= kTernaryLiveL0WvPayloadWords) {
                return false;
            }
            const uint32_t packed = (uint32_t)payload_words[word_idx].to_uint();
            const uint32_t code = (packed >> (slot * 2u)) & 0x3u;

            quant_w_t w = 0;
            if (code == (uint32_t)TERNARY_CODE_ZERO) {
                w = quant_w_t(0);
            } else if (code == (uint32_t)TERNARY_CODE_POS) {
                w = quant_w_t(1);
            } else if (code == (uint32_t)TERNARY_CODE_NEG) {
                w = quant_w_t(-1);
            } else {
                return false;
            }

            const quant_act_t x = quant_act_from_bits(x_row[in]);
            acc += quant_acc_t(x) * quant_acc_t(w);
        }
        const u32_t q_bits = quant_bits_from_acc(acc / inv_sw);
        out_row[out] = q_bits;
        out_act_q_row[out] = q_bits;
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
