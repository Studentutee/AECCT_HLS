#pragma once
// Minimal design-side live ternary consumer helper for QLM_L0_WQ.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "QuantDesc.h"
#include "gen/WeightStreamOrder.h"

namespace aecct {

static inline QuantLinearMeta ternary_linear_live_l0_wq_meta() {
    return kQuantLinearMeta[(uint32_t)QLM_L0_WQ];
}

static inline bool ternary_linear_live_l0_wq_read_inv_sw_bits(
    const u32_t* sram,
    u32_t param_base_word,
    u32_t& out_inv_sw_bits
) {
    const QuantLinearMeta meta = ternary_linear_live_l0_wq_meta();
    const uint32_t param_base = (uint32_t)param_base_word.to_uint();
    const ParamMeta inv_meta = kParamMeta[meta.inv_sw_param_id];
    if (inv_meta.len_w == 0u) {
        return false;
    }
    out_inv_sw_bits = sram[param_base + inv_meta.offset_w];
    return true;
}

static inline bool ternary_linear_live_l0_wq_decode_code(
    const u32_t* sram,
    u32_t param_base_word,
    uint32_t out_idx,
    uint32_t in_idx,
    uint32_t& out_code
) {
    const QuantLinearMeta meta = ternary_linear_live_l0_wq_meta();
    if (out_idx >= meta.rows || in_idx >= meta.cols) {
        return false;
    }

    const uint32_t elem_idx = out_idx * meta.cols + in_idx;
    if (elem_idx >= meta.num_weights) {
        return false;
    }

    const uint32_t word_idx = (elem_idx >> 4);
    if (word_idx >= meta.payload_words_2b) {
        return false;
    }

    const uint32_t slot = (elem_idx & 15u);
    const uint32_t valid_in_last_word = meta.last_word_valid_count;
    const uint32_t valid_in_word = ((word_idx + 1u) == meta.payload_words_2b) ? valid_in_last_word : 16u;
    if (valid_in_word == 0u || valid_in_word > 16u || slot >= valid_in_word) {
        return false;
    }

    const uint32_t param_base = (uint32_t)param_base_word.to_uint();
    const uint32_t payload_base = param_base + kParamMeta[meta.weight_param_id].offset_w;
    const uint32_t word = (uint32_t)sram[payload_base + word_idx].to_uint();
    out_code = (word >> (slot * 2u)) & 0x3u;
    return true;
}

static inline bool ternary_linear_live_l0_wq_decode_weight(
    const u32_t* sram,
    u32_t param_base_word,
    uint32_t out_idx,
    uint32_t in_idx,
    quant_w_t& out_w
) {
    uint32_t code = 0u;
    if (!ternary_linear_live_l0_wq_decode_code(sram, param_base_word, out_idx, in_idx, code)) {
        return false;
    }

    if (code == (uint32_t)TERNARY_CODE_ZERO) {
        out_w = quant_w_t(0);
        return true;
    }
    if (code == (uint32_t)TERNARY_CODE_POS) {
        out_w = quant_w_t(1);
        return true;
    }
    if (code == (uint32_t)TERNARY_CODE_NEG) {
        out_w = quant_w_t(-1);
        return true;
    }
    return false;
}

static inline bool ternary_linear_live_l0_wq_compute_q_elem(
    const u32_t* sram,
    u32_t param_base_word,
    u32_t x_row_base_word,
    uint32_t out_idx,
    u32_t& out_q_bits,
    u32_t& out_inv_sw_bits
) {
    const QuantLinearMeta meta = ternary_linear_live_l0_wq_meta();
    if (out_idx >= meta.rows) {
        return false;
    }
    if (!ternary_linear_live_l0_wq_read_inv_sw_bits(sram, param_base_word, out_inv_sw_bits)) {
        return false;
    }

    fp32_t inv_sw_fp = fp32_from_bits(out_inv_sw_bits);
    quant_acc_t inv_sw = inv_sw_fp.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
    if (inv_sw == quant_acc_t(0)) {
        return false;
    }

    const uint32_t x_base = (uint32_t)x_row_base_word.to_uint();
    quant_acc_t acc = 0;
    for (uint32_t in = 0; in < meta.cols; ++in) {
        quant_w_t w = 0;
        if (!ternary_linear_live_l0_wq_decode_weight(sram, param_base_word, out_idx, in, w)) {
            return false;
        }
        quant_act_t x = quant_act_from_bits(sram[x_base + in]);
        acc += quant_acc_t(x) * quant_acc_t(w);
    }

    out_q_bits = quant_bits_from_acc(acc / inv_sw);
    return true;
}

} // namespace aecct
