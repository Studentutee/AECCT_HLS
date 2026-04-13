#pragma once
// Minimal design-side live ternary consumer helper for QLM_L0_WQ.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "QuantDesc.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"

namespace aecct {

static inline QuantLinearMeta ternary_linear_live_meta(QuantLinearMatrixId matrix_id) {
    return kQuantLinearMeta[(uint32_t)matrix_id];
}

static inline uint32_t ternary_linear_live_fp16_branch_param_base_w() {
    return storage_words_to_legacy_words_ceil(
        sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_BASE_WORD16);
}

static inline bool ternary_linear_live_uses_fp16_branch_params(const u32_t& param_base_word) {
    return (uint32_t)param_base_word.to_uint() == ternary_linear_live_fp16_branch_param_base_w();
}

template<typename SramView>
static inline u16_t ternary_linear_live_read_word16_from_u32_sram(
    const SramView& sram,
    uint32_t base_word32,
    uint32_t word16_offset
) {
    const uint32_t addr_word32 = base_word32 + (word16_offset >> 1);
    const uint32_t lane_idx = word16_offset & 1u;
    return unpack_fp16_lane(sram[addr_word32], lane_idx);
}

template<typename SramView>
static inline bool ternary_linear_live_read_inv_sw_bits_view(
    const SramView& sram,
    u32_t param_base_word,
    QuantLinearMatrixId matrix_id,
    u32_t& out_inv_sw_bits
) {
    const QuantLinearMeta meta = ternary_linear_live_meta(matrix_id);
    const uint32_t param_base = (uint32_t)param_base_word.to_uint();
    if (!ternary_linear_live_uses_fp16_branch_params(param_base_word)) {
        const ParamMeta inv_meta = kParamMeta[meta.inv_sw_param_id];
        if (inv_meta.len_w == 0u) {
            return false;
        }
        out_inv_sw_bits = sram[param_base + inv_meta.offset_w];
        return true;
    }

    WeightId inv_wid;
    if (!quant_linear_matrix_id_to_inv_sw_weight_id((uint32_t)matrix_id, inv_wid)) {
        return false;
    }
    const Fp16BranchStorageDesc desc = fp16_branch_weight_storage_desc(inv_wid);
    if (desc.words16 == 0u) {
        return false;
    }
    const u16_t inv_lane = ternary_linear_live_read_word16_from_u32_sram(
        sram, param_base, desc.offset_words16);
    out_inv_sw_bits = fp32_bits_from_fp16_lane(inv_lane);
    return true;
}

template<typename SramView>
static inline bool ternary_linear_live_load_payload_words_view(
    const SramView& sram,
    u32_t param_base_word,
    QuantLinearMatrixId matrix_id,
    u32_t* out_payload_words,
    uint32_t payload_words_capacity
) {
    const QuantLinearMeta meta = ternary_linear_live_meta(matrix_id);
    if (payload_words_capacity < meta.payload_words_2b) {
        return false;
    }
    const uint32_t param_base = (uint32_t)param_base_word.to_uint();
    if (!ternary_linear_live_uses_fp16_branch_params(param_base_word)) {
        const ParamMeta payload_meta = kParamMeta[meta.weight_param_id];
        if (payload_meta.len_w < meta.payload_words_2b) {
            return false;
        }
        for (uint32_t i = 0u; i < meta.payload_words_2b; ++i) {
            out_payload_words[i] = sram[param_base + payload_meta.offset_w + i];
        }
        return true;
    }

    WeightId payload_wid;
    if (!quant_linear_matrix_id_to_weight_id((uint32_t)matrix_id, payload_wid)) {
        return false;
    }
    const Fp16BranchStorageDesc desc = fp16_branch_weight_storage_desc(payload_wid);
    for (uint32_t i = 0u; i < meta.payload_words_2b; ++i) {
        const uint32_t lo_idx = desc.offset_words16 + i * 2u;
        const u16_t lo = ternary_linear_live_read_word16_from_u32_sram(sram, param_base, lo_idx);
        u16_t hi = (u16_t)0u;
        if ((i * 2u + 1u) < desc.words16) {
            hi = ternary_linear_live_read_word16_from_u32_sram(sram, param_base, lo_idx + 1u);
        }
        out_payload_words[i] = pack_fp16_lanes(lo, hi);
    }
    return true;
}

static inline bool ternary_linear_live_read_inv_sw_bits(
    const u32_t* sram,
    u32_t param_base_word,
    QuantLinearMatrixId matrix_id,
    u32_t& out_inv_sw_bits
) {
    return ternary_linear_live_read_inv_sw_bits_view(
        sram, param_base_word, matrix_id, out_inv_sw_bits);
}

template<typename SramView>
static inline bool ternary_linear_live_decode_code_view(
    const SramView& sram,
    u32_t param_base_word,
    QuantLinearMatrixId matrix_id,
    uint32_t out_idx,
    uint32_t in_idx,
    uint32_t& out_code
) {
    const QuantLinearMeta meta = ternary_linear_live_meta(matrix_id);
    if (out_idx >= meta.rows || in_idx >= meta.cols) {
        return false;
    }

    const uint32_t elem_idx = out_idx * meta.cols + in_idx;
    if (elem_idx >= meta.num_weights) {
        return false;
    }

    const uint32_t param_base = (uint32_t)param_base_word.to_uint();
    if (!ternary_linear_live_uses_fp16_branch_params(param_base_word)) {
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
        const uint32_t payload_base = param_base + kParamMeta[meta.weight_param_id].offset_w;
        const uint32_t word = (uint32_t)sram[payload_base + word_idx].to_uint();
        out_code = (word >> (slot * 2u)) & 0x3u;
        return true;
    }

    WeightId payload_wid;
    if (!quant_linear_matrix_id_to_weight_id((uint32_t)matrix_id, payload_wid)) {
        return false;
    }
    const Fp16BranchStorageDesc desc = fp16_branch_weight_storage_desc(payload_wid);
    const uint32_t word16_idx = (elem_idx >> 3);
    if (word16_idx >= desc.words16) {
        return false;
    }
    const uint32_t slot = (elem_idx & 7u);
    uint32_t valid_in_word16 = meta.num_weights - word16_idx * 8u;
    if (valid_in_word16 > 8u) {
        valid_in_word16 = 8u;
    }
    if (valid_in_word16 == 0u || slot >= valid_in_word16) {
        return false;
    }
    const u16_t word16 = ternary_linear_live_read_word16_from_u32_sram(
        sram, param_base, desc.offset_words16 + word16_idx);
    out_code = (((uint32_t)word16.to_uint()) >> (slot * 2u)) & 0x3u;
    return true;
}

static inline bool ternary_linear_live_decode_code(
    const u32_t* sram,
    u32_t param_base_word,
    QuantLinearMatrixId matrix_id,
    uint32_t out_idx,
    uint32_t in_idx,
    uint32_t& out_code
) {
    return ternary_linear_live_decode_code_view(
        sram, param_base_word, matrix_id, out_idx, in_idx, out_code);
}

static inline bool ternary_linear_live_decode_weight(
    const u32_t* sram,
    u32_t param_base_word,
    QuantLinearMatrixId matrix_id,
    uint32_t out_idx,
    uint32_t in_idx,
    quant_w_t& out_w
) {
    uint32_t code = 0u;
    if (!ternary_linear_live_decode_code(sram, param_base_word, matrix_id, out_idx, in_idx, code)) {
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

static inline bool ternary_linear_live_decode_weight_i8(
    const u32_t* sram,
    u32_t param_base_word,
    QuantLinearMatrixId matrix_id,
    uint32_t out_idx,
    uint32_t in_idx,
    quant_w_i8_t& out_w
) {
    uint32_t code = 0u;
    if (!ternary_linear_live_decode_code(sram, param_base_word, matrix_id, out_idx, in_idx, code)) {
        return false;
    }
    return quant_decode_ternary_weight_i8(code, out_w);
}

static inline bool ternary_linear_live_compute_q_elem_i8_acc16(
    const u32_t* sram,
    u32_t param_base_word,
    QuantLinearMatrixId matrix_id,
    u32_t x_row_base_word,
    uint32_t out_idx,
    u32_t& out_q_bits,
    u32_t& out_inv_sw_bits
) {
    const QuantLinearMeta meta = ternary_linear_live_meta(matrix_id);
    if (out_idx >= meta.rows) {
        return false;
    }
    if (!ternary_linear_live_read_inv_sw_bits(sram, param_base_word, matrix_id, out_inv_sw_bits)) {
        return false;
    }

    const uint32_t x_base = (uint32_t)x_row_base_word.to_uint();
    quant_acc_i16_t acc = 0;
    TERNARY_LINEAR_LIVE_I8ACC16_COL_LOOP: for (uint32_t in = 0u; in < meta.cols; ++in) {
        quant_w_i8_t w = 0;
        if (!ternary_linear_live_decode_weight_i8(sram, param_base_word, matrix_id, out_idx, in, w)) {
            return false;
        }
        const quant_act_i8_t x = quant_act_i8_from_word(sram[x_base + in]);
        acc = quant_acc_i16_saturating_madd(acc, x, w);
    }

    out_q_bits = quant_word_from_acc_i16(acc);
    return true;
}

static inline bool ternary_linear_live_compute_q_elem(
    const u32_t* sram,
    u32_t param_base_word,
    QuantLinearMatrixId matrix_id,
    u32_t x_row_base_word,
    uint32_t out_idx,
    u32_t& out_q_bits,
    u32_t& out_inv_sw_bits
) {
    return ternary_linear_live_compute_q_elem_i8_acc16(
        sram, param_base_word, matrix_id, x_row_base_word, out_idx, out_q_bits, out_inv_sw_bits);
}

static inline QuantLinearMeta ternary_linear_live_l0_wq_meta() {
    return ternary_linear_live_meta(QLM_L0_WQ);
}

static inline bool ternary_linear_live_l0_wq_read_inv_sw_bits(
    const u32_t* sram,
    u32_t param_base_word,
    u32_t& out_inv_sw_bits
) {
    return ternary_linear_live_read_inv_sw_bits(sram, param_base_word, QLM_L0_WQ, out_inv_sw_bits);
}

static inline bool ternary_linear_live_l0_wq_decode_code(
    const u32_t* sram,
    u32_t param_base_word,
    uint32_t out_idx,
    uint32_t in_idx,
    uint32_t& out_code
) {
    return ternary_linear_live_decode_code(sram, param_base_word, QLM_L0_WQ, out_idx, in_idx, out_code);
}

static inline bool ternary_linear_live_l0_wq_decode_weight(
    const u32_t* sram,
    u32_t param_base_word,
    uint32_t out_idx,
    uint32_t in_idx,
    quant_w_t& out_w
) {
    return ternary_linear_live_decode_weight(sram, param_base_word, QLM_L0_WQ, out_idx, in_idx, out_w);
}

static inline bool ternary_linear_live_l0_wq_compute_q_elem(
    const u32_t* sram,
    u32_t param_base_word,
    u32_t x_row_base_word,
    uint32_t out_idx,
    u32_t& out_q_bits,
    u32_t& out_inv_sw_bits
) {
    return ternary_linear_live_compute_q_elem(
        sram, param_base_word, QLM_L0_WQ, x_row_base_word, out_idx, out_q_bits, out_inv_sw_bits);
}

static inline QuantLinearMeta ternary_linear_live_l0_wk_meta() {
    return ternary_linear_live_meta(QLM_L0_WK);
}

static inline bool ternary_linear_live_l0_wk_read_inv_sw_bits(
    const u32_t* sram,
    u32_t param_base_word,
    u32_t& out_inv_sw_bits
) {
    return ternary_linear_live_read_inv_sw_bits(sram, param_base_word, QLM_L0_WK, out_inv_sw_bits);
}

static inline bool ternary_linear_live_l0_wk_decode_code(
    const u32_t* sram,
    u32_t param_base_word,
    uint32_t out_idx,
    uint32_t in_idx,
    uint32_t& out_code
) {
    return ternary_linear_live_decode_code(sram, param_base_word, QLM_L0_WK, out_idx, in_idx, out_code);
}

static inline bool ternary_linear_live_l0_wk_decode_weight(
    const u32_t* sram,
    u32_t param_base_word,
    uint32_t out_idx,
    uint32_t in_idx,
    quant_w_t& out_w
) {
    return ternary_linear_live_decode_weight(sram, param_base_word, QLM_L0_WK, out_idx, in_idx, out_w);
}

static inline bool ternary_linear_live_l0_wk_compute_q_elem(
    const u32_t* sram,
    u32_t param_base_word,
    u32_t x_row_base_word,
    uint32_t out_idx,
    u32_t& out_q_bits,
    u32_t& out_inv_sw_bits
) {
    return ternary_linear_live_compute_q_elem(
        sram, param_base_word, QLM_L0_WK, x_row_base_word, out_idx, out_q_bits, out_inv_sw_bits);
}

static inline QuantLinearMeta ternary_linear_live_l0_wv_meta() {
    return ternary_linear_live_meta(QLM_L0_WV);
}

static inline bool ternary_linear_live_l0_wv_read_inv_sw_bits(
    const u32_t* sram,
    u32_t param_base_word,
    u32_t& out_inv_sw_bits
) {
    return ternary_linear_live_read_inv_sw_bits(sram, param_base_word, QLM_L0_WV, out_inv_sw_bits);
}

static inline bool ternary_linear_live_l0_wv_decode_code(
    const u32_t* sram,
    u32_t param_base_word,
    uint32_t out_idx,
    uint32_t in_idx,
    uint32_t& out_code
) {
    return ternary_linear_live_decode_code(sram, param_base_word, QLM_L0_WV, out_idx, in_idx, out_code);
}

static inline bool ternary_linear_live_l0_wv_decode_weight(
    const u32_t* sram,
    u32_t param_base_word,
    uint32_t out_idx,
    uint32_t in_idx,
    quant_w_t& out_w
) {
    return ternary_linear_live_decode_weight(sram, param_base_word, QLM_L0_WV, out_idx, in_idx, out_w);
}

static inline bool ternary_linear_live_l0_wv_compute_q_elem(
    const u32_t* sram,
    u32_t param_base_word,
    u32_t x_row_base_word,
    uint32_t out_idx,
    u32_t& out_q_bits,
    u32_t& out_inv_sw_bits
) {
    return ternary_linear_live_compute_q_elem(
        sram, param_base_word, QLM_L0_WV, x_row_base_word, out_idx, out_q_bits, out_inv_sw_bits);
}

} // namespace aecct
