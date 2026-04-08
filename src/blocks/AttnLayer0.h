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

static inline QuantLinearMatrixId attn_q_matrix_id_for_layer(uint32_t layer_id) {
    return (layer_id == 0u) ? QLM_L0_WQ : QLM_L1_WQ;
}

static inline QuantLinearMatrixId attn_k_matrix_id_for_layer(uint32_t layer_id) {
    return (layer_id == 0u) ? QLM_L0_WK : QLM_L1_WK;
}

static inline QuantLinearMatrixId attn_v_matrix_id_for_layer(uint32_t layer_id) {
    return (layer_id == 0u) ? QLM_L0_WV : QLM_L1_WV;
}

static inline bool attn_qkv_bias_param_id_for_matrix(
    QuantLinearMatrixId matrix_id,
    uint32_t& out_bias_param_id
) {
    switch (matrix_id) {
        case QLM_L0_WQ: out_bias_param_id = 0u; return true;
        case QLM_L0_WK: out_bias_param_id = 1u; return true;
        case QLM_L0_WV: out_bias_param_id = 2u; return true;
        case QLM_L1_WQ: out_bias_param_id = 8u; return true;
        case QLM_L1_WK: out_bias_param_id = 9u; return true;
        case QLM_L1_WV: out_bias_param_id = 10u; return true;
        default: return false;
    }
}

static inline uint32_t attn_qkv_input_sx_slot_for_layer(uint32_t layer_id) {
    return (layer_id == 0u) ? 0u : 4u;
}

static inline bool attn_qkv_input_sx_param_id(uint32_t& out_param_id) {
    return weight_id_to_param_id(QUANT_SX_8, out_param_id);
}

static inline bool attn_src_mask_param_id(uint32_t& out_param_id) {
    return weight_id_to_param_id(SRC_MASK, out_param_id);
}

static inline fp32_t attn_qkv_quantize_int8_symmetric(fp32_t x, fp32_t s_x) {
    fp32_t q = (x * s_x).round();
    if (q > fp32_t(127.0f)) { q = fp32_t(127.0f); }
    if (q < fp32_t(-127.0f)) { q = fp32_t(-127.0f); }
    return q;
}

static inline bool attn_dense_qkv_gate_ok(
    const QuantLinearMeta& meta,
    QuantLinearMatrixId expected_matrix_id,
    uint32_t param_base_word,
    uint32_t token_count,
    uint32_t d_model,
    uint32_t layer_id
) {
    if (param_base_word == 0u || token_count == 0u) {
        return false;
    }
    if (meta.matrix_id != (uint32_t)expected_matrix_id) {
        return false;
    }
    if (meta.rows != d_model || meta.cols != d_model) {
        return false;
    }
    if (meta.num_weights != (meta.rows * meta.cols)) {
        return false;
    }
    const ParamMeta weight_meta = kParamMeta[meta.weight_param_id];
    if (weight_meta.len_w < meta.num_weights) {
        return false;
    }
    uint32_t bias_param_id = 0u;
    if (!attn_qkv_bias_param_id_for_matrix(expected_matrix_id, bias_param_id)) {
        return false;
    }
    const ParamMeta bias_meta = kParamMeta[bias_param_id];
    if (bias_meta.len_w < meta.rows) {
        return false;
    }
    const ParamMeta inv_sw_meta = kParamMeta[meta.inv_sw_param_id];
    if (inv_sw_meta.len_w == 0u) {
        return false;
    }
    uint32_t sx_param_id = 0u;
    if (!attn_qkv_input_sx_param_id(sx_param_id)) {
        return false;
    }
    const ParamMeta sx_meta = kParamMeta[sx_param_id];
    const uint32_t sx_slot = attn_qkv_input_sx_slot_for_layer(layer_id);
    if (sx_meta.len_w <= sx_slot) {
        return false;
    }
    return true;
}

template<typename SramView>
static inline bool attn_materialize_qkv_dense_fp32_contract(
    SramView& sram,
    uint32_t param_base_word,
    const QuantLinearMeta& meta,
    QuantLinearMatrixId matrix_id,
    uint32_t layer_id,
    uint32_t token_count,
    uint32_t d_model,
    uint32_t x_in_base,
    uint32_t out_base,
    uint32_t out_act_q_base,
    u32_t& out_inv_sw_bits
) {
    if (!attn_dense_qkv_gate_ok(
            meta,
            matrix_id,
            param_base_word,
            token_count,
            d_model,
            layer_id)) {
        return false;
    }

    const ParamMeta weight_meta = kParamMeta[meta.weight_param_id];
    const ParamMeta inv_sw_meta = kParamMeta[meta.inv_sw_param_id];
    uint32_t bias_param_id = 0u;
    if (!attn_qkv_bias_param_id_for_matrix(matrix_id, bias_param_id)) {
        return false;
    }
    const ParamMeta bias_meta = kParamMeta[bias_param_id];
    uint32_t sx_param_id = 0u;
    if (!attn_qkv_input_sx_param_id(sx_param_id)) {
        return false;
    }
    const ParamMeta sx_meta = kParamMeta[sx_param_id];
    const uint32_t sx_slot = attn_qkv_input_sx_slot_for_layer(layer_id);

    const uint32_t weight_base = param_base_word + weight_meta.offset_w;
    const uint32_t bias_base = param_base_word + bias_meta.offset_w;
    const uint32_t sx_addr = param_base_word + sx_meta.offset_w + sx_slot;
    const uint32_t inv_sw_addr = param_base_word + inv_sw_meta.offset_w;
    const u32_t s_x_bits = sram[sx_addr];
    const u32_t inv_sw_bits = sram[inv_sw_addr];
    if ((uint32_t)s_x_bits.to_uint() == 0u || (uint32_t)inv_sw_bits.to_uint() == 0u) {
        return false;
    }

    const fp32_t s_x = fp32_from_bits(s_x_bits);
    const fp32_t inv_sw = fp32_from_bits(inv_sw_bits);
    const fp32_t inv_scale = inv_sw / s_x;

    ATTN_DENSE_QKV_TOKEN_LOOP: for (uint32_t t = 0u; t < token_count; ++t) {
        const uint32_t x_row_base = x_in_base + t * d_model;
        const uint32_t out_row_base = out_base + t * d_model;
        const uint32_t out_act_q_row_base = out_act_q_base + t * d_model;
        ATTN_DENSE_QKV_OUT_LOOP: for (uint32_t out = 0u; out < meta.rows; ++out) {
            fp32_t acc = fp32_from_bits(sram[bias_base + out]);
            const uint32_t w_row_base = weight_base + out * meta.cols;
            ATTN_DENSE_QKV_IN_LOOP: for (uint32_t in = 0u; in < meta.cols; ++in) {
                const fp32_t x_fp = fp32_from_bits(sram[x_row_base + in]);
                const fp32_t q_fp = attn_qkv_quantize_int8_symmetric(x_fp, s_x);
                const fp32_t w_fp = fp32_from_bits(sram[w_row_base + in]);
                acc += q_fp * (w_fp * inv_scale);
            }
            const u32_t out_bits = bits_from_fp32(acc);
            sram[out_row_base + out] = out_bits;
            sram[out_act_q_row_base + out] = out_bits;
        }
    }

    out_inv_sw_bits = inv_sw_bits;
    return true;
}

template<typename SramView>
static inline bool attn_src_mask_bit_row_major(
    const SramView& sram,
    uint32_t src_mask_base,
    uint32_t src_mask_rows,
    uint32_t src_mask_cols,
    uint32_t token_idx,
    uint32_t key_idx
) {
    if (token_idx >= src_mask_rows || key_idx >= src_mask_cols) {
        return false;
    }
    const uint32_t bit_idx = token_idx * src_mask_cols + key_idx;
    const uint32_t word_idx = bit_idx >> 5;
    const uint32_t shift = bit_idx & 31u;
    const uint32_t packed = (uint32_t)sram[src_mask_base + word_idx].to_uint();
    return ((packed >> shift) & 1u) != 0u;
}

template<typename SramView>
static inline bool attn_score_masked_by_src_contract(
    const SramView& sram,
    bool src_mask_ready,
    uint32_t src_mask_base,
    uint32_t src_mask_rows,
    uint32_t src_mask_cols,
    uint32_t n_heads,
    uint32_t head_idx,
    uint32_t token_idx,
    uint32_t key_idx
) {
    if (!src_mask_ready) {
        return false;
    }
    const bool src_mask_bit = attn_src_mask_bit_row_major(
        sram,
        src_mask_base,
        src_mask_rows,
        src_mask_cols,
        token_idx,
        key_idx);
    const bool is_var_i = (token_idx < (uint32_t)CODE_N);
    const bool is_var_j = (key_idx < (uint32_t)CODE_N);
    uint32_t one_ring_head_count = (n_heads >> 1);
    if (one_ring_head_count == 0u) {
        one_ring_head_count = 1u;
    }
    const bool use_one_ring = (head_idx < one_ring_head_count);
    if (use_one_ring) {
        if (is_var_i == is_var_j) {
            return true;
        }
        return src_mask_bit;
    }
    if (is_var_i == is_var_j) {
        return src_mask_bit;
    }
    return true;
}

static inline fp32_t attn_inv_sqrt_d_head_fp32(uint32_t d_head) {
    u32_t bits = (u32_t)0x3F800000u;
    if (d_head == 2u) { bits = (u32_t)0x3F3504F3u; }
    else if (d_head == 4u) { bits = (u32_t)0x3F000000u; }
    else if (d_head == 8u) { bits = (u32_t)0x3EB504F3u; }
    else if (d_head == 16u) { bits = (u32_t)0x3E800000u; }
    else if (d_head == 32u) { bits = (u32_t)0x3E3504F3u; }
    else if (d_head == 64u) { bits = (u32_t)0x3E000000u; }
    return fp32_from_bits(bits);
}

static inline bool attn_score_debug_dump_slot_for_probe(
    uint32_t h,
    uint32_t t,
    uint32_t n_heads,
    uint32_t token_count,
    uint32_t tensor_words,
    uint32_t& out_slot
) {
    out_slot = 0u;
    if (h == 0u && t == 0u) {
        out_slot = 0u;
    } else if (h == 0u && t == 16u) {
        out_slot = 1u;
    } else if (h == 4u && t == 35u) {
        out_slot = 2u;
    } else {
        return false;
    }
    if (h >= n_heads || t >= token_count) {
        return false;
    }
    const uint32_t slot_end = (out_slot + 1u) * token_count;
    if (slot_end > tensor_words) {
        return false;
    }
    return true;
}

static inline fp32_t attn_softmax_fp32_exp_lut(fp32_t x) {
    static const uint32_t kAttnSoftmaxExpLutBits[SoftmaxApproxCfg::EXP_LUT_SIZE] = {
        0x3F800000u, 0x3F743B65u, 0x3F690146u, 0x3F5E4B46u, 0x3F541351u, 0x3F4A539Du, 0x3F4106A3u, 0x3F38271Cu,
        0x3F2FB000u, 0x3F279C83u, 0x3F1FE810u, 0x3F188E48u, 0x3F118B02u, 0x3F0ADA42u, 0x3F04783Eu, 0x3EFCC2AEu,
        0x3EF12432u, 0x3EE60E72u, 0x3EDB7B25u, 0x3ED1644Bu, 0x3EC7C42Cu, 0x3EBE9553u, 0x3EB5D28Au, 0x3EAD76DBu,
        0x3EA57D87u, 0x3E9DE20Au, 0x3E96A013u, 0x3E8FB384u, 0x3E891871u, 0x3E82CB1Au, 0x3E798FDBu, 0x3E6E1703u,
        0x3E63252Cu, 0x3E58B421u, 0x3E4EBDF6u, 0x3E453D06u, 0x3E3C2BECu, 0x3E338585u, 0x3E2B44E9u, 0x3E23656Bu,
        0x3E1BE293u, 0x3E14B81Eu, 0x3E0DE1FEu, 0x3E075C51u, 0x3E012365u, 0x3DF66764u, 0x3DEB13B6u, 0x3DE04554u,
        0x3DD5F61Cu, 0x3DCC2037u, 0x3DC2BE10u, 0x3DB9CA56u, 0x3DB13FF4u, 0x3DA91A14u, 0x3DA15417u, 0x3D99E994u,
        0x3D92D656u, 0x3D8C165Cu, 0x3D85A5D0u, 0x3D7F0217u, 0x3D734928u, 0x3D681A2Cu, 0x3D5D6ECBu, 0x3D5340F9u,
        0x3D498AF1u, 0x3D404730u, 0x3D377076u, 0x3D2F01BFu, 0x3D26F645u, 0x3D1F4976u, 0x3D17F6F9u, 0x3D10FAA7u,
        0x3D0A508Au, 0x3D03F4DBu, 0x3CFBC7FCu, 0x3CF03506u, 0x3CE52A45u, 0x3CDAA175u, 0x3CD0949Cu, 0x3CC6FE09u,
        0x3CBDD84Cu, 0x3CB51E34u, 0x3CACCACFu, 0x3CA4D964u, 0x3C9D4572u, 0x3C960AAEu, 0x3C8F24FDu, 0x3C889077u,
        0x3C824961u, 0x3C789855u, 0x3C6D2ADEu, 0x3C6243E2u, 0x3C57DD32u, 0x3C4DF0E8u, 0x3C447965u, 0x3C3B714Au,
        0x3C32D377u, 0x3C2A9B0Bu, 0x3C22C35Bu, 0x3C1B47F6u, 0x3C14249Du, 0x3C0D5545u, 0x3C06D610u, 0x3C00A34Fu,
        0x3BF57300u, 0x3BEA2A8Eu, 0x3BDF66E3u, 0x3BD521E5u, 0x3BCB55C1u, 0x3BC1FCE9u, 0x3BB91210u, 0x3BB09027u,
        0x3BA8725Cu, 0x3BA0B414u, 0x3B9950ECu, 0x3B9244B3u, 0x3B8B8B6Au, 0x3B852141u, 0x3B7E052Bu, 0x3B7257DCu,
        0x3B6733F7u, 0x3B5C932Cu, 0x3B526F72u, 0x3B48C30Bu, 0x3B3F887Bu, 0x3B36BA85u, 0x3B2E542Cu, 0x3B2650ACu,
        0x3B1EAB7Au, 0x3B176040u, 0x3B106ADCu, 0x3B09C75Bu, 0x3B0371FAu, 0x3AFACE42u, 0x3AEF46C8u, 0x3AE446FAu,
        0x3AD9C89Cu, 0x3ACFC5BCu, 0x3AC638ACu, 0x3ABD1C01u, 0x3AB46A90u, 0x3AAC1F6Du, 0x3AA435E3u, 0x3A9CA976u,
        0x3A9575DDu, 0x3A8E9703u, 0x3A880904u, 0x3A81C828u, 0x3A77A1C4u, 0x3A6C3FA3u, 0x3A616377u, 0x3A570718u,
        0x3A4D24A6u, 0x3A43B687u, 0x3A3AB760u, 0x3A32221Au, 0x3A29F1D4u, 0x3A2221ECu, 0x3A1AADF3u, 0x3A1391AEu,
        0x3A0CC917u, 0x3A065054u, 0x3A0023B9u, 0x39F47F8Eu, 0x39E9424Du, 0x39DE8950u, 0x39D44E81u, 0x39CA8C15u,
        0x39C13C82u, 0x39B85A81u, 0x39AFE108u, 0x39A7CB4Au, 0x39A014B0u, 0x3998B8DCu, 0x3991B3A0u, 0x398B0103u,
        0x39849D36u, 0x397D0939u, 0x3971677Fu, 0x39664EA7u, 0x395BB866u, 0x39519EBBu, 0x3947FBECu, 0x393ECA83u,
        0x39360549u, 0x392DA744u, 0x3925ABB7u, 0x391E0E1Au, 0x3916CA1Cu, 0x390FDB9Fu, 0x39093EB4u, 0x3902EF9Bu,
        0x38F9D581u, 0x38EE5975u, 0x38E36490u, 0x38D8F09Bu, 0x38CEF7A9u, 0x38C57411u, 0x38BC6070u, 0x38B3B79Fu,
        0x38AB74B6u, 0x38A39305u, 0x389C0E14u, 0x3894E1A0u, 0x388E0997u, 0x38878218u, 0x3881476Fu, 0x3876AC28u,
        0x386B5551u, 0x386083EBu, 0x385631D2u, 0x384C592Eu, 0x3842F469u, 0x3839FE2Fu, 0x3831716Cu, 0x38294946u,
        0x3821811Du, 0x381A1488u, 0x3812FF51u, 0x380C3D74u, 0x3805CB1Cu, 0x37FF4942u, 0x37F38D0Eu, 0x37E85AF3u,
        0x37DDAC98u, 0x37D37BEFu, 0x37C9C330u, 0x37C07CD9u, 0x37B7A3A8u, 0x37AF3297u, 0x37A724DDu, 0x379F75EAu,
        0x37982162u, 0x3791231Du, 0x378A7724u, 0x378419AEu, 0x377C0E40u, 0x37707810u, 0x37656A3Au, 0x375ADE79u,
        0x3750CED2u, 0x37473592u, 0x373E0D47u, 0x373550C0u, 0x372CFB08u, 0x37250766u, 0x371D7156u, 0x3716348Eu,
        0x370F4CF0u, 0x3708B694u, 0x37026DBDu, 0x36F8DDB6u, 0x36ED6D0Eu, 0x36E28307u, 0x36D81970u, 0x36CE2A62u
    };
    fp32_t x_clamped = x;
    const fp32_t fp32_zero = fp32_t(0.0f);
    const fp32_t fp32_neg_t = fp32_t(-12.0f);
    if (x_clamped > fp32_zero) {
        x_clamped = fp32_zero;
    }
    if (x_clamped < fp32_neg_t) {
        x_clamped = fp32_neg_t;
    }
    const fp32_t idxf = (fp32_zero - x_clamped) * fp32_t(21.25f);
    int idx = (idxf + fp32_t(0.5f)).to_float();
    if (idx < 0) {
        idx = 0;
    }
    if (idx >= SoftmaxApproxCfg::EXP_LUT_SIZE) {
        idx = SoftmaxApproxCfg::EXP_LUT_SIZE - 1;
    }
    return fp32_from_bits((u32_t)kAttnSoftmaxExpLutBits[idx]);
}

static inline fp32_t attn_softmax_fp32_rcp_lut_nr(fp32_t sumexp) {
    fp32_t sumexp_safe = sumexp;
    const fp32_t fp32_eps = fp32_t(1.0e-6f);
    if (sumexp_safe < fp32_eps) {
        sumexp_safe = fp32_eps;
    }
    fp32_t sumexp_for_idx = sumexp_safe;
    const fp32_t fp32_one = fp32_t(1.0f);
    const fp32_t fp32_max = fp32_t(256.0f);
    if (sumexp_for_idx < fp32_one) {
        sumexp_for_idx = fp32_one;
    }
    if (sumexp_for_idx > fp32_max) {
        sumexp_for_idx = fp32_max;
    }
    int idx = ((sumexp_for_idx - fp32_one) + fp32_t(0.5f)).to_float();
    if (idx < 0) {
        idx = 0;
    }
    if (idx >= SoftmaxApproxCfg::RCP_LUT_SIZE) {
        idx = SoftmaxApproxCfg::RCP_LUT_SIZE - 1;
    }
    const fp32_t inv0 = fp32_t(1.0) / fp32_t((double)(idx + 1));
    const fp32_t two = fp32_t(2.0f);
    return inv0 * (two - (sumexp_safe * inv0));
}

static inline fp32_t attn_fp32_roundtrip(fp32_t x) {
    return fp32_from_bits(bits_from_fp32(x));
}

// local-only boundary descriptor for Top-fed prebuilt handoff flags.
// This keeps the AttnLayer0 seam contractized without changing Top ownership policy.
struct AttnLayer0PrebuiltHandoffDesc {
    bool kv_prebuilt_from_top_managed;
    bool q_prebuilt_from_top_managed;
    bool score_prebuilt_from_top_managed;
    bool out_prebuilt_from_top_managed;
    bool out_topfed_payload_enable;
    const u32_t* out_topfed_payload_words;
    u32_t out_topfed_payload_words_valid;
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
    desc.out_topfed_payload_enable = false;
    desc.out_topfed_payload_words = 0;
    desc.out_topfed_payload_words_valid = (u32_t)0u;
    return desc;
}

// local-only deeper ownership seam:
// caller can provide top-fed OUT payload that is consumed inside AttnLayer0 OUT stage.
static inline AttnLayer0PrebuiltHandoffDesc make_attn_layer0_prebuilt_handoff_desc(
    bool kv_prebuilt_from_top_managed,
    bool q_prebuilt_from_top_managed,
    bool score_prebuilt_from_top_managed,
    bool out_prebuilt_from_top_managed,
    bool out_topfed_payload_enable,
    const u32_t* out_topfed_payload_words,
    u32_t out_topfed_payload_words_valid
) {
    AttnLayer0PrebuiltHandoffDesc desc = make_attn_layer0_prebuilt_handoff_desc(
        kv_prebuilt_from_top_managed,
        q_prebuilt_from_top_managed,
        score_prebuilt_from_top_managed,
        out_prebuilt_from_top_managed);
    desc.out_topfed_payload_enable = out_topfed_payload_enable;
    desc.out_topfed_payload_words = out_topfed_payload_words;
    desc.out_topfed_payload_words_valid = out_topfed_payload_words_valid;
    return desc;
}

template<unsigned STAGE_MODE, typename SramView>
// AttnLayer0 executes stage-scoped attention work in SRAM windows owned by Top.
// Ownership boundary: this block only consumes the base offsets provided by Top.
// Reading rule: start from the stage comments below and do not treat this file as a
// single straight-line kernel; it is an accepted attention datapath mixed with
// migration-era prebuilt/top-fed seams.
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
    AttnLayer0PrebuiltHandoffDesc prebuilt_handoff = make_attn_layer0_prebuilt_handoff_desc(),
    u32_t layer_id_word = (u32_t)0
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
    const uint32_t layer_id = (uint32_t)layer_id_word.to_uint();
    if (n_heads == 0u) { n_heads = (uint32_t)ATTN_N_HEADS; }
    if (d_head == 0u) { d_head = d_model / n_heads; }
    quant_acc_t inv_sqrt_d_head = attn_inv_sqrt_d_head(d_head);
    bool q_dense_enabled_latched = false;
    bool k_dense_enabled_latched = false;
    bool v_dense_enabled_latched = false;
    bool q_live_ok_latched = false;
    bool k_live_ok_latched = false;
    bool v_live_ok_latched = false;

    // QKV stage:
    // X_WORK + W_REGION metadata/payload -> Q/K/V (plus act_q mirrors) in attention scratch windows.
    // C++20 compatibility: keep stage gating runtime-form while STAGE_MODE stays compile-time constant.
    if (STAGE_MODE == ATTN_STAGE_QKV || STAGE_MODE == ATTN_STAGE_FULL) {
        const uint32_t param_base = (uint32_t)param_base_word.to_uint();
        const QuantLinearMatrixId q_matrix_id = attn_q_matrix_id_for_layer(layer_id);
        const QuantLinearMatrixId k_matrix_id = attn_k_matrix_id_for_layer(layer_id);
        const QuantLinearMatrixId v_matrix_id = attn_v_matrix_id_for_layer(layer_id);
        const QuantLinearMeta live_q_meta = ternary_linear_live_meta(q_matrix_id);
        const QuantLinearMeta live_k_meta = ternary_linear_live_meta(k_matrix_id);
        const QuantLinearMeta live_v_meta = ternary_linear_live_meta(v_matrix_id);
        const bool live_q_enabled =
            attn_live_qkv_gate_ok(live_q_meta, q_matrix_id, param_base, token_count, d_model);
        const bool live_k_enabled =
            attn_live_qkv_gate_ok(live_k_meta, k_matrix_id, param_base, token_count, d_model);
        const bool live_v_enabled =
            attn_live_qkv_gate_ok(live_v_meta, v_matrix_id, param_base, token_count, d_model);
        // P11AD mainline hook: Top-owned Q prebuild bypasses only Q generation.
        // Top may mark Q as prebuilt; this block then consumes Q SRAM as-is.
        const bool skip_q_materialization = prebuilt_handoff.q_prebuilt_from_top_managed;
        // P11AC mainline hook: Top-owned K/V prebuild bypasses only K/V generation.
        // Top may mark K/V as prebuilt; this block then skips local KV materialization only.
        const bool skip_kv_materialization = prebuilt_handoff.kv_prebuilt_from_top_managed;
        const bool dense_q_enabled =
            attn_dense_qkv_gate_ok(live_q_meta, q_matrix_id, param_base, token_count, d_model, layer_id);
        const bool dense_k_enabled =
            attn_dense_qkv_gate_ok(live_k_meta, k_matrix_id, param_base, token_count, d_model, layer_id);
        const bool dense_v_enabled =
            attn_dense_qkv_gate_ok(live_v_meta, v_matrix_id, param_base, token_count, d_model, layer_id);
        q_dense_enabled_latched = dense_q_enabled;
        k_dense_enabled_latched = dense_k_enabled;
        v_dense_enabled_latched = dense_v_enabled;
        bool live_q_ok = false;
        bool live_k_ok = false;
        bool live_v_ok = false;
        u32_t live_q_inv_sw_bits = (u32_t)0u;
        u32_t live_k_inv_sw_bits = (u32_t)0u;
        u32_t live_v_inv_sw_bits = (u32_t)0u;

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

        if (!skip_q_materialization) {
            const uint32_t q_act_q_base = (uint32_t)sc.q_act_q_base_word.to_uint();
            if (dense_q_enabled) {
                live_q_ok = attn_materialize_qkv_dense_fp32_contract(
                    sram,
                    param_base,
                    live_q_meta,
                    q_matrix_id,
                    layer_id,
                    token_count,
                    d_model,
                    x_in_base,
                    q_base,
                    q_act_q_base,
                    live_q_inv_sw_bits);
            } else if (live_q_enabled) {
                live_q_ok = true;
                bool use_q_split_top = (layer_id == 0u);
#if defined(AECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE)
                // Preferred Q path: fixed-shape split-kernel call over ternary leaf kernel.
                if (use_q_split_top) {
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
                }
#else
                use_q_split_top = false;
#endif
                if (!use_q_split_top) {
                    ATTN_Q_FALLBACK_TOKEN_LOOP: for (uint32_t t = 0; t < token_count && live_q_ok; ++t) {
                        const uint32_t x_row_base = x_in_base + t * d_model;
                        const uint32_t q_row_base = q_base + t * d_model;
                        const uint32_t q_act_q_row_base = q_act_q_base + t * d_model;
                        ATTN_Q_FALLBACK_OUTPUT_COL_LOOP: for (uint32_t out = 0; out < live_q_meta.rows; ++out) {
                            u32_t q_bits = 0;
                            u32_t inv_sw_bits = 0;
                            if (!ternary_linear_live_compute_q_elem(
                                    sram,
                                    param_base_word,
                                    q_matrix_id,
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
                }
            }
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

        if (!skip_kv_materialization) {
            const uint32_t k_act_q_base = (uint32_t)sc.k_act_q_base_word.to_uint();
            if (dense_k_enabled) {
                live_k_ok = attn_materialize_qkv_dense_fp32_contract(
                    sram,
                    param_base,
                    live_k_meta,
                    k_matrix_id,
                    layer_id,
                    token_count,
                    d_model,
                    x_in_base,
                    k_base,
                    k_act_q_base,
                    live_k_inv_sw_bits);
            } else if (live_k_enabled) {
                live_k_ok = true;
#if defined(AECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE)
                // Preferred K path: fixed-shape split-kernel call over ternary leaf kernel.
                bool use_k_split_top = (layer_id == 0u);
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
                            if (!ternary_linear_live_compute_q_elem(
                                    sram,
                                    param_base_word,
                                    k_matrix_id,
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
                        if (!ternary_linear_live_compute_q_elem(
                                sram,
                                param_base_word,
                                k_matrix_id,
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
            }
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

        if (!skip_kv_materialization) {
            const uint32_t v_act_q_base = (uint32_t)sc.v_act_q_base_word.to_uint();
            if (dense_v_enabled) {
                live_v_ok = attn_materialize_qkv_dense_fp32_contract(
                    sram,
                    param_base,
                    live_v_meta,
                    v_matrix_id,
                    layer_id,
                    token_count,
                    d_model,
                    x_in_base,
                    v_base,
                    v_act_q_base,
                    live_v_inv_sw_bits);
            } else if (live_v_enabled) {
                live_v_ok = true;
#if defined(AECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE)
                // Preferred V path: fixed-shape split-kernel call over ternary leaf kernel.
                bool use_v_split_top = (layer_id == 0u);
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
                            if (!ternary_linear_live_compute_q_elem(
                                    sram,
                                    param_base_word,
                                    v_matrix_id,
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
                        if (!ternary_linear_live_compute_q_elem(
                                sram,
                                param_base_word,
                                v_matrix_id,
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
            }
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
        q_live_ok_latched = live_q_ok;
        k_live_ok_latched = live_k_ok;
        v_live_ok_latched = live_v_ok;
    }

    if (STAGE_MODE == ATTN_STAGE_SCORES || STAGE_MODE == ATTN_STAGE_FULL) {
        // Stage ownership seam: prebuilt score means AE already committed score rows for AF consumption.
        if (prebuilt_handoff.score_prebuilt_from_top_managed) {
            // Top-managed AE/AF path already produced score/pre/post for this layer invocation.
            // Keep existing AC/AD hooks untouched and skip duplicate score/softmax execution only.
        } else {
        const uint32_t param_base = (uint32_t)param_base_word.to_uint();
        uint32_t src_mask_base = 0u;
        uint32_t src_mask_rows = 0u;
        uint32_t src_mask_cols = 0u;
        bool src_mask_ready = false;
        const bool dense_tail_fp32_enabled =
            q_dense_enabled_latched &&
            k_dense_enabled_latched &&
            v_dense_enabled_latched &&
            q_live_ok_latched &&
            k_live_ok_latched &&
            v_live_ok_latched;
        const fp32_t inv_sqrt_d_head_fp32 = attn_inv_sqrt_d_head_fp32(d_head);
        uint32_t src_mask_param_id = 0u;
        if (param_base != 0u && attn_src_mask_param_id(src_mask_param_id)) {
            const ParamMeta src_mask_meta = kParamMeta[src_mask_param_id];
            const uint32_t total_mask_bits = src_mask_meta.d0 * src_mask_meta.d1;
            const uint32_t total_mask_words = (total_mask_bits + 31u) >> 5;
            if (src_mask_meta.dtype == (uint32_t)PARAM_DTYPE_BITPACK &&
                src_mask_meta.ndims >= 2u &&
                src_mask_meta.d0 > 0u &&
                src_mask_meta.d1 > 0u &&
                src_mask_meta.len_w >= total_mask_words &&
                token_count <= src_mask_meta.d0 &&
                token_count <= src_mask_meta.d1) {
                src_mask_base = param_base + src_mask_meta.offset_w;
                src_mask_rows = src_mask_meta.d0;
                src_mask_cols = src_mask_meta.d1;
                src_mask_ready = true;
            }
        }
        const softmax_score_t score_mask_floor = softmax_score_t(-SoftmaxApproxCfg::SOFTMAX_NEG_T);
        const u32_t score_mask_neg_inf_bits = (u32_t)0xFF800000u;
        const fp32_t fp32_zero = fp32_t(0.0f);
        const fp32_t fp32_one_v = fp32_t(1.0f);
        const fp32_t fp32_neg_inf = fp32_from_bits(score_mask_neg_inf_bits);
        // Scores stage:
        // Q/K/V -> scaled dot products -> softmax probabilities -> pre-concat accumulation.
        // Optional debug dumps are emitted only for (t=0, h=0) into score/softmax windows.
        softmax_score_t score_row[N_NODES];
        softmax_prob_t prob_row[N_NODES];
        fp32_t score_row_fp32[N_NODES];
        fp32_t prob_row_fp32[N_NODES];
        bool masked_row[N_NODES];
        softmax_score_t unmasked_score_row[N_NODES];
        softmax_prob_t unmasked_prob_row[N_NODES];
        uint32_t unmasked_keys[N_NODES];
        ATTN_SCORE_TOKEN_LOOP: for (uint32_t t = 0; t < token_count; ++t) {
            ATTN_SCORE_HEAD_LOOP: for (uint32_t h = 0; h < n_heads; ++h) {
                uint32_t head_col_base = h * d_head;
                const uint32_t q_row_base = q_base + t * d_model + head_col_base;
                uint32_t unmasked_count = 0u;
                bool dense_multi_ctx_ready = false;
                fp32_t dense_multi_ctx_row[D_MODEL];
                ATTN_DENSE_MULTI_CTX_INIT_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
                    dense_multi_ctx_row[d] = fp32_zero;
                }

                ATTN_SCORE_KEY_TOKEN_LOOP: for (uint32_t j = 0; j < token_count; ++j) {
                    const bool masked = attn_score_masked_by_src_contract(
                        sram,
                        src_mask_ready,
                        src_mask_base,
                        src_mask_rows,
                        src_mask_cols,
                        n_heads,
                        h,
                        t,
                        j);
                    masked_row[j] = masked;
                    if (masked) {
                        score_row[j] = score_mask_floor;
                        prob_row[j] = softmax_prob_t(0);
                        score_row_fp32[j] = fp32_neg_inf;
                        prob_row_fp32[j] = fp32_zero;
                        continue;
                    }
                    const uint32_t k_row_base = k_base + j * d_model + head_col_base;
                    if (dense_tail_fp32_enabled) {
                        fp32_t dot_fp = fp32_zero;
                        ATTN_SCORE_DOT_COL_DENSE_LOOP: for (uint32_t d = 0; d < d_head; ++d) {
                            const fp32_t qv_fp = fp32_from_bits(sram[q_row_base + d]);
                            const fp32_t kv_fp = fp32_from_bits(sram[k_row_base + d]);
                            dot_fp += qv_fp * kv_fp;
                        }
                        const fp32_t score_fp = dot_fp * inv_sqrt_d_head_fp32;
                        score_row[j] =
                            score_fp.template convert_to_ac_fixed<18, 6, true, AC_RND, AC_SAT>(false);
                        score_row_fp32[j] = score_fp;
                    } else {
                        quant_acc_t dot = 0;
                        ATTN_SCORE_DOT_COL_FIXED_LOOP: for (uint32_t d = 0; d < d_head; ++d) {
                            quant_act_t qv = quant_act_from_bits(sram[q_row_base + d]);
                            quant_act_t kv = quant_act_from_bits(sram[k_row_base + d]);
                            dot += quant_acc_t(qv) * quant_acc_t(kv);
                        }
                        score_row[j] = softmax_score_t(dot * inv_sqrt_d_head);
                        score_row_fp32[j] = fp32_t(score_row[j]);
                    }
                    unmasked_score_row[unmasked_count] = score_row[j];
                    unmasked_keys[unmasked_count] = j;
                    ++unmasked_count;
                }

                if (unmasked_count == 0u) {
                    ATTN_SCORE_MASK_ALL_ZERO_LOOP: for (uint32_t j = 0; j < token_count; ++j) {
                        prob_row[j] = softmax_prob_t(0);
                        prob_row_fp32[j] = fp32_zero;
                    }
                } else if (unmasked_count == 1u) {
                    const uint32_t only_key = unmasked_keys[0];
                    ATTN_SCORE_MASK_ONEHOT_LOOP: for (uint32_t j = 0; j < token_count; ++j) {
                        if (j == only_key) {
                            prob_row[j] = softmax_prob_t(1);
                            prob_row_fp32[j] = fp32_one_v;
                        } else {
                            prob_row[j] = softmax_prob_t(0);
                            prob_row_fp32[j] = fp32_zero;
                        }
                    }
                } else if (dense_tail_fp32_enabled) {
                    fp32_t online_max = fp32_zero;
                    fp32_t online_sumexp = fp32_zero;
                    bool online_init = false;
                    fp32_t online_ctx_row[D_MODEL];
                    ATTN_DENSE_ONLINE_CTX_INIT_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
                        online_ctx_row[d] = fp32_zero;
                    }
                    ATTN_DENSE_ONLINE_KEY_LOOP: for (uint32_t u = 0u; u < unmasked_count; ++u) {
                        const uint32_t key = unmasked_keys[u];
                        const fp32_t score_fp = score_row_fp32[key];
                        if (!online_init) {
                            online_max = score_fp;
                            online_sumexp = fp32_one_v;
                            ATTN_DENSE_ONLINE_CTX_BOOT_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
                                const uint32_t v_idx = v_base + key * d_model + head_col_base + d;
                                online_ctx_row[d] = fp32_from_bits(sram[v_idx]);
                            }
                            online_init = true;
                            continue;
                        }
                        if (score_fp > online_max) {
                            const fp32_t alpha =
                                attn_fp32_roundtrip(attn_softmax_fp32_exp_lut(online_max - score_fp));
                            online_sumexp = (online_sumexp * alpha) + fp32_one_v;
                            ATTN_DENSE_ONLINE_CTX_RENORM_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
                                const uint32_t v_idx = v_base + key * d_model + head_col_base + d;
                                const fp32_t v_fp = fp32_from_bits(sram[v_idx]);
                                online_ctx_row[d] = (online_ctx_row[d] * alpha) + v_fp;
                            }
                            online_max = score_fp;
                        } else {
                            const fp32_t beta =
                                attn_fp32_roundtrip(attn_softmax_fp32_exp_lut(score_fp - online_max));
                            online_sumexp += beta;
                            ATTN_DENSE_ONLINE_CTX_ACC_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
                                const uint32_t v_idx = v_base + key * d_model + head_col_base + d;
                                const fp32_t v_fp = fp32_from_bits(sram[v_idx]);
                                online_ctx_row[d] += beta * v_fp;
                            }
                        }
                    }
                    if (online_init) {
                        const fp32_t inv_sumexp_fp =
                            attn_fp32_roundtrip(attn_softmax_fp32_rcp_lut_nr(online_sumexp));
                        ATTN_DENSE_ONLINE_PROB_MATERIALIZE_LOOP: for (uint32_t u = 0u; u < unmasked_count; ++u) {
                            const uint32_t key = unmasked_keys[u];
                            const fp32_t w_fp =
                                attn_fp32_roundtrip(attn_softmax_fp32_exp_lut(score_row_fp32[key] - online_max));
                            const fp32_t prob_fp = attn_fp32_roundtrip(w_fp * inv_sumexp_fp);
                            prob_row_fp32[key] = prob_fp;
                            prob_row[key] =
                                prob_fp.template convert_to_ac_fixed<18, 2, false, AC_RND, AC_SAT>(false);
                        }
                        ATTN_DENSE_ONLINE_CTX_FINAL_LOOP: for (uint32_t d = 0u; d < d_head; ++d) {
                            dense_multi_ctx_row[d] =
                                attn_fp32_roundtrip(online_ctx_row[d] * inv_sumexp_fp);
                        }
                        dense_multi_ctx_ready = true;
                    } else {
                        ATTN_DENSE_ONLINE_EMPTY_PROB_LOOP: for (uint32_t j = 0; j < token_count; ++j) {
                            prob_row[j] = softmax_prob_t(0);
                            prob_row_fp32[j] = fp32_zero;
                        }
                    }
                } else {
                    SoftmaxApprox<N_NODES>(unmasked_score_row, unmasked_prob_row, unmasked_count);
                    uint32_t unmasked_idx = 0u;
                    ATTN_SCORE_MASK_SCATTER_LOOP: for (uint32_t j = 0; j < token_count; ++j) {
                        if (masked_row[j]) {
                            prob_row[j] = softmax_prob_t(0);
                            prob_row_fp32[j] = fp32_zero;
                            continue;
                        }
                        prob_row[j] = unmasked_prob_row[unmasked_idx];
                        prob_row_fp32[j] = fp32_t(unmasked_prob_row[unmasked_idx]);
                        ++unmasked_idx;
                    }
                }

                uint32_t dump_slot = 0u;
                if (attn_score_debug_dump_slot_for_probe(
                        h, t, n_heads, token_count, tensor_words, dump_slot)) {
                    const uint32_t score_dump_base = score_base + dump_slot * token_count;
                    const uint32_t prob_dump_base = softmax_base + dump_slot * token_count;
                    ATTN_SCORE_DEBUG_DUMP_LOOP: for (uint32_t j = 0; j < token_count; ++j) {
                        if (masked_row[j]) {
                            sram[score_dump_base + j] = score_mask_neg_inf_bits;
                            sram[prob_dump_base + j] = (u32_t)0u;
                        } else {
                            sram[score_dump_base + j] = bits_from_fp32(score_row_fp32[j]);
                            sram[prob_dump_base + j] = bits_from_fp32(prob_row_fp32[j]);
                        }
                    }
                }

                if (dense_tail_fp32_enabled && unmasked_count == 1u) {
                    const uint32_t only_key = unmasked_keys[0];
                    ATTN_PRECONCAT_DENSE_ONEHOT_COPY_LOOP: for (uint32_t d = 0; d < d_head; ++d) {
                        const uint32_t v_idx = v_base + only_key * d_model + head_col_base + d;
                        const uint32_t out_idx = pre_base + t * d_model + head_col_base + d;
                        sram[out_idx] = sram[v_idx];
                    }
                } else if (dense_tail_fp32_enabled && dense_multi_ctx_ready) {
                    ATTN_PRECONCAT_DENSE_ONLINE_CTX_COPY_LOOP: for (uint32_t d = 0; d < d_head; ++d) {
                        const uint32_t out_idx = pre_base + t * d_model + head_col_base + d;
                        sram[out_idx] = bits_from_fp32(dense_multi_ctx_row[d]);
                    }
                } else if (dense_tail_fp32_enabled) {
                    ATTN_PRECONCAT_DENSE_HEAD_COL_LOOP: for (uint32_t d = 0; d < d_head; ++d) {
                        fp32_t acc_fp = fp32_zero;
                        ATTN_PRECONCAT_DENSE_KEY_TOKEN_ACC_LOOP: for (uint32_t j = 0; j < token_count; ++j) {
                            if (masked_row[j]) {
                                continue;
                            }
                            const uint32_t v_idx = v_base + j * d_model + head_col_base + d;
                            const fp32_t vv_fp = fp32_from_bits(sram[v_idx]);
                            acc_fp += prob_row_fp32[j] * vv_fp;
                        }
                        const uint32_t out_idx = pre_base + t * d_model + head_col_base + d;
                        sram[out_idx] = bits_from_fp32(acc_fp);
                    }
                } else {
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
        }

        // Write-back boundary between pre-concat scratch and post-concat tensor.
        ATTN_POSTCONCAT_COPY_LOOP: for (uint32_t i = 0; i < tensor_words; ++i) {
            sram[post_base + i] = sram[pre_base + i];
        }
        }
    }

    if (STAGE_MODE == ATTN_STAGE_OUT || STAGE_MODE == ATTN_STAGE_FULL) {
        const uint32_t out_topfed_valid_raw =
            (uint32_t)prebuilt_handoff.out_topfed_payload_words_valid.to_uint();
        uint32_t out_topfed_valid = out_topfed_valid_raw;
        if (out_topfed_valid > tensor_words) {
            out_topfed_valid = tensor_words;
        }
        const bool out_topfed_ready =
            prebuilt_handoff.out_topfed_payload_enable &&
            (prebuilt_handoff.out_topfed_payload_words != 0) &&
            (out_topfed_valid >= tensor_words);

        // OUT priority 1: consume full Top-fed payload when descriptor coverage is complete.
        if (out_topfed_ready) {
            // Deeper consume site:
            // when caller-fed payload is valid, OUT stage consumes it before local scratch fallback.
            ATTN_OUT_TOPFED_PAYLOAD_CONSUME_LOOP: for (uint32_t i = 0; i < tensor_words; ++i) {
                sram[out_base + i] = prebuilt_handoff.out_topfed_payload_words[i];
            }
            return;
        }

        // OUT priority 2: prebuilt output already committed by Top-managed AF path.
        // Ownership seam: committed prebuilt output must not be overwritten by local fallback copy.
        if (prebuilt_handoff.out_prebuilt_from_top_managed) {
            // Top-managed AF path already wrote final output for this layer invocation.
            return;
        }

        // OUT priority 3: descriptor was enabled but not complete, so copy local post-concat as fallback.
        if (prebuilt_handoff.out_topfed_payload_enable) {
            // Invalid/short top-fed payload falls back to local post-concat consume path.
            ATTN_OUT_TOPFED_INVALID_FALLBACK_LOOP: for (uint32_t i = 0; i < tensor_words; ++i) {
                sram[out_base + i] = sram[post_base + i];
            }
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
// Public pointer-based attention entry.
// This is the compatibility/public wrapper used by TransformerLayer. The real work
// still lives in AttnLayer0CoreWindow(); this wrapper only forwards the caller-owned
// SRAM window and optional handoff descriptor.
static inline void AttnLayer0(
    u32_t* sram,
    const AttnCfg& cfg,
    u32_t x_in_base_word,
    u32_t attn_out_base_word,
    const AttnScratch& sc,
    u32_t param_base_word,
    AttnLayer0PrebuiltHandoffDesc prebuilt_handoff,
    u32_t layer_id_word = (u32_t)0
) {
    AttnLayer0CoreWindow<STAGE_MODE, u32_t*>(
        sram,
        cfg,
        x_in_base_word,
        attn_out_base_word,
        sc,
        param_base_word,
        prebuilt_handoff,
        layer_id_word
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
    bool out_prebuilt_from_top_managed = false,
    u32_t layer_id_word = (u32_t)0
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
            out_prebuilt_from_top_managed),
        layer_id_word
    );
}

// P00-011AN: first deep Attn boundary bridge.
// This keeps the first deep call-site boundary array-shaped for Catapult-facing paths.
// Internal compute semantics remain the same by forwarding to the accepted AttnLayer0 core.
template<unsigned STAGE_MODE, uint32_t SRAM_WORDS>
// Array-shaped bridge for Catapult-facing compile-prep and explicit ownership review.
// The bridge does not change datapath semantics; it only preserves a fixed-shape
// window boundary for Top-managed call sites.
static inline void AttnLayer0TopManagedWindowBridge(
    u32_t (&sram_window)[SRAM_WORDS],
    const AttnCfg& cfg,
    u32_t x_in_base_word,
    u32_t attn_out_base_word,
    const AttnScratch& sc,
    u32_t param_base_word,
    AttnLayer0PrebuiltHandoffDesc prebuilt_handoff,
    u32_t layer_id_word = (u32_t)0
) {
    AttnLayer0CoreWindow<STAGE_MODE, u32_t (&)[SRAM_WORDS]>(
        sram_window,
        cfg,
        x_in_base_word,
        attn_out_base_word,
        sc,
        param_base_word,
        prebuilt_handoff,
        layer_id_word
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
    bool out_prebuilt_from_top_managed = false,
    u32_t layer_id_word = (u32_t)0
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
            out_prebuilt_from_top_managed),
        layer_id_word
    );
}

} // namespace aecct
