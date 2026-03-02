#pragma once
// Layer0 FFN block.

#include <cstdint>

#include "AecctTypes.h"
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

static inline quant_w_t ffn_weight_from_sram(const u32_t* sram, uint32_t param_base_word, uint32_t param_id, uint32_t elem_idx) {
    return quant_w_t(quant_act_from_bits(sram[ffn_param_addr_word(param_base_word, param_id, elem_idx)]));
}

template<unsigned STAGE_MODE>
static inline void FFNLayer0(
    u32_t* sram,
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

    if constexpr (STAGE_MODE == FFN_STAGE_W1 || STAGE_MODE == FFN_STAGE_FULL) {
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

    if constexpr (STAGE_MODE == FFN_STAGE_RELU || STAGE_MODE == FFN_STAGE_FULL) {
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

    if constexpr (STAGE_MODE == FFN_STAGE_W2 || STAGE_MODE == FFN_STAGE_FULL) {
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

} // namespace aecct