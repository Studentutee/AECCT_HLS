#pragma once
// Layer0 FFN block.

#include <cstdint>

#include "AecctTypes.h"
#include "FfnDescBringup.h"
#include "QuantDesc.h"
#include "weights.h"

namespace aecct {

template<unsigned STAGE_MODE>
static inline void FFNLayer0(
    u32_t* sram,
    const FfnCfg& cfg,
    u32_t x_in_base_word,
    const FfnScratch& sc,
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

    if constexpr (STAGE_MODE == FFN_STAGE_W1 || STAGE_MODE == FFN_STAGE_FULL) {
        for (uint32_t t = 0; t < token_count; ++t) {
            uint32_t x_row = x_in_base + t * d_model;
            uint32_t h_row = w1_base + t * d_ffn;
            for (uint32_t j = 0; j < d_ffn; ++j) {
                quant_acc_t acc = use_layer1
                    ? quant_acc_t(w_decoder_layers_1_feed_forward_w_1_bias[j])
                    : quant_acc_t(w_decoder_layers_0_feed_forward_w_1_bias[j]);
                uint32_t w_row = j * d_model;
                for (uint32_t i = 0; i < d_model; ++i) {
                    quant_act_t x = quant_act_from_bits(sram[x_row + i]);
                    quant_w_t w = use_layer1
                        ? quant_w_t(w_decoder_layers_1_feed_forward_w_1_weight[w_row + i])
                        : quant_w_t(w_decoder_layers_0_feed_forward_w_1_weight[w_row + i]);
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
                quant_acc_t acc = use_layer1
                    ? quant_acc_t(w_decoder_layers_1_feed_forward_w_2_bias[i])
                    : quant_acc_t(w_decoder_layers_0_feed_forward_w_2_bias[i]);
                uint32_t w_row = i * d_ffn;
                for (uint32_t j = 0; j < d_ffn; ++j) {
                    quant_act_t a = quant_act_from_bits(sram[a_row + j]);
                    quant_w_t w = use_layer1
                        ? quant_w_t(w_decoder_layers_1_feed_forward_w_2_weight[w_row + j])
                        : quant_w_t(w_decoder_layers_0_feed_forward_w_2_weight[w_row + j]);
                    acc += quant_acc_t(a) * quant_acc_t(w);
                }
                sram[y_row + i] = quant_bits_from_acc(acc);
            }
        }
    }
}

} // namespace aecct
