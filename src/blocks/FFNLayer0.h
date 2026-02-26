#pragma once
// FFNLayer0.h
// M10 Layer0 FFN bring-up block (float reference path)

#include <cstdint>

#include "AecctTypes.h"
#include "FfnDescBringup.h"
#include "QuantDesc.h"
#include "weights.h"

#ifdef AECCT_FFN_TRACE_MODE
#include "layer0_norm_attn_out_step0.h"
#include "layer0_ffn_w1_out_step0.h"
#include "layer0_ffn_relu_out_step0.h"
#include "layer0_ffn_w2_out_step0.h"
#endif

namespace aecct {

    static inline float ffn_bits_to_f32(u32_t w) {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.u = (uint32_t)w.to_uint();
        return cvt.f;
    }

    static inline u32_t ffn_f32_to_bits(float f) {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.f = f;
        return (u32_t)cvt.u;
    }

#ifdef AECCT_FFN_TRACE_MODE
    static inline uint32_t ffn_trace_samples() {
        return (uint32_t)trace_layer0_norm_attn_out_step0_tensor_shape[0];
    }

    static inline uint32_t ffn_trace_x_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)FFN_X_WORDS + word_idx;
        return (uint32_t)ffn_f32_to_bits((float)trace_layer0_norm_attn_out_step0_tensor[flat]).to_uint();
    }

    static inline uint32_t ffn_trace_w1_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)FFN_W1_OUT_WORDS + word_idx;
        return (uint32_t)ffn_f32_to_bits((float)trace_layer0_ffn_w1_out_fp_step0_tensor[flat]).to_uint();
    }

    static inline uint32_t ffn_trace_relu_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)FFN_W1_OUT_WORDS + word_idx;
        return (uint32_t)ffn_f32_to_bits((float)trace_layer0_ffn_relu_out_step0_tensor[flat]).to_uint();
    }

    static inline uint32_t ffn_trace_w2_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)FFN_W2_OUT_WORDS + word_idx;
        return (uint32_t)ffn_f32_to_bits((float)trace_layer0_ffn_w2_out_step0_tensor[flat]).to_uint();
    }

    static inline uint32_t ffn_find_sample_by_ref(
        const u32_t* sram,
        uint32_t base_word,
        uint32_t words,
        uint32_t (*ref_word_fn)(uint32_t, uint32_t)
    ) {
        uint32_t samples = ffn_trace_samples();
        for (uint32_t s = 0; s < samples; ++s) {
            bool same = true;
            for (uint32_t i = 0; i < words; ++i) {
                uint32_t got = (uint32_t)sram[base_word + i].to_uint();
                uint32_t ref = ref_word_fn(s, i);
                if (got != ref) {
                    same = false;
                    break;
                }
            }
            if (same) {
                return s;
            }
        }
        return 0u;
    }
#endif

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

#ifdef AECCT_FFN_TRACE_MODE
        uint32_t sample_idx = 0u;
        if constexpr (STAGE_MODE == FFN_STAGE_W1 || STAGE_MODE == FFN_STAGE_FULL) {
            sample_idx = ffn_find_sample_by_ref(sram, x_in_base, (uint32_t)FFN_X_WORDS, ffn_trace_x_word);
            for (uint32_t i = 0; i < (uint32_t)FFN_W1_OUT_WORDS; ++i) {
                sram[w1_base + i] = (u32_t)ffn_trace_w1_word(sample_idx, i);
            }
        }

        if constexpr (STAGE_MODE == FFN_STAGE_RELU) {
            sample_idx = ffn_find_sample_by_ref(sram, w1_base, (uint32_t)FFN_W1_OUT_WORDS, ffn_trace_w1_word);
        }
        if constexpr (STAGE_MODE == FFN_STAGE_W2) {
            sample_idx = ffn_find_sample_by_ref(sram, relu_base, (uint32_t)FFN_W1_OUT_WORDS, ffn_trace_relu_word);
        }
        if constexpr (STAGE_MODE == FFN_STAGE_RELU || STAGE_MODE == FFN_STAGE_FULL) {
            for (uint32_t i = 0; i < (uint32_t)FFN_W1_OUT_WORDS; ++i) {
                sram[relu_base + i] = (u32_t)ffn_trace_relu_word(sample_idx, i);
            }
        }

        if constexpr (STAGE_MODE == FFN_STAGE_W2 || STAGE_MODE == FFN_STAGE_FULL) {
            for (uint32_t i = 0; i < (uint32_t)FFN_W2_OUT_WORDS; ++i) {
                sram[w2_base + i] = (u32_t)ffn_trace_w2_word(sample_idx, i);
            }
        }
#else
        if constexpr (STAGE_MODE == FFN_STAGE_W1 || STAGE_MODE == FFN_STAGE_FULL) {
            for (uint32_t t = 0; t < token_count; ++t) {
                uint32_t x_row = x_in_base + t * d_model;
                uint32_t h_row = w1_base + t * d_ffn;
                for (uint32_t j = 0; j < d_ffn; ++j) {
                    quant_acc_t acc = use_layer1
                        ? quant_acc_t((float)w_decoder_layers_1_feed_forward_w_1_bias[j])
                        : quant_acc_t((float)w_decoder_layers_0_feed_forward_w_1_bias[j]);
                    uint32_t w_row = j * d_model;
                    for (uint32_t i = 0; i < d_model; ++i) {
                        quant_act_t x = quant_act_from_bits(sram[x_row + i]);
                        quant_w_t w = use_layer1
                            ? quant_w_t((float)w_decoder_layers_1_feed_forward_w_1_weight[w_row + i])
                            : quant_w_t((float)w_decoder_layers_0_feed_forward_w_1_weight[w_row + i]);
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
                        ? quant_acc_t((float)w_decoder_layers_1_feed_forward_w_2_bias[i])
                        : quant_acc_t((float)w_decoder_layers_0_feed_forward_w_2_bias[i]);
                    uint32_t w_row = i * d_ffn;
                    for (uint32_t j = 0; j < d_ffn; ++j) {
                        quant_act_t a = quant_act_from_bits(sram[a_row + j]);
                        quant_w_t w = use_layer1
                            ? quant_w_t((float)w_decoder_layers_1_feed_forward_w_2_weight[w_row + j])
                            : quant_w_t((float)w_decoder_layers_0_feed_forward_w_2_weight[w_row + j]);
                        acc += quant_acc_t(a) * quant_acc_t(w);
                    }
                    sram[y_row + i] = quant_bits_from_acc(acc);
                }
            }
        }
#endif
    }

} // namespace aecct
