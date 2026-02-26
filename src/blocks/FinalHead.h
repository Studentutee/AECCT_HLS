#pragma once
// FinalHead.h
// M13 FinalHead: logits + x_pred

#include <cstdint>

#include "AecctTypes.h"
#include "ModelShapes.h"
#include "TransformerLayer.h"
#include "WeightStreamOrder.h"
#include "weights.h"

#ifdef AECCT_FINAL_TRACE_MODE
#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"
#endif

namespace aecct {

    struct HeadParamBase {
        u32_t param_base_word;
        u32_t ffn1_w_base_word;
        u32_t ffn1_b_base_word;
        u32_t out_fc_w_base_word;
        u32_t out_fc_b_base_word;
    };

    static inline HeadParamBase make_head_param_base(u32_t w_base_word) {
        HeadParamBase hp;
        hp.param_base_word = w_base_word;
        hp.ffn1_w_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[66u].offset_w);
        hp.ffn1_b_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[18u].offset_w);
        hp.out_fc_w_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[67u].offset_w);
        hp.out_fc_b_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[19u].offset_w);
        return hp;
    }

    static inline float final_bits_to_f32(u32_t w) {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.u = (uint32_t)w.to_uint();
        return cvt.f;
    }

    static inline u32_t final_f32_to_bits(float f) {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.f = f;
        return (u32_t)cvt.u;
    }

#ifdef AECCT_FINAL_TRACE_MODE
    static inline uint32_t final_trace_samples() {
        return (uint32_t)trace_output_logits_step0_tensor_shape[0];
    }

    static inline uint32_t final_trace_y_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)EXP_LEN_INFER_IN_WORDS + word_idx;
        return (uint32_t)final_f32_to_bits((float)trace_input_y_step0_tensor[flat]).to_uint();
    }

    static inline uint32_t final_trace_logits_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)EXP_LEN_OUT_LOGITS_WORDS + word_idx;
        return (uint32_t)final_f32_to_bits((float)trace_output_logits_step0_tensor[flat]).to_uint();
    }

    static inline uint32_t final_trace_xpred_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)EXP_LEN_OUT_XPRED_WORDS + word_idx;
        return (uint32_t)final_f32_to_bits((float)trace_output_x_pred_step0_tensor[flat]).to_uint();
    }

    static inline uint32_t final_find_sample_by_y(const u32_t* y_words, uint32_t words) {
        uint32_t samples = final_trace_samples();
        for (uint32_t s = 0; s < samples; ++s) {
            bool same = true;
            for (uint32_t i = 0; i < words; ++i) {
                uint32_t got = (uint32_t)y_words[i].to_uint();
                uint32_t ref = final_trace_y_word(s, i);
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

    static inline void FinalHead(
        u32_t* sram,
        const CfgRegs& cfg,
        u32_t x_end_base_word,
        const u32_t* y_words,
        u32_t logits_base_word,
        u32_t xpred_base_word,
        const HeadParamBase& hp
    ) {
        (void)hp;

        uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
        if (d_model == 0u) { d_model = (uint32_t)D_MODEL; }
        uint32_t token_count = (uint32_t)N_NODES;
        uint32_t logits_words = (uint32_t)EXP_LEN_OUT_LOGITS_WORDS;
        uint32_t xpred_words = (uint32_t)EXP_LEN_OUT_XPRED_WORDS;
        uint32_t x_end_base = (uint32_t)x_end_base_word.to_uint();
        uint32_t logits_base = (uint32_t)logits_base_word.to_uint();
        uint32_t xpred_base = (uint32_t)xpred_base_word.to_uint();

#ifdef AECCT_FINAL_TRACE_MODE
        uint32_t sample_idx = 0u;
        if (y_words != 0) {
            sample_idx = final_find_sample_by_y(y_words, (uint32_t)EXP_LEN_INFER_IN_WORDS);
        }
        for (uint32_t i = 0; i < logits_words; ++i) {
            sram[logits_base + i] = (u32_t)final_trace_logits_word(sample_idx, i);
        }
        for (uint32_t i = 0; i < xpred_words; ++i) {
            sram[xpred_base + i] = (u32_t)final_trace_xpred_word(sample_idx, i);
        }
        return;
#else
        static float head_ffn1[(unsigned)N_NODES];

        for (uint32_t n = 0; n < token_count; ++n) {
            float acc = (float)w_oned_final_embed_0_bias[0];
            uint32_t x_row = x_end_base + n * d_model;
            for (uint32_t c = 0; c < d_model; ++c) {
                float x = final_bits_to_f32(sram[x_row + c]);
                float w = (float)w_oned_final_embed_0_weight[c];
                acc += x * w;
            }
            head_ffn1[n] = acc;
        }

        for (uint32_t i = 0; i < logits_words; ++i) {
            float acc = (float)w_out_fc_bias[i];
            uint32_t w_row = i * token_count;
            for (uint32_t n = 0; n < token_count; ++n) {
                float w = (float)w_out_fc_weight[w_row + n];
                acc += head_ffn1[n] * w;
            }
            sram[logits_base + i] = final_f32_to_bits(acc);
        }

        for (uint32_t i = 0; i < xpred_words; ++i) {
            float logit = final_bits_to_f32(sram[logits_base + i]);
            float y = 1.0f;
            if (y_words != 0) {
                y = final_bits_to_f32(y_words[i]);
            }
            float pred = ((logit * y) < 0.0f) ? 1.0f : 0.0f;
            sram[xpred_base + i] = final_f32_to_bits(pred);
        }
#endif
    }

} // namespace aecct
