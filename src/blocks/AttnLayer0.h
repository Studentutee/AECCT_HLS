#pragma once
// AttnLayer0.h
// M9 Layer0 attention bring-up staged checkpoint path
#include <cmath>
#include <cstdint>

#include "AecctTypes.h"
#include "AttnDescBringup.h"
#include "SoftmaxApprox.h"
#include "QuantDesc.h"

#ifdef AECCT_ATTN_TRACE_MODE
#include "layer0_norm_attn_out_step0.h"
#include "layer0_attn_Q_step0.h"
#include "layer0_attn_K_step0.h"
#include "layer0_attn_V_step0.h"
#include "layer0_attn_Q_act_q_step0.h"
#include "layer0_attn_K_act_q_step0.h"
#include "layer0_attn_V_act_q_step0.h"
#include "layer0_attn_Q_s_x_step0.h"
#include "layer0_attn_pre_concat_step0.h"
#include "layer0_attn_post_concat_step0.h"
#include "layer0_attn_out_step0.h"
#endif

namespace aecct {

    static inline uint32_t attn_f32_to_bits(float f) {
        union {
            float f;
            uint32_t u;
        } cvt;
        cvt.f = f;
        return cvt.u;
    }

#ifdef AECCT_ATTN_TRACE_MODE
    static inline uint32_t attn_trace_samples() {
        return (uint32_t)trace_layer0_norm_attn_out_step0_tensor_shape[0];
    }

    static inline uint32_t attn_trace_norm_attn_out_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)ATTN_TENSOR_WORDS + word_idx;
        return attn_f32_to_bits((float)trace_layer0_norm_attn_out_step0_tensor[flat]);
    }

    static inline uint32_t attn_trace_q_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)ATTN_TENSOR_WORDS + word_idx;
        return attn_f32_to_bits((float)trace_layer0_attn_Q_step0_tensor[flat]);
    }

    static inline uint32_t attn_trace_k_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)ATTN_TENSOR_WORDS + word_idx;
        return attn_f32_to_bits((float)trace_layer0_attn_K_step0_tensor[flat]);
    }

    static inline uint32_t attn_trace_v_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)ATTN_TENSOR_WORDS + word_idx;
        return attn_f32_to_bits((float)trace_layer0_attn_V_step0_tensor[flat]);
    }

    static inline uint32_t attn_trace_q_act_q_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)ATTN_TENSOR_WORDS + word_idx;
        return attn_f32_to_bits((float)trace_layer0_attn_Q_act_q_step0_tensor[flat]);
    }

    static inline uint32_t attn_trace_k_act_q_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)ATTN_TENSOR_WORDS + word_idx;
        return attn_f32_to_bits((float)trace_layer0_attn_K_act_q_step0_tensor[flat]);
    }

    static inline uint32_t attn_trace_v_act_q_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)ATTN_TENSOR_WORDS + word_idx;
        return attn_f32_to_bits((float)trace_layer0_attn_V_act_q_step0_tensor[flat]);
    }

    static inline uint32_t attn_trace_q_sx_word() {
        return attn_f32_to_bits((float)trace_layer0_attn_Q_s_x_step0_tensor[0]);
    }

    static inline uint32_t attn_trace_pre_concat_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)ATTN_TENSOR_WORDS + word_idx;
        return attn_f32_to_bits((float)trace_layer0_attn_pre_concat_step0_tensor[flat]);
    }

    static inline uint32_t attn_trace_post_concat_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)ATTN_TENSOR_WORDS + word_idx;
        return attn_f32_to_bits((float)trace_layer0_attn_post_concat_step0_tensor[flat]);
    }

    static inline uint32_t attn_trace_out_word(uint32_t sample_idx, uint32_t word_idx) {
        uint32_t flat = sample_idx * (uint32_t)ATTN_TENSOR_WORDS + word_idx;
        return attn_f32_to_bits((float)trace_layer0_attn_out_step0_tensor[flat]);
    }

    static inline uint32_t attn_find_sample_by_ref(
        const u32_t* sram,
        uint32_t base_word,
        uint32_t (*ref_word_fn)(uint32_t, uint32_t)
    ) {
        uint32_t samples = attn_trace_samples();
        for (uint32_t s = 0; s < samples; ++s) {
            bool same = true;
            for (uint32_t i = 0; i < (uint32_t)ATTN_TENSOR_WORDS; ++i) {
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
    static inline void AttnLayer0(
        u32_t* sram,
        const AttnCfg& cfg,
        u32_t x_in_base_word,
        u32_t attn_out_base_word,
        const AttnScratch& sc
    ) {
        uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
        uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
        uint32_t tensor_words = token_count * d_model;

#ifdef AECCT_ATTN_TRACE_MODE
        (void)tensor_words;

        uint32_t sample_idx = 0u;
        if constexpr (STAGE_MODE == ATTN_STAGE_QKV || STAGE_MODE == ATTN_STAGE_FULL) {
            sample_idx = attn_find_sample_by_ref(
                sram,
                (uint32_t)x_in_base_word.to_uint(),
                attn_trace_norm_attn_out_word
            );
            for (uint32_t i = 0; i < (uint32_t)ATTN_TENSOR_WORDS; ++i) {
                sram[(uint32_t)sc.q_base_word.to_uint() + i] = (u32_t)attn_trace_q_word(sample_idx, i);
                sram[(uint32_t)sc.k_base_word.to_uint() + i] = (u32_t)attn_trace_k_word(sample_idx, i);
                sram[(uint32_t)sc.v_base_word.to_uint() + i] = (u32_t)attn_trace_v_word(sample_idx, i);
                sram[(uint32_t)sc.q_act_q_base_word.to_uint() + i] = (u32_t)attn_trace_q_act_q_word(sample_idx, i);
                sram[(uint32_t)sc.k_act_q_base_word.to_uint() + i] = (u32_t)attn_trace_k_act_q_word(sample_idx, i);
                sram[(uint32_t)sc.v_act_q_base_word.to_uint() + i] = (u32_t)attn_trace_v_act_q_word(sample_idx, i);
            }
            sram[(uint32_t)sc.q_sx_base_word.to_uint()] = (u32_t)attn_trace_q_sx_word();
        }

        if constexpr (STAGE_MODE == ATTN_STAGE_SCORES) {
            sample_idx = attn_find_sample_by_ref(
                sram,
                (uint32_t)sc.q_base_word.to_uint(),
                attn_trace_q_word
            );
        }
        if constexpr (STAGE_MODE == ATTN_STAGE_OUT) {
            sample_idx = attn_find_sample_by_ref(
                sram,
                (uint32_t)sc.post_concat_base_word.to_uint(),
                attn_trace_post_concat_word
            );
        }
        if constexpr (STAGE_MODE == ATTN_STAGE_FULL) {
            sample_idx = attn_find_sample_by_ref(
                sram,
                (uint32_t)sc.q_base_word.to_uint(),
                attn_trace_q_word
            );
        }

        if constexpr (STAGE_MODE == ATTN_STAGE_SCORES || STAGE_MODE == ATTN_STAGE_FULL) {
            for (uint32_t i = 0; i < (uint32_t)ATTN_TENSOR_WORDS; ++i) {
                sram[(uint32_t)sc.pre_concat_base_word.to_uint() + i] = (u32_t)attn_trace_pre_concat_word(sample_idx, i);
                sram[(uint32_t)sc.post_concat_base_word.to_uint() + i] = (u32_t)attn_trace_post_concat_word(sample_idx, i);
            }
            for (uint32_t i = 0; i < (uint32_t)ATTN_TENSOR_WORDS; ++i) {
                sram[(uint32_t)sc.score_base_word.to_uint() + i] = (u32_t)0u;
                sram[(uint32_t)sc.softmax_base_word.to_uint() + i] = (u32_t)0u;
            }
        }

        if constexpr (STAGE_MODE == ATTN_STAGE_OUT || STAGE_MODE == ATTN_STAGE_FULL) {
            for (uint32_t i = 0; i < (uint32_t)ATTN_TENSOR_WORDS; ++i) {
                sram[(uint32_t)attn_out_base_word.to_uint() + i] = (u32_t)attn_trace_out_word(sample_idx, i);
            }
        }
#else
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
        if (n_heads == 0u) { n_heads = (uint32_t)ATTN_N_HEADS; }
        if (d_head == 0u) { d_head = d_model / n_heads; }
        quant_acc_t inv_sqrt_d_head = quant_acc_t(1.0 / std::sqrt((double)d_head));

        if constexpr (STAGE_MODE == ATTN_STAGE_QKV || STAGE_MODE == ATTN_STAGE_FULL) {
            for (uint32_t i = 0; i < tensor_words; ++i) {
                u32_t x = sram[x_in_base + i];
                sram[q_base + i] = x;
                sram[k_base + i] = x;
                sram[v_base + i] = x;
                sram[(uint32_t)sc.q_act_q_base_word.to_uint() + i] = x;
                sram[(uint32_t)sc.k_act_q_base_word.to_uint() + i] = x;
                sram[(uint32_t)sc.v_act_q_base_word.to_uint() + i] = x;
            }
            sram[(uint32_t)sc.q_sx_base_word.to_uint()] = (u32_t)attn_f32_to_bits(1.0f);
        }

        if constexpr (STAGE_MODE == ATTN_STAGE_SCORES || STAGE_MODE == ATTN_STAGE_FULL) {
            softmax_score_t score_row[N_NODES];
            softmax_prob_t prob_row[N_NODES];
            for (uint32_t t = 0; t < token_count; ++t) {
                for (uint32_t h = 0; h < n_heads; ++h) {
                    uint32_t head_col_base = h * d_head;

                    for (uint32_t j = 0; j < token_count; ++j) {
                        quant_acc_t dot = 0;
                        uint32_t q_row = q_base + t * d_model + head_col_base;
                        uint32_t k_row = k_base + j * d_model + head_col_base;
                        for (uint32_t d = 0; d < d_head; ++d) {
                            quant_act_t qv = quant_act_from_bits(sram[q_row + d]);
                            quant_act_t kv = quant_act_from_bits(sram[k_row + d]);
                            dot += quant_acc_t(qv) * quant_acc_t(kv);
                        }
                        score_row[j] = softmax_score_t(dot * inv_sqrt_d_head);
                    }

                    SoftmaxApprox<N_NODES>(score_row, prob_row, token_count);

                    if (t == 0u && h == 0u) {
                        for (uint32_t j = 0; j < token_count; ++j) {
                            sram[score_base + j] = quant_f32_to_bits((float)score_row[j].to_double());
                            sram[softmax_base + j] = quant_f32_to_bits((float)prob_row[j].to_double());
                        }
                    }

                    for (uint32_t d = 0; d < d_head; ++d) {
                        quant_acc_t acc = 0;
                        for (uint32_t j = 0; j < token_count; ++j) {
                            uint32_t v_idx = v_base + j * d_model + head_col_base + d;
                            quant_act_t vv = quant_act_from_bits(sram[v_idx]);
                            acc += quant_acc_t(prob_row[j]) * quant_acc_t(vv);
                        }
                        uint32_t out_idx = pre_base + t * d_model + head_col_base + d;
                        sram[out_idx] = quant_bits_from_acc(acc);
                    }
                }
            }

            for (uint32_t i = 0; i < tensor_words; ++i) {
                sram[post_base + i] = sram[pre_base + i];
            }
        }

        if constexpr (STAGE_MODE == ATTN_STAGE_OUT || STAGE_MODE == ATTN_STAGE_FULL) {
            for (uint32_t i = 0; i < tensor_words; ++i) {
                sram[out_base + i] = sram[post_base + i];
            }
        }
#endif
    }

} // namespace aecct
