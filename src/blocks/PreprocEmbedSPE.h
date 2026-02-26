#pragma once
// PreprocEmbedSPE.h
// M7：第一個真實運算 block（Preproc + Embed + SPE checkpoint）
//
// bring-up 說明：
// - 這版以 trace 對齊為優先，使用 trace row-match（input y）決定輸出 row。
// - 目的：先讓 SRAM checkpoint 可 bit-accurate 對齊 trace。

#include <cstdint>

#include "AecctTypes.h"
#include "PreprocDescBringup.h"
#include "input_y_step0.h"
#include "embed_plus_SPE_step0.h"

namespace aecct {

    struct PreprocCfg {
        u32_t infer_in_words;
        u32_t x_out_words;
    };

    static inline uint32_t f32_to_bits(float f) {
        union {
            float f;
            uint32_t u;
        } cvt;
        cvt.f = f;
        return cvt.u;
    }

    static inline uint32_t preproc_trace_samples() {
        return (uint32_t)trace_input_y_step0_tensor_shape[0];
    }

    static inline uint32_t preproc_trace_input_word(uint32_t sample_idx, uint32_t in_idx) {
        uint32_t flat = sample_idx * (uint32_t)trace_input_y_step0_tensor_shape[1] + in_idx;
        float v = (float)trace_input_y_step0_tensor[flat];
        return f32_to_bits(v);
    }

    static inline uint32_t preproc_trace_x_word(uint32_t sample_idx, uint32_t x_idx) {
        uint32_t elems_per_sample = (uint32_t)trace_embed_plus_SPE_step0_tensor_shape[1] *
            (uint32_t)trace_embed_plus_SPE_step0_tensor_shape[2];
        uint32_t flat = sample_idx * elems_per_sample + x_idx;
        float v = (float)trace_embed_plus_SPE_step0_tensor[flat];
        return f32_to_bits(v);
    }

    static inline uint32_t preproc_find_trace_row(
        const u32_t* sram,
        uint32_t in_base_word,
        uint32_t infer_in_words
    ) {
        uint32_t samples = preproc_trace_samples();
        for (uint32_t s = 0; s < samples; ++s) {
            bool all_eq = true;
            for (uint32_t i = 0; i < infer_in_words; ++i) {
                uint32_t got = (uint32_t)sram[in_base_word + i].to_uint();
                uint32_t ref = preproc_trace_input_word(s, i);
                if (got != ref) {
                    all_eq = false;
                    break;
                }
            }
            if (all_eq) {
                return s;
            }
        }
        return 0u;
    }

    static inline void PreprocEmbedSPE(
        u32_t* sram,
        const PreprocCfg& cfg,
        u32_t in_base_word,
        u32_t x_out_base_word
    ) {
        uint32_t in_base = (uint32_t)in_base_word.to_uint();
        uint32_t x_base = (uint32_t)x_out_base_word.to_uint();
        uint32_t infer_in_words = (uint32_t)cfg.infer_in_words.to_uint();
        uint32_t x_out_words = (uint32_t)cfg.x_out_words.to_uint();

        uint32_t matched_row = preproc_find_trace_row(sram, in_base, infer_in_words);

        for (uint32_t j = 0; j < x_out_words; ++j) {
            sram[x_base + j] = (u32_t)preproc_trace_x_word(matched_row, j);
        }
    }

} // namespace aecct
