#pragma once
// LayerNormBlock.h
// M8：LayerNorm 兩 pass（non in-place）

#include <cmath>
#include <cstdint>

#include "AecctTypes.h"
#include "LayerNormDesc.h"

namespace aecct {

    struct LayerNormCfg {
        u32_t token_count;
        u32_t d_model;
        float eps;
    };

    static inline float ln_bits_to_f32(u32_t w) {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.u = (uint32_t)w.to_uint();
        return cvt.f;
    }

    static inline u32_t ln_f32_to_bits(float f) {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.f = f;
        return (u32_t)cvt.u;
    }

    static inline void LayerNormBlock(
        u32_t* sram,
        const LayerNormCfg& cfg,
        u32_t x_in_base_word,
        u32_t x_out_base_word,
        u32_t gamma_base_word,
        u32_t beta_base_word
    ) {
        uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
        uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
        float eps = cfg.eps;

        uint32_t x_in_base = (uint32_t)x_in_base_word.to_uint();
        uint32_t x_out_base = (uint32_t)x_out_base_word.to_uint();
        uint32_t gamma_base = (uint32_t)gamma_base_word.to_uint();
        uint32_t beta_base = (uint32_t)beta_base_word.to_uint();

        for (uint32_t t = 0; t < token_count; ++t) {
            uint32_t row_in_base = x_in_base + t * d_model;
            uint32_t row_out_base = x_out_base + t * d_model;

            // pass1：計算 mean / var
            float sum = 0.0f;
            float sq_sum = 0.0f;
            for (uint32_t c = 0; c < d_model; ++c) {
                float x = ln_bits_to_f32(sram[row_in_base + c]);
                sum += x;
                sq_sum += (x * x);
            }
            float inv_n = 1.0f / (float)d_model;
            float mean = sum * inv_n;
            float var = sq_sum * inv_n - mean * mean;
            float inv_std = 1.0f / std::sqrt(var + eps);

            // pass2：normalize + affine，寫到 x_out（non in-place）
            for (uint32_t c = 0; c < d_model; ++c) {
                float x = ln_bits_to_f32(sram[row_in_base + c]);
                float g = ln_bits_to_f32(sram[gamma_base + c]);
                float b = ln_bits_to_f32(sram[beta_base + c]);
                float y = ((x - mean) * inv_std) * g + b;
                sram[row_out_base + c] = ln_f32_to_bits(y);
            }
        }
    }

} // namespace aecct
