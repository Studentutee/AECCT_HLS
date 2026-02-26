// tb_layernorm_m8.cpp
// M8 block TB：LayerNorm checkpoint 驗證（layer0 sublayer1 norm）

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "AecctTypes.h"
#include "LayerNormDesc.h"
#include "blocks/LayerNormBlock.h"
#include "layer0_ffn_w2_out_step0.h"
#include "layer0_norm_attn_out_step0.h"
#include "layer0_norm_ffn_out_step0.h"
#include "weights.h"

static uint32_t f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = f;
    return cvt.u;
}

static float bits_to_f32(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = u;
    return cvt.f;
}

int main() {
    aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
    for (uint32_t i = 0; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = (aecct::u32_t)0u;
    }

    const uint32_t sample_idx = 0u;
    const uint32_t elems_per_sample = (uint32_t)aecct::LN_X_TOTAL_WORDS;

    // 1) 組 LN 輸入：Add2 = layer0_ffn_w2_out + layer0_norm_attn_out
    for (uint32_t i = 0; i < elems_per_sample; ++i) {
        uint32_t flat = sample_idx * elems_per_sample + i;
        float a = (float)trace_layer0_ffn_w2_out_step0_tensor[flat];
        float b = (float)trace_layer0_norm_attn_out_step0_tensor[flat];
        sram[aecct::LN_X_IN_BASE_WORD_DEFAULT + i] = (aecct::u32_t)f32_to_bits(a + b);
    }

    // 2) 寫 gamma / beta（decoder.layers.0.sublayer.1.norm）
    for (uint32_t c = 0; c < (uint32_t)aecct::LN_D_MODEL; ++c) {
        float g = (float)w_decoder_layers_0_sublayer_1_norm_weight[c];
        float b = (float)w_decoder_layers_0_sublayer_1_norm_bias[c];
        sram[aecct::LN_GAMMA_BASE_WORD_DEFAULT + c] = (aecct::u32_t)f32_to_bits(g);
        sram[aecct::LN_BETA_BASE_WORD_DEFAULT + c] = (aecct::u32_t)f32_to_bits(b);
    }

    // 3) 呼叫 LN block（non in-place）
    aecct::LayerNormCfg cfg;
    cfg.token_count = (aecct::u32_t)aecct::LN_TOKEN_COUNT;
    cfg.d_model = (aecct::u32_t)aecct::LN_D_MODEL;
    cfg.eps = aecct::LN_EPS;

    aecct::LayerNormBlock(
        sram,
        cfg,
        (aecct::u32_t)aecct::LN_X_IN_BASE_WORD_DEFAULT,
        (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD_DEFAULT,
        (aecct::u32_t)aecct::LN_GAMMA_BASE_WORD_DEFAULT,
        (aecct::u32_t)aecct::LN_BETA_BASE_WORD_DEFAULT
    );

    // 4) 比對 golden：先 exact-bit，若不過再看 abs_err<=1e-5
    bool exact_ok = true;
    double max_abs_err = 0.0;
    uint32_t max_err_idx = 0u;
    for (uint32_t i = 0; i < elems_per_sample; ++i) {
        uint32_t flat = sample_idx * elems_per_sample + i;
        uint32_t got_bits = (uint32_t)sram[aecct::LN_X_OUT_BASE_WORD_DEFAULT + i].to_uint();
        float got = bits_to_f32(got_bits);
        float ref = (float)trace_layer0_norm_ffn_out_step0_tensor[flat];
        uint32_t ref_bits = f32_to_bits(ref);
        if (got_bits != ref_bits) {
            exact_ok = false;
            double err = std::fabs((double)got - (double)ref);
            if (err > max_abs_err) {
                max_abs_err = err;
                max_err_idx = i;
            }
        }
    }

    if (exact_ok) {
        std::printf("PASS: tb_layernorm_m8 exact-bit match\n");
        return 0;
    }

    std::printf("INFO: exact-bit mismatch, max_abs_err=%.9g at idx=%u\n",
        max_abs_err, (unsigned)max_err_idx);
    if (max_abs_err <= 1.0e-5) {
        std::printf("PASS: tb_layernorm_m8 abs_err<=1e-5\n");
        return 0;
    }

    std::printf("ERROR: tb_layernorm_m8 abs_err too large\n");
    return 1;
}
