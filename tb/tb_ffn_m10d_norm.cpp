#include <cmath>
#include <cstdint>
#include <cstdio>

#include "AecctTypes.h"
#include "FfnDescBringup.h"
#include "LayerNormDesc.h"
#include "gen/SramMap.h"
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

static int compare_tensor(
    const char* name,
    const aecct::u32_t* sram,
    uint32_t base_word,
    const double* trace_tensor,
    uint32_t sample_idx,
    uint32_t words,
    double tol
) {
    bool exact_ok = true;
    double max_abs_err = 0.0;
    uint32_t max_idx = 0u;
    for (uint32_t i = 0; i < words; ++i) {
        uint32_t got_bits = (uint32_t)sram[base_word + i].to_uint();
        float ref = (float)trace_tensor[sample_idx * words + i];
        uint32_t ref_bits = f32_to_bits(ref);
        if (got_bits != ref_bits) {
            exact_ok = false;
            double err = std::fabs((double)bits_to_f32(got_bits) - (double)ref);
            if (err > max_abs_err) {
                max_abs_err = err;
                max_idx = i;
            }
        }
    }

    if (exact_ok) {
        std::printf("PASS: %s exact-bit match\n", name);
        return 0;
    }

    std::printf("INFO: %s exact mismatch, max_abs_err=%.9g idx=%u\n", name, max_abs_err, (unsigned)max_idx);
    if (max_abs_err <= tol) {
        std::printf("PASS: %s abs_err<=%.1e\n", name, tol);
        return 0;
    }

    std::printf("ERROR: %s mismatch\n", name);
    return 1;
}

int main() {
    static aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
    for (uint32_t i = 0; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = (aecct::u32_t)0u;
    }

    const uint32_t sample_idx = 0u;
    const uint32_t words = (uint32_t)aecct::FFN_X_WORDS;
    aecct::FfnScratch sc = aecct::default_ffn_scratch();

    uint32_t x_in_base = (uint32_t)aecct::FFN_X_IN_BASE_WORD_DEFAULT;
    uint32_t w2_base = (uint32_t)sc.w2_out_base_word.to_uint();
    uint32_t add2_base = (uint32_t)sc.add2_base_word.to_uint();
    uint32_t ln_out_base = (uint32_t)sc.ln_out_base_word.to_uint();
    uint32_t gamma_base = (uint32_t)sc.ln_gamma_base_word.to_uint();
    uint32_t beta_base = (uint32_t)sc.ln_beta_base_word.to_uint();

    for (uint32_t i = 0; i < words; ++i) {
        float x = (float)trace_layer0_norm_attn_out_step0_tensor[sample_idx * words + i];
        float w2 = (float)trace_layer0_ffn_w2_out_step0_tensor[sample_idx * words + i];
        sram[x_in_base + i] = (aecct::u32_t)f32_to_bits(x);
        sram[w2_base + i] = (aecct::u32_t)f32_to_bits(w2);
        sram[add2_base + i] = (aecct::u32_t)f32_to_bits(x + w2);
    }

    for (uint32_t c = 0; c < (uint32_t)aecct::FFN_D_MODEL; ++c) {
        sram[gamma_base + c] = (aecct::u32_t)f32_to_bits((float)w_decoder_layers_0_sublayer_1_norm_weight[c]);
        sram[beta_base + c] = (aecct::u32_t)f32_to_bits((float)w_decoder_layers_0_sublayer_1_norm_bias[c]);
    }

    aecct::LayerNormCfg cfg;
    cfg.token_count = (aecct::u32_t)aecct::FFN_TOKEN_COUNT;
    cfg.d_model = (aecct::u32_t)aecct::FFN_D_MODEL;
    cfg.eps = aecct::LN_EPS;

    aecct::LayerNormBlock(
        sram,
        cfg,
        (aecct::u32_t)add2_base,
        (aecct::u32_t)ln_out_base,
        (aecct::u32_t)gamma_base,
        (aecct::u32_t)beta_base
    );

    int rc = compare_tensor(
        "layer0_norm_ffn_out",
        sram,
        ln_out_base,
        trace_layer0_norm_ffn_out_step0_tensor,
        sample_idx,
        words,
        1.0e-5
    );

    if (rc != 0) {
        std::printf("ERROR: tb_ffn_m10d_norm failed\n");
        return 1;
    }

    std::printf("PASS: tb_ffn_m10d_norm\n");
    return 0;
}

