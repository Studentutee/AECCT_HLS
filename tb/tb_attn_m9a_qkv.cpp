#define AECCT_ATTN_TRACE_MODE 1

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "AecctTypes.h"
#include "AttnDescBringup.h"
#include "SramMap.h"
#include "blocks/AttnLayer0.h"
#include "layer0_norm_attn_out_step0.h"
#include "layer0_attn_Q_step0.h"
#include "layer0_attn_K_step0.h"
#include "layer0_attn_V_step0.h"
#include "layer0_attn_Q_act_q_step0.h"
#include "layer0_attn_K_act_q_step0.h"
#include "layer0_attn_V_act_q_step0.h"
#include "layer0_attn_Q_s_x_step0.h"

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
    uint32_t max_got_bits = 0u;
    uint32_t max_ref_bits = 0u;

    for (uint32_t i = 0; i < words; ++i) {
        uint32_t got_bits = (uint32_t)sram[base_word + i].to_uint();
        float ref_f = (float)trace_tensor[sample_idx * words + i];
        uint32_t ref_bits = f32_to_bits(ref_f);
        if (got_bits != ref_bits) {
            exact_ok = false;
            double err = std::fabs((double)bits_to_f32(got_bits) - (double)ref_f);
            if (err > max_abs_err) {
                max_abs_err = err;
                max_idx = i;
                max_got_bits = got_bits;
                max_ref_bits = ref_bits;
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

    std::printf("ERROR: %s mismatch. idx=%u got=0x%08X ref=0x%08X\n",
        name, (unsigned)max_idx, (unsigned)max_got_bits, (unsigned)max_ref_bits);
    return 1;
}

static int compare_scalar(
    const char* name,
    uint32_t got_bits,
    float ref_f,
    double tol
) {
    uint32_t ref_bits = f32_to_bits(ref_f);
    if (got_bits == ref_bits) {
        std::printf("PASS: %s exact-bit match\n", name);
        return 0;
    }

    double err = std::fabs((double)bits_to_f32(got_bits) - (double)ref_f);
    std::printf("INFO: %s exact mismatch, abs_err=%.9g\n", name, err);
    if (err <= tol) {
        std::printf("PASS: %s abs_err<=%.1e\n", name, tol);
        return 0;
    }

    std::printf("ERROR: %s mismatch. got=0x%08X ref=0x%08X\n",
        name, (unsigned)got_bits, (unsigned)ref_bits);
    return 1;
}

int main() {
    static aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
    for (uint32_t i = 0; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = (aecct::u32_t)0u;
    }

    const uint32_t sample_idx = 0u;
    const uint32_t words = (uint32_t)aecct::ATTN_TENSOR_WORDS;

    for (uint32_t i = 0; i < words; ++i) {
        float x = (float)trace_layer0_norm_attn_out_step0_tensor[sample_idx * words + i];
        sram[aecct::ATTN_X_IN_BASE_WORD_DEFAULT + i] = (aecct::u32_t)f32_to_bits(x);
    }

    aecct::AttnCfg cfg;
    cfg.token_count = (aecct::u32_t)aecct::ATTN_TOKEN_COUNT;
    cfg.d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
    cfg.n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
    cfg.d_head = (aecct::u32_t)aecct::ATTN_D_HEAD;

    aecct::AttnScratch sc = aecct::default_attn_scratch();

    aecct::AttnLayer0<aecct::ATTN_STAGE_QKV>(
        sram,
        cfg,
        (aecct::u32_t)aecct::ATTN_X_IN_BASE_WORD_DEFAULT,
        (aecct::u32_t)aecct::ATTN_OUT_BASE_WORD_DEFAULT,
        sc
    );

    int rc = 0;
    rc |= compare_tensor("Q", sram, (uint32_t)sc.q_base_word.to_uint(), trace_layer0_attn_Q_step0_tensor, sample_idx, words, 1.0e-5);
    rc |= compare_tensor("K", sram, (uint32_t)sc.k_base_word.to_uint(), trace_layer0_attn_K_step0_tensor, sample_idx, words, 1.0e-5);
    rc |= compare_tensor("V", sram, (uint32_t)sc.v_base_word.to_uint(), trace_layer0_attn_V_step0_tensor, sample_idx, words, 1.0e-5);

    rc |= compare_tensor("Q_act_q", sram, (uint32_t)sc.q_act_q_base_word.to_uint(), trace_layer0_attn_Q_act_q_step0_tensor, sample_idx, words, 1.0e-5);
    rc |= compare_tensor("K_act_q", sram, (uint32_t)sc.k_act_q_base_word.to_uint(), trace_layer0_attn_K_act_q_step0_tensor, sample_idx, words, 1.0e-5);
    rc |= compare_tensor("V_act_q", sram, (uint32_t)sc.v_act_q_base_word.to_uint(), trace_layer0_attn_V_act_q_step0_tensor, sample_idx, words, 1.0e-5);

    rc |= compare_scalar(
        "Q_s_x",
        (uint32_t)sram[(uint32_t)sc.q_sx_base_word.to_uint()].to_uint(),
        (float)trace_layer0_attn_Q_s_x_step0_tensor[0],
        1.0e-5
    );

    if (rc != 0) {
        std::printf("ERROR: tb_attn_m9a_qkv failed\n");
        return 1;
    }

    std::printf("PASS: tb_attn_m9a_qkv\n");
    return 0;
}