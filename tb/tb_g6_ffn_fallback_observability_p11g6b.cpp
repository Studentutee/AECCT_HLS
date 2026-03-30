// P00-G6-SUBB-FFN-FALLBACK-OBS: targeted W1/W2 reject-stage observability harmonization (local-only).
#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>

#include "Top.h"

#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#define AECCT_HAS_SCVERIFY 1
#else
#define AECCT_HAS_SCVERIFY 0
#endif

#if !AECCT_HAS_SCVERIFY
#ifndef CCS_MAIN
#define CCS_MAIN(...) int main(__VA_ARGS__)
#endif
#ifndef CCS_RETURN
#define CCS_RETURN(x) return (x)
#endif
#endif

namespace {

static bool expect_u32(uint32_t got, uint32_t exp, const char* fail_label) {
    if (got != exp) {
        std::printf("[p11g6b][FAIL] %s got=%u exp=%u\n", fail_label, (unsigned)got, (unsigned)exp);
        return false;
    }
    return true;
}

static bool expect_true(bool cond, const char* fail_label) {
    if (!cond) {
        std::printf("[p11g6b][FAIL] %s\n", fail_label);
        return false;
    }
    return true;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;

    static aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
    for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = 0;
    }

    const uint32_t token_count = 2u;
    const uint32_t d_model = 4u;
    const uint32_t d_ffn = 2u;
    const uint32_t x_words = token_count * d_model;
    const uint32_t w1_weight_words = d_ffn * d_model;
    const uint32_t w2_input_words = token_count * d_ffn;
    const uint32_t w2_weight_words = d_model * d_ffn;
    const uint32_t w2_bias_words = d_model;

    const uint32_t x_base = (uint32_t)sram_map::X_PAGE0_BASE_W;
    const uint32_t w1_base = (uint32_t)sram_map::BASE_SCR_K_W;
    const uint32_t relu_base = (uint32_t)sram_map::BASE_SCR_V_W;
    const uint32_t w2_base = (uint32_t)sram_map::BASE_SCR_FINAL_SCALAR_W;
    const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;

    aecct::FfnCfg cfg;
    cfg.token_count = (aecct::u32_t)token_count;
    cfg.d_model = (aecct::u32_t)d_model;
    cfg.d_ffn = (aecct::u32_t)d_ffn;

    aecct::FfnScratch sc = aecct::default_ffn_scratch();
    sc.w1_out_base_word = (aecct::u32_t)w1_base;
    sc.relu_out_base_word = (aecct::u32_t)relu_base;
    sc.w2_out_base_word = (aecct::u32_t)w2_base;

    static aecct::u32_t topfed_x[aecct::FFN_X_WORDS];
    static aecct::u32_t topfed_w1[aecct::FFN_W1_WEIGHT_WORDS];
    static aecct::u32_t topfed_w2_input[aecct::FFN_W2_INPUT_WORDS];
    static aecct::u32_t topfed_w2_weight[aecct::FFN_W2_WEIGHT_WORDS];
    static aecct::u32_t topfed_w2_bias[aecct::FFN_W2_BIAS_WORDS];
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_X_WORDS; ++i) { topfed_x[i] = aecct::bits_from_fp32(aecct::fp32_t(1.0f)); }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_WEIGHT_WORDS; ++i) { topfed_w1[i] = aecct::bits_from_fp32(aecct::fp32_t(1.0f)); }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_INPUT_WORDS; ++i) { topfed_w2_input[i] = aecct::bits_from_fp32(aecct::fp32_t(1.0f)); }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_WEIGHT_WORDS; ++i) { topfed_w2_weight[i] = aecct::bits_from_fp32(aecct::fp32_t(1.0f)); }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_BIAS_WORDS; ++i) { topfed_w2_bias[i] = aecct::bits_from_fp32(aecct::fp32_t(0.0f)); }

    aecct::u32_t reject_flag = (aecct::u32_t)0u;
    aecct::u32_t reject_stage = (aecct::u32_t)99u;
    aecct::u32_t fallback_touch_counter = (aecct::u32_t)0u;
    aecct::u32_t* sram_ptr = sram;

    // Case A: strict W1 reject on missing descriptor-ready.
    const aecct::u32_t w1_sentinel = aecct::bits_from_fp32(aecct::fp32_t(-13.0f));
    for (uint32_t i = 0u; i < token_count * d_ffn; ++i) {
        sram[w1_base + i] = w1_sentinel;
    }
    aecct::FFNLayer0<aecct::FFN_STAGE_W1>(
        sram_ptr,
        cfg,
        (aecct::u32_t)x_base,
        sc,
        (aecct::u32_t)param_base,
        (aecct::u32_t)0u,
        topfed_x,
        topfed_w1,
        (aecct::u32_t)(w1_weight_words - 1u),
        0,
        (aecct::u32_t)0u,
        0,
        (aecct::u32_t)0u,
        0,
        (aecct::u32_t)0u,
        (aecct::u32_t)aecct::FFN_POLICY_REQUIRE_W1_TOPFED,
        &reject_flag,
        &fallback_touch_counter,
        (aecct::u32_t)x_words,
        0,
        (aecct::u32_t)0u,
        &reject_stage
    );
    if (!expect_u32((uint32_t)reject_flag.to_uint(), 1u, "caseA reject flag") ||
        !expect_u32((uint32_t)reject_stage.to_uint(), (uint32_t)aecct::FFN_REJECT_STAGE_W1, "caseA reject stage") ||
        !expect_u32((uint32_t)fallback_touch_counter.to_uint(), 0u, "caseA fallback touch")) {
        CCS_RETURN(1);
    }
    for (uint32_t i = 0u; i < token_count * d_ffn; ++i) {
        if (!expect_u32((uint32_t)sram[w1_base + i].to_uint(), (uint32_t)w1_sentinel.to_uint(), "caseA no stale")) {
            CCS_RETURN(1);
        }
    }
    std::printf("G6FFN_SUBWAVE_B_REJECT_STAGE_W1 PASS\n");

    // Case B: strict W2 reject on missing descriptor-ready.
    const aecct::u32_t w2_sentinel = aecct::bits_from_fp32(aecct::fp32_t(-29.0f));
    for (uint32_t i = 0u; i < token_count * d_model; ++i) {
        sram[w2_base + i] = w2_sentinel;
    }
    aecct::FFNLayer0<aecct::FFN_STAGE_W2>(
        sram_ptr,
        cfg,
        (aecct::u32_t)x_base,
        sc,
        (aecct::u32_t)param_base,
        (aecct::u32_t)0u,
        0,
        0,
        (aecct::u32_t)0u,
        topfed_w2_input,
        (aecct::u32_t)w2_input_words,
        topfed_w2_weight,
        (aecct::u32_t)w2_weight_words,
        topfed_w2_bias,
        (aecct::u32_t)(w2_bias_words - 1u),
        (aecct::u32_t)aecct::FFN_POLICY_REQUIRE_W2_TOPFED,
        &reject_flag,
        &fallback_touch_counter,
        (aecct::u32_t)0u,
        0,
        (aecct::u32_t)0u,
        &reject_stage
    );
    if (!expect_u32((uint32_t)reject_flag.to_uint(), 1u, "caseB reject flag") ||
        !expect_u32((uint32_t)reject_stage.to_uint(), (uint32_t)aecct::FFN_REJECT_STAGE_W2, "caseB reject stage") ||
        !expect_u32((uint32_t)fallback_touch_counter.to_uint(), 0u, "caseB fallback touch")) {
        CCS_RETURN(1);
    }
    for (uint32_t i = 0u; i < token_count * d_model; ++i) {
        if (!expect_u32((uint32_t)sram[w2_base + i].to_uint(), (uint32_t)w2_sentinel.to_uint(), "caseB no stale")) {
            CCS_RETURN(1);
        }
    }
    std::printf("G6FFN_SUBWAVE_B_REJECT_STAGE_W2 PASS\n");
    std::printf("G6FFN_SUBWAVE_B_NO_STALE_ON_REJECT PASS\n");

    // Case C: non-strict fallback path should be observable but not rejected.
    aecct::FFNLayer0<aecct::FFN_STAGE_W1>(
        sram_ptr,
        cfg,
        (aecct::u32_t)x_base,
        sc,
        (aecct::u32_t)param_base,
        (aecct::u32_t)0u,
        0,
        0,
        (aecct::u32_t)0u,
        0,
        (aecct::u32_t)0u,
        0,
        (aecct::u32_t)0u,
        0,
        (aecct::u32_t)0u,
        (aecct::u32_t)aecct::FFN_POLICY_NONE,
        &reject_flag,
        &fallback_touch_counter,
        (aecct::u32_t)0u,
        0,
        (aecct::u32_t)0u,
        &reject_stage
    );
    if (!expect_u32((uint32_t)reject_flag.to_uint(), 0u, "caseC reject flag") ||
        !expect_u32((uint32_t)reject_stage.to_uint(), (uint32_t)aecct::FFN_REJECT_STAGE_NONE, "caseC reject stage") ||
        !expect_true((uint32_t)fallback_touch_counter.to_uint() > 0u, "caseC fallback touch observed")) {
        CCS_RETURN(1);
    }
    std::printf("G6FFN_SUBWAVE_B_NONSTRICT_FALLBACK_OBS PASS\n");

    std::printf("PASS: tb_g6_ffn_fallback_observability_p11g6b\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
