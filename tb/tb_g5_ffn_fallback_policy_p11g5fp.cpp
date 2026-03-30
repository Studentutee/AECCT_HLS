// P00-G5-FFN-FALLBACK: targeted fallback policy tightening validation (local-only).
// Scope:
// - Ensure strict top-fed W2 policy keeps top-fed path as primary consume path.
// - Ensure fallback is controlled by explicit policy/descriptor readiness.
// - Ensure reject path leaves no stale/partial output state.

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
        std::printf("[p11g5fp][FAIL] %s got=%u exp=%u\n", fail_label, (unsigned)got, (unsigned)exp);
        return false;
    }
    return true;
}

static bool expect_true(bool cond, const char* fail_label) {
    if (!cond) {
        std::printf("[p11g5fp][FAIL] %s\n", fail_label);
        return false;
    }
    return true;
}

static bool addr_ok(uint32_t addr_word) {
    return addr_word < (uint32_t)sram_map::SRAM_WORDS_TOTAL;
}

static bool range_ok(uint32_t base, uint32_t len) {
    const uint32_t total = (uint32_t)sram_map::SRAM_WORDS_TOTAL;
    return (base < total) && (len <= (total - base));
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;

    static aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
    for (uint32_t i = 0; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = 0;
    }

    const uint32_t token_count = 2u;
    const uint32_t d_model = 4u;
    const uint32_t d_ffn = 2u;
    const uint32_t w2_input_words = token_count * d_ffn;
    const uint32_t w2_weight_words = d_model * d_ffn;
    const uint32_t w2_bias_words = d_model;

    const uint32_t x_base = (uint32_t)sram_map::X_PAGE0_BASE_W;
    const uint32_t w1_base = (uint32_t)sram_map::BASE_SCR_K_W;
    const uint32_t relu_base = (uint32_t)sram_map::BASE_SCR_V_W;
    const uint32_t w2_base = (uint32_t)sram_map::BASE_SCR_FINAL_SCALAR_W;
    const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;

    if (!expect_true(range_ok(relu_base, w2_input_words), "relu range") ||
        !expect_true(range_ok(w2_base, token_count * d_model), "w2 out range")) {
        CCS_RETURN(1);
    }

    aecct::FfnCfg cfg;
    cfg.token_count = (aecct::u32_t)token_count;
    cfg.d_model = (aecct::u32_t)d_model;
    cfg.d_ffn = (aecct::u32_t)d_ffn;

    aecct::FfnScratch sc = aecct::default_ffn_scratch();
    sc.w1_out_base_word = (aecct::u32_t)w1_base;
    sc.relu_out_base_word = (aecct::u32_t)relu_base;
    sc.w2_out_base_word = (aecct::u32_t)w2_base;

    static aecct::u32_t topfed_w2_input[aecct::FFN_W2_INPUT_WORDS];
    static aecct::u32_t topfed_w2_weight[aecct::FFN_W2_WEIGHT_WORDS];
    static aecct::u32_t topfed_w2_bias[aecct::FFN_W2_BIAS_WORDS];
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_INPUT_WORDS; ++i) { topfed_w2_input[i] = 0; }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_WEIGHT_WORDS; ++i) { topfed_w2_weight[i] = 0; }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_BIAS_WORDS; ++i) { topfed_w2_bias[i] = 0; }

    // token0 a: [10,20], token1 a: [30,40]
    topfed_w2_input[0] = aecct::bits_from_fp32(aecct::fp32_t(10.0f));
    topfed_w2_input[1] = aecct::bits_from_fp32(aecct::fp32_t(20.0f));
    topfed_w2_input[2] = aecct::bits_from_fp32(aecct::fp32_t(30.0f));
    topfed_w2_input[3] = aecct::bits_from_fp32(aecct::fp32_t(40.0f));

    // W2 rows: [1,1], [2,2], [3,3], [4,4]
    topfed_w2_weight[0] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    topfed_w2_weight[1] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    topfed_w2_weight[2] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_w2_weight[3] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_w2_weight[4] = aecct::bits_from_fp32(aecct::fp32_t(3.0f));
    topfed_w2_weight[5] = aecct::bits_from_fp32(aecct::fp32_t(3.0f));
    topfed_w2_weight[6] = aecct::bits_from_fp32(aecct::fp32_t(4.0f));
    topfed_w2_weight[7] = aecct::bits_from_fp32(aecct::fp32_t(4.0f));

    topfed_w2_bias[0] = aecct::bits_from_fp32(aecct::fp32_t(0.0f));
    topfed_w2_bias[1] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    topfed_w2_bias[2] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_w2_bias[3] = aecct::bits_from_fp32(aecct::fp32_t(3.0f));

    const uint32_t w2_weight_id = 39u;
    const uint32_t w2_bias_id = 5u;
    const aecct::u32_t legacy_bias_bits = aecct::bits_from_fp32(aecct::fp32_t(100.0f));
    const aecct::u32_t legacy_weight_bits = aecct::bits_from_fp32(aecct::fp32_t(9.0f));
    static aecct::u32_t legacy_bias_snapshot[4];
    static aecct::u32_t legacy_weight_snapshot[8];
    static aecct::u32_t legacy_relu_snapshot[4];

    for (uint32_t t = 0u; t < token_count; ++t) {
        for (uint32_t j = 0u; j < d_ffn; ++j) {
            const uint32_t addr = relu_base + t * d_ffn + j;
            if (!expect_true(addr_ok(addr), "relu addr")) { CCS_RETURN(1); }
            sram[addr] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
            legacy_relu_snapshot[t * d_ffn + j] = sram[addr];
        }
    }
    for (uint32_t i = 0u; i < d_model; ++i) {
        const uint32_t b_addr = aecct::ffn_param_addr_word(param_base, w2_bias_id, i);
        if (!expect_true(addr_ok(b_addr), "w2 bias addr")) { CCS_RETURN(1); }
        sram[b_addr] = legacy_bias_bits;
        legacy_bias_snapshot[i] = legacy_bias_bits;
        const uint32_t w_row = i * d_ffn;
        for (uint32_t j = 0u; j < d_ffn; ++j) {
            const uint32_t idx = w_row + j;
            const uint32_t w_addr = aecct::ffn_param_addr_word(param_base, w2_weight_id, idx);
            if (!expect_true(addr_ok(w_addr), "w2 weight addr")) { CCS_RETURN(1); }
            sram[w_addr] = legacy_weight_bits;
            legacy_weight_snapshot[idx] = legacy_weight_bits;
        }
    }

    // Case A: strict top-fed policy, all descriptors ready => primary path consume, no fallback touches.
    aecct::u32_t reject_flag = (aecct::u32_t)0u;
    aecct::u32_t fallback_touch_counter = (aecct::u32_t)0u;
    aecct::u32_t* sram_ptr = sram;
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
        (aecct::u32_t)w2_bias_words,
        (aecct::u32_t)aecct::FFN_POLICY_REQUIRE_W2_TOPFED,
        &reject_flag,
        &fallback_touch_counter
    );
    if (!expect_u32((uint32_t)reject_flag.to_uint(), 0u, "caseA reject flag") ||
        !expect_u32((uint32_t)fallback_touch_counter.to_uint(), 0u, "caseA fallback touch counter")) {
        CCS_RETURN(1);
    }

    uint32_t legacy_mismatch_count = 0u;
    for (uint32_t t = 0u; t < token_count; ++t) {
        for (uint32_t i = 0u; i < d_model; ++i) {
            const uint32_t y_idx = t * d_model + i;
            const uint32_t y_addr = w2_base + y_idx;
            aecct::quant_acc_t exp_acc = aecct::ffn_bias_from_word(topfed_w2_bias[i]);
            aecct::quant_acc_t legacy_acc = aecct::ffn_bias_from_word(legacy_bias_bits);
            const uint32_t w_row = i * d_ffn;
            for (uint32_t j = 0u; j < d_ffn; ++j) {
                const uint32_t a_idx = t * d_ffn + j;
                const uint32_t w_idx = w_row + j;
                const aecct::quant_act_t a_topfed = aecct::quant_act_from_bits(topfed_w2_input[a_idx]);
                const aecct::quant_w_t w_topfed = aecct::quant_act_from_bits(topfed_w2_weight[w_idx]);
                exp_acc += aecct::quant_acc_t(a_topfed) * aecct::quant_acc_t(w_topfed);
                const aecct::quant_act_t a_legacy = aecct::quant_act_from_bits(legacy_relu_snapshot[a_idx]);
                const aecct::quant_w_t w_legacy = aecct::quant_act_from_bits(legacy_weight_snapshot[w_idx]);
                legacy_acc += aecct::quant_acc_t(a_legacy) * aecct::quant_acc_t(w_legacy);
            }
            const uint32_t exp_bits = (uint32_t)aecct::quant_bits_from_acc(exp_acc).to_uint();
            const uint32_t legacy_bits = (uint32_t)aecct::quant_bits_from_acc(legacy_acc).to_uint();
            const uint32_t got_bits = (uint32_t)sram[y_addr].to_uint();
            if (!expect_u32(got_bits, exp_bits, "caseA expected compare")) {
                CCS_RETURN(1);
            }
            if (got_bits != legacy_bits) {
                ++legacy_mismatch_count;
            }
        }
    }
    if (!expect_true(legacy_mismatch_count > 0u, "caseA legacy mismatch aggregate")) {
        CCS_RETURN(1);
    }
    std::printf("G5FFN_FALLBACK_POLICY_TOPFED_PRIMARY PASS\n");
    std::printf("G5FFN_FALLBACK_POLICY_EXPECTED_COMPARE PASS\n");

    // Case B: strict top-fed policy, descriptor not ready => controlled reject + no stale output.
    const aecct::u32_t sentinel = aecct::bits_from_fp32(aecct::fp32_t(-77.0f));
    for (uint32_t idx = 0u; idx < token_count * d_model; ++idx) {
        sram[w2_base + idx] = sentinel;
    }
    reject_flag = (aecct::u32_t)0u;
    fallback_touch_counter = (aecct::u32_t)0u;
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
        (aecct::u32_t)(w2_weight_words - 1u), // force descriptor-not-ready reject
        topfed_w2_bias,
        (aecct::u32_t)w2_bias_words,
        (aecct::u32_t)aecct::FFN_POLICY_REQUIRE_W2_TOPFED,
        &reject_flag,
        &fallback_touch_counter
    );
    if (!expect_u32((uint32_t)reject_flag.to_uint(), 1u, "caseB reject flag") ||
        !expect_u32((uint32_t)fallback_touch_counter.to_uint(), 0u, "caseB fallback touch counter")) {
        CCS_RETURN(1);
    }
    for (uint32_t idx = 0u; idx < token_count * d_model; ++idx) {
        if (!expect_u32((uint32_t)sram[w2_base + idx].to_uint(), (uint32_t)sentinel.to_uint(), "caseB no stale output")) {
            CCS_RETURN(1);
        }
    }
    std::printf("G5FFN_FALLBACK_POLICY_CONTROLLED_FALLBACK PASS\n");
    std::printf("G5FFN_FALLBACK_POLICY_NO_STALE_STATE PASS\n");

    // No spurious touch on legacy source regions in both cases.
    for (uint32_t i = 0u; i < d_model; ++i) {
        const uint32_t b_addr = aecct::ffn_param_addr_word(param_base, w2_bias_id, i);
        if (!expect_u32((uint32_t)sram[b_addr].to_uint(), (uint32_t)legacy_bias_snapshot[i].to_uint(), "legacy bias touched")) {
            CCS_RETURN(1);
        }
    }
    for (uint32_t idx = 0u; idx < w2_weight_words; ++idx) {
        const uint32_t w_addr = aecct::ffn_param_addr_word(param_base, w2_weight_id, idx);
        if (!expect_u32((uint32_t)sram[w_addr].to_uint(), (uint32_t)legacy_weight_snapshot[idx].to_uint(), "legacy weight touched")) {
            CCS_RETURN(1);
        }
    }
    for (uint32_t idx = 0u; idx < w2_input_words; ++idx) {
        const uint32_t addr = relu_base + idx;
        if (!expect_u32((uint32_t)sram[addr].to_uint(), (uint32_t)legacy_relu_snapshot[idx].to_uint(), "legacy relu touched")) {
            CCS_RETURN(1);
        }
    }
    std::printf("G5FFN_FALLBACK_POLICY_NO_SPURIOUS_TOUCH PASS\n");
    std::printf("PASS: tb_g5_ffn_fallback_policy_p11g5fp\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
