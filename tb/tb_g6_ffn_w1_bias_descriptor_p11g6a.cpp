// P00-G6-SUBA-FFN-W1-BIAS: targeted W1 bias descriptor validation (local-only).
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
        std::printf("[p11g6a][FAIL] %s got=%u exp=%u\n", fail_label, (unsigned)got, (unsigned)exp);
        return false;
    }
    return true;
}

static bool expect_true(bool cond, const char* fail_label) {
    if (!cond) {
        std::printf("[p11g6a][FAIL] %s\n", fail_label);
        return false;
    }
    return true;
}

static bool addr_ok(uint32_t addr_word) {
    return addr_word < (uint32_t)sram_map::SRAM_WORDS_TOTAL;
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
    const uint32_t w1_bias_words = d_ffn;

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

    static aecct::u32_t topfed_x_words[aecct::FFN_X_WORDS];
    static aecct::u32_t topfed_w1_words[aecct::FFN_W1_WEIGHT_WORDS];
    static aecct::u32_t topfed_w1_bias[aecct::FFN_W1_BIAS_WORDS];
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_X_WORDS; ++i) { topfed_x_words[i] = 0; }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_WEIGHT_WORDS; ++i) { topfed_w1_words[i] = 0; }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_BIAS_WORDS; ++i) { topfed_w1_bias[i] = 0; }

    // top-fed x token rows:
    // t0=[1,2,3,4], t1=[5,6,7,8]
    for (uint32_t i = 0u; i < x_words; ++i) {
        topfed_x_words[i] = aecct::bits_from_fp32(aecct::fp32_t(1.0f + (float)i));
    }

    // top-fed w1 rows: j0=[1,1,1,1], j1=[2,2,2,2]
    for (uint32_t i = 0u; i < d_model; ++i) {
        topfed_w1_words[i] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
        topfed_w1_words[d_model + i] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    }

    // top-fed W1 bias: [10, 20]
    topfed_w1_bias[0] = aecct::bits_from_fp32(aecct::fp32_t(10.0f));
    topfed_w1_bias[1] = aecct::bits_from_fp32(aecct::fp32_t(20.0f));

    const uint32_t w1_bias_id = 4u;
    const uint32_t w1_weight_id = 36u;
    static aecct::u32_t legacy_x_snapshot[8];
    static aecct::u32_t legacy_w1_weight_snapshot[8];
    static aecct::u32_t legacy_w1_bias_snapshot[2];

    // Legacy SRAM payloads intentionally diverge from top-fed descriptors.
    for (uint32_t t = 0u; t < token_count; ++t) {
        for (uint32_t i = 0u; i < d_model; ++i) {
            const uint32_t idx = t * d_model + i;
            sram[x_base + idx] = aecct::bits_from_fp32(aecct::fp32_t(50.0f + (float)idx));
            legacy_x_snapshot[idx] = sram[x_base + idx];
        }
    }
    for (uint32_t j = 0u; j < d_ffn; ++j) {
        const uint32_t b_addr = aecct::ffn_param_addr_word(param_base, w1_bias_id, j);
        if (!expect_true(addr_ok(b_addr), "w1 bias addr")) { CCS_RETURN(1); }
        sram[b_addr] = aecct::bits_from_fp32(aecct::fp32_t(100.0f + (float)j));
        legacy_w1_bias_snapshot[j] = sram[b_addr];
        const uint32_t w_row = j * d_model;
        for (uint32_t i = 0u; i < d_model; ++i) {
            const uint32_t idx = w_row + i;
            const uint32_t w_addr = aecct::ffn_param_addr_word(param_base, w1_weight_id, idx);
            if (!expect_true(addr_ok(w_addr), "w1 weight addr")) { CCS_RETURN(1); }
            sram[w_addr] = aecct::bits_from_fp32(aecct::fp32_t(9.0f));
            legacy_w1_weight_snapshot[idx] = sram[w_addr];
        }
    }

    aecct::u32_t reject_flag = (aecct::u32_t)0u;
    aecct::u32_t fallback_touch_counter = (aecct::u32_t)0u;
    aecct::u32_t reject_stage = (aecct::u32_t)99u;
    aecct::u32_t* sram_ptr = sram;

    aecct::FFNLayer0<aecct::FFN_STAGE_W1>(
        sram_ptr,
        cfg,
        (aecct::u32_t)x_base,
        sc,
        (aecct::u32_t)param_base,
        (aecct::u32_t)0u,
        topfed_x_words,
        topfed_w1_words,
        (aecct::u32_t)w1_weight_words,
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
        topfed_w1_bias,
        (aecct::u32_t)w1_bias_words,
        &reject_stage
    );

    if (!expect_u32((uint32_t)reject_flag.to_uint(), 0u, "reject flag") ||
        !expect_u32((uint32_t)reject_stage.to_uint(), (uint32_t)aecct::FFN_REJECT_STAGE_NONE, "reject stage") ||
        !expect_u32((uint32_t)fallback_touch_counter.to_uint(), 0u, "fallback touch counter")) {
        CCS_RETURN(1);
    }

    uint32_t legacy_mismatch_count = 0u;
    for (uint32_t t = 0u; t < token_count; ++t) {
        for (uint32_t j = 0u; j < d_ffn; ++j) {
            aecct::quant_acc_t exp_acc = aecct::ffn_bias_from_word(topfed_w1_bias[j]);
            aecct::quant_acc_t legacy_acc = aecct::ffn_bias_from_word(legacy_w1_bias_snapshot[j]);
            const uint32_t w_row = j * d_model;
            for (uint32_t i = 0u; i < d_model; ++i) {
                const uint32_t x_idx = t * d_model + i;
                const uint32_t w_idx = w_row + i;
                exp_acc += aecct::quant_acc_t(aecct::quant_act_from_bits(topfed_x_words[x_idx])) *
                           aecct::quant_acc_t(aecct::quant_act_from_bits(topfed_w1_words[w_idx]));
                legacy_acc += aecct::quant_acc_t(aecct::quant_act_from_bits(legacy_x_snapshot[x_idx])) *
                              aecct::quant_acc_t(aecct::quant_act_from_bits(legacy_w1_weight_snapshot[w_idx]));
            }
            const uint32_t out_addr = w1_base + t * d_ffn + j;
            const uint32_t got_bits = (uint32_t)sram[out_addr].to_uint();
            const uint32_t exp_bits = (uint32_t)aecct::quant_bits_from_acc(exp_acc).to_uint();
            const uint32_t legacy_bits = (uint32_t)aecct::quant_bits_from_acc(legacy_acc).to_uint();
            if (!expect_u32(got_bits, exp_bits, "expected compare")) {
                CCS_RETURN(1);
            }
            if (got_bits != legacy_bits) {
                ++legacy_mismatch_count;
            }
        }
    }
    if (!expect_true(legacy_mismatch_count > 0u, "legacy mismatch aggregate")) {
        CCS_RETURN(1);
    }
    std::printf("G6FFN_SUBWAVE_A_W1_BIAS_TOPFED_PATH PASS\n");
    std::printf("G6FFN_SUBWAVE_A_W1_BIAS_EXPECTED_COMPARE PASS\n");

    for (uint32_t idx = 0u; idx < x_words; ++idx) {
        if (!expect_u32((uint32_t)sram[x_base + idx].to_uint(), (uint32_t)legacy_x_snapshot[idx].to_uint(), "legacy x touched")) {
            CCS_RETURN(1);
        }
    }
    for (uint32_t idx = 0u; idx < w1_weight_words; ++idx) {
        const uint32_t w_addr = aecct::ffn_param_addr_word(param_base, w1_weight_id, idx);
        if (!expect_u32((uint32_t)sram[w_addr].to_uint(), (uint32_t)legacy_w1_weight_snapshot[idx].to_uint(), "legacy w1 weight touched")) {
            CCS_RETURN(1);
        }
    }
    for (uint32_t idx = 0u; idx < w1_bias_words; ++idx) {
        const uint32_t b_addr = aecct::ffn_param_addr_word(param_base, w1_bias_id, idx);
        if (!expect_u32((uint32_t)sram[b_addr].to_uint(), (uint32_t)legacy_w1_bias_snapshot[idx].to_uint(), "legacy w1 bias touched")) {
            CCS_RETURN(1);
        }
    }

    std::printf("G6FFN_SUBWAVE_A_W1_BIAS_NO_SPURIOUS_TOUCH PASS\n");
    std::printf("PASS: tb_g6_ffn_w1_bias_descriptor_p11g6a\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
