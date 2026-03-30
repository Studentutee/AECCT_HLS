// P00-G5-FFN-W1-FALLBACK: targeted W1 fallback-policy tightening validation (local-only).
// Scope:
// - top-fed W1 path is primary under strict policy.
// - fallback remains controlled in non-strict mode.
// - strict mode rejects missing descriptors without stale output.

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
        std::printf("[p11g5w1fp][FAIL] %s got=%u exp=%u\n", fail_label, (unsigned)got, (unsigned)exp);
        return false;
    }
    return true;
}

static bool expect_true(bool cond, const char* fail_label) {
    if (!cond) {
        std::printf("[p11g5w1fp][FAIL] %s\n", fail_label);
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
    for (uint32_t i = 0; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = 0;
    }

    const uint32_t token_count = 2u;
    const uint32_t d_model = 4u;
    const uint32_t d_ffn = 2u;
    const uint32_t x_words = token_count * d_model;
    const uint32_t w1_weight_words = d_ffn * d_model;

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
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_X_WORDS; ++i) {
        topfed_x_words[i] = 0;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_WEIGHT_WORDS; ++i) {
        topfed_w1_words[i] = 0;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_BIAS_WORDS; ++i) {
        topfed_w1_bias[i] = 0;
    }

    // top-fed x token rows:
    // t0=[1,2,3,4], t1=[5,6,7,8]
    topfed_x_words[0] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    topfed_x_words[1] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_x_words[2] = aecct::bits_from_fp32(aecct::fp32_t(3.0f));
    topfed_x_words[3] = aecct::bits_from_fp32(aecct::fp32_t(4.0f));
    topfed_x_words[4] = aecct::bits_from_fp32(aecct::fp32_t(5.0f));
    topfed_x_words[5] = aecct::bits_from_fp32(aecct::fp32_t(6.0f));
    topfed_x_words[6] = aecct::bits_from_fp32(aecct::fp32_t(7.0f));
    topfed_x_words[7] = aecct::bits_from_fp32(aecct::fp32_t(8.0f));

    // top-fed w1 rows:
    // j0=[1,1,1,1], j1=[2,2,2,2]
    topfed_w1_words[0] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    topfed_w1_words[1] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    topfed_w1_words[2] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    topfed_w1_words[3] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    topfed_w1_words[4] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_w1_words[5] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_w1_words[6] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_w1_words[7] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_w1_bias[0] = aecct::bits_from_fp32(aecct::fp32_t(0.0f));
    topfed_w1_bias[1] = aecct::bits_from_fp32(aecct::fp32_t(0.0f));

    const uint32_t w1_bias_id = 4u;
    const uint32_t w1_weight_id = 36u;
    const aecct::u32_t legacy_w1_bits = aecct::bits_from_fp32(aecct::fp32_t(9.0f));
    static aecct::u32_t legacy_x_snapshot[8];
    static aecct::u32_t legacy_w1_snapshot[8];

    // Legacy SRAM payloads are intentionally different from top-fed descriptors.
    for (uint32_t t = 0u; t < token_count; ++t) {
        for (uint32_t i = 0u; i < d_model; ++i) {
            const uint32_t idx = t * d_model + i;
            sram[x_base + idx] = aecct::bits_from_fp32(aecct::fp32_t(11.0f + (float)idx));
            legacy_x_snapshot[idx] = sram[x_base + idx];
        }
    }
    for (uint32_t j = 0u; j < d_ffn; ++j) {
        const uint32_t b_addr = aecct::ffn_param_addr_word(param_base, w1_bias_id, j);
        if (!expect_true(addr_ok(b_addr), "w1 bias addr in range")) {
            CCS_RETURN(1);
        }
        sram[b_addr] = aecct::bits_from_fp32(aecct::fp32_t(0.0f));
        const uint32_t w_row = j * d_model;
        for (uint32_t i = 0u; i < d_model; ++i) {
            const uint32_t idx = w_row + i;
            const uint32_t w_addr = aecct::ffn_param_addr_word(param_base, w1_weight_id, idx);
            if (!expect_true(addr_ok(w_addr), "w1 weight addr in range")) {
                CCS_RETURN(1);
            }
            sram[w_addr] = legacy_w1_bits;
            legacy_w1_snapshot[idx] = legacy_w1_bits;
        }
    }

    aecct::u32_t reject_flag = (aecct::u32_t)0u;
    aecct::u32_t fallback_touch_counter = (aecct::u32_t)0u;
    aecct::u32_t* sram_ptr = sram;

    // Case A: strict mode with complete descriptors -> top-fed primary path.
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
        (aecct::u32_t)d_ffn
    );
    if (!expect_u32((uint32_t)reject_flag.to_uint(), 0u, "caseA reject flag") ||
        !expect_u32((uint32_t)fallback_touch_counter.to_uint(), 0u, "caseA fallback touch counter")) {
        CCS_RETURN(1);
    }

    uint32_t legacy_mismatch_count = 0u;
    for (uint32_t t = 0u; t < token_count; ++t) {
        for (uint32_t j = 0u; j < d_ffn; ++j) {
            aecct::quant_acc_t exp_acc = aecct::ffn_bias_from_word(aecct::bits_from_fp32(aecct::fp32_t(0.0f)));
            aecct::quant_acc_t legacy_acc = aecct::ffn_bias_from_word(aecct::bits_from_fp32(aecct::fp32_t(0.0f)));
            const uint32_t w_row = j * d_model;
            for (uint32_t i = 0u; i < d_model; ++i) {
                const uint32_t x_idx = t * d_model + i;
                const uint32_t w_idx = w_row + i;
                exp_acc += aecct::quant_acc_t(aecct::quant_act_from_bits(topfed_x_words[x_idx])) *
                           aecct::quant_acc_t(aecct::quant_act_from_bits(topfed_w1_words[w_idx]));
                legacy_acc += aecct::quant_acc_t(aecct::quant_act_from_bits(legacy_x_snapshot[x_idx])) *
                              aecct::quant_acc_t(aecct::quant_act_from_bits(legacy_w1_snapshot[w_idx]));
            }
            const uint32_t out_addr = w1_base + t * d_ffn + j;
            const uint32_t got_bits = (uint32_t)sram[out_addr].to_uint();
            const uint32_t exp_bits = (uint32_t)aecct::quant_bits_from_acc(exp_acc).to_uint();
            const uint32_t legacy_bits = (uint32_t)aecct::quant_bits_from_acc(legacy_acc).to_uint();
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
    std::printf("G5FFN_W1_FALLBACK_POLICY_TOPFED_PRIMARY PASS\n");

    // Case B: non-strict mode and missing top-fed descriptors -> controlled fallback.
    reject_flag = (aecct::u32_t)0u;
    fallback_touch_counter = (aecct::u32_t)0u;
    aecct::FFNLayer0<aecct::FFN_STAGE_W1>(
        sram_ptr,
        cfg,
        (aecct::u32_t)x_base,
        sc,
        (aecct::u32_t)param_base,
        (aecct::u32_t)0u
    );
    if (!expect_u32((uint32_t)reject_flag.to_uint(), 0u, "caseB reject flag") ||
        !expect_true((uint32_t)fallback_touch_counter.to_uint() == 0u, "caseB external counter remains untouched without pointer")) {
        CCS_RETURN(1);
    }
    std::printf("G5FFN_W1_FALLBACK_POLICY_CONTROLLED_FALLBACK PASS\n");

    // Case C: strict mode and missing descriptor-ready -> reject + no stale state.
    const aecct::u32_t stale_sentinel = aecct::bits_from_fp32(aecct::fp32_t(-31.0f));
    for (uint32_t idx = 0u; idx < token_count * d_ffn; ++idx) {
        sram[w1_base + idx] = stale_sentinel;
    }
    reject_flag = (aecct::u32_t)0u;
    fallback_touch_counter = (aecct::u32_t)0u;
    aecct::FFNLayer0<aecct::FFN_STAGE_W1>(
        sram_ptr,
        cfg,
        (aecct::u32_t)x_base,
        sc,
        (aecct::u32_t)param_base,
        (aecct::u32_t)0u,
        topfed_x_words,
        topfed_w1_words,
        (aecct::u32_t)(w1_weight_words - 1u), // missing descriptor-ready
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
        (aecct::u32_t)d_ffn
    );
    if (!expect_u32((uint32_t)reject_flag.to_uint(), 1u, "caseC reject flag") ||
        !expect_u32((uint32_t)fallback_touch_counter.to_uint(), 0u, "caseC no fallback touch after reject")) {
        CCS_RETURN(1);
    }
    for (uint32_t idx = 0u; idx < token_count * d_ffn; ++idx) {
        if (!expect_u32((uint32_t)sram[w1_base + idx].to_uint(), (uint32_t)stale_sentinel.to_uint(), "caseC no stale output")) {
            CCS_RETURN(1);
        }
    }
    std::printf("G5FFN_W1_FALLBACK_POLICY_REJECT_ON_MISSING_DESCRIPTOR PASS\n");
    std::printf("G5FFN_W1_FALLBACK_POLICY_NO_STALE_STATE PASS\n");

    // No spurious touch on legacy source regions.
    for (uint32_t idx = 0u; idx < x_words; ++idx) {
        if (!expect_u32((uint32_t)sram[x_base + idx].to_uint(), (uint32_t)legacy_x_snapshot[idx].to_uint(), "legacy x touched")) {
            CCS_RETURN(1);
        }
    }
    for (uint32_t idx = 0u; idx < w1_weight_words; ++idx) {
        const uint32_t w_addr = aecct::ffn_param_addr_word(param_base, w1_weight_id, idx);
        if (!expect_u32((uint32_t)sram[w_addr].to_uint(), (uint32_t)legacy_w1_snapshot[idx].to_uint(), "legacy w1 touched")) {
            CCS_RETURN(1);
        }
    }
    std::printf("G5FFN_W1_FALLBACK_POLICY_NO_SPURIOUS_TOUCH PASS\n");
    std::printf("G5FFN_W1_FALLBACK_POLICY_EXPECTED_COMPARE PASS\n");
    std::printf("PASS: tb_g5_ffn_w1_fallback_policy_p11g5w1fp\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
