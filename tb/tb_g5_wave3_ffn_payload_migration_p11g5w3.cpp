// P00-G5-WAVE3: targeted FFN payload migration validation (local-only).
// Scope:
// - Validate FFN W1 stage consumes caller-fed topfed_x payload when provided.
// - Validate legacy SRAM x payload is not used as the main consume source in this path.

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
        std::printf("[p11g5w3][FAIL] %s got=%u exp=%u\n", fail_label, (unsigned)got, (unsigned)exp);
        return false;
    }
    return true;
}

static bool expect_true(bool cond, const char* fail_label) {
    if (!cond) {
        std::printf("[p11g5w3][FAIL] %s\n", fail_label);
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
    const uint32_t x_words = token_count * d_model;

    const uint32_t x_base = (uint32_t)sram_map::X_WORK_BASE_W;
    const uint32_t w1_base = (uint32_t)sram_map::BASE_SCR_K_W;
    const uint32_t relu_base = (uint32_t)sram_map::BASE_SCR_V_W;
    const uint32_t w2_base = (uint32_t)sram_map::BASE_SCR_FINAL_SCALAR_W;
    const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;

    if (!expect_true(range_ok(x_base, x_words), "x range") ||
        !expect_true(range_ok(w1_base, token_count * d_ffn), "w1_out range") ||
        !expect_true(range_ok(relu_base, token_count * d_ffn), "relu_out range") ||
        !expect_true(range_ok(w2_base, token_count * d_model), "w2_out range")) {
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

    // Legacy SRAM x payload intentionally carries different values.
    static aecct::u32_t legacy_x_bits[8];
    for (uint32_t i = 0u; i < x_words; ++i) {
        legacy_x_bits[i] = aecct::bits_from_fp32(aecct::fp32_t(100.0f + (float)i));
        sram[x_base + i] = legacy_x_bits[i];
    }

    static aecct::u32_t topfed_x_words[aecct::FFN_X_WORDS];
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_X_WORDS; ++i) {
        topfed_x_words[i] = 0;
    }
    topfed_x_words[0] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    topfed_x_words[1] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_x_words[2] = aecct::bits_from_fp32(aecct::fp32_t(3.0f));
    topfed_x_words[3] = aecct::bits_from_fp32(aecct::fp32_t(4.0f));
    topfed_x_words[4] = aecct::bits_from_fp32(aecct::fp32_t(5.0f));
    topfed_x_words[5] = aecct::bits_from_fp32(aecct::fp32_t(6.0f));
    topfed_x_words[6] = aecct::bits_from_fp32(aecct::fp32_t(7.0f));
    topfed_x_words[7] = aecct::bits_from_fp32(aecct::fp32_t(8.0f));

    const uint32_t w1_bias_id = 4u;
    const uint32_t w1_weight_id = 36u;
    FFN_W1_PARAM_SETUP_LOOP: for (uint32_t j = 0u; j < d_ffn; ++j) {
        const uint32_t bias_addr = aecct::ffn_param_addr_word(param_base, w1_bias_id, j);
        if (!expect_true(addr_ok(bias_addr), "w1 bias addr in range")) {
            CCS_RETURN(1);
        }
        sram[bias_addr] = aecct::bits_from_fp32(aecct::fp32_t(0.0f));

        const uint32_t w_row = j * d_model;
        FFN_W1_WEIGHT_SETUP_LOOP: for (uint32_t i = 0u; i < d_model; ++i) {
            const uint32_t w_addr = aecct::ffn_param_addr_word(param_base, w1_weight_id, w_row + i);
            if (!expect_true(addr_ok(w_addr), "w1 weight addr in range")) {
                CCS_RETURN(1);
            }
            sram[w_addr] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
        }
    }

    aecct::u32_t* sram_ptr = sram;
    aecct::FFNLayer0<aecct::FFN_STAGE_W1>(
        sram_ptr,
        cfg,
        (aecct::u32_t)x_base,
        sc,
        (aecct::u32_t)param_base,
        (aecct::u32_t)0u,
        topfed_x_words
    );

    const uint32_t expect_token0 =
        (uint32_t)aecct::bits_from_fp32(aecct::fp32_t(10.0f)).to_uint();
    const uint32_t expect_token1 =
        (uint32_t)aecct::bits_from_fp32(aecct::fp32_t(26.0f)).to_uint();
    const uint32_t sentinel_token0 =
        (uint32_t)aecct::bits_from_fp32(aecct::fp32_t(406.0f)).to_uint();
    const uint32_t sentinel_token1 =
        (uint32_t)aecct::bits_from_fp32(aecct::fp32_t(422.0f)).to_uint();

    FFN_W1_OUTPUT_CHECK_LOOP: for (uint32_t j = 0u; j < d_ffn; ++j) {
        if (!expect_u32(
                (uint32_t)sram[w1_base + 0u * d_ffn + j].to_uint(),
                expect_token0,
                "ffn topfed token0 expected compare")) {
            CCS_RETURN(1);
        }
        if (!expect_u32(
                (uint32_t)sram[w1_base + 1u * d_ffn + j].to_uint(),
                expect_token1,
                "ffn topfed token1 expected compare")) {
            CCS_RETURN(1);
        }
        if (!expect_true(
                (uint32_t)sram[w1_base + 0u * d_ffn + j].to_uint() != sentinel_token0,
                "ffn output must not match legacy-sram token0 sum")) {
            CCS_RETURN(1);
        }
        if (!expect_true(
                (uint32_t)sram[w1_base + 1u * d_ffn + j].to_uint() != sentinel_token1,
                "ffn output must not match legacy-sram token1 sum")) {
            CCS_RETURN(1);
        }
    }
    std::printf("G5W3_FFN_TOPFED_PAYLOAD_PATH PASS\n");

    FFN_X_NO_SPURIOUS_TOUCH_LOOP: for (uint32_t i = 0u; i < x_words; ++i) {
        if (!expect_u32(
                (uint32_t)sram[x_base + i].to_uint(),
                (uint32_t)legacy_x_bits[i].to_uint(),
                "legacy x payload changed unexpectedly")) {
            CCS_RETURN(1);
        }
    }
    std::printf("G5W3_FFN_NO_SPURIOUS_SRAM_TOUCH PASS\n");
    std::printf("G5W3_FFN_EXPECTED_COMPARE PASS\n");
    std::printf("PASS: tb_g5_wave3_ffn_payload_migration_p11g5w3\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
