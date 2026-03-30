// P00-G5-FFN-CLOSURE: targeted FFN subwave A/B/C/D validation (local-only).
// Scope:
// - Subwave A: W2 input activation tile path consumes caller-fed descriptor payload.
// - Subwave B: W2 weight tile path consumes caller-fed descriptor payload.
// - Subwave C: W2 bias path consumes caller-fed descriptor payload.
// - Subwave D: fallback boundary stays compatibility-only (top-fed path wins when provided).

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
        std::printf("[p11g5fc][FAIL] %s got=%u exp=%u\n", fail_label, (unsigned)got, (unsigned)exp);
        return false;
    }
    return true;
}

static bool expect_true(bool cond, const char* fail_label) {
    if (!cond) {
        std::printf("[p11g5fc][FAIL] %s\n", fail_label);
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

    if (!expect_true(range_ok(relu_base, w2_input_words), "relu input range") ||
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

    static aecct::u32_t topfed_w2_input[aecct::FFN_W2_INPUT_WORDS];
    static aecct::u32_t topfed_w2_weight[aecct::FFN_W2_WEIGHT_WORDS];
    static aecct::u32_t topfed_w2_bias[aecct::FFN_W2_BIAS_WORDS];
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_INPUT_WORDS; ++i) {
        topfed_w2_input[i] = 0;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_WEIGHT_WORDS; ++i) {
        topfed_w2_weight[i] = 0;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_BIAS_WORDS; ++i) {
        topfed_w2_bias[i] = 0;
    }

    // token0 a: [10,20], token1 a: [30,40]
    topfed_w2_input[0] = aecct::bits_from_fp32(aecct::fp32_t(10.0f));
    topfed_w2_input[1] = aecct::bits_from_fp32(aecct::fp32_t(20.0f));
    topfed_w2_input[2] = aecct::bits_from_fp32(aecct::fp32_t(30.0f));
    topfed_w2_input[3] = aecct::bits_from_fp32(aecct::fp32_t(40.0f));

    // W2 rows for output i=0..3
    // row0 [1,1], row1 [2,2], row2 [3,3], row3 [4,4]
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

    // Poison legacy SRAM paths to ensure top-fed descriptors are consumed.
    for (uint32_t t = 0u; t < token_count; ++t) {
        for (uint32_t j = 0u; j < d_ffn; ++j) {
            const uint32_t addr = relu_base + t * d_ffn + j;
            if (!expect_true(addr_ok(addr), "relu addr range")) {
                CCS_RETURN(1);
            }
            sram[addr] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
            legacy_relu_snapshot[t * d_ffn + j] = sram[addr];
        }
    }
    for (uint32_t i = 0u; i < d_model; ++i) {
        const uint32_t b_addr = aecct::ffn_param_addr_word(param_base, w2_bias_id, i);
        if (!expect_true(addr_ok(b_addr), "w2 bias addr range")) {
            CCS_RETURN(1);
        }
        sram[b_addr] = legacy_bias_bits;
        legacy_bias_snapshot[i] = legacy_bias_bits;
        const uint32_t w_row = i * d_ffn;
        for (uint32_t j = 0u; j < d_ffn; ++j) {
            const uint32_t idx = w_row + j;
            const uint32_t w_addr = aecct::ffn_param_addr_word(param_base, w2_weight_id, idx);
            if (!expect_true(addr_ok(w_addr), "w2 weight addr range")) {
                CCS_RETURN(1);
            }
            sram[w_addr] = legacy_weight_bits;
            legacy_weight_snapshot[idx] = legacy_weight_bits;
        }
    }

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
        (aecct::u32_t)w2_bias_words
    );

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
            if (!expect_u32(got_bits, exp_bits, "w2 expected compare")) {
                CCS_RETURN(1);
            }
            if (got_bits != legacy_bits) {
                ++legacy_mismatch_count;
            }
        }
    }
    if (!expect_true(legacy_mismatch_count > 0u, "legacy fallback mismatch aggregate check")) {
        CCS_RETURN(1);
    }
    std::printf("G5FFN_SUBWAVE_A_W2_INPUT_TOPFED_PATH PASS\n");
    std::printf("G5FFN_SUBWAVE_B_W2_WEIGHT_TOPFED_PATH PASS\n");
    std::printf("G5FFN_SUBWAVE_C_W2_BIAS_TOPFED_PATH PASS\n");

    // Ensure no unexpected write touches on legacy descriptor source SRAM words.
    for (uint32_t i = 0u; i < d_model; ++i) {
        const uint32_t b_addr = aecct::ffn_param_addr_word(param_base, w2_bias_id, i);
        if (!expect_u32((uint32_t)sram[b_addr].to_uint(), (uint32_t)legacy_bias_snapshot[i].to_uint(), "legacy w2 bias touched")) {
            CCS_RETURN(1);
        }
    }
    for (uint32_t idx = 0u; idx < w2_weight_words; ++idx) {
        const uint32_t w_addr = aecct::ffn_param_addr_word(param_base, w2_weight_id, idx);
        if (!expect_u32((uint32_t)sram[w_addr].to_uint(), (uint32_t)legacy_weight_snapshot[idx].to_uint(), "legacy w2 weight touched")) {
            CCS_RETURN(1);
        }
    }
    for (uint32_t idx = 0u; idx < w2_input_words; ++idx) {
        const uint32_t addr = relu_base + idx;
        if (!expect_u32((uint32_t)sram[addr].to_uint(), (uint32_t)legacy_relu_snapshot[idx].to_uint(), "legacy relu input touched")) {
            CCS_RETURN(1);
        }
    }
    std::printf("G5FFN_SUBWAVE_D_FALLBACK_BOUNDARY PASS\n");
    std::printf("G5FFN_SUBWAVE_ABCD_NO_SPURIOUS_SRAM_TOUCH PASS\n");
    std::printf("G5FFN_SUBWAVE_ABCD_EXPECTED_COMPARE PASS\n");
    std::printf("PASS: tb_g5_ffn_closure_campaign_p11g5fc\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
