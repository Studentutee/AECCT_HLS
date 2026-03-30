// P00-G5-WAVE1: targeted payload migration validation for LayerNormBlock + FinalHead (local-only).
// Scope:
// - Validate LayerNormBlock consumes Top-fed gamma/beta payload when provided.
// - Validate FinalHead consumes Top-fed final-scalar payload when provided.

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
        std::printf("[p11g5w1][FAIL] %s got=%u exp=%u\n", fail_label, (unsigned)got, (unsigned)exp);
        return false;
    }
    return true;
}

static bool expect_true(bool cond, const char* fail_label) {
    if (!cond) {
        std::printf("[p11g5w1][FAIL] %s\n", fail_label);
        return false;
    }
    return true;
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

    // ---------------- Wave1-A: LayerNorm top-fed affine payload ----------------
    const uint32_t ln_x_in_base = 0u;
    const uint32_t ln_x_out_base = 256u;
    const uint32_t ln_gamma_base = 512u;
    const uint32_t ln_beta_base = 768u;
    const uint32_t ln_d_model = 4u;
    const uint32_t ln_tokens = 1u;
    if (!expect_true(range_ok(ln_x_in_base, 64u), "ln_x_in range") ||
        !expect_true(range_ok(ln_x_out_base, 64u), "ln_x_out range") ||
        !expect_true(range_ok(ln_gamma_base, 64u), "ln_gamma range") ||
        !expect_true(range_ok(ln_beta_base, 64u), "ln_beta range")) {
        CCS_RETURN(1);
    }

    aecct::LayerNormCfg ln_cfg;
    ln_cfg.token_count = (aecct::u32_t)ln_tokens;
    ln_cfg.d_model = (aecct::u32_t)ln_d_model;
    ln_cfg.eps_bits = aecct::LN_EPS_BITS;

    aecct::LayerNormBlockContract ln_contract;
    aecct::clear_layernorm_contract(ln_contract);
    ln_contract.start = true;
    ln_contract.phase_id = aecct::PHASE_END_LN;
    ln_contract.x_work_base_word = (aecct::u32_t)ln_x_in_base;
    ln_contract.gamma_base_word = (aecct::u32_t)ln_gamma_base;
    ln_contract.beta_base_word = (aecct::u32_t)ln_beta_base;
    ln_contract.token_range = aecct::make_token_range((aecct::u32_t)0u, (aecct::u32_t)ln_tokens);
    ln_contract.tile_range = aecct::make_tile_range((aecct::u32_t)0u, (aecct::u32_t)1u);

    for (uint32_t c = 0; c < ln_d_model; ++c) {
        sram[ln_x_in_base + c] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
        sram[ln_gamma_base + c] = aecct::bits_from_fp32(aecct::fp32_t(9.0f)); // should be ignored in top-fed path
        sram[ln_beta_base + c] = aecct::bits_from_fp32(aecct::fp32_t(9.0f));  // should be ignored in top-fed path
    }

    aecct::u32_t topfed_gamma[aecct::LN_D_MODEL];
    aecct::u32_t topfed_beta[aecct::LN_D_MODEL];
    for (uint32_t c = 0; c < (uint32_t)aecct::LN_D_MODEL; ++c) {
        topfed_gamma[c] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
        topfed_beta[c] = 0;
    }
    topfed_beta[0] = aecct::bits_from_fp32(aecct::fp32_t(0.125f));
    topfed_beta[1] = aecct::bits_from_fp32(aecct::fp32_t(0.250f));
    topfed_beta[2] = aecct::bits_from_fp32(aecct::fp32_t(0.375f));
    topfed_beta[3] = aecct::bits_from_fp32(aecct::fp32_t(0.500f));

    aecct::u32_t* sram_ptr = sram;
    aecct::LayerNormBlockCoreWindow<aecct::u32_t*>(
        sram_ptr,
        ln_cfg,
        (aecct::u32_t)ln_x_in_base,
        (aecct::u32_t)ln_x_out_base,
        ln_contract,
        topfed_gamma,
        topfed_beta
    );

    for (uint32_t c = 0; c < ln_d_model; ++c) {
        if (!expect_u32(
                (uint32_t)sram[ln_x_out_base + c].to_uint(),
                (uint32_t)topfed_beta[c].to_uint(),
                "layernorm topfed beta output mismatch")) {
            CCS_RETURN(1);
        }
    }
    std::printf("G5W1_LN_TOPFED_AFFINE_NO_SPURIOUS_SRAM_TOUCH PASS\n");

    // ---------------- Wave1-B: FinalHead top-fed scalar payload ----------------
    const uint32_t fh_x_end_base = 2048u;
    const uint32_t fh_logits_base = 4096u;
    const uint32_t fh_xpred_base = 4352u;
    const uint32_t fh_final_scalar_base = 4608u;
    const uint32_t fh_out_fc_w_base = 4864u;
    const uint32_t fh_out_fc_b_base = 9216u;
    if (!expect_true(range_ok(fh_x_end_base, 256u), "fh_x_end range") ||
        !expect_true(range_ok(fh_logits_base, 512u), "fh_logits range") ||
        !expect_true(range_ok(fh_xpred_base, 512u), "fh_xpred range") ||
        !expect_true(range_ok(fh_final_scalar_base, 256u), "fh_final_scalar range") ||
        !expect_true(range_ok(fh_out_fc_w_base, 4096u), "fh_out_fc_w range") ||
        !expect_true(range_ok(fh_out_fc_b_base, 1024u), "fh_out_fc_b range")) {
        CCS_RETURN(1);
    }

    const uint32_t token_count = (uint32_t)N_NODES;
    aecct::CfgRegs cfg;
    cfg.d_model = (aecct::u32_t)1u;
    cfg.n_heads = (aecct::u32_t)1u;
    cfg.d_ffn = (aecct::u32_t)1u;
    cfg.n_layers = (aecct::u32_t)1u;

    aecct::FinalHeadContract fh_contract;
    aecct::clear_final_head_contract(fh_contract);
    fh_contract.start = true;
    fh_contract.phase_id = aecct::PHASE_FINAL_HEAD;
    fh_contract.x_work_base_word = (aecct::u32_t)fh_x_end_base;
    fh_contract.final_scalar_base_word = (aecct::u32_t)fh_final_scalar_base;
    fh_contract.token_range = aecct::make_token_range((aecct::u32_t)0u, (aecct::u32_t)2u);
    fh_contract.tile_range = aecct::make_tile_range((aecct::u32_t)0u, (aecct::u32_t)1u);

    aecct::HeadParamBase hp;
    hp.param_base_word = 0;
    hp.ffn1_w_base_word = 0;
    hp.ffn1_b_base_word = 0;
    hp.out_fc_w_base_word = (aecct::u32_t)fh_out_fc_w_base;
    hp.out_fc_b_base_word = (aecct::u32_t)fh_out_fc_b_base;

    // x_end payload intentionally different from top-fed scalar payload.
    sram[fh_x_end_base + 0u] = aecct::bits_from_fp32(aecct::fp32_t(10.0f));
    sram[fh_x_end_base + 1u] = aecct::bits_from_fp32(aecct::fp32_t(10.0f));

    // logits class 0 uses token0+token1, with bias 0 and weights 1.
    sram[fh_out_fc_b_base + 0u] = aecct::bits_from_fp32(aecct::fp32_t(0.0f));
    sram[fh_out_fc_w_base + 0u * token_count + 0u] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    sram[fh_out_fc_w_base + 0u * token_count + 1u] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));

    static aecct::u32_t topfed_final_scalar[N_NODES];
    for (uint32_t t = 0; t < token_count; ++t) {
        topfed_final_scalar[t] = 0;
    }
    topfed_final_scalar[0] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_final_scalar[1] = aecct::bits_from_fp32(aecct::fp32_t(3.0f));

    (void)aecct::FinalHeadCorePassABTopManaged<aecct::u32_t*>(
        sram_ptr,
        cfg,
        (aecct::u32_t)fh_x_end_base,
        (const aecct::u32_t*)0,
        (aecct::u32_t)fh_logits_base,
        (aecct::u32_t)fh_xpred_base,
        hp,
        fh_contract,
        (aecct::data_ch_t*)0,
        (aecct::u32_t)aecct::FINAL_HEAD_OUTMODE_NONE,
        topfed_final_scalar
    );

    const uint32_t expect_logits_bits =
        (uint32_t)aecct::bits_from_fp32(aecct::fp32_t(5.0f)).to_uint();
    if (!expect_u32(
            (uint32_t)sram[fh_logits_base + 0u].to_uint(),
            expect_logits_bits,
            "finalhead topfed scalar logits mismatch")) {
        CCS_RETURN(1);
    }
    std::printf("G5W1_FINALHEAD_TOPFED_SCALAR_PATH PASS\n");

    std::printf("PASS: tb_g5_wave1_payload_migration_p11g5w1\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
