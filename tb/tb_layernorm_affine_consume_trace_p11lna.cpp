// Minimal LayerNorm affine consume observability probe.
// Scope: top-fed/fallback seam visibility only (local-only evidence).

#ifndef __SYNTHESIS__

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "AecctTypes.h"
#include "LayerNormDesc.h"
#include "blocks/LayerNormBlock.h"

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

static float bits_to_f32(aecct::u32_t bits) {
    union {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = (uint32_t)bits.to_uint();
    return cvt.f;
}

static bool expect_true(bool cond, const char* label) {
    if (!cond) {
        std::printf("[p11lna][FAIL] %s\n", label);
        return false;
    }
    return true;
}

static bool expect_close(float got, float exp, float tol, const char* label) {
    const float err = std::fabs(got - exp);
    if (err > tol) {
        std::printf(
            "[p11lna][FAIL] %s got=%.8f exp=%.8f err=%.8f tol=%.8f\n",
            label,
            got,
            exp,
            err,
            tol);
        return false;
    }
    return true;
}

static bool expect_u32_eq(aecct::u32_t got, uint32_t exp, const char* label) {
    const uint32_t got_u32 = (uint32_t)got.to_uint();
    if (got_u32 != exp) {
        std::printf("[p11lna][FAIL] %s got=%u exp=%u\n", label, (unsigned)got_u32, (unsigned)exp);
        return false;
    }
    return true;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;

    static aecct::u32_t sram[4096];
    for (uint32_t i = 0; i < 4096u; ++i) {
        sram[i] = 0;
    }

    const uint32_t x_in_base = 0u;
    const uint32_t x_out_base = 512u;
    const uint32_t gamma_base = 1024u;
    const uint32_t beta_base = 1536u;
    const uint32_t d_model = 2u;
    const uint32_t token_count = 1u;

    aecct::LayerNormCfg cfg;
    cfg.token_count = (aecct::u32_t)token_count;
    cfg.d_model = (aecct::u32_t)d_model;
    cfg.eps_bits = aecct::bits_from_fp32(aecct::fp32_t(0.0f));

    aecct::LayerNormBlockContract contract;
    aecct::clear_layernorm_contract(contract);
    contract.start = true;
    contract.phase_id = aecct::PHASE_END_LN;
    contract.x_work_base_word = (aecct::u32_t)x_in_base;
    contract.gamma_base_word = (aecct::u32_t)gamma_base;
    contract.beta_base_word = (aecct::u32_t)beta_base;
    contract.token_range = aecct::make_token_range((aecct::u32_t)0u, (aecct::u32_t)token_count);
    contract.tile_range = aecct::make_tile_range((aecct::u32_t)0u, (aecct::u32_t)1u);

    sram[x_in_base + 0u] = aecct::bits_from_fp32(aecct::fp32_t(1.0f));
    sram[x_in_base + 1u] = aecct::bits_from_fp32(aecct::fp32_t(3.0f));
    sram[gamma_base + 0u] = aecct::bits_from_fp32(aecct::fp32_t(5.0f));
    sram[gamma_base + 1u] = aecct::bits_from_fp32(aecct::fp32_t(5.0f));
    sram[beta_base + 0u] = aecct::bits_from_fp32(aecct::fp32_t(10.0f));
    sram[beta_base + 1u] = aecct::bits_from_fp32(aecct::fp32_t(20.0f));

    aecct::u32_t topfed_gamma[aecct::LN_D_MODEL];
    aecct::u32_t topfed_beta[aecct::LN_D_MODEL];
    for (uint32_t c = 0; c < (uint32_t)aecct::LN_D_MODEL; ++c) {
        topfed_gamma[c] = 0;
        topfed_beta[c] = 0;
    }
    topfed_gamma[0] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_gamma[1] = aecct::bits_from_fp32(aecct::fp32_t(2.0f));
    topfed_beta[0] = aecct::bits_from_fp32(aecct::fp32_t(30.0f));
    topfed_beta[1] = aecct::bits_from_fp32(aecct::fp32_t(40.0f));

    auto read_out_f32 = [&](uint32_t idx) -> float {
        return bits_to_f32(sram[x_out_base + idx]);
    };

    aecct::LayerNormAffineConsumeTrace trace;
    aecct::u32_t* sram_ptr = sram;

    // Case A: top-fed gamma+beta both present.
    aecct::LayerNormBlockCoreWindow<aecct::u32_t*>(
        sram_ptr,
        cfg,
        (aecct::u32_t)x_in_base,
        (aecct::u32_t)x_out_base,
        contract,
        topfed_gamma,
        topfed_beta,
        &trace
    );
    if (!expect_true(trace.saw_topfed_gamma && trace.saw_topfed_beta, "caseA saw topfed pair") ||
        !expect_true(trace.used_topfed_gamma && trace.used_topfed_beta, "caseA used topfed pair") ||
        !expect_true(!trace.used_fallback_gamma && !trace.used_fallback_beta, "caseA no fallback") ||
        !expect_u32_eq(trace.topfed_gamma_words_consumed, d_model, "caseA topfed gamma words") ||
        !expect_u32_eq(trace.topfed_beta_words_consumed, d_model, "caseA topfed beta words") ||
        !expect_u32_eq(trace.fallback_gamma_words_consumed, 0u, "caseA fallback gamma words") ||
        !expect_u32_eq(trace.fallback_beta_words_consumed, 0u, "caseA fallback beta words") ||
        !expect_close(read_out_f32(0u), 28.0f, 1.0e-4f, "caseA out0") ||
        !expect_close(read_out_f32(1u), 42.0f, 1.0e-4f, "caseA out1")) {
        CCS_RETURN(1);
    }

    // Case B: no top-fed payload, pure SRAM fallback.
    aecct::LayerNormBlockCoreWindow<aecct::u32_t*>(
        sram_ptr,
        cfg,
        (aecct::u32_t)x_in_base,
        (aecct::u32_t)x_out_base,
        contract,
        (const aecct::u32_t*)0,
        (const aecct::u32_t*)0,
        &trace
    );
    if (!expect_true(!trace.saw_topfed_gamma && !trace.saw_topfed_beta, "caseB saw no topfed") ||
        !expect_true(trace.used_fallback_gamma && trace.used_fallback_beta, "caseB used fallback pair") ||
        !expect_true(!trace.used_topfed_gamma && !trace.used_topfed_beta, "caseB no topfed use") ||
        !expect_u32_eq(trace.topfed_gamma_words_consumed, 0u, "caseB topfed gamma words") ||
        !expect_u32_eq(trace.topfed_beta_words_consumed, 0u, "caseB topfed beta words") ||
        !expect_u32_eq(trace.fallback_gamma_words_consumed, d_model, "caseB fallback gamma words") ||
        !expect_u32_eq(trace.fallback_beta_words_consumed, d_model, "caseB fallback beta words") ||
        !expect_close(read_out_f32(0u), 5.0f, 1.0e-4f, "caseB out0") ||
        !expect_close(read_out_f32(1u), 25.0f, 1.0e-4f, "caseB out1")) {
        CCS_RETURN(1);
    }

    // Case C: mixed source (gamma top-fed, beta fallback).
    aecct::LayerNormBlockCoreWindow<aecct::u32_t*>(
        sram_ptr,
        cfg,
        (aecct::u32_t)x_in_base,
        (aecct::u32_t)x_out_base,
        contract,
        topfed_gamma,
        (const aecct::u32_t*)0,
        &trace
    );
    if (!expect_true(trace.saw_topfed_gamma && !trace.saw_topfed_beta, "caseC mixed saw") ||
        !expect_true(trace.used_topfed_gamma && !trace.used_topfed_beta, "caseC gamma topfed only") ||
        !expect_true(!trace.used_fallback_gamma && trace.used_fallback_beta, "caseC beta fallback only") ||
        !expect_u32_eq(trace.topfed_gamma_words_consumed, d_model, "caseC topfed gamma words") ||
        !expect_u32_eq(trace.topfed_beta_words_consumed, 0u, "caseC topfed beta words") ||
        !expect_u32_eq(trace.fallback_gamma_words_consumed, 0u, "caseC fallback gamma words") ||
        !expect_u32_eq(trace.fallback_beta_words_consumed, d_model, "caseC fallback beta words") ||
        !expect_close(read_out_f32(0u), 8.0f, 1.0e-4f, "caseC out0") ||
        !expect_close(read_out_f32(1u), 22.0f, 1.0e-4f, "caseC out1")) {
        CCS_RETURN(1);
    }

    std::printf("PASS: tb_layernorm_affine_consume_trace_p11lna\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
