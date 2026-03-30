// P00-G5-WAVE2: targeted payload migration validation for PreprocEmbedSPE input payload (local-only).
// Scope:
// - Validate PreprocEmbedSPE consumes Top-fed infer input payload when provided.

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
        std::printf("[p11g5w2][FAIL] %s got=%u exp=%u\n", fail_label, (unsigned)got, (unsigned)exp);
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

    const uint32_t in_base = 0u;
    const uint32_t x_base = 256u;
    const uint32_t infer_in_words = 8u;
    const uint32_t x_out_words = 8u;
    if (!range_ok(in_base, 64u) || !range_ok(x_base, 64u)) {
        std::printf("[p11g5w2][FAIL] range check failed\n");
        CCS_RETURN(1);
    }

    aecct::PreprocCfg cfg;
    cfg.infer_in_words = (aecct::u32_t)infer_in_words;
    cfg.x_out_words = (aecct::u32_t)x_out_words;

    aecct::PreprocBlockContract contract;
    aecct::clear_preproc_contract(contract);
    contract.start = true;
    contract.phase_id = aecct::PHASE_PREPROC;
    contract.x_work_base_word = (aecct::u32_t)x_base;
    contract.token_range = aecct::make_token_range((aecct::u32_t)0u, (aecct::u32_t)2u);
    contract.tile_range = aecct::make_tile_range((aecct::u32_t)0u, (aecct::u32_t)2u);

    // SRAM source intentionally different from top-fed payload.
    for (uint32_t i = 0u; i < infer_in_words; ++i) {
        sram[in_base + i] = (aecct::u32_t)0xDEADBEEFu;
    }

    aecct::u32_t topfed_payload[aecct::PREPROC_IN_WORDS_EXPECTED];
    for (uint32_t i = 0u; i < (uint32_t)aecct::PREPROC_IN_WORDS_EXPECTED; ++i) {
        topfed_payload[i] = 0;
    }
    topfed_payload[0] = (aecct::u32_t)11u;
    topfed_payload[1] = (aecct::u32_t)22u;
    topfed_payload[2] = (aecct::u32_t)33u;
    topfed_payload[3] = (aecct::u32_t)44u;
    topfed_payload[4] = (aecct::u32_t)55u;
    topfed_payload[5] = (aecct::u32_t)66u;
    topfed_payload[6] = (aecct::u32_t)77u;
    topfed_payload[7] = (aecct::u32_t)88u;

    aecct::u32_t* sram_ptr = sram;
    aecct::PreprocEmbedSPECoreWindow<aecct::u32_t*>(
        sram_ptr,
        cfg,
        (aecct::u32_t)in_base,
        (aecct::u32_t)x_base,
        contract,
        topfed_payload
    );

    for (uint32_t i = 0u; i < infer_in_words; ++i) {
        if (!expect_u32(
                (uint32_t)sram[x_base + i].to_uint(),
                (uint32_t)topfed_payload[i].to_uint(),
                "preproc topfed payload mismatch")) {
            CCS_RETURN(1);
        }
    }

    std::printf("G5W2_PREPROC_TOPFED_INPUT_PATH PASS\n");
    std::printf("PASS: tb_g5_wave2_preproc_payload_migration_p11g5w2\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
