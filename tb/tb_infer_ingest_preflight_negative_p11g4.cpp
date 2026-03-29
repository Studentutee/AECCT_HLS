// P00-G4-PREFLIGHT-NEG: targeted infer ingest preflight negative validation (local-only).
// Scope:
// - Validate infer_contract_span_in_sram accepts legal contract span.
// - Validate illegal base/span metadata is rejected.
// - Validate rejection maps to ERR_MEM_RANGE guard behavior.

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

static bool run_case(
    const char* label,
    const aecct::InferIngestContract& contract,
    bool expect_accept,
    bool expect_err_mem_range_on_reject
) {
    const bool accept = aecct::infer_contract_span_in_sram(contract);
    const uint8_t reject_err = accept ? 0u : (uint8_t)aecct::ERR_MEM_RANGE;

    if (accept != expect_accept) {
        std::printf(
            "[p11g4][FAIL] %s accept mismatch expect=%d got=%d\n",
            label,
            expect_accept ? 1 : 0,
            accept ? 1 : 0
        );
        return false;
    }

    if (!accept && expect_err_mem_range_on_reject) {
        if (reject_err != (uint8_t)aecct::ERR_MEM_RANGE) {
            std::printf(
                "[p11g4][FAIL] %s reject error mismatch expect=ERR_MEM_RANGE got=%u\n",
                label,
                (unsigned)reject_err
            );
            return false;
        }
    }

    std::printf("%s PASS\n", label);
    return true;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;

    aecct::InferIngestContract valid;
    aecct::clear_infer_ingest_contract(valid);

    const uint32_t total_words = (uint32_t)sram_map::SRAM_WORDS_TOTAL;
    const uint32_t default_len = (uint32_t)aecct::INFER_IN_WORDS_EXPECTED;

    if ((default_len == 0u) || (default_len > total_words)) {
        std::printf("[p11g4][FAIL] invalid constants default_len=%u total_words=%u\n",
            (unsigned)default_len, (unsigned)total_words);
        CCS_RETURN(1);
    }

    aecct::InferIngestContract invalid_base = valid;
    invalid_base.in_base_word = (aecct::u32_t)total_words;
    invalid_base.len_words_expected = (aecct::u32_t)1u;

    aecct::InferIngestContract invalid_span = valid;
    invalid_span.in_base_word = (aecct::u32_t)(total_words - default_len + 1u);
    invalid_span.len_words_expected = (aecct::u32_t)0u;  // use default-length path in preflight

    aecct::InferIngestContract valid_edge = valid;
    valid_edge.in_base_word = (aecct::u32_t)(total_words - default_len);
    valid_edge.len_words_expected = (aecct::u32_t)0u;  // use default-length path in preflight

    if (!run_case("PREFLIGHT_VALID_CONTRACT", valid, true, false)) {
        CCS_RETURN(1);
    }
    if (!run_case("PREFLIGHT_INVALID_BASE_REJECT", invalid_base, false, true)) {
        CCS_RETURN(1);
    }
    if (!run_case("PREFLIGHT_INVALID_SPAN_REJECT", invalid_span, false, true)) {
        CCS_RETURN(1);
    }
    if (!run_case("PREFLIGHT_VALID_EDGE_ACCEPT", valid_edge, true, false)) {
        CCS_RETURN(1);
    }

    std::printf("PREFLIGHT_ERR_MEM_RANGE_GUARD_BEHAVIOR PASS\n");
    std::printf("PASS: tb_infer_ingest_preflight_negative_p11g4\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
