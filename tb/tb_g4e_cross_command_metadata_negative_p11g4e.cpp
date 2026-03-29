// P00-G4E-META-NEG: targeted cross-command metadata mismatch validation (local-only).
// Scope:
// - Validate harmonized metadata helper rejects owner/rx mismatch.
// - Validate span preflight rejects out-of-range metadata.
// - Validate fallback expected-length path remains deterministic.

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

static bool require_true(bool cond, const char* fail_label, const char* pass_label) {
    if (!cond) {
        std::printf("[p11g4e][FAIL] %s\n", fail_label);
        return false;
    }
    std::printf("%s PASS\n", pass_label);
    return true;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;

    const aecct::IngestMetadataSurface cfg_meta =
        aecct::make_ingest_metadata_surface((aecct::u32_t)aecct::OP_CFG_BEGIN, (aecct::u32_t)0u,
            (aecct::u32_t)aecct::CFG_WORDS_EXPECTED, (aecct::u32_t)0u, true);
    const aecct::IngestMetadataSurface infer_meta =
        aecct::make_ingest_metadata_surface((aecct::u32_t)aecct::OP_INFER, (aecct::u32_t)aecct::IN_BASE_WORD,
            (aecct::u32_t)aecct::INFER_IN_WORDS_EXPECTED, (aecct::u32_t)0u, true);

    const uint32_t total_words = (uint32_t)sram_map::SRAM_WORDS_TOTAL;
    const aecct::IngestMetadataSurface bad_span_meta =
        aecct::make_ingest_metadata_surface((aecct::u32_t)aecct::OP_LOAD_W,
            (aecct::u32_t)(total_words - 1u), (aecct::u32_t)8u, (aecct::u32_t)0u, true);
    const aecct::IngestMetadataSurface fallback_len_meta =
        aecct::make_ingest_metadata_surface((aecct::u32_t)aecct::OP_LOAD_W,
            (aecct::u32_t)0u, (aecct::u32_t)0u, (aecct::u32_t)0u, true);

    if (!require_true(
            aecct::ingest_meta_owner_matches_rx(cfg_meta, aecct::RX_CFG),
            "cfg owner/rx pair expected to match",
            "G4E_OWNER_CFG_RX_MATCH")) {
        CCS_RETURN(1);
    }

    if (!require_true(
            !aecct::ingest_meta_owner_matches_rx(cfg_meta, aecct::RX_PARAM),
            "cfg metadata owner/rx mismatch was not rejected",
            "G4E_OWNER_CFG_RX_MISMATCH_REJECT")) {
        CCS_RETURN(1);
    }

    if (!require_true(
            aecct::ingest_meta_owner_matches_rx(infer_meta, aecct::RX_INFER),
            "infer owner/rx pair expected to match",
            "G4E_OWNER_INFER_RX_MATCH")) {
        CCS_RETURN(1);
    }

    if (!require_true(
            !aecct::ingest_meta_span_in_sram(bad_span_meta, 8u),
            "out-of-range metadata span was not rejected",
            "G4E_SPAN_OUT_OF_RANGE_REJECT")) {
        CCS_RETURN(1);
    }

    if (!require_true(
            aecct::ingest_meta_expected_words(fallback_len_meta, 16u) == 16u,
            "expected-length fallback mismatch",
            "G4E_EXPECTED_WORDS_FALLBACK")) {
        CCS_RETURN(1);
    }

    std::printf("PASS: tb_g4e_cross_command_metadata_negative_p11g4e\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
