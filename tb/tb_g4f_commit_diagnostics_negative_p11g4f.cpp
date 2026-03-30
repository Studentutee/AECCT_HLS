// P00-G4F-COMMIT-DIAG-NEG: targeted commit-time diagnostics mismatch validation (local-only).
// Scope:
// - Validate commit-time diagnostics helper owner/rx mismatch mapping.
// - Validate length mismatch mapping for CFG and PARAM.
// - Validate span mismatch mapping and deterministic acceptance path.

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

static bool expect_u8(uint8_t got, uint8_t exp, const char* pass_label, const char* fail_label) {
    if (got != exp) {
        std::printf("[p11g4f][FAIL] %s got=%u exp=%u\n", fail_label, (unsigned)got, (unsigned)exp);
        return false;
    }
    std::printf("%s PASS\n", pass_label);
    return true;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;

    const aecct::IngestMetadataSurface cfg_len_mismatch =
        aecct::make_ingest_metadata_surface(
            (aecct::u32_t)aecct::OP_CFG_BEGIN,
            (aecct::u32_t)0u,
            (aecct::u32_t)aecct::CFG_WORDS_EXPECTED,
            (aecct::u32_t)(aecct::CFG_WORDS_EXPECTED - 1u),
            true
        );
    const uint8_t cfg_diag = aecct::ingest_commit_diag_error(
        cfg_len_mismatch,
        aecct::RX_CFG,
        (uint32_t)aecct::CFG_WORDS_EXPECTED,
        (uint8_t)aecct::ERR_CFG_LEN_MISMATCH,
        false
    );
    if (!expect_u8(cfg_diag, (uint8_t)aecct::ERR_CFG_LEN_MISMATCH,
            "G4F_CFG_LEN_MISMATCH_MAPPING", "cfg len mismatch mapping")) {
        CCS_RETURN(1);
    }

    const aecct::IngestMetadataSurface param_len_mismatch =
        aecct::make_ingest_metadata_surface(
            (aecct::u32_t)aecct::OP_LOAD_W,
            (aecct::u32_t)sram_map::W_REGION_BASE,
            (aecct::u32_t)aecct::PARAM_WORDS_EXPECTED,
            (aecct::u32_t)(aecct::PARAM_WORDS_EXPECTED - 3u),
            true
        );
    const uint8_t param_diag = aecct::ingest_commit_diag_error(
        param_len_mismatch,
        aecct::RX_PARAM,
        (uint32_t)aecct::PARAM_WORDS_EXPECTED,
        (uint8_t)aecct::ERR_PARAM_LEN_MISMATCH,
        true
    );
    if (!expect_u8(param_diag, (uint8_t)aecct::ERR_PARAM_LEN_MISMATCH,
            "G4F_PARAM_LEN_MISMATCH_MAPPING", "param len mismatch mapping")) {
        CCS_RETURN(1);
    }

    const aecct::IngestMetadataSurface infer_owner_mismatch =
        aecct::make_ingest_metadata_surface(
            (aecct::u32_t)aecct::OP_INFER,
            (aecct::u32_t)aecct::IN_BASE_WORD,
            (aecct::u32_t)aecct::INFER_IN_WORDS_EXPECTED,
            (aecct::u32_t)aecct::INFER_IN_WORDS_EXPECTED,
            true
        );
    const uint8_t infer_owner_diag = aecct::ingest_commit_diag_error(
        infer_owner_mismatch,
        aecct::RX_PARAM,
        (uint32_t)aecct::INFER_IN_WORDS_EXPECTED,
        (uint8_t)aecct::ERR_BAD_STATE,
        true
    );
    if (!expect_u8(infer_owner_diag, (uint8_t)aecct::ERR_BAD_STATE,
            "G4F_OWNER_RX_MISMATCH_MAPPING", "owner/rx mismatch mapping")) {
        CCS_RETURN(1);
    }

    const aecct::IngestMetadataSurface infer_span_mismatch =
        aecct::make_ingest_metadata_surface(
            (aecct::u32_t)aecct::OP_INFER,
            (aecct::u32_t)(sram_map::SRAM_WORDS_TOTAL - 1u),
            (aecct::u32_t)8u,
            (aecct::u32_t)8u,
            true
        );
    const uint8_t infer_span_diag = aecct::ingest_commit_diag_error(
        infer_span_mismatch,
        aecct::RX_INFER,
        (uint32_t)aecct::INFER_IN_WORDS_EXPECTED,
        (uint8_t)aecct::ERR_BAD_STATE,
        true
    );
    if (!expect_u8(infer_span_diag, (uint8_t)aecct::ERR_MEM_RANGE,
            "G4F_SPAN_MISMATCH_MAPPING", "span mismatch mapping")) {
        CCS_RETURN(1);
    }

    const aecct::IngestMetadataSurface accept_meta =
        aecct::make_ingest_metadata_surface(
            (aecct::u32_t)aecct::OP_LOAD_W,
            (aecct::u32_t)sram_map::W_REGION_BASE,
            (aecct::u32_t)aecct::PARAM_WORDS_EXPECTED,
            (aecct::u32_t)aecct::PARAM_WORDS_EXPECTED,
            true
        );
    const uint8_t accept_diag = aecct::ingest_commit_diag_error(
        accept_meta,
        aecct::RX_PARAM,
        (uint32_t)aecct::PARAM_WORDS_EXPECTED,
        (uint8_t)aecct::ERR_PARAM_LEN_MISMATCH,
        true
    );
    if (!expect_u8(accept_diag, (uint8_t)aecct::ERR_OK,
            "G4F_ACCEPTANCE_DETERMINISTIC", "acceptance deterministic mapping")) {
        CCS_RETURN(1);
    }

    std::printf("PASS: tb_g4f_commit_diagnostics_negative_p11g4f\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
