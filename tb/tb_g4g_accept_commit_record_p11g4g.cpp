// P00-G4G-ACCEPT-COMMIT-RECORD: targeted accepted-commit metadata record harmonization validation (local-only).
// Scope:
// - Validate accepted commit record is deterministic across CFG/PARAM/INFER metadata paths.
// - Validate reject path does not overwrite the previously accepted record (no stale-state pollution).

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
        std::printf("[p11g4g][FAIL] %s got=%u exp=%u\n", fail_label, (unsigned)got, (unsigned)exp);
        return false;
    }
    return true;
}

static bool expect_bool(bool got, bool exp, const char* fail_label) {
    if (got != exp) {
        std::printf("[p11g4g][FAIL] %s got=%d exp=%d\n", fail_label, got ? 1 : 0, exp ? 1 : 0);
        return false;
    }
    return true;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;

    aecct::AcceptedCommitMetadataRecord record =
        aecct::make_invalid_accepted_commit_metadata_record();

    const aecct::IngestMetadataSurface cfg_meta =
        aecct::make_ingest_metadata_surface(
            (aecct::u32_t)aecct::OP_CFG_BEGIN,
            (aecct::u32_t)0u,
            (aecct::u32_t)aecct::CFG_WORDS_EXPECTED,
            (aecct::u32_t)aecct::CFG_WORDS_EXPECTED,
            true
        );
    const uint8_t cfg_diag = aecct::ingest_commit_diag_and_record(
        record,
        cfg_meta,
        aecct::RX_CFG,
        (uint32_t)aecct::CFG_WORDS_EXPECTED,
        (uint8_t)aecct::ERR_CFG_LEN_MISMATCH,
        false,
        (aecct::u32_t)0u,
        false
    );
    if (!expect_u32((uint32_t)cfg_diag, (uint32_t)aecct::ERR_OK, "cfg diag should accept")) {
        CCS_RETURN(1);
    }
    if (!expect_bool(record.valid, true, "cfg accept should mark record valid") ||
        !expect_u32((uint32_t)record.owner_opcode.to_uint(), (uint32_t)aecct::OP_CFG_BEGIN, "cfg owner opcode") ||
        !expect_u32((uint32_t)record.rx_state.to_uint(), (uint32_t)aecct::RX_CFG, "cfg rx state") ||
        !expect_bool(record.phase_valid, false, "cfg phase_valid should be false")) {
        CCS_RETURN(1);
    }
    std::printf("G4G_ACCEPT_RECORD_CFG_DETERMINISTIC PASS\n");

    const aecct::IngestMetadataSurface param_meta =
        aecct::make_ingest_metadata_surface(
            (aecct::u32_t)aecct::OP_LOAD_W,
            (aecct::u32_t)sram_map::W_REGION_BASE,
            (aecct::u32_t)aecct::PARAM_WORDS_EXPECTED,
            (aecct::u32_t)aecct::PARAM_WORDS_EXPECTED,
            true
        );
    const uint8_t param_diag = aecct::ingest_commit_diag_and_record(
        record,
        param_meta,
        aecct::RX_PARAM,
        (uint32_t)aecct::PARAM_WORDS_EXPECTED,
        (uint8_t)aecct::ERR_PARAM_LEN_MISMATCH,
        true,
        (aecct::u32_t)0u,
        false
    );
    if (!expect_u32((uint32_t)param_diag, (uint32_t)aecct::ERR_OK, "param diag should accept")) {
        CCS_RETURN(1);
    }
    if (!expect_u32((uint32_t)record.owner_opcode.to_uint(), (uint32_t)aecct::OP_LOAD_W, "param owner opcode") ||
        !expect_u32((uint32_t)record.base_word.to_uint(), (uint32_t)sram_map::W_REGION_BASE, "param base") ||
        !expect_u32((uint32_t)record.rx_state.to_uint(), (uint32_t)aecct::RX_PARAM, "param rx state") ||
        !expect_bool(record.phase_valid, false, "param phase_valid should be false")) {
        CCS_RETURN(1);
    }
    std::printf("G4G_ACCEPT_RECORD_PARAM_DETERMINISTIC PASS\n");

    const uint32_t prev_owner = (uint32_t)record.owner_opcode.to_uint();
    const uint32_t prev_base = (uint32_t)record.base_word.to_uint();
    const uint32_t prev_rx = (uint32_t)record.rx_state.to_uint();
    const uint32_t prev_len_expected = (uint32_t)record.len_words_expected.to_uint();
    const uint32_t prev_len_valid = (uint32_t)record.len_words_valid.to_uint();
    const bool prev_phase_valid = record.phase_valid;

    const aecct::IngestMetadataSurface infer_bad_span_meta =
        aecct::make_ingest_metadata_surface(
            (aecct::u32_t)aecct::OP_INFER,
            (aecct::u32_t)((uint32_t)sram_map::SRAM_WORDS_TOTAL - 1u),
            (aecct::u32_t)8u,
            (aecct::u32_t)8u,
            true
        );
    const uint8_t infer_bad_diag = aecct::ingest_commit_diag_and_record(
        record,
        infer_bad_span_meta,
        aecct::RX_INFER,
        (uint32_t)aecct::INFER_IN_WORDS_EXPECTED,
        (uint8_t)aecct::ERR_BAD_STATE,
        true,
        (aecct::u32_t)aecct::PHASE_PREPROC,
        true
    );
    if (!expect_u32((uint32_t)infer_bad_diag, (uint32_t)aecct::ERR_MEM_RANGE, "infer bad span should reject")) {
        CCS_RETURN(1);
    }
    if (!expect_u32((uint32_t)record.owner_opcode.to_uint(), prev_owner, "reject must not overwrite owner") ||
        !expect_u32((uint32_t)record.base_word.to_uint(), prev_base, "reject must not overwrite base") ||
        !expect_u32((uint32_t)record.rx_state.to_uint(), prev_rx, "reject must not overwrite rx") ||
        !expect_u32((uint32_t)record.len_words_expected.to_uint(), prev_len_expected, "reject must not overwrite len expected") ||
        !expect_u32((uint32_t)record.len_words_valid.to_uint(), prev_len_valid, "reject must not overwrite len valid") ||
        !expect_bool(record.phase_valid, prev_phase_valid, "reject must not overwrite phase_valid")) {
        CCS_RETURN(1);
    }
    std::printf("G4G_REJECT_NO_STALE_STATE PASS\n");

    const aecct::IngestMetadataSurface infer_meta =
        aecct::make_ingest_metadata_surface(
            (aecct::u32_t)aecct::OP_INFER,
            (aecct::u32_t)aecct::IN_BASE_WORD,
            (aecct::u32_t)aecct::INFER_IN_WORDS_EXPECTED,
            (aecct::u32_t)aecct::INFER_IN_WORDS_EXPECTED,
            true
        );
    const uint8_t infer_diag = aecct::ingest_commit_diag_and_record(
        record,
        infer_meta,
        aecct::RX_INFER,
        (uint32_t)aecct::INFER_IN_WORDS_EXPECTED,
        (uint8_t)aecct::ERR_BAD_STATE,
        true,
        (aecct::u32_t)aecct::PHASE_PREPROC,
        true
    );
    if (!expect_u32((uint32_t)infer_diag, (uint32_t)aecct::ERR_OK, "infer diag should accept")) {
        CCS_RETURN(1);
    }
    if (!expect_u32((uint32_t)record.owner_opcode.to_uint(), (uint32_t)aecct::OP_INFER, "infer owner opcode") ||
        !expect_u32((uint32_t)record.rx_state.to_uint(), (uint32_t)aecct::RX_INFER, "infer rx state") ||
        !expect_bool(record.phase_valid, true, "infer phase_valid should be true") ||
        !expect_u32((uint32_t)record.phase_id.to_uint(), (uint32_t)aecct::PHASE_PREPROC, "infer phase_id")) {
        CCS_RETURN(1);
    }
    std::printf("G4G_ACCEPT_RECORD_INFER_PHASE_VALID PASS\n");

    std::printf("PASS: tb_g4g_accept_commit_record_p11g4g\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
