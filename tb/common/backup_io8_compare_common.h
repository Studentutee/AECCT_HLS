#ifndef TB_COMMON_BACKUP_IO8_COMPARE_COMMON_H_
#define TB_COMMON_BACKUP_IO8_COMPARE_COMMON_H_

#include <cstdint>
#include <cstdio>

namespace backup_io8_compare_common {

struct DualCompareSummary {
    uint32_t old_baseline_mismatch_count;
    uint32_t dut_aligned_mismatch_count;
};

static inline DualCompareSummary make_dual_compare_summary() {
    DualCompareSummary s;
    s.old_baseline_mismatch_count = 0u;
    s.dut_aligned_mismatch_count = 0u;
    return s;
}

static inline void dual_compare_update(
    DualCompareSummary& s,
    bool old_baseline_match,
    bool dut_aligned_match
) {
    if (!old_baseline_match) {
        s.old_baseline_mismatch_count += 1u;
    }
    if (!dut_aligned_match) {
        s.dut_aligned_mismatch_count += 1u;
    }
}

static inline bool dual_compare_old_exact(const DualCompareSummary& s) {
    return s.old_baseline_mismatch_count == 0u;
}

static inline bool dual_compare_aligned_exact(const DualCompareSummary& s) {
    return s.dut_aligned_mismatch_count == 0u;
}

static inline void print_dual_compare_summary(
    const char* tag,
    uint32_t sample_id,
    const DualCompareSummary& s
) {
    std::printf(
        "[backup_io8][%s] sample=%u old_baseline_mismatch=%u dut_aligned_mismatch=%u old_baseline_exact=%u dut_aligned_exact=%u\n",
        tag,
        (unsigned)sample_id,
        (unsigned)s.old_baseline_mismatch_count,
        (unsigned)s.dut_aligned_mismatch_count,
        (unsigned)(dual_compare_old_exact(s) ? 1u : 0u),
        (unsigned)(dual_compare_aligned_exact(s) ? 1u : 0u)
    );
}

static inline void print_fixed_sample_summary(
    uint32_t sample_id,
    uint32_t boundary_bucket,
    uint32_t end_norm_old_mismatch,
    uint32_t end_norm_dut_aligned_mismatch,
    uint32_t st_old_mismatch,
    uint32_t st_dut_aligned_mismatch,
    uint32_t logit_old_mismatch,
    uint32_t logit_dut_aligned_mismatch,
    bool end_norm_exact,
    bool st_exact,
    bool logit_exact,
    bool xpred_exact
) {
    std::printf(
        "[backup_io8][fixed_sample_summary] sample=%u boundary_bucket=%u end_norm_old_mismatch=%u end_norm_dut_aligned_mismatch=%u st_old_mismatch=%u st_dut_aligned_mismatch=%u logit_old_mismatch=%u logit_dut_aligned_mismatch=%u end_norm_exact=%u st_exact=%u logit_exact=%u xpred_exact=%u\n",
        (unsigned)sample_id,
        (unsigned)boundary_bucket,
        (unsigned)end_norm_old_mismatch,
        (unsigned)end_norm_dut_aligned_mismatch,
        (unsigned)st_old_mismatch,
        (unsigned)st_dut_aligned_mismatch,
        (unsigned)logit_old_mismatch,
        (unsigned)logit_dut_aligned_mismatch,
        (unsigned)(end_norm_exact ? 1u : 0u),
        (unsigned)(st_exact ? 1u : 0u),
        (unsigned)(logit_exact ? 1u : 0u),
        (unsigned)(xpred_exact ? 1u : 0u)
    );
}

} // namespace backup_io8_compare_common

#endif // TB_COMMON_BACKUP_IO8_COMPARE_COMMON_H_
