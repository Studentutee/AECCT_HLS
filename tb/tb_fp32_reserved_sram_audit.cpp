#ifndef __SYNTHESIS__

#include <cstdint>
#include <cstdio>

#include "gen/ModelShapes.h"
#include "gen/SramMap.h"

namespace {

struct AuditRow {
    const char* name;
    uint32_t payload_elems;
    uint32_t current_storage_word16;
    uint32_t shadow_storage_word16;
    uint32_t delta_storage_word16;
    bool is_reserved_only;
};

static uint32_t align_up_storage_words_local(uint32_t x) {
    return align_up_storage_words(x, ALIGN_STORAGE_WORD16);
}

static void print_row(const AuditRow& row) {
    std::printf(
        "[fp32_audit] section=%s elems=%u current_word16=%u shadow_word16=%u delta=%u reserved_only=%u\n",
        row.name,
        (unsigned)row.payload_elems,
        (unsigned)row.current_storage_word16,
        (unsigned)row.shadow_storage_word16,
        (unsigned)row.delta_storage_word16,
        row.is_reserved_only ? 1u : 0u);
}

} // namespace

int main() {
    const AuditRow rows[] = {
        {"X_PAGE0_ACTIVE", (uint32_t)ELEMS_X, (uint32_t)sram_map::X_PAGE0_WORDS_WORD16,
            align_up_storage_words_local((uint32_t)ELEMS_X),
            (uint32_t)sram_map::X_PAGE0_WORDS_WORD16 - align_up_storage_words_local((uint32_t)ELEMS_X), false},
        {"X_PAGE1_COMPAT", (uint32_t)ELEMS_X, (uint32_t)sram_map::X_PAGE1_WORDS_WORD16,
            align_up_storage_words_local((uint32_t)ELEMS_X),
            (uint32_t)sram_map::X_PAGE1_WORDS_WORD16 - align_up_storage_words_local((uint32_t)ELEMS_X), true},
        {"SCR_K", (uint32_t)ELEMS_X, (uint32_t)sram_map::SIZE_SCR_K_WORD16,
            align_up_storage_words_local((uint32_t)ELEMS_X),
            (uint32_t)sram_map::SIZE_SCR_K_WORD16 - align_up_storage_words_local((uint32_t)ELEMS_X), false},
        {"SCR_V", (uint32_t)ELEMS_X, (uint32_t)sram_map::SIZE_SCR_V_WORD16,
            align_up_storage_words_local((uint32_t)ELEMS_X),
            (uint32_t)sram_map::SIZE_SCR_V_WORD16 - align_up_storage_words_local((uint32_t)ELEMS_X), false},
        {"FINAL_SCALAR_BUF", (uint32_t)N_NODES, (uint32_t)sram_map::FINAL_SCALAR_BUF_WORDS_WORD16,
            align_up_storage_words_local((uint32_t)N_NODES),
            (uint32_t)sram_map::FINAL_SCALAR_BUF_WORDS_WORD16 - align_up_storage_words_local((uint32_t)N_NODES), false},
    };

    uint32_t total_current = 0u;
    uint32_t total_shadow = 0u;
    uint32_t total_delta = 0u;
    uint32_t total_current_no_reserved = 0u;
    uint32_t total_shadow_no_reserved = 0u;
    uint32_t total_delta_no_reserved = 0u;

    ROW_LOOP: for (uint32_t i = 0u; i < (uint32_t)(sizeof(rows) / sizeof(rows[0])); ++i) {
        print_row(rows[i]);
        total_current += rows[i].current_storage_word16;
        total_shadow += rows[i].shadow_storage_word16;
        total_delta += rows[i].delta_storage_word16;
        if (!rows[i].is_reserved_only) {
            total_current_no_reserved += rows[i].current_storage_word16;
            total_shadow_no_reserved += rows[i].shadow_storage_word16;
            total_delta_no_reserved += rows[i].delta_storage_word16;
        }
    }

    const uint32_t logical_x_work_shadow_keep_both_pages =
        align_up_storage_words_local((uint32_t)ELEMS_X) * 2u;
    const uint32_t logical_x_work_shadow_single_page =
        align_up_storage_words_local((uint32_t)ELEMS_X);

    std::printf(
        "[fp32_audit] logical_x_work current_word16=%u shadow_keep_compat_pages=%u shadow_single_page=%u\n",
        (unsigned)sram_map::SIZE_X_WORK_WORD16,
        (unsigned)logical_x_work_shadow_keep_both_pages,
        (unsigned)logical_x_work_shadow_single_page);

    std::printf(
        "[fp32_audit][SUMMARY] total_current_word16=%u total_shadow_word16=%u total_delta=%u\n",
        (unsigned)total_current,
        (unsigned)total_shadow,
        (unsigned)total_delta);
    std::printf(
        "[fp32_audit][SUMMARY_NO_RESERVED] total_current_word16=%u total_shadow_word16=%u total_delta=%u\n",
        (unsigned)total_current_no_reserved,
        (unsigned)total_shadow_no_reserved,
        (unsigned)total_delta_no_reserved);

    if (total_shadow >= total_current) {
        std::printf("[fp32_audit][FAIL] shadow projection did not reduce storage\n");
        return 1;
    }
    if (total_shadow_no_reserved >= total_current_no_reserved) {
        std::printf("[fp32_audit][FAIL] no-reserved shadow projection did not reduce storage\n");
        return 1;
    }

    std::printf("PASS: tb_fp32_reserved_sram_audit\n");
    return 0;
}

#endif // __SYNTHESIS__
