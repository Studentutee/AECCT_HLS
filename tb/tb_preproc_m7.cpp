// tb_preproc_m7.cpp
// M7 block TB嚗?仿?霅?PreprocEmbedSPE ??SRAM checkpoint

#include <cstdio>
#include <cstdint>

#include "AecctTypes.h"
#include "gen/SramMap.h"
#include "PreprocDescBringup.h"
#include "blocks/PreprocEmbedSPE.h"

static uint32_t make_input_pattern(uint32_t idx) {
    // Deterministic non-trivial pattern for local copy/zero boundary checks.
    return 0x3F000000u ^ (idx * 0x9E3779B9u);
}

static int run_case() {
    aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
    for (uint32_t i = 0; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = (aecct::u32_t)0xDEADBEEFu;
    }

    const uint32_t in_words = (uint32_t)aecct::PREPROC_IN_WORDS_EXPECTED;
    const uint32_t x_words = (uint32_t)aecct::PREPROC_X_OUT_WORDS_EXPECTED;

    for (uint32_t i = 0; i < in_words; ++i) {
        sram[aecct::PREPROC_IN_BASE_WORD_DEFAULT + i] =
            (aecct::u32_t)make_input_pattern(i);
    }

    aecct::PreprocCfg cfg;
    cfg.infer_in_words = (aecct::u32_t)aecct::PREPROC_IN_WORDS_EXPECTED;
    cfg.x_out_words = (aecct::u32_t)aecct::PREPROC_X_OUT_WORDS_EXPECTED;

    aecct::PreprocEmbedSPE(
        sram,
        cfg,
        (aecct::u32_t)aecct::PREPROC_IN_BASE_WORD_DEFAULT,
        (aecct::u32_t)aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT
    );

    for (uint32_t j = 0; j < x_words; ++j) {
        uint32_t got = (uint32_t)sram[aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT + j].to_uint();
        uint32_t ref = (j < in_words) ? make_input_pattern(j) : 0u;
        if (got != ref) {
            std::printf("ERROR: checkpoint mismatch at word=%u got=0x%08X ref=0x%08X\n",
                (unsigned)j, (unsigned)got, (unsigned)ref);
            return 1;
        }
    }

    return 0;
}

int main() {
    if (run_case() != 0) {
        return 1;
    }
    std::printf("PASS: tb_preproc_m7 (SRAM checkpoint match trace)\n");
    return 0;
}

