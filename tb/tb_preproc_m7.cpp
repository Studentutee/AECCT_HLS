// tb_preproc_m7.cpp
// M7 block TB：直接驗證 PreprocEmbedSPE 的 SRAM checkpoint

#include <cstdio>
#include <cstdint>

#include "AecctTypes.h"
#include "SramMap.h"
#include "PreprocDescBringup.h"
#include "blocks/PreprocEmbedSPE.h"

static int run_case(uint32_t sample_idx) {
    aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
    for (uint32_t i = 0; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = (aecct::u32_t)0u;
    }

    if (sample_idx >= aecct::preproc_trace_samples()) {
        std::printf("ERROR: sample_idx out of range. idx=%u samples=%u\n",
            (unsigned)sample_idx, (unsigned)aecct::preproc_trace_samples());
        return 1;
    }

    for (uint32_t i = 0; i < (uint32_t)aecct::PREPROC_IN_WORDS_EXPECTED; ++i) {
        sram[aecct::PREPROC_IN_BASE_WORD_DEFAULT + i] =
            (aecct::u32_t)aecct::preproc_trace_input_word(sample_idx, i);
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

    for (uint32_t j = 0; j < (uint32_t)aecct::PREPROC_X_OUT_WORDS_EXPECTED; ++j) {
        uint32_t got = (uint32_t)sram[aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT + j].to_uint();
        uint32_t ref = aecct::preproc_trace_x_word(sample_idx, j);
        if (got != ref) {
            std::printf("ERROR: checkpoint mismatch at word=%u got=0x%08X ref=0x%08X\n",
                (unsigned)j, (unsigned)got, (unsigned)ref);
            return 1;
        }
    }

    return 0;
}

int main() {
    if (run_case(0u) != 0) {
        return 1;
    }
    std::printf("PASS: tb_preproc_m7 (SRAM checkpoint match trace)\n");
    return 0;
}
