#pragma once
// PreprocDescBringup.h
// M7 preproc shape/layout single source of truth

#include "gen/ModelShapes.h"
#include "gen/SramMap.h"

namespace aecct {

    // INFER payload words (input y)
    static const unsigned PREPROC_IN_WORDS_EXPECTED = (unsigned)EXP_LEN_INFER_IN_WORDS;

    // Preproc output X words, layout: [token][d_model]
    static const unsigned PREPROC_X_OUT_WORDS_EXPECTED = (unsigned)WORDS_X_FP32;
    static const unsigned PREPROC_X_TOKEN_STRIDE_WORDS = (unsigned)D_MODEL;

    // Default SRAM locations (word address)
    static const unsigned PREPROC_IN_BASE_WORD_DEFAULT = (unsigned)sram_map::BASE_SCRATCH_W;
    static const unsigned PREPROC_X_OUT_BASE_WORD_DEFAULT = (unsigned)sram_map::X_PAGE0_BASE_W;

} // namespace aecct
