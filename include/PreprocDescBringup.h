#pragma once
// PreprocDescBringup.h
// M7 bring-up：PreprocEmbedSPE 需要的長度與 layout（single source of truth）

#include "ModelShapes.h"
#include "SramMap.h"

namespace aecct {

    // INFER 輸入 y 長度（words）
    static const unsigned PREPROC_IN_WORDS_EXPECTED = (unsigned)EXP_LEN_INFER_IN_WORDS;

    // Preproc 輸出 X 長度（words），layout = [token_idx][d_model] 線性展開
    static const unsigned PREPROC_X_OUT_WORDS_EXPECTED = (unsigned)WORDS_X_FP32;
    static const unsigned PREPROC_X_TOKEN_STRIDE_WORDS = (unsigned)D_MODEL;

    // 預設 SRAM 基底位址（word address）
    static const unsigned PREPROC_IN_BASE_WORD_DEFAULT = (unsigned)sram_map::BASE_SCRATCH_W;
    static const unsigned PREPROC_X_OUT_BASE_WORD_DEFAULT = (unsigned)sram_map::X_PAGE0_BASE_W;

} // namespace aecct
