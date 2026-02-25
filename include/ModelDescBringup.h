#pragma once
// ModelDescBringup.h
// M3 bring-up 用的 cfg layout 定義（single source of truth）
// Top 與 TB 必須共用此檔，不可各自硬編。

namespace aecct {

    enum CfgWordIndex : unsigned {
        CFG_IDX_MAGIC = 0,
        CFG_IDX_CODE_N = 1,
        CFG_IDX_CODE_C = 2,
        CFG_IDX_D_MODEL = 3,
        CFG_IDX_N_HEADS = 4,
        CFG_IDX_D_HEAD = 5,
        CFG_IDX_D_FFN = 6,
        CFG_IDX_D_LPE = 7,
        CFG_IDX_N_LAYERS = 8,
        CFG_IDX_OUT_LEN_X_PRED = 9,
        CFG_IDX_OUT_LEN_LOGITS = 10,
        CFG_WORDS_EXPECTED = 11
    };

} // namespace aecct
