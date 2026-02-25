#pragma once
// SramMapBringup.h
// M2 bring-up 用的簡化 SRAM 記憶體映射（word-address）
// 後續 M3/M4 可由正式 SramMap.h 取代。

namespace aecct {

    // 四個 logical regions（單位：u32 words）
    enum SramMapBringupConst : unsigned {
        X0_BASE_WORD = 0,
        X0_WORDS = 256,

        X1_BASE_WORD = X0_BASE_WORD + X0_WORDS,
        X1_WORDS = 256,

        SCR_BASE_WORD = X1_BASE_WORD + X1_WORDS,
        SCR_WORDS = 256,

        W_BASE_WORD = SCR_BASE_WORD + SCR_WORDS,
        W_WORDS = 256,

        SRAM_TOTAL_WORDS = W_BASE_WORD + W_WORDS,

        // M2 reset 只初始化每個 region 前段，避免大範圍清寫
        INIT_WORDS = 64
    };

} // namespace aecct
