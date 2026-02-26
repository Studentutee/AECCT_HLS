#pragma once
// LayerNormDesc.h
// M8：LayerNormBlock 參數與 layout（single source of truth）

#include "ModelShapes.h"
#include "SramMap.h"

namespace aecct {

    // LN 向量維度與 token 數（layout: [token][d_model]）
    static const unsigned LN_TOKEN_COUNT = (unsigned)N_NODES;
    static const unsigned LN_D_MODEL = (unsigned)D_MODEL;
    static const unsigned LN_X_TOTAL_WORDS = (unsigned)(LN_TOKEN_COUNT * LN_D_MODEL);

    // PyTorch LayerNorm 預設 eps
    static constexpr float LN_EPS = 1.0e-5f;

    // M8 預設 SRAM 區域
    static const unsigned LN_X_IN_BASE_WORD_DEFAULT = (unsigned)sram_map::X_PAGE0_BASE_W;
    static const unsigned LN_X_OUT_BASE_WORD_DEFAULT = (unsigned)sram_map::X_PAGE1_BASE_W;
    static const unsigned LN_GAMMA_BASE_WORD_DEFAULT = (unsigned)sram_map::W_REGION_BASE;
    static const unsigned LN_BETA_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + LN_D_MODEL);

} // namespace aecct
