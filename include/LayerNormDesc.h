#pragma once
// LayerNormDesc.h
// M8 layernorm shape/layout single source of truth

#include "gen/ModelShapes.h"
#include "gen/SramMap.h"

namespace aecct {

    // LN tensor layout: [token][d_model]
    static const unsigned LN_TOKEN_COUNT = (unsigned)N_NODES;
    static const unsigned LN_D_MODEL = (unsigned)D_MODEL;
    static const unsigned LN_X_TOTAL_WORDS = (unsigned)(LN_TOKEN_COUNT * LN_D_MODEL);

    // LayerNorm epsilon
    static constexpr float LN_EPS = 1.0e-5f;

    // Default SRAM locations (word address)
    static const unsigned LN_X_IN_BASE_WORD_DEFAULT = (unsigned)sram_map::X_PAGE0_BASE_W;
    static const unsigned LN_X_OUT_BASE_WORD_DEFAULT = (unsigned)sram_map::X_PAGE1_BASE_W;
    static const unsigned LN_GAMMA_BASE_WORD_DEFAULT = (unsigned)sram_map::W_REGION_BASE;
    static const unsigned LN_BETA_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + LN_D_MODEL);

} // namespace aecct
