#pragma once
// LayerNorm shape/layout single source of truth.

#include "AecctTypes.h"
#include "gen/ModelShapes.h"
#include "gen/SramMap.h"

namespace aecct {

static const unsigned LN_TOKEN_COUNT = (unsigned)N_NODES;
static const unsigned LN_D_MODEL = (unsigned)D_MODEL;
static const unsigned LN_X_TOTAL_WORDS = (unsigned)(LN_TOKEN_COUNT * LN_D_MODEL);

// 1.0e-5 in IEEE754 binary32.
static const u32_t LN_EPS_BITS = (u32_t)0x3727C5ACu;

static const unsigned LN_X_IN_BASE_WORD_DEFAULT = (unsigned)sram_map::X_PAGE0_BASE_W;
static const unsigned LN_X_OUT_BASE_WORD_DEFAULT = (unsigned)sram_map::X_PAGE1_BASE_W;
static const unsigned LN_GAMMA_BASE_WORD_DEFAULT = (unsigned)sram_map::W_REGION_BASE;
static const unsigned LN_BETA_BASE_WORD_DEFAULT = (unsigned)(sram_map::W_REGION_BASE + LN_D_MODEL);

} // namespace aecct
