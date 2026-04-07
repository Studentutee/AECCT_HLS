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
// Runtime LN gamma/beta are staging buffers loaded from PARAM by Top.
// Keep them outside W_REGION/PARAM to avoid runtime overwrite on persistent data.
static const unsigned LN_RUNTIME_STAGE_WORDS = (unsigned)align_up_words((uint32_t)(LN_D_MODEL * 2u), (uint32_t)ALIGN_WORDS);
static const unsigned LN_RUNTIME_STAGE_BASE_WORD_DEFAULT =
    (unsigned)(sram_map::BACKUP_RUNTIME_SCRATCH_BASE_W + sram_map::BACKUP_RUNTIME_SCRATCH_WORDS - LN_RUNTIME_STAGE_WORDS);
static const unsigned LN_GAMMA_BASE_WORD_DEFAULT = (unsigned)LN_RUNTIME_STAGE_BASE_WORD_DEFAULT;
static const unsigned LN_BETA_BASE_WORD_DEFAULT = (unsigned)(LN_RUNTIME_STAGE_BASE_WORD_DEFAULT + LN_D_MODEL);

static_assert(
    (uint32_t)LN_RUNTIME_STAGE_BASE_WORD_DEFAULT >=
        ((uint32_t)sram_map::W_REGION_BASE + (uint32_t)sram_map::W_REGION_WORDS),
    "LayerNorm runtime staging base must not overlap W_REGION/PARAM");
static_assert(
    ((uint32_t)LN_BETA_BASE_WORD_DEFAULT + (uint32_t)LN_D_MODEL) <=
        ((uint32_t)sram_map::BACKUP_RUNTIME_SCRATCH_BASE_W + (uint32_t)sram_map::BACKUP_RUNTIME_SCRATCH_WORDS),
    "LayerNorm runtime staging window exceeds backup runtime scratch region");

} // namespace aecct
