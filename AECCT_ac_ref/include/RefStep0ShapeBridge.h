#pragma once

#include "ModelShapes.h"
#include "SramMap.h"

namespace ModelShapes {

static const int T_TOKENS = static_cast<int>(N_NODES);
static const int D_MODEL = static_cast<int>(::D_MODEL);
static const int OUT_DIM = static_cast<int>(ELEMS_LOGITS);
static const int N_LAYERS = static_cast<int>(::N_LAYERS);
static const int N_VARS = static_cast<int>(ELEMS_X_PRED);
static const int N_HEADS = static_cast<int>(::N_HEAD);
static const int D_HEAD = static_cast<int>(::D_HEAD);
static const int D_FFN = static_cast<int>(::D_FFN);

static const int ALIGN_WORDS = static_cast<int>(::ALIGN_WORDS);
static const int PAGE_WORDS = static_cast<int>(align_up_words(static_cast<uint32_t>(T_TOKENS * D_MODEL), static_cast<uint32_t>(ALIGN_WORDS)));

static const int X_REGION_BASE = static_cast<int>(sram_map::X_PAGE0_BASE_W);
static const int PAGE0_BASE = static_cast<int>(sram_map::X_PAGE0_BASE_W);
static const int PAGE1_BASE = static_cast<int>(sram_map::X_PAGE1_BASE_W);

static const int SCRATCH_BASE = static_cast<int>(sram_map::BASE_SCRATCH_W);
static const int SCRATCH_WORDS = static_cast<int>(sram_map::SIZE_SCRATCH_W);

static const int SCR_K_BASE = static_cast<int>(sram_map::BASE_SCR_K_W);
static const int SCR_K_WORDS = static_cast<int>(sram_map::SIZE_SCR_K_W);
static const int SCR_V_BASE = static_cast<int>(sram_map::BASE_SCR_V_W);
static const int SCR_V_WORDS = static_cast<int>(sram_map::SIZE_SCR_V_W);

static const int SCR_FINAL_SCALAR_WORDS = static_cast<int>(align_up_words(static_cast<uint32_t>(T_TOKENS), static_cast<uint32_t>(ALIGN_WORDS)));
static const int SCR_FINAL_SCALAR_BASE = static_cast<int>(sram_map::BASE_SCR_K_W);

static const int XPRED_WORDS = (N_VARS + 31) / 32;

static const bool HAS_CUSTOM_VAR_TO_CLASS_MAP = false;
static_assert((OUT_DIM == N_VARS) || HAS_CUSTOM_VAR_TO_CLASS_MAP,
  "Provide SSOT map_var_to_class when OUT_DIM != N_VARS");

static inline int map_var_to_class(int var_idx) {
  // SSOT mapping for current model (identity).
  return var_idx;
}

} // namespace ModelShapes
