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
static const int STORAGE_ALIGN_WORDS = static_cast<int>(::SRAM_WORDS_PER_BEAT);

// Canonical single-workspace bridge constants.
// Legacy *_WORDS values below remain in aggregate bridge words so existing
// synth/ref helpers do not silently resize fp32 buffers in this patch.
// New *_STORAGE_WORDS / *_BASE_WORD16 values expose v12.1 storage units.
static const int X_WORK_WORDS = static_cast<int>(align_up_words(static_cast<uint32_t>(T_TOKENS * D_MODEL), static_cast<uint32_t>(ALIGN_WORDS)));
static const int X_WORK_BASE = static_cast<int>(sram_map::X_WORK_BASE_W);
static const int X_WORK_STORAGE_WORDS = static_cast<int>(sram_map::SIZE_X_WORK_WORD16);
static const int X_WORK_BASE_WORD16 = static_cast<int>(sram_map::BASE_X_WORK_WORD16);

// Legacy compatibility aliases.
// PAGE1_BASE is a virtual alias only and must not drive real X storage sizing
// or swap-based scheduling semantics.
static const int PAGE_WORDS = X_WORK_WORDS;
static const int PAGE0_BASE = X_WORK_BASE;
static const int PAGE1_BASE = X_WORK_BASE + X_WORK_WORDS;
static const int X_REGION_BASE = X_WORK_BASE;

static const int SCRATCH_BASE = static_cast<int>(sram_map::BASE_SCRATCH_W);
static const int SCRATCH_WORDS = static_cast<int>(sram_map::SIZE_SCRATCH_W);
static const int SCRATCH_BASE_WORD16 = static_cast<int>(sram_map::BASE_SCRATCH_WORD16);
static const int SCRATCH_STORAGE_WORDS = static_cast<int>(sram_map::SIZE_SCRATCH_WORD16);

static const int SCR_K_BASE = static_cast<int>(sram_map::BASE_SCR_K_W);
static const int SCR_K_WORDS = static_cast<int>(sram_map::SIZE_SCR_K_W);
static const int SCR_K_BASE_WORD16 = static_cast<int>(sram_map::BASE_SCR_K_WORD16);
static const int SCR_K_STORAGE_WORDS = static_cast<int>(sram_map::SIZE_SCR_K_WORD16);
static const int SCR_V_BASE = static_cast<int>(sram_map::BASE_SCR_V_W);
static const int SCR_V_WORDS = static_cast<int>(sram_map::SIZE_SCR_V_W);
static const int SCR_V_BASE_WORD16 = static_cast<int>(sram_map::BASE_SCR_V_WORD16);
static const int SCR_V_STORAGE_WORDS = static_cast<int>(sram_map::SIZE_SCR_V_WORD16);

static const int SCR_FINAL_SCALAR_WORDS = static_cast<int>(sram_map::SCR_FINAL_SCALAR_WORDS);
static const int SCR_FINAL_SCALAR_BASE = static_cast<int>(sram_map::SCR_FINAL_SCALAR_BASE);
static const int SCR_FINAL_SCALAR_STORAGE_WORDS = static_cast<int>(sram_map::SCR_FINAL_SCALAR_WORDS_WORD16);
static const int SCR_FINAL_SCALAR_BASE_WORD16 = static_cast<int>(sram_map::SCR_FINAL_SCALAR_BASE_WORD16);

static const int XPRED_WORDS = (N_VARS + 31) / 32;
static const int XPRED_STORAGE_WORDS = static_cast<int>(storage_words_bits(static_cast<uint32_t>(N_VARS)));
static const int LOGITS_STORAGE_WORDS = static_cast<int>(storage_words_fp32(static_cast<uint32_t>(OUT_DIM)));

static const bool HAS_CUSTOM_VAR_TO_CLASS_MAP = false;
static_assert((OUT_DIM == N_VARS) || HAS_CUSTOM_VAR_TO_CLASS_MAP,
  "Provide SSOT map_var_to_class when OUT_DIM != N_VARS");

static inline int map_var_to_class(int var_idx) {
  // SSOT mapping for current model (identity).
  return var_idx;
}

} // namespace ModelShapes
