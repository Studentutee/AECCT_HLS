// SramMap.h
#pragma once
#include <cstdint>
#include "ModelShapes.h"

// ============================================================
// SramMap.h (legacy aggregate-word bridge + v12.1 storage-word helpers)
// ------------------------------------------------------------
// - Existing *_W symbols stay in legacy aggregate words to avoid a broad refactor.
// - New *_WORD16 / *_STORAGE_* helpers expose v12.1 semantics:
//     * SRAM word = 16 bits
//     * SRAM beat = 8 words = 128 bits
//     * base_word / len_words for the new profile should use 16-bit storage words.
//
// Step2 convergence note:
// - Baseline storage semantics are single-X_WORK:
//     * W_REGION, X_WORK, SCRATCH, IO_REGION
// - Legacy dual-page names (X_PAGE0/X_PAGE1) are kept only as compatibility
//   aliases for existing bring-up tests and transitional code paths.
// - Physical implementation MAY still map to one flat SRAM word space.
//
// Notes:
// - X_WORK is the only baseline shared working area.
// - SCR_K / SCR_V / FINAL_SCALAR are SCRATCH sub-regions.
// - [compat] X_PAGE0 / X_PAGE1 names remain aliases; they are not a separate
//   baseline taxonomy in this step.
// - No dedicated DEBUG SRAM region (D1 debug is "halt + READ_MEM").
// - [legacy] Separate BIAS/WEIGHT regions are still exposed for compatibility.
// ============================================================

static const uint32_t SRAM_WORD_BYTES = BYTES_PER_WORD;
static const uint32_t SRAM_WORD_LANES = W_LANES;
static_assert(SRAM_WORD_BYTES == 16u, "Legacy aggregate beat must remain 16 bytes during bridge.");
static_assert(SRAM_WORD_LANES == 8u, "Legacy aggregate beat must remain 8 lanes during bridge.");

static const uint32_t SRAM_STORAGE_WORD_BITS = ::SRAM_STORAGE_WORD_BITS;
static const uint32_t SRAM_STORAGE_WORD_BYTES = ::SRAM_STORAGE_WORD_BYTES;
static const uint32_t SRAM_STORAGE_WORDS_PER_BEAT = ::SRAM_WORDS_PER_BEAT;
static const uint32_t SRAM_BEAT_BITS = ::SRAM_BEAT_BITS;
static const uint32_t SRAM_STORAGE_WORDS_PER_LEGACY_WORD =
    (LEGACY_U32_WORD_BITS / SRAM_STORAGE_WORD_BITS);
static_assert(SRAM_STORAGE_WORD_BITS == 16u, "v12.1 storage word must be 16 bits");
static_assert(SRAM_STORAGE_WORDS_PER_BEAT == 8u, "v12.1 beat must contain 8 storage words");
static_assert(SRAM_BEAT_BITS == 128u, "v12.1 beat must be 128 bits");
static_assert(SRAM_STORAGE_WORDS_PER_LEGACY_WORD == 2u,
              "Legacy u32 word must map to two 16-bit storage words.");

// Alignment unit remains legacy aggregate-word addressed in the old map.
static const uint32_t ALIGN_WORDS = 16u;
static_assert((ALIGN_WORDS % W_LANES) == 0, "ALIGN_WORDS must be a multiple of W_LANES");

constexpr uint32_t align_up_words(uint32_t x, uint32_t a) {
  return ((x + a - 1u) / a) * a;
}

constexpr uint32_t legacy_words_to_storage_words(uint32_t words) {
  return words * SRAM_STORAGE_WORDS_PER_LEGACY_WORD;
}

constexpr uint32_t storage_words_to_legacy_words_ceil(uint32_t words) {
  return ceil_div_u32(words, SRAM_STORAGE_WORDS_PER_LEGACY_WORD);
}

constexpr uint32_t align_up_storage_words(uint32_t x, uint32_t a) {
  return ((x + a - 1u) / a) * a;
}

constexpr uint32_t align_up_storage_words_to_beat(uint32_t x) {
  return align_up_storage_words(x, SRAM_STORAGE_WORDS_PER_BEAT);
}

namespace sram_map {

// ------------------------------------------------------------
// Baseline storage taxonomy (single-X_WORK semantics)
// ------------------------------------------------------------
enum StorageClass : uint8_t {
  CLASS_W_REGION = 0,
  CLASS_X_WORK = 1,
  CLASS_SCRATCH = 2,
  CLASS_IO_REGION = 3,
  CLASS_TOKEN_LOCAL = 4,
  CLASS_SMALL_SCRATCH = 5,
  CLASS_INVALID = 255
};

enum AliasGroup : uint8_t {
  ALIAS_W_PERSIST = 0,
  ALIAS_X_WORK = 1,
  ALIAS_SCR_K = 2,
  ALIAS_SCR_V = 3,
  ALIAS_FINAL_SCALAR = 4,
  ALIAS_IO_STAGING = 5,
  ALIAS_LOCAL_ONLY = 6,
  ALIAS_INVALID = 255
};

// ------------------------------------------------------------
// Legacy compatibility region decode classes
// ------------------------------------------------------------
enum SramRegion : uint8_t {
  REG_X_PAGE0 = 0,
  REG_X_PAGE1 = 1,
  REG_SCRATCH = 2,
  REG_W_REGION = 3,
  REG_INVALID = 255
};

// ----------------------------
// X ping-pong
// ----------------------------
// X is fp32 [N_NODES, D_MODEL] => WORDS_X_FP32
static const uint32_t X_PAGE_WORDS = align_up_words(WORDS_X_FP32, ALIGN_WORDS);

static const uint32_t BASE_X_PING_W = 0;
static const uint32_t SIZE_X_PING_W = X_PAGE_WORDS;

static const uint32_t BASE_X_PONG_W = BASE_X_PING_W + SIZE_X_PING_W;
static const uint32_t SIZE_X_PONG_W = X_PAGE_WORDS;

// Baseline single-X_WORK window (covers both legacy page aliases).
static const uint32_t BASE_X_WORK_W = BASE_X_PING_W;
static const uint32_t SIZE_X_WORK_W = (SIZE_X_PING_W + SIZE_X_PONG_W);

// Legacy compatibility aliases (same values as above).
static const uint32_t X_PAGE0_BASE_W = BASE_X_PING_W;
static const uint32_t X_PAGE0_WORDS  = SIZE_X_PING_W;
static const uint32_t X_PAGE1_BASE_W = BASE_X_PONG_W;
static const uint32_t X_PAGE1_WORDS  = SIZE_X_PONG_W;

// ----------------------------
// SCRATCH (S1: KV cache + FinalHead scalar)
// ----------------------------
// SCR_K: fp32 [N_NODES, D_MODEL]
// SCR_V: fp32 [N_NODES, D_MODEL]
// SCR_FINAL_SCALAR: fp32 [N_NODES]
static const uint32_t BASE_SCRATCH_W = BASE_X_PONG_W + SIZE_X_PONG_W;

static const uint32_t BASE_SCR_K_W = align_up_words(BASE_SCRATCH_W, ALIGN_WORDS);
static const uint32_t SIZE_SCR_K_W = X_PAGE_WORDS;

static const uint32_t BASE_SCR_V_W = BASE_SCR_K_W + SIZE_SCR_K_W;
static const uint32_t SIZE_SCR_V_W = X_PAGE_WORDS;

static const uint32_t BASE_SCR_FINAL_SCALAR_W =
  align_up_words(BASE_SCR_V_W + SIZE_SCR_V_W, ALIGN_WORDS);
static const uint32_t SIZE_SCR_FINAL_SCALAR_W = align_up_words(N_NODES, ALIGN_WORDS);

// Alias names used by v11.12/v12 bridge and reports.
static const uint32_t SCR_FINAL_SCALAR_BASE_W = BASE_SCR_FINAL_SCALAR_W;
static const uint32_t SCR_FINAL_SCALAR_WORDS = SIZE_SCR_FINAL_SCALAR_W;
static const uint32_t SCR_FINAL_SCALAR_BASE = SCR_FINAL_SCALAR_BASE_W;

static const uint32_t SIZE_SCRATCH_W =
  (BASE_SCR_FINAL_SCALAR_W + SIZE_SCR_FINAL_SCALAR_W) - BASE_SCRATCH_W;

// ----------------------------
// [legacy] BIAS region (fp32 words)
// ----------------------------
static const uint32_t BASE_BIAS_W = align_up_words(BASE_SCRATCH_W + SIZE_SCRATCH_W, ALIGN_WORDS);
static const uint32_t SIZE_BIAS_W = align_up_words(EXP_LEN_BIAS_WORDS, ALIGN_WORDS);

// ----------------------------
// [legacy] WEIGHT region
// ----------------------------
// Includes:
// - BCH parity-check H (bitpack) first
// - then model weights (fp32)
// - plus src_mask (bitpack) and other fp32 tensors
static const uint32_t BASE_W_W = BASE_BIAS_W + SIZE_BIAS_W;
static const uint32_t SIZE_W_W = align_up_words(EXP_LEN_W_WORDS, ALIGN_WORDS);

// ----------------------------
// W_REGION (v11.4+ main path)
// ----------------------------
// Unified PARAM stream is written starting at runtime param_base_word.
// SET_W_BASE must range-check param_base_word against this allowed region.
static const uint32_t W_REGION_BASE  = BASE_BIAS_W;
static const uint32_t W_REGION_WORDS = (SIZE_BIAS_W + SIZE_W_W);

// Suggested default for TB bring-up (if no special placement is needed).
static const uint32_t PARAM_BASE_DEFAULT = W_REGION_BASE;

// ----------------------------
// END / sizing
// ----------------------------
static const uint32_t END_W = BASE_W_W + SIZE_W_W;

// ----------------------------
// backup profile runtime scratch extension (local-only)
// ----------------------------
// This window is reserved for bring-up runtime intermediates so they do not
// alias unified PARAM / W_REGION.
static const uint32_t BACKUP_RUNTIME_SCRATCH_BASE_W = align_up_words(END_W, ALIGN_WORDS);
static const uint32_t BACKUP_RUNTIME_SCRATCH_WORDS = align_up_words(65536u, ALIGN_WORDS);

// IO_REGION is channel-oriented in current bring-up path, so SRAM window is
// kept as an empty placeholder for taxonomy completeness.
static const uint32_t IO_REGION_BASE_W = END_W;
static const uint32_t IO_REGION_WORDS = 0;

// Minimum required SRAM depth (words) for this memory map.
static const uint32_t SRAM_WORDS_MIN_REQUIRED = END_W;

// NOTE: Your actual SRAM depth may be larger.
// For TB bring-up, you may set SRAM_WORDS_TOTAL = SRAM_WORDS_MIN_REQUIRED.
static const uint32_t SRAM_WORDS_TOTAL =
  BACKUP_RUNTIME_SCRATCH_BASE_W + BACKUP_RUNTIME_SCRATCH_WORDS;

// v12.1 storage-word aliases for ref-model / loader migration.
static const uint32_t BASE_X_WORK_WORD16 = legacy_words_to_storage_words(BASE_X_WORK_W);
static const uint32_t SIZE_X_WORK_WORD16 = legacy_words_to_storage_words(SIZE_X_WORK_W);
static const uint32_t BASE_SCRATCH_WORD16 = legacy_words_to_storage_words(BASE_SCRATCH_W);
static const uint32_t SIZE_SCRATCH_WORD16 = legacy_words_to_storage_words(SIZE_SCRATCH_W);
static const uint32_t BASE_SCR_K_WORD16 = legacy_words_to_storage_words(BASE_SCR_K_W);
static const uint32_t SIZE_SCR_K_WORD16 = legacy_words_to_storage_words(SIZE_SCR_K_W);
static const uint32_t BASE_SCR_V_WORD16 = legacy_words_to_storage_words(BASE_SCR_V_W);
static const uint32_t SIZE_SCR_V_WORD16 = legacy_words_to_storage_words(SIZE_SCR_V_W);
static const uint32_t SCR_FINAL_SCALAR_BASE_WORD16 = legacy_words_to_storage_words(SCR_FINAL_SCALAR_BASE_W);
static const uint32_t SCR_FINAL_SCALAR_WORDS_WORD16 = legacy_words_to_storage_words(SCR_FINAL_SCALAR_WORDS);
static const uint32_t W_REGION_BASE_WORD16 = legacy_words_to_storage_words(W_REGION_BASE);
static const uint32_t W_REGION_WORDS_WORD16 = legacy_words_to_storage_words(W_REGION_WORDS);
static const uint32_t PARAM_BASE_DEFAULT_WORD16 = legacy_words_to_storage_words(PARAM_BASE_DEFAULT);
static const uint32_t SRAM_STORAGE_WORDS_MIN_REQUIRED = legacy_words_to_storage_words(SRAM_WORDS_MIN_REQUIRED);
static const uint32_t SRAM_STORAGE_WORDS_TOTAL = legacy_words_to_storage_words(SRAM_WORDS_TOTAL);

// ------------------------------------------------------------
// Region decode helpers (purely by addr_word range)
// ------------------------------------------------------------
static inline bool in_range(uint32_t addr_w, uint32_t base_w, uint32_t size_w) {
  return (addr_w >= base_w) && (addr_w < (base_w + size_w));
}

static inline SramRegion region_of_addr(uint32_t addr_w) {
  if (in_range(addr_w, X_PAGE0_BASE_W, X_PAGE0_WORDS)) return REG_X_PAGE0;
  if (in_range(addr_w, X_PAGE1_BASE_W, X_PAGE1_WORDS)) return REG_X_PAGE1;
  if (in_range(addr_w, BASE_SCRATCH_W, SIZE_SCRATCH_W)) return REG_SCRATCH;
  if (in_range(addr_w, BACKUP_RUNTIME_SCRATCH_BASE_W, BACKUP_RUNTIME_SCRATCH_WORDS)) return REG_SCRATCH;
  if (in_range(addr_w, W_REGION_BASE, W_REGION_WORDS)) return REG_W_REGION;
  return REG_INVALID;
}

static inline StorageClass storage_class_of_addr(uint32_t addr_w) {
  if (in_range(addr_w, BASE_X_WORK_W, SIZE_X_WORK_W)) return CLASS_X_WORK;
  if (in_range(addr_w, BASE_SCRATCH_W, SIZE_SCRATCH_W)) return CLASS_SCRATCH;
  if (in_range(addr_w, BACKUP_RUNTIME_SCRATCH_BASE_W, BACKUP_RUNTIME_SCRATCH_WORDS)) return CLASS_SCRATCH;
  if (in_range(addr_w, W_REGION_BASE, W_REGION_WORDS)) return CLASS_W_REGION;
  if (in_range(addr_w, IO_REGION_BASE_W, IO_REGION_WORDS)) return CLASS_IO_REGION;
  return CLASS_INVALID;
}

} // namespace sram_map
