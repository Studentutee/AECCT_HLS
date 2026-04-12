// SramMap.h
#pragma once
#include <cstdint>
#include "ModelShapes.h"

// ============================================================
// SramMap.h (legacy aggregate-word bridge + v12.1 storage-word helpers)
// ------------------------------------------------------------
// - Existing *_W symbols stay in legacy logical u32 words to avoid a broad
//   refactor of current bring-up code paths.
// - New *_WORD16 / *_STORAGE_* helpers expose the v12.1 storage contract:
//     * SRAM word = 16 bits
//     * SRAM beat = 8 words = 128 bits
//     * base_word / len_words for the new profile should use 16-bit storage words.
//
// Readability / usage intent:
// - This file now separates three things that were previously easy to mix:
//     1) logical windows (for example X_WORK / W_REGION / SCRATCH)
//     2) concrete physical sub-sections (for example SCR_K / SCR_V /
//        FINAL_SCALAR_BUF)
//     3) actual payload words versus padded region capacity
// - The goal is to make later LOAD_W correctness and READ_MEM / compare-side
//   boundary checks easier to reason about without changing the DUT math path.
//
// Step2 convergence note:
// - Baseline storage semantics are single-X_WORK:
//     * W_REGION, X_WORK, SCRATCH, IO_REGION
// - Legacy dual-page wording has been retired from the active map.
//   A second physical slice is still exposed only as an X_WORK compatibility
//   slice for transitional bring-up code paths.
// - Physical implementation MAY still map to one flat SRAM word space.
//
// Notes:
// - X_WORK is the only baseline shared working area.
// - SCR_K / SCR_V / FINAL_SCALAR_BUF are SCRATCH sub-regions.
// - [compat] A second X_WORK slice remains only for transitional code paths;
//   it is not a separate baseline taxonomy in this step.
// - No dedicated DEBUG SRAM region (D1 debug is "halt + READ_MEM").
// - [legacy] Separate BIAS/WEIGHT regions are still exposed for compatibility.
// ============================================================

static const uint32_t SRAM_WORD_BYTES = BYTES_PER_WORD;
static const uint32_t SRAM_WORD_LANES = W_LANES;
static_assert(SRAM_WORD_BYTES == 16u, "Legacy aggregate beat must remain 16 bytes during bridge.");
static_assert(SRAM_WORD_LANES == 8u, "Legacy aggregate beat must remain 8 lanes during bridge.");

static const uint32_t SRAM_STORAGE_WORDS_PER_BEAT = ::SRAM_WORDS_PER_BEAT;
static const uint32_t SRAM_STORAGE_WORDS_PER_LEGACY_WORD =
    (LEGACY_U32_WORD_BITS / ::SRAM_STORAGE_WORD_BITS);
static_assert(::SRAM_STORAGE_WORD_BITS == 16u, "v12.1 storage word must be 16 bits");
static_assert(SRAM_STORAGE_WORDS_PER_BEAT == 8u, "v12.1 beat must contain 8 storage words");
static_assert(::SRAM_BEAT_BITS == 128u, "v12.1 beat must be 128 bits");
static_assert(SRAM_STORAGE_WORDS_PER_LEGACY_WORD == 2u,
              "Legacy u32 word must map to two 16-bit storage words.");

// Alignment unit remains legacy logical-word addressed in the old map.
static const uint32_t ALIGN_WORDS = 16u;
static const uint32_t ALIGN_STORAGE_WORD16 =
    (ALIGN_WORDS * SRAM_STORAGE_WORDS_PER_LEGACY_WORD);
static_assert((ALIGN_WORDS % W_LANES) == 0, "ALIGN_WORDS must be a multiple of W_LANES");
static_assert((ALIGN_STORAGE_WORD16 % SRAM_STORAGE_WORDS_PER_BEAT) == 0u,
              "Legacy alignment must stay beat-aligned after u32->word16 conversion.");

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

// Describes what kind of bytes a section/window is expected to hold.
// This is intentionally small and synth-safe: later checkers can convert it
// to human-readable strings outside of design code.
enum PayloadClass : uint8_t {
  PAYLOAD_FP32_TENSOR = 0,
  PAYLOAD_FP32_VECTOR = 1,
  PAYLOAD_BITPACK = 2,
  PAYLOAD_PARAM_STREAM = 3,
  PAYLOAD_EMPTY = 4,
  PAYLOAD_COMPAT_ALIAS = 5,
  PAYLOAD_RUNTIME_SCRATCH = 6,
  PAYLOAD_MIXED_PERSIST = 7,
  PAYLOAD_FP16_TENSOR = 8,
  PAYLOAD_FP16_VECTOR = 9,
  PAYLOAD_INVALID = 255
};

// Distinguish concrete non-overlapping sub-sections from logical overlay
// windows such as X_WORK / SCRATCH / W_REGION / PARAM_STREAM_DEFAULT.
enum SectionKind : uint8_t {
  SECTION_PHYSICAL = 0,
  SECTION_LOGICAL = 1
};

enum SectionFlags : uint32_t {
  SECTION_FLAG_NONE = 0u,
  SECTION_FLAG_LOAD_W_CRITICAL = 1u << 0,
  SECTION_FLAG_READ_MEM_VISIBLE = 1u << 1,
  SECTION_FLAG_COMPARE_CRITICAL = 1u << 2,
  SECTION_FLAG_LEGACY_COMPAT = 1u << 3,
  SECTION_FLAG_PADDING_PRESENT = 1u << 4
};

// ------------------------------------------------------------
// Region decode classes (single-X_WORK baseline)
// ------------------------------------------------------------
enum SramRegion : uint8_t {
  REG_X_WORK = 0,
  REG_SCRATCH = 1,
  REG_W_REGION = 2,
  REG_INVALID = 255
};

// Fine-grain section IDs used by focused checkers / READ_MEM audits.
// Overlapping logical windows are allowed; use section_kind()/section_flags()
// to understand whether the entry is a concrete non-overlapping sub-section or
// a logical coverage window.
enum SectionId : uint8_t {
  SEC_X_WORK = 0,
  SEC_SCR_K = 1,
  SEC_SCR_V = 2,
  SEC_FINAL_SCALAR_BUF = 3,
  SEC_SCRATCH = 4,
  SEC_BIAS_LEGACY = 5,
  SEC_WEIGHT_LEGACY = 6,
  SEC_W_REGION = 7,
  SEC_PARAM_STREAM_DEFAULT = 8,
  SEC_IO_REGION = 9,
  SEC_BACKUP_RUNTIME_SCRATCH = 10,
  SEC_INVALID = 255
};

struct SectionDesc {
  uint8_t id;
  uint8_t kind;
  uint8_t storage_class;
  uint8_t alias_group;
  uint8_t payload_class;
  uint8_t reserved0;
  uint16_t reserved1;
  uint32_t base_w;
  uint32_t words_w;
  uint32_t base_word16;
  uint32_t words_word16;
  uint32_t align_words;
  uint32_t align_word16;
  uint32_t payload_words_w;
  uint32_t payload_words_word16;
  uint32_t flags;
};

// ----------------------------
// X_WORK (single physical working buffer)
// ----------------------------
// X is live fp16 storage [N_NODES, D_MODEL] packed into legacy u32 words
static const uint32_t X_WORK_SLICE_WORDS = align_up_words(WORDS_X_FP16_PACKED, ALIGN_WORDS);
static const uint32_t BASE_X_WORK_W = 0;
static const uint32_t SIZE_X_WORK_W = X_WORK_SLICE_WORDS;

// Public aliases used by active code and checkers.
static const uint32_t X_WORK_BASE_W = BASE_X_WORK_W;
static const uint32_t X_WORK_WORDS  = SIZE_X_WORK_W;

// ----------------------------
// SCRATCH (S1: KV cache + FinalHead scalar)
// ----------------------------
// SCR_K: fp16 [N_NODES, D_MODEL]
// SCR_V: fp16 [N_NODES, D_MODEL]
// FINAL_SCALAR_BUF: fp16 [N_NODES]
static const uint32_t BASE_SCRATCH_W = BASE_X_WORK_W + SIZE_X_WORK_W;

static const uint32_t BASE_SCR_K_W = align_up_words(BASE_SCRATCH_W, ALIGN_WORDS);
static const uint32_t SIZE_SCR_K_W = align_up_words(WORDS_X_FP16_PACKED, ALIGN_WORDS);

static const uint32_t BASE_SCR_V_W = BASE_SCR_K_W + SIZE_SCR_K_W;
static const uint32_t SIZE_SCR_V_W = align_up_words(WORDS_X_FP16_PACKED, ALIGN_WORDS);

static const uint32_t BASE_SCR_FINAL_SCALAR_W =
  align_up_words(BASE_SCR_V_W + SIZE_SCR_V_W, ALIGN_WORDS);
static const uint32_t SIZE_SCR_FINAL_SCALAR_W = align_up_words(legacy_u32_words_fp16_packed(N_NODES), ALIGN_WORDS);

// Alias names used by v11.12/v12 bridge and reports.
static const uint32_t SCR_FINAL_SCALAR_BASE_W = BASE_SCR_FINAL_SCALAR_W;
static const uint32_t SCR_FINAL_SCALAR_WORDS = SIZE_SCR_FINAL_SCALAR_W;
static const uint32_t SCR_FINAL_SCALAR_BASE = SCR_FINAL_SCALAR_BASE_W;

// New explicit BUF aliases for readability on the storage/debug side.
static const uint32_t FINAL_SCALAR_BUF_BASE_W = BASE_SCR_FINAL_SCALAR_W;
static const uint32_t FINAL_SCALAR_BUF_WORDS = SIZE_SCR_FINAL_SCALAR_W;

static const uint32_t SIZE_SCRATCH_W =
  (BASE_SCR_FINAL_SCALAR_W + SIZE_SCR_FINAL_SCALAR_W) - BASE_SCRATCH_W;

// ----------------------------
// FP16 branch bias / weight persistent region
// ----------------------------
// Bias tensors and non-linear/plain weights use fp16 storage words (1 value =
// 1 word16). Quant-linear ternary payload remains packed, and inv_s_w / other
// metadata lanes also live in word16 space. Legacy u32 views below are kept
// only as a ceil(word16/2) bridge for existing bring-up code paths.
static const uint32_t BASE_BIAS_W = align_up_words(BASE_SCRATCH_W + SIZE_SCRATCH_W, ALIGN_WORDS);
static const uint32_t FP16_BRANCH_TOTAL_BIAS_WORDS16 = 832u;
static const uint32_t FP16_BRANCH_TOTAL_WEIGHT_WORDS16 = 10853u;
static const uint32_t FP16_BRANCH_TOTAL_PARAM_WORDS16 = 11685u;

static const uint32_t SIZE_BIAS_PAYLOAD_WORD16 = FP16_BRANCH_TOTAL_BIAS_WORDS16;
static const uint32_t SIZE_BIAS_PAYLOAD_W = storage_words_to_legacy_words_ceil(SIZE_BIAS_PAYLOAD_WORD16);
static const uint32_t SIZE_BIAS_W = align_up_words(SIZE_BIAS_PAYLOAD_W, ALIGN_WORDS);
static const uint32_t SIZE_BIAS_PADDING_W = (SIZE_BIAS_W - SIZE_BIAS_PAYLOAD_W);
static const uint32_t SIZE_BIAS_PADDING_WORD16 =
    (legacy_words_to_storage_words(SIZE_BIAS_W) - SIZE_BIAS_PAYLOAD_WORD16);

// ----------------------------
// [branch] WEIGHT region
// ----------------------------
// Includes:
// - BCH parity-check H / SRC_MASK bitpack metadata
// - Quant-linear ternary payload matrices
// - fp16 inv_s_w metadata
// - fp16 non-linear / final-head / plain weights
static const uint32_t BASE_W_W = BASE_BIAS_W + SIZE_BIAS_W;
static const uint32_t SIZE_WEIGHT_PAYLOAD_WORD16 = FP16_BRANCH_TOTAL_WEIGHT_WORDS16;
static const uint32_t SIZE_WEIGHT_PAYLOAD_W = storage_words_to_legacy_words_ceil(SIZE_WEIGHT_PAYLOAD_WORD16);
static const uint32_t SIZE_W_W = align_up_words(SIZE_WEIGHT_PAYLOAD_W, ALIGN_WORDS);
static const uint32_t SIZE_WEIGHT_PADDING_W = (SIZE_W_W - SIZE_WEIGHT_PAYLOAD_W);
static const uint32_t SIZE_WEIGHT_PADDING_WORD16 =
    (legacy_words_to_storage_words(SIZE_W_W) - SIZE_WEIGHT_PAYLOAD_WORD16);

// ----------------------------
// W_REGION (fp16 branch main path)
// ----------------------------
// Unified PARAM stream keeps using legacy u32 command lengths for transition,
// but the authoritative storage contract is now expressed in 16-bit words.
static const uint32_t W_REGION_BASE  = BASE_BIAS_W;
static const uint32_t W_REGION_WORDS = (SIZE_BIAS_W + SIZE_W_W);
static const uint32_t W_REGION_PAYLOAD_WORDS = (SIZE_BIAS_PAYLOAD_W + SIZE_WEIGHT_PAYLOAD_W);
static const uint32_t W_REGION_PADDING_WORDS = (W_REGION_WORDS - W_REGION_PAYLOAD_WORDS);
static const uint32_t W_REGION_PAYLOAD_WORDS_WORD16 =
    (SIZE_BIAS_PAYLOAD_WORD16 + SIZE_WEIGHT_PAYLOAD_WORD16);
static const uint32_t W_REGION_PADDING_WORDS_WORD16 =
    (legacy_words_to_storage_words(W_REGION_WORDS) - W_REGION_PAYLOAD_WORDS_WORD16);

// Suggested default for TB bring-up (if no special placement is needed).
static const uint32_t PARAM_BASE_DEFAULT = W_REGION_BASE;

// Explicit default PARAM stream window inside W_REGION.
// This is the actual LOAD_W payload length, not the padded capacity of W_REGION.
static const uint32_t PARAM_STREAM_DEFAULT_BASE_W = PARAM_BASE_DEFAULT;
static const uint32_t PARAM_STREAM_DEFAULT_WORDS = W_REGION_PAYLOAD_WORDS;
static const uint32_t PARAM_STREAM_DEFAULT_WORDS_WORD16 = FP16_BRANCH_TOTAL_PARAM_WORDS16;

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

// Minimum required SRAM depth (legacy u32 words) for this memory map.
static const uint32_t SRAM_WORDS_MIN_REQUIRED = END_W;

// NOTE: Your actual SRAM depth may be larger.
// For TB bring-up, you may set SRAM_WORDS_TOTAL = SRAM_WORDS_MIN_REQUIRED.
static const uint32_t SRAM_WORDS_TOTAL =
  BACKUP_RUNTIME_SCRATCH_BASE_W + BACKUP_RUNTIME_SCRATCH_WORDS;

// v12.1 storage-word aliases for ref-model / loader migration.
static const uint32_t BASE_X_WORK_WORD16 = legacy_words_to_storage_words(BASE_X_WORK_W);
static const uint32_t SIZE_X_WORK_WORD16 = legacy_words_to_storage_words(SIZE_X_WORK_W);
static const uint32_t X_WORK_BASE_WORD16 = BASE_X_WORK_WORD16;
static const uint32_t X_WORK_WORD16 = SIZE_X_WORK_WORD16;
static const uint32_t BASE_SCRATCH_WORD16 = legacy_words_to_storage_words(BASE_SCRATCH_W);
static const uint32_t SIZE_SCRATCH_WORD16 = legacy_words_to_storage_words(SIZE_SCRATCH_W);
static const uint32_t BASE_SCR_K_WORD16 = legacy_words_to_storage_words(BASE_SCR_K_W);
static const uint32_t SIZE_SCR_K_WORD16 = legacy_words_to_storage_words(SIZE_SCR_K_W);
static const uint32_t BASE_SCR_V_WORD16 = legacy_words_to_storage_words(BASE_SCR_V_W);
static const uint32_t SIZE_SCR_V_WORD16 = legacy_words_to_storage_words(SIZE_SCR_V_W);
static const uint32_t SCR_FINAL_SCALAR_BASE_WORD16 = legacy_words_to_storage_words(SCR_FINAL_SCALAR_BASE_W);
static const uint32_t SCR_FINAL_SCALAR_WORDS_WORD16 = legacy_words_to_storage_words(SCR_FINAL_SCALAR_WORDS);
static const uint32_t FINAL_SCALAR_BUF_BASE_WORD16 = legacy_words_to_storage_words(FINAL_SCALAR_BUF_BASE_W);
static const uint32_t FINAL_SCALAR_BUF_WORDS_WORD16 = legacy_words_to_storage_words(FINAL_SCALAR_BUF_WORDS);
static const uint32_t BASE_BIAS_WORD16 = legacy_words_to_storage_words(BASE_BIAS_W);
static const uint32_t SIZE_BIAS_WORD16 = legacy_words_to_storage_words(SIZE_BIAS_W);
static const uint32_t BASE_WEIGHT_WORD16 = legacy_words_to_storage_words(BASE_W_W);
static const uint32_t SIZE_WEIGHT_WORD16 = legacy_words_to_storage_words(SIZE_W_W);
static const uint32_t W_REGION_BASE_WORD16 = legacy_words_to_storage_words(W_REGION_BASE);
static const uint32_t W_REGION_WORDS_WORD16 = legacy_words_to_storage_words(W_REGION_WORDS);
static const uint32_t PARAM_BASE_DEFAULT_WORD16 = legacy_words_to_storage_words(PARAM_BASE_DEFAULT);
static const uint32_t PARAM_STREAM_DEFAULT_BASE_WORD16 = legacy_words_to_storage_words(PARAM_STREAM_DEFAULT_BASE_W);
static const uint32_t IO_REGION_BASE_WORD16 = legacy_words_to_storage_words(IO_REGION_BASE_W);
static const uint32_t IO_REGION_WORDS_WORD16 = legacy_words_to_storage_words(IO_REGION_WORDS);
static const uint32_t BACKUP_RUNTIME_SCRATCH_BASE_WORD16 = legacy_words_to_storage_words(BACKUP_RUNTIME_SCRATCH_BASE_W);
static const uint32_t BACKUP_RUNTIME_SCRATCH_WORDS_WORD16 = legacy_words_to_storage_words(BACKUP_RUNTIME_SCRATCH_WORDS);
static const uint32_t SRAM_STORAGE_WORDS_MIN_REQUIRED = legacy_words_to_storage_words(SRAM_WORDS_MIN_REQUIRED);
static const uint32_t SRAM_STORAGE_WORDS_TOTAL = legacy_words_to_storage_words(SRAM_WORDS_TOTAL);

static_assert(PARAM_STREAM_DEFAULT_WORDS_WORD16 == FP16_BRANCH_TOTAL_PARAM_WORDS16,
              "fp16 branch PARAM stream word16 count must match frozen branch contract.");
static_assert(W_REGION_BASE == PARAM_BASE_DEFAULT,
              "Default PARAM base must stay at the start of W_REGION.");
static_assert((PARAM_STREAM_DEFAULT_BASE_W + PARAM_STREAM_DEFAULT_WORDS) <= (W_REGION_BASE + W_REGION_WORDS),
              "Default PARAM stream must fit inside W_REGION.");
static_assert(FINAL_SCALAR_BUF_BASE_W == SCR_FINAL_SCALAR_BASE_W,
              "FINAL_SCALAR_BUF alias must match legacy SCR_FINAL_SCALAR base.");
static_assert(FINAL_SCALAR_BUF_WORDS == SCR_FINAL_SCALAR_WORDS,
              "FINAL_SCALAR_BUF alias must match legacy SCR_FINAL_SCALAR words.");

static constexpr SectionDesc kSectionTable[] = {
  { SEC_X_WORK, SECTION_PHYSICAL, CLASS_X_WORK, ALIAS_X_WORK, PAYLOAD_FP16_TENSOR, 0u, 0u,
    BASE_X_WORK_W, SIZE_X_WORK_W, BASE_X_WORK_WORD16, SIZE_X_WORK_WORD16,
    ALIGN_WORDS, ALIGN_STORAGE_WORD16,
    WORDS_X_FP16_PACKED, storage_words_fp16(ELEMS_X),
    SECTION_FLAG_READ_MEM_VISIBLE | SECTION_FLAG_COMPARE_CRITICAL },
  { SEC_SCR_K, SECTION_PHYSICAL, CLASS_SCRATCH, ALIAS_SCR_K, PAYLOAD_FP16_TENSOR, 0u, 0u,
    BASE_SCR_K_W, SIZE_SCR_K_W, BASE_SCR_K_WORD16, SIZE_SCR_K_WORD16,
    ALIGN_WORDS, ALIGN_STORAGE_WORD16,
    WORDS_X_FP16_PACKED, storage_words_fp16(ELEMS_X),
    SECTION_FLAG_READ_MEM_VISIBLE | SECTION_FLAG_COMPARE_CRITICAL },
  { SEC_SCR_V, SECTION_PHYSICAL, CLASS_SCRATCH, ALIAS_SCR_V, PAYLOAD_FP16_TENSOR, 0u, 0u,
    BASE_SCR_V_W, SIZE_SCR_V_W, BASE_SCR_V_WORD16, SIZE_SCR_V_WORD16,
    ALIGN_WORDS, ALIGN_STORAGE_WORD16,
    WORDS_X_FP16_PACKED, storage_words_fp16(ELEMS_X),
    SECTION_FLAG_READ_MEM_VISIBLE | SECTION_FLAG_COMPARE_CRITICAL },
  { SEC_FINAL_SCALAR_BUF, SECTION_PHYSICAL, CLASS_SCRATCH, ALIAS_FINAL_SCALAR, PAYLOAD_FP16_VECTOR, 0u, 0u,
    FINAL_SCALAR_BUF_BASE_W, FINAL_SCALAR_BUF_WORDS,
    FINAL_SCALAR_BUF_BASE_WORD16, FINAL_SCALAR_BUF_WORDS_WORD16,
    ALIGN_WORDS, ALIGN_STORAGE_WORD16,
    legacy_u32_words_fp16_packed(N_NODES), storage_words_fp16(N_NODES),
    SECTION_FLAG_READ_MEM_VISIBLE | SECTION_FLAG_COMPARE_CRITICAL | SECTION_FLAG_PADDING_PRESENT },
  { SEC_SCRATCH, SECTION_LOGICAL, CLASS_SCRATCH, ALIAS_LOCAL_ONLY, PAYLOAD_RUNTIME_SCRATCH, 0u, 0u,
    BASE_SCRATCH_W, SIZE_SCRATCH_W, BASE_SCRATCH_WORD16, SIZE_SCRATCH_WORD16,
    ALIGN_WORDS, ALIGN_STORAGE_WORD16,
    SIZE_SCRATCH_W, SIZE_SCRATCH_WORD16,
    SECTION_FLAG_READ_MEM_VISIBLE | SECTION_FLAG_COMPARE_CRITICAL },
  { SEC_BIAS_LEGACY, SECTION_PHYSICAL, CLASS_W_REGION, ALIAS_W_PERSIST, PAYLOAD_MIXED_PERSIST, 0u, 0u,
    BASE_BIAS_W, SIZE_BIAS_W, BASE_BIAS_WORD16, SIZE_BIAS_WORD16,
    ALIGN_WORDS, ALIGN_STORAGE_WORD16,
    SIZE_BIAS_PAYLOAD_W, SIZE_BIAS_PAYLOAD_WORD16,
    SECTION_FLAG_LOAD_W_CRITICAL | SECTION_FLAG_READ_MEM_VISIBLE | SECTION_FLAG_PADDING_PRESENT | SECTION_FLAG_LEGACY_COMPAT },
  { SEC_WEIGHT_LEGACY, SECTION_PHYSICAL, CLASS_W_REGION, ALIAS_W_PERSIST, PAYLOAD_MIXED_PERSIST, 0u, 0u,
    BASE_W_W, SIZE_W_W, BASE_WEIGHT_WORD16, SIZE_WEIGHT_WORD16,
    ALIGN_WORDS, ALIGN_STORAGE_WORD16,
    SIZE_WEIGHT_PAYLOAD_W, SIZE_WEIGHT_PAYLOAD_WORD16,
    SECTION_FLAG_LOAD_W_CRITICAL | SECTION_FLAG_READ_MEM_VISIBLE | SECTION_FLAG_PADDING_PRESENT | SECTION_FLAG_LEGACY_COMPAT },
  { SEC_W_REGION, SECTION_LOGICAL, CLASS_W_REGION, ALIAS_W_PERSIST, PAYLOAD_MIXED_PERSIST, 0u, 0u,
    W_REGION_BASE, W_REGION_WORDS, W_REGION_BASE_WORD16, W_REGION_WORDS_WORD16,
    ALIGN_WORDS, ALIGN_STORAGE_WORD16,
    W_REGION_PAYLOAD_WORDS, W_REGION_PAYLOAD_WORDS_WORD16,
    SECTION_FLAG_LOAD_W_CRITICAL | SECTION_FLAG_READ_MEM_VISIBLE | SECTION_FLAG_PADDING_PRESENT },
  { SEC_PARAM_STREAM_DEFAULT, SECTION_LOGICAL, CLASS_W_REGION, ALIAS_W_PERSIST, PAYLOAD_PARAM_STREAM, 0u, 0u,
    PARAM_STREAM_DEFAULT_BASE_W, PARAM_STREAM_DEFAULT_WORDS,
    PARAM_STREAM_DEFAULT_BASE_WORD16, PARAM_STREAM_DEFAULT_WORDS_WORD16,
    W_LANES, legacy_words_to_storage_words(W_LANES),
    PARAM_STREAM_DEFAULT_WORDS, PARAM_STREAM_DEFAULT_WORDS_WORD16,
    SECTION_FLAG_LOAD_W_CRITICAL | SECTION_FLAG_READ_MEM_VISIBLE },
  { SEC_IO_REGION, SECTION_LOGICAL, CLASS_IO_REGION, ALIAS_IO_STAGING, PAYLOAD_EMPTY, 0u, 0u,
    IO_REGION_BASE_W, IO_REGION_WORDS, IO_REGION_BASE_WORD16, IO_REGION_WORDS_WORD16,
    ALIGN_WORDS, ALIGN_STORAGE_WORD16,
    IO_REGION_WORDS, IO_REGION_WORDS_WORD16,
    SECTION_FLAG_NONE },
  { SEC_BACKUP_RUNTIME_SCRATCH, SECTION_PHYSICAL, CLASS_SCRATCH, ALIAS_LOCAL_ONLY, PAYLOAD_RUNTIME_SCRATCH, 0u, 0u,
    BACKUP_RUNTIME_SCRATCH_BASE_W, BACKUP_RUNTIME_SCRATCH_WORDS,
    BACKUP_RUNTIME_SCRATCH_BASE_WORD16, BACKUP_RUNTIME_SCRATCH_WORDS_WORD16,
    ALIGN_WORDS, ALIGN_STORAGE_WORD16,
    BACKUP_RUNTIME_SCRATCH_WORDS, BACKUP_RUNTIME_SCRATCH_WORDS_WORD16,
    SECTION_FLAG_READ_MEM_VISIBLE | SECTION_FLAG_COMPARE_CRITICAL | SECTION_FLAG_LEGACY_COMPAT }
};

static const uint32_t SECTION_COUNT =
  (uint32_t)(sizeof(kSectionTable) / sizeof(kSectionTable[0]));

// ------------------------------------------------------------
// Generic helpers
// ------------------------------------------------------------
static inline bool in_range(uint32_t addr_w, uint32_t base_w, uint32_t size_w) {
  return (addr_w >= base_w) && (addr_w < (base_w + size_w));
}

static inline bool range_fits(uint32_t base_w, uint32_t words_w,
                              uint32_t container_base_w, uint32_t container_words_w) {
  const unsigned long long begin = (unsigned long long)base_w;
  const unsigned long long end_excl = begin + (unsigned long long)words_w;
  const unsigned long long container_begin = (unsigned long long)container_base_w;
  const unsigned long long container_end_excl = container_begin + (unsigned long long)container_words_w;
  return (begin >= container_begin) && (end_excl <= container_end_excl);
}

static inline bool is_legacy_word_aligned(uint32_t addr_w, uint32_t align_words) {
  return (align_words == 0u) ? true : ((addr_w % align_words) == 0u);
}

static inline bool is_storage_word_aligned(uint32_t addr_word16, uint32_t align_word16) {
  return (align_word16 == 0u) ? true : ((addr_word16 % align_word16) == 0u);
}

static inline bool is_storage_beat_aligned_word16(uint32_t addr_word16) {
  return ((addr_word16 % SRAM_STORAGE_WORDS_PER_BEAT) == 0u);
}

static inline const SectionDesc& section_desc(const SectionId id) {
  const uint32_t idx = (uint32_t)id;
  if (idx < SECTION_COUNT) {
    return kSectionTable[idx];
  }
  return kSectionTable[0];
}

static inline uint32_t section_flags(const SectionId id) {
  const uint32_t idx = (uint32_t)id;
  if (idx < SECTION_COUNT) {
    return kSectionTable[idx].flags;
  }
  return SECTION_FLAG_NONE;
}

static inline bool section_has_flag(const SectionId id, const uint32_t flag) {
  return ((section_flags(id) & flag) != 0u);
}

static inline bool section_affects_load_w(const SectionId id) {
  return section_has_flag(id, SECTION_FLAG_LOAD_W_CRITICAL);
}

static inline bool section_affects_compare(const SectionId id) {
  return section_has_flag(id, SECTION_FLAG_COMPARE_CRITICAL);
}

static inline bool section_is_read_mem_visible(const SectionId id) {
  return section_has_flag(id, SECTION_FLAG_READ_MEM_VISIBLE);
}

static inline bool section_is_logical_window(const SectionId id) {
  const uint32_t idx = (uint32_t)id;
  return (idx < SECTION_COUNT) ? (kSectionTable[idx].kind == SECTION_LOGICAL) : false;
}

static inline bool section_contains_addr(const SectionId id, uint32_t addr_w) {
  const uint32_t idx = (uint32_t)id;
  if (idx >= SECTION_COUNT) {
    return false;
  }
  return in_range(addr_w, kSectionTable[idx].base_w, kSectionTable[idx].words_w);
}

static inline bool section_payload_fits_capacity(const SectionId id) {
  const uint32_t idx = (uint32_t)id;
  if (idx >= SECTION_COUNT) {
    return false;
  }
  return (kSectionTable[idx].payload_words_w <= kSectionTable[idx].words_w);
}

static inline bool default_param_stream_fits_w_region(void) {
  return range_fits(PARAM_STREAM_DEFAULT_BASE_W, PARAM_STREAM_DEFAULT_WORDS,
                    W_REGION_BASE, W_REGION_WORDS);
}

// Decode the most specific non-overlapping physical section first.
static inline SectionId physical_section_of_addr(uint32_t addr_w) {
  if (in_range(addr_w, BASE_X_WORK_W, SIZE_X_WORK_W)) return SEC_X_WORK;
  if (in_range(addr_w, BASE_SCR_K_W, SIZE_SCR_K_W)) return SEC_SCR_K;
  if (in_range(addr_w, BASE_SCR_V_W, SIZE_SCR_V_W)) return SEC_SCR_V;
  if (in_range(addr_w, FINAL_SCALAR_BUF_BASE_W, FINAL_SCALAR_BUF_WORDS)) return SEC_FINAL_SCALAR_BUF;
  if (in_range(addr_w, BASE_BIAS_W, SIZE_BIAS_W)) return SEC_BIAS_LEGACY;
  if (in_range(addr_w, BASE_W_W, SIZE_W_W)) return SEC_WEIGHT_LEGACY;
  if (in_range(addr_w, BACKUP_RUNTIME_SCRATCH_BASE_W, BACKUP_RUNTIME_SCRATCH_WORDS)) return SEC_BACKUP_RUNTIME_SCRATCH;
  return SEC_INVALID;
}

// ------------------------------------------------------------
// Region decode helpers (purely by addr_word range)
// ------------------------------------------------------------
static inline SramRegion region_of_addr(uint32_t addr_w) {
  if (in_range(addr_w, BASE_X_WORK_W, SIZE_X_WORK_W)) return REG_X_WORK;
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
