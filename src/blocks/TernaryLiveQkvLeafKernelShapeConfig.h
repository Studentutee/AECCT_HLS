#pragma once
// P00-011T: compile-time shape SSOT for current supported QKV live-cut/compile-prep build.
// Runtime metadata/config performs validation against these compile-time supported shapes only.
// This header does not imply runtime-variable top interfaces.
// It defines expected constants only; it does not perform materialization logic.

#include <cstdint>

namespace aecct {

// Current compile-time supported shapes for QKV split/materialize surfaces.
static constexpr uint32_t kQkvCtSupportedL0WqRows = 32u;
static constexpr uint32_t kQkvCtSupportedL0WqCols = 32u;
static constexpr uint32_t kQkvCtSupportedL0WkRows = 32u;
static constexpr uint32_t kQkvCtSupportedL0WkCols = 32u;
static constexpr uint32_t kQkvCtSupportedL0WvRows = 32u;
static constexpr uint32_t kQkvCtSupportedL0WvCols = 32u;

// P00-011U: payload-metadata expectation SSOT bridge.
// Expected payload metadata must stay on one active source chain.
static constexpr uint32_t kQkvCtPackedWordElems = 16u;

static inline constexpr uint32_t qkv_ct_payload_words_from_num_weights(uint32_t num_weights) {
    return (num_weights + (kQkvCtPackedWordElems - 1u)) / kQkvCtPackedWordElems;
}

static inline constexpr uint32_t qkv_ct_last_word_valid_count_from_num_weights(uint32_t num_weights) {
    const uint32_t rem = (num_weights % kQkvCtPackedWordElems);
    return (rem == 0u) ? kQkvCtPackedWordElems : rem;
}

static constexpr uint32_t kQkvCtExpectedL0WqNumWeights = kQkvCtSupportedL0WqRows * kQkvCtSupportedL0WqCols;
static constexpr uint32_t kQkvCtExpectedL0WqPayloadWords = qkv_ct_payload_words_from_num_weights(kQkvCtExpectedL0WqNumWeights);
static constexpr uint32_t kQkvCtExpectedL0WqLastWordValidCount =
    qkv_ct_last_word_valid_count_from_num_weights(kQkvCtExpectedL0WqNumWeights);
static constexpr uint32_t kQkvCtSupportedL0WqPayloadWords = kQkvCtExpectedL0WqPayloadWords;

static constexpr uint32_t kQkvCtExpectedL0WkNumWeights = kQkvCtSupportedL0WkRows * kQkvCtSupportedL0WkCols;
static constexpr uint32_t kQkvCtExpectedL0WkPayloadWords = qkv_ct_payload_words_from_num_weights(kQkvCtExpectedL0WkNumWeights);
static constexpr uint32_t kQkvCtExpectedL0WkLastWordValidCount =
    qkv_ct_last_word_valid_count_from_num_weights(kQkvCtExpectedL0WkNumWeights);
static constexpr uint32_t kQkvCtSupportedL0WkPayloadWords = kQkvCtExpectedL0WkPayloadWords;

static constexpr uint32_t kQkvCtExpectedL0WvNumWeights = kQkvCtSupportedL0WvRows * kQkvCtSupportedL0WvCols;
static constexpr uint32_t kQkvCtExpectedL0WvPayloadWords = qkv_ct_payload_words_from_num_weights(kQkvCtExpectedL0WvNumWeights);
static constexpr uint32_t kQkvCtExpectedL0WvLastWordValidCount =
    qkv_ct_last_word_valid_count_from_num_weights(kQkvCtExpectedL0WvNumWeights);
static constexpr uint32_t kQkvCtSupportedL0WvPayloadWords = kQkvCtExpectedL0WvPayloadWords;

static_assert(kQkvCtExpectedL0WqNumWeights == (kQkvCtSupportedL0WqRows * kQkvCtSupportedL0WqCols),
              "L0_WQ expected num_weights must match rows*cols");
static_assert(kQkvCtExpectedL0WkNumWeights == (kQkvCtSupportedL0WkRows * kQkvCtSupportedL0WkCols),
              "L0_WK expected num_weights must match rows*cols");
static_assert(kQkvCtExpectedL0WvNumWeights == (kQkvCtSupportedL0WvRows * kQkvCtSupportedL0WvCols),
              "L0_WV expected num_weights must match rows*cols");

static_assert(kQkvCtExpectedL0WqPayloadWords == qkv_ct_payload_words_from_num_weights(kQkvCtExpectedL0WqNumWeights),
              "L0_WQ expected payload words must be derived from expected num_weights");
static_assert(kQkvCtExpectedL0WkPayloadWords == qkv_ct_payload_words_from_num_weights(kQkvCtExpectedL0WkNumWeights),
              "L0_WK expected payload words must be derived from expected num_weights");
static_assert(kQkvCtExpectedL0WvPayloadWords == qkv_ct_payload_words_from_num_weights(kQkvCtExpectedL0WvNumWeights),
              "L0_WV expected payload words must be derived from expected num_weights");

static_assert(kQkvCtExpectedL0WqLastWordValidCount ==
                  qkv_ct_last_word_valid_count_from_num_weights(kQkvCtExpectedL0WqNumWeights),
              "L0_WQ expected last_word_valid_count must be derived from expected num_weights");
static_assert(kQkvCtExpectedL0WkLastWordValidCount ==
                  qkv_ct_last_word_valid_count_from_num_weights(kQkvCtExpectedL0WkNumWeights),
              "L0_WK expected last_word_valid_count must be derived from expected num_weights");
static_assert(kQkvCtExpectedL0WvLastWordValidCount ==
                  qkv_ct_last_word_valid_count_from_num_weights(kQkvCtExpectedL0WvNumWeights),
              "L0_WV expected last_word_valid_count must be derived from expected num_weights");

static_assert((kQkvCtExpectedL0WqLastWordValidCount >= 1u) && (kQkvCtExpectedL0WqLastWordValidCount <= kQkvCtPackedWordElems),
              "L0_WQ expected last_word_valid_count must be in range");
static_assert((kQkvCtExpectedL0WkLastWordValidCount >= 1u) && (kQkvCtExpectedL0WkLastWordValidCount <= kQkvCtPackedWordElems),
              "L0_WK expected last_word_valid_count must be in range");
static_assert((kQkvCtExpectedL0WvLastWordValidCount >= 1u) && (kQkvCtExpectedL0WvLastWordValidCount <= kQkvCtPackedWordElems),
              "L0_WV expected last_word_valid_count must be in range");

// Legacy-style names are re-exports from SSOT only.
// They must remain aliases and not become a second active definition point.
static constexpr uint32_t kTernaryLiveL0WqRows = kQkvCtSupportedL0WqRows;
static constexpr uint32_t kTernaryLiveL0WqCols = kQkvCtSupportedL0WqCols;
static constexpr uint32_t kTernaryLiveL0WqPayloadWords = kQkvCtSupportedL0WqPayloadWords;

static constexpr uint32_t kTernaryLiveL0WkRows = kQkvCtSupportedL0WkRows;
static constexpr uint32_t kTernaryLiveL0WkCols = kQkvCtSupportedL0WkCols;
static constexpr uint32_t kTernaryLiveL0WkPayloadWords = kQkvCtSupportedL0WkPayloadWords;

static constexpr uint32_t kTernaryLiveL0WvRows = kQkvCtSupportedL0WvRows;
static constexpr uint32_t kTernaryLiveL0WvCols = kQkvCtSupportedL0WvCols;
static constexpr uint32_t kTernaryLiveL0WvPayloadWords = kQkvCtSupportedL0WvPayloadWords;

} // namespace aecct
