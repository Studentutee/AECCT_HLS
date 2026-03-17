#pragma once
// P00-011V: validation-only continuity fence for QKV SSOT expectations against
// the authoritative local-build WeightStreamOrder metadata surface.

#include <cstdint>

#include "TernaryLiveQkvLeafKernelShapeConfig.h"
#include "gen/WeightStreamOrder.h"

namespace aecct {

static constexpr const QuantLinearMeta& kQkvWoMetaL0Wq = kQuantLinearMeta[(uint32_t)QLM_L0_WQ];
static constexpr const QuantLinearMeta& kQkvWoMetaL0Wk = kQuantLinearMeta[(uint32_t)QLM_L0_WK];
static constexpr const QuantLinearMeta& kQkvWoMetaL0Wv = kQuantLinearMeta[(uint32_t)QLM_L0_WV];

static_assert(kQkvWoMetaL0Wq.matrix_id == (uint32_t)QLM_L0_WQ, "L0_WQ matrix_id continuity mismatch");
static_assert(kQkvWoMetaL0Wq.rows == kQkvCtSupportedL0WqRows, "L0_WQ rows continuity mismatch");
static_assert(kQkvWoMetaL0Wq.cols == kQkvCtSupportedL0WqCols, "L0_WQ cols continuity mismatch");
static_assert(kQkvWoMetaL0Wq.num_weights == kQkvCtExpectedL0WqNumWeights, "L0_WQ num_weights continuity mismatch");
static_assert(kQkvWoMetaL0Wq.payload_words_2b == kQkvCtExpectedL0WqPayloadWords,
              "L0_WQ payload_words_2b continuity mismatch");
static_assert(kQkvWoMetaL0Wq.last_word_valid_count == kQkvCtExpectedL0WqLastWordValidCount,
              "L0_WQ last_word_valid_count continuity mismatch");

static_assert(kQkvWoMetaL0Wk.matrix_id == (uint32_t)QLM_L0_WK, "L0_WK matrix_id continuity mismatch");
static_assert(kQkvWoMetaL0Wk.rows == kQkvCtSupportedL0WkRows, "L0_WK rows continuity mismatch");
static_assert(kQkvWoMetaL0Wk.cols == kQkvCtSupportedL0WkCols, "L0_WK cols continuity mismatch");
static_assert(kQkvWoMetaL0Wk.num_weights == kQkvCtExpectedL0WkNumWeights, "L0_WK num_weights continuity mismatch");
static_assert(kQkvWoMetaL0Wk.payload_words_2b == kQkvCtExpectedL0WkPayloadWords,
              "L0_WK payload_words_2b continuity mismatch");
static_assert(kQkvWoMetaL0Wk.last_word_valid_count == kQkvCtExpectedL0WkLastWordValidCount,
              "L0_WK last_word_valid_count continuity mismatch");

static_assert(kQkvWoMetaL0Wv.matrix_id == (uint32_t)QLM_L0_WV, "L0_WV matrix_id continuity mismatch");
static_assert(kQkvWoMetaL0Wv.rows == kQkvCtSupportedL0WvRows, "L0_WV rows continuity mismatch");
static_assert(kQkvWoMetaL0Wv.cols == kQkvCtSupportedL0WvCols, "L0_WV cols continuity mismatch");
static_assert(kQkvWoMetaL0Wv.num_weights == kQkvCtExpectedL0WvNumWeights, "L0_WV num_weights continuity mismatch");
static_assert(kQkvWoMetaL0Wv.payload_words_2b == kQkvCtExpectedL0WvPayloadWords,
              "L0_WV payload_words_2b continuity mismatch");
static_assert(kQkvWoMetaL0Wv.last_word_valid_count == kQkvCtExpectedL0WvLastWordValidCount,
              "L0_WV last_word_valid_count continuity mismatch");

} // namespace aecct
