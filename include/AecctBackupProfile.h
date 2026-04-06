#pragma once
// Backup demonstration profile for fp16/io8/inline/ln1p branch.
// This header is intentionally side-effect free in the first cut:
// - No existing design path is changed just by including this file.
// - It provides a single place to freeze branch-local decisions.

#include <cstdint>

#include "AecctTypes.h"
#include "ModelShapes.h"

namespace aecct {

// -----------------------------------------------------------------------------
// Branch identity / posture
// -----------------------------------------------------------------------------
static const bool BACKUP_PROFILE_ENABLE = true;
static const bool BACKUP_PROFILE_KEEP_TOP_SHARED_SRAM_OWNER = true;
static const bool BACKUP_PROFILE_KEEP_PROFILE_RUNTIME = true;

// -----------------------------------------------------------------------------
// Numeric / dataflow targets for this branch
// -----------------------------------------------------------------------------
// Branch intent:
// - main tensor path aims to move from fp32 to fp16
// - native linear path keeps ternary weights + INT8 activations
// - accumulator is intentionally frozen to INT16 for the current demo profile
// - IO payload is serialized over 8-bit data channels
static const bool BACKUP_PROFILE_MAIN_FP16 = true;
static const bool BACKUP_PROFILE_LINEAR_TERNARY = true;
static const bool BACKUP_PROFILE_LINEAR_ACT_INT8 = true;
static const bool BACKUP_PROFILE_LINEAR_ACC_INT16 = true;
static const bool BACKUP_PROFILE_IO8_SERIAL = true;

// Inline / LN one-pass are staged features. Keep them explicit so the branch can
// reach a local testable state before enabling the riskier changes.
static const bool BACKUP_PROFILE_INLINE_LEAF = false;
static const bool BACKUP_PROFILE_LN_ONEPASS = false;

// -----------------------------------------------------------------------------
// External / internal transport targets
// -----------------------------------------------------------------------------
static const uint32_t BACKUP_DATA_IO_BITS = 8;
static const uint32_t BACKUP_SRAM_LANE_BITS = 16;
static const uint32_t BACKUP_SRAM_WORD_LANES = 8;
static const uint32_t BACKUP_SRAM_WORD_BITS =
    BACKUP_SRAM_LANE_BITS * BACKUP_SRAM_WORD_LANES;
static const uint32_t BACKUP_SRAM_WORD_BYTES = BACKUP_SRAM_WORD_BITS / 8u;

// For IO8 serialization:
// - one fp16 lane = 2 bytes
// - one 128-bit SRAM word = 16 bytes
static const uint32_t BACKUP_IO_BYTES_PER_FP16 = 2;
static const uint32_t BACKUP_IO_BYTES_PER_SRAM_WORD = BACKUP_SRAM_WORD_BYTES;
static const uint32_t BACKUP_FP16_LANES_PER_SRAM_WORD = BACKUP_SRAM_WORD_LANES;

// -----------------------------------------------------------------------------
// Current frozen-shape accumulator assumptions
// -----------------------------------------------------------------------------
// Current repo-frozen shapes (ModelShapes.h):
// - D_MODEL = 32
// - D_FFN   = 128
// With ternary weights {-1,0,+1} and INT8 activation magnitude bounded to 127,
// the worst-case signed dot magnitude is K * 127.
// This backup branch intentionally assumes INT16 accumulate is acceptable for the
// current frozen shapes. If wider runtime shapes are enabled later, revalidation is
// mandatory before keeping INT16 accumulate.
static const uint32_t BACKUP_LINEAR_REDUCTION_QKV = D_MODEL;
static const uint32_t BACKUP_LINEAR_REDUCTION_WO = D_MODEL;
static const uint32_t BACKUP_LINEAR_REDUCTION_FFN1 = D_MODEL;
static const uint32_t BACKUP_LINEAR_REDUCTION_FFN2 = D_FFN;

static const int32_t BACKUP_INT8_ABS_MAX = 127;
static const int32_t BACKUP_INT16_ABS_MAX = 32767;

static const int32_t BACKUP_WORST_QKV_ACC =
    (int32_t)BACKUP_LINEAR_REDUCTION_QKV * BACKUP_INT8_ABS_MAX;
static const int32_t BACKUP_WORST_FFN2_ACC =
    (int32_t)BACKUP_LINEAR_REDUCTION_FFN2 * BACKUP_INT8_ABS_MAX;

static_assert(BACKUP_WORST_QKV_ACC <= BACKUP_INT16_ABS_MAX,
              "Backup INT16 accumulator assumption fails for current QKV/WO shape.");
static_assert(BACKUP_WORST_FFN2_ACC <= BACKUP_INT16_ABS_MAX,
              "Backup INT16 accumulator assumption fails for current FFN2 shape.");

// -----------------------------------------------------------------------------
// Optional future compile-time gate hooks
// -----------------------------------------------------------------------------
// These symbolic IDs are provided so branch-local code can switch behavior
// without scattering magic numbers / strings.
static const uint32_t BACKUP_MODE_LEGACY = 0;
static const uint32_t BACKUP_MODE_FP16_IO8 = 1;
static const uint32_t BACKUP_ACTIVE_MODE = BACKUP_MODE_FP16_IO8;

} // namespace aecct
