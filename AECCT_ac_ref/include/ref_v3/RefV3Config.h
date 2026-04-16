#pragma once

#include "RefStep0ShapeBridge.h"

namespace aecct_ref {
namespace ref_v3 {

static const int REFV3_LAYER0_ID = 0;
static const int REFV3_LAYER1_ID = 1;

static const int REFV3_TOKENS_T = ModelShapes::T_TOKENS;
static const int REFV3_D_MODEL = ModelShapes::D_MODEL;
static const int REFV3_HEADS = ModelShapes::N_HEADS;
static const int REFV3_D_HEAD = ModelShapes::D_HEAD;
static const int REFV3_FF_DIM = ModelShapes::D_FFN;
static const int REFV3_VAR_N = ModelShapes::N_VARS;
static const int REFV3_ATTN_MATRIX_ELEMS = REFV3_TOKENS_T * REFV3_D_MODEL;

#ifndef REFV3_LN_INV_SQRT_SYNTH_POLICY
#define REFV3_LN_INV_SQRT_SYNTH_POLICY 0
#endif
static const int REFV3_LN_INV_SQRT_SYNTH_LUT_ONLY = 0;
static const int REFV3_LN_INV_SQRT_SYNTH_LUT_PLUS_NR1 = 1;
static_assert(
  REFV3_LN_INV_SQRT_SYNTH_POLICY == REFV3_LN_INV_SQRT_SYNTH_LUT_ONLY ||
    REFV3_LN_INV_SQRT_SYNTH_POLICY == REFV3_LN_INV_SQRT_SYNTH_LUT_PLUS_NR1,
  "REFV3_LN_INV_SQRT_SYNTH_POLICY must be LUT_ONLY(0) or LUT_PLUS_NR1(1)");

#ifndef REFV3_SOFTMAX_EXP_SYNTH_POLICY
#define REFV3_SOFTMAX_EXP_SYNTH_POLICY 0
#endif
static const int REFV3_SOFTMAX_EXP_SYNTH_LUT_ONLY = 0;
static const int REFV3_SOFTMAX_EXP_SYNTH_LERP_LUT = 1;
static_assert(
  REFV3_SOFTMAX_EXP_SYNTH_POLICY == REFV3_SOFTMAX_EXP_SYNTH_LUT_ONLY ||
    REFV3_SOFTMAX_EXP_SYNTH_POLICY == REFV3_SOFTMAX_EXP_SYNTH_LERP_LUT,
  "REFV3_SOFTMAX_EXP_SYNTH_POLICY must be LUT_ONLY(0) or LERP_LUT(1)");

#ifndef REFV3_SOFTMAX_RCP_SYNTH_POLICY
#define REFV3_SOFTMAX_RCP_SYNTH_POLICY 0
#endif
static const int REFV3_SOFTMAX_RCP_SYNTH_LUT_ONLY = 0;
static const int REFV3_SOFTMAX_RCP_SYNTH_LUT_PLUS_NR1 = 1;
static_assert(
  REFV3_SOFTMAX_RCP_SYNTH_POLICY == REFV3_SOFTMAX_RCP_SYNTH_LUT_ONLY ||
    REFV3_SOFTMAX_RCP_SYNTH_POLICY == REFV3_SOFTMAX_RCP_SYNTH_LUT_PLUS_NR1,
  "REFV3_SOFTMAX_RCP_SYNTH_POLICY must be LUT_ONLY(0) or LUT_PLUS_NR1(1)");

#ifndef REFV3_LN_BASELINE_EXTRA_NR_ITERS
#define REFV3_LN_BASELINE_EXTRA_NR_ITERS 6
#endif
static_assert(REFV3_LN_BASELINE_EXTRA_NR_ITERS >= 0,
              "REFV3_LN_BASELINE_EXTRA_NR_ITERS must be non-negative");

static_assert(REFV3_TOKENS_T > 0, "REFV3_TOKENS_T must be positive");
static_assert(REFV3_D_MODEL > 0, "REFV3_D_MODEL must be positive");
static_assert(REFV3_HEADS > 0, "REFV3_HEADS must be positive");
static_assert(REFV3_D_HEAD > 0, "REFV3_D_HEAD must be positive");
static_assert(REFV3_FF_DIM > 0, "REFV3_FF_DIM must be positive");
static_assert(REFV3_HEADS * REFV3_D_HEAD == REFV3_D_MODEL,
              "REFV3_HEADS * REFV3_D_HEAD must equal REFV3_D_MODEL");

static inline int REFV3_flatten_row_major_index(int token_row, int dim_col) {
  return (token_row * REFV3_D_MODEL) + dim_col;
}

} // namespace ref_v3
} // namespace aecct_ref
