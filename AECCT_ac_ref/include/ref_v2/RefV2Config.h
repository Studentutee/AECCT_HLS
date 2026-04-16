#pragma once

#include "RefStep0ShapeBridge.h"

namespace aecct_ref {
namespace ref_v2 {

static const int REFV2_LAYER0_ID = 0;
static const int REFV2_LAYER1_ID = 1;

static const int REFV2_TOKENS_T = ModelShapes::T_TOKENS;
static const int REFV2_D_MODEL = ModelShapes::D_MODEL;
static const int REFV2_HEADS = ModelShapes::N_HEADS;
static const int REFV2_D_HEAD = ModelShapes::D_HEAD;
static const int REFV2_FF_DIM = ModelShapes::D_FFN;
static const int REFV2_VAR_N = ModelShapes::N_VARS;
static const int REFV2_ATTN_MATRIX_ELEMS = REFV2_TOKENS_T * REFV2_D_MODEL;

#ifndef REFV2_LN_BASELINE_EXTRA_NR_ITERS
#define REFV2_LN_BASELINE_EXTRA_NR_ITERS 6
#endif
static_assert(REFV2_LN_BASELINE_EXTRA_NR_ITERS >= 0,
              "REFV2_LN_BASELINE_EXTRA_NR_ITERS must be non-negative");

static_assert(REFV2_TOKENS_T > 0, "REFV2_TOKENS_T must be positive");
static_assert(REFV2_D_MODEL > 0, "REFV2_D_MODEL must be positive");
static_assert(REFV2_HEADS > 0, "REFV2_HEADS must be positive");
static_assert(REFV2_D_HEAD > 0, "REFV2_D_HEAD must be positive");
static_assert(REFV2_FF_DIM > 0, "REFV2_FF_DIM must be positive");
static_assert(REFV2_HEADS * REFV2_D_HEAD == REFV2_D_MODEL,
              "REFV2_HEADS * REFV2_D_HEAD must equal REFV2_D_MODEL");

static inline int refv2_flatten_row_major_index(int token_row, int dim_col) {
  return (token_row * REFV2_D_MODEL) + dim_col;
}

} // namespace ref_v2
} // namespace aecct_ref
