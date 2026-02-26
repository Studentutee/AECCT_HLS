#pragma once
// SoftmaxDesc.h
// M15a softmax approximation single source of truth

#include <ac_fixed.h>
#include "ModelShapes.h"

struct SoftmaxApproxCfg {
  // x = score - max(score), then clamp to [-SOFTMAX_NEG_T, 0]
  static const int SOFTMAX_NEG_T = 12;
  static const int EXP_LUT_SIZE = 256;
  static const int RCP_LUT_SIZE = 256;
  static const int EPS_POW2_NEG = 16;
};

// Softmax reciprocal LUT mapping range.
// Must match SoftmaxApproxLutData.h generation.
static const int SOFTMAX_SUMEXP_MAX = 256;

// Quantization format (single source of truth).
typedef ac_fixed<18, 6, true, AC_RND, AC_SAT> softmax_x_t;
typedef ac_fixed<18, 2, false, AC_RND, AC_SAT> softmax_exp_t;
typedef ac_fixed<24, 10, false, AC_RND, AC_SAT> softmax_sum_t;
typedef ac_fixed<18, 2, false, AC_RND, AC_SAT> softmax_inv_t;
typedef ac_fixed<18, 6, true, AC_RND, AC_SAT> softmax_score_t;
typedef ac_fixed<18, 2, false, AC_RND, AC_SAT> softmax_prob_t;
typedef ac_int<16, false> softmax_idx_t;

// Precomputed scales, avoids divider inference in datapath.
// EXP index: (EXP_LUT_SIZE-1) / SOFTMAX_NEG_T = 255 / 12 = 21.25
static const softmax_x_t SOFTMAX_EXP_IDX_SCALE = softmax_x_t(21.25);

// RCP index: (RCP_LUT_SIZE-1) / (SOFTMAX_SUMEXP_MAX-1) = 255/255 = 1.0
static const softmax_sum_t SOFTMAX_RCP_IDX_SCALE = softmax_sum_t(1.0);
