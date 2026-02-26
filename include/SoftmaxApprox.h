//======================================================================
// SoftmaxApprox.h
//----------------------------------------------------------------------
// Purpose:
//   Synthesizable softmax approximation utilities for Catapult HLS.
//
//   Chosen scheme (per spec):
//     - exp(x) via LUT, where x = score - max_score <= 0
//     - 1/sumexp via reciprocal LUT
//     - No std::exp, no division in datapath (use multiply by inv_sumexp)
//
// Notes:
//   1) This header is intended to be the single source of truth for:
//        - SOFTMAX_NEG_T, LUT sizes, EPS, mapping rules
//        - exp/rcp LUT data (either embedded here or included externally)
//   2) You may choose to keep LUT data in a separate generated header
//      "SoftmaxApproxLutData.h" to avoid huge diffs in version control.
//   3) All math here is fixed mapping + table lookup; synthesizable.
//======================================================================

#ifndef SOFTMAX_APPROX_H
#define SOFTMAX_APPROX_H

#include <ac_fixed.h>
#include <ac_int.h>

//----------------------------------------------------------------------
// Configuration (compile-time constants)
//----------------------------------------------------------------------

struct SoftmaxApproxCfg {
  // Clamp range for x = score - max_score
  // x is expected to be <= 0; clamp to [-SOFTMAX_NEG_T, 0].
  static const int SOFTMAX_NEG_T = 12; // recommended bring-up: 10~12

  // LUT sizes (power-of-two not required, but convenient)
  static const int EXP_LUT_SIZE = 256;
  static const int RCP_LUT_SIZE = 256;

  // EPS floor for sumexp to avoid reciprocal overflow/invalid index
  // EPS = 2^(-EPS_POW2_NEG) by default.
  static const int EPS_POW2_NEG = 16;  // 2^-16 ≈ 1.5e-5
};

//----------------------------------------------------------------------
// Datatypes
//   Tune widths as needed.
//
//   x type: clamped score range [-T, 0]
//   exp type: exp(x) in (0, 1]
//   sum type: sum of exp, range approx [1, SUMEXP_MAX]
//   inv type: 1/sum, range approx (0, 1]
//----------------------------------------------------------------------

// x in [-T, 0], signed
typedef ac_fixed<18, 6, true>  softmax_x_t;

// exp(x) in (0, 1], unsigned
typedef ac_fixed<18, 2, false> softmax_exp_t;

// sumexp up to SOFTMAX_SUMEXP_MAX (override below), unsigned
typedef ac_fixed<24, 10, false> softmax_sum_t;

// inv_sumexp in (0, 1], unsigned
typedef ac_fixed<18, 2, false> softmax_inv_t;

// LUT index type
typedef ac_int<16, false> softmax_idx_t;

//----------------------------------------------------------------------
// Index mapping scales (precomputed constants; no '/' in mapping logic)
// IMPORTANT: These must match the LUT generation parameters in
// SoftmaxApproxLutData.h. If you change SOFTMAX_NEG_T / LUT sizes /
// SOFTMAX_SUMEXP_MAX, regenerate the LUT data AND update these scales.
//----------------------------------------------------------------------

// For default generation: SOFTMAX_NEG_T=12, EXP_LUT_SIZE=256 => 255/12 = 21.25
static const softmax_x_t SOFTMAX_EXP_IDX_SCALE = softmax_x_t(21.25);

// For default generation: SOFTMAX_SUMEXP_MAX=256, RCP_LUT_SIZE=256 => 255/255 = 1.0
static const softmax_sum_t SOFTMAX_RCP_IDX_SCALE = softmax_sum_t(1.0);

//----------------------------------------------------------------------
// Optional: Specify SUMEXP_MAX for reciprocal mapping.
// Override this macro from ModelShapes.h (e.g., N_NODES) before including.
//----------------------------------------------------------------------

#ifndef SOFTMAX_SUMEXP_MAX
#define SOFTMAX_SUMEXP_MAX 256
#endif

//----------------------------------------------------------------------
// LUT data provisioning
//   Option A (default): include generated data header.
//   Option B: embed LUT arrays directly here.
//
// Expected names:
//   - g_exp_lut[SoftmaxApproxCfg::EXP_LUT_SIZE]
//   - g_rcp_lut[SoftmaxApproxCfg::RCP_LUT_SIZE]
//----------------------------------------------------------------------

#ifndef SOFTMAX_LUT_EMBEDDED
#define SOFTMAX_LUT_EMBEDDED 0
#endif

#if SOFTMAX_LUT_EMBEDDED
// Embed arrays here (PLACEHOLDERS).
// Replace these placeholders with real LUT values.
static const softmax_exp_t g_exp_lut[SoftmaxApproxCfg::EXP_LUT_SIZE] = {
  softmax_exp_t(1.0)
};

static const softmax_inv_t g_rcp_lut[SoftmaxApproxCfg::RCP_LUT_SIZE] = {
  softmax_inv_t(1.0)
};

#else
#include "SoftmaxApproxLutData.h"
#endif

//----------------------------------------------------------------------
// clamp x to [-T, 0]
//----------------------------------------------------------------------

static inline softmax_x_t softmax_clamp_x(softmax_x_t x) {
#pragma hls_inline
  const softmax_x_t neg_t = softmax_x_t(-SoftmaxApproxCfg::SOFTMAX_NEG_T);
  if (x > softmax_x_t(0))  x = softmax_x_t(0);
  if (x < neg_t)           x = neg_t;
  return x;
}

//----------------------------------------------------------------------
// x in [-T, 0] -> exp LUT index in [0, EXP_LUT_SIZE-1]
//   x = 0   -> idx 0
//   x = -T  -> idx EXP_LUT_SIZE-1
//   idx = round((-x) * EXP_IDX_SCALE)  // EXP_IDX_SCALE=(EXP_LUT_SIZE-1)/T
//----------------------------------------------------------------------

static inline softmax_idx_t softmax_exp_idx(softmax_x_t x_clamped) {
#pragma hls_inline
  const softmax_x_t mag = softmax_x_t(0) - x_clamped; // -x in [0, T]
  // Precomputed scale avoids divider inference in HLS
  softmax_x_t idx_f = mag * SOFTMAX_EXP_IDX_SCALE;
  softmax_idx_t idx = softmax_idx_t(idx_f + softmax_x_t(0.5)); // round
  if (idx < 0) idx = 0;
  if (idx > (SoftmaxApproxCfg::EXP_LUT_SIZE - 1)) idx = (SoftmaxApproxCfg::EXP_LUT_SIZE - 1);
  return idx;
}

static inline softmax_exp_t softmax_exp_lut(softmax_x_t x) {
#pragma hls_inline
  softmax_x_t x_c = softmax_clamp_x(x);
  softmax_idx_t idx = softmax_exp_idx(x_c);
  return g_exp_lut[idx];
}

//----------------------------------------------------------------------
// EPS helpers
//----------------------------------------------------------------------

static inline softmax_sum_t softmax_eps() {
#pragma hls_inline
  // EPS = 2^-k
  const int k = SoftmaxApproxCfg::EPS_POW2_NEG;
  softmax_sum_t eps = softmax_sum_t(1);
  eps = eps >> k;
  return eps;
}

static inline softmax_sum_t softmax_sum_floor(softmax_sum_t sumexp) {
#pragma hls_inline
  softmax_sum_t eps = softmax_eps();
  if (sumexp < eps) sumexp = eps;
  return sumexp;
}

//----------------------------------------------------------------------
// sumexp in [1, SUMEXP_MAX] -> rcp LUT index in [0, RCP_LUT_SIZE-1]
//   sumexp = 1          -> idx 0
//   sumexp = SUMEXP_MAX -> idx RCP_LUT_SIZE-1
//   idx = round((sumexp - 1) * RCP_IDX_SCALE)  // RCP_IDX_SCALE=(RCP_LUT_SIZE-1)/(SUMEXP_MAX-1)
//----------------------------------------------------------------------

static inline softmax_idx_t softmax_rcp_idx(softmax_sum_t sumexp_safe) {
#pragma hls_inline
  const softmax_sum_t one = softmax_sum_t(1);
  const softmax_sum_t maxv = softmax_sum_t(SOFTMAX_SUMEXP_MAX);
  softmax_sum_t s = sumexp_safe;

  if (s < one)  s = one;
  if (s > maxv) s = maxv;

  const softmax_sum_t numer = s - one;

  // Precomputed scale avoids divider inference in HLS
  softmax_sum_t idx_f = numer * SOFTMAX_RCP_IDX_SCALE;
  softmax_idx_t idx = softmax_idx_t(idx_f + softmax_sum_t(0.5));
  if (idx < 0) idx = 0;
  if (idx > (SoftmaxApproxCfg::RCP_LUT_SIZE - 1)) idx = (SoftmaxApproxCfg::RCP_LUT_SIZE - 1);
  return idx;
}

static inline softmax_inv_t softmax_rcp_lut(softmax_sum_t sumexp) {
#pragma hls_inline
  softmax_sum_t s = softmax_sum_floor(sumexp);
  softmax_idx_t idx = softmax_rcp_idx(s);
  return g_rcp_lut[idx];
}

#endif // SOFTMAX_APPROX_H
