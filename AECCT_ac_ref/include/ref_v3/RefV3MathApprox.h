#pragma once

#include "InvSqrtApprox.h"
#include "ref_v3/RefV3Types.h"
#include "SoftmaxApprox.h"
#include "ref_v3/RefV3Config.h"

namespace aecct_ref {
namespace ref_v3 {

// Fixed-weight stripped-track scales for layer0 active path.
static const refv3_fp_t REFV3_SCALE_L0_IN_S_X(40.336463928222656);
static const refv3_fp_t REFV3_SCALE_L0_O_S_X(110.63162231445312);
static const refv3_fp_t REFV3_SCALE_L0_FF1_S_X(42.850276947021484);
static const refv3_fp_t REFV3_SCALE_L0_FF2_S_X(57.637298583984375);

static const refv3_fp_t REFV3_SCALE_L0_ATTN_Q_S_W(3.4775071144104004);
static const refv3_fp_t REFV3_SCALE_L0_ATTN_K_S_W(3.2122509479522705);
static const refv3_fp_t REFV3_SCALE_L0_ATTN_V_S_W(8.15131664276123);
static const refv3_fp_t REFV3_SCALE_L0_ATTN_O_S_W(10.102375984191895);
static const refv3_fp_t REFV3_SCALE_L0_FFN_W1_S_W(8.694713592529297);
static const refv3_fp_t REFV3_SCALE_L0_FFN_W2_S_W(4.8022541999816895);

static const refv3_fp_t REFV3_INV_L0_ATTN_Q =
  refv3_fp_t(1.0f) / (REFV3_SCALE_L0_IN_S_X * REFV3_SCALE_L0_ATTN_Q_S_W);
static const refv3_fp_t REFV3_INV_L0_ATTN_K =
  refv3_fp_t(1.0f) / (REFV3_SCALE_L0_IN_S_X * REFV3_SCALE_L0_ATTN_K_S_W);
static const refv3_fp_t REFV3_INV_L0_ATTN_V =
  refv3_fp_t(1.0f) / (REFV3_SCALE_L0_IN_S_X * REFV3_SCALE_L0_ATTN_V_S_W);
static const refv3_fp_t REFV3_INV_L0_ATTN_O =
  refv3_fp_t(1.0f) / (REFV3_SCALE_L0_O_S_X * REFV3_SCALE_L0_ATTN_O_S_W);
static const refv3_fp_t REFV3_INV_L0_FFN_W1 =
  refv3_fp_t(1.0f) / (REFV3_SCALE_L0_FF1_S_X * REFV3_SCALE_L0_FFN_W1_S_W);
static const refv3_fp_t REFV3_INV_L0_FFN_W2 =
  refv3_fp_t(1.0f) / (REFV3_SCALE_L0_FF2_S_X * REFV3_SCALE_L0_FFN_W2_S_W);

static const refv3_fp_t REFV3_INV_D_MODEL =
  refv3_fp_t(1.0f) / refv3_fp_t(REFV3_D_MODEL);

static inline refv3_fp_t REFV3_ln_inv_sqrt_synth(refv3_fp_t x) {
  const refv3_fp_t zero(0.0f);
  const refv3_fp_t one(1.0f);
  refv3_fp_t x_safe = x;
  if (x_safe != x_safe || x_safe <= zero) {
    x_safe = one;
  }

#if REFV3_LN_INV_MODE == REFV3_LN_INV_MODE_LUT_NR1
  refv3_fp_t inv_std = ref_inv_sqrt_lut_plus_nr1(x_safe);
#else
  refv3_fp_t inv_std = ref_inv_sqrt_lut_only(x_safe);
#endif
  if (inv_std != inv_std || inv_std <= zero) {
    inv_std = ref_inv_sqrt_lut_only(x_safe);
  }
  return inv_std;
}

static inline refv3_fp_t REFV3_softmax_exp_synth(refv3_fp_t x) {
#if REFV3_SOFTMAX_EXP_SYNTH_POLICY == REFV3_SOFTMAX_EXP_SYNTH_LERP_LUT
  return ref_softmax_exp_lerp_lut(x);
#else
  return ref_softmax_exp_lut_only(x);
#endif
}

static inline refv3_fp_t REFV3_softmax_rcp_synth(refv3_fp_t sumexp) {
  const refv3_fp_t zero(0.0f);
  if (sumexp <= zero) {
    return zero;
  }

#if REFV3_SOFTMAX_RCP_MODE == REFV3_SOFTMAX_RCP_MODE_LUT_NR1
  return ref_softmax_rcp_lut_plus_nr1(sumexp);
#else
  return ref_softmax_rcp_lut_only(sumexp);
#endif
}

} // namespace ref_v3
} // namespace aecct_ref
