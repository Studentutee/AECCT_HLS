#pragma once

#include "InvSqrtApprox.h"
#include "ref_v3/RefV3Types.h"
#include "SoftmaxApprox.h"
#include "ref_v3/RefV3Config.h"

namespace aecct_ref {
namespace ref_v3 {

// Fixed-weight stripped-track scales for layer0 active path.
static constexpr float REFV3_SCALE_L0_IN_S_X = 40.336463928222656f;
static constexpr float REFV3_SCALE_L0_O_S_X = 110.63162231445312f;
static constexpr float REFV3_SCALE_L0_FF1_S_X = 42.850276947021484f;
static constexpr float REFV3_SCALE_L0_FF2_S_X = 57.637298583984375f;

static constexpr float REFV3_SCALE_L0_ATTN_Q_S_W = 3.4775071144104004f;
static constexpr float REFV3_SCALE_L0_ATTN_K_S_W = 3.2122509479522705f;
static constexpr float REFV3_SCALE_L0_ATTN_V_S_W = 8.15131664276123f;
static constexpr float REFV3_SCALE_L0_ATTN_O_S_W = 10.102375984191895f;
static constexpr float REFV3_SCALE_L0_FFN_W1_S_W = 8.694713592529297f;
static constexpr float REFV3_SCALE_L0_FFN_W2_S_W = 4.8022541999816895f;

// Reciprocal constants are precomputed to keep synthesis away from fp16 operator/.
static constexpr float REFV3_INV_L0_ATTN_Q = 0.0071290908541314845f;
static constexpr float REFV3_INV_L0_ATTN_K = 0.0077177856170683968f;
static constexpr float REFV3_INV_L0_ATTN_V = 0.00304140610051462f;
static constexpr float REFV3_INV_L0_ATTN_O = 0.000894740696358606f;
static constexpr float REFV3_INV_L0_FFN_W1 = 0.0026840529927247346f;
static constexpr float REFV3_INV_L0_FFN_W2 = 0.0036128608702035883f;
static constexpr float REFV3_INV_L1_ATTN_Q = 0.0043993235050510809f;
static constexpr float REFV3_INV_L1_ATTN_K = 0.0071688982173430315f;
static constexpr float REFV3_INV_L1_ATTN_V = 0.0049903884588218062f;
static constexpr float REFV3_INV_L1_ATTN_O = 0.0043067652734030569f;
static constexpr float REFV3_INV_L1_FFN_W1 = 0.002454141838255364f;
static constexpr float REFV3_INV_L1_FFN_W2 = 0.0055658957032067415f;
static constexpr float REFV3_INV_D_MODEL = 0.03125f;

static inline refv3_fp_t REFV3_attn_inv_sxsw_const(int lid, int linear_id) {
  static constexpr float k_inv_attn[2][4] = {
    {REFV3_INV_L0_ATTN_Q, REFV3_INV_L0_ATTN_K, REFV3_INV_L0_ATTN_V, REFV3_INV_L0_ATTN_O},
    {REFV3_INV_L1_ATTN_Q, REFV3_INV_L1_ATTN_K, REFV3_INV_L1_ATTN_V, REFV3_INV_L1_ATTN_O}
  };
  const int layer_idx = (lid == REFV3_LAYER1_ID) ? 1 : 0;
  int linear_idx = linear_id;
  if (linear_idx < 0) {
    linear_idx = 0;
  }
  if (linear_idx > 3) {
    linear_idx = 3;
  }
  return refv3_fp_t(k_inv_attn[layer_idx][linear_idx]);
}

static inline refv3_fp_t REFV3_ffn_w1_inv_sxsw_const(int lid) {
  return refv3_fp_t((lid == REFV3_LAYER1_ID) ? REFV3_INV_L1_FFN_W1 : REFV3_INV_L0_FFN_W1);
}

static inline refv3_fp_t REFV3_ffn_w2_inv_sxsw_const(int lid) {
  return refv3_fp_t((lid == REFV3_LAYER1_ID) ? REFV3_INV_L1_FFN_W2 : REFV3_INV_L0_FFN_W2);
}

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
