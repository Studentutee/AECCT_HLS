#pragma once

#include <cmath>

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

static constexpr float REFV3_INV_L0_ATTN_Q =
  1.0f / (REFV3_SCALE_L0_IN_S_X * REFV3_SCALE_L0_ATTN_Q_S_W);
static constexpr float REFV3_INV_L0_ATTN_K =
  1.0f / (REFV3_SCALE_L0_IN_S_X * REFV3_SCALE_L0_ATTN_K_S_W);
static constexpr float REFV3_INV_L0_ATTN_V =
  1.0f / (REFV3_SCALE_L0_IN_S_X * REFV3_SCALE_L0_ATTN_V_S_W);
static constexpr float REFV3_INV_L0_ATTN_O =
  1.0f / (REFV3_SCALE_L0_O_S_X * REFV3_SCALE_L0_ATTN_O_S_W);
static constexpr float REFV3_INV_L0_FFN_W1 =
  1.0f / (REFV3_SCALE_L0_FF1_S_X * REFV3_SCALE_L0_FFN_W1_S_W);
static constexpr float REFV3_INV_L0_FFN_W2 =
  1.0f / (REFV3_SCALE_L0_FF2_S_X * REFV3_SCALE_L0_FFN_W2_S_W);

static constexpr float REFV3_INV_D_MODEL = 1.0f / static_cast<float>(REFV3_D_MODEL);

static inline refv3_fp_t REFV3_softmax_rcp_lut_safe(refv3_fp_t sumexp) {
  const refv3_fp_t zero(0.0f);
  if (sumexp <= zero) {
    return zero;
  }
  return ref_softmax_rcp_lut(sumexp);
}

static inline float REFV3_inv_sqrt_nr1_or_lut(float x) {
  float x_safe = x;
  if (!std::isfinite(x_safe) || x_safe <= 0.0f) {
    x_safe = 1.0f;
  }

  float inv_std = ref_inv_sqrt_nr1_approx(refv3_fp_t(x_safe)).to_float();
  if (!std::isfinite(inv_std) || inv_std <= 0.0f) {
    inv_std = ref_inv_sqrt_approx(refv3_fp_t(x_safe)).to_float();
  }
  return inv_std;
}

} // namespace ref_v3
} // namespace aecct_ref
