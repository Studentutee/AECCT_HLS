#include "../../include/ref_v3/RefV3LayerNormBlock.h"
#include "../../include/ref_v3/RefV3MathApprox.h"
#include "../../include/ref_v3/RefV3WeightsFp16LocalOnly.h"

#include <cmath>

namespace aecct_ref {
namespace ref_v3 {
namespace {

static void layernorm_token_32_local(
  const RefRunConfig& run_cfg,
  const refv3_fp_t x_token[REFV3_D_MODEL],
  const RefV3TernaryLinearParams& params,
  refv3_fp_t y_token[REFV3_D_MODEL]) {
  const refv3_fp_t eps(1.0e-5f);
  const refv3_fp_t inv_d(REFV3_INV_D_MODEL);
  const refv3_fp_t zero(0.0f);
  const refv3_fp_t one(1.0f);

  auto sanitize_input = [](refv3_fp_t v) -> refv3_fp_t {
    return (v == v) ? v : refv3_fp_t(0.0f);
  };
  auto sanitize_output = [](refv3_fp_t v) -> refv3_fp_t {
    return (v == v) ? v : refv3_fp_t(0.0f);
  };

  // Host-only exact reference path; synthesis surface must stay on LUT-based approximation.
#if !defined(__SYNTHESIS__) && !defined(REFV3_SYNTH_ONLY)
  if (run_cfg.legacy.ln_mode == RefLayerNormMode::LN_EXACT_REFERENCE) {
    const double eps_host = 1.0e-5;
    double sum = 0.0;
    REFV3_LN_EXACT_SUM_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      sum += static_cast<double>(x_token[d].to_float());
    }
    const double mean = sum / static_cast<double>(REFV3_D_MODEL);

    double var_acc = 0.0;
    REFV3_LN_EXACT_VAR_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      const double dv = static_cast<double>(x_token[d].to_float()) - mean;
      var_acc += (dv * dv);
    }
    const double var = var_acc / static_cast<double>(REFV3_D_MODEL);
    const double inv_std = 1.0 / std::sqrt(var + eps_host);

    REFV3_LN_EXACT_OUT_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      const double xv = static_cast<double>(x_token[d].to_float());
      const double xn = (xv - mean) * inv_std;
      const double wv = static_cast<double>(refv3_linear_weight_fp_at(params, d).to_float());
      const double bv = static_cast<double>(refv3_linear_bias_fp_at(params, d).to_float());
      const double yi =
        (xn * wv) +
        bv;
      y_token[d] = refv3_fp_t(static_cast<float>(yi));
    }
    return;
  }
#endif

// Host-only legacy compare path; synthesis surface must keep a single baseline path.
#if !defined(__SYNTHESIS__) && !defined(REFV3_SYNTH_ONLY)
  if (run_cfg.legacy.ln_mode == RefLayerNormMode::LN_SUM_SUMSQ_APPROX) {
    refv3_fp_t sum = zero;
    refv3_fp_t sumsq = zero;
    REFV3_LN_SUMSQ_ACC_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      const refv3_fp_t xv = sanitize_input(x_token[d]);
      sum += xv;
      sumsq += xv * xv;
    }
    const refv3_fp_t mean = sum * inv_d;
    const refv3_fp_t ex2 = sumsq * inv_d;
    const refv3_fp_t mean_sq = mean * mean;
    const refv3_fp_t var_raw = ex2 - mean_sq;

    refv3_fp_t var_final = var_raw;
    if (var_final != var_final || var_final < zero) {
      var_final = zero;
    }
    refv3_fp_t var_eps = var_final + eps;
    if (var_eps != var_eps || var_eps < eps) {
      var_eps = eps;
    }

    refv3_fp_t inv_std = REFV3_ln_inv_sqrt_synth(var_eps);
    inv_std = sanitize_output(inv_std);

    REFV3_LN_SUMSQ_OUT_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      const refv3_fp_t xv = sanitize_input(x_token[d]);
      const refv3_fp_t xn = (xv - mean) * inv_std;
      const refv3_fp_t yi =
        (xn * refv3_linear_weight_fp_at(params, d)) +
        refv3_linear_bias_fp_at(params, d);
      y_token[d] = sanitize_output(yi);
    }
    return;
  }
#else
  (void)run_cfg;
#endif

  refv3_fp_t sum = zero;
  REFV3_LN_BASE_SUM_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
    const refv3_fp_t xv = sanitize_input(x_token[d]);
    sum += xv;
  }

  const refv3_fp_t mean = sum * inv_d;
  refv3_fp_t var_acc = zero;
  REFV3_LN_BASE_VAR_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
    const refv3_fp_t xv = sanitize_input(x_token[d]);
    const refv3_fp_t delta = xv - mean;
    var_acc += delta * delta;
  }
  const refv3_fp_t var_raw = var_acc * inv_d;

  refv3_fp_t x_eps_safe = var_raw + eps;
  if (x_eps_safe != x_eps_safe || x_eps_safe <= zero) {
    x_eps_safe = eps;
  }
  if (x_eps_safe != x_eps_safe || x_eps_safe <= zero) {
    x_eps_safe = one;
  }

  refv3_fp_t inv_std = REFV3_ln_inv_sqrt_synth(x_eps_safe);
  if (inv_std != inv_std || inv_std <= zero) {
    inv_std = one;
  }

  REFV3_LN_BASE_OUT_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
    const refv3_fp_t xv = sanitize_input(x_token[d]);
    const refv3_fp_t xn = (xv - mean) * inv_std;
    refv3_fp_t yi =
      (xn * refv3_linear_weight_fp_at(params, d)) +
      refv3_linear_bias_fp_at(params, d);
    yi = sanitize_output(yi);
    y_token[d] = yi;
  }
}

} // namespace

RefV3LayerNormBlock::RefV3LayerNormBlock() {}

namespace {

static bool refv3_layernorm_run_with_explicit_params(
  int expected_layer_id,
  const RefV3TernaryLinearParams& ln_params,
  const RefRunConfig& run_cfg,
  ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
  RefV3AttentionPayloadHeader header_ref;
  bool header_init = false;
  bool token_seen[REFV3_TOKENS_T];
  refv3_fp_t token_ln_out[REFV3_D_MODEL];

  REFV3_LN_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_LN_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3AttentionTokenVectorPayload token_payload = in_token_ch.read();
    if (!REFV3_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }

    if (!header_init) {
      header_ref = token_payload.header;
      header_init = true;
    } else {
      if (token_payload.header.layer_id != header_ref.layer_id ||
          token_payload.header.token_rows != header_ref.token_rows ||
          token_payload.header.dim_cols != header_ref.dim_cols) {
        return false;
      }
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV3_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    layernorm_token_32_local(run_cfg, token_payload.token_vec, ln_params, token_ln_out);

    RefV3AttentionTokenVectorPayload out_payload;
    out_payload.header = token_payload.header;
    out_payload.token_row = token_payload.token_row;
    REFV3_LN_TOKEN_OUT_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      out_payload.token_vec[dim] = token_ln_out[dim];
    }
    out_token_ch.write(out_payload);
  }

  return true;
}

} // namespace

bool RefV3LayerNormBlock::run(int lid,
                              const RefRunConfig& run_cfg,
                              ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                              ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  const RefV3TernaryLinearParams ln_params = refv3_layernorm0_params_fp_local_only(lid);
  return run_with_params(lid, ln_params, run_cfg, in_token_ch, out_token_ch);
}

bool RefV3LayerNormBlock::run_with_params(
  int expected_layer_id,
  const RefV3TernaryLinearParams& ln_params,
  const RefRunConfig& run_cfg,
  ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const {
  if (expected_layer_id != REFV3_LAYER0_ID && expected_layer_id != REFV3_LAYER1_ID) {
    return false;
  }
  return refv3_layernorm_run_with_explicit_params(
    expected_layer_id,
    ln_params,
    run_cfg,
    in_token_ch,
    out_token_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
