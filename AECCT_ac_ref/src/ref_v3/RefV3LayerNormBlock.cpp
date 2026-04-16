#include "../../include/ref_v3/RefV3LayerNormBlock.h"
#include "../../include/ref_v3/RefV3MathApprox.h"

#include <cmath>

#include "weights.h"

namespace aecct_ref {
namespace ref_v3 {
namespace {

static void layernorm_token_32_local(
  const RefRunConfig& run_cfg,
  const refv3_fp_t x_token[REFV3_D_MODEL],
  const double w[REFV3_D_MODEL],
  const double b[REFV3_D_MODEL],
  refv3_fp_t y_token[REFV3_D_MODEL]) {
  const float eps = 1.0e-5f;
  const float inv_d = REFV3_INV_D_MODEL;

  auto sanitize_input = [](float v) -> float {
    return std::isfinite(v) ? v : 0.0f;
  };
  auto sanitize_output = [](float v) -> float {
    return std::isfinite(v) ? v : 0.0f;
  };

  // Host-only exact reference path; synthesis surface must stay on LUT-based approximation.
#if !defined(__SYNTHESIS__) && !defined(REFV3_SYNTH_ONLY)
  if (run_cfg.legacy.ln_mode == RefLayerNormMode::LN_EXACT_REFERENCE) {
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
    const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

    REFV3_LN_EXACT_OUT_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      const double xv = static_cast<double>(x_token[d].to_float());
      const double xn = (xv - mean) * inv_std;
      const double yi = (xn * w[d]) + b[d];
      y_token[d] = refv3_fp_t(static_cast<float>(yi));
    }
    return;
  }
#endif

  if (run_cfg.legacy.ln_mode == RefLayerNormMode::LN_SUM_SUMSQ_APPROX) {
    float sum = 0.0f;
    float sumsq = 0.0f;
    REFV3_LN_SUMSQ_ACC_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      const float xv = sanitize_input(x_token[d].to_float());
      sum += xv;
      sumsq += xv * xv;
    }
    const float mean = sum * inv_d;
    const float ex2 = sumsq * inv_d;
    const float mean_sq = mean * mean;
    const float var_raw = ex2 - mean_sq;

    float var_final = var_raw;
    if (!std::isfinite(var_final) || var_final < 0.0f) {
      var_final = 0.0f;
    }
    float var_eps = var_final + eps;
    if (!std::isfinite(var_eps) || var_eps < eps) {
      var_eps = eps;
    }

    float inv_std = REFV3_inv_sqrt_nr1_or_lut(var_eps);
    inv_std = sanitize_output(inv_std);

    REFV3_LN_SUMSQ_OUT_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      const float xv = sanitize_input(x_token[d].to_float());
      const float xn = (xv - mean) * inv_std;
      const float yi = (xn * static_cast<float>(w[d])) + static_cast<float>(b[d]);
      y_token[d] = refv3_fp_t(sanitize_output(yi));
    }
    return;
  }

  float sum = 0.0f;
  REFV3_LN_BASE_SUM_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
    const float xv = sanitize_input(x_token[d].to_float());
    sum += xv;
  }

  const float mean = sum * inv_d;
  float var_acc = 0.0f;
  REFV3_LN_BASE_VAR_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
    const float xv = sanitize_input(x_token[d].to_float());
    const float delta = xv - mean;
    var_acc += delta * delta;
  }
  const float var_raw = var_acc * inv_d;

  float x_eps_safe = var_raw + eps;
  if (!std::isfinite(x_eps_safe) || x_eps_safe <= 0.0f) {
    x_eps_safe = eps;
  }
  if (!std::isfinite(x_eps_safe) || x_eps_safe <= 0.0f) {
    x_eps_safe = 1.0f;
  }

  float inv_std = REFV3_inv_sqrt_nr1_or_lut(x_eps_safe);
  REFV3_LN_BASE_NR_REFINE_LOOP:
  for (int nr_iter = 0; nr_iter < REFV3_LN_BASELINE_EXTRA_NR_ITERS; ++nr_iter) {
    const float inv_sq = inv_std * inv_std;
    const float inv_nr = inv_std * (1.5f - (0.5f * x_eps_safe * inv_sq));
    if (std::isfinite(inv_nr) && inv_nr > 0.0f) {
      inv_std = inv_nr;
    }
  }
  if (!std::isfinite(inv_std) || inv_std <= 0.0f) {
    inv_std = 1.0f;
  }

  REFV3_LN_BASE_OUT_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
    const float xv = sanitize_input(x_token[d].to_float());
    const float xn = (xv - mean) * inv_std;
    float yi = (xn * static_cast<float>(w[d])) + static_cast<float>(b[d]);
    yi = sanitize_output(yi);
    y_token[d] = refv3_fp_t(yi);
  }
}

} // namespace

RefV3LayerNormBlock::RefV3LayerNormBlock() {}

bool RefV3LayerNormBlock::run(int lid,
                              const RefRunConfig& run_cfg,
                              ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                              ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  RefV3AttentionPayloadHeader header_ref;
  bool header_init = false;
  bool token_seen[REFV3_TOKENS_T];
  refv3_fp_t token_ln_out[REFV3_D_MODEL];
  const double* const ln_weight = (lid == REFV3_LAYER0_ID)
                                    ? w_decoder_layers_0_sublayer_0_norm_weight
                                    : w_decoder_layers_1_sublayer_0_norm_weight;
  const double* const ln_bias = (lid == REFV3_LAYER0_ID)
                                  ? w_decoder_layers_0_sublayer_0_norm_bias
                                  : w_decoder_layers_1_sublayer_0_norm_bias;

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

    layernorm_token_32_local(run_cfg, token_payload.token_vec, ln_weight, ln_bias, token_ln_out);

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

} // namespace ref_v3
} // namespace aecct_ref
