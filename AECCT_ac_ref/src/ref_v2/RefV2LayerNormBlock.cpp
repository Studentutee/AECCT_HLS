#include "../../include/ref_v2/RefV2LayerNormBlock.h"
#include "../../include/ref_v2/RefV2MathApprox.h"

#include <cmath>

#include "weights.h"

namespace aecct_ref {
namespace ref_v2 {
namespace {

static void layernorm_token_32_local(
  const RefRunConfig& run_cfg,
  const ref_fp32_t x_token[REFV2_D_MODEL],
  const double w[REFV2_D_MODEL],
  const double b[REFV2_D_MODEL],
  ref_fp32_t y_token[REFV2_D_MODEL]) {
  const float eps = 1.0e-5f;
  const float inv_d = REFV2_INV_D_MODEL;

  auto sanitize_input = [](float v) -> float {
    return std::isfinite(v) ? v : 0.0f;
  };
  auto sanitize_output = [](float v) -> float {
    return std::isfinite(v) ? v : 0.0f;
  };

  if (run_cfg.legacy.ln_mode == RefLayerNormMode::LN_EXACT_REFERENCE) {
    double sum = 0.0;
    REFV2_LN_EXACT_SUM_LOOP: for (int d = 0; d < REFV2_D_MODEL; ++d) {
      sum += static_cast<double>(x_token[d].to_float());
    }
    const double mean = sum / static_cast<double>(REFV2_D_MODEL);

    double var_acc = 0.0;
    REFV2_LN_EXACT_VAR_LOOP: for (int d = 0; d < REFV2_D_MODEL; ++d) {
      const double dv = static_cast<double>(x_token[d].to_float()) - mean;
      var_acc += (dv * dv);
    }
    const double var = var_acc / static_cast<double>(REFV2_D_MODEL);
    const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

    REFV2_LN_EXACT_OUT_LOOP: for (int d = 0; d < REFV2_D_MODEL; ++d) {
      const double xv = static_cast<double>(x_token[d].to_float());
      const double xn = (xv - mean) * inv_std;
      const double yi = (xn * w[d]) + b[d];
      y_token[d] = ref_fp32_t(static_cast<float>(yi));
    }
    return;
  }

  if (run_cfg.legacy.ln_mode == RefLayerNormMode::LN_SUM_SUMSQ_APPROX) {
    float sum = 0.0f;
    float sumsq = 0.0f;
    REFV2_LN_SUMSQ_ACC_LOOP: for (int d = 0; d < REFV2_D_MODEL; ++d) {
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

    float inv_std = refv2_inv_sqrt_nr1_or_lut(var_eps);
    inv_std = sanitize_output(inv_std);

    REFV2_LN_SUMSQ_OUT_LOOP: for (int d = 0; d < REFV2_D_MODEL; ++d) {
      const float xv = sanitize_input(x_token[d].to_float());
      const float xn = (xv - mean) * inv_std;
      const float yi = (xn * static_cast<float>(w[d])) + static_cast<float>(b[d]);
      y_token[d] = ref_fp32_t(sanitize_output(yi));
    }
    return;
  }

  float sum = 0.0f;
  REFV2_LN_BASE_SUM_LOOP: for (int d = 0; d < REFV2_D_MODEL; ++d) {
    const float xv = sanitize_input(x_token[d].to_float());
    sum += xv;
  }

  const float mean = sum * inv_d;
  float var_acc = 0.0f;
  REFV2_LN_BASE_VAR_LOOP: for (int d = 0; d < REFV2_D_MODEL; ++d) {
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

  float inv_std = refv2_inv_sqrt_nr1_or_lut(x_eps_safe);
  REFV2_LN_BASE_NR_REFINE_LOOP: for (int nr_iter = 0; nr_iter < 6; ++nr_iter) {
    const float inv_sq = inv_std * inv_std;
    const float inv_nr = inv_std * (1.5f - (0.5f * x_eps_safe * inv_sq));
    if (std::isfinite(inv_nr) && inv_nr > 0.0f) {
      inv_std = inv_nr;
    }
  }
  if (!std::isfinite(inv_std) || inv_std <= 0.0f) {
    inv_std = 1.0f;
  }

  REFV2_LN_BASE_OUT_LOOP: for (int d = 0; d < REFV2_D_MODEL; ++d) {
    const float xv = sanitize_input(x_token[d].to_float());
    const float xn = (xv - mean) * inv_std;
    float yi = (xn * static_cast<float>(w[d])) + static_cast<float>(b[d]);
    yi = sanitize_output(yi);
    y_token[d] = ref_fp32_t(yi);
  }
}

} // namespace

RefV2LayerNormBlock::RefV2LayerNormBlock() {}

bool RefV2LayerNormBlock::run(const RefRunConfig& run_cfg,
                              ac_channel<RefV2AttentionTokenVectorPayload>& in_token_ch,
                              ac_channel<RefV2AttentionTokenVectorPayload>& out_token_ch) const {
  RefV2AttentionPayloadHeader header_ref;
  bool header_init = false;
  bool token_seen[REFV2_TOKENS_T];
  ref_fp32_t token_matrix[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t token_ln_out[REFV2_D_MODEL];

  REFV2_LN_TOKEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
    REFV2_LN_TOKEN_INIT_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      token_matrix[token][dim] = ref_fp32_t(0.0f);
    }
  }

  REFV2_LN_TOKEN_READ_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2AttentionTokenVectorPayload token_payload = in_token_ch.read();
    if (!refv2_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != REFV2_LAYER0_ID) {
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
    if (token < 0 || token >= REFV2_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV2_LN_TOKEN_PACK_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      token_matrix[token][dim] = token_payload.token_vec[dim];
    }
  }

  const double* const ln0_w = w_decoder_layers_0_sublayer_0_norm_weight;
  const double* const ln0_b = w_decoder_layers_0_sublayer_0_norm_bias;

  REFV2_LN_TOKEN_OUT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    RefV2AttentionTokenVectorPayload token_payload;
    token_payload.header = header_ref;
    token_payload.token_row = ac_int<16, false>(token);

    layernorm_token_32_local(run_cfg, token_matrix[token], ln0_w, ln0_b, token_ln_out);
    REFV2_LN_TOKEN_OUT_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = token_ln_out[dim];
    }
    out_token_ch.write(token_payload);
  }

  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
