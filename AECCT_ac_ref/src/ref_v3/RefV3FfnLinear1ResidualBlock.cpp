#include "../../include/ref_v3/RefV3FfnLinear1ResidualBlock.h"
#include "../../include/ref_v3/RefV3MathApprox.h"
#include "../../include/ref_v3/RefV3WeightsFp16LocalOnly.h"

namespace aecct_ref {
namespace ref_v3 {
namespace {

static inline ac_int<16, true> accumulate_ternary_mac_i16_local(
  ac_int<16, true> acc_i16,
  ac_int<8, true> qx_i8,
  ac_int<2, true> ternary_sign_i2) {
  const ac_int<16, true> mac_i16 = qx_i8 * ternary_sign_i2;
  return acc_i16 + mac_i16;
}

static void quant_linear_token_128_to32_native(
  const refv3_fp_t x[REFV3_FF_DIM],
  const RefV3TernaryLinearParams& params,
  refv3_fp_t s_x,
  refv3_fp_t inv_sxsw_const,
  refv3_fp_t y[REFV3_D_MODEL]) {
  const refv3_fp_t inv = inv_sxsw_const;

  ac_int<8, true> qx_i8[REFV3_FF_DIM];
  REFV3_FFN_L1_QUANT_LOOP: for (int i = 0; i < REFV3_FF_DIM; ++i) {
    qx_i8[i] = refv3_quantize_int8_local(x[i], s_x);
  }

  REFV3_FFN_L1_OUT_LOOP: for (int o = 0; o < REFV3_D_MODEL; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV3_FF_DIM;
    REFV3_FFN_L1_INNER_LOOP: for (int i = 0; i < REFV3_FF_DIM; ++i) {
      acc_i16 = accumulate_ternary_mac_i16_local(
        acc_i16,
        qx_i8[i],
        refv3_ternary_weight_sign_at(params, base + i));
    }
    const refv3_fp_t bias_island = refv3_linear_bias_fp_at(params, o);
    const refv3_fp_t acc_island(acc_i16.to_int());
    y[o] = bias_island + (acc_island * inv);
  }
}

} // namespace

RefV3FfnLinear1ResidualBlock::RefV3FfnLinear1ResidualBlock() {}

bool RefV3FfnLinear1ResidualBlock::run(
  int lid,
  ac_channel<RefV3FfnHiddenTokenPayload>& in_hidden_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& in_residual_token_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  const RefV3TernaryLinearParams ff2_params = refv3_ffn_w2_params_fp_local_only(lid);
  const refv3_fp_t ff2_s_x = refv3_ffn_w2_s_x_fp_local_only(lid);
  const refv3_fp_t ff2_s_w = refv3_ffn_w2_s_w_fp_local_only(lid);
  const refv3_fp_t inv_ffn_w2 = refv3_fp_t(1.0f) / (ff2_s_x * ff2_s_w);

  bool token_seen[REFV3_TOKENS_T];
  refv3_fp_t linear1_out_buf[REFV3_D_MODEL];

  REFV3_FFN_L1_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_FFN_L1_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3FfnHiddenTokenPayload hidden_payload = in_hidden_ch.read();
    const RefV3AttentionTokenVectorPayload residual_payload = in_residual_token_ch.read();

    if (!REFV3_payload_header_matches_shape(hidden_payload.header) ||
        !REFV3_payload_header_matches_shape(residual_payload.header)) {
      return false;
    }
    if (hidden_payload.header.layer_id.to_int() != expected_layer_id ||
        residual_payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }
    if (hidden_payload.header.layer_id != residual_payload.header.layer_id ||
        hidden_payload.header.token_rows != residual_payload.header.token_rows ||
        hidden_payload.header.dim_cols != residual_payload.header.dim_cols) {
      return false;
    }
    if (hidden_payload.token_row != residual_payload.token_row) {
      return false;
    }

    const int token = residual_payload.token_row.to_int();
    if (token < 0 || token >= REFV3_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    quant_linear_token_128_to32_native(
      hidden_payload.hidden_vec,
      ff2_params,
      ff2_s_x,
      inv_ffn_w2,
      linear1_out_buf);

    RefV3AttentionTokenVectorPayload out_payload;
    out_payload.header = residual_payload.header;
    out_payload.token_row = residual_payload.token_row;
    REFV3_FFN_L1_TOKEN_OUT_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      out_payload.token_vec[dim] = linear1_out_buf[dim] + residual_payload.token_vec[dim];
    }
    out_token_ch.write(out_payload);
  }

  return true;
}

} // namespace ref_v3
} // namespace aecct_ref
