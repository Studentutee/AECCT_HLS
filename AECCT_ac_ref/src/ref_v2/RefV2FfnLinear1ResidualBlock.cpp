#include "../../include/ref_v2/RefV2FfnLinear1ResidualBlock.h"
#include "../../include/ref_v2/RefV2MathApprox.h"

#include <cmath>

#include "weights.h"

namespace aecct_ref {
namespace ref_v2 {
namespace {

static inline ac_int<8, true> quantize_int8_to_i8_local(ref_fp32_t x, float s_x) {
  const double scaled = static_cast<double>(x.to_float()) * static_cast<double>(s_x);
  if (scaled >= 127.0) {
    return ac_int<8, true>(127);
  }
  if (scaled <= -127.0) {
    return ac_int<8, true>(-127);
  }
  return ac_int<8, true>(static_cast<signed char>(std::lround(scaled)));
}

static inline ac_int<2, true> decode_ternary_weight_sign_i2_local(double w) {
  if (w == 1.0 || w == 1.0f) return ac_int<2, true>(1);
  if (w == -1.0 || w == -1.0f) return ac_int<2, true>(-1);
  if (w == 0.0 || w == -0.0 || w == 0.0f || w == -0.0f) return ac_int<2, true>(0);
  if (w >= 0.5) return ac_int<2, true>(1);
  if (w <= -0.5) return ac_int<2, true>(-1);
  return ac_int<2, true>(0);
}

static inline ac_int<16, true> accumulate_ternary_mac_i16_local(
  ac_int<16, true> acc_i16,
  ac_int<8, true> qx_i8,
  ac_int<2, true> ternary_sign_i2) {
  const ac_int<16, true> mac_i16 = qx_i8 * ternary_sign_i2;
  return acc_i16 + mac_i16;
}

static void quant_linear_token_128_to32_native(
  const ref_fp32_t x[REFV2_FF_DIM],
  const double w[REFV2_D_MODEL * REFV2_FF_DIM],
  const double b[REFV2_D_MODEL],
  float s_x,
  float inv_sxsw_const,
  ref_fp32_t y[REFV2_D_MODEL]) {
  const ref_fp32_t inv(inv_sxsw_const);

  ac_int<8, true> qx_i8[REFV2_FF_DIM];
  REFV2_FFN_L1_QUANT_LOOP: for (int i = 0; i < REFV2_FF_DIM; ++i) {
    qx_i8[i] = quantize_int8_to_i8_local(x[i], s_x);
  }

  REFV2_FFN_L1_OUT_LOOP: for (int o = 0; o < REFV2_D_MODEL; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV2_FF_DIM;
    REFV2_FFN_L1_INNER_LOOP: for (int i = 0; i < REFV2_FF_DIM; ++i) {
      acc_i16 = accumulate_ternary_mac_i16_local(
        acc_i16,
        qx_i8[i],
        decode_ternary_weight_sign_i2_local(w[base + i]));
    }
    const ref_fp32_t bias_island(static_cast<float>(b[o]));
    const ref_fp32_t acc_island(acc_i16.to_int());
    y[o] = bias_island + (acc_island * inv);
  }
}

} // namespace

RefV2FfnLinear1ResidualBlock::RefV2FfnLinear1ResidualBlock() {}

bool RefV2FfnLinear1ResidualBlock::run(
  int lid,
  ac_channel<RefV2FfnHiddenTokenPayload>& in_hidden_ch,
  ac_channel<RefV2AttentionTokenVectorPayload>& in_residual_token_ch,
  ac_channel<RefV2AttentionTokenVectorPayload>& out_token_ch) const {
  if (lid != REFV2_LAYER0_ID && lid != REFV2_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  const double* const ff2_weight = (lid == REFV2_LAYER0_ID)
                                     ? w_decoder_layers_0_feed_forward_w_2_weight
                                     : w_decoder_layers_1_feed_forward_w_2_weight;
  const double* const ff2_bias = (lid == REFV2_LAYER0_ID)
                                   ? w_decoder_layers_0_feed_forward_w_2_bias
                                   : w_decoder_layers_1_feed_forward_w_2_bias;
  const float ff2_s_x = (lid == REFV2_LAYER0_ID)
                          ? static_cast<float>(l0_ff2_s_x)
                          : static_cast<float>(l1_ff2_s_x);
  const float ff2_s_w = (lid == REFV2_LAYER0_ID)
                          ? static_cast<float>(w_decoder_layers_0_feed_forward_w_2_s_w[0])
                          : static_cast<float>(w_decoder_layers_1_feed_forward_w_2_s_w[0]);
  const float inv_ffn_w2 = 1.0f / (ff2_s_x * ff2_s_w);

  bool token_seen[REFV2_TOKENS_T];
  ref_fp32_t linear1_out_buf[REFV2_D_MODEL];

  REFV2_FFN_L1_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV2_FFN_L1_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2FfnHiddenTokenPayload hidden_payload = in_hidden_ch.read();
    const RefV2AttentionTokenVectorPayload residual_payload = in_residual_token_ch.read();

    if (!refv2_payload_header_matches_shape(hidden_payload.header) ||
        !refv2_payload_header_matches_shape(residual_payload.header)) {
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
    if (token < 0 || token >= REFV2_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    quant_linear_token_128_to32_native(
      hidden_payload.hidden_vec,
      ff2_weight,
      ff2_bias,
      ff2_s_x,
      inv_ffn_w2,
      linear1_out_buf);

    RefV2AttentionTokenVectorPayload out_payload;
    out_payload.header = residual_payload.header;
    out_payload.token_row = residual_payload.token_row;
    REFV2_FFN_L1_TOKEN_OUT_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      out_payload.token_vec[dim] = linear1_out_buf[dim] + residual_payload.token_vec[dim];
    }
    out_token_ch.write(out_payload);
  }

  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
