#include "../../include/ref_v2/RefV2FfnLinear0ReluBlock.h"
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

static void quant_linear_token_32_to128_native(
  const ref_fp32_t x[REFV2_D_MODEL],
  const double w[REFV2_FF_DIM * REFV2_D_MODEL],
  const double b[REFV2_FF_DIM],
  float s_x,
  float inv_sxsw_const,
  ref_fp32_t y[REFV2_FF_DIM]) {
  const ref_fp32_t inv(inv_sxsw_const);

  ac_int<8, true> qx_i8[REFV2_D_MODEL];
  REFV2_FFN_L0_QUANT_LOOP: for (int i = 0; i < REFV2_D_MODEL; ++i) {
    qx_i8[i] = quantize_int8_to_i8_local(x[i], s_x);
  }

  REFV2_FFN_L0_OUT_LOOP: for (int o = 0; o < REFV2_FF_DIM; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV2_D_MODEL;
    REFV2_FFN_L0_INNER_LOOP: for (int i = 0; i < REFV2_D_MODEL; ++i) {
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

RefV2FfnLinear0ReluBlock::RefV2FfnLinear0ReluBlock() {}

bool RefV2FfnLinear0ReluBlock::run(
  ac_channel<RefV2AttentionTokenVectorPayload>& in_token_ch,
  ac_channel<RefV2FfnHiddenTokenPayload>& out_hidden_ch) const {
  RefV2AttentionPayloadHeader header_ref;
  bool header_init = false;
  bool token_seen[REFV2_TOKENS_T];
  ref_fp32_t linear0_out_buf[REFV2_FF_DIM];
  const ref_fp32_t zero(0.0f);

  REFV2_FFN_L0_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV2_FFN_L0_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
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

    quant_linear_token_32_to128_native(
      token_payload.token_vec,
      w_decoder_layers_0_feed_forward_w_1_weight,
      w_decoder_layers_0_feed_forward_w_1_bias,
      REFV2_SCALE_L0_FF1_S_X,
      REFV2_INV_L0_FFN_W1,
      linear0_out_buf);

    RefV2FfnHiddenTokenPayload hidden_payload;
    hidden_payload.header = token_payload.header;
    hidden_payload.token_row = token_payload.token_row;
    REFV2_FFN_L0_RELU_LOOP: for (int i = 0; i < REFV2_FF_DIM; ++i) {
      hidden_payload.hidden_vec[i] = (linear0_out_buf[i] < zero) ? zero : linear0_out_buf[i];
    }
    out_hidden_ch.write(hidden_payload);
  }

  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
