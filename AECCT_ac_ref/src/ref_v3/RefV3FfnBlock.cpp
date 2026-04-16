#include "../../include/ref_v3/RefV3FfnBlock.h"
#include "../../include/ref_v3/RefV3MathApprox.h"

#include <cmath>

#include "weights.h"

namespace aecct_ref {
namespace ref_v3 {
namespace {

static inline ac_int<8, true> quantize_int8_to_i8_local(refv3_fp_t x, float s_x) {
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
  const refv3_fp_t x[REFV3_D_MODEL],
  const double w[REFV3_FF_DIM * REFV3_D_MODEL],
  const double b[REFV3_FF_DIM],
  float s_x,
  float inv_sxsw_const,
  refv3_fp_t y[REFV3_FF_DIM]) {
  const refv3_fp_t inv(inv_sxsw_const);

  ac_int<8, true> qx_i8[REFV3_D_MODEL];
  REFV3_FFN_W1_QUANT_LOOP: for (int i = 0; i < REFV3_D_MODEL; ++i) {
    qx_i8[i] = quantize_int8_to_i8_local(x[i], s_x);
  }

  REFV3_FFN_W1_OUT_LOOP: for (int o = 0; o < REFV3_FF_DIM; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV3_D_MODEL;
    REFV3_FFN_W1_INNER_LOOP: for (int i = 0; i < REFV3_D_MODEL; ++i) {
      acc_i16 = accumulate_ternary_mac_i16_local(
        acc_i16,
        qx_i8[i],
        decode_ternary_weight_sign_i2_local(w[base + i]));
    }
    const refv3_fp_t bias_island(static_cast<float>(b[o]));
    const refv3_fp_t acc_island(acc_i16.to_int());
    y[o] = bias_island + (acc_island * inv);
  }
}

static void quant_linear_token_128_to32_native(
  const refv3_fp_t x[REFV3_FF_DIM],
  const double w[REFV3_D_MODEL * REFV3_FF_DIM],
  const double b[REFV3_D_MODEL],
  float s_x,
  float inv_sxsw_const,
  refv3_fp_t y[REFV3_D_MODEL]) {
  const refv3_fp_t inv(inv_sxsw_const);

  ac_int<8, true> qx_i8[REFV3_FF_DIM];
  REFV3_FFN_W2_QUANT_LOOP: for (int i = 0; i < REFV3_FF_DIM; ++i) {
    qx_i8[i] = quantize_int8_to_i8_local(x[i], s_x);
  }

  REFV3_FFN_W2_OUT_LOOP: for (int o = 0; o < REFV3_D_MODEL; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV3_FF_DIM;
    REFV3_FFN_W2_INNER_LOOP: for (int i = 0; i < REFV3_FF_DIM; ++i) {
      acc_i16 = accumulate_ternary_mac_i16_local(
        acc_i16,
        qx_i8[i],
        decode_ternary_weight_sign_i2_local(w[base + i]));
    }
    const refv3_fp_t bias_island(static_cast<float>(b[o]));
    const refv3_fp_t acc_island(acc_i16.to_int());
    y[o] = bias_island + (acc_island * inv);
  }
}

} // namespace

RefV3FfnBlock::RefV3FfnBlock() {}

bool RefV3FfnBlock::run(ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                        ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const {
  RefV3AttentionPayloadHeader header_ref;
  bool header_init = false;
  bool token_seen[REFV3_TOKENS_T];
  refv3_fp_t ffn1_token_buf[REFV3_FF_DIM];
  refv3_fp_t ffn2_token_buf[REFV3_D_MODEL];
  const refv3_fp_t zero(0.0f);

  REFV3_FFN_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_FFN_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3AttentionTokenVectorPayload token_payload = in_token_ch.read();
    if (!REFV3_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != REFV3_LAYER0_ID) {
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

    quant_linear_token_32_to128_native(
      token_payload.token_vec,
      w_decoder_layers_0_feed_forward_w_1_weight,
      w_decoder_layers_0_feed_forward_w_1_bias,
      REFV3_SCALE_L0_FF1_S_X,
      REFV3_INV_L0_FFN_W1,
      ffn1_token_buf);

    REFV3_FFN_RELU_LOOP: for (int i = 0; i < REFV3_FF_DIM; ++i) {
      if (ffn1_token_buf[i] < zero) {
        ffn1_token_buf[i] = zero;
      }
    }

    quant_linear_token_128_to32_native(
      ffn1_token_buf,
      w_decoder_layers_0_feed_forward_w_2_weight,
      w_decoder_layers_0_feed_forward_w_2_bias,
      REFV3_SCALE_L0_FF2_S_X,
      REFV3_INV_L0_FFN_W2,
      ffn2_token_buf);

    RefV3AttentionTokenVectorPayload out_payload;
    out_payload.header = token_payload.header;
    out_payload.token_row = token_payload.token_row;
    REFV3_FFN_TOKEN_OUT_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      out_payload.token_vec[dim] = ffn2_token_buf[dim] + token_payload.token_vec[dim];
    }
    out_token_ch.write(out_payload);
  }

  return true;
}

} // namespace ref_v3
} // namespace aecct_ref
