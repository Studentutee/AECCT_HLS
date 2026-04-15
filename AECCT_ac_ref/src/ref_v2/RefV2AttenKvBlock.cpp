#include "../../include/ref_v2/RefV2AttenKvBlock.h"

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

static void quant_linear_token_32_to32_native(
  const ref_fp32_t x[REFV2_D_MODEL],
  const double w[REFV2_D_MODEL * REFV2_D_MODEL],
  const double b[REFV2_D_MODEL],
  float s_x,
  float s_w,
  ref_fp32_t y[REFV2_D_MODEL]) {
  const ref_fp32_t inv =
    ref_fp32_t(1.0f) / (ref_fp32_t(s_x) * ref_fp32_t(s_w));

  ac_int<8, true> qx_i8[REFV2_D_MODEL];
  KV_QUANTIZE_I8_LOOP: for (int i = 0; i < REFV2_D_MODEL; ++i) {
    qx_i8[i] = quantize_int8_to_i8_local(x[i], s_x);
  }

  KV_LINEAR_OUTER_LOOP: for (int o = 0; o < REFV2_D_MODEL; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV2_D_MODEL;
    KV_LINEAR_INNER_LOOP: for (int i = 0; i < REFV2_D_MODEL; ++i) {
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

RefV2AttenKvBlock::RefV2AttenKvBlock() {}

bool RefV2AttenKvBlock::run(const RefV2AttentionInputPayload& in_payload,
                            RefV2AttentionKPayload* out_k_payload,
                            RefV2AttentionVPayload* out_v_payload) const {
  if (out_k_payload == 0 || out_v_payload == 0) {
    return false;
  }
  if (!refv2_payload_header_matches_shape(in_payload.header)) {
    return false;
  }
  if (in_payload.header.layer_id.to_int() != REFV2_LAYER0_ID) {
    return false;
  }

  out_k_payload->header = in_payload.header;
  out_v_payload->header = in_payload.header;

  const float s_x_in = static_cast<float>(l0_in_s_x);
  const float s_w_k = static_cast<float>(w_decoder_layers_0_self_attn_linears_1_s_w[0]);
  const float s_w_v = static_cast<float>(w_decoder_layers_0_self_attn_linears_2_s_w[0]);

  REFV2_KV_BLOCK_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    const int base = token * REFV2_D_MODEL;
    quant_linear_token_32_to32_native(
      &in_payload.x_flat[base],
      w_decoder_layers_0_self_attn_linears_1_weight,
      w_decoder_layers_0_self_attn_linears_1_bias,
      s_x_in,
      s_w_k,
      &out_k_payload->k_flat[base]);
    quant_linear_token_32_to32_native(
      &in_payload.x_flat[base],
      w_decoder_layers_0_self_attn_linears_2_weight,
      w_decoder_layers_0_self_attn_linears_2_bias,
      s_x_in,
      s_w_v,
      &out_v_payload->v_flat[base]);
  }

  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
