#include "../../include/ref_v3/RefV3FfnLinear0ReluBlock.h"
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

static void quant_linear_token_32_to128_native(
  const refv3_fp_t x[REFV3_D_MODEL],
  const RefV3TernaryLinearParams& params,
  refv3_fp_t s_x,
  refv3_fp_t inv_sxsw_const,
  refv3_fp_t y[REFV3_FF_DIM]) {
  const refv3_fp_t inv = inv_sxsw_const;

  ac_int<8, true> qx_i8[REFV3_D_MODEL];
  REFV3_FFN_L0_QUANT_LOOP: for (int i = 0; i < REFV3_D_MODEL; ++i) {
    qx_i8[i] = refv3_quantize_int8_local(x[i], s_x);
  }

  REFV3_FFN_L0_OUT_LOOP: for (int o = 0; o < REFV3_FF_DIM; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV3_D_MODEL;
    REFV3_FFN_L0_INNER_LOOP: for (int i = 0; i < REFV3_D_MODEL; ++i) {
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

RefV3FfnLinear0ReluBlock::RefV3FfnLinear0ReluBlock() {}

bool RefV3FfnLinear0ReluBlock::run(
  int lid,
  ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
  ac_channel<RefV3FfnHiddenTokenPayload>& out_hidden_ch) const {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  const RefV3TernaryLinearParams ff1_params = refv3_ffn_w1_params_fp_local_only(lid);
  const refv3_fp_t ff1_s_x = refv3_ffn_w1_s_x_fp_local_only(lid);
  const refv3_fp_t ff1_s_w = refv3_ffn_w1_s_w_fp_local_only(lid);
  const refv3_fp_t inv_ffn_w1 = refv3_fp_t(1.0f) / (ff1_s_x * ff1_s_w);

  RefV3AttentionPayloadHeader header_ref;
  bool header_init = false;
  bool token_seen[REFV3_TOKENS_T];
  refv3_fp_t linear0_out_buf[REFV3_FF_DIM];
  const refv3_fp_t zero(0.0f);

  REFV3_FFN_L0_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_FFN_L0_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
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

    quant_linear_token_32_to128_native(
      token_payload.token_vec,
      ff1_params,
      ff1_s_x,
      inv_ffn_w1,
      linear0_out_buf);

    RefV3FfnHiddenTokenPayload hidden_payload;
    hidden_payload.header = token_payload.header;
    hidden_payload.token_row = token_payload.token_row;
    REFV3_FFN_L0_RELU_LOOP: for (int i = 0; i < REFV3_FF_DIM; ++i) {
      hidden_payload.hidden_vec[i] = (linear0_out_buf[i] < zero) ? zero : linear0_out_buf[i];
    }
    out_hidden_ch.write(hidden_payload);
  }

  return true;
}

} // namespace ref_v3
} // namespace aecct_ref
