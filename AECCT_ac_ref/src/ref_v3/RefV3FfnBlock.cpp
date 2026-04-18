#include "../../include/ref_v3/RefV3FfnBlock.h"
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
  REFV3_FFN_W1_QUANT_LOOP: for (int i = 0; i < REFV3_D_MODEL; ++i) {
    qx_i8[i] = refv3_quantize_int8_local(x[i], s_x);
  }

  REFV3_FFN_W1_OUT_LOOP: for (int o = 0; o < REFV3_FF_DIM; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV3_D_MODEL;
    REFV3_FFN_W1_INNER_LOOP: for (int i = 0; i < REFV3_D_MODEL; ++i) {
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

static void quant_linear_token_128_to32_native(
  const refv3_fp_t x[REFV3_FF_DIM],
  const RefV3TernaryLinearParams& params,
  refv3_fp_t s_x,
  refv3_fp_t inv_sxsw_const,
  refv3_fp_t y[REFV3_D_MODEL]) {
  const refv3_fp_t inv = inv_sxsw_const;

  ac_int<8, true> qx_i8[REFV3_FF_DIM];
  REFV3_FFN_W2_QUANT_LOOP: for (int i = 0; i < REFV3_FF_DIM; ++i) {
    qx_i8[i] = refv3_quantize_int8_local(x[i], s_x);
  }

  REFV3_FFN_W2_OUT_LOOP: for (int o = 0; o < REFV3_D_MODEL; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV3_FF_DIM;
    REFV3_FFN_W2_INNER_LOOP: for (int i = 0; i < REFV3_FF_DIM; ++i) {
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

RefV3FfnBlock::RefV3FfnBlock() {}

bool RefV3FfnBlock::run(ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                        ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const {
  RefV3AttentionPayloadHeader header_ref;
  bool header_init = false;
  bool token_seen[REFV3_TOKENS_T];
  refv3_fp_t ffn1_token_buf[REFV3_FF_DIM];
  refv3_fp_t ffn2_token_buf[REFV3_D_MODEL];
  const RefV3TernaryLinearParams ff1_params = refv3_ffn_w1_params_fp_local_only(REFV3_LAYER0_ID);
  const RefV3TernaryLinearParams ff2_params = refv3_ffn_w2_params_fp_local_only(REFV3_LAYER0_ID);
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
      ff1_params,
      refv3_fp_t(REFV3_SCALE_L0_FF1_S_X),
      refv3_fp_t(REFV3_INV_L0_FFN_W1),
      ffn1_token_buf);

    REFV3_FFN_RELU_LOOP: for (int i = 0; i < REFV3_FF_DIM; ++i) {
      if (ffn1_token_buf[i] < zero) {
        ffn1_token_buf[i] = zero;
      }
    }

    quant_linear_token_128_to32_native(
      ffn1_token_buf,
      ff2_params,
      refv3_fp_t(REFV3_SCALE_L0_FF2_S_X),
      refv3_fp_t(REFV3_INV_L0_FFN_W2),
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
