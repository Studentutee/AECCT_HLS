#include "../../include/ref_v3/RefV3AttenKvBlock.h"
#include "../../include/ref_v3/RefV3MathApprox.h"

#include "weights.h"

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

static void quant_linear_token_32_to32_native(
  const refv3_fp_t x[REFV3_D_MODEL],
  const double w[REFV3_D_MODEL * REFV3_D_MODEL],
  const double b[REFV3_D_MODEL],
  refv3_fp_t s_x,
  refv3_fp_t inv_sxsw_const,
  refv3_fp_t y[REFV3_D_MODEL]) {
  const refv3_fp_t inv = inv_sxsw_const;

  ac_int<8, true> qx_i8[REFV3_D_MODEL];
  KV_QUANTIZE_I8_LOOP: for (int i = 0; i < REFV3_D_MODEL; ++i) {
    qx_i8[i] = refv3_quantize_int8_local(x[i], s_x);
  }

  KV_LINEAR_OUTER_LOOP: for (int o = 0; o < REFV3_D_MODEL; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV3_D_MODEL;
    KV_LINEAR_INNER_LOOP: for (int i = 0; i < REFV3_D_MODEL; ++i) {
      acc_i16 = accumulate_ternary_mac_i16_local(
        acc_i16,
        qx_i8[i],
        refv3_decode_ternary_weight_sign_i2_local(w[base + i]));
    }
    const refv3_fp_t bias_island = refv3_fp_from_double(b[o]);
    const refv3_fp_t acc_island(acc_i16.to_int());
    y[o] = bias_island + (acc_island * inv);
  }
}

} // namespace

RefV3AttenKvBlock::RefV3AttenKvBlock() {}

bool RefV3AttenKvBlock::run(int lid,
                            ac_channel<RefV3AttentionTokenVectorPayload>& in_x_token_ch,
                            ac_channel<RefV3AttentionKPayload>& out_k_payload_ch,
                            ac_channel<RefV3AttentionVPayload>& out_v_payload_ch) const {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  const refv3_fp_t s_x_in = (lid == REFV3_LAYER0_ID)
                              ? refv3_fp_from_double(l0_in_s_x)
                              : refv3_fp_from_double(l1_in_s_x);
  const double* const k_weight = (lid == REFV3_LAYER0_ID)
                                   ? w_decoder_layers_0_self_attn_linears_1_weight
                                   : w_decoder_layers_1_self_attn_linears_1_weight;
  const double* const k_bias = (lid == REFV3_LAYER0_ID)
                                 ? w_decoder_layers_0_self_attn_linears_1_bias
                                 : w_decoder_layers_1_self_attn_linears_1_bias;
  const double* const v_weight = (lid == REFV3_LAYER0_ID)
                                   ? w_decoder_layers_0_self_attn_linears_2_weight
                                   : w_decoder_layers_1_self_attn_linears_2_weight;
  const double* const v_bias = (lid == REFV3_LAYER0_ID)
                                 ? w_decoder_layers_0_self_attn_linears_2_bias
                                 : w_decoder_layers_1_self_attn_linears_2_bias;
  const refv3_fp_t k_s_w = (lid == REFV3_LAYER0_ID)
                             ? refv3_fp_from_double(w_decoder_layers_0_self_attn_linears_1_s_w[0])
                             : refv3_fp_from_double(w_decoder_layers_1_self_attn_linears_1_s_w[0]);
  const refv3_fp_t v_s_w = (lid == REFV3_LAYER0_ID)
                             ? refv3_fp_from_double(w_decoder_layers_0_self_attn_linears_2_s_w[0])
                             : refv3_fp_from_double(w_decoder_layers_1_self_attn_linears_2_s_w[0]);
  const refv3_fp_t inv_attn_k = refv3_fp_t(1.0f) / (s_x_in * k_s_w);
  const refv3_fp_t inv_attn_v = refv3_fp_t(1.0f) / (s_x_in * v_s_w);

  RefV3AttentionKPayload out_k_payload;
  RefV3AttentionVPayload out_v_payload;
  RefV3AttentionPayloadHeader header_ref;
  bool header_init = false;
  refv3_fp_t in_token_buf[REFV3_D_MODEL];

  bool token_seen[REFV3_TOKENS_T];
  REFV3_KV_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_KV_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3AttentionTokenVectorPayload token_payload = in_x_token_ch.read();
    if (!REFV3_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }
    if (!header_init) {
      header_ref = token_payload.header;
      out_k_payload.header = header_ref;
      out_v_payload.header = header_ref;
      header_init = true;
    } else {
      if (token_payload.header.layer_id != header_ref.layer_id ||
          token_payload.header.token_rows != header_ref.token_rows ||
          token_payload.header.dim_cols != header_ref.dim_cols) {
        return false;
      }
    }

    const int token_row = token_payload.token_row.to_int();
    if (token_row < 0 || token_row >= REFV3_TOKENS_T) {
      return false;
    }
    if (token_seen[token_row]) {
      return false;
    }
    token_seen[token_row] = true;

    REFV3_KV_TOKEN_COPY_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      in_token_buf[dim] = token_payload.token_vec[dim];
    }

    const int base = token_row * REFV3_D_MODEL;
    quant_linear_token_32_to32_native(
      in_token_buf,
      k_weight,
      k_bias,
      s_x_in,
      inv_attn_k,
      &out_k_payload.k_flat[base]);
    quant_linear_token_32_to32_native(
      in_token_buf,
      v_weight,
      v_bias,
      s_x_in,
      inv_attn_v,
      &out_v_payload.v_flat[base]);
  }

  out_k_payload_ch.write(out_k_payload);
  out_v_payload_ch.write(out_v_payload);

  return true;
}

} // namespace ref_v3
} // namespace aecct_ref
