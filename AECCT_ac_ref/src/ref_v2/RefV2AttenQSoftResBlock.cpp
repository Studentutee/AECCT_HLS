#include "../../include/ref_v2/RefV2AttenQSoftResBlock.h"

#include <cmath>

#include "../../include/SoftmaxApprox.h"
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
  QSOFT_QUANTIZE_I8_LOOP: for (int i = 0; i < REFV2_D_MODEL; ++i) {
    qx_i8[i] = quantize_int8_to_i8_local(x[i], s_x);
  }

  QSOFT_LINEAR_OUTER_LOOP: for (int o = 0; o < REFV2_D_MODEL; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV2_D_MODEL;
    QSOFT_LINEAR_INNER_LOOP: for (int i = 0; i < REFV2_D_MODEL; ++i) {
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

static bool is_layer0_attn_masked_token_pair(int head_idx, int q_token, int k_token) {
  const bool src_masked = (w_src_mask[q_token * REFV2_TOKENS_T + k_token].to_int() != 0);
  const bool q_is_var = (q_token < REFV2_VAR_N);
  const bool k_is_var = (k_token < REFV2_VAR_N);

  bool one_ring_masked = true;
  bool second_ring_masked = true;
  if (q_is_var && k_is_var) {
    one_ring_masked = true;
    second_ring_masked = src_masked;
  } else if (q_is_var != k_is_var) {
    one_ring_masked = src_masked;
    second_ring_masked = true;
  } else {
    one_ring_masked = true;
    second_ring_masked = src_masked;
  }
  return (head_idx < 4) ? one_ring_masked : second_ring_masked;
}

} // namespace

RefV2AttenQSoftResBlock::RefV2AttenQSoftResBlock() {}

bool RefV2AttenQSoftResBlock::run(const RefRunConfig& run_cfg,
                                  ac_channel<RefV2AttentionTokenVectorPayload>& query_token_ch,
                                  ac_channel<RefV2AttentionKPayload>& in_k_payload_ch,
                                  ac_channel<RefV2AttentionVPayload>& in_v_payload_ch,
                                  ac_channel<RefV2AttentionTokenVectorPayload>& out_token_ch) const {
  RefV2AttentionInputPayload query_payload;
  RefV2AttentionKPayload in_k_payload;
  RefV2AttentionVPayload in_v_payload;
  RefV2AttentionOutputPayload out_payload;

  query_payload.header.layer_id = ac_int<8, false>(REFV2_LAYER0_ID);
  query_payload.header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
  query_payload.header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);

  in_k_payload = in_k_payload_ch.read();
  in_v_payload = in_v_payload_ch.read();
  if (!refv2_payload_header_matches_shape(in_k_payload.header) ||
      !refv2_payload_header_matches_shape(in_v_payload.header)) {
    return false;
  }
  if (in_k_payload.header.layer_id.to_int() != REFV2_LAYER0_ID ||
      in_v_payload.header.layer_id.to_int() != REFV2_LAYER0_ID) {
    return false;
  }

  bool query_token_seen[REFV2_TOKENS_T];
  REFV2_QSOFTRES_QUERY_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    query_token_seen[token] = false;
  }

  REFV2_QSOFTRES_QUERY_TOKEN_READ_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2AttentionTokenVectorPayload query_token_payload = query_token_ch.read();
    if (!refv2_payload_header_matches_shape(query_token_payload.header)) {
      return false;
    }
    if (query_token_payload.header.layer_id.to_int() != REFV2_LAYER0_ID) {
      return false;
    }

    const int token_row = query_token_payload.token_row.to_int();
    if (token_row < 0 || token_row >= REFV2_TOKENS_T) {
      return false;
    }
    if (query_token_seen[token_row]) {
      return false;
    }
    query_token_seen[token_row] = true;

    REFV2_QSOFTRES_QUERY_TOKEN_PACK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token_row, dim);
      query_payload.x_flat[idx] = query_token_payload.token_vec[dim];
    }
  }

  out_payload.header = query_payload.header;

  ref_fp32_t q_vec[REFV2_D_MODEL];
  ref_fp32_t head_ctx_buf[REFV2_HEADS][REFV2_D_HEAD];
  ref_fp32_t out_acc_tile[REFV2_D_MODEL];
  ref_fp32_t softmax_acc_tile[REFV2_D_HEAD];
  ref_fp32_t ln_token_buf[REFV2_D_MODEL];

  const float s_x_q = static_cast<float>(l0_in_s_x);
  const float s_x_o = static_cast<float>(l0_o_s_x);
  const float s_w_q = static_cast<float>(w_decoder_layers_0_self_attn_linears_0_s_w[0]);
  const float s_w_o = static_cast<float>(w_decoder_layers_0_self_attn_linears_3_s_w[0]);
  const bool use_softmax_exact =
    (run_cfg.legacy.algo_variant == RefAlgoVariant::RESERVED_SOFTMAX_ALT);

  const ref_fp32_t inv_sqrt_dh(0.5f);
  const ref_fp32_t zero(0.0f);

  REFV2_QSOFTRES_QTOKEN_LOOP: for (int q_token = 0; q_token < REFV2_TOKENS_T; ++q_token) {
    const int q_base = q_token * REFV2_D_MODEL;
    REFV2_QSOFTRES_PREP_DIM_LOOP: for (int d = 0; d < REFV2_D_MODEL; ++d) {
      ln_token_buf[d] = query_payload.x_flat[q_base + d];
      out_acc_tile[d] = zero;
    }

    quant_linear_token_32_to32_native(
      &query_payload.x_flat[q_base],
      w_decoder_layers_0_self_attn_linears_0_weight,
      w_decoder_layers_0_self_attn_linears_0_bias,
      s_x_q,
      s_w_q,
      q_vec);

    REFV2_QSOFTRES_HEAD_LOOP: for (int h = 0; h < REFV2_HEADS; ++h) {
      const int head_base = h * REFV2_D_HEAD;

      if (use_softmax_exact) {
        bool has_valid = false;
        ref_fp32_t max_score = zero;

        REFV2_QSOFTRES_EXACT_MAX_TOKEN_LOOP: for (int k_token = 0; k_token < REFV2_TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          ref_fp32_t dot = zero;
          REFV2_QSOFTRES_EXACT_MAX_DH_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
            const int idx = (k_token * REFV2_D_MODEL) + head_base + dh;
            dot += q_vec[head_base + dh] * in_k_payload.k_flat[idx];
          }
          const ref_fp32_t score = dot * inv_sqrt_dh;
          if (!has_valid || score > max_score) {
            max_score = score;
          }
          has_valid = true;
        }

        if (!has_valid) {
          REFV2_QSOFTRES_EXACT_NO_VALID_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
            head_ctx_buf[h][dh] = zero;
          }
          continue;
        }

        ref_fp32_t sumexp = zero;
        REFV2_QSOFTRES_EXACT_CLR_ACC_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
          softmax_acc_tile[dh] = zero;
        }

        REFV2_QSOFTRES_EXACT_ACC_TOKEN_LOOP: for (int k_token = 0; k_token < REFV2_TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          ref_fp32_t dot = zero;
          REFV2_QSOFTRES_EXACT_ACC_DH_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
            const int idx = (k_token * REFV2_D_MODEL) + head_base + dh;
            dot += q_vec[head_base + dh] * in_k_payload.k_flat[idx];
          }
          const ref_fp32_t score = dot * inv_sqrt_dh;
          const ref_fp32_t w(static_cast<float>(std::exp((score - max_score).to_float())));
          sumexp += w;
          REFV2_QSOFTRES_EXACT_VACC_DH_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
            const int idx = (k_token * REFV2_D_MODEL) + head_base + dh;
            softmax_acc_tile[dh] += w * in_v_payload.v_flat[idx];
          }
        }

        ref_fp32_t inv_sumexp = zero;
        if (sumexp > zero) {
          inv_sumexp = ref_fp32_t(1.0f) / sumexp;
        }
        REFV2_QSOFTRES_EXACT_NORM_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
          head_ctx_buf[h][dh] = softmax_acc_tile[dh] * inv_sumexp;
        }
      } else {
        bool online_init = false;
        ref_fp32_t online_max = zero;
        ref_fp32_t online_sumexp = zero;

        REFV2_QSOFTRES_APPROX_CLR_ACC_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
          softmax_acc_tile[dh] = zero;
        }

        REFV2_QSOFTRES_APPROX_TOKEN_LOOP: for (int k_token = 0; k_token < REFV2_TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }

          ref_fp32_t dot = zero;
          REFV2_QSOFTRES_APPROX_DH_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
            const int idx = (k_token * REFV2_D_MODEL) + head_base + dh;
            dot += q_vec[head_base + dh] * in_k_payload.k_flat[idx];
          }
          const ref_fp32_t score = dot * inv_sqrt_dh;

          if (!online_init) {
            online_max = score;
            online_sumexp = ref_fp32_t(1.0f);
            REFV2_QSOFTRES_APPROX_INIT_ACC_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
              const int idx = (k_token * REFV2_D_MODEL) + head_base + dh;
              softmax_acc_tile[dh] = in_v_payload.v_flat[idx];
            }
            online_init = true;
            continue;
          }

          if (score > online_max) {
            const ref_fp32_t rescale = ref_softmax_exp_dispatch(
              online_max - score,
              run_cfg.legacy.softmax_exp_mode);
            online_sumexp = (online_sumexp * rescale) + ref_fp32_t(1.0f);
            REFV2_QSOFTRES_APPROX_RESCALE_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
              const int idx = (k_token * REFV2_D_MODEL) + head_base + dh;
              softmax_acc_tile[dh] =
                (softmax_acc_tile[dh] * rescale) + in_v_payload.v_flat[idx];
            }
            online_max = score;
            continue;
          }

          const ref_fp32_t w = ref_softmax_exp_dispatch(
            score - online_max,
            run_cfg.legacy.softmax_exp_mode);
          online_sumexp += w;
          REFV2_QSOFTRES_APPROX_ACC_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
            const int idx = (k_token * REFV2_D_MODEL) + head_base + dh;
            softmax_acc_tile[dh] += w * in_v_payload.v_flat[idx];
          }
        }

        if (!online_init) {
          REFV2_QSOFTRES_APPROX_NO_VALID_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
            head_ctx_buf[h][dh] = zero;
          }
          continue;
        }

        const ref_fp32_t inv_sumexp = ref_softmax_rcp_lut(online_sumexp);
        REFV2_QSOFTRES_APPROX_NORM_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
          head_ctx_buf[h][dh] = softmax_acc_tile[dh] * inv_sumexp;
        }
      }
    }

    REFV2_QSOFTRES_PACK_CTX_LOOP: for (int h = 0; h < REFV2_HEADS; ++h) {
      const int head_base = h * REFV2_D_HEAD;
      REFV2_QSOFTRES_PACK_CTX_DH_LOOP: for (int dh = 0; dh < REFV2_D_HEAD; ++dh) {
        q_vec[head_base + dh] = head_ctx_buf[h][dh];
      }
    }

    quant_linear_token_32_to32_native(
      q_vec,
      w_decoder_layers_0_self_attn_linears_3_weight,
      w_decoder_layers_0_self_attn_linears_3_bias,
      s_x_o,
      s_w_o,
      out_acc_tile);

    REFV2_QSOFTRES_WRITEBACK_LOOP: for (int d = 0; d < REFV2_D_MODEL; ++d) {
      out_payload.out_flat[q_base + d] = out_acc_tile[d] + ln_token_buf[d];
    }
  }

  // Boundary ownership: Top owns X_WORK write-back, this block only streams token vectors out.
  REFV2_QSOFTRES_OUT_TOKEN_STREAM_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    RefV2AttentionTokenVectorPayload out_token_payload;
    out_token_payload.header = out_payload.header;
    out_token_payload.token_row = ac_int<16, false>(token);

    REFV2_QSOFTRES_OUT_TOKEN_STREAM_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      out_token_payload.token_vec[dim] = out_payload.out_flat[idx];
    }
    out_token_ch.write(out_token_payload);
  }

  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
