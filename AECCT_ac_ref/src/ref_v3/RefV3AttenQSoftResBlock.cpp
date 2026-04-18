#include "../../include/ref_v3/RefV3AttenQSoftResBlock.h"
#include "../../include/ref_v3/RefV3MathApprox.h"
#include "../../include/ref_v3/RefV3WeightsFp16LocalOnly.h"

#include <cmath>

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
  const RefV3TernaryLinearParams& params,
  refv3_fp_t s_x,
  refv3_fp_t inv_sxsw_const,
  refv3_fp_t y[REFV3_D_MODEL]) {
  const refv3_fp_t inv = inv_sxsw_const;

  ac_int<8, true> qx_i8[REFV3_D_MODEL];
  QSOFT_QUANTIZE_I8_LOOP: for (int i = 0; i < REFV3_D_MODEL; ++i) {
    qx_i8[i] = refv3_quantize_int8_local(x[i], s_x);
  }

  QSOFT_LINEAR_OUTER_LOOP: for (int o = 0; o < REFV3_D_MODEL; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * REFV3_D_MODEL;
    QSOFT_LINEAR_INNER_LOOP: for (int i = 0; i < REFV3_D_MODEL; ++i) {
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

static bool is_layer0_attn_masked_token_pair(int head_idx, int q_token, int k_token) {
  const bool src_masked = (w_src_mask[q_token * REFV3_TOKENS_T + k_token].to_int() != 0);
  const bool q_is_var = (q_token < REFV3_VAR_N);
  const bool k_is_var = (k_token < REFV3_VAR_N);

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

RefV3AttenQSoftResBlock::RefV3AttenQSoftResBlock() {}

bool RefV3AttenQSoftResBlock::run(int lid,
                                  const RefRunConfig& run_cfg,
                                  ac_channel<RefV3AttentionInputPayload>& in_xwork_ch,
                                  ac_channel<RefV3AttentionKPayload>& in_k_payload_ch,
                                  ac_channel<RefV3AttentionVPayload>& in_v_payload_ch,
                                  ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  const RefV3AttentionInputPayload xwork_payload = in_xwork_ch.read();
  if (!REFV3_payload_header_matches_shape(xwork_payload.header)) {
    return false;
  }
  if (xwork_payload.header.layer_id.to_int() != expected_layer_id) {
    return false;
  }

  const refv3_fp_t s_x_q = refv3_attn_input_s_x_fp_local_only(lid);
  const refv3_fp_t s_x_o = refv3_attn_output_s_x_fp_local_only(lid);
  const RefV3TernaryLinearParams q_params = refv3_attn_linear_params_fp_local_only(lid, 0);
  const RefV3TernaryLinearParams o_params = refv3_attn_linear_params_fp_local_only(lid, 3);
  const refv3_fp_t inv_attn_q = REFV3_attn_inv_sxsw_const(lid, 0);
  const refv3_fp_t inv_attn_o = REFV3_attn_inv_sxsw_const(lid, 3);

  RefV3AttentionKPayload in_k_payload;
  RefV3AttentionVPayload in_v_payload;
  const RefV3AttentionPayloadHeader header_ref = xwork_payload.header;

  in_k_payload = in_k_payload_ch.read();
  in_v_payload = in_v_payload_ch.read();
  if (!REFV3_payload_header_matches_shape(in_k_payload.header) ||
      !REFV3_payload_header_matches_shape(in_v_payload.header)) {
    return false;
  }
  if (in_k_payload.header.layer_id.to_int() != expected_layer_id ||
      in_v_payload.header.layer_id.to_int() != expected_layer_id) {
    return false;
  }
  if (in_k_payload.header.layer_id != xwork_payload.header.layer_id ||
      in_k_payload.header.token_rows != xwork_payload.header.token_rows ||
      in_k_payload.header.dim_cols != xwork_payload.header.dim_cols ||
      in_v_payload.header.layer_id != xwork_payload.header.layer_id) {
    return false;
  }
  if (in_v_payload.header.token_rows != xwork_payload.header.token_rows ||
      in_v_payload.header.dim_cols != xwork_payload.header.dim_cols) {
    return false;
  }

  refv3_fp_t q_vec[REFV3_D_MODEL];
  refv3_fp_t head_ctx_buf[REFV3_HEADS][REFV3_D_HEAD];
  refv3_fp_t out_acc_tile[REFV3_D_MODEL];
  refv3_fp_t softmax_acc_tile[REFV3_D_HEAD];
  refv3_fp_t query_token_buf[REFV3_D_MODEL];
  refv3_fp_t ln_token_buf[REFV3_D_MODEL];

#if !defined(__SYNTHESIS__) && !defined(REFV3_SYNTH_ONLY)
  const bool use_softmax_exact =
    (run_cfg.legacy.algo_variant == RefAlgoVariant::RESERVED_SOFTMAX_ALT);
#else
  (void)run_cfg;
#endif

  const refv3_fp_t inv_sqrt_dh(0.5f);
  const refv3_fp_t zero(0.0f);

  // Attention residual base is sourced from full-matrix X_WORK, not query token FIFO.
  REFV3_QSOFTRES_TOKEN_LOOP: for (int q_token = 0; q_token < REFV3_TOKENS_T; ++q_token) {
    REFV3_QSOFTRES_PREP_DIM_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      const int x_idx = REFV3_flatten_row_major_index(q_token, d);
      const refv3_fp_t x_token_value = xwork_payload.x_flat[x_idx];
      query_token_buf[d] = x_token_value;
      ln_token_buf[d] = x_token_value;
      out_acc_tile[d] = zero;
    }

    quant_linear_token_32_to32_native(
      query_token_buf,
      q_params,
      s_x_q,
      inv_attn_q,
      q_vec);

    REFV3_QSOFTRES_HEAD_LOOP: for (int h = 0; h < REFV3_HEADS; ++h) {
      const int head_base = h * REFV3_D_HEAD;

// Host-only exact reference path; synthesis surface must stay on LUT-based softmax approximation.
#if !defined(__SYNTHESIS__) && !defined(REFV3_SYNTH_ONLY)
      if (use_softmax_exact) {
        bool has_valid = false;
        refv3_fp_t max_score = zero;

        REFV3_QSOFTRES_EXACT_MAX_TOKEN_LOOP: for (int k_token = 0; k_token < REFV3_TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          refv3_fp_t dot = zero;
          REFV3_QSOFTRES_EXACT_MAX_DH_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
            const int idx = (k_token * REFV3_D_MODEL) + head_base + dh;
            dot += q_vec[head_base + dh] * in_k_payload.k_flat[idx];
          }
          const refv3_fp_t score = dot * inv_sqrt_dh;
          if (!has_valid || score > max_score) {
            max_score = score;
          }
          has_valid = true;
        }

        if (!has_valid) {
          REFV3_QSOFTRES_EXACT_NO_VALID_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
            head_ctx_buf[h][dh] = zero;
          }
          continue;
        }

        refv3_fp_t sumexp = zero;
        REFV3_QSOFTRES_EXACT_CLR_ACC_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
          softmax_acc_tile[dh] = zero;
        }

        REFV3_QSOFTRES_EXACT_ACC_TOKEN_LOOP: for (int k_token = 0; k_token < REFV3_TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          refv3_fp_t dot = zero;
          REFV3_QSOFTRES_EXACT_ACC_DH_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
            const int idx = (k_token * REFV3_D_MODEL) + head_base + dh;
            dot += q_vec[head_base + dh] * in_k_payload.k_flat[idx];
          }
          const refv3_fp_t score = dot * inv_sqrt_dh;
          const refv3_fp_t w(static_cast<float>(std::exp((score - max_score).to_float())));
          sumexp += w;
          REFV3_QSOFTRES_EXACT_VACC_DH_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
            const int idx = (k_token * REFV3_D_MODEL) + head_base + dh;
            softmax_acc_tile[dh] += w * in_v_payload.v_flat[idx];
          }
        }

        refv3_fp_t inv_sumexp = zero;
        if (sumexp > zero) {
          inv_sumexp = REFV3_softmax_rcp_synth(sumexp);
        }
        REFV3_QSOFTRES_EXACT_NORM_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
          head_ctx_buf[h][dh] = softmax_acc_tile[dh] * inv_sumexp;
        }
      } else
#endif
      {
        bool online_init = false;
        refv3_fp_t online_max = zero;
        refv3_fp_t online_sumexp = zero;

        REFV3_QSOFTRES_APPROX_CLR_ACC_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
          softmax_acc_tile[dh] = zero;
        }

        REFV3_QSOFTRES_APPROX_TOKEN_LOOP: for (int k_token = 0; k_token < REFV3_TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }

          refv3_fp_t dot = zero;
          REFV3_QSOFTRES_APPROX_DH_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
            const int idx = (k_token * REFV3_D_MODEL) + head_base + dh;
            dot += q_vec[head_base + dh] * in_k_payload.k_flat[idx];
          }
          const refv3_fp_t score = dot * inv_sqrt_dh;

          if (!online_init) {
            online_max = score;
            online_sumexp = refv3_fp_t(1.0f);
            REFV3_QSOFTRES_APPROX_INIT_ACC_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
              const int idx = (k_token * REFV3_D_MODEL) + head_base + dh;
              softmax_acc_tile[dh] = in_v_payload.v_flat[idx];
            }
            online_init = true;
            continue;
          }

          if (score > online_max) {
            const refv3_fp_t rescale = REFV3_softmax_exp_synth(online_max - score);
            online_sumexp = (online_sumexp * rescale) + refv3_fp_t(1.0f);
            REFV3_QSOFTRES_APPROX_RESCALE_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
              const int idx = (k_token * REFV3_D_MODEL) + head_base + dh;
              softmax_acc_tile[dh] =
                (softmax_acc_tile[dh] * rescale) + in_v_payload.v_flat[idx];
            }
            online_max = score;
            continue;
          }

          const refv3_fp_t w = REFV3_softmax_exp_synth(score - online_max);
          online_sumexp += w;
          REFV3_QSOFTRES_APPROX_ACC_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
            const int idx = (k_token * REFV3_D_MODEL) + head_base + dh;
            softmax_acc_tile[dh] += w * in_v_payload.v_flat[idx];
          }
        }

        if (!online_init) {
          REFV3_QSOFTRES_APPROX_NO_VALID_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
            head_ctx_buf[h][dh] = zero;
          }
          continue;
        }

        const refv3_fp_t inv_sumexp = REFV3_softmax_rcp_synth(online_sumexp);
        REFV3_QSOFTRES_APPROX_NORM_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
          head_ctx_buf[h][dh] = softmax_acc_tile[dh] * inv_sumexp;
        }
      }
    }

    REFV3_QSOFTRES_PACK_CTX_LOOP: for (int h = 0; h < REFV3_HEADS; ++h) {
      const int head_base = h * REFV3_D_HEAD;
      REFV3_QSOFTRES_PACK_CTX_DH_LOOP: for (int dh = 0; dh < REFV3_D_HEAD; ++dh) {
        q_vec[head_base + dh] = head_ctx_buf[h][dh];
      }
    }

    quant_linear_token_32_to32_native(
      q_vec,
      o_params,
      s_x_o,
      inv_attn_o,
      out_acc_tile);

    RefV3AttentionTokenVectorPayload out_token_payload;
    out_token_payload.header = header_ref;
    out_token_payload.token_row = ac_int<16, false>(q_token);
    REFV3_QSOFTRES_TOKEN_OUT_DIM_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      out_token_payload.token_vec[d] = out_acc_tile[d] + ln_token_buf[d];
    }
    out_token_ch.write(out_token_payload);
  }

  return true;
}

bool RefV3AttenQSoftResBlock::run(int lid,
                                  const RefRunConfig& run_cfg,
                                  ac_channel<RefV3AttentionTokenVectorPayload>& query_token_ch,
                                  ac_channel<RefV3AttentionKPayload>& in_k_payload_ch,
                                  ac_channel<RefV3AttentionVPayload>& in_v_payload_ch,
                                  ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const {
  RefV3AttentionInputPayload xwork_payload;
  bool header_init = false;
  bool query_token_seen[REFV3_TOKENS_T];
  REFV3_QSOFTRES_COMPAT_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    query_token_seen[token] = false;
  }

  REFV3_QSOFTRES_COMPAT_READ_QUERY_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3AttentionTokenVectorPayload query_token_payload = query_token_ch.read();
    if (!REFV3_payload_header_matches_shape(query_token_payload.header)) {
      return false;
    }
    if (query_token_payload.header.layer_id.to_int() != lid) {
      return false;
    }
    if (!header_init) {
      xwork_payload.header = query_token_payload.header;
      header_init = true;
    } else {
      if (query_token_payload.header.layer_id != xwork_payload.header.layer_id ||
          query_token_payload.header.token_rows != xwork_payload.header.token_rows ||
          query_token_payload.header.dim_cols != xwork_payload.header.dim_cols) {
        return false;
      }
    }

    const int q_token = query_token_payload.token_row.to_int();
    if (q_token < 0 || q_token >= REFV3_TOKENS_T) {
      return false;
    }
    if (query_token_seen[q_token]) {
      return false;
    }
    query_token_seen[q_token] = true;

    REFV3_QSOFTRES_COMPAT_COPY_DIM_LOOP: for (int d = 0; d < REFV3_D_MODEL; ++d) {
      const int x_idx = REFV3_flatten_row_major_index(q_token, d);
      xwork_payload.x_flat[x_idx] = query_token_payload.token_vec[d];
    }
  }

  if (!header_init) {
    return false;
  }

  ac_channel<RefV3AttentionInputPayload> in_xwork_ch;
  in_xwork_ch.write(xwork_payload);
  return run(lid, run_cfg, in_xwork_ch, in_k_payload_ch, in_v_payload_ch, out_token_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
