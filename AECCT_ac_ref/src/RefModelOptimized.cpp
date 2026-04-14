#include "../include/RefModelOptimized.h"

#include <cassert>
#include <cmath>

#include "../include/SoftmaxApprox.h"
#include "weights.h"

namespace aecct_ref {
namespace {

static inline ref_fp32_t make_ref_fp32(float x) {
  return ref_fp32_t(x);
}

template<ac_ieee_float_format FloatFormat>
static inline ref_fp32_t make_ref_fp32_from_island(ac_ieee_float<FloatFormat> x) {
  return ref_fp32_t(x.to_float());
}

} // namespace

RefModelOptimized::RefModelOptimized()
  : last_staged_sample_index_(-1),
    phase_a_valid_(false),
    layer0_attn_writeback_valid_(false) {
  run_cfg_ = make_fp32_baseline_run_config();
  legacy_ref_.set_run_config(run_cfg_);
  clear_formal_storage();
}

void RefModelOptimized::set_run_config(const RefRunConfig& cfg) {
  run_cfg_ = cfg;
  legacy_ref_.set_run_config(cfg);
}

RefRunConfig RefModelOptimized::get_run_config() const {
  return run_cfg_;
}

void RefModelOptimized::set_numeric_config(const RefOptimizedNumericConfig& cfg) {
  numeric_cfg_ = cfg;
}

RefOptimizedNumericConfig RefModelOptimized::get_numeric_config() const {
  return numeric_cfg_;
}

void RefModelOptimized::infer_step0(const RefModelIO& io) {
  if (io.input_y_fp32 != nullptr && io.B > 0 && io.N >= VAR_N) {
    for (int b = 0; b < io.B; ++b) {
      if (stage_step0_phase_a(io, b)) {
        (void)run_step0_layer0_attention_writeback();
      }
    }
  }

  // Layer0 attention writeback (Step 4) is ported in the optimized path.
  // Remaining downstream phases still complete on the legacy path.
  legacy_ref_.infer_step0(io);
}

bool RefModelOptimized::stage_step0_phase_a(const RefModelIO& io, int batch_index) {
  if (io.input_y_fp32 == nullptr) {
    return false;
  }
  if (io.N < VAR_N || io.B <= 0) {
    return false;
  }
  if (batch_index < 0 || batch_index >= io.B) {
    return false;
  }

  if (numeric_cfg_.float_mode == REF_OPT_FLOAT16) {
    return stage_step0_phase_a_with_float<binary16>(io, batch_index);
  }
  return stage_step0_phase_a_with_float<binary32>(io, batch_index);
}

int RefModelOptimized::last_staged_sample_index() const {
  return last_staged_sample_index_;
}

bool RefModelOptimized::phase_a_valid() const {
  return phase_a_valid_;
}

bool RefModelOptimized::layer0_attn_writeback_valid() const {
  return layer0_attn_writeback_valid_;
}

const ref_fp32_t& RefModelOptimized::x_work(int token, int dim) const {
  assert(token >= 0 && token < TOKENS_T);
  assert(dim >= 0 && dim < D_MODEL);
  return x_work_[token][dim];
}

const ref_fp32_t& RefModelOptimized::scr_k(int token, int dim) const {
  assert(token >= 0 && token < TOKENS_T);
  assert(dim >= 0 && dim < D_MODEL);
  return scr_k_[token][dim];
}

const ref_fp32_t& RefModelOptimized::scr_v(int token, int dim) const {
  assert(token >= 0 && token < TOKENS_T);
  assert(dim >= 0 && dim < D_MODEL);
  return scr_v_[token][dim];
}

const ref_fp32_t& RefModelOptimized::final_scalar_buf(int token) const {
  assert(token >= 0 && token < TOKENS_T);
  return final_scalar_buf_[token];
}

bool RefModelOptimized::run_step0_layer0_attention_writeback() {
  if (!phase_a_valid_) {
    return false;
  }

  if (numeric_cfg_.float_mode == REF_OPT_FLOAT16) {
    materialize_layer0_attention_writeback_from_x_work<binary16>();
  } else {
    materialize_layer0_attention_writeback_from_x_work<binary32>();
  }
  layer0_attn_writeback_valid_ = true;
  return true;
}

void RefModelOptimized::clear_formal_storage() {
  for (int t = 0; t < TOKENS_T; ++t) {
    final_scalar_buf_[t] = make_ref_fp32(0.0f);
    for (int d = 0; d < D_MODEL; ++d) {
      x_work_[t][d] = make_ref_fp32(0.0f);
      scr_k_[t][d] = make_ref_fp32(0.0f);
      scr_v_[t][d] = make_ref_fp32(0.0f);
      q_vec_[d] = make_ref_fp32(0.0f);
      out_acc_tile_[d] = make_ref_fp32(0.0f);
      ln_token_buf_[d] = make_ref_fp32(0.0f);
    }
  }
  for (int h = 0; h < HEADS; ++h) {
    for (int dh = 0; dh < D_HEAD; ++dh) {
      head_ctx_buf_[h][dh] = make_ref_fp32(0.0f);
      softmax_acc_tile_[dh] = make_ref_fp32(0.0f);
    }
  }
  for (int i = 0; i < FF_DIM; ++i) {
    ffn1_tile_buf_[i] = make_ref_fp32(0.0f);
  }
  last_staged_sample_index_ = -1;
  phase_a_valid_ = false;
  layer0_attn_writeback_valid_ = false;
}

template<ac_ieee_float_format FloatFormat>
bool RefModelOptimized::stage_step0_phase_a_with_float(
  const RefModelIO& io,
  int batch_index) {
  clear_formal_storage();
  build_preproc_x_work_from_input<FloatFormat>(&io.input_y_fp32[batch_index * io.N]);
  materialize_layer0_kv_from_x_work<FloatFormat>();
  last_staged_sample_index_ = batch_index;
  phase_a_valid_ = true;
  layer0_attn_writeback_valid_ = false;
  return true;
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::build_preproc_x_work_from_input(const double* input_y_fp32) {
  ac_ieee_float<FloatFormat> node_feature[TOKENS_T];
  ac_int<1, false> y_hard[VAR_N];

  for (int i = 0; i < VAR_N; ++i) {
    const ac_ieee_float<FloatFormat> y(static_cast<float>(input_y_fp32[i]));
    node_feature[i] = float_abs_local<FloatFormat>(y);
    y_hard[i] = (y < ac_ieee_float<FloatFormat>(0.0f)) ? ac_int<1, false>(1) : ac_int<1, false>(0);
  }
  for (int c = 0; c < CHECK_N; ++c) {
    ac_int<1, false> parity = 0;
    for (int v = 0; v < VAR_N; ++v) {
      if (h_H[c * VAR_N + v].to_int() != 0) {
        parity = ac_int<1, false>(parity ^ y_hard[v]);
      }
    }
    node_feature[VAR_N + c] =
      (parity == 0) ? ac_ieee_float<FloatFormat>(1.0f) : ac_ieee_float<FloatFormat>(-1.0f);
  }

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int k = 0; k < 24; ++k) {
      const ac_ieee_float<FloatFormat> embed =
        node_feature[t] * ac_ieee_float<FloatFormat>(static_cast<float>(w_src_embed[t * 24 + k]));
      x_work_[t][k] = make_ref_fp32_from_island<FloatFormat>(embed);
    }
    for (int k = 0; k < 8; ++k) {
      const ac_ieee_float<FloatFormat> lpe =
        ac_ieee_float<FloatFormat>(static_cast<float>(w_lpe_token[t * 8 + k]));
      x_work_[t][24 + k] = make_ref_fp32_from_island<FloatFormat>(lpe);
    }
  }
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_layer0_kv_from_x_work() {
  // Free-point rule:
  // X_WORK remains the source of truth until both SCR_K and SCR_V have been
  // fully materialized. No X_WORK overwrite is allowed in this phase.
  const float s_x_in = static_cast<float>(l0_in_s_x);
  const float s_w_k = static_cast<float>(w_decoder_layers_0_self_attn_linears_1_s_w[0]);
  const float s_w_v = static_cast<float>(w_decoder_layers_0_self_attn_linears_2_s_w[0]);

  for (int t = 0; t < TOKENS_T; ++t) {
    quant_linear_token_32_to32_native<FloatFormat>(
      x_work_[t],
      w_decoder_layers_0_self_attn_linears_1_weight,
      w_decoder_layers_0_self_attn_linears_1_bias,
      s_x_in,
      s_w_k,
      scr_k_[t]);
    quant_linear_token_32_to32_native<FloatFormat>(
      x_work_[t],
      w_decoder_layers_0_self_attn_linears_2_weight,
      w_decoder_layers_0_self_attn_linears_2_bias,
      s_x_in,
      s_w_v,
      scr_v_[t]);
  }
}

bool RefModelOptimized::is_layer0_attn_masked_token_pair(
  int head_idx,
  int q_token,
  int k_token) {
  assert(head_idx >= 0 && head_idx < HEADS);
  assert(q_token >= 0 && q_token < TOKENS_T);
  assert(k_token >= 0 && k_token < TOKENS_T);
  const bool src_masked = (w_src_mask[q_token * TOKENS_T + k_token].to_int() != 0);
  const bool q_is_var = (q_token < VAR_N);
  const bool k_is_var = (k_token < VAR_N);

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

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_layer0_attention_writeback_from_x_work() {
  // Free-point rule:
  // SCR_K/SCR_V remain the K/V source of truth for the whole layer-0 attention
  // traversal. X_WORK writeback is committed only after Wo + residual per token.
  const float s_x_in = static_cast<float>(l0_in_s_x);
  const float s_x_o = static_cast<float>(l0_o_s_x);
  const float s_w_q = static_cast<float>(w_decoder_layers_0_self_attn_linears_0_s_w[0]);
  const float s_w_o = static_cast<float>(w_decoder_layers_0_self_attn_linears_3_s_w[0]);
  const ref_fp32_t inv_sqrt_dh = make_ref_fp32(0.5f); // 1 / sqrt(4)
  const bool use_softmax_exact = (run_cfg_.legacy.algo_variant == RefAlgoVariant::RESERVED_SOFTMAX_ALT);

  for (int q_token = 0; q_token < TOKENS_T; ++q_token) {
    for (int d = 0; d < D_MODEL; ++d) {
      ln_token_buf_[d] = x_work_[q_token][d];
      out_acc_tile_[d] = make_ref_fp32(0.0f);
    }

    quant_linear_token_32_to32_native<FloatFormat>(
      x_work_[q_token],
      w_decoder_layers_0_self_attn_linears_0_weight,
      w_decoder_layers_0_self_attn_linears_0_bias,
      s_x_in,
      s_w_q,
      q_vec_);

    for (int h = 0; h < HEADS; ++h) {
      const int base = h * D_HEAD;
      if (use_softmax_exact) {
        bool has_valid = false;
        ref_fp32_t max_score = make_ref_fp32(0.0f);
        for (int k_token = 0; k_token < TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          ref_fp32_t dot = make_ref_fp32(0.0f);
          for (int dh = 0; dh < D_HEAD; ++dh) {
            dot += q_vec_[base + dh] * scr_k_[k_token][base + dh];
          }
          const ref_fp32_t score = dot * inv_sqrt_dh;
          if (!has_valid || score > max_score) {
            max_score = score;
          }
          has_valid = true;
        }

        if (!has_valid) {
          for (int dh = 0; dh < D_HEAD; ++dh) {
            head_ctx_buf_[h][dh] = make_ref_fp32(0.0f);
          }
          continue;
        }

        ref_fp32_t sumexp = make_ref_fp32(0.0f);
        for (int dh = 0; dh < D_HEAD; ++dh) {
          softmax_acc_tile_[dh] = make_ref_fp32(0.0f);
        }
        for (int k_token = 0; k_token < TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          ref_fp32_t dot = make_ref_fp32(0.0f);
          for (int dh = 0; dh < D_HEAD; ++dh) {
            dot += q_vec_[base + dh] * scr_k_[k_token][base + dh];
          }
          const ref_fp32_t score = dot * inv_sqrt_dh;
          const ref_fp32_t w = make_ref_fp32(std::exp((score - max_score).to_float()));
          sumexp += w;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            softmax_acc_tile_[dh] += w * scr_v_[k_token][base + dh];
          }
        }

        ref_fp32_t inv_sumexp = make_ref_fp32(0.0f);
        if (sumexp > make_ref_fp32(0.0f)) {
          inv_sumexp = make_ref_fp32(1.0f) / sumexp;
        }
        for (int dh = 0; dh < D_HEAD; ++dh) {
          head_ctx_buf_[h][dh] = softmax_acc_tile_[dh] * inv_sumexp;
        }
      } else {
        bool online_init = false;
        ref_fp32_t online_max = make_ref_fp32(0.0f);
        ref_fp32_t online_sumexp = make_ref_fp32(0.0f);
        for (int dh = 0; dh < D_HEAD; ++dh) {
          softmax_acc_tile_[dh] = make_ref_fp32(0.0f);
        }

        for (int k_token = 0; k_token < TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          ref_fp32_t dot = make_ref_fp32(0.0f);
          for (int dh = 0; dh < D_HEAD; ++dh) {
            dot += q_vec_[base + dh] * scr_k_[k_token][base + dh];
          }
          const ref_fp32_t score = dot * inv_sqrt_dh;

          if (!online_init) {
            online_max = score;
            online_sumexp = make_ref_fp32(1.0f);
            for (int dh = 0; dh < D_HEAD; ++dh) {
              softmax_acc_tile_[dh] = scr_v_[k_token][base + dh];
            }
            online_init = true;
            continue;
          }

          if (score > online_max) {
            const ref_fp32_t rescale = ref_softmax_exp_dispatch(
              online_max - score,
              run_cfg_.legacy.softmax_exp_mode);
            online_sumexp = (online_sumexp * rescale) + make_ref_fp32(1.0f);
            for (int dh = 0; dh < D_HEAD; ++dh) {
              softmax_acc_tile_[dh] = (softmax_acc_tile_[dh] * rescale) + scr_v_[k_token][base + dh];
            }
            online_max = score;
            continue;
          }

          const ref_fp32_t w = ref_softmax_exp_dispatch(
            score - online_max,
            run_cfg_.legacy.softmax_exp_mode);
          online_sumexp += w;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            softmax_acc_tile_[dh] += w * scr_v_[k_token][base + dh];
          }
        }

        if (!online_init) {
          for (int dh = 0; dh < D_HEAD; ++dh) {
            head_ctx_buf_[h][dh] = make_ref_fp32(0.0f);
          }
          continue;
        }

        const ref_fp32_t inv_sumexp = ref_softmax_rcp_lut(online_sumexp);
        for (int dh = 0; dh < D_HEAD; ++dh) {
          head_ctx_buf_[h][dh] = softmax_acc_tile_[dh] * inv_sumexp;
        }
      }
    }

    for (int h = 0; h < HEADS; ++h) {
      const int base = h * D_HEAD;
      for (int dh = 0; dh < D_HEAD; ++dh) {
        q_vec_[base + dh] = head_ctx_buf_[h][dh];
      }
    }

    quant_linear_token_32_to32_native<FloatFormat>(
      q_vec_,
      w_decoder_layers_0_self_attn_linears_3_weight,
      w_decoder_layers_0_self_attn_linears_3_bias,
      s_x_o,
      s_w_o,
      out_acc_tile_);

    for (int d = 0; d < D_MODEL; ++d) {
      x_work_[q_token][d] = out_acc_tile_[d] + ln_token_buf_[d];
    }
  }
}

ref_fp32_t RefModelOptimized::fp32_abs_local(ref_fp32_t x) {
  return (x < make_ref_fp32(0.0f)) ? (make_ref_fp32(0.0f) - x) : x;
}

template<ac_ieee_float_format FloatFormat>
ac_ieee_float<FloatFormat> RefModelOptimized::float_abs_local(
  ac_ieee_float<FloatFormat> x) {
  return (x < ac_ieee_float<FloatFormat>(0.0f))
    ? (ac_ieee_float<FloatFormat>(0.0f) - x)
    : x;
}

ac_int<8, true> RefModelOptimized::quantize_int8_to_i8_local(ref_fp32_t x, float s_x) {
  const double scaled = static_cast<double>(x.to_float()) * static_cast<double>(s_x);
  if (scaled >= 127.0) {
    return ac_int<8, true>(127);
  }
  if (scaled <= -127.0) {
    return ac_int<8, true>(-127);
  }
  return ac_int<8, true>(static_cast<signed char>(std::lround(scaled)));
}

ac_int<2, true> RefModelOptimized::decode_ternary_weight_sign_i2_local(double w) {
  if (w == 1.0 || w == 1.0f) return ac_int<2, true>(1);
  if (w == -1.0 || w == -1.0f) return ac_int<2, true>(-1);
  if (w == 0.0 || w == -0.0 || w == 0.0f || w == -0.0f) return ac_int<2, true>(0);
  if (w >= 0.5) return ac_int<2, true>(1);
  if (w <= -0.5) return ac_int<2, true>(-1);
  return ac_int<2, true>(0);
}

ac_int<16, true> RefModelOptimized::accumulate_ternary_mac_i16_local(
  ac_int<16, true> acc_i16,
  ac_int<8, true> qx_i8,
  ac_int<2, true> ternary_sign_i2) {
  const ac_int<16, true> mac_i16 = qx_i8 * ternary_sign_i2;
  const ac_int<16, true> next_acc_i16 = acc_i16 + mac_i16;
  return next_acc_i16;
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::quant_linear_token_32_to32_native(
  const ref_fp32_t x[D_MODEL],
  const double w[D_MODEL * D_MODEL],
  const double b[D_MODEL],
  float s_x,
  float s_w,
  ref_fp32_t y[D_MODEL]) {
  const ac_ieee_float<FloatFormat> inv =
    ac_ieee_float<FloatFormat>(1.0f) /
    (ac_ieee_float<FloatFormat>(s_x) * ac_ieee_float<FloatFormat>(s_w));
  ac_int<8, true> qx_i8[D_MODEL];
  for (int i = 0; i < D_MODEL; ++i) {
    qx_i8[i] = quantize_int8_to_i8_local(x[i], s_x);
  }
  for (int o = 0; o < D_MODEL; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * D_MODEL;
    for (int i = 0; i < D_MODEL; ++i) {
      acc_i16 = accumulate_ternary_mac_i16_local(
        acc_i16,
        qx_i8[i],
        decode_ternary_weight_sign_i2_local(w[base + i]));
    }
    const ac_ieee_float<FloatFormat> bias_island(static_cast<float>(b[o]));
    const ac_ieee_float<FloatFormat> acc_island(acc_i16.to_int());
    const ac_ieee_float<FloatFormat> y_island = bias_island + (acc_island * inv);
    y[o] = make_ref_fp32_from_island<FloatFormat>(y_island);
  }
}

template bool RefModelOptimized::stage_step0_phase_a_with_float<binary16>(
  const RefModelIO& io,
  int batch_index);
template bool RefModelOptimized::stage_step0_phase_a_with_float<binary32>(
  const RefModelIO& io,
  int batch_index);

template void RefModelOptimized::build_preproc_x_work_from_input<binary16>(
  const double* input_y_fp32);
template void RefModelOptimized::build_preproc_x_work_from_input<binary32>(
  const double* input_y_fp32);

template void RefModelOptimized::materialize_layer0_kv_from_x_work<binary16>();
template void RefModelOptimized::materialize_layer0_kv_from_x_work<binary32>();
template void RefModelOptimized::materialize_layer0_attention_writeback_from_x_work<binary16>();
template void RefModelOptimized::materialize_layer0_attention_writeback_from_x_work<binary32>();

template ac_ieee_float<binary16> RefModelOptimized::float_abs_local<binary16>(
  ac_ieee_float<binary16> x);
template ac_ieee_float<binary32> RefModelOptimized::float_abs_local<binary32>(
  ac_ieee_float<binary32> x);

template void RefModelOptimized::quant_linear_token_32_to32_native<binary16>(
  const ref_fp32_t x[D_MODEL],
  const double w[D_MODEL * D_MODEL],
  const double b[D_MODEL],
  float s_x,
  float s_w,
  ref_fp32_t y[D_MODEL]);
template void RefModelOptimized::quant_linear_token_32_to32_native<binary32>(
  const ref_fp32_t x[D_MODEL],
  const double w[D_MODEL * D_MODEL],
  const double b[D_MODEL],
  float s_x,
  float s_w,
  ref_fp32_t y[D_MODEL]);

} // namespace aecct_ref
