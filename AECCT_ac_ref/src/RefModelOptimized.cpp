#include "../include/RefModelOptimized.h"

#include <cassert>
#include <cmath>
#include <cstdint>

#include "weights.h"

namespace aecct_ref {
namespace {

static inline ref_fp32_t make_ref_fp32(float x) {
  return ref_fp32_t(x);
}

} // namespace

RefModelOptimized::RefModelOptimized()
  : last_staged_sample_index_(-1),
    phase_a_valid_(false) {
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

void RefModelOptimized::infer_step0(const RefModelIO& io) {
  if (io.input_y_fp32 != nullptr && io.B > 0 && io.N >= VAR_N) {
    for (int b = 0; b < io.B; ++b) {
      (void)stage_step0_phase_a(io, b);
    }
  }

  // Step 4+ are not ported yet. Keep functional completion on the legacy path
  // while the optimized storage pipeline is being built up incrementally.
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

  clear_formal_storage();
  build_preproc_x_work_from_input(&io.input_y_fp32[batch_index * io.N]);
  materialize_layer0_kv_from_x_work();
  last_staged_sample_index_ = batch_index;
  phase_a_valid_ = true;
  return true;
}

int RefModelOptimized::last_staged_sample_index() const {
  return last_staged_sample_index_;
}

bool RefModelOptimized::phase_a_valid() const {
  return phase_a_valid_;
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
}

void RefModelOptimized::build_preproc_x_work_from_input(const double* input_y_fp32) {
  ref_fp32_t node_feature[TOKENS_T];
  int y_hard[VAR_N];

  for (int i = 0; i < VAR_N; ++i) {
    const ref_fp32_t y = make_ref_fp32(static_cast<float>(input_y_fp32[i]));
    node_feature[i] = fp32_abs_local(y);
    y_hard[i] = (y < make_ref_fp32(0.0f)) ? 1 : 0;
  }
  for (int c = 0; c < CHECK_N; ++c) {
    int parity = 0;
    for (int v = 0; v < VAR_N; ++v) {
      if (h_H[c * VAR_N + v].to_int() != 0) {
        parity ^= y_hard[v];
      }
    }
    node_feature[VAR_N + c] = (parity == 0) ? make_ref_fp32(1.0f) : make_ref_fp32(-1.0f);
  }

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int k = 0; k < 24; ++k) {
      x_work_[t][k] =
        node_feature[t] * make_ref_fp32(static_cast<float>(w_src_embed[t * 24 + k]));
    }
    for (int k = 0; k < 8; ++k) {
      x_work_[t][24 + k] = make_ref_fp32(static_cast<float>(w_lpe_token[t * 8 + k]));
    }
  }
}

void RefModelOptimized::materialize_layer0_kv_from_x_work() {
  // Free-point rule:
  // X_WORK remains the source of truth until both SCR_K and SCR_V have been
  // fully materialized. No X_WORK overwrite is allowed in this phase.
  const float s_x_in = static_cast<float>(l0_in_s_x);
  const float s_w_k = static_cast<float>(w_decoder_layers_0_self_attn_linears_1_s_w[0]);
  const float s_w_v = static_cast<float>(w_decoder_layers_0_self_attn_linears_2_s_w[0]);

  for (int t = 0; t < TOKENS_T; ++t) {
    quant_linear_token_32_to32_native(
      x_work_[t],
      w_decoder_layers_0_self_attn_linears_1_weight,
      w_decoder_layers_0_self_attn_linears_1_bias,
      s_x_in,
      s_w_k,
      scr_k_[t]);
    quant_linear_token_32_to32_native(
      x_work_[t],
      w_decoder_layers_0_self_attn_linears_2_weight,
      w_decoder_layers_0_self_attn_linears_2_bias,
      s_x_in,
      s_w_v,
      scr_v_[t]);
  }
}

ref_fp32_t RefModelOptimized::fp32_abs_local(ref_fp32_t x) {
  return (x < make_ref_fp32(0.0f)) ? (make_ref_fp32(0.0f) - x) : x;
}

int16_t RefModelOptimized::quantize_int8_to_i16_local(ref_fp32_t x, float s_x) {
  int32_t q = static_cast<int32_t>(std::lround(x.to_float() * s_x));
  if (q > 127) q = 127;
  if (q < -127) q = -127;
  return static_cast<int16_t>(q);
}

int16_t RefModelOptimized::decode_ternary_weight_sign_i16_local(double w) {
  if (w == 1.0 || w == 1.0f) return static_cast<int16_t>(1);
  if (w == -1.0 || w == -1.0f) return static_cast<int16_t>(-1);
  if (w == 0.0 || w == -0.0 || w == 0.0f || w == -0.0f) return static_cast<int16_t>(0);
  if (w >= 0.5) return static_cast<int16_t>(1);
  if (w <= -0.5) return static_cast<int16_t>(-1);
  return static_cast<int16_t>(0);
}

int16_t RefModelOptimized::accumulate_ternary_mac_i16_local(
  int16_t acc_i16,
  int16_t qx_i16,
  int16_t ternary_sign_i16) {
  int32_t sum =
    static_cast<int32_t>(acc_i16) +
    (static_cast<int32_t>(qx_i16) * static_cast<int32_t>(ternary_sign_i16));
  if (sum > 32767) sum = 32767;
  if (sum < -32768) sum = -32768;
  return static_cast<int16_t>(sum);
}

void RefModelOptimized::quant_linear_token_32_to32_native(
  const ref_fp32_t x[D_MODEL],
  const double w[D_MODEL * D_MODEL],
  const double b[D_MODEL],
  float s_x,
  float s_w,
  ref_fp32_t y[D_MODEL]) {
  const ref_fp32_t inv =
    make_ref_fp32(1.0f) / (make_ref_fp32(s_x) * make_ref_fp32(s_w));
  int16_t qx_i16[D_MODEL];
  for (int i = 0; i < D_MODEL; ++i) {
    qx_i16[i] = quantize_int8_to_i16_local(x[i], s_x);
  }
  for (int o = 0; o < D_MODEL; ++o) {
    int16_t acc_i16 = 0;
    const int base = o * D_MODEL;
    for (int i = 0; i < D_MODEL; ++i) {
      acc_i16 = accumulate_ternary_mac_i16_local(
        acc_i16,
        qx_i16[i],
        decode_ternary_weight_sign_i16_local(w[base + i]));
    }
    y[o] = make_ref_fp32(static_cast<float>(b[o])) +
           (make_ref_fp32(static_cast<float>(acc_i16)) * inv);
  }
}

} // namespace aecct_ref
