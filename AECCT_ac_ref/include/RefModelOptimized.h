#pragma once

#include "RefModel.h"

namespace aecct_ref {

class RefModelOptimized {
public:
  RefModelOptimized();

  void set_run_config(const RefRunConfig& cfg);
  RefRunConfig get_run_config() const;

  // Partial optimized pipeline:
  // Step 2/3 currently materialize preproc into X_WORK and layer-0 K/V into
  // SCR_K / SCR_V. Downstream phases are still completed by the legacy
  // RefModel path until later steps are ported.
  void infer_step0(const RefModelIO& io);

  // Stage the optimized major storage for one sample without running the
  // remaining attention/FFN/finalhead phases.
  bool stage_step0_phase_a(const RefModelIO& io, int batch_index = 0);

  int last_staged_sample_index() const;
  bool phase_a_valid() const;

  const ref_fp32_t& x_work(int token, int dim) const;
  const ref_fp32_t& scr_k(int token, int dim) const;
  const ref_fp32_t& scr_v(int token, int dim) const;
  const ref_fp32_t& final_scalar_buf(int token) const;

private:
  static const int TOKENS_T = 75;
  static const int VAR_N = 63;
  static const int CHECK_N = 12;
  static const int D_MODEL = 32;
  static const int HEADS = 8;
  static const int D_HEAD = 4;
  static const int FF_DIM = 128;

  void clear_formal_storage();
  void build_preproc_x_work_from_input(const double* input_y_fp32);
  void materialize_layer0_kv_from_x_work();

  static ref_fp32_t fp32_abs_local(ref_fp32_t x);
  static int16_t quantize_int8_to_i16_local(ref_fp32_t x, float s_x);
  static int16_t decode_ternary_weight_sign_i16_local(double w);
  static int16_t accumulate_ternary_mac_i16_local(
    int16_t acc_i16,
    int16_t qx_i16,
    int16_t ternary_sign_i16);
  static void quant_linear_token_32_to32_native(
    const ref_fp32_t x[D_MODEL],
    const double w[D_MODEL * D_MODEL],
    const double b[D_MODEL],
    float s_x,
    float s_w,
    ref_fp32_t y[D_MODEL]);

private:
  RefRunConfig run_cfg_;
  RefModel legacy_ref_;

  // Formal major storage.
  ref_fp32_t x_work_[TOKENS_T][D_MODEL];
  ref_fp32_t scr_k_[TOKENS_T][D_MODEL];
  ref_fp32_t scr_v_[TOKENS_T][D_MODEL];
  ref_fp32_t final_scalar_buf_[TOKENS_T];

  // Local / tile storage skeleton for later steps.
  ref_fp32_t q_vec_[D_MODEL];
  ref_fp32_t head_ctx_buf_[HEADS][D_HEAD];
  ref_fp32_t out_acc_tile_[D_MODEL];
  ref_fp32_t softmax_acc_tile_[D_HEAD];
  ref_fp32_t ffn1_tile_buf_[FF_DIM];
  ref_fp32_t ln_token_buf_[D_MODEL];

  int last_staged_sample_index_;
  bool phase_a_valid_;
};

} // namespace aecct_ref
