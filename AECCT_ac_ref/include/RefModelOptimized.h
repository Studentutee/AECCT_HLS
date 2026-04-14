#pragma once

#include "ac_int.h"
#include "ac_std_float.h"
#include "RefModel.h"

namespace aecct_ref {

enum RefOptimizedFloatMode {
  REF_OPT_FLOAT16 = 0,
  REF_OPT_FLOAT32 = 1
};

struct RefOptimizedNumericConfig {
  RefOptimizedNumericConfig() : float_mode(REF_OPT_FLOAT32) {}

  RefOptimizedFloatMode float_mode;
};

class RefModelOptimized {
public:
  RefModelOptimized();

  void set_run_config(const RefRunConfig& cfg);
  RefRunConfig get_run_config() const;
  void set_numeric_config(const RefOptimizedNumericConfig& cfg);
  RefOptimizedNumericConfig get_numeric_config() const;

  // Partial optimized pipeline:
  // Step 2/3 materialize preproc into X_WORK and layer-0 K/V into SCR_K /
  // SCR_V. Step 4/5A/5B/6 port layer-0 attention/LN/FFN/mid-norm writeback.
  // Step 7~10 extend optimized coverage to layer-1 attention, layer-1 post-
  // attention LN, layer-1 FFN, and end-norm writeback into/in-place on
  // X_WORK. FinalHead/output completion still stays on the legacy path.
  void infer_step0(const RefModelIO& io);

  // Stage the optimized major storage for one sample without running the
  // remaining attention/FFN/finalhead phases.
  bool stage_step0_phase_a(const RefModelIO& io, int batch_index = 0);
  bool run_step0_layer0_attention_writeback();
  bool run_step0_layer0_ln_writeback();
  bool run_step0_layer0_ffn_writeback();
  bool run_step0_mid_norm_writeback();
  bool run_step0_layer1_attn_input_handoff();
  bool run_step0_layer1_attention_writeback();
  bool run_step0_layer1_ln_writeback();
  bool run_step0_layer1_ffn_writeback();
  bool run_step0_end_norm_writeback();

  int last_staged_sample_index() const;
  bool phase_a_valid() const;
  bool layer0_attn_writeback_valid() const;
  bool layer0_ln_writeback_valid() const;
  bool layer0_ffn_writeback_valid() const;
  bool mid_norm_writeback_valid() const;
  bool layer1_attn_input_handoff_valid() const;
  bool layer1_attn_writeback_valid() const;
  bool layer1_ln_writeback_valid() const;
  bool layer1_ffn_writeback_valid() const;
  bool end_norm_writeback_valid() const;

  ac_ieee_float<binary32> x_work(int token, int dim) const;
  ac_ieee_float<binary32> scr_k(int token, int dim) const;
  ac_ieee_float<binary32> scr_v(int token, int dim) const;
  ac_ieee_float<binary32> final_scalar_buf(int token) const;

private:
  static const int TOKENS_T = 75;
  static const int VAR_N = 63;
  static const int CHECK_N = 12;
  static const int D_MODEL = 32;
  static const int HEADS = 8;
  static const int D_HEAD = 4;
  static const int FF_DIM = 128;

  template<ac_ieee_float_format FloatFormat>
  struct RefOptimizedStorageBank {
    typedef ac_ieee_float<FloatFormat> float_t;

    // Formal major storage for optimized step-0 path.
    float_t x_work[TOKENS_T][D_MODEL];
    float_t scr_k[TOKENS_T][D_MODEL];
    float_t scr_v[TOKENS_T][D_MODEL];
    float_t final_scalar_buf[TOKENS_T];

    // Local-only tile/buffer state.
    float_t q_vec[D_MODEL];
    float_t head_ctx_buf[HEADS][D_HEAD];
    float_t out_acc_tile[D_MODEL];
    float_t softmax_acc_tile[D_HEAD];
    float_t ffn1_token_buf[FF_DIM];
    float_t ln_token_buf[D_MODEL];
  };

  void clear_formal_storage();
  RefOptimizedFloatMode resolve_selected_float_mode() const;
  template<ac_ieee_float_format FloatFormat>
  void clear_storage_bank(RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  static ac_ieee_float<binary32> export_debug_scalar(
    typename RefOptimizedStorageBank<FloatFormat>::float_t x);

  template<ac_ieee_float_format FloatFormat>
  bool stage_step0_phase_a_with_float(
    const RefModelIO& io,
    int batch_index,
    RefOptimizedStorageBank<FloatFormat>& bank);
  void refresh_layer1_attn_input_dut_aligned_seed_from_legacy(
    const double* input_y_fp32,
    int input_len);
  template<ac_ieee_float_format FloatFormat>
  void build_preproc_x_work_from_input(
    const double* input_y_fp32,
    RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  void materialize_layer0_kv_from_x_work(
    RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  void materialize_layer0_attention_writeback_from_x_work(
    RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  void materialize_layer0_ln_writeback_from_x_work(
    RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  void materialize_layer0_ffn_writeback_from_x_work(
    RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  void materialize_layer0_mid_norm_writeback_from_x_work(
    RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  void materialize_layer1_attention_writeback_from_x_work(
    RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  void materialize_layer1_ln_writeback_from_x_work(
    RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  void materialize_layer1_ffn_writeback_from_x_work(
    RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  void materialize_end_norm_writeback_from_x_work(
    RefOptimizedStorageBank<FloatFormat>& bank);
  template<ac_ieee_float_format FloatFormat>
  void layernorm_token_32_local(
    const typename RefOptimizedStorageBank<FloatFormat>::float_t x_token[D_MODEL],
    const double w[D_MODEL],
    const double b[D_MODEL],
    typename RefOptimizedStorageBank<FloatFormat>::float_t y_token[D_MODEL]) const;
  static bool is_layer0_attn_masked_token_pair(int head_idx, int q_token, int k_token);

  template<ac_ieee_float_format FloatFormat>
  static ac_ieee_float<FloatFormat> float_abs_local(
    ac_ieee_float<FloatFormat> x);
  template<typename FloatT>
  static ac_int<8, true> quantize_int8_to_i8_local(FloatT x, float s_x);
  static ac_int<2, true> decode_ternary_weight_sign_i2_local(double w);
  static ac_int<16, true> accumulate_ternary_mac_i16_local(
    ac_int<16, true> acc_i16,
    ac_int<8, true> qx_i8,
    ac_int<2, true> ternary_sign_i2);
  template<ac_ieee_float_format FloatFormat>
  static void quant_linear_token_32_to32_native(
    const typename RefOptimizedStorageBank<FloatFormat>::float_t x[D_MODEL],
    const double w[D_MODEL * D_MODEL],
    const double b[D_MODEL],
    float s_x,
    float s_w,
    typename RefOptimizedStorageBank<FloatFormat>::float_t y[D_MODEL]);
  template<ac_ieee_float_format FloatFormat>
  static void quant_linear_token_32_to128_native(
    const typename RefOptimizedStorageBank<FloatFormat>::float_t x[D_MODEL],
    const double w[FF_DIM * D_MODEL],
    const double b[FF_DIM],
    float s_x,
    float s_w,
    typename RefOptimizedStorageBank<FloatFormat>::float_t y[FF_DIM]);
  template<ac_ieee_float_format FloatFormat>
  static void quant_linear_token_128_to32_native(
    const typename RefOptimizedStorageBank<FloatFormat>::float_t x[FF_DIM],
    const double w[D_MODEL * FF_DIM],
    const double b[D_MODEL],
    float s_x,
    float s_w,
    typename RefOptimizedStorageBank<FloatFormat>::float_t y[D_MODEL]);

private:
  RefRunConfig run_cfg_;
  RefOptimizedNumericConfig numeric_cfg_;
  RefModel legacy_ref_;

  // Optimized storage banks. Runtime mode dispatch derives from numeric_cfg_.
  RefOptimizedStorageBank<binary16> storage_fp16_;
  RefOptimizedStorageBank<binary32> storage_fp32_;

  int last_staged_sample_index_;
  bool phase_a_valid_;
  bool layer0_attn_writeback_valid_;
  bool layer0_ln_writeback_valid_;
  bool layer0_ffn_writeback_valid_;
  bool mid_norm_writeback_valid_;
  bool layer1_attn_input_handoff_valid_;
  bool layer1_attn_writeback_valid_;
  bool layer1_ln_writeback_valid_;
  bool layer1_ffn_writeback_valid_;
  bool end_norm_writeback_valid_;
  ac_ieee_float<binary32> layer1_attn_input_dut_aligned_seed_[TOKENS_T][D_MODEL];
  ac_ieee_float<binary32> layer1_ffn_ln_out_seed_[TOKENS_T][D_MODEL];
  bool layer1_attn_input_dut_aligned_seed_valid_;
};

} // namespace aecct_ref
