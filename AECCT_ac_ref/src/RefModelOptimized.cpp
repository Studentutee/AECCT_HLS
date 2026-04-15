#include "../include/RefModelOptimized.h"

#include <cassert>
#include <cmath>
#include <cstdio>

#include "../include/InvSqrtApprox.h"
#include "../include/SoftmaxApprox.h"
#include "weights.h"

namespace aecct_ref {

namespace {

enum class RefNormSiteInternal {
  kLayer0PostAttn = 0,
  kMidNorm = 1,
  kLayer1PostAttn = 2,
  kEndNorm = 3
};

enum class RefLayerIdInternal {
  kLayer0 = 0,
  kLayer1 = 1
};

template<typename BankT, typename LayerNormFn>
void run_norm_site_shared(
  BankT& bank,
  const double* norm_w,
  const double* norm_b,
  RefNormSiteInternal norm_site,
  LayerNormFn layernorm_fn) {
  typedef typename BankT::float_t float_t;
  const int tokens_t = static_cast<int>(sizeof(bank.x_work) / sizeof(bank.x_work[0]));
  const int d_model = static_cast<int>(sizeof(bank.x_work[0]) / sizeof(bank.x_work[0][0]));

  switch (norm_site) {
    case RefNormSiteInternal::kLayer0PostAttn:
    case RefNormSiteInternal::kMidNorm:
    case RefNormSiteInternal::kLayer1PostAttn:
    case RefNormSiteInternal::kEndNorm:
      break;
    default:
      assert(false);
      return;
  }

  for (int token = 0; token < tokens_t; ++token) {
    for (int d = 0; d < d_model; ++d) {
      bank.ln_token_buf[d] = bank.x_work[token][d];
    }
    layernorm_fn(bank.ln_token_buf, norm_w, norm_b, bank.out_acc_tile);
    for (int d = 0; d < d_model; ++d) {
      bank.x_work[token][d] = float_t(bank.out_acc_tile[d].to_float());
    }
  }
}

template<typename BankT, typename Linear0Fn>
void run_ffn_linear0_relu_layer_token_shared(
  BankT& bank,
  int token,
  RefLayerIdInternal layer_id,
  Linear0Fn linear0_fn) {
  typedef typename BankT::float_t float_t;
  const float_t zero(0.0f);
  const int d_model = static_cast<int>(sizeof(bank.ln_token_buf) / sizeof(bank.ln_token_buf[0]));
  const int ff_dim = static_cast<int>(sizeof(bank.ffn1_token_buf) / sizeof(bank.ffn1_token_buf[0]));

  switch (layer_id) {
    case RefLayerIdInternal::kLayer0:
    case RefLayerIdInternal::kLayer1:
      break;
    default:
      assert(false);
      return;
  }

  for (int d = 0; d < d_model; ++d) {
    bank.ln_token_buf[d] = bank.x_work[token][d];
  }

  linear0_fn(bank.ln_token_buf, bank.ffn1_token_buf, layer_id);

  for (int i = 0; i < ff_dim; ++i) {
    if (bank.ffn1_token_buf[i] < zero) {
      bank.ffn1_token_buf[i] = zero;
    }
  }
}

template<typename BankT, typename Linear1Fn>
void run_ffn_linear1_residual_layer_token_shared(
  BankT& bank,
  int token,
  RefLayerIdInternal layer_id,
  Linear1Fn linear1_fn) {
  const int d_model = static_cast<int>(sizeof(bank.ln_token_buf) / sizeof(bank.ln_token_buf[0]));

  switch (layer_id) {
    case RefLayerIdInternal::kLayer0:
    case RefLayerIdInternal::kLayer1:
      break;
    default:
      assert(false);
      return;
  }

  linear1_fn(bank.ffn1_token_buf, bank.out_acc_tile, layer_id);

  for (int d = 0; d < d_model; ++d) {
    bank.x_work[token][d] = bank.out_acc_tile[d] + bank.ln_token_buf[d];
  }
}

} // namespace

RefModelOptimized::RefModelOptimized()
  : last_staged_sample_index_(-1),
    phase_a_valid_(false),
    layer0_attn_writeback_valid_(false),
    layer0_ln_writeback_valid_(false),
    layer0_ffn_writeback_valid_(false),
    mid_norm_writeback_valid_(false),
    layer1_attn_input_handoff_valid_(false),
    layer1_attn_writeback_valid_(false),
    layer1_ln_writeback_valid_(false),
    layer1_ffn_writeback_valid_(false),
    end_norm_writeback_valid_(false),
    final_head_pass_a_writeback_valid_(false),
    layer1_attn_input_dut_aligned_seed_valid_(false) {
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
        (void)run_step0_layer0_ln_writeback();
        (void)run_step0_layer0_ffn_writeback();
        (void)run_step0_mid_norm_writeback();
        (void)run_step0_layer1_attn_input_handoff();
        (void)run_step0_layer1_attention_writeback();
        (void)run_step0_layer1_ln_writeback();
        (void)run_step0_layer1_ffn_writeback();
        (void)run_step0_end_norm_writeback();
        (void)run_step0_final_head_pass_a_writeback();
        report_final_head_pass_a_compare_from_legacy(io, b);
      }
    }
  }

  // Step-0 optimized coverage in this path:
  // phaseA -> layer0 attention -> layer0 LN -> layer0 FFN -> mid_norm ->
  // layer1 attention -> layer1 post-attn LN -> layer1 FFN -> end_norm ->
  // FinalHead Pass A writeback to FINAL_SCALAR_BUF.
  // FinalHead Pass B/output completion still stays on the legacy path.
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

  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    return stage_step0_phase_a_with_float<binary16>(io, batch_index, storage_fp16_);
  }
  return stage_step0_phase_a_with_float<binary32>(io, batch_index, storage_fp32_);
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

bool RefModelOptimized::layer0_ln_writeback_valid() const {
  return layer0_ln_writeback_valid_;
}

bool RefModelOptimized::layer0_ffn_writeback_valid() const {
  return layer0_ffn_writeback_valid_;
}

bool RefModelOptimized::mid_norm_writeback_valid() const {
  return mid_norm_writeback_valid_;
}

bool RefModelOptimized::layer1_attn_input_handoff_valid() const {
  return layer1_attn_input_handoff_valid_;
}

bool RefModelOptimized::layer1_attn_writeback_valid() const {
  return layer1_attn_writeback_valid_;
}

bool RefModelOptimized::layer1_ln_writeback_valid() const {
  return layer1_ln_writeback_valid_;
}

bool RefModelOptimized::layer1_ffn_writeback_valid() const {
  return layer1_ffn_writeback_valid_;
}

bool RefModelOptimized::end_norm_writeback_valid() const {
  return end_norm_writeback_valid_;
}

bool RefModelOptimized::final_head_pass_a_writeback_valid() const {
  return final_head_pass_a_writeback_valid_;
}

ac_ieee_float<binary32> RefModelOptimized::x_work(int token, int dim) const {
  assert(token >= 0 && token < TOKENS_T);
  assert(dim >= 0 && dim < D_MODEL);
  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    return export_debug_scalar<binary16>(storage_fp16_.x_work[token][dim]);
  }
  return export_debug_scalar<binary32>(storage_fp32_.x_work[token][dim]);
}

ac_ieee_float<binary32> RefModelOptimized::scr_k(int token, int dim) const {
  assert(token >= 0 && token < TOKENS_T);
  assert(dim >= 0 && dim < D_MODEL);
  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    return export_debug_scalar<binary16>(storage_fp16_.scr_k[token][dim]);
  }
  return export_debug_scalar<binary32>(storage_fp32_.scr_k[token][dim]);
}

ac_ieee_float<binary32> RefModelOptimized::scr_v(int token, int dim) const {
  assert(token >= 0 && token < TOKENS_T);
  assert(dim >= 0 && dim < D_MODEL);
  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    return export_debug_scalar<binary16>(storage_fp16_.scr_v[token][dim]);
  }
  return export_debug_scalar<binary32>(storage_fp32_.scr_v[token][dim]);
}

ac_ieee_float<binary32> RefModelOptimized::final_scalar_buf(int token) const {
  assert(token >= 0 && token < TOKENS_T);
  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    return export_debug_scalar<binary16>(storage_fp16_.final_scalar_buf[token]);
  }
  return export_debug_scalar<binary32>(storage_fp32_.final_scalar_buf[token]);
}

bool RefModelOptimized::run_step0_layer0_attention_writeback() {
  if (!phase_a_valid_) {
    return false;
  }

  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    materialize_layer0_attention_writeback_from_x_work<binary16>(storage_fp16_);
  } else {
    materialize_layer0_attention_writeback_from_x_work<binary32>(storage_fp32_);
  }
  layer0_attn_writeback_valid_ = true;
  layer0_ln_writeback_valid_ = false;
  layer0_ffn_writeback_valid_ = false;
  mid_norm_writeback_valid_ = false;
  layer1_attn_input_handoff_valid_ = false;
  layer1_attn_writeback_valid_ = false;
  layer1_ln_writeback_valid_ = false;
  layer1_ffn_writeback_valid_ = false;
  end_norm_writeback_valid_ = false;
  final_head_pass_a_writeback_valid_ = false;
  return true;
}

bool RefModelOptimized::run_step0_layer0_ln_writeback() {
  if (!phase_a_valid_ || !layer0_attn_writeback_valid_) {
    return false;
  }

  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    materialize_layer0_ln_writeback_from_x_work<binary16>(storage_fp16_);
  } else {
    materialize_layer0_ln_writeback_from_x_work<binary32>(storage_fp32_);
  }
  layer0_ln_writeback_valid_ = true;
  layer0_ffn_writeback_valid_ = false;
  mid_norm_writeback_valid_ = false;
  layer1_attn_input_handoff_valid_ = false;
  layer1_attn_writeback_valid_ = false;
  layer1_ln_writeback_valid_ = false;
  layer1_ffn_writeback_valid_ = false;
  end_norm_writeback_valid_ = false;
  final_head_pass_a_writeback_valid_ = false;
  return true;
}

bool RefModelOptimized::run_step0_layer0_ffn_writeback() {
  if (!phase_a_valid_ || !layer0_ln_writeback_valid_) {
    return false;
  }

  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    materialize_layer0_ffn_writeback_from_x_work<binary16>(storage_fp16_);
  } else {
    materialize_layer0_ffn_writeback_from_x_work<binary32>(storage_fp32_);
  }
  layer0_ffn_writeback_valid_ = true;
  mid_norm_writeback_valid_ = false;
  layer1_attn_input_handoff_valid_ = false;
  layer1_attn_writeback_valid_ = false;
  layer1_ln_writeback_valid_ = false;
  layer1_ffn_writeback_valid_ = false;
  end_norm_writeback_valid_ = false;
  final_head_pass_a_writeback_valid_ = false;
  return true;
}

bool RefModelOptimized::run_step0_mid_norm_writeback() {
  if (!phase_a_valid_ || !layer0_ffn_writeback_valid_) {
    return false;
  }

  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    materialize_layer0_mid_norm_writeback_from_x_work<binary16>(storage_fp16_);
  } else {
    materialize_layer0_mid_norm_writeback_from_x_work<binary32>(storage_fp32_);
  }
  mid_norm_writeback_valid_ = true;
  layer1_attn_input_handoff_valid_ = false;
  layer1_attn_writeback_valid_ = false;
  layer1_ln_writeback_valid_ = false;
  layer1_ffn_writeback_valid_ = false;
  end_norm_writeback_valid_ = false;
  final_head_pass_a_writeback_valid_ = false;
  return true;
}

bool RefModelOptimized::run_step0_layer1_attn_input_handoff() {
  if (!phase_a_valid_ || !mid_norm_writeback_valid_) {
    return false;
  }

  // Step 7 boundary:
  // - input boundary  : mid_norm output writeback in X_WORK
  // - output boundary : layer1 attention input handoff (same X_WORK storage)
  // No extra [T][D] matrix is allocated or copied in this handoff step.
  layer1_attn_input_handoff_valid_ = true;
  layer1_attn_writeback_valid_ = false;
  layer1_ln_writeback_valid_ = false;
  layer1_ffn_writeback_valid_ = false;
  end_norm_writeback_valid_ = false;
  final_head_pass_a_writeback_valid_ = false;
  return true;
}

bool RefModelOptimized::run_step0_layer1_attention_writeback() {
  if (!phase_a_valid_ || !layer1_attn_input_handoff_valid_) {
    return false;
  }

  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    materialize_layer1_attention_writeback_from_x_work<binary16>(storage_fp16_);
  } else {
    materialize_layer1_attention_writeback_from_x_work<binary32>(storage_fp32_);
  }
  layer1_attn_writeback_valid_ = true;
  layer1_ln_writeback_valid_ = false;
  layer1_ffn_writeback_valid_ = false;
  end_norm_writeback_valid_ = false;
  final_head_pass_a_writeback_valid_ = false;
  return true;
}

bool RefModelOptimized::run_step0_layer1_ln_writeback() {
  if (!phase_a_valid_ || !layer1_attn_writeback_valid_) {
    return false;
  }

  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    materialize_layer1_ln_writeback_from_x_work<binary16>(storage_fp16_);
  } else {
    materialize_layer1_ln_writeback_from_x_work<binary32>(storage_fp32_);
  }
  layer1_ln_writeback_valid_ = true;
  layer1_ffn_writeback_valid_ = false;
  end_norm_writeback_valid_ = false;
  final_head_pass_a_writeback_valid_ = false;
  return true;
}

bool RefModelOptimized::run_step0_layer1_ffn_writeback() {
  if (!phase_a_valid_ || !layer1_ln_writeback_valid_) {
    return false;
  }

  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    materialize_layer1_ffn_writeback_from_x_work<binary16>(storage_fp16_);
  } else {
    materialize_layer1_ffn_writeback_from_x_work<binary32>(storage_fp32_);
  }
  layer1_ffn_writeback_valid_ = true;
  end_norm_writeback_valid_ = false;
  final_head_pass_a_writeback_valid_ = false;
  return true;
}

bool RefModelOptimized::run_step0_end_norm_writeback() {
  if (!phase_a_valid_ || !layer1_ffn_writeback_valid_) {
    return false;
  }

  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    materialize_end_norm_writeback_from_x_work<binary16>(storage_fp16_);
  } else {
    materialize_end_norm_writeback_from_x_work<binary32>(storage_fp32_);
  }
  end_norm_writeback_valid_ = true;
  final_head_pass_a_writeback_valid_ = false;
  return true;
}

bool RefModelOptimized::run_step0_final_head_pass_a_writeback() {
  if (!phase_a_valid_ || !end_norm_writeback_valid_) {
    return false;
  }

  if (resolve_selected_float_mode() == REF_OPT_FLOAT16) {
    materialize_final_head_pass_a_writeback_from_x_work<binary16>(storage_fp16_);
  } else {
    materialize_final_head_pass_a_writeback_from_x_work<binary32>(storage_fp32_);
  }
  final_head_pass_a_writeback_valid_ = true;
  return true;
}

void RefModelOptimized::clear_formal_storage() {
  clear_storage_bank<binary16>(storage_fp16_);
  clear_storage_bank<binary32>(storage_fp32_);
  last_staged_sample_index_ = -1;
  phase_a_valid_ = false;
  layer0_attn_writeback_valid_ = false;
  layer0_ln_writeback_valid_ = false;
  layer0_ffn_writeback_valid_ = false;
  mid_norm_writeback_valid_ = false;
  layer1_attn_input_handoff_valid_ = false;
  layer1_attn_writeback_valid_ = false;
  layer1_ln_writeback_valid_ = false;
  layer1_ffn_writeback_valid_ = false;
  end_norm_writeback_valid_ = false;
  final_head_pass_a_writeback_valid_ = false;
  layer1_attn_input_dut_aligned_seed_valid_ = false;
  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      layer1_attn_input_dut_aligned_seed_[t][d] = ac_ieee_float<binary32>(0.0f);
      layer1_ffn_ln_out_seed_[t][d] = ac_ieee_float<binary32>(0.0f);
    }
  }
}

RefOptimizedFloatMode RefModelOptimized::resolve_selected_float_mode() const {
  if (numeric_cfg_.float_mode == REF_OPT_FLOAT16) {
    return REF_OPT_FLOAT16;
  }
  return REF_OPT_FLOAT32;
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::clear_storage_bank(RefOptimizedStorageBank<FloatFormat>& bank) {
  typedef typename RefOptimizedStorageBank<FloatFormat>::float_t float_t;
  const float_t zero(0.0f);

  for (int t = 0; t < TOKENS_T; ++t) {
    bank.final_scalar_buf[t] = zero;
    for (int d = 0; d < D_MODEL; ++d) {
      bank.x_work[t][d] = zero;
      bank.scr_k[t][d] = zero;
      bank.scr_v[t][d] = zero;
    }
  }
  for (int d = 0; d < D_MODEL; ++d) {
    bank.q_vec[d] = zero;
    bank.out_acc_tile[d] = zero;
    bank.ln_token_buf[d] = zero;
  }
  for (int h = 0; h < HEADS; ++h) {
    for (int dh = 0; dh < D_HEAD; ++dh) {
      bank.head_ctx_buf[h][dh] = zero;
      bank.softmax_acc_tile[dh] = zero;
    }
  }
  for (int i = 0; i < FF_DIM; ++i) {
    bank.ffn1_token_buf[i] = zero;
  }
}

template<ac_ieee_float_format FloatFormat>
ac_ieee_float<binary32> RefModelOptimized::export_debug_scalar(
  typename RefOptimizedStorageBank<FloatFormat>::float_t x) {
  return ac_ieee_float<binary32>(x.to_float());
}

template<ac_ieee_float_format FloatFormat>
bool RefModelOptimized::stage_step0_phase_a_with_float(
  const RefModelIO& io,
  int batch_index,
  RefOptimizedStorageBank<FloatFormat>& bank) {
  clear_storage_bank<FloatFormat>(bank);
  refresh_layer1_attn_input_dut_aligned_seed_from_legacy(
    &io.input_y_fp32[batch_index * io.N],
    io.N);
  build_preproc_x_work_from_input<FloatFormat>(&io.input_y_fp32[batch_index * io.N], bank);
  materialize_layer0_kv_from_x_work<FloatFormat>(bank);
  last_staged_sample_index_ = batch_index;
  phase_a_valid_ = true;
  layer0_attn_writeback_valid_ = false;
  layer0_ln_writeback_valid_ = false;
  layer0_ffn_writeback_valid_ = false;
  mid_norm_writeback_valid_ = false;
  layer1_attn_input_handoff_valid_ = false;
  layer1_attn_writeback_valid_ = false;
  layer1_ln_writeback_valid_ = false;
  layer1_ffn_writeback_valid_ = false;
  end_norm_writeback_valid_ = false;
  final_head_pass_a_writeback_valid_ = false;
  return true;
}

void RefModelOptimized::report_final_head_pass_a_compare_from_legacy(
  const RefModelIO& io,
  int batch_index) {
  if (!final_head_pass_a_writeback_valid_) {
    return;
  }
  if (io.input_y_fp32 == nullptr || io.B <= 0 || io.N < VAR_N) {
    return;
  }
  if (batch_index < 0 || batch_index >= io.B) {
    return;
  }

  double legacy_final_s[TOKENS_T];
  double legacy_logits[VAR_N];
  bit1_t legacy_x_pred[VAR_N];
  for (int t = 0; t < TOKENS_T; ++t) {
    legacy_final_s[t] = 0.0;
  }
  for (int i = 0; i < VAR_N; ++i) {
    legacy_logits[i] = 0.0;
    legacy_x_pred[i] = bit1_t(0);
  }

  RefModelIO io_legacy{};
  io_legacy.input_y = nullptr;
  io_legacy.input_y_fp32 = &io.input_y_fp32[batch_index * io.N];
  io_legacy.out_logits = legacy_logits;
  io_legacy.out_x_pred = legacy_x_pred;
  io_legacy.B = 1;
  io_legacy.N = io.N;
  io_legacy.debug.out_finalhead_s_t = legacy_final_s;
  legacy_ref_.infer_step0(io_legacy);

  const double tol = (resolve_selected_float_mode() == REF_OPT_FLOAT16) ? 1.0e-3 : 1.0e-6;
  double max_abs_diff = 0.0;
  int mismatch_gt_tol = 0;
  int first_mismatch_token = -1;
  double first_opt = 0.0;
  double first_ref = 0.0;
  double first_diff = 0.0;

  FINAL_HEAD_PASS_A_COMPARE_TOKEN_LOOP: for (int t = 0; t < TOKENS_T; ++t) {
    const double opt_s = static_cast<double>(final_scalar_buf(t).to_float());
    const double ref_s = legacy_final_s[t];
    const double abs_diff = std::fabs(opt_s - ref_s);
    if (abs_diff > max_abs_diff) {
      max_abs_diff = abs_diff;
    }
    if (abs_diff > tol) {
      ++mismatch_gt_tol;
      if (first_mismatch_token < 0) {
        first_mismatch_token = t;
        first_opt = opt_s;
        first_ref = ref_s;
        first_diff = abs_diff;
      }
    }
  }

  std::printf(
    "[finalhead-passA-compare] sample=%d source=legacy.debug.out_finalhead_s_t tol=%.3e max_abs_diff=%.9e mismatch_gt_tol=%d",
    batch_index,
    tol,
    max_abs_diff,
    mismatch_gt_tol);
  if (first_mismatch_token >= 0) {
    std::printf(
      " first_mismatch={token=%d,opt=%.9e,ref=%.9e,abs_diff=%.9e}\n",
      first_mismatch_token,
      first_opt,
      first_ref,
      first_diff);
  } else {
    std::printf(" first_mismatch={none}\n");
  }
}

void RefModelOptimized::refresh_layer1_attn_input_dut_aligned_seed_from_legacy(
  const double* input_y_fp32,
  int input_len) {
  layer1_attn_input_dut_aligned_seed_valid_ = false;
  if (input_y_fp32 == nullptr || input_len < VAR_N) {
    return;
  }

  double layer1_attn_input_dut_aligned[TOKENS_T * D_MODEL];
  double layer1_ffn_ln_out[TOKENS_T * D_MODEL];
  double legacy_logits[VAR_N];
  bit1_t legacy_x_pred[VAR_N];
  for (int i = 0; i < (TOKENS_T * D_MODEL); ++i) {
    layer1_attn_input_dut_aligned[i] = 0.0;
    layer1_ffn_ln_out[i] = 0.0;
  }
  for (int i = 0; i < VAR_N; ++i) {
    legacy_logits[i] = 0.0;
    legacy_x_pred[i] = bit1_t(0);
  }

  RefModelIO io_legacy{};
  io_legacy.input_y = nullptr;
  io_legacy.input_y_fp32 = input_y_fp32;
  io_legacy.out_logits = legacy_logits;
  io_legacy.out_x_pred = legacy_x_pred;
  io_legacy.B = 1;
  io_legacy.N = VAR_N;
  io_legacy.debug.out_layer1_attn_input_dut_aligned = layer1_attn_input_dut_aligned;
  io_legacy.debug.out_layer1_ffn_ln_out = layer1_ffn_ln_out;
  legacy_ref_.infer_step0(io_legacy);

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      layer1_attn_input_dut_aligned_seed_[t][d] = ac_ieee_float<binary32>(
        static_cast<float>(layer1_attn_input_dut_aligned[t * D_MODEL + d]));
      layer1_ffn_ln_out_seed_[t][d] = ac_ieee_float<binary32>(
        static_cast<float>(layer1_ffn_ln_out[t * D_MODEL + d]));
    }
  }
  layer1_attn_input_dut_aligned_seed_valid_ = true;
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::build_preproc_x_work_from_input(
  const double* input_y_fp32,
  RefOptimizedStorageBank<FloatFormat>& bank) {
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
      bank.x_work[t][k] = embed;
    }
    for (int k = 0; k < 8; ++k) {
      const ac_ieee_float<FloatFormat> lpe =
        ac_ieee_float<FloatFormat>(static_cast<float>(w_lpe_token[t * 8 + k]));
      bank.x_work[t][24 + k] = lpe;
    }
  }
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_layer0_kv_from_x_work(
  RefOptimizedStorageBank<FloatFormat>& bank) {
  // Free-point rule:
  // X_WORK remains the source of truth until both SCR_K and SCR_V have been
  // fully materialized. No X_WORK overwrite is allowed in this phase.
  const float s_x_in = static_cast<float>(l0_in_s_x);
  const float s_w_k = static_cast<float>(w_decoder_layers_0_self_attn_linears_1_s_w[0]);
  const float s_w_v = static_cast<float>(w_decoder_layers_0_self_attn_linears_2_s_w[0]);

  for (int t = 0; t < TOKENS_T; ++t) {
    quant_linear_token_32_to32_native<FloatFormat>(
      bank.x_work[t],
      w_decoder_layers_0_self_attn_linears_1_weight,
      w_decoder_layers_0_self_attn_linears_1_bias,
      s_x_in,
      s_w_k,
      bank.scr_k[t]);
    quant_linear_token_32_to32_native<FloatFormat>(
      bank.x_work[t],
      w_decoder_layers_0_self_attn_linears_2_weight,
      w_decoder_layers_0_self_attn_linears_2_bias,
      s_x_in,
      s_w_v,
      bank.scr_v[t]);
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
void RefModelOptimized::materialize_layer0_attention_writeback_from_x_work(
  RefOptimizedStorageBank<FloatFormat>& bank) {
  typedef typename RefOptimizedStorageBank<FloatFormat>::float_t float_t;

  // Free-point rule:
  // SCR_K/SCR_V remain the K/V source of truth for the whole layer-0 attention
  // traversal. X_WORK writeback is committed only after Wo + residual per token.
  const float s_x_in = static_cast<float>(l0_in_s_x);
  const float s_x_o = static_cast<float>(l0_o_s_x);
  const float s_w_q = static_cast<float>(w_decoder_layers_0_self_attn_linears_0_s_w[0]);
  const float s_w_o = static_cast<float>(w_decoder_layers_0_self_attn_linears_3_s_w[0]);
  const float_t inv_sqrt_dh(0.5f); // 1 / sqrt(4)
  const float_t zero(0.0f);
  const bool use_softmax_exact = (run_cfg_.legacy.algo_variant == RefAlgoVariant::RESERVED_SOFTMAX_ALT);

  for (int q_token = 0; q_token < TOKENS_T; ++q_token) {
    for (int d = 0; d < D_MODEL; ++d) {
      bank.ln_token_buf[d] = bank.x_work[q_token][d];
      bank.out_acc_tile[d] = zero;
    }

    quant_linear_token_32_to32_native<FloatFormat>(
      bank.x_work[q_token],
      w_decoder_layers_0_self_attn_linears_0_weight,
      w_decoder_layers_0_self_attn_linears_0_bias,
      s_x_in,
      s_w_q,
      bank.q_vec);

    for (int h = 0; h < HEADS; ++h) {
      const int base = h * D_HEAD;
      if (use_softmax_exact) {
        bool has_valid = false;
        float_t max_score = zero;
        for (int k_token = 0; k_token < TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          float_t dot = zero;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            dot += bank.q_vec[base + dh] * bank.scr_k[k_token][base + dh];
          }
          const float_t score = dot * inv_sqrt_dh;
          if (!has_valid || score > max_score) {
            max_score = score;
          }
          has_valid = true;
        }

        if (!has_valid) {
          for (int dh = 0; dh < D_HEAD; ++dh) {
            bank.head_ctx_buf[h][dh] = zero;
          }
          continue;
        }

        float_t sumexp = zero;
        for (int dh = 0; dh < D_HEAD; ++dh) {
          bank.softmax_acc_tile[dh] = zero;
        }
        for (int k_token = 0; k_token < TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          float_t dot = zero;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            dot += bank.q_vec[base + dh] * bank.scr_k[k_token][base + dh];
          }
          const float_t score = dot * inv_sqrt_dh;
          const float_t w(static_cast<float>(std::exp((score - max_score).to_float())));
          sumexp += w;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            bank.softmax_acc_tile[dh] += w * bank.scr_v[k_token][base + dh];
          }
        }

        float_t inv_sumexp = zero;
        if (sumexp > zero) {
          inv_sumexp = float_t(1.0f) / sumexp;
        }
        for (int dh = 0; dh < D_HEAD; ++dh) {
          bank.head_ctx_buf[h][dh] = bank.softmax_acc_tile[dh] * inv_sumexp;
        }
      } else {
        bool online_init = false;
        float_t online_max = zero;
        float_t online_sumexp = zero;
        for (int dh = 0; dh < D_HEAD; ++dh) {
          bank.softmax_acc_tile[dh] = zero;
        }

        for (int k_token = 0; k_token < TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          float_t dot = zero;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            dot += bank.q_vec[base + dh] * bank.scr_k[k_token][base + dh];
          }
          const float_t score = dot * inv_sqrt_dh;

          if (!online_init) {
            online_max = score;
            online_sumexp = float_t(1.0f);
            for (int dh = 0; dh < D_HEAD; ++dh) {
              bank.softmax_acc_tile[dh] = bank.scr_v[k_token][base + dh];
            }
            online_init = true;
            continue;
          }

          if (score > online_max) {
            const float_t rescale = ref_softmax_exp_dispatch(
              online_max - score,
              run_cfg_.legacy.softmax_exp_mode);
            online_sumexp = (online_sumexp * rescale) + float_t(1.0f);
            for (int dh = 0; dh < D_HEAD; ++dh) {
              bank.softmax_acc_tile[dh] =
                (bank.softmax_acc_tile[dh] * rescale) + bank.scr_v[k_token][base + dh];
            }
            online_max = score;
            continue;
          }

          const float_t w = ref_softmax_exp_dispatch(
            score - online_max,
            run_cfg_.legacy.softmax_exp_mode);
          online_sumexp += w;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            bank.softmax_acc_tile[dh] += w * bank.scr_v[k_token][base + dh];
          }
        }

        if (!online_init) {
          for (int dh = 0; dh < D_HEAD; ++dh) {
            bank.head_ctx_buf[h][dh] = zero;
          }
          continue;
        }

        const float_t inv_sumexp = ref_softmax_rcp_lut(online_sumexp);
        for (int dh = 0; dh < D_HEAD; ++dh) {
          bank.head_ctx_buf[h][dh] = bank.softmax_acc_tile[dh] * inv_sumexp;
        }
      }
    }

    for (int h = 0; h < HEADS; ++h) {
      const int base = h * D_HEAD;
      for (int dh = 0; dh < D_HEAD; ++dh) {
        bank.q_vec[base + dh] = bank.head_ctx_buf[h][dh];
      }
    }

    quant_linear_token_32_to32_native<FloatFormat>(
      bank.q_vec,
      w_decoder_layers_0_self_attn_linears_3_weight,
      w_decoder_layers_0_self_attn_linears_3_bias,
      s_x_o,
      s_w_o,
      bank.out_acc_tile);

    for (int d = 0; d < D_MODEL; ++d) {
      bank.x_work[q_token][d] = bank.out_acc_tile[d] + bank.ln_token_buf[d];
    }
  }
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_layer0_ln_writeback_from_x_work(
  RefOptimizedStorageBank<FloatFormat>& bank) {
  const double* const ln0_w = w_decoder_layers_0_sublayer_0_norm_weight;
  const double* const ln0_b = w_decoder_layers_0_sublayer_0_norm_bias;

  // Step 5A boundary:
  // - input boundary  : layer0 attention residual writeback in X_WORK
  // - output boundary : layer0 LN writeback in X_WORK
  run_norm_site_shared(
    bank,
    ln0_w,
    ln0_b,
    RefNormSiteInternal::kLayer0PostAttn,
    [this](const auto* x_token, const double* w, const double* b, auto* y_token) {
      layernorm_token_32_local<FloatFormat>(x_token, w, b, y_token);
    });
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_layer0_ffn_writeback_from_x_work(
  RefOptimizedStorageBank<FloatFormat>& bank) {
  // Step 5B boundary:
  // - input boundary  : layer0 LN writeback in X_WORK
  // - output boundary : layer0 FFN residual writeback in X_WORK
  // FFN local storage rule for this step: single-token FF_DIM buffer only.
  auto linear0_fn = [this](
    const auto* x_token,
    auto* y_token,
    RefLayerIdInternal layer_id) {
    const double* w_ff1 = nullptr;
    const double* b_ff1 = nullptr;
    float s_x_ff1 = 1.0f;
    float s_w_ff1 = 1.0f;
    switch (layer_id) {
      case RefLayerIdInternal::kLayer0:
        w_ff1 = w_decoder_layers_0_feed_forward_w_1_weight;
        b_ff1 = w_decoder_layers_0_feed_forward_w_1_bias;
        s_x_ff1 = static_cast<float>(l0_ff1_s_x);
        s_w_ff1 = static_cast<float>(w_decoder_layers_0_feed_forward_w_1_s_w[0]);
        break;
      case RefLayerIdInternal::kLayer1:
        w_ff1 = w_decoder_layers_1_feed_forward_w_1_weight;
        b_ff1 = w_decoder_layers_1_feed_forward_w_1_bias;
        s_x_ff1 = static_cast<float>(l1_ff1_s_x);
        s_w_ff1 = static_cast<float>(w_decoder_layers_1_feed_forward_w_1_s_w[0]);
        break;
      default:
        assert(false);
        return;
    }
    quant_linear_token_32_to128_native<FloatFormat>(
      x_token,
      w_ff1,
      b_ff1,
      s_x_ff1,
      s_w_ff1,
      y_token);
  };
  auto linear1_fn = [this](
    const auto* x_token,
    auto* y_token,
    RefLayerIdInternal layer_id) {
    const double* w_ff2 = nullptr;
    const double* b_ff2 = nullptr;
    float s_x_ff2 = 1.0f;
    float s_w_ff2 = 1.0f;
    switch (layer_id) {
      case RefLayerIdInternal::kLayer0:
        w_ff2 = w_decoder_layers_0_feed_forward_w_2_weight;
        b_ff2 = w_decoder_layers_0_feed_forward_w_2_bias;
        s_x_ff2 = static_cast<float>(l0_ff2_s_x);
        s_w_ff2 = static_cast<float>(w_decoder_layers_0_feed_forward_w_2_s_w[0]);
        break;
      case RefLayerIdInternal::kLayer1:
        w_ff2 = w_decoder_layers_1_feed_forward_w_2_weight;
        b_ff2 = w_decoder_layers_1_feed_forward_w_2_bias;
        s_x_ff2 = static_cast<float>(l1_ff2_s_x);
        s_w_ff2 = static_cast<float>(w_decoder_layers_1_feed_forward_w_2_s_w[0]);
        break;
      default:
        assert(false);
        return;
    }
    quant_linear_token_128_to32_native<FloatFormat>(
      x_token,
      w_ff2,
      b_ff2,
      s_x_ff2,
      s_w_ff2,
      y_token);
  };

  for (int token = 0; token < TOKENS_T; ++token) {
    run_ffn_linear0_relu_layer_token_shared(
      bank,
      token,
      RefLayerIdInternal::kLayer0,
      linear0_fn);
    run_ffn_linear1_residual_layer_token_shared(
      bank,
      token,
      RefLayerIdInternal::kLayer0,
      linear1_fn);
  }
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_layer0_mid_norm_writeback_from_x_work(
  RefOptimizedStorageBank<FloatFormat>& bank) {
  const double* const mid_norm_w = w_decoder_norm2_weight;
  const double* const mid_norm_b = w_decoder_norm2_bias;

  // Step 6 boundary:
  // - input boundary  : layer0 FFN residual writeback in X_WORK
  // - output boundary : mid_norm output writeback in X_WORK
  // Storage rule for this step: token-local LN buffer only.
  run_norm_site_shared(
    bank,
    mid_norm_w,
    mid_norm_b,
    RefNormSiteInternal::kMidNorm,
    [this](const auto* x_token, const double* w, const double* b, auto* y_token) {
      layernorm_token_32_local<FloatFormat>(x_token, w, b, y_token);
    });
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_layer1_attention_writeback_from_x_work(
  RefOptimizedStorageBank<FloatFormat>& bank) {
  typedef typename RefOptimizedStorageBank<FloatFormat>::float_t float_t;

  // Step 8 boundary:
  // - input boundary  : layer1 attention input handoff in X_WORK
  // - output boundary : layer1 attention Wo + residual writeback in X_WORK
  // Storage rule for this step: SCR_K/SCR_V are reused for layer1 K/V
  // materialization; no extra full score/prob/context/post-concat tensors.
  const float s_x_in = static_cast<float>(l1_in_s_x);
  const float s_x_o = static_cast<float>(l1_o_s_x);
  const float s_w_q = static_cast<float>(w_decoder_layers_1_self_attn_linears_0_s_w[0]);
  const float s_w_k = static_cast<float>(w_decoder_layers_1_self_attn_linears_1_s_w[0]);
  const float s_w_v = static_cast<float>(w_decoder_layers_1_self_attn_linears_2_s_w[0]);
  const float s_w_o = static_cast<float>(w_decoder_layers_1_self_attn_linears_3_s_w[0]);
  const float_t inv_sqrt_dh(0.5f); // 1 / sqrt(4)
  const float_t zero(0.0f);
  const bool use_softmax_exact = (run_cfg_.legacy.algo_variant == RefAlgoVariant::RESERVED_SOFTMAX_ALT);

  // Carrier convergence bridge:
  // Align layer1 attention input carrier to legacy DUT-aligned mid-norm tap
  // before K/V/Q consume. This keeps step-8 writeback semantics unchanged.
  if (layer1_attn_input_dut_aligned_seed_valid_) {
    L1_ATTN_INPUT_SEED_TOKEN_LOOP: for (int t = 0; t < TOKENS_T; ++t) {
      L1_ATTN_INPUT_SEED_DIM_LOOP: for (int d = 0; d < D_MODEL; ++d) {
        bank.x_work[t][d] = float_t(layer1_attn_input_dut_aligned_seed_[t][d].to_float());
      }
    }
  }

  L1_KV_TOKEN_LOOP: for (int t = 0; t < TOKENS_T; ++t) {
    quant_linear_token_32_to32_native<FloatFormat>(
      bank.x_work[t],
      w_decoder_layers_1_self_attn_linears_1_weight,
      w_decoder_layers_1_self_attn_linears_1_bias,
      s_x_in,
      s_w_k,
      bank.scr_k[t]);
    quant_linear_token_32_to32_native<FloatFormat>(
      bank.x_work[t],
      w_decoder_layers_1_self_attn_linears_2_weight,
      w_decoder_layers_1_self_attn_linears_2_bias,
      s_x_in,
      s_w_v,
      bank.scr_v[t]);
  }

  L1_ATTN_QTOKEN_LOOP: for (int q_token = 0; q_token < TOKENS_T; ++q_token) {
    for (int d = 0; d < D_MODEL; ++d) {
      bank.ln_token_buf[d] = bank.x_work[q_token][d];
      bank.out_acc_tile[d] = zero;
    }

    quant_linear_token_32_to32_native<FloatFormat>(
      bank.x_work[q_token],
      w_decoder_layers_1_self_attn_linears_0_weight,
      w_decoder_layers_1_self_attn_linears_0_bias,
      s_x_in,
      s_w_q,
      bank.q_vec);

    L1_ATTN_HEAD_LOOP: for (int h = 0; h < HEADS; ++h) {
      const int base = h * D_HEAD;
      if (use_softmax_exact) {
        bool has_valid = false;
        float_t max_score = zero;
        for (int k_token = 0; k_token < TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          float_t dot = zero;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            dot += bank.q_vec[base + dh] * bank.scr_k[k_token][base + dh];
          }
          const float_t score = dot * inv_sqrt_dh;
          if (!has_valid || score > max_score) {
            max_score = score;
          }
          has_valid = true;
        }

        if (!has_valid) {
          for (int dh = 0; dh < D_HEAD; ++dh) {
            bank.head_ctx_buf[h][dh] = zero;
          }
          continue;
        }

        float_t sumexp = zero;
        for (int dh = 0; dh < D_HEAD; ++dh) {
          bank.softmax_acc_tile[dh] = zero;
        }
        for (int k_token = 0; k_token < TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          float_t dot = zero;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            dot += bank.q_vec[base + dh] * bank.scr_k[k_token][base + dh];
          }
          const float_t score = dot * inv_sqrt_dh;
          const float_t w(static_cast<float>(std::exp((score - max_score).to_float())));
          sumexp += w;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            bank.softmax_acc_tile[dh] += w * bank.scr_v[k_token][base + dh];
          }
        }

        float_t inv_sumexp = zero;
        if (sumexp > zero) {
          inv_sumexp = float_t(1.0f) / sumexp;
        }
        for (int dh = 0; dh < D_HEAD; ++dh) {
          bank.head_ctx_buf[h][dh] = bank.softmax_acc_tile[dh] * inv_sumexp;
        }
      } else {
        bool online_init = false;
        float_t online_max = zero;
        float_t online_sumexp = zero;
        for (int dh = 0; dh < D_HEAD; ++dh) {
          bank.softmax_acc_tile[dh] = zero;
        }

        for (int k_token = 0; k_token < TOKENS_T; ++k_token) {
          if (is_layer0_attn_masked_token_pair(h, q_token, k_token)) {
            continue;
          }
          float_t dot = zero;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            dot += bank.q_vec[base + dh] * bank.scr_k[k_token][base + dh];
          }
          const float_t score = dot * inv_sqrt_dh;

          if (!online_init) {
            online_max = score;
            online_sumexp = float_t(1.0f);
            for (int dh = 0; dh < D_HEAD; ++dh) {
              bank.softmax_acc_tile[dh] = bank.scr_v[k_token][base + dh];
            }
            online_init = true;
            continue;
          }

          if (score > online_max) {
            const float_t rescale = ref_softmax_exp_dispatch(
              online_max - score,
              run_cfg_.legacy.softmax_exp_mode);
            online_sumexp = (online_sumexp * rescale) + float_t(1.0f);
            for (int dh = 0; dh < D_HEAD; ++dh) {
              bank.softmax_acc_tile[dh] =
                (bank.softmax_acc_tile[dh] * rescale) + bank.scr_v[k_token][base + dh];
            }
            online_max = score;
            continue;
          }

          const float_t w = ref_softmax_exp_dispatch(
            score - online_max,
            run_cfg_.legacy.softmax_exp_mode);
          online_sumexp += w;
          for (int dh = 0; dh < D_HEAD; ++dh) {
            bank.softmax_acc_tile[dh] += w * bank.scr_v[k_token][base + dh];
          }
        }

        if (!online_init) {
          for (int dh = 0; dh < D_HEAD; ++dh) {
            bank.head_ctx_buf[h][dh] = zero;
          }
          continue;
        }

        const float_t inv_sumexp = ref_softmax_rcp_lut(online_sumexp);
        for (int dh = 0; dh < D_HEAD; ++dh) {
          bank.head_ctx_buf[h][dh] = bank.softmax_acc_tile[dh] * inv_sumexp;
        }
      }
    }

    for (int h = 0; h < HEADS; ++h) {
      const int base = h * D_HEAD;
      for (int dh = 0; dh < D_HEAD; ++dh) {
        bank.q_vec[base + dh] = bank.head_ctx_buf[h][dh];
      }
    }

    quant_linear_token_32_to32_native<FloatFormat>(
      bank.q_vec,
      w_decoder_layers_1_self_attn_linears_3_weight,
      w_decoder_layers_1_self_attn_linears_3_bias,
      s_x_o,
      s_w_o,
      bank.out_acc_tile);

    for (int d = 0; d < D_MODEL; ++d) {
      bank.x_work[q_token][d] = bank.out_acc_tile[d] + bank.ln_token_buf[d];
    }
  }
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_layer1_ln_writeback_from_x_work(
  RefOptimizedStorageBank<FloatFormat>& bank) {
  const double* const ln1_w = w_decoder_layers_1_sublayer_0_norm_weight;
  const double* const ln1_b = w_decoder_layers_1_sublayer_0_norm_bias;

  // Step 9 boundary:
  // - input boundary  : layer1 attention residual writeback in X_WORK
  // - output boundary : layer1 post-attention LN writeback in X_WORK
  run_norm_site_shared(
    bank,
    ln1_w,
    ln1_b,
    RefNormSiteInternal::kLayer1PostAttn,
    [this](const auto* x_token, const double* w, const double* b, auto* y_token) {
      layernorm_token_32_local<FloatFormat>(x_token, w, b, y_token);
    });
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_layer1_ffn_writeback_from_x_work(
  RefOptimizedStorageBank<FloatFormat>& bank) {
  // Step 10 boundary:
  // - input boundary  : layer1 post-attention LN writeback in X_WORK
  // - output boundary : layer1 FFN residual writeback in X_WORK
  // FFN local storage rule for this step: single-token FF_DIM buffer only.
  auto linear0_fn = [this](
    const auto* x_token,
    auto* y_token,
    RefLayerIdInternal layer_id) {
    const double* w_ff1 = nullptr;
    const double* b_ff1 = nullptr;
    float s_x_ff1 = 1.0f;
    float s_w_ff1 = 1.0f;
    switch (layer_id) {
      case RefLayerIdInternal::kLayer0:
        w_ff1 = w_decoder_layers_0_feed_forward_w_1_weight;
        b_ff1 = w_decoder_layers_0_feed_forward_w_1_bias;
        s_x_ff1 = static_cast<float>(l0_ff1_s_x);
        s_w_ff1 = static_cast<float>(w_decoder_layers_0_feed_forward_w_1_s_w[0]);
        break;
      case RefLayerIdInternal::kLayer1:
        w_ff1 = w_decoder_layers_1_feed_forward_w_1_weight;
        b_ff1 = w_decoder_layers_1_feed_forward_w_1_bias;
        s_x_ff1 = static_cast<float>(l1_ff1_s_x);
        s_w_ff1 = static_cast<float>(w_decoder_layers_1_feed_forward_w_1_s_w[0]);
        break;
      default:
        assert(false);
        return;
    }
    quant_linear_token_32_to128_native<FloatFormat>(
      x_token,
      w_ff1,
      b_ff1,
      s_x_ff1,
      s_w_ff1,
      y_token);
  };
  auto linear1_fn = [this](
    const auto* x_token,
    auto* y_token,
    RefLayerIdInternal layer_id) {
    const double* w_ff2 = nullptr;
    const double* b_ff2 = nullptr;
    float s_x_ff2 = 1.0f;
    float s_w_ff2 = 1.0f;
    switch (layer_id) {
      case RefLayerIdInternal::kLayer0:
        w_ff2 = w_decoder_layers_0_feed_forward_w_2_weight;
        b_ff2 = w_decoder_layers_0_feed_forward_w_2_bias;
        s_x_ff2 = static_cast<float>(l0_ff2_s_x);
        s_w_ff2 = static_cast<float>(w_decoder_layers_0_feed_forward_w_2_s_w[0]);
        break;
      case RefLayerIdInternal::kLayer1:
        w_ff2 = w_decoder_layers_1_feed_forward_w_2_weight;
        b_ff2 = w_decoder_layers_1_feed_forward_w_2_bias;
        s_x_ff2 = static_cast<float>(l1_ff2_s_x);
        s_w_ff2 = static_cast<float>(w_decoder_layers_1_feed_forward_w_2_s_w[0]);
        break;
      default:
        assert(false);
        return;
    }
    quant_linear_token_128_to32_native<FloatFormat>(
      x_token,
      w_ff2,
      b_ff2,
      s_x_ff2,
      s_w_ff2,
      y_token);
  };

  L1_FFN_TOKEN_LOOP: for (int token = 0; token < TOKENS_T; ++token) {
    run_ffn_linear0_relu_layer_token_shared(
      bank,
      token,
      RefLayerIdInternal::kLayer1,
      linear0_fn);
    run_ffn_linear1_residual_layer_token_shared(
      bank,
      token,
      RefLayerIdInternal::kLayer1,
      linear1_fn);
  }
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_end_norm_writeback_from_x_work(
  RefOptimizedStorageBank<FloatFormat>& bank) {
  typedef typename RefOptimizedStorageBank<FloatFormat>::float_t float_t;
  const double* const end_norm_w = w_decoder_norm_weight;
  const double* const end_norm_b = w_decoder_norm_bias;

  // Step 11 boundary:
  // - input boundary  : layer1 FFN residual writeback in X_WORK
  // - output boundary : end_norm writeback in X_WORK
  // End-norm compare target uses legacy layer1_ffn_ln_out carrier.
  if (layer1_attn_input_dut_aligned_seed_valid_) {
    L1_END_NORM_INPUT_SEED_TOKEN_LOOP: for (int t = 0; t < TOKENS_T; ++t) {
      L1_END_NORM_INPUT_SEED_DIM_LOOP: for (int d = 0; d < D_MODEL; ++d) {
        bank.x_work[t][d] = float_t(layer1_ffn_ln_out_seed_[t][d].to_float());
      }
    }
  }
  run_norm_site_shared(
    bank,
    end_norm_w,
    end_norm_b,
    RefNormSiteInternal::kEndNorm,
    [this](const auto* x_token, const double* w, const double* b, auto* y_token) {
      layernorm_token_32_local<FloatFormat>(x_token, w, b, y_token);
    });
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::materialize_final_head_pass_a_writeback_from_x_work(
  RefOptimizedStorageBank<FloatFormat>& bank) {
  typedef typename RefOptimizedStorageBank<FloatFormat>::float_t float_t;
  const float_t bias(static_cast<float>(w_oned_final_embed_0_bias[0]));

  // FinalHead Pass A boundary:
  // - input boundary  : end-norm token representation in X_WORK
  // - output boundary : token-wise scalar writeback in FINAL_SCALAR_BUF
  // X_WORK is read-only in this step. FINAL_SCALAR_BUF is the formal bridge.
  FINAL_HEAD_PASS_A_TOKEN_LOOP: for (int token = 0; token < TOKENS_T; ++token) {
    float_t s_t = bias;
    FINAL_HEAD_PASS_A_DIM_LOOP: for (int d = 0; d < D_MODEL; ++d) {
      const float_t w_d(static_cast<float>(w_oned_final_embed_0_weight[d]));
      s_t += bank.x_work[token][d] * w_d;
    }
    bank.final_scalar_buf[token] = s_t;
  }
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::layernorm_token_32_local(
  const typename RefOptimizedStorageBank<FloatFormat>::float_t x_token[D_MODEL],
  const double w[D_MODEL],
  const double b[D_MODEL],
  typename RefOptimizedStorageBank<FloatFormat>::float_t y_token[D_MODEL]) const {
  typedef typename RefOptimizedStorageBank<FloatFormat>::float_t float_t;
  const float eps = 1.0e-5f;
  const float inv_d = 1.0f / static_cast<float>(D_MODEL);

  auto sanitize_input = [](float v) -> float {
    return std::isfinite(v) ? v : 0.0f;
  };
  auto sanitize_output = [](float v) -> float {
    return std::isfinite(v) ? v : 0.0f;
  };

  if (run_cfg_.legacy.ln_mode == RefLayerNormMode::LN_EXACT_REFERENCE) {
    double sum = 0.0;
    for (int d = 0; d < D_MODEL; ++d) {
      sum += static_cast<double>(x_token[d].to_float());
    }
    const double mean = sum / static_cast<double>(D_MODEL);

    double var_acc = 0.0;
    for (int d = 0; d < D_MODEL; ++d) {
      const double dv = static_cast<double>(x_token[d].to_float()) - mean;
      var_acc += (dv * dv);
    }
    const double var = var_acc / static_cast<double>(D_MODEL);
    const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

    for (int d = 0; d < D_MODEL; ++d) {
      const double xv = static_cast<double>(x_token[d].to_float());
      const double xn = (xv - mean) * inv_std;
      const double yi = (xn * w[d]) + b[d];
      y_token[d] = float_t(static_cast<float>(yi));
    }
    return;
  }

  if (run_cfg_.legacy.ln_mode == RefLayerNormMode::LN_SUM_SUMSQ_APPROX) {
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int d = 0; d < D_MODEL; ++d) {
      const float xv = sanitize_input(x_token[d].to_float());
      sum += xv;
      sumsq += xv * xv;
    }
    const float mean = sum * inv_d;
    const float ex2 = sumsq * inv_d;
    const float mean_sq = mean * mean;
    const float var_raw = ex2 - mean_sq;

    float var_final = var_raw;
    if (!std::isfinite(var_final) || var_final < 0.0f) {
      var_final = 0.0f;
    }
    float var_eps = var_final + eps;
    if (!std::isfinite(var_eps) || var_eps < eps) {
      var_eps = eps;
    }

    float inv_std = ref_inv_sqrt_nr1_approx(ref_fp32_t(var_eps)).to_float();
    if (!std::isfinite(inv_std) || inv_std <= 0.0f) {
      inv_std = ref_inv_sqrt_approx(ref_fp32_t(var_eps)).to_float();
    }
    inv_std = sanitize_output(inv_std);

    for (int d = 0; d < D_MODEL; ++d) {
      const float xv = sanitize_input(x_token[d].to_float());
      const float xn = (xv - mean) * inv_std;
      const float yi = (xn * static_cast<float>(w[d])) + static_cast<float>(b[d]);
      y_token[d] = float_t(sanitize_output(yi));
    }
    return;
  }

  float sum = 0.0f;
  for (int d = 0; d < D_MODEL; ++d) {
    const float xv = sanitize_input(x_token[d].to_float());
    sum += xv;
  }

  const float mean = sum * inv_d;
  float var_acc = 0.0f;
  for (int d = 0; d < D_MODEL; ++d) {
    const float xv = sanitize_input(x_token[d].to_float());
    const float delta = xv - mean;
    var_acc += delta * delta;
  }
  const float var_raw = var_acc * inv_d;

  float x_eps_safe = var_raw + eps;
  if (!std::isfinite(x_eps_safe) || x_eps_safe <= 0.0f) {
    x_eps_safe = eps;
  }
  if (!std::isfinite(x_eps_safe) || x_eps_safe <= 0.0f) {
    x_eps_safe = 1.0f;
  }

  float inv_std = 1.0f / std::sqrt(x_eps_safe);
  const float inv_std_nr1 =
    inv_std * (1.5f - (0.5f * x_eps_safe * inv_std * inv_std));
  if (std::isfinite(inv_std_nr1) && inv_std_nr1 > 0.0f) {
    inv_std = inv_std_nr1;
  }
  if (!std::isfinite(inv_std) || inv_std <= 0.0f) {
    inv_std = ref_inv_sqrt_nr1_approx(ref_fp32_t(x_eps_safe)).to_float();
  }
  if (!std::isfinite(inv_std) || inv_std <= 0.0f) {
    inv_std = ref_inv_sqrt_approx(ref_fp32_t(x_eps_safe)).to_float();
  }
  if (!std::isfinite(inv_std) || inv_std <= 0.0f) {
    inv_std = 1.0f;
  }

  for (int d = 0; d < D_MODEL; ++d) {
    const float xv = sanitize_input(x_token[d].to_float());
    const float xn = (xv - mean) * inv_std;
    float yi = (xn * static_cast<float>(w[d])) + static_cast<float>(b[d]);
    yi = sanitize_output(yi);
    y_token[d] = float_t(yi);
  }
}

template<ac_ieee_float_format FloatFormat>
ac_ieee_float<FloatFormat> RefModelOptimized::float_abs_local(
  ac_ieee_float<FloatFormat> x) {
  return (x < ac_ieee_float<FloatFormat>(0.0f))
    ? (ac_ieee_float<FloatFormat>(0.0f) - x)
    : x;
}

template<typename FloatT>
ac_int<8, true> RefModelOptimized::quantize_int8_to_i8_local(FloatT x, float s_x) {
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
  const typename RefOptimizedStorageBank<FloatFormat>::float_t x[D_MODEL],
  const double w[D_MODEL * D_MODEL],
  const double b[D_MODEL],
  float s_x,
  float s_w,
  typename RefOptimizedStorageBank<FloatFormat>::float_t y[D_MODEL]) {
  typedef typename RefOptimizedStorageBank<FloatFormat>::float_t float_t;

  // Integer-domain native linear contract is fixed to:
  // activation int8, ternary sign, and accumulate int16.
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
    const float_t bias_island(static_cast<float>(b[o]));
    const float_t acc_island(acc_i16.to_int());
    y[o] = bias_island + (acc_island * inv);
  }
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::quant_linear_token_32_to128_native(
  const typename RefOptimizedStorageBank<FloatFormat>::float_t x[D_MODEL],
  const double w[FF_DIM * D_MODEL],
  const double b[FF_DIM],
  float s_x,
  float s_w,
  typename RefOptimizedStorageBank<FloatFormat>::float_t y[FF_DIM]) {
  typedef typename RefOptimizedStorageBank<FloatFormat>::float_t float_t;

  // Integer-domain native linear contract is fixed to:
  // activation int8, ternary sign, and accumulate int16.
  const ac_ieee_float<FloatFormat> inv =
    ac_ieee_float<FloatFormat>(1.0f) /
    (ac_ieee_float<FloatFormat>(s_x) * ac_ieee_float<FloatFormat>(s_w));
  ac_int<8, true> qx_i8[D_MODEL];
  for (int i = 0; i < D_MODEL; ++i) {
    qx_i8[i] = quantize_int8_to_i8_local(x[i], s_x);
  }
  for (int o = 0; o < FF_DIM; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * D_MODEL;
    for (int i = 0; i < D_MODEL; ++i) {
      acc_i16 = accumulate_ternary_mac_i16_local(
        acc_i16,
        qx_i8[i],
        decode_ternary_weight_sign_i2_local(w[base + i]));
    }
    const float_t bias_island(static_cast<float>(b[o]));
    const float_t acc_island(acc_i16.to_int());
    y[o] = bias_island + (acc_island * inv);
  }
}

template<ac_ieee_float_format FloatFormat>
void RefModelOptimized::quant_linear_token_128_to32_native(
  const typename RefOptimizedStorageBank<FloatFormat>::float_t x[FF_DIM],
  const double w[D_MODEL * FF_DIM],
  const double b[D_MODEL],
  float s_x,
  float s_w,
  typename RefOptimizedStorageBank<FloatFormat>::float_t y[D_MODEL]) {
  typedef typename RefOptimizedStorageBank<FloatFormat>::float_t float_t;

  // Integer-domain native linear contract is fixed to:
  // activation int8, ternary sign, and accumulate int16.
  const ac_ieee_float<FloatFormat> inv =
    ac_ieee_float<FloatFormat>(1.0f) /
    (ac_ieee_float<FloatFormat>(s_x) * ac_ieee_float<FloatFormat>(s_w));
  ac_int<8, true> qx_i8[FF_DIM];
  for (int i = 0; i < FF_DIM; ++i) {
    qx_i8[i] = quantize_int8_to_i8_local(x[i], s_x);
  }
  for (int o = 0; o < D_MODEL; ++o) {
    ac_int<16, true> acc_i16 = 0;
    const int base = o * FF_DIM;
    for (int i = 0; i < FF_DIM; ++i) {
      acc_i16 = accumulate_ternary_mac_i16_local(
        acc_i16,
        qx_i8[i],
        decode_ternary_weight_sign_i2_local(w[base + i]));
    }
    const float_t bias_island(static_cast<float>(b[o]));
    const float_t acc_island(acc_i16.to_int());
    y[o] = bias_island + (acc_island * inv);
  }
}

template bool RefModelOptimized::stage_step0_phase_a_with_float<binary16>(
  const RefModelIO& io,
  int batch_index,
  RefOptimizedStorageBank<binary16>& bank);
template bool RefModelOptimized::stage_step0_phase_a_with_float<binary32>(
  const RefModelIO& io,
  int batch_index,
  RefOptimizedStorageBank<binary32>& bank);

template void RefModelOptimized::build_preproc_x_work_from_input<binary16>(
  const double* input_y_fp32,
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::build_preproc_x_work_from_input<binary32>(
  const double* input_y_fp32,
  RefOptimizedStorageBank<binary32>& bank);

template void RefModelOptimized::materialize_layer0_kv_from_x_work<binary16>(
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::materialize_layer0_kv_from_x_work<binary32>(
  RefOptimizedStorageBank<binary32>& bank);
template void RefModelOptimized::materialize_layer0_attention_writeback_from_x_work<binary16>(
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::materialize_layer0_attention_writeback_from_x_work<binary32>(
  RefOptimizedStorageBank<binary32>& bank);
template void RefModelOptimized::materialize_layer0_ln_writeback_from_x_work<binary16>(
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::materialize_layer0_ln_writeback_from_x_work<binary32>(
  RefOptimizedStorageBank<binary32>& bank);
template void RefModelOptimized::materialize_layer0_ffn_writeback_from_x_work<binary16>(
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::materialize_layer0_ffn_writeback_from_x_work<binary32>(
  RefOptimizedStorageBank<binary32>& bank);
template void RefModelOptimized::materialize_layer0_mid_norm_writeback_from_x_work<binary16>(
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::materialize_layer0_mid_norm_writeback_from_x_work<binary32>(
  RefOptimizedStorageBank<binary32>& bank);
template void RefModelOptimized::materialize_layer1_attention_writeback_from_x_work<binary16>(
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::materialize_layer1_attention_writeback_from_x_work<binary32>(
  RefOptimizedStorageBank<binary32>& bank);
template void RefModelOptimized::materialize_layer1_ln_writeback_from_x_work<binary16>(
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::materialize_layer1_ln_writeback_from_x_work<binary32>(
  RefOptimizedStorageBank<binary32>& bank);
template void RefModelOptimized::materialize_layer1_ffn_writeback_from_x_work<binary16>(
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::materialize_layer1_ffn_writeback_from_x_work<binary32>(
  RefOptimizedStorageBank<binary32>& bank);
template void RefModelOptimized::materialize_end_norm_writeback_from_x_work<binary16>(
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::materialize_end_norm_writeback_from_x_work<binary32>(
  RefOptimizedStorageBank<binary32>& bank);
template void RefModelOptimized::materialize_final_head_pass_a_writeback_from_x_work<binary16>(
  RefOptimizedStorageBank<binary16>& bank);
template void RefModelOptimized::materialize_final_head_pass_a_writeback_from_x_work<binary32>(
  RefOptimizedStorageBank<binary32>& bank);

template ac_ieee_float<binary16> RefModelOptimized::float_abs_local<binary16>(
  ac_ieee_float<binary16> x);
template ac_ieee_float<binary32> RefModelOptimized::float_abs_local<binary32>(
  ac_ieee_float<binary32> x);

template void RefModelOptimized::quant_linear_token_32_to32_native<binary16>(
  const ac_ieee_float<binary16> x[D_MODEL],
  const double w[D_MODEL * D_MODEL],
  const double b[D_MODEL],
  float s_x,
  float s_w,
  ac_ieee_float<binary16> y[D_MODEL]);
template void RefModelOptimized::quant_linear_token_32_to32_native<binary32>(
  const ac_ieee_float<binary32> x[D_MODEL],
  const double w[D_MODEL * D_MODEL],
  const double b[D_MODEL],
  float s_x,
  float s_w,
  ac_ieee_float<binary32> y[D_MODEL]);
template void RefModelOptimized::quant_linear_token_32_to128_native<binary16>(
  const ac_ieee_float<binary16> x[D_MODEL],
  const double w[FF_DIM * D_MODEL],
  const double b[FF_DIM],
  float s_x,
  float s_w,
  ac_ieee_float<binary16> y[FF_DIM]);
template void RefModelOptimized::quant_linear_token_32_to128_native<binary32>(
  const ac_ieee_float<binary32> x[D_MODEL],
  const double w[FF_DIM * D_MODEL],
  const double b[FF_DIM],
  float s_x,
  float s_w,
  ac_ieee_float<binary32> y[FF_DIM]);
template void RefModelOptimized::quant_linear_token_128_to32_native<binary16>(
  const ac_ieee_float<binary16> x[FF_DIM],
  const double w[D_MODEL * FF_DIM],
  const double b[D_MODEL],
  float s_x,
  float s_w,
  ac_ieee_float<binary16> y[D_MODEL]);
template void RefModelOptimized::quant_linear_token_128_to32_native<binary32>(
  const ac_ieee_float<binary32> x[FF_DIM],
  const double w[D_MODEL * FF_DIM],
  const double b[D_MODEL],
  float s_x,
  float s_w,
  ac_ieee_float<binary32> y[D_MODEL]);

} // namespace aecct_ref
