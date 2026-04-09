#pragma once
#include <cstddef>

#include "RefAlgoVariant.h"
#include "RefFragGroupConfig.h"
#include "RefLayerNormMode.h"
#include "RefPrecisionMode.h"
#include "RefStageConfig.h"
#include "RefTypes.h"

namespace aecct_ref {

struct RefModelIO {
  // Flattened pointers (row-major):
  // input_y: [B, N] (quantized path)
  const act_t* input_y = nullptr;
  // input_y_fp32: [B, N] optional exact FP32/F64 bring-up path
  const double* input_y_fp32 = nullptr;
  // output logits: [B, N] as double for convenient comparison
  double* out_logits = nullptr;
  // output x_pred: [B, N] (0/1)
  bit1_t* out_x_pred = nullptr;
  // optional FinalHead scalar s_t dump: [B, 75], row-major
  double* out_finalhead_s_t = nullptr;
  // optional end_norm dump: [B, 75, 32], row-major
  double* out_end_norm = nullptr;
  // optional layer1_ffn_ln_out dump: [B, 75, 32], row-major
  double* out_layer1_ffn_ln_out = nullptr;
  // optional layer0_ffn_ln_out dump: [B, 75, 32], row-major
  double* out_layer0_ffn_ln_out = nullptr;
  // optional layer0_ffn1_out dump: [B, 75, 128], row-major
  double* out_layer0_ffn1_out = nullptr;
  // optional layer0_relu_out dump: [B, 75, 128], row-major
  double* out_layer0_relu_out = nullptr;
  // optional layer0_ffn2_out dump: [B, 75, 32], row-major
  double* out_layer0_ffn2_out = nullptr;
  // optional layer0 W2 quant-contract raw out dump (pre-residual): [B, 75, 32], row-major
  double* out_layer0_ffn_w2_quant_raw_out = nullptr;
  // optional layer0 W2 raw-quant qx trace: [B, 75, 128], row-major
  double* out_layer0_ffn_w2_quant_raw_qx = nullptr;
  // optional layer0 W2 raw-quant weight*inv_scale trace: [B, 32, 128], row-major
  double* out_layer0_ffn_w2_quant_raw_weight_scaled = nullptr;
  // optional layer0 W2 raw-quant bias-domain trace: [B, 32], row-major
  double* out_layer0_ffn_w2_quant_raw_bias_domain = nullptr;
  // optional layer0 W2 raw-quant partial-acc trace (focus dims 0..2): [B, 75, 3, 129], row-major
  double* out_layer0_ffn_w2_quant_raw_partial_acc_focus = nullptr;
  // optional layer0_sublayer0_attn_input dump: [B, 75, 32], row-major
  double* out_layer0_attn_input = nullptr;
  // optional layer0_attention_post_concat dump: [B, 75, 32], row-major
  double* out_layer0_post_concat = nullptr;
  // optional layer0_sublayer0_attn_out dump: [B, 75, 32], row-major
  double* out_layer0_attn_out = nullptr;
  // optional layer0_sublayer0_pre_ln_input dump: [B, 75, 32], row-major
  double* out_layer0_pre_ln_input = nullptr;
  // optional layer0_ln_out dump: [B, 75, 32], row-major
  double* out_layer0_ln_out = nullptr;
  // optional layer0 residual add sum dump: [B, 75, 32], row-major
  double* out_layer0_residual_add_out = nullptr;
  // optional layer0 sublayer1 LN input dump: [B, 75, 32], row-major
  double* out_layer0_sublayer1_ln_in = nullptr;
  // optional layer1_ffn2_out dump: [B, 75, 32], row-major
  double* out_layer1_ffn2_out = nullptr;
  // optional layer1_sublayer0_attn_out dump: [B, 75, 32], row-major
  double* out_layer1_attn_out = nullptr;
  // optional layer1_attention_post_concat dump: [B, 75, 32], row-major
  double* out_layer1_post_concat = nullptr;
  // optional layer1_q dump: [B, 75, 32], row-major
  double* out_layer1_q = nullptr;
  // optional layer1_attn_input(mid_norm) dump: [B, 75, 32], row-major
  double* out_layer1_attn_input = nullptr;
  // optional layer1_sublayer0_pre_ln_input dump: [B, 75, 32], row-major
  double* out_layer1_pre_ln_input = nullptr;
  // optional layer1_sublayer0_ln_out dump (FFN input): [B, 75, 32], row-major
  double* out_layer1_ln_out = nullptr;
  // optional layer1_ffn1_out dump: [B, 75, 128], row-major
  double* out_layer1_ffn1_out = nullptr;
  // optional layer1_relu_out dump: [B, 75, 128], row-major
  double* out_layer1_relu_out = nullptr;

  int B = 0;
  int N = 0;
};

struct RefDumpConfig {
  bool enabled;
  const char* dump_dir;
  int pattern_index;
};

struct RefRunConfig {
  RefPrecisionMode precision_mode = RefPrecisionMode::BASELINE_FP32;
  RefAlgoVariant algo_variant = RefAlgoVariant::BASELINE_SPEC_FLOW;
  RefLayerNormMode ln_mode = RefLayerNormMode::LN_BASELINE;
  RefFinalHeadExploreStage finalhead_stage = RefFinalHeadExploreStage::S0;
  RefFragGroup frag_group = RefFragGroup::NONE;
};

bool is_fp32_baseline_mode(RefPrecisionMode mode);
bool is_fp16_experiment_mode(RefPrecisionMode mode);
RefRunConfig make_fp32_baseline_run_config();
RefRunConfig make_fp16_experiment_run_config();

class RefModel {
public:
  RefModel();

  void set_run_config(const RefRunConfig& cfg);
  RefRunConfig get_run_config() const;

  void set_dump_config(const RefDumpConfig& cfg);
  void clear_dump_config();

  // Step-0 reference path aligned to algorithm_ref.ipynb.
  void infer_step0(const RefModelIO& io) const;

private:
  RefRunConfig run_cfg_;
  RefDumpConfig dump_cfg_;
};

} // namespace aecct_ref
