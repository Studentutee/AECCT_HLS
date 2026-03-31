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
  const act_t* input_y;
  // input_y_fp32: [B, N] optional exact FP32/F64 bring-up path
  const double* input_y_fp32;
  // output logits: [B, N] as double for convenient comparison
  double* out_logits;
  // output x_pred: [B, N] (0/1)
  bit1_t* out_x_pred;
  // optional FinalHead scalar s_t dump: [B, 75], row-major
  double* out_finalhead_s_t;

  int B;
  int N;
};

struct RefDumpConfig {
  bool enabled;
  const char* dump_dir;
  int pattern_index;
};

struct RefRunConfig {
  RefPrecisionMode precision_mode;
  RefAlgoVariant algo_variant;
  RefLayerNormMode ln_mode = RefLayerNormMode::LN_BASELINE;
  RefFinalHeadExploreStage finalhead_stage = RefFinalHeadExploreStage::S0;
  RefFragGroup frag_group = RefFragGroup::NONE;
};

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
