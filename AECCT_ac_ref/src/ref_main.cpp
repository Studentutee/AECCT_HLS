#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../include/RefExperimentMetrics.h"
#include "../include/RefE4M3Helpers.h"
#include "../include/RefFullQuantStats.h"
#include "../include/RefModel.h"
#include "../include/RefPrecisionMode.h"

#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"
#include "weights.h"

namespace aecct_ref {

void set_ref_ln_debug_enabled(bool enabled);
void reset_ref_ln_debug_trace();
int get_ref_ln_debug_entry_count();
const char* ref_ln_debug_site_name(int site_id);
bool get_ref_ln_debug_entry(
  int index,
  int* out_entry_seq,
  int* out_site_call_index,
  int* out_sample_index,
  int* out_site,
  int* out_token,
  float* out_sum,
  float* out_sumsq,
  float* out_mean,
  float* out_ex2,
  float* out_mean_sq,
  float* out_var_raw,
  float* out_var_final,
  float* out_var_from_residual,
  float* out_var_from_ex2,
  float* out_eps_applied,
  float* out_inv_std_input,
  float* out_sum_seq,
  float* out_sumsq_seq,
  float* out_sum_tile,
  float* out_sumsq_tile,
  float* out_x_eps,
  float* out_inv_std_used,
  float* out_inv_std_true,
  float* out_inv_std_seed,
  float* out_inv_std_nr1,
  int* out_var_negative_before_clamp,
  int* out_clamp_triggered,
  int* out_eps_dominates_var,
  int* out_sanitize_input_count,
  float* out_x,
  int out_x_len,
  float* out_y,
  int out_y_len
);

void set_ref_ln_upstream_debug_enabled(bool enabled);
void reset_ref_ln_upstream_debug_trace();
int get_ref_ln_upstream_debug_entry_count();
const char* ref_ln_upstream_boundary_name(int boundary_id);
bool get_ref_ln_upstream_debug_entry(
  int index,
  int* out_entry_seq,
  int* out_sample_index,
  int* out_layer_idx,
  int* out_boundary,
  int* out_boundary_call_index,
  int* out_token,
  float* out_x,
  int out_x_len
);

} // namespace aecct_ref

namespace {

enum class CliRunMode : unsigned char {
  COMPARE = 0,
  BASELINE_ONLY = 1,
  EXPERIMENT_ONLY = 2,
  EVAL_BASELINE = 3,
  EVAL_EXPERIMENT = 4,
  EVAL_COMPARE = 5,
  EXPLORE = 6
};

enum class CliParseResult : unsigned char {
  OK = 0,
  HELP = 1,
  ERROR = 2
};

struct CliOptions {
  CliRunMode run_mode;
  int pattern_index;
  int pattern_begin;
  int pattern_count;
  int topk;
  bool summary_only;
  bool ln_debug;
  bool ln_var_debug;
  bool ln_input_debug;
  bool ln_upstream_debug;
  bool xpred_focused_inspect;
  bool io16_selfcheck;
  aecct_ref::RefStep0OutputMode io16_output_mode;
  std::string summary_csv_path;
  aecct_ref::RefAlgoVariant algo_variant;
  aecct_ref::RefSoftmaxExpMode softmax_exp_mode;
  aecct_ref::RefLayerNormMode ln_mode;
  aecct_ref::RefLayerNormMode experiment_ln_mode;
  aecct_ref::RefFinalHeadExploreStage finalhead_stage;
  aecct_ref::RefPrecisionMode experiment_precision_mode;
  bool experiment_precision_explicit;
  aecct_ref::RefFragGroup frag_group;
};

struct PatternRange {
  int begin;
  int count;
};

struct RefRunOutputs {
  std::vector<double> logits;
  std::vector<aecct_ref::bit1_t> x_pred;
  aecct_ref::Metrics logits_vs_golden;
  std::size_t x_pred_match_count;
  std::size_t x_pred_total_count;
};

struct PerPatternCompareRow {
  int pattern_index;
  aecct_ref::RefExperimentCompareMetrics cmp;
  aecct_ref::Metrics baseline_vs_golden;
  aecct_ref::Metrics experiment_vs_golden;
};

struct BatchCompareSummary {
  int total_patterns_scanned;
  std::size_t patterns_with_xpred_flip;
  std::size_t patterns_with_sign_flip;
  std::size_t total_flipped_bits;
  double worst_logits_mse;
  int worst_logits_mse_pattern;
  double worst_max_abs_diff;
  int worst_max_abs_diff_pattern;
  double worst_min_margin;
  int worst_min_margin_pattern;
  std::size_t max_sign_flip_count;
  int max_sign_flip_pattern;
  std::size_t total_sign_flip_count;
  std::vector<double> per_pattern_min_margin;
  aecct_ref::RefNonFiniteCounters baseline_nonfinite_total;
  aecct_ref::RefNonFiniteCounters experiment_nonfinite_total;
};

struct DistributionStats {
  double min_v;
  double max_v;
  double mean_v;
  double median_v;
};

struct GoldenAggregateMetrics {
  double se_sum;
  double ae_sum;
  double max_abs;
  std::size_t logits_count;
  std::size_t x_pred_match_count;
  std::size_t x_pred_total_count;
};

struct PerfBreakdownSec {
  double startup_init_s;
  double baseline_model_s;
  double experiment_path_s;
  double compare_aggregation_s;
  double file_io_s;
  double total_s;
};

struct EvalPatternRow {
  int pattern_index;
  std::size_t bit_errors;
  std::size_t evaluated_bits;
  std::size_t x_pred_match_count;
  int frame_error_flag;
};

struct EvalComparePatternRow {
  int pattern_index;
  std::size_t evaluated_bits;
  std::size_t baseline_bit_errors;
  std::size_t experiment_bit_errors;
  std::size_t baseline_x_pred_match_count;
  std::size_t experiment_x_pred_match_count;
  int baseline_frame_error_flag;
  int experiment_frame_error_flag;
};

struct EvalAggregateStats {
  int total_patterns;
  std::size_t total_bits;
  std::size_t total_bit_errors;
  std::size_t frame_error_count;
  std::size_t x_pred_match_count;
  double ber;
  double fer;
  double x_pred_match_ratio;
};

static constexpr int STAGE_TOKENS = 75;
static constexpr int STAGE_D_MODEL = 32;

struct StageTensorDiffSummary {
  const char* name = "";
  std::size_t total_count = 0;
  std::size_t mismatch_count = 0;
  double max_abs_diff = 0.0;
  bool exact_equal = true;
  bool has_first_mismatch = false;
  int first_pattern = -1;
  int first_token = -1;
  int first_dim = -1;
  double first_baseline_value = 0.0;
  double first_experiment_value = 0.0;
};

struct StageBitDiffSummary {
  const char* name = "";
  std::size_t total_count = 0;
  std::size_t mismatch_count = 0;
  bool exact_equal = true;
  bool has_first_mismatch = false;
  int first_pattern = -1;
  int first_index = -1;
  int first_baseline_value = 0;
  int first_experiment_value = 0;
};

static constexpr int LN_DEBUG_D_MODEL = 32;

struct LnDebugEntryRow {
  int entry_seq = 0;
  int site_call_index = 0;
  int sample_index = 0;
  int site = 0;
  int token = 0;
  float sum = 0.0f;
  float sumsq = 0.0f;
  float mean = 0.0f;
  float ex2 = 0.0f;
  float mean_sq = 0.0f;
  float var_raw = 0.0f;
  float var_final = 0.0f;
  float var_from_residual = 0.0f;
  float var_from_ex2 = 0.0f;
  float eps_applied = 0.0f;
  float inv_std_input = 0.0f;
  float sum_seq = 0.0f;
  float sumsq_seq = 0.0f;
  float sum_tile = 0.0f;
  float sumsq_tile = 0.0f;
  float x_eps = 0.0f;
  float inv_std_used = 0.0f;
  float inv_std_true = 0.0f;
  float inv_std_seed = 0.0f;
  float inv_std_nr1 = 0.0f;
  int var_negative_before_clamp = 0;
  int clamp_triggered = 0;
  int eps_dominates_var = 0;
  int sanitize_input_count = 0;
  float x[LN_DEBUG_D_MODEL]{};
  float y[LN_DEBUG_D_MODEL]{};
};

struct LnUpstreamEntryRow {
  int entry_seq = 0;
  int sample_index = 0;
  int layer_idx = 0;
  int boundary = 0;
  int boundary_call_index = 0;
  int token = 0;
  float x[LN_DEBUG_D_MODEL]{};
};

struct LnUpstreamBoundaryDiag {
  int pattern = -1;
  int token = -1;
  int boundary = -1;
  int entry_seq_base = -1;
  int entry_seq_exp = -1;
  int call_index_base = -1;
  int call_index_exp = -1;
  float max_abs_diff = 0.0f;
  float mean_abs_diff = 0.0f;
  float mse = 0.0f;
  float l2 = 0.0f;
};

enum class LnVarFirstDivergenceStage : int {
  NONE = 0,
  SUM_OR_SUMSQ = 1,
  MOMENT_DERIVED = 2,
  VAR_FORMULA = 3,
  POST_VAR_HANDLING = 4
};

struct LnVarTokenDiag {
  int pattern = -1;
  int site = -1;
  int token = -1;
  float base_sum = 0.0f;
  float exp_sum = 0.0f;
  float base_sumsq = 0.0f;
  float exp_sumsq = 0.0f;
  float base_mean = 0.0f;
  float exp_mean = 0.0f;
  float base_ex2 = 0.0f;
  float exp_ex2 = 0.0f;
  float base_mean_sq = 0.0f;
  float exp_mean_sq = 0.0f;
  float base_var_raw = 0.0f;
  float exp_var_raw = 0.0f;
  float base_var_final = 0.0f;
  float exp_var_final = 0.0f;
  float base_eps_applied = 0.0f;
  float exp_eps_applied = 0.0f;
  float base_inv_std_input = 0.0f;
  float exp_inv_std_input = 0.0f;
  float base_var_from_residual = 0.0f;
  float exp_var_from_residual = 0.0f;
  float base_var_from_ex2 = 0.0f;
  float exp_var_from_ex2 = 0.0f;
  float base_sum_seq = 0.0f;
  float exp_sum_seq = 0.0f;
  float base_sum_tile = 0.0f;
  float exp_sum_tile = 0.0f;
  float base_sumsq_seq = 0.0f;
  float exp_sumsq_seq = 0.0f;
  float base_sumsq_tile = 0.0f;
  float exp_sumsq_tile = 0.0f;
  float y_max_abs_diff = 0.0f;
  float y_mean_abs_diff = 0.0f;
  float var_raw_abs_diff = 0.0f;
  float var_final_abs_diff = 0.0f;
  LnVarFirstDivergenceStage first_stage = LnVarFirstDivergenceStage::NONE;
};

struct LnInputTokenDiag {
  int pattern = -1;
  int site = -1;
  int token = -1;
  int entry_seq_base = -1;
  int entry_seq_exp = -1;
  int site_call_index_base = -1;
  int site_call_index_exp = -1;
  float input_max_abs_diff = 0.0f;
  float input_mean_abs_diff = 0.0f;
  float input_mse = 0.0f;
  float input_l2 = 0.0f;
  float var_raw_abs_diff = 0.0f;
  float var_final_abs_diff = 0.0f;
  float base_probe_mean = 0.0f;
  float base_probe_var_residual = 0.0f;
  float base_probe_var_ex2 = 0.0f;
  float base_probe_var_gap_abs = 0.0f;
  float base_probe_var_gap_rel = 0.0f;
  float exp_probe_mean = 0.0f;
  float exp_probe_var_residual = 0.0f;
  float exp_probe_var_ex2 = 0.0f;
  float exp_probe_var_gap_abs = 0.0f;
  float exp_probe_var_gap_rel = 0.0f;
};

struct LnDebugTokenDiag {
  int pattern = -1;
  int site = -1;
  int token = -1;
  float mean_base = 0.0f;
  float mean_exp = 0.0f;
  float var_base = 0.0f;
  float var_exp = 0.0f;
  float var_abs_diff = 0.0f;
  float x_eps = 0.0f;
  float inv_std_base_true = 0.0f;
  float inv_std_exp_true = 0.0f;
  float inv_std_exp_nr1 = 0.0f;
  float inv_std_seed = 0.0f;
  float inv_std_cross_abs_diff = 0.0f;
  float inv_std_cross_rel_diff = 0.0f;
  float seed_abs_err = 0.0f;
  float seed_rel_err = 0.0f;
  float nr_abs_err = 0.0f;
  float nr_rel_err = 0.0f;
  float y_max_abs_diff = 0.0f;
  float y_mean_abs_diff = 0.0f;
  int var_negative_before_clamp = 0;
  int clamp_triggered = 0;
  int eps_dominates_var = 0;
  int sanitize_input_count = 0;
};

static inline std::chrono::steady_clock::time_point now_tp() {
  return std::chrono::steady_clock::now();
}

static inline double elapsed_sec(
  const std::chrono::steady_clock::time_point& t0,
  const std::chrono::steady_clock::time_point& t1
) {
  return std::chrono::duration<double>(t1 - t0).count();
}

static const char* run_mode_to_string(CliRunMode mode) {
  switch (mode) {
    case CliRunMode::COMPARE: return "compare";
    case CliRunMode::BASELINE_ONLY: return "baseline";
    case CliRunMode::EXPERIMENT_ONLY: return "experiment";
    case CliRunMode::EVAL_BASELINE: return "eval-baseline";
    case CliRunMode::EVAL_EXPERIMENT: return "eval-experiment";
    case CliRunMode::EVAL_COMPARE: return "eval-compare";
    case CliRunMode::EXPLORE: return "explore";
    default: return "unknown";
  }
}

static bool run_mode_uses_experiment_config(CliRunMode mode) {
  return mode == CliRunMode::EXPERIMENT_ONLY ||
         mode == CliRunMode::COMPARE ||
         mode == CliRunMode::EVAL_EXPERIMENT ||
         mode == CliRunMode::EVAL_COMPARE ||
         mode == CliRunMode::EXPLORE;
}

static bool run_mode_has_dual_path(CliRunMode mode) {
  return mode == CliRunMode::COMPARE ||
         mode == CliRunMode::EVAL_COMPARE ||
         mode == CliRunMode::EXPLORE;
}

static void print_usage() {
  std::printf("Usage: ref_sim [pattern_index] [options]\n");
  std::printf("Options:\n");
  std::printf("  --mode compare|baseline|experiment|eval-baseline|eval-experiment|eval-compare|explore\n");
  std::printf("  --pattern N\n");
  std::printf("  --pattern-begin N --pattern-count M\n");
  std::printf("  --topk K\n");
  std::printf("  --stage S0|S1|S2|S3|S4\n");
  std::printf("  --precision-exp MODE\n");
  std::printf("      baseline_fp32 = baseline (default)\n");
  std::printf("      generic_e4m3_* / full_e4m3_* / int8_fixedexp_* / fp16_replace_fp32_global = experiment-only\n");
  std::printf("      values: baseline_fp32|generic_e4m3_finalhead|full_e4m3_nonlinear_stress|generic_e4m3_frag_bisect|generic_e4m3_except_g5|generic_e4m3_g5_g4|generic_e4m3_g5_g1|generic_e4m3_g5_g3|generic_e4m3_g5_g2|generic_e4m3_g2_embed_only|generic_e4m3_g2_spe_only|generic_e4m3_g2_preproc_assembly|generic_e4m3_g2_prelayer_handoff|int8_fixedexp_zone3_embed_g2|int8_fixedexp_zone4_embed_g2|fp16_replace_fp32_global\n");
  std::printf("  --frag-group NONE|G1|G2|G3|G4|G5|C1|C2|C3|C4\n");
  std::printf("  --summary-only\n");
  std::printf("  --quiet (alias of --summary-only)\n");
  std::printf("  --ln-debug\n");
  std::printf("  --ln-var-debug\n");
  std::printf("  --ln-input-debug\n");
  std::printf("  --ln-upstream-debug\n");
  std::printf("  --xpred-focused-inspect (debug-only: print x_pred one-bit positions for single-pattern run)\n");
  std::printf("  --io16-selfcheck (single-pattern only: build ref io16 image + readback/unpack selfcheck)\n");
  std::printf("  --io16-output xpred|logits (default: xpred)\n");
  std::printf("  --summary-csv PATH\n");
  std::printf("  --algo baseline_spec_flow|reserved_softmax_alt|reserved_finalhead_alt\n");
  std::printf("  --softmax-exp-mode baseline_nearest_lut|v2_lerp_lut|v3_base2_reserved\n");
  std::printf("      baseline_nearest_lut = current baseline (default)\n");
  std::printf("      v2_lerp_lut = softmax_v2 candidate 2\n");
  std::printf("      v3_base2_reserved = reserved, not used in this task\n");
  std::printf("  --ln-mode ln_baseline|ln_sum_sumsq_approx\n");
  std::printf("  --ln-mode-exp ln_baseline|ln_sum_sumsq_approx\n");
  std::printf("  --help\n");
  std::printf("Examples:\n");
  std::printf("  ref_sim --mode baseline --pattern 0\n");
  std::printf("  ref_sim --mode experiment --precision-exp fp16_replace_fp32_global --pattern 0\n");
  std::printf("  ref_sim --mode compare --precision-exp fp16_replace_fp32_global --pattern-begin 0 --pattern-count 32\n");
}

static bool parse_run_mode(const char* text, CliRunMode& mode) {
  if (std::strcmp(text, "compare") == 0) {
    mode = CliRunMode::COMPARE;
    return true;
  }
  if (std::strcmp(text, "baseline") == 0) {
    mode = CliRunMode::BASELINE_ONLY;
    return true;
  }
  if (std::strcmp(text, "experiment") == 0) {
    mode = CliRunMode::EXPERIMENT_ONLY;
    return true;
  }
  if (std::strcmp(text, "eval-baseline") == 0) {
    mode = CliRunMode::EVAL_BASELINE;
    return true;
  }
  if (std::strcmp(text, "eval-experiment") == 0) {
    mode = CliRunMode::EVAL_EXPERIMENT;
    return true;
  }
  if (std::strcmp(text, "eval-compare") == 0) {
    mode = CliRunMode::EVAL_COMPARE;
    return true;
  }
  if (std::strcmp(text, "explore") == 0) {
    mode = CliRunMode::EXPLORE;
    return true;
  }
  return false;
}

static bool parse_algo_variant(const char* text, aecct_ref::RefAlgoVariant& variant) {
  if (std::strcmp(text, "baseline_spec_flow") == 0) {
    variant = aecct_ref::RefAlgoVariant::BASELINE_SPEC_FLOW;
    return true;
  }
  if (std::strcmp(text, "reserved_softmax_alt") == 0) {
    variant = aecct_ref::RefAlgoVariant::RESERVED_SOFTMAX_ALT;
    return true;
  }
  if (std::strcmp(text, "reserved_finalhead_alt") == 0) {
    variant = aecct_ref::RefAlgoVariant::RESERVED_FINALHEAD_ALT;
    return true;
  }
  return false;
}

static bool parse_softmax_exp_mode(const char* text, aecct_ref::RefSoftmaxExpMode& mode) {
  if (std::strcmp(text, "baseline_nearest_lut") == 0) {
    mode = aecct_ref::RefSoftmaxExpMode::BASELINE_NEAREST_LUT;
    return true;
  }
  if (std::strcmp(text, "v2_lerp_lut") == 0) {
    mode = aecct_ref::RefSoftmaxExpMode::V2_LERP_LUT;
    return true;
  }
  if (std::strcmp(text, "v3_base2_reserved") == 0) {
    mode = aecct_ref::RefSoftmaxExpMode::V3_BASE2_RESERVED;
    return true;
  }
  return false;
}

static bool parse_ln_mode(const char* text, aecct_ref::RefLayerNormMode& mode) {
  if (std::strcmp(text, "ln_baseline") == 0 ||
      std::strcmp(text, "baseline") == 0 ||
      std::strcmp(text, "LN_BASELINE") == 0) {
    mode = aecct_ref::RefLayerNormMode::LN_BASELINE;
    return true;
  }
  if (std::strcmp(text, "ln_sum_sumsq_approx") == 0 ||
      std::strcmp(text, "sum_sumsq_approx") == 0 ||
      std::strcmp(text, "LN_SUM_SUMSQ_APPROX") == 0) {
    mode = aecct_ref::RefLayerNormMode::LN_SUM_SUMSQ_APPROX;
    return true;
  }
  return false;
}

static bool parse_finalhead_stage(const char* text, aecct_ref::RefFinalHeadExploreStage& stage) {
  if (std::strcmp(text, "S0") == 0 || std::strcmp(text, "s0") == 0) {
    stage = aecct_ref::RefFinalHeadExploreStage::S0;
    return true;
  }
  if (std::strcmp(text, "S1") == 0 || std::strcmp(text, "s1") == 0) {
    stage = aecct_ref::RefFinalHeadExploreStage::S1;
    return true;
  }
  if (std::strcmp(text, "S2") == 0 || std::strcmp(text, "s2") == 0) {
    stage = aecct_ref::RefFinalHeadExploreStage::S2;
    return true;
  }
  if (std::strcmp(text, "S3") == 0 || std::strcmp(text, "s3") == 0) {
    stage = aecct_ref::RefFinalHeadExploreStage::S3;
    return true;
  }
  if (std::strcmp(text, "S4") == 0 || std::strcmp(text, "s4") == 0) {
    stage = aecct_ref::RefFinalHeadExploreStage::S4;
    return true;
  }
  return false;
}

static bool parse_precision_mode(const char* text, aecct_ref::RefPrecisionMode& mode) {
  if (std::strcmp(text, "baseline_fp32") == 0 || std::strcmp(text, "fp32") == 0) {
    mode = aecct_ref::RefPrecisionMode::BASELINE_FP32;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_finalhead") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD;
    return true;
  }
  if (std::strcmp(text, "full_e4m3_nonlinear_stress") == 0) {
    mode = aecct_ref::RefPrecisionMode::FULL_E4M3_NONLINEAR_STRESS;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_frag_bisect") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_FRAG_BISECT;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_except_g5") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_EXCEPT_G5;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_g5_g4") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G4;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_g5_g1") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G1;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_g5_g3") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G3;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_g5_g2") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G2;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_g2_embed_only") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_EMBED_ONLY;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_g2_spe_only") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_SPE_ONLY;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_g2_preproc_assembly") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_PREPROC_ASSEMBLY;
    return true;
  }
  if (std::strcmp(text, "generic_e4m3_g2_prelayer_handoff") == 0) {
    mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_PRELAYER_HANDOFF;
    return true;
  }
  if (std::strcmp(text, "int8_fixedexp_zone3_embed_g2") == 0) {
    mode = aecct_ref::RefPrecisionMode::INT8_FIXEDEXP_ZONE3_EMBED_G2;
    return true;
  }
  if (std::strcmp(text, "int8_fixedexp_zone4_embed_g2") == 0) {
    mode = aecct_ref::RefPrecisionMode::INT8_FIXEDEXP_ZONE4_EMBED_G2;
    return true;
  }
  if (std::strcmp(text, "fp16_replace_fp32_global") == 0 ||
      std::strcmp(text, "fp16_experiment") == 0 ||
      std::strcmp(text, "fp16") == 0) {
    mode = aecct_ref::RefPrecisionMode::FP16_REPLACE_FP32_GLOBAL;
    return true;
  }
  return false;
}

static bool parse_frag_group(const char* text, aecct_ref::RefFragGroup& g) {
  if (std::strcmp(text, "NONE") == 0 || std::strcmp(text, "none") == 0) {
    g = aecct_ref::RefFragGroup::NONE;
    return true;
  }
  if (std::strcmp(text, "G1") == 0 || std::strcmp(text, "g1") == 0) {
    g = aecct_ref::RefFragGroup::G1_LAYERNORM;
    return true;
  }
  if (std::strcmp(text, "G2") == 0 || std::strcmp(text, "g2") == 0) {
    g = aecct_ref::RefFragGroup::G2_RESIDUAL;
    return true;
  }
  if (std::strcmp(text, "G3") == 0 || std::strcmp(text, "g3") == 0) {
    g = aecct_ref::RefFragGroup::G3_ATTN_CONTEXT;
    return true;
  }
  if (std::strcmp(text, "G4") == 0 || std::strcmp(text, "g4") == 0) {
    g = aecct_ref::RefFragGroup::G4_SOFTMAX_NEIGHBORHOOD;
    return true;
  }
  if (std::strcmp(text, "G5") == 0 || std::strcmp(text, "g5") == 0) {
    g = aecct_ref::RefFragGroup::G5_PREPROC_EMBED;
    return true;
  }
  if (std::strcmp(text, "C1") == 0 || std::strcmp(text, "c1") == 0) {
    g = aecct_ref::RefFragGroup::C1_G1_G2;
    return true;
  }
  if (std::strcmp(text, "C2") == 0 || std::strcmp(text, "c2") == 0) {
    g = aecct_ref::RefFragGroup::C2_G1_G3;
    return true;
  }
  if (std::strcmp(text, "C3") == 0 || std::strcmp(text, "c3") == 0) {
    g = aecct_ref::RefFragGroup::C3_G2_G3;
    return true;
  }
  if (std::strcmp(text, "C4") == 0 || std::strcmp(text, "c4") == 0) {
    g = aecct_ref::RefFragGroup::C4_G1_G4;
    return true;
  }
  return false;
}

static bool parse_io16_output_mode(const char* text, aecct_ref::RefStep0OutputMode& mode) {
  if (std::strcmp(text, "xpred") == 0 || std::strcmp(text, "x_pred") == 0) {
    mode = aecct_ref::RefStep0OutputMode::X_PRED;
    return true;
  }
  if (std::strcmp(text, "logits") == 0) {
    mode = aecct_ref::RefStep0OutputMode::LOGITS;
    return true;
  }
  return false;
}

static CliParseResult parse_cli(int argc, char** argv, CliOptions& opts) {
  opts.run_mode = CliRunMode::COMPARE;
  opts.pattern_index = -1;
  opts.pattern_begin = -1;
  opts.pattern_count = -1;
  opts.topk = 5;
  opts.summary_only = false;
  opts.ln_debug = false;
  opts.ln_var_debug = false;
  opts.ln_input_debug = false;
  opts.ln_upstream_debug = false;
  opts.xpred_focused_inspect = false;
  opts.io16_selfcheck = false;
  opts.io16_output_mode = aecct_ref::RefStep0OutputMode::X_PRED;
  opts.summary_csv_path.clear();
  opts.algo_variant = aecct_ref::RefAlgoVariant::BASELINE_SPEC_FLOW;
  opts.softmax_exp_mode = aecct_ref::RefSoftmaxExpMode::BASELINE_NEAREST_LUT;
  opts.ln_mode = aecct_ref::RefLayerNormMode::LN_BASELINE;
  opts.experiment_ln_mode = aecct_ref::RefLayerNormMode::LN_BASELINE;
  opts.finalhead_stage = aecct_ref::RefFinalHeadExploreStage::S0;
  opts.experiment_precision_mode = aecct_ref::RefPrecisionMode::BASELINE_FP32;
  opts.experiment_precision_explicit = false;
  opts.frag_group = aecct_ref::RefFragGroup::NONE;

  bool positional_pattern_used = false;
  bool ln_mode_exp_set = false;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      return CliParseResult::HELP;
    }
    if (std::strcmp(arg, "--mode") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --mode\n");
        return CliParseResult::ERROR;
      }
      if (!parse_run_mode(argv[++i], opts.run_mode)) {
        std::printf("Unsupported mode: %s\n", argv[i]);
        return CliParseResult::ERROR;
      }
      continue;
    }
    if (std::strcmp(arg, "--pattern") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --pattern\n");
        return CliParseResult::ERROR;
      }
      opts.pattern_index = std::atoi(argv[++i]);
      positional_pattern_used = true;
      continue;
    }
    if (std::strcmp(arg, "--pattern-begin") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --pattern-begin\n");
        return CliParseResult::ERROR;
      }
      opts.pattern_begin = std::atoi(argv[++i]);
      continue;
    }
    if (std::strcmp(arg, "--pattern-count") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --pattern-count\n");
        return CliParseResult::ERROR;
      }
      opts.pattern_count = std::atoi(argv[++i]);
      continue;
    }
    if (std::strcmp(arg, "--topk") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --topk\n");
        return CliParseResult::ERROR;
      }
      opts.topk = std::atoi(argv[++i]);
      continue;
    }
    if (std::strcmp(arg, "--stage") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --stage\n");
        return CliParseResult::ERROR;
      }
      if (!parse_finalhead_stage(argv[++i], opts.finalhead_stage)) {
        std::printf("Unsupported stage: %s\n", argv[i]);
        return CliParseResult::ERROR;
      }
      continue;
    }
    if (std::strcmp(arg, "--summary-only") == 0 || std::strcmp(arg, "--quiet") == 0) {
      opts.summary_only = true;
      continue;
    }
    if (std::strcmp(arg, "--ln-debug") == 0) {
      opts.ln_debug = true;
      continue;
    }
    if (std::strcmp(arg, "--ln-var-debug") == 0) {
      opts.ln_var_debug = true;
      continue;
    }
    if (std::strcmp(arg, "--ln-input-debug") == 0) {
      opts.ln_input_debug = true;
      continue;
    }
    if (std::strcmp(arg, "--ln-upstream-debug") == 0) {
      opts.ln_upstream_debug = true;
      continue;
    }
    if (std::strcmp(arg, "--xpred-focused-inspect") == 0) {
      opts.xpred_focused_inspect = true;
      continue;
    }
    if (std::strcmp(arg, "--io16-selfcheck") == 0) {
      opts.io16_selfcheck = true;
      continue;
    }
    if (std::strcmp(arg, "--io16-output") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --io16-output\n");
        return CliParseResult::ERROR;
      }
      if (!parse_io16_output_mode(argv[++i], opts.io16_output_mode)) {
        std::printf("Unsupported --io16-output value: %s\n", argv[i]);
        return CliParseResult::ERROR;
      }
      continue;
    }
    if (std::strcmp(arg, "--frag-group") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --frag-group\n");
        return CliParseResult::ERROR;
      }
      if (!parse_frag_group(argv[++i], opts.frag_group)) {
        std::printf("Unsupported --frag-group value: %s\n", argv[i]);
        return CliParseResult::ERROR;
      }
      continue;
    }
    if (std::strcmp(arg, "--precision-exp") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --precision-exp\n");
        return CliParseResult::ERROR;
      }
      if (!parse_precision_mode(argv[++i], opts.experiment_precision_mode)) {
        std::printf("Unsupported --precision-exp value: %s\n", argv[i]);
        return CliParseResult::ERROR;
      }
      opts.experiment_precision_explicit = true;
      continue;
    }
    if (std::strcmp(arg, "--summary-csv") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --summary-csv\n");
        return CliParseResult::ERROR;
      }
      opts.summary_csv_path = argv[++i];
      continue;
    }
    if (std::strcmp(arg, "--algo") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --algo\n");
        return CliParseResult::ERROR;
      }
      if (!parse_algo_variant(argv[++i], opts.algo_variant)) {
        std::printf("Unsupported algo variant: %s\n", argv[i]);
        return CliParseResult::ERROR;
      }
      continue;
    }
    if (std::strcmp(arg, "--softmax-exp-mode") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --softmax-exp-mode\n");
        return CliParseResult::ERROR;
      }
      if (!parse_softmax_exp_mode(argv[++i], opts.softmax_exp_mode)) {
        std::printf("Unsupported --softmax-exp-mode value: %s\n", argv[i]);
        return CliParseResult::ERROR;
      }
      continue;
    }
    if (std::strcmp(arg, "--ln-mode") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --ln-mode\n");
        return CliParseResult::ERROR;
      }
      if (!parse_ln_mode(argv[++i], opts.ln_mode)) {
        std::printf("Unsupported LN mode: %s\n", argv[i]);
        return CliParseResult::ERROR;
      }
      continue;
    }
    if (std::strcmp(arg, "--ln-mode-exp") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --ln-mode-exp\n");
        return CliParseResult::ERROR;
      }
      if (!parse_ln_mode(argv[++i], opts.experiment_ln_mode)) {
        std::printf("Unsupported experimental LN mode: %s\n", argv[i]);
        return CliParseResult::ERROR;
      }
      ln_mode_exp_set = true;
      continue;
    }
    if (arg[0] == '-') {
      std::printf("Unknown flag: %s\n", arg);
      return CliParseResult::ERROR;
    }
    if (positional_pattern_used) {
      std::printf("Unexpected positional arg: %s\n", arg);
      return CliParseResult::ERROR;
    }
    opts.pattern_index = std::atoi(arg);
    positional_pattern_used = true;
  }

  if (opts.topk <= 0) {
    std::printf("--topk must be > 0\n");
    return CliParseResult::ERROR;
  }
  if (!ln_mode_exp_set) {
    opts.experiment_ln_mode = opts.ln_mode;
  }
  return CliParseResult::OK;
}

static bool resolve_pattern_range(const CliOptions& opts, int total_patterns, PatternRange& range) {
  if (opts.pattern_index >= 0) {
    if (opts.pattern_begin >= 0 || opts.pattern_count >= 0) {
      std::printf("Do not mix --pattern with --pattern-begin/--pattern-count\n");
      return false;
    }
    if (opts.pattern_index >= total_patterns) {
      std::printf("pattern_index out of range: %d (valid [0, %d))\n",
        opts.pattern_index, total_patterns);
      return false;
    }
    range.begin = opts.pattern_index;
    range.count = 1;
    return true;
  }

  if (opts.pattern_begin >= 0 || opts.pattern_count >= 0) {
    const int begin = (opts.pattern_begin >= 0) ? opts.pattern_begin : 0;
    const int count = (opts.pattern_count >= 0) ? opts.pattern_count : (total_patterns - begin);
    if (begin < 0 || begin >= total_patterns) {
      std::printf("pattern-begin out of range: %d (valid [0, %d))\n", begin, total_patterns);
      return false;
    }
    if (count <= 0) {
      std::printf("pattern-count must be > 0\n");
      return false;
    }
    if (begin + count > total_patterns) {
      std::printf("pattern range overflow: begin=%d count=%d total=%d\n",
        begin, count, total_patterns);
      return false;
    }
    range.begin = begin;
    range.count = count;
    return true;
  }

  range.begin = 0;
  range.count = total_patterns;
  return true;
}

static inline bool precision_mode_anchors_to_finalhead_s0(aecct_ref::RefPrecisionMode mode) {
  return mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_FRAG_BISECT ||
         mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_EXCEPT_G5 ||
         mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G4 ||
         mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G1 ||
         mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G3 ||
         mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G2 ||
         mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_EMBED_ONLY ||
         mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_SPE_ONLY ||
         mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_PREPROC_ASSEMBLY ||
         mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_PRELAYER_HANDOFF ||
         mode == aecct_ref::RefPrecisionMode::INT8_FIXEDEXP_ZONE3_EMBED_G2 ||
         mode == aecct_ref::RefPrecisionMode::INT8_FIXEDEXP_ZONE4_EMBED_G2;
}

static inline bool precision_mode_requires_frag_group(aecct_ref::RefPrecisionMode mode) {
  return mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_FRAG_BISECT;
}

static std::string precision_mode_output_suffix(
  aecct_ref::RefPrecisionMode mode,
  aecct_ref::RefFragGroup group
) {
  if (mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_FRAG_BISECT) {
    return "_frag_" + std::string(aecct_ref::to_string(group));
  }
  if (mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_EXCEPT_G5) {
    return "_except_g5";
  }
  if (mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G4) {
    return "_g5_g4";
  }
  if (mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G1) {
    return "_g5_g1";
  }
  if (mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G3) {
    return "_g5_g3";
  }
  if (mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G5_G2) {
    return "_g5_g2";
  }
  if (mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_EMBED_ONLY) {
    return "_g2_embed_only";
  }
  if (mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_SPE_ONLY) {
    return "_g2_spe_only";
  }
  if (mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_PREPROC_ASSEMBLY) {
    return "_g2_preproc_assembly";
  }
  if (mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_PRELAYER_HANDOFF) {
    return "_g2_prelayer_handoff";
  }
  if (mode == aecct_ref::RefPrecisionMode::INT8_FIXEDEXP_ZONE3_EMBED_G2) {
    return "_int8fx_z3_embed_g2";
  }
  if (mode == aecct_ref::RefPrecisionMode::INT8_FIXEDEXP_ZONE4_EMBED_G2) {
    return "_int8fx_z4_embed_g2";
  }
  if (mode == aecct_ref::RefPrecisionMode::FP16_REPLACE_FP32_GLOBAL) {
    return "_fp16_replace_fp32_global";
  }
  return std::string();
}

static std::size_t compute_x_pred_match_count(
  const double* golden_x_pred,
  const std::vector<aecct_ref::bit1_t>& predicted_x_pred
) {
  std::size_t match = 0;
  for (std::size_t i = 0; i < predicted_x_pred.size(); ++i) {
    const int g = (golden_x_pred[i] != 0.0) ? 1 : 0;
    const int p = predicted_x_pred[i].to_int();
    if (g == p) {
      match++;
    }
  }
  return match;
}

static void run_ref_single_pattern(
  aecct_ref::RefModel& model,
  int pattern_index,
  int n_vars,
  RefRunOutputs& out
) {
  const double* input_ptr = &trace_input_y_step0_tensor[pattern_index * n_vars];
  const double* golden_logits = &trace_output_logits_step0_tensor[pattern_index * n_vars];
  const double* golden_x_pred = &trace_output_x_pred_step0_tensor[pattern_index * n_vars];

  out.logits.assign(static_cast<std::size_t>(n_vars), 0.0);
  out.x_pred.assign(static_cast<std::size_t>(n_vars), aecct_ref::bit1_t(0));

  aecct_ref::RefModelIO io{};
  io.input_y = nullptr;
  io.input_y_fp32 = input_ptr;
  io.out_logits = out.logits.data();
  io.out_x_pred = out.x_pred.data();
  io.out_finalhead_s_t = nullptr;
  io.B = 1;
  io.N = n_vars;
  model.infer_step0(io);

  out.logits_vs_golden = aecct_ref::compute_metrics(golden_logits, out.logits.data(), out.logits.size());
  out.x_pred_match_count = compute_x_pred_match_count(golden_x_pred, out.x_pred);
  out.x_pred_total_count = out.x_pred.size();
}

static void run_ref_batch(
  aecct_ref::RefModel& model,
  const PatternRange& range,
  int n_vars,
  std::vector<double>& logits,
  std::vector<aecct_ref::bit1_t>& x_pred,
  double* out_finalhead_s_t,
  double* out_layer1_attn_input = nullptr,
  double* out_end_norm = nullptr
) {
  const int run_b = range.count;
  const std::size_t logits_count = static_cast<std::size_t>(run_b * n_vars);
  logits.assign(logits_count, 0.0);
  x_pred.assign(logits_count, aecct_ref::bit1_t(0));

  aecct_ref::RefModelIO io{};
  io.input_y = nullptr;
  io.input_y_fp32 = &trace_input_y_step0_tensor[range.begin * n_vars];
  io.out_logits = logits.data();
  io.out_x_pred = x_pred.data();
  io.out_finalhead_s_t = out_finalhead_s_t;
  io.out_layer1_attn_input = out_layer1_attn_input;
  io.out_end_norm = out_end_norm;
  io.B = run_b;
  io.N = n_vars;
  model.infer_step0(io);
}

static inline aecct_ref::ref_fp32_t sign_fp32_local(aecct_ref::ref_fp32_t x) {
  if (x > aecct_ref::ref_fp32_t(0.0f)) return aecct_ref::ref_fp32_t(1.0f);
  if (x < aecct_ref::ref_fp32_t(0.0f)) return aecct_ref::ref_fp32_t(-1.0f);
  return aecct_ref::ref_fp32_t(0.0f);
}

static void run_experiment_from_baseline_finalhead(
  const PatternRange& range,
  int n_vars,
  aecct_ref::RefFinalHeadExploreStage stage,
  const std::vector<double>& baseline_finalhead_s_t,
  std::vector<double>& experiment_logits,
  std::vector<aecct_ref::bit1_t>& experiment_x_pred
) {
  static constexpr int kTokensT = 75;
  static constexpr int kVars = 63;
  const int run_b = range.count;
  const std::size_t logits_count = static_cast<std::size_t>(run_b * n_vars);
  experiment_logits.assign(logits_count, 0.0);
  experiment_x_pred.assign(logits_count, aecct_ref::bit1_t(0));
  const bool use_s0 = aecct_ref::stage_uses_island_s0(stage);
  const bool use_s1 = aecct_ref::stage_uses_island_s1(stage);
  const bool use_s3 = aecct_ref::stage_uses_island_s3(stage);

  for (int b = 0; b < run_b; ++b) {
    aecct_ref::ref_fp32_t out_fc_in[kTokensT];
    for (int t = 0; t < kTokensT; ++t) {
      const double s_t = baseline_finalhead_s_t[static_cast<std::size_t>(b * kTokensT + t)];
      aecct_ref::ref_fp32_t x = aecct_ref::ref_fp32_t(static_cast<float>(s_t));
      if (use_s3) {
        x = aecct_ref::roundtrip_through_generic_e4m3(x);
      }
      if (use_s0) {
        x = aecct_ref::roundtrip_through_generic_e4m3(x);
      }
      out_fc_in[t] = x;
    }

    const int src_pattern = range.begin + b;
    const double* input_y = &trace_input_y_step0_tensor[src_pattern * n_vars];
    for (int n = 0; n < kVars; ++n) {
      aecct_ref::ref_fp32_t acc = aecct_ref::ref_fp32_t(static_cast<float>(w_out_fc_bias[n]));
      for (int t = 0; t < kTokensT; ++t) {
        aecct_ref::ref_fp32_t mul_in = out_fc_in[t];
        if (use_s1) {
          mul_in = aecct_ref::roundtrip_through_generic_e4m3(mul_in);
        }
        acc += aecct_ref::ref_fp32_t(static_cast<float>(w_out_fc_weight[n * kTokensT + t])) * mul_in;
      }

      experiment_logits[static_cast<std::size_t>(b * n_vars + n)] = static_cast<double>(acc.to_float());
      const aecct_ref::ref_fp32_t y = aecct_ref::ref_fp32_t(static_cast<float>(input_y[n]));
      const aecct_ref::ref_fp32_t decision = acc * sign_fp32_local(y);
      experiment_x_pred[static_cast<std::size_t>(b * n_vars + n)] =
        aecct_ref::bit1_t((decision < aecct_ref::ref_fp32_t(0.0f)) ? 1 : 0);
    }

    for (int n = kVars; n < n_vars; ++n) {
      experiment_logits[static_cast<std::size_t>(b * n_vars + n)] = 0.0;
      experiment_x_pred[static_cast<std::size_t>(b * n_vars + n)] = aecct_ref::bit1_t(0);
    }
  }
}

static aecct_ref::Metrics compute_pattern_logits_vs_golden(
  const std::vector<double>& logits,
  int batch_index,
  int pattern_index,
  int n_vars
) {
  const double* golden_logits = &trace_output_logits_step0_tensor[pattern_index * n_vars];
  const double* pred_logits = &logits[static_cast<std::size_t>(batch_index * n_vars)];
  return aecct_ref::compute_metrics(golden_logits, pred_logits, static_cast<std::size_t>(n_vars));
}

static std::size_t compute_pattern_xpred_match_vs_golden(
  const std::vector<aecct_ref::bit1_t>& x_pred,
  int batch_index,
  int pattern_index,
  int n_vars
) {
  const double* golden_x_pred = &trace_output_x_pred_step0_tensor[pattern_index * n_vars];
  const aecct_ref::bit1_t* pred_x = &x_pred[static_cast<std::size_t>(batch_index * n_vars)];
  std::size_t match = 0;
  for (int i = 0; i < n_vars; ++i) {
    const int g = (golden_x_pred[i] != 0.0) ? 1 : 0;
    const int p = pred_x[i].to_int();
    if (g == p) {
      match++;
    }
  }
  return match;
}

static StageTensorDiffSummary compare_stage_tensor_diff(
  const char* name,
  const double* baseline,
  const double* experiment,
  int pattern_begin,
  int pattern_count,
  int per_pattern_outer,
  int per_outer_inner
) {
  StageTensorDiffSummary s{};
  s.name = name;
  const std::size_t per_pattern_count =
    static_cast<std::size_t>(per_pattern_outer) * static_cast<std::size_t>(per_outer_inner);
  s.total_count = static_cast<std::size_t>(pattern_count) * per_pattern_count;
  s.exact_equal = true;
  s.max_abs_diff = 0.0;

  for (std::size_t i = 0; i < s.total_count; ++i) {
    const double b = baseline[i];
    const double e = experiment[i];
    const double abs_diff = std::fabs(b - e);
    if (abs_diff > s.max_abs_diff) {
      s.max_abs_diff = abs_diff;
    }
    if (b != e) {
      s.mismatch_count++;
      s.exact_equal = false;
      if (!s.has_first_mismatch) {
        const std::size_t pattern_offset = (per_pattern_count > 0U) ? (i / per_pattern_count) : 0U;
        const std::size_t rem = (per_pattern_count > 0U) ? (i % per_pattern_count) : 0U;
        const int token = (per_outer_inner > 0)
          ? static_cast<int>(rem / static_cast<std::size_t>(per_outer_inner))
          : 0;
        const int dim = (per_outer_inner > 1)
          ? static_cast<int>(rem % static_cast<std::size_t>(per_outer_inner))
          : -1;
        s.has_first_mismatch = true;
        s.first_pattern = pattern_begin + static_cast<int>(pattern_offset);
        s.first_token = token;
        s.first_dim = dim;
        s.first_baseline_value = b;
        s.first_experiment_value = e;
      }
    }
  }
  return s;
}

static StageBitDiffSummary compare_xpred_diff(
  const char* name,
  const std::vector<aecct_ref::bit1_t>& baseline,
  const std::vector<aecct_ref::bit1_t>& experiment,
  int pattern_begin,
  int pattern_count,
  int n_vars
) {
  StageBitDiffSummary s{};
  s.name = name;
  const std::size_t expected = static_cast<std::size_t>(pattern_count * n_vars);
  if (baseline.size() < expected || experiment.size() < expected) {
    return s;
  }
  s.total_count = expected;
  s.exact_equal = true;
  for (std::size_t i = 0; i < expected; ++i) {
    const int b = baseline[i].to_int();
    const int e = experiment[i].to_int();
    if (b != e) {
      s.mismatch_count++;
      s.exact_equal = false;
      if (!s.has_first_mismatch) {
        const std::size_t pattern_offset = static_cast<std::size_t>(i / static_cast<std::size_t>(n_vars));
        const std::size_t idx = static_cast<std::size_t>(i % static_cast<std::size_t>(n_vars));
        s.has_first_mismatch = true;
        s.first_pattern = pattern_begin + static_cast<int>(pattern_offset);
        s.first_index = static_cast<int>(idx);
        s.first_baseline_value = b;
        s.first_experiment_value = e;
      }
    }
  }
  return s;
}

static void print_stage_tensor_diff_summary(const StageTensorDiffSummary& s, const char* coord_label) {
  std::printf("[stage:%s] exact=%s mismatch=%zu/%zu max_abs=%.9e",
    s.name,
    s.exact_equal ? "YES" : "NO",
    s.mismatch_count,
    s.total_count,
    s.max_abs_diff);
  if (s.has_first_mismatch) {
    if (s.first_dim >= 0) {
      std::printf(" first_mismatch(pattern=%d,%s=%d,dim=%d,b=%.9e,e=%.9e)\n",
        s.first_pattern,
        coord_label,
        s.first_token,
        s.first_dim,
        s.first_baseline_value,
        s.first_experiment_value);
    } else {
      std::printf(" first_mismatch(pattern=%d,%s=%d,b=%.9e,e=%.9e)\n",
        s.first_pattern,
        coord_label,
        s.first_token,
        s.first_baseline_value,
        s.first_experiment_value);
    }
    return;
  }
  std::printf(" first_mismatch(none)\n");
}

static void print_stage_bit_diff_summary(const StageBitDiffSummary& s) {
  std::printf("[stage:%s] exact=%s mismatch=%zu/%zu",
    s.name,
    s.exact_equal ? "YES" : "NO",
    s.mismatch_count,
    s.total_count);
  if (s.has_first_mismatch) {
    std::printf(" first_mismatch(pattern=%d,index=%d,b=%d,e=%d)\n",
      s.first_pattern,
      s.first_index,
      s.first_baseline_value,
      s.first_experiment_value);
    return;
  }
  std::printf(" first_mismatch(none)\n");
}

static bool write_stage_compare_summary_txt(
  const std::string& path,
  const StageTensorDiffSummary& mid_norm,
  const StageTensorDiffSummary& layer1_attn_input,
  const StageTensorDiffSummary& end_norm,
  const StageTensorDiffSummary& s_t,
  const StageTensorDiffSummary& logits,
  const StageBitDiffSummary& x_pred
) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
  if (!ofs.good()) {
    return false;
  }
  ofs.setf(std::ios::scientific);
  ofs << std::setprecision(9);
  ofs << "=== Ref-vs-Ref Stage Compare (baseline FP32 vs experiment) ===\n";
  const StageTensorDiffSummary tensors[] = {
    mid_norm, layer1_attn_input, end_norm, s_t, logits
  };
  const char* labels[] = {"token", "token", "token", "token", "index"};
  for (int i = 0; i < 5; ++i) {
    const StageTensorDiffSummary& s = tensors[i];
    ofs << "[stage:" << s.name << "] exact=" << (s.exact_equal ? "YES" : "NO")
        << " mismatch=" << s.mismatch_count << "/" << s.total_count
        << " max_abs=" << s.max_abs_diff;
    if (s.has_first_mismatch) {
      ofs << " first_mismatch(pattern=" << s.first_pattern
          << "," << labels[i] << "=" << s.first_token;
      if (s.first_dim >= 0) {
        ofs << ",dim=" << s.first_dim;
      }
      ofs << ",b=" << s.first_baseline_value
          << ",e=" << s.first_experiment_value << ")";
    } else {
      ofs << " first_mismatch(none)";
    }
    ofs << "\n";
  }
  ofs << "[stage:" << x_pred.name << "] exact=" << (x_pred.exact_equal ? "YES" : "NO")
      << " mismatch=" << x_pred.mismatch_count << "/" << x_pred.total_count;
  if (x_pred.has_first_mismatch) {
    ofs << " first_mismatch(pattern=" << x_pred.first_pattern
        << ",index=" << x_pred.first_index
        << ",b=" << x_pred.first_baseline_value
        << ",e=" << x_pred.first_experiment_value << ")";
  } else {
    ofs << " first_mismatch(none)";
  }
  ofs << "\n";
  return true;
}

static EvalPatternRow build_eval_pattern_row(
  const aecct_ref::bit1_t* pred_x,
  int pattern_index,
  int n_vars
) {
  EvalPatternRow row{};
  row.pattern_index = pattern_index;
  row.evaluated_bits = static_cast<std::size_t>(n_vars);
  const double* target_x = &trace_output_x_pred_step0_tensor[pattern_index * n_vars];
  for (int i = 0; i < n_vars; ++i) {
    const int target = (target_x[i] != 0.0) ? 1 : 0;
    const int pred = pred_x[i].to_int();
    if (target == pred) {
      row.x_pred_match_count++;
    } else {
      row.bit_errors++;
    }
  }
  row.frame_error_flag = (row.bit_errors > 0U) ? 1 : 0;
  return row;
}

static void init_eval_aggregate(EvalAggregateStats& s) {
  s.total_patterns = 0;
  s.total_bits = 0;
  s.total_bit_errors = 0;
  s.frame_error_count = 0;
  s.x_pred_match_count = 0;
  s.ber = 0.0;
  s.fer = 0.0;
  s.x_pred_match_ratio = 0.0;
}

static void update_eval_aggregate(EvalAggregateStats& s, const EvalPatternRow& row) {
  s.total_patterns += 1;
  s.total_bits += row.evaluated_bits;
  s.total_bit_errors += row.bit_errors;
  s.frame_error_count += static_cast<std::size_t>(row.frame_error_flag);
  s.x_pred_match_count += row.x_pred_match_count;
}

static void finalize_eval_aggregate(EvalAggregateStats& s) {
  s.ber = (s.total_bits > 0U)
    ? (static_cast<double>(s.total_bit_errors) / static_cast<double>(s.total_bits))
    : 0.0;
  s.fer = (s.total_patterns > 0)
    ? (static_cast<double>(s.frame_error_count) / static_cast<double>(s.total_patterns))
    : 0.0;
  s.x_pred_match_ratio = (s.total_bits > 0U)
    ? (static_cast<double>(s.x_pred_match_count) / static_cast<double>(s.total_bits))
    : 0.0;
}

static void print_vs_golden_summary(const char* tag, const RefRunOutputs& run) {
  const double match_ratio = (run.x_pred_total_count > 0U)
    ? (100.0 * static_cast<double>(run.x_pred_match_count) /
       static_cast<double>(run.x_pred_total_count))
    : 0.0;

  std::printf("=== %s vs golden ===\n", tag);
  std::printf("logits MSE    : %.9e\n", run.logits_vs_golden.mse);
  std::printf("logits RMSE   : %.9e\n", run.logits_vs_golden.rmse);
  std::printf("logits MAE    : %.9e\n", run.logits_vs_golden.mae);
  std::printf("logits MaxAbs : %.9e\n", run.logits_vs_golden.max_abs);
  std::printf("x_pred match  : %.2f%% (%zu / %zu)\n",
    match_ratio, run.x_pred_match_count, run.x_pred_total_count);
}

static std::string format_topk_entries(const std::vector<aecct_ref::RefExperimentCompareMetrics::TopKEntry>& entries) {
  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss << std::setprecision(6);
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (i > 0U) {
      oss << "|";
    }
    oss << entries[i].index << ":" << entries[i].value << ":" << entries[i].abs_value;
  }
  return oss.str();
}

static inline std::uint64_t make_ln_debug_key(int sample_index, int site, int token) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sample_index)) << 32) |
         (static_cast<std::uint64_t>(static_cast<std::uint16_t>(site)) << 16) |
         static_cast<std::uint64_t>(static_cast<std::uint16_t>(token));
}

static bool collect_ln_debug_entries(std::vector<LnDebugEntryRow>& out) {
  out.clear();
  const int count = aecct_ref::get_ref_ln_debug_entry_count();
  if (count <= 0) {
    return true;
  }
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    LnDebugEntryRow e{};
    if (!aecct_ref::get_ref_ln_debug_entry(
          i,
          &e.entry_seq,
          &e.site_call_index,
          &e.sample_index,
          &e.site,
          &e.token,
          &e.sum,
          &e.sumsq,
          &e.mean,
          &e.ex2,
          &e.mean_sq,
          &e.var_raw,
          &e.var_final,
          &e.var_from_residual,
          &e.var_from_ex2,
          &e.eps_applied,
          &e.inv_std_input,
          &e.sum_seq,
          &e.sumsq_seq,
          &e.sum_tile,
          &e.sumsq_tile,
          &e.x_eps,
          &e.inv_std_used,
          &e.inv_std_true,
          &e.inv_std_seed,
          &e.inv_std_nr1,
          &e.var_negative_before_clamp,
          &e.clamp_triggered,
          &e.eps_dominates_var,
          &e.sanitize_input_count,
          e.x,
          LN_DEBUG_D_MODEL,
          e.y,
          LN_DEBUG_D_MODEL)) {
      return false;
    }
    out.push_back(e);
  }
  return true;
}

static bool collect_ln_upstream_debug_entries(std::vector<LnUpstreamEntryRow>& out) {
  out.clear();
  const int count = aecct_ref::get_ref_ln_upstream_debug_entry_count();
  if (count <= 0) {
    return true;
  }
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    LnUpstreamEntryRow e{};
    if (!aecct_ref::get_ref_ln_upstream_debug_entry(
          i,
          &e.entry_seq,
          &e.sample_index,
          &e.layer_idx,
          &e.boundary,
          &e.boundary_call_index,
          &e.token,
          e.x,
          LN_DEBUG_D_MODEL)) {
      return false;
    }
    out.push_back(e);
  }
  return true;
}

static void dump_ln_debug_report(
  int pattern_begin,
  int pattern_count,
  const std::vector<LnDebugEntryRow>& baseline_entries,
  const std::vector<LnDebugEntryRow>& experiment_entries,
  const std::string& out_path
) {
  std::unordered_map<std::uint64_t, std::size_t> exp_index;
  exp_index.reserve(experiment_entries.size());
  for (std::size_t i = 0; i < experiment_entries.size(); ++i) {
    exp_index[make_ln_debug_key(
      experiment_entries[i].sample_index,
      experiment_entries[i].site,
      experiment_entries[i].token)] = i;
  }

  std::vector<LnDebugTokenDiag> token_diags;
  token_diags.reserve(baseline_entries.size());

  double seed_abs_err_sum = 0.0;
  double nr_abs_err_sum = 0.0;
  double inv_std_abs_diff_sum = 0.0;
  double var_abs_diff_sum = 0.0;
  double seed_rel_err_sum = 0.0;
  double nr_rel_err_sum = 0.0;
  double inv_std_rel_diff_sum = 0.0;
  std::size_t matched = 0U;
  std::size_t var_neg_count = 0U;
  std::size_t clamp_count = 0U;
  std::size_t eps_dom_count = 0U;
  std::size_t sanitize_count = 0U;
  std::size_t small_var_count = 0U;
  double var_raw_min = std::numeric_limits<double>::infinity();
  double var_raw_max = -std::numeric_limits<double>::infinity();
  double var_final_min = std::numeric_limits<double>::infinity();
  double var_final_max = -std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i < baseline_entries.size(); ++i) {
    const LnDebugEntryRow& b = baseline_entries[i];
    const auto it = exp_index.find(make_ln_debug_key(b.sample_index, b.site, b.token));
    if (it == exp_index.end()) {
      continue;
    }
    const LnDebugEntryRow& e = experiment_entries[it->second];

    LnDebugTokenDiag d{};
    d.pattern = pattern_begin + b.sample_index;
    d.site = b.site;
    d.token = b.token;
    d.mean_base = b.mean;
    d.mean_exp = e.mean;
    d.var_base = b.var_final;
    d.var_exp = e.var_final;
    d.var_abs_diff = static_cast<float>(std::fabs(static_cast<double>(e.var_final) - static_cast<double>(b.var_final)));
    d.x_eps = e.x_eps;
    d.inv_std_base_true = b.inv_std_true;
    d.inv_std_exp_true = e.inv_std_true;
    d.inv_std_exp_nr1 = e.inv_std_nr1;
    d.inv_std_seed = e.inv_std_seed;
    d.inv_std_cross_abs_diff = static_cast<float>(std::fabs(static_cast<double>(d.inv_std_exp_nr1) - static_cast<double>(d.inv_std_base_true)));
    const double inv_cross_ref = std::max(std::fabs(static_cast<double>(d.inv_std_base_true)), 1.0e-12);
    d.inv_std_cross_rel_diff = static_cast<float>(d.inv_std_cross_abs_diff / inv_cross_ref);
    const double inv_local_ref = std::max(std::fabs(static_cast<double>(d.inv_std_exp_true)), 1.0e-12);
    d.seed_abs_err = static_cast<float>(std::fabs(static_cast<double>(d.inv_std_seed) - static_cast<double>(d.inv_std_exp_true)));
    d.seed_rel_err = static_cast<float>(d.seed_abs_err / inv_local_ref);
    d.nr_abs_err = static_cast<float>(std::fabs(static_cast<double>(d.inv_std_exp_nr1) - static_cast<double>(d.inv_std_exp_true)));
    d.nr_rel_err = static_cast<float>(d.nr_abs_err / inv_local_ref);
    d.var_negative_before_clamp = e.var_negative_before_clamp;
    d.clamp_triggered = e.clamp_triggered;
    d.eps_dominates_var = e.eps_dominates_var;
    d.sanitize_input_count = e.sanitize_input_count;

    double y_abs_sum = 0.0;
    double y_abs_max = 0.0;
    for (int k = 0; k < LN_DEBUG_D_MODEL; ++k) {
      const double ad = std::fabs(static_cast<double>(e.y[k]) - static_cast<double>(b.y[k]));
      y_abs_sum += ad;
      if (ad > y_abs_max) {
        y_abs_max = ad;
      }
    }
    d.y_max_abs_diff = static_cast<float>(y_abs_max);
    d.y_mean_abs_diff = static_cast<float>(y_abs_sum / static_cast<double>(LN_DEBUG_D_MODEL));
    token_diags.push_back(d);
    matched++;

    seed_abs_err_sum += d.seed_abs_err;
    nr_abs_err_sum += d.nr_abs_err;
    inv_std_abs_diff_sum += d.inv_std_cross_abs_diff;
    var_abs_diff_sum += d.var_abs_diff;
    seed_rel_err_sum += d.seed_rel_err;
    nr_rel_err_sum += d.nr_rel_err;
    inv_std_rel_diff_sum += d.inv_std_cross_rel_diff;

    if (d.var_negative_before_clamp != 0) var_neg_count++;
    if (d.clamp_triggered != 0) clamp_count++;
    if (d.eps_dominates_var != 0) eps_dom_count++;
    sanitize_count += static_cast<std::size_t>((d.sanitize_input_count > 0) ? d.sanitize_input_count : 0);
    if (e.var_final < 1.0e-4f) {
      small_var_count++;
    }
    if (e.var_raw < var_raw_min) var_raw_min = e.var_raw;
    if (e.var_raw > var_raw_max) var_raw_max = e.var_raw;
    if (e.var_final < var_final_min) var_final_min = e.var_final;
    if (e.var_final > var_final_max) var_final_max = e.var_final;
  }

  std::vector<LnDebugTokenDiag> pattern_worst(static_cast<std::size_t>(pattern_count));
  std::vector<int> pattern_has(static_cast<std::size_t>(pattern_count), 0);
  LnDebugTokenDiag global_worst{};
  bool has_global = false;
  for (std::size_t i = 0; i < token_diags.size(); ++i) {
    const LnDebugTokenDiag& d = token_diags[i];
    const int off = d.pattern - pattern_begin;
    if (off < 0 || off >= pattern_count) {
      continue;
    }
    if (pattern_has[static_cast<std::size_t>(off)] == 0 ||
        d.y_max_abs_diff > pattern_worst[static_cast<std::size_t>(off)].y_max_abs_diff) {
      pattern_worst[static_cast<std::size_t>(off)] = d;
      pattern_has[static_cast<std::size_t>(off)] = 1;
    }
    if (!has_global || d.y_max_abs_diff > global_worst.y_max_abs_diff) {
      global_worst = d;
      has_global = true;
    }
  }

  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss << std::setprecision(9);
  oss << "=== LN Debug Diagnosis ===\n";
  oss << "baseline_entries: " << baseline_entries.size() << "\n";
  oss << "experiment_entries: " << experiment_entries.size() << "\n";
  oss << "matched_entries: " << matched << "\n";
  if (matched > 0U) {
    const double inv = 1.0 / static_cast<double>(matched);
    oss << "mean_seed_abs_err: " << (seed_abs_err_sum * inv) << "\n";
    oss << "mean_nr_abs_err  : " << (nr_abs_err_sum * inv) << "\n";
    oss << "mean_seed_rel_err: " << (seed_rel_err_sum * inv) << "\n";
    oss << "mean_nr_rel_err  : " << (nr_rel_err_sum * inv) << "\n";
    oss << "mean_invstd_cross_abs_diff(base_true vs exp_nr1): " << (inv_std_abs_diff_sum * inv) << "\n";
    oss << "mean_invstd_cross_rel_diff(base_true vs exp_nr1): " << (inv_std_rel_diff_sum * inv) << "\n";
    oss << "mean_var_abs_diff(base vs exp): " << (var_abs_diff_sum * inv) << "\n";
    oss << "var_negative_before_clamp_count(exp): " << var_neg_count << "\n";
    oss << "clamp_trigger_count(exp): " << clamp_count << "\n";
    oss << "eps_dominates_var_count(exp): " << eps_dom_count << "\n";
    oss << "small_var_count(exp, var<1e-4): " << small_var_count << "\n";
    oss << "sanitize_input_total(exp): " << sanitize_count << "\n";
    oss << "var_raw_range(exp): [" << var_raw_min << ", " << var_raw_max << "]\n";
    oss << "var_final_range(exp): [" << var_final_min << ", " << var_final_max << "]\n";
  }
  oss << "\n";

  for (int p = 0; p < pattern_count; ++p) {
    if (pattern_has[static_cast<std::size_t>(p)] == 0) {
      continue;
    }
    const LnDebugTokenDiag& d = pattern_worst[static_cast<std::size_t>(p)];
    oss << "[Pattern " << d.pattern << "]\n";
    oss << "[Token " << d.token << "] [Site " << aecct_ref::ref_ln_debug_site_name(d.site) << "]\n";
    oss << "mean_base / mean_exp: " << d.mean_base << " / " << d.mean_exp << "\n";
    oss << "var_base  / var_exp : " << d.var_base << " / " << d.var_exp << "\n";
    oss << "var_abs_diff         : " << d.var_abs_diff << "\n";
    oss << "x(var+eps)           : " << d.x_eps << "\n";
    oss << "inv_std_base_true    : " << d.inv_std_base_true << "\n";
    oss << "inv_std_exp_true     : " << d.inv_std_exp_true << "\n";
    oss << "inv_std_exp_nr1      : " << d.inv_std_exp_nr1 << "\n";
    oss << "inv_std_cross_abs_diff(base_true vs exp_nr1): " << d.inv_std_cross_abs_diff << "\n";
    oss << "inv_std_cross_rel_diff(base_true vs exp_nr1): " << d.inv_std_cross_rel_diff << "\n";
    oss << "y0(seed)             : " << d.inv_std_seed << "\n";
    oss << "y1(NR1)              : " << d.inv_std_exp_nr1 << "\n";
    oss << "y_true(1/sqrt(x_exp)): " << d.inv_std_exp_true << "\n";
    oss << "seed_abs_err / rel   : " << d.seed_abs_err << " / " << d.seed_rel_err << "\n";
    oss << "nr_abs_err / rel     : " << d.nr_abs_err << " / " << d.nr_rel_err << "\n";
    oss << "LN y max_abs_diff    : " << d.y_max_abs_diff << "\n";
    oss << "LN y mean_abs_diff   : " << d.y_mean_abs_diff << "\n";
    oss << "var_negative/clamp/eps_dom/sanitize_count(exp): "
        << d.var_negative_before_clamp << " / "
        << d.clamp_triggered << " / "
        << d.eps_dominates_var << " / "
        << d.sanitize_input_count << "\n\n";
  }

  if (has_global) {
    oss << "[Global worst token across batch]\n";
    oss << "pattern/token/site: " << global_worst.pattern << " / "
        << global_worst.token << " / "
        << aecct_ref::ref_ln_debug_site_name(global_worst.site) << "\n";
    oss << "LN y max_abs_diff : " << global_worst.y_max_abs_diff << "\n";
    oss << "LN y mean_abs_diff: " << global_worst.y_mean_abs_diff << "\n";
    oss << "inv_std base_true / exp_true / exp_nr1: " << global_worst.inv_std_base_true
        << " / " << global_worst.inv_std_exp_true
        << " / " << global_worst.inv_std_exp_nr1 << "\n";
    oss << "seed_abs_err / nr_abs_err: " << global_worst.seed_abs_err
        << " / " << global_worst.nr_abs_err << "\n";
  }

  const std::string report = oss.str();
  std::printf("%s", report.c_str());

  if (!out_path.empty()) {
    std::filesystem::path p(out_path);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream ofs(out_path.c_str(), std::ios::out | std::ios::trunc);
    if (ofs.good()) {
      ofs << report;
    }
  }
}

static inline float abs_diff_f(float a, float b) {
  return static_cast<float>(std::fabs(static_cast<double>(a) - static_cast<double>(b)));
}

static inline float rel_diff_f(float a, float b) {
  const double absd = std::fabs(static_cast<double>(a) - static_cast<double>(b));
  const double ref = std::max(
    std::max(std::fabs(static_cast<double>(a)), std::fabs(static_cast<double>(b))),
    1.0e-12);
  return static_cast<float>(absd / ref);
}

static inline bool is_material_diff(float a, float b) {
  const double absd = std::fabs(static_cast<double>(a) - static_cast<double>(b));
  const double ref = std::max(
    std::max(std::fabs(static_cast<double>(a)), std::fabs(static_cast<double>(b))),
    1.0);
  const double rel = absd / ref;
  return absd > 1.0e-5 || rel > 1.0e-3;
}

static const char* ln_var_stage_to_string(LnVarFirstDivergenceStage s) {
  switch (s) {
    case LnVarFirstDivergenceStage::SUM_OR_SUMSQ:
      return "summation/sumsq accumulation";
    case LnVarFirstDivergenceStage::MOMENT_DERIVED:
      return "derived moments (mean/ex2/mean_sq)";
    case LnVarFirstDivergenceStage::VAR_FORMULA:
      return "variance formation (ex2-mean_sq vs residual)";
    case LnVarFirstDivergenceStage::POST_VAR_HANDLING:
      return "post-var handling (clamp/eps/inv_std_input)";
    default:
      return "none";
  }
}

static LnVarFirstDivergenceStage detect_first_var_divergence(const LnVarTokenDiag& d) {
  if (is_material_diff(d.base_sum, d.exp_sum) || is_material_diff(d.base_sumsq, d.exp_sumsq)) {
    return LnVarFirstDivergenceStage::SUM_OR_SUMSQ;
  }
  if (is_material_diff(d.base_mean, d.exp_mean) ||
      is_material_diff(d.base_ex2, d.exp_ex2) ||
      is_material_diff(d.base_mean_sq, d.exp_mean_sq)) {
    return LnVarFirstDivergenceStage::MOMENT_DERIVED;
  }
  if (is_material_diff(d.base_var_raw, d.exp_var_raw)) {
    return LnVarFirstDivergenceStage::VAR_FORMULA;
  }
  if (is_material_diff(d.base_var_final, d.exp_var_final) ||
      is_material_diff(d.base_eps_applied, d.exp_eps_applied) ||
      is_material_diff(d.base_inv_std_input, d.exp_inv_std_input)) {
    return LnVarFirstDivergenceStage::POST_VAR_HANDLING;
  }
  return LnVarFirstDivergenceStage::NONE;
}

static void append_var_pair_line(std::ostringstream& oss, const char* label, float base_v, float exp_v) {
  oss << label
      << " base/exp=" << base_v << " / " << exp_v
      << " abs_diff=" << abs_diff_f(base_v, exp_v)
      << " rel_diff=" << rel_diff_f(base_v, exp_v)
      << "\n";
}

static void append_ln_var_token_details(std::ostringstream& oss, const LnVarTokenDiag& d) {
  oss << "[Pattern " << d.pattern << "] [Site " << aecct_ref::ref_ln_debug_site_name(d.site)
      << "] [Token " << d.token << "]\n";
  oss << "first_divergence_stage: " << ln_var_stage_to_string(d.first_stage) << "\n";
  oss << "Baseline:\n";
  oss << "  sum=" << d.base_sum
      << " sumsq=" << d.base_sumsq
      << " mean=" << d.base_mean
      << " ex2=" << d.base_ex2
      << " mean_sq=" << d.base_mean_sq
      << " var_raw=" << d.base_var_raw
      << " var_final=" << d.base_var_final
      << " eps_applied=" << d.base_eps_applied
      << " inv_std_input=" << d.base_inv_std_input
      << "\n";
  oss << "  var(residual)=" << d.base_var_from_residual
      << " var(ex2-mean_sq)=" << d.base_var_from_ex2
      << " order(sum seq/tile)=" << d.base_sum_seq << " / " << d.base_sum_tile
      << " order(sumsq seq/tile)=" << d.base_sumsq_seq << " / " << d.base_sumsq_tile
      << "\n";
  oss << "Experimental:\n";
  oss << "  sum=" << d.exp_sum
      << " sumsq=" << d.exp_sumsq
      << " mean=" << d.exp_mean
      << " ex2=" << d.exp_ex2
      << " mean_sq=" << d.exp_mean_sq
      << " var_raw=" << d.exp_var_raw
      << " var_final=" << d.exp_var_final
      << " eps_applied=" << d.exp_eps_applied
      << " inv_std_input=" << d.exp_inv_std_input
      << "\n";
  oss << "  var(residual)=" << d.exp_var_from_residual
      << " var(ex2-mean_sq)=" << d.exp_var_from_ex2
      << " order(sum seq/tile)=" << d.exp_sum_seq << " / " << d.exp_sum_tile
      << " order(sumsq seq/tile)=" << d.exp_sumsq_seq << " / " << d.exp_sumsq_tile
      << "\n";
  oss << "Differences:\n";
  append_var_pair_line(oss, "  sum           ", d.base_sum, d.exp_sum);
  append_var_pair_line(oss, "  sumsq         ", d.base_sumsq, d.exp_sumsq);
  append_var_pair_line(oss, "  mean          ", d.base_mean, d.exp_mean);
  append_var_pair_line(oss, "  ex2           ", d.base_ex2, d.exp_ex2);
  append_var_pair_line(oss, "  mean_sq       ", d.base_mean_sq, d.exp_mean_sq);
  append_var_pair_line(oss, "  var_raw       ", d.base_var_raw, d.exp_var_raw);
  append_var_pair_line(oss, "  var_final     ", d.base_var_final, d.exp_var_final);
  append_var_pair_line(oss, "  eps_applied   ", d.base_eps_applied, d.exp_eps_applied);
  append_var_pair_line(oss, "  inv_std_input ", d.base_inv_std_input, d.exp_inv_std_input);
  append_var_pair_line(oss, "  var(residual) ", d.base_var_from_residual, d.exp_var_from_residual);
  append_var_pair_line(oss, "  var(ex2 path) ", d.base_var_from_ex2, d.exp_var_from_ex2);
  oss << "  order_delta_base(sum/sumsq): "
      << abs_diff_f(d.base_sum_seq, d.base_sum_tile) << " / "
      << abs_diff_f(d.base_sumsq_seq, d.base_sumsq_tile) << "\n";
  oss << "  order_delta_exp(sum/sumsq): "
      << abs_diff_f(d.exp_sum_seq, d.exp_sum_tile) << " / "
      << abs_diff_f(d.exp_sumsq_seq, d.exp_sumsq_tile) << "\n";
  oss << "  LN output max/mean abs diff: "
      << d.y_max_abs_diff << " / " << d.y_mean_abs_diff << "\n\n";
}

static void dump_ln_var_debug_report(
  int pattern_begin,
  int pattern_count,
  const std::vector<LnDebugEntryRow>& baseline_entries,
  const std::vector<LnDebugEntryRow>& experiment_entries,
  const std::string& out_path
) {
  std::unordered_map<std::uint64_t, std::size_t> exp_index;
  exp_index.reserve(experiment_entries.size());
  for (std::size_t i = 0; i < experiment_entries.size(); ++i) {
    exp_index[make_ln_debug_key(
      experiment_entries[i].sample_index,
      experiment_entries[i].site,
      experiment_entries[i].token)] = i;
  }

  std::vector<LnVarTokenDiag> token_diags;
  token_diags.reserve(baseline_entries.size());
  double sum_abs_diff_sum = 0.0;
  double sumsq_abs_diff_sum = 0.0;
  double mean_abs_diff_sum = 0.0;
  double ex2_abs_diff_sum = 0.0;
  double mean_sq_abs_diff_sum = 0.0;
  double var_raw_abs_diff_sum = 0.0;
  double var_final_abs_diff_sum = 0.0;
  double eps_abs_diff_sum = 0.0;
  double inv_input_abs_diff_sum = 0.0;
  double base_formula_gap_sum = 0.0;
  double exp_formula_gap_sum = 0.0;
  double base_formula_gap_max = 0.0;
  double exp_formula_gap_max = 0.0;
  double base_order_sum_abs = 0.0;
  double exp_order_sum_abs = 0.0;
  double base_order_sumsq_abs = 0.0;
  double exp_order_sumsq_abs = 0.0;
  std::size_t cancellation_amp_count = 0U;
  std::size_t order_significant_count = 0U;
  std::size_t matched = 0U;
  int stage_hist[5] = {0, 0, 0, 0, 0};

  for (std::size_t i = 0; i < baseline_entries.size(); ++i) {
    const LnDebugEntryRow& b = baseline_entries[i];
    const auto it = exp_index.find(make_ln_debug_key(b.sample_index, b.site, b.token));
    if (it == exp_index.end()) {
      continue;
    }
    const LnDebugEntryRow& e = experiment_entries[it->second];
    LnVarTokenDiag d{};
    d.pattern = pattern_begin + b.sample_index;
    d.site = b.site;
    d.token = b.token;
    d.base_sum = b.sum;
    d.exp_sum = e.sum;
    d.base_sumsq = b.sumsq;
    d.exp_sumsq = e.sumsq;
    d.base_mean = b.mean;
    d.exp_mean = e.mean;
    d.base_ex2 = b.ex2;
    d.exp_ex2 = e.ex2;
    d.base_mean_sq = b.mean_sq;
    d.exp_mean_sq = e.mean_sq;
    d.base_var_raw = b.var_raw;
    d.exp_var_raw = e.var_raw;
    d.base_var_final = b.var_final;
    d.exp_var_final = e.var_final;
    d.base_eps_applied = b.eps_applied;
    d.exp_eps_applied = e.eps_applied;
    d.base_inv_std_input = b.inv_std_input;
    d.exp_inv_std_input = e.inv_std_input;
    d.base_var_from_residual = b.var_from_residual;
    d.exp_var_from_residual = e.var_from_residual;
    d.base_var_from_ex2 = b.var_from_ex2;
    d.exp_var_from_ex2 = e.var_from_ex2;
    d.base_sum_seq = b.sum_seq;
    d.exp_sum_seq = e.sum_seq;
    d.base_sum_tile = b.sum_tile;
    d.exp_sum_tile = e.sum_tile;
    d.base_sumsq_seq = b.sumsq_seq;
    d.exp_sumsq_seq = e.sumsq_seq;
    d.base_sumsq_tile = b.sumsq_tile;
    d.exp_sumsq_tile = e.sumsq_tile;
    d.var_raw_abs_diff = abs_diff_f(b.var_raw, e.var_raw);
    d.var_final_abs_diff = abs_diff_f(b.var_final, e.var_final);
    d.first_stage = detect_first_var_divergence(d);

    double y_abs_sum = 0.0;
    double y_abs_max = 0.0;
    for (int k = 0; k < LN_DEBUG_D_MODEL; ++k) {
      const double ad = std::fabs(static_cast<double>(e.y[k]) - static_cast<double>(b.y[k]));
      y_abs_sum += ad;
      if (ad > y_abs_max) {
        y_abs_max = ad;
      }
    }
    d.y_max_abs_diff = static_cast<float>(y_abs_max);
    d.y_mean_abs_diff = static_cast<float>(y_abs_sum / static_cast<double>(LN_DEBUG_D_MODEL));
    token_diags.push_back(d);
    matched++;

    sum_abs_diff_sum += abs_diff_f(d.base_sum, d.exp_sum);
    sumsq_abs_diff_sum += abs_diff_f(d.base_sumsq, d.exp_sumsq);
    mean_abs_diff_sum += abs_diff_f(d.base_mean, d.exp_mean);
    ex2_abs_diff_sum += abs_diff_f(d.base_ex2, d.exp_ex2);
    mean_sq_abs_diff_sum += abs_diff_f(d.base_mean_sq, d.exp_mean_sq);
    var_raw_abs_diff_sum += d.var_raw_abs_diff;
    var_final_abs_diff_sum += d.var_final_abs_diff;
    eps_abs_diff_sum += abs_diff_f(d.base_eps_applied, d.exp_eps_applied);
    inv_input_abs_diff_sum += abs_diff_f(d.base_inv_std_input, d.exp_inv_std_input);
    const double base_formula_gap = abs_diff_f(d.base_var_from_residual, d.base_var_from_ex2);
    const double exp_formula_gap = abs_diff_f(d.exp_var_from_residual, d.exp_var_from_ex2);
    base_formula_gap_sum += base_formula_gap;
    exp_formula_gap_sum += exp_formula_gap;
    if (base_formula_gap > base_formula_gap_max) {
      base_formula_gap_max = base_formula_gap;
    }
    if (exp_formula_gap > exp_formula_gap_max) {
      exp_formula_gap_max = exp_formula_gap;
    }
    const float base_sum_order = abs_diff_f(d.base_sum_seq, d.base_sum_tile);
    const float exp_sum_order = abs_diff_f(d.exp_sum_seq, d.exp_sum_tile);
    const float base_sumsq_order = abs_diff_f(d.base_sumsq_seq, d.base_sumsq_tile);
    const float exp_sumsq_order = abs_diff_f(d.exp_sumsq_seq, d.exp_sumsq_tile);
    base_order_sum_abs += base_sum_order;
    exp_order_sum_abs += exp_sum_order;
    base_order_sumsq_abs += base_sumsq_order;
    exp_order_sumsq_abs += exp_sumsq_order;
    if (abs_diff_f(d.base_sum, d.exp_sum) < 1.0e-3f &&
        abs_diff_f(d.base_sumsq, d.exp_sumsq) < 1.0e-3f &&
        d.var_raw_abs_diff > 1.0e-2f) {
      cancellation_amp_count++;
    }
    if (exp_sum_order > (0.25f * std::max(abs_diff_f(d.base_sum, d.exp_sum), 1.0e-12f)) ||
        exp_sumsq_order > (0.25f * std::max(abs_diff_f(d.base_sumsq, d.exp_sumsq), 1.0e-12f))) {
      order_significant_count++;
    }
    stage_hist[static_cast<int>(d.first_stage)]++;
  }

  std::vector<LnVarTokenDiag> pattern_worst(static_cast<std::size_t>(pattern_count));
  std::vector<int> pattern_has(static_cast<std::size_t>(pattern_count), 0);
  for (std::size_t i = 0; i < token_diags.size(); ++i) {
    const LnVarTokenDiag& d = token_diags[i];
    const int off = d.pattern - pattern_begin;
    if (off < 0 || off >= pattern_count) {
      continue;
    }
    if (pattern_has[static_cast<std::size_t>(off)] == 0 ||
        d.var_final_abs_diff > pattern_worst[static_cast<std::size_t>(off)].var_final_abs_diff) {
      pattern_worst[static_cast<std::size_t>(off)] = d;
      pattern_has[static_cast<std::size_t>(off)] = 1;
    }
  }

  std::vector<LnVarTokenDiag> sorted_by_var_final = token_diags;
  std::sort(sorted_by_var_final.begin(), sorted_by_var_final.end(),
    [](const LnVarTokenDiag& a, const LnVarTokenDiag& b) {
      return a.var_final_abs_diff > b.var_final_abs_diff;
    });
  std::vector<LnVarTokenDiag> sorted_by_var_raw = token_diags;
  std::sort(sorted_by_var_raw.begin(), sorted_by_var_raw.end(),
    [](const LnVarTokenDiag& a, const LnVarTokenDiag& b) {
      return a.var_raw_abs_diff > b.var_raw_abs_diff;
    });

  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss << std::setprecision(9);
  oss << "=== LN Var Alignment Diagnosis ===\n";
  oss << "baseline_entries: " << baseline_entries.size() << "\n";
  oss << "experiment_entries: " << experiment_entries.size() << "\n";
  oss << "matched_entries: " << matched << "\n";
  oss << "baseline_variance_formula(actual): var = mean((x - mean)^2)\n";
  oss << "baseline_variance_formula(reconstructed): var = E[x^2] - mean^2\n";
  oss << "experimental_variance_formula(actual): var = E[x^2] - mean^2\n";
  oss << "mathematical_equivalence(ideal arithmetic): yes\n";
  oss << "accumulator_types: baseline(sum=float,var_acc=float) experimental(sum=float,sumsq=float)\n";
  oss << "accumulation_order: baseline=sequential(0..31), experimental=tiled(tile=8, tile-major)\n";
  if (matched > 0U) {
    const double inv = 1.0 / static_cast<double>(matched);
    oss << "mean_abs_diff(sum): " << (sum_abs_diff_sum * inv) << "\n";
    oss << "mean_abs_diff(sumsq): " << (sumsq_abs_diff_sum * inv) << "\n";
    oss << "mean_abs_diff(mean): " << (mean_abs_diff_sum * inv) << "\n";
    oss << "mean_abs_diff(ex2): " << (ex2_abs_diff_sum * inv) << "\n";
    oss << "mean_abs_diff(mean_sq): " << (mean_sq_abs_diff_sum * inv) << "\n";
    oss << "mean_abs_diff(var_raw): " << (var_raw_abs_diff_sum * inv) << "\n";
    oss << "mean_abs_diff(var_final): " << (var_final_abs_diff_sum * inv) << "\n";
    oss << "mean_abs_diff(eps_applied): " << (eps_abs_diff_sum * inv) << "\n";
    oss << "mean_abs_diff(inv_std_input): " << (inv_input_abs_diff_sum * inv) << "\n";
    oss << "mean_abs_gap_base(var_residual vs var_ex2): " << (base_formula_gap_sum * inv) << "\n";
    oss << "mean_abs_gap_exp(var_residual vs var_ex2): " << (exp_formula_gap_sum * inv) << "\n";
    oss << "max_abs_gap_base(var_residual vs var_ex2): " << base_formula_gap_max << "\n";
    oss << "max_abs_gap_exp(var_residual vs var_ex2): " << exp_formula_gap_max << "\n";
    oss << "mean_order_delta_base(sum/sumsq): " << (base_order_sum_abs * inv) << " / "
        << (base_order_sumsq_abs * inv) << "\n";
    oss << "mean_order_delta_exp(sum/sumsq): " << (exp_order_sum_abs * inv) << " / "
        << (exp_order_sumsq_abs * inv) << "\n";
  }
  oss << "first_divergence_histogram:\n";
  oss << "  none: " << stage_hist[static_cast<int>(LnVarFirstDivergenceStage::NONE)] << "\n";
  oss << "  summation/sumsq: " << stage_hist[static_cast<int>(LnVarFirstDivergenceStage::SUM_OR_SUMSQ)] << "\n";
  oss << "  derived moments: " << stage_hist[static_cast<int>(LnVarFirstDivergenceStage::MOMENT_DERIVED)] << "\n";
  oss << "  variance formation: " << stage_hist[static_cast<int>(LnVarFirstDivergenceStage::VAR_FORMULA)] << "\n";
  oss << "  post-var handling: " << stage_hist[static_cast<int>(LnVarFirstDivergenceStage::POST_VAR_HANDLING)] << "\n";
  oss << "cancellation_amplification_count(sum/sumsq close but var_raw large): "
      << cancellation_amp_count << "\n";
  oss << "order_significant_count(exp order delta >=25% of base-exp sum/sumsq diff): "
      << order_significant_count << "\n\n";

  oss << "=== Worst Token Per Pattern (rank by |var_final_base - var_final_exp|) ===\n";
  for (int p = 0; p < pattern_count; ++p) {
    if (pattern_has[static_cast<std::size_t>(p)] == 0) {
      continue;
    }
    append_ln_var_token_details(oss, pattern_worst[static_cast<std::size_t>(p)]);
  }

  if (!sorted_by_var_final.empty()) {
    oss << "=== Global Worst Token Across Batch (by |var_final diff|) ===\n";
    append_ln_var_token_details(oss, sorted_by_var_final.front());
  }

  oss << "=== Top-3 by |var_final_base - var_final_exp| ===\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(3U, sorted_by_var_final.size()); ++i) {
    oss << "[Rank " << (i + 1U) << "]\n";
    append_ln_var_token_details(oss, sorted_by_var_final[i]);
  }

  oss << "=== Top-3 by |var_raw_base - var_raw_exp| ===\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(3U, sorted_by_var_raw.size()); ++i) {
    oss << "[Rank " << (i + 1U) << "]\n";
    append_ln_var_token_details(oss, sorted_by_var_raw[i]);
  }

  const std::string report = oss.str();
  std::printf("%s", report.c_str());
  if (!out_path.empty()) {
    std::filesystem::path p(out_path);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream ofs(out_path.c_str(), std::ios::out | std::ios::trunc);
    if (ofs.good()) {
      ofs << report;
    }
  }
}

struct LnProbeStats {
  float mean = 0.0f;
  float var_residual = 0.0f;
  float var_ex2 = 0.0f;
  float var_gap_abs = 0.0f;
  float var_gap_rel = 0.0f;
};

static inline float ln_probe_sanitize(float x) {
  return std::isfinite(x) ? x : 0.0f;
}

static LnProbeStats compute_ln_probe_stats(const float x[LN_DEBUG_D_MODEL]) {
  LnProbeStats s{};
  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int i = 0; i < LN_DEBUG_D_MODEL; ++i) {
    const float xv = ln_probe_sanitize(x[i]);
    sum += xv;
    sumsq += xv * xv;
  }
  s.mean = sum / static_cast<float>(LN_DEBUG_D_MODEL);
  const float ex2 = sumsq / static_cast<float>(LN_DEBUG_D_MODEL);
  float var_res_acc = 0.0f;
  for (int i = 0; i < LN_DEBUG_D_MODEL; ++i) {
    const float xv = ln_probe_sanitize(x[i]);
    const float d = xv - s.mean;
    var_res_acc += d * d;
  }
  s.var_residual = var_res_acc / static_cast<float>(LN_DEBUG_D_MODEL);
  s.var_ex2 = ex2 - (s.mean * s.mean);
  s.var_gap_abs = abs_diff_f(s.var_residual, s.var_ex2);
  s.var_gap_rel = rel_diff_f(s.var_residual, s.var_ex2);
  return s;
}

static void append_ln_input_token_details(std::ostringstream& oss, const LnInputTokenDiag& d) {
  oss << "[Pattern " << d.pattern << "] [Site " << aecct_ref::ref_ln_debug_site_name(d.site)
      << "] [Token " << d.token << "]\n";
  oss << "pairing_meta: entry_seq(base/exp)=" << d.entry_seq_base << " / " << d.entry_seq_exp
      << " site_call_index(base/exp)=" << d.site_call_index_base << " / " << d.site_call_index_exp << "\n";
  oss << "input_diff: max_abs=" << d.input_max_abs_diff
      << " mean_abs=" << d.input_mean_abs_diff
      << " mse=" << d.input_mse
      << " l2=" << d.input_l2 << "\n";
  oss << "var_diff: |raw|=" << d.var_raw_abs_diff
      << " |final|=" << d.var_final_abs_diff << "\n";
  oss << "same_input_probe(base): mean=" << d.base_probe_mean
      << " var_residual=" << d.base_probe_var_residual
      << " var_ex2=" << d.base_probe_var_ex2
      << " gap_abs=" << d.base_probe_var_gap_abs
      << " gap_rel=" << d.base_probe_var_gap_rel << "\n";
  oss << "same_input_probe(exp): mean=" << d.exp_probe_mean
      << " var_residual=" << d.exp_probe_var_residual
      << " var_ex2=" << d.exp_probe_var_ex2
      << " gap_abs=" << d.exp_probe_var_gap_abs
      << " gap_rel=" << d.exp_probe_var_gap_rel << "\n\n";
}

static void dump_ln_input_debug_report(
  int pattern_begin,
  int pattern_count,
  const std::vector<LnDebugEntryRow>& baseline_entries,
  const std::vector<LnDebugEntryRow>& experiment_entries,
  const std::string& out_path
) {
  std::unordered_map<std::uint64_t, std::size_t> exp_index;
  exp_index.reserve(experiment_entries.size());
  std::unordered_map<std::uint64_t, int> base_key_count;
  std::unordered_map<std::uint64_t, int> exp_key_count;
  base_key_count.reserve(baseline_entries.size());
  exp_key_count.reserve(experiment_entries.size());
  for (std::size_t i = 0; i < baseline_entries.size(); ++i) {
    base_key_count[make_ln_debug_key(
      baseline_entries[i].sample_index,
      baseline_entries[i].site,
      baseline_entries[i].token)] += 1;
  }
  for (std::size_t i = 0; i < experiment_entries.size(); ++i) {
    const std::uint64_t key = make_ln_debug_key(
      experiment_entries[i].sample_index,
      experiment_entries[i].site,
      experiment_entries[i].token);
    exp_key_count[key] += 1;
    exp_index[key] = i;
  }

  std::vector<LnInputTokenDiag> token_diags;
  token_diags.reserve(baseline_entries.size());
  std::size_t matched = 0U;
  std::size_t site_call_index_mismatch_count = 0U;
  std::size_t input_nonzero_count = 0U;
  double mean_input_max_abs_diff_sum = 0.0;
  double mean_input_mean_abs_diff_sum = 0.0;
  double mean_input_mse_sum = 0.0;
  double mean_base_probe_gap_abs_sum = 0.0;
  double mean_exp_probe_gap_abs_sum = 0.0;
  double mean_base_probe_gap_rel_sum = 0.0;
  double mean_exp_probe_gap_rel_sum = 0.0;

  for (std::size_t i = 0; i < baseline_entries.size(); ++i) {
    const LnDebugEntryRow& b = baseline_entries[i];
    const std::uint64_t key = make_ln_debug_key(b.sample_index, b.site, b.token);
    const auto it = exp_index.find(key);
    if (it == exp_index.end()) {
      continue;
    }
    const LnDebugEntryRow& e = experiment_entries[it->second];
    LnInputTokenDiag d{};
    d.pattern = pattern_begin + b.sample_index;
    d.site = b.site;
    d.token = b.token;
    d.entry_seq_base = b.entry_seq;
    d.entry_seq_exp = e.entry_seq;
    d.site_call_index_base = b.site_call_index;
    d.site_call_index_exp = e.site_call_index;

    double abs_sum = 0.0;
    double sq_sum = 0.0;
    double abs_max = 0.0;
    for (int k = 0; k < LN_DEBUG_D_MODEL; ++k) {
      const double ad = std::fabs(static_cast<double>(b.x[k]) - static_cast<double>(e.x[k]));
      abs_sum += ad;
      sq_sum += ad * ad;
      if (ad > abs_max) {
        abs_max = ad;
      }
    }
    d.input_max_abs_diff = static_cast<float>(abs_max);
    d.input_mean_abs_diff = static_cast<float>(abs_sum / static_cast<double>(LN_DEBUG_D_MODEL));
    d.input_mse = static_cast<float>(sq_sum / static_cast<double>(LN_DEBUG_D_MODEL));
    d.input_l2 = static_cast<float>(std::sqrt(sq_sum));
    d.var_raw_abs_diff = abs_diff_f(b.var_raw, e.var_raw);
    d.var_final_abs_diff = abs_diff_f(b.var_final, e.var_final);
    const LnProbeStats base_probe = compute_ln_probe_stats(b.x);
    const LnProbeStats exp_probe = compute_ln_probe_stats(e.x);
    d.base_probe_mean = base_probe.mean;
    d.base_probe_var_residual = base_probe.var_residual;
    d.base_probe_var_ex2 = base_probe.var_ex2;
    d.base_probe_var_gap_abs = base_probe.var_gap_abs;
    d.base_probe_var_gap_rel = base_probe.var_gap_rel;
    d.exp_probe_mean = exp_probe.mean;
    d.exp_probe_var_residual = exp_probe.var_residual;
    d.exp_probe_var_ex2 = exp_probe.var_ex2;
    d.exp_probe_var_gap_abs = exp_probe.var_gap_abs;
    d.exp_probe_var_gap_rel = exp_probe.var_gap_rel;
    token_diags.push_back(d);
    matched++;

    if (d.site_call_index_base != d.site_call_index_exp) {
      site_call_index_mismatch_count++;
    }
    if (d.input_max_abs_diff > 0.0f) {
      input_nonzero_count++;
    }
    mean_input_max_abs_diff_sum += d.input_max_abs_diff;
    mean_input_mean_abs_diff_sum += d.input_mean_abs_diff;
    mean_input_mse_sum += d.input_mse;
    mean_base_probe_gap_abs_sum += d.base_probe_var_gap_abs;
    mean_exp_probe_gap_abs_sum += d.exp_probe_var_gap_abs;
    mean_base_probe_gap_rel_sum += d.base_probe_var_gap_rel;
    mean_exp_probe_gap_rel_sum += d.exp_probe_var_gap_rel;
  }

  std::size_t base_duplicate_keys = 0U;
  for (auto it = base_key_count.begin(); it != base_key_count.end(); ++it) {
    if (it->second > 1) {
      base_duplicate_keys++;
    }
  }
  std::size_t exp_duplicate_keys = 0U;
  for (auto it = exp_key_count.begin(); it != exp_key_count.end(); ++it) {
    if (it->second > 1) {
      exp_duplicate_keys++;
    }
  }

  std::vector<int> pattern_first_div_site(static_cast<std::size_t>(pattern_count), -1);
  int first_div_site_hist[7] = {0, 0, 0, 0, 0, 0, 0}; // 0..5 site, 6 none
  const float first_div_tol = 1.0e-8f;
  for (std::size_t i = 0; i < token_diags.size(); ++i) {
    const LnInputTokenDiag& d = token_diags[i];
    const int off = d.pattern - pattern_begin;
    if (off < 0 || off >= pattern_count) {
      continue;
    }
    if (d.input_max_abs_diff > first_div_tol) {
      const int prev = pattern_first_div_site[static_cast<std::size_t>(off)];
      if (prev < 0 || d.site < prev) {
        pattern_first_div_site[static_cast<std::size_t>(off)] = d.site;
      }
    }
  }
  for (int i = 0; i < pattern_count; ++i) {
    const int s = pattern_first_div_site[static_cast<std::size_t>(i)];
    if (s >= 0 && s < 6) {
      first_div_site_hist[s] += 1;
    } else {
      first_div_site_hist[6] += 1;
    }
  }

  std::vector<LnInputTokenDiag> sorted_by_input = token_diags;
  std::sort(sorted_by_input.begin(), sorted_by_input.end(),
    [](const LnInputTokenDiag& a, const LnInputTokenDiag& b) {
      return a.input_max_abs_diff > b.input_max_abs_diff;
    });
  std::vector<LnInputTokenDiag> sorted_by_var = token_diags;
  std::sort(sorted_by_var.begin(), sorted_by_var.end(),
    [](const LnInputTokenDiag& a, const LnInputTokenDiag& b) {
      return a.var_final_abs_diff > b.var_final_abs_diff;
    });

  std::unordered_map<std::uint64_t, int> top_input_keys;
  for (std::size_t i = 0; i < std::min<std::size_t>(3U, sorted_by_input.size()); ++i) {
    top_input_keys[make_ln_debug_key(
      sorted_by_input[i].pattern - pattern_begin,
      sorted_by_input[i].site,
      sorted_by_input[i].token)] = 1;
  }
  int overlap_top3 = 0;
  for (std::size_t i = 0; i < std::min<std::size_t>(3U, sorted_by_var.size()); ++i) {
    const std::uint64_t key = make_ln_debug_key(
      sorted_by_var[i].pattern - pattern_begin,
      sorted_by_var[i].site,
      sorted_by_var[i].token);
    if (top_input_keys.find(key) != top_input_keys.end()) {
      overlap_top3++;
    }
  }

  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss << std::setprecision(9);
  oss << "=== LN Input Alignment Diagnosis ===\n";
  oss << "pairing_key_used: (pattern, site, token)\n";
  oss << "baseline_entries: " << baseline_entries.size() << "\n";
  oss << "experiment_entries: " << experiment_entries.size() << "\n";
  oss << "matched_entries: " << matched << "\n";
  oss << "pairing_audit_key_duplicates(base/exp): " << base_duplicate_keys
      << " / " << exp_duplicate_keys << "\n";
  oss << "site_call_index_mismatch_count: " << site_call_index_mismatch_count << "\n";
  if (matched > 0U) {
    const double inv = 1.0 / static_cast<double>(matched);
    oss << "input_nonzero_diff_count: " << input_nonzero_count << "\n";
    oss << "mean_input_max_abs_diff: " << (mean_input_max_abs_diff_sum * inv) << "\n";
    oss << "mean_input_mean_abs_diff: " << (mean_input_mean_abs_diff_sum * inv) << "\n";
    oss << "mean_input_mse: " << (mean_input_mse_sum * inv) << "\n";
    oss << "mean_same_input_probe_gap_abs(base): " << (mean_base_probe_gap_abs_sum * inv) << "\n";
    oss << "mean_same_input_probe_gap_abs(exp): " << (mean_exp_probe_gap_abs_sum * inv) << "\n";
    oss << "mean_same_input_probe_gap_rel(base): " << (mean_base_probe_gap_rel_sum * inv) << "\n";
    oss << "mean_same_input_probe_gap_rel(exp): " << (mean_exp_probe_gap_rel_sum * inv) << "\n";
  }
  oss << "first_upstream_divergence_site_histogram(per-pattern):\n";
  oss << "  L0_SUB0_LN: " << first_div_site_hist[0] << "\n";
  oss << "  L0_SUB1_LN: " << first_div_site_hist[1] << "\n";
  oss << "  MID_NORM: " << first_div_site_hist[2] << "\n";
  oss << "  L1_SUB0_LN: " << first_div_site_hist[3] << "\n";
  oss << "  L1_SUB1_LN: " << first_div_site_hist[4] << "\n";
  oss << "  END_NORM: " << first_div_site_hist[5] << "\n";
  oss << "  no_divergence: " << first_div_site_hist[6] << "\n";
  oss << "top3_overlap_count(input_diff_rank vs var_diff_rank): " << overlap_top3 << "\n\n";

  oss << "=== Top-3 Worst Tokens by LN Input Diff ===\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(3U, sorted_by_input.size()); ++i) {
    oss << "[Rank " << (i + 1U) << "]\n";
    append_ln_input_token_details(oss, sorted_by_input[i]);
  }

  oss << "=== Top-3 Worst Tokens by |var_final_base - var_final_exp| ===\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(3U, sorted_by_var.size()); ++i) {
    oss << "[Rank " << (i + 1U) << "]\n";
    append_ln_input_token_details(oss, sorted_by_var[i]);
  }

  const std::string report = oss.str();
  std::printf("%s", report.c_str());
  if (!out_path.empty()) {
    std::filesystem::path p(out_path);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream ofs(out_path.c_str(), std::ios::out | std::ios::trunc);
    if (ofs.good()) {
      ofs << report;
    }
  }
}

static inline std::uint64_t make_ln_upstream_key(int sample_index, int layer_idx, int boundary, int token) {
  return (static_cast<std::uint64_t>(static_cast<std::uint16_t>(sample_index)) << 48) |
         (static_cast<std::uint64_t>(static_cast<std::uint16_t>(layer_idx)) << 32) |
         (static_cast<std::uint64_t>(static_cast<std::uint16_t>(boundary)) << 16) |
         static_cast<std::uint64_t>(static_cast<std::uint16_t>(token));
}

static void append_ln_upstream_diag_line(std::ostringstream& oss, const LnUpstreamBoundaryDiag& d) {
  oss << "[Pattern " << d.pattern << "] [Token " << d.token << "] [Boundary "
      << aecct_ref::ref_ln_upstream_boundary_name(d.boundary) << "]\n";
  oss << "pairing_meta: entry_seq(base/exp)=" << d.entry_seq_base << " / " << d.entry_seq_exp
      << " call_index(base/exp)=" << d.call_index_base << " / " << d.call_index_exp << "\n";
  oss << "diff: max_abs=" << d.max_abs_diff
      << " mean_abs=" << d.mean_abs_diff
      << " mse=" << d.mse
      << " l2=" << d.l2 << "\n\n";
}

static void dump_ln_upstream_debug_report(
  int pattern_begin,
  int pattern_count,
  const std::vector<LnUpstreamEntryRow>& baseline_entries,
  const std::vector<LnUpstreamEntryRow>& experiment_entries,
  const std::string& out_path
) {
  std::unordered_map<std::uint64_t, std::size_t> exp_index;
  exp_index.reserve(experiment_entries.size());
  std::unordered_map<std::uint64_t, int> base_key_count;
  std::unordered_map<std::uint64_t, int> exp_key_count;
  base_key_count.reserve(baseline_entries.size());
  exp_key_count.reserve(experiment_entries.size());
  for (std::size_t i = 0; i < baseline_entries.size(); ++i) {
    const std::uint64_t key = make_ln_upstream_key(
      baseline_entries[i].sample_index,
      baseline_entries[i].layer_idx,
      baseline_entries[i].boundary,
      baseline_entries[i].token);
    base_key_count[key] += 1;
  }
  for (std::size_t i = 0; i < experiment_entries.size(); ++i) {
    const std::uint64_t key = make_ln_upstream_key(
      experiment_entries[i].sample_index,
      experiment_entries[i].layer_idx,
      experiment_entries[i].boundary,
      experiment_entries[i].token);
    exp_key_count[key] += 1;
    exp_index[key] = i;
  }

  std::vector<LnUpstreamBoundaryDiag> diags;
  diags.reserve(baseline_entries.size());
  std::size_t matched = 0U;
  std::size_t call_index_mismatch_count = 0U;
  std::size_t nonzero_diff_count = 0U;
  double mean_max_abs_sum = 0.0;
  double mean_mean_abs_sum = 0.0;
  double mean_mse_sum = 0.0;

  for (std::size_t i = 0; i < baseline_entries.size(); ++i) {
    const LnUpstreamEntryRow& b = baseline_entries[i];
    const std::uint64_t key = make_ln_upstream_key(b.sample_index, b.layer_idx, b.boundary, b.token);
    const auto it = exp_index.find(key);
    if (it == exp_index.end()) {
      continue;
    }
    const LnUpstreamEntryRow& e = experiment_entries[it->second];
    LnUpstreamBoundaryDiag d{};
    d.pattern = pattern_begin + b.sample_index;
    d.token = b.token;
    d.boundary = b.boundary;
    d.entry_seq_base = b.entry_seq;
    d.entry_seq_exp = e.entry_seq;
    d.call_index_base = b.boundary_call_index;
    d.call_index_exp = e.boundary_call_index;
    double abs_sum = 0.0;
    double sq_sum = 0.0;
    double abs_max = 0.0;
    for (int k = 0; k < LN_DEBUG_D_MODEL; ++k) {
      const double ad = std::fabs(static_cast<double>(b.x[k]) - static_cast<double>(e.x[k]));
      abs_sum += ad;
      sq_sum += ad * ad;
      if (ad > abs_max) {
        abs_max = ad;
      }
    }
    d.max_abs_diff = static_cast<float>(abs_max);
    d.mean_abs_diff = static_cast<float>(abs_sum / static_cast<double>(LN_DEBUG_D_MODEL));
    d.mse = static_cast<float>(sq_sum / static_cast<double>(LN_DEBUG_D_MODEL));
    d.l2 = static_cast<float>(std::sqrt(sq_sum));
    diags.push_back(d);
    matched++;

    if (d.call_index_base != d.call_index_exp) {
      call_index_mismatch_count++;
    }
    if (d.max_abs_diff > 0.0f) {
      nonzero_diff_count++;
    }
    mean_max_abs_sum += d.max_abs_diff;
    mean_mean_abs_sum += d.mean_abs_diff;
    mean_mse_sum += d.mse;
  }

  std::size_t base_duplicate_keys = 0U;
  for (auto it = base_key_count.begin(); it != base_key_count.end(); ++it) {
    if (it->second > 1) {
      base_duplicate_keys++;
    }
  }
  std::size_t exp_duplicate_keys = 0U;
  for (auto it = exp_key_count.begin(); it != exp_key_count.end(); ++it) {
    if (it->second > 1) {
      exp_duplicate_keys++;
    }
  }

  const int kBoundaryCount = 4;
  const int boundary_order[kBoundaryCount] = {0, 1, 2, 3};
  int earliest_hist[kBoundaryCount + 1] = {0, 0, 0, 0, 0}; // +1 no-div
  struct TokenBoundaryMetrics {
    float max_abs[kBoundaryCount] = {0.0f, 0.0f, 0.0f, 0.0f};
    int has[kBoundaryCount] = {0, 0, 0, 0};
  };
  std::unordered_map<std::uint64_t, TokenBoundaryMetrics> token_boundary;
  token_boundary.reserve(static_cast<std::size_t>(pattern_count * 75));
  for (std::size_t i = 0; i < diags.size(); ++i) {
    const LnUpstreamBoundaryDiag& d = diags[i];
    const int off = d.pattern - pattern_begin;
    if (off < 0 || off >= pattern_count) {
      continue;
    }
    if (d.boundary < 0 || d.boundary >= kBoundaryCount) {
      continue;
    }
    const std::uint64_t tk = (static_cast<std::uint64_t>(static_cast<std::uint16_t>(off)) << 16) |
                             static_cast<std::uint64_t>(static_cast<std::uint16_t>(d.token));
    TokenBoundaryMetrics& m = token_boundary[tk];
    m.max_abs[d.boundary] = d.max_abs_diff;
    m.has[d.boundary] = 1;
  }
  const float tol = 1.0e-8f;
  for (int off = 0; off < pattern_count; ++off) {
    for (int token = 0; token < 75; ++token) {
      const std::uint64_t tk = (static_cast<std::uint64_t>(static_cast<std::uint16_t>(off)) << 16) |
                               static_cast<std::uint64_t>(static_cast<std::uint16_t>(token));
      const auto it = token_boundary.find(tk);
      if (it == token_boundary.end()) {
        continue;
      }
      const TokenBoundaryMetrics& m = it->second;
      int earliest = -1;
      for (int bi = 0; bi < kBoundaryCount; ++bi) {
        const int b = boundary_order[bi];
        if (m.has[b] != 0 && m.max_abs[b] > tol) {
          earliest = b;
          break;
        }
      }
      if (earliest >= 0 && earliest < kBoundaryCount) {
        earliest_hist[earliest] += 1;
      } else {
        earliest_hist[kBoundaryCount] += 1;
      }
    }
  }

  std::vector<LnUpstreamBoundaryDiag> sorted_by_diff = diags;
  std::sort(sorted_by_diff.begin(), sorted_by_diff.end(),
    [](const LnUpstreamBoundaryDiag& a, const LnUpstreamBoundaryDiag& b) {
      return a.max_abs_diff > b.max_abs_diff;
    });

  std::vector<LnUpstreamBoundaryDiag> producer_boundary_diags;
  producer_boundary_diags.reserve(diags.size());
  for (std::size_t i = 0; i < diags.size(); ++i) {
    if (diags[i].boundary == 3) {
      producer_boundary_diags.push_back(diags[i]);
    }
  }
  std::sort(producer_boundary_diags.begin(), producer_boundary_diags.end(),
    [](const LnUpstreamBoundaryDiag& a, const LnUpstreamBoundaryDiag& b) {
      return a.max_abs_diff > b.max_abs_diff;
    });

  std::vector<LnUpstreamBoundaryDiag> assembly_sum_diags;
  std::vector<LnUpstreamBoundaryDiag> assembly_out_diags;
  for (std::size_t i = 0; i < diags.size(); ++i) {
    if (diags[i].boundary == 2) {
      assembly_sum_diags.push_back(diags[i]);
    } else if (diags[i].boundary == 3) {
      assembly_out_diags.push_back(diags[i]);
    }
  }
  std::unordered_map<std::uint64_t, std::size_t> assembly_sum_idx;
  assembly_sum_idx.reserve(assembly_sum_diags.size());
  for (std::size_t i = 0; i < assembly_sum_diags.size(); ++i) {
    const std::uint64_t key = make_ln_upstream_key(
      assembly_sum_diags[i].pattern - pattern_begin, 0, 2, assembly_sum_diags[i].token);
    assembly_sum_idx[key] = i;
  }
  double assembly_out_minus_sum_mean = 0.0;
  std::size_t assembly_out_more_count = 0U;
  std::size_t assembly_pair_count = 0U;
  for (std::size_t i = 0; i < assembly_out_diags.size(); ++i) {
    const std::uint64_t key = make_ln_upstream_key(
      assembly_out_diags[i].pattern - pattern_begin, 0, 2, assembly_out_diags[i].token);
    const auto it = assembly_sum_idx.find(key);
    if (it == assembly_sum_idx.end()) {
      continue;
    }
    const float delta = assembly_out_diags[i].max_abs_diff - assembly_sum_diags[it->second].max_abs_diff;
    assembly_out_minus_sum_mean += delta;
    if (delta > 1.0e-8f) {
      assembly_out_more_count++;
    }
    assembly_pair_count++;
  }

  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss << std::setprecision(9);
  oss << "=== LN Upstream Divergence Diagnosis (L0_SUB1 path) ===\n";
  oss << "pairing_key_used: (pattern, layer_idx, boundary, token)\n";
  oss << "baseline_entries: " << baseline_entries.size() << "\n";
  oss << "experiment_entries: " << experiment_entries.size() << "\n";
  oss << "matched_entries: " << matched << "\n";
  oss << "pairing_audit_key_duplicates(base/exp): " << base_duplicate_keys << " / "
      << exp_duplicate_keys << "\n";
  oss << "call_index_mismatch_count: " << call_index_mismatch_count << "\n";
  if (matched > 0U) {
    const double inv = 1.0 / static_cast<double>(matched);
    oss << "nonzero_diff_count: " << nonzero_diff_count << "\n";
    oss << "mean_max_abs_diff: " << (mean_max_abs_sum * inv) << "\n";
    oss << "mean_mean_abs_diff: " << (mean_mean_abs_sum * inv) << "\n";
    oss << "mean_mse: " << (mean_mse_sum * inv) << "\n";
  }
  if (assembly_pair_count > 0U) {
    const double inv = 1.0 / static_cast<double>(assembly_pair_count);
    oss << "assembly_out_minus_sum_mean(max_abs): " << (assembly_out_minus_sum_mean * inv) << "\n";
    oss << "assembly_out_more_count: " << assembly_out_more_count << "\n";
  }
  oss << "boundary_order_map:\n";
  for (int b = 0; b < kBoundaryCount; ++b) {
    oss << "  [" << b << "] " << aecct_ref::ref_ln_upstream_boundary_name(b) << "\n";
  }
  oss << "earliest_divergence_boundary_histogram:\n";
  for (int b = 0; b < kBoundaryCount; ++b) {
    oss << "  " << aecct_ref::ref_ln_upstream_boundary_name(b) << ": " << earliest_hist[b] << "\n";
  }
  oss << "  no_divergence: " << earliest_hist[kBoundaryCount] << "\n\n";

  oss << "=== Top-3 Worst Boundary Entries (global) ===\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(3U, sorted_by_diff.size()); ++i) {
    oss << "[Rank " << (i + 1U) << "]\n";
    append_ln_upstream_diag_line(oss, sorted_by_diff[i]);
  }

  oss << "=== Top-3 Worst Producer Boundary Entries (ffn_ln_in_residual_ffn) ===\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(3U, producer_boundary_diags.size()); ++i) {
    oss << "[Rank " << (i + 1U) << "]\n";
    append_ln_upstream_diag_line(oss, producer_boundary_diags[i]);
  }

  const std::string report = oss.str();
  std::printf("%s", report.c_str());
  if (!out_path.empty()) {
    std::filesystem::path p(out_path);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream ofs(out_path.c_str(), std::ios::out | std::ios::trunc);
    if (ofs.good()) {
      ofs << report;
    }
  }
}

static bool write_compare_csv(const std::string& path, const std::vector<PerPatternCompareRow>& rows) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }

  std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
  if (!ofs.good()) {
    return false;
  }

  ofs << "pattern"
      << ",logits_mse"
      << ",logits_rmse"
      << ",logits_mae"
      << ",logits_max_abs_diff"
      << ",x_pred_mismatch_count"
      << ",x_pred_mismatch_ratio"
      << ",baseline_min_abs_margin"
      << ",experiment_min_abs_margin"
      << ",sign_flip_count"
      << ",baseline_nan_count"
      << ",baseline_inf_count"
      << ",experiment_nan_count"
      << ",experiment_inf_count"
      << ",baseline_vs_golden_mse"
      << ",experiment_vs_golden_mse"
      << ",baseline_topk_smallest_abs"
      << ",experiment_topk_smallest_abs"
      << "\n";

  ofs.setf(std::ios::scientific);
  ofs << std::setprecision(9);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const PerPatternCompareRow& r = rows[i];
    ofs << r.pattern_index
        << "," << r.cmp.logits_diff.mse
        << "," << r.cmp.logits_diff.rmse
        << "," << r.cmp.logits_diff.mae
        << "," << r.cmp.logits_diff.max_abs
        << "," << r.cmp.x_pred_mismatch_count
        << "," << r.cmp.x_pred_mismatch_ratio
        << "," << r.cmp.baseline_min_abs_margin
        << "," << r.cmp.experiment_min_abs_margin
        << "," << r.cmp.sign_flip_count
        << "," << r.cmp.baseline_logits_nonfinite.nan_count
        << "," << r.cmp.baseline_logits_nonfinite.inf_count
        << "," << r.cmp.experiment_logits_nonfinite.nan_count
        << "," << r.cmp.experiment_logits_nonfinite.inf_count
        << "," << r.baseline_vs_golden.mse
        << "," << r.experiment_vs_golden.mse
        << ",\"" << format_topk_entries(r.cmp.baseline_topk_smallest_abs) << "\""
        << ",\"" << format_topk_entries(r.cmp.experiment_topk_smallest_abs) << "\""
        << "\n";
  }
  return ofs.good();
}

static bool write_eval_single_csv(const std::string& path, const std::vector<EvalPatternRow>& rows) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
  if (!ofs.good()) {
    return false;
  }

  ofs << "pattern"
      << ",evaluated_bits"
      << ",bit_errors"
      << ",ber"
      << ",x_pred_match_count"
      << ",x_pred_match_ratio"
      << ",frame_error_flag"
      << "\n";
  ofs.setf(std::ios::scientific);
  ofs << std::setprecision(9);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const EvalPatternRow& r = rows[i];
    const double ber = (r.evaluated_bits > 0U)
      ? (static_cast<double>(r.bit_errors) / static_cast<double>(r.evaluated_bits))
      : 0.0;
    const double match_ratio = (r.evaluated_bits > 0U)
      ? (static_cast<double>(r.x_pred_match_count) / static_cast<double>(r.evaluated_bits))
      : 0.0;
    ofs << r.pattern_index
        << "," << r.evaluated_bits
        << "," << r.bit_errors
        << "," << ber
        << "," << r.x_pred_match_count
        << "," << match_ratio
        << "," << r.frame_error_flag
        << "\n";
  }
  return ofs.good();
}

static bool write_eval_compare_csv(const std::string& path, const std::vector<EvalComparePatternRow>& rows) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
  if (!ofs.good()) {
    return false;
  }

  ofs << "pattern"
      << ",evaluated_bits"
      << ",baseline_bit_errors"
      << ",baseline_ber"
      << ",baseline_x_pred_match_count"
      << ",baseline_x_pred_match_ratio"
      << ",baseline_frame_error_flag"
      << ",experiment_bit_errors"
      << ",experiment_ber"
      << ",experiment_x_pred_match_count"
      << ",experiment_x_pred_match_ratio"
      << ",experiment_frame_error_flag"
      << ",delta_bit_errors"
      << "\n";
  ofs.setf(std::ios::scientific);
  ofs << std::setprecision(9);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const EvalComparePatternRow& r = rows[i];
    const double b_ber = (r.evaluated_bits > 0U)
      ? (static_cast<double>(r.baseline_bit_errors) / static_cast<double>(r.evaluated_bits))
      : 0.0;
    const double e_ber = (r.evaluated_bits > 0U)
      ? (static_cast<double>(r.experiment_bit_errors) / static_cast<double>(r.evaluated_bits))
      : 0.0;
    const double b_match = (r.evaluated_bits > 0U)
      ? (static_cast<double>(r.baseline_x_pred_match_count) / static_cast<double>(r.evaluated_bits))
      : 0.0;
    const double e_match = (r.evaluated_bits > 0U)
      ? (static_cast<double>(r.experiment_x_pred_match_count) / static_cast<double>(r.evaluated_bits))
      : 0.0;
    const std::size_t delta_bit_errors = (r.experiment_bit_errors >= r.baseline_bit_errors)
      ? (r.experiment_bit_errors - r.baseline_bit_errors)
      : (r.baseline_bit_errors - r.experiment_bit_errors);
    ofs << r.pattern_index
        << "," << r.evaluated_bits
        << "," << r.baseline_bit_errors
        << "," << b_ber
        << "," << r.baseline_x_pred_match_count
        << "," << b_match
        << "," << r.baseline_frame_error_flag
        << "," << r.experiment_bit_errors
        << "," << e_ber
        << "," << r.experiment_x_pred_match_count
        << "," << e_match
        << "," << r.experiment_frame_error_flag
        << "," << delta_bit_errors
        << "\n";
  }
  return ofs.good();
}

static bool write_eval_single_summary_txt(
  const std::string& path,
  const char* tag,
  const EvalAggregateStats& stats,
  const std::vector<EvalPatternRow>& rows,
  const aecct_ref::RefFullQuantStats* full_stats = nullptr
) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
  if (!ofs.good()) {
    return false;
  }
  ofs.setf(std::ios::scientific);
  ofs << std::setprecision(9);
  ofs << "=== Evaluator Summary (" << tag << ") ===\n";
  ofs << "total patterns             : " << stats.total_patterns << "\n";
  ofs << "total evaluated bits       : " << stats.total_bits << "\n";
  ofs << "total bit errors           : " << stats.total_bit_errors << "\n";
  ofs << "BER                        : " << stats.ber << "\n";
  ofs << "frame error count          : " << stats.frame_error_count << "\n";
  ofs << "FER                        : " << stats.fer << "\n";
  ofs << "x_pred/target match count  : " << stats.x_pred_match_count << "\n";
  ofs << "x_pred/target match ratio  : " << stats.x_pred_match_ratio << "\n";
  if (full_stats != nullptr) {
    ofs << "\n";
    ofs << "=== Quant Stress Counters (experiment) ===\n";
    ofs << "int8 clamp count           : " << full_stats->int_linear.int8_clamp_count << "\n";
    ofs << "int16 overflow count       : " << full_stats->int_linear.int16_overflow_count << "\n";
    ofs << "dequant restore count      : " << full_stats->int_linear.dequant_restore_count << "\n";
    ofs << "e4m3 roundtrip count       : " << full_stats->e4m3.roundtrip_count << "\n";
    ofs << "e4m3 roundtrip g1/g2/g3/g4/g5 : "
        << full_stats->e4m3.roundtrip_g1_count << "/"
        << full_stats->e4m3.roundtrip_g2_count << "/"
        << full_stats->e4m3.roundtrip_g3_count << "/"
        << full_stats->e4m3.roundtrip_g4_count << "/"
        << full_stats->e4m3.roundtrip_g5_count << "\n";
    ofs << "e4m3 roundtrip g5 sub (embed/spe/assembly/prelayer) : "
        << full_stats->e4m3.roundtrip_g5_embed_count << "/"
        << full_stats->e4m3.roundtrip_g5_spe_count << "/"
        << full_stats->e4m3.roundtrip_g5_preproc_assembly_count << "/"
        << full_stats->e4m3.roundtrip_g5_prelayer_handoff_count << "\n";
    ofs << "e4m3 nan in/out            : " << full_stats->e4m3.nan_in_count
        << "/" << full_stats->e4m3.nan_out_count << "\n";
    ofs << "e4m3 inf in/out            : " << full_stats->e4m3.inf_in_count
        << "/" << full_stats->e4m3.inf_out_count << "\n";
    ofs << "first nonfinite block      : "
        << (full_stats->e4m3.first_nonfinite_block.empty()
            ? "none" : full_stats->e4m3.first_nonfinite_block)
        << "\n";
    ofs << "first int16 overflow block : "
        << (full_stats->int_linear.first_int16_overflow_block.empty()
            ? "none" : full_stats->int_linear.first_int16_overflow_block)
        << "\n";
    ofs << "int8 fixedexp roundtrip    : " << full_stats->int8_fixedexp.roundtrip_count << "\n";
    ofs << "int8 fixedexp clamp count  : " << full_stats->int8_fixedexp.clamp_count << "\n";
    ofs << "int8 fixedexp zone z1/z2/z3/z4 : "
        << full_stats->int8_fixedexp.zone1_count << "/"
        << full_stats->int8_fixedexp.zone2_count << "/"
        << full_stats->int8_fixedexp.zone3_count << "/"
        << full_stats->int8_fixedexp.zone4_count << "\n";
    ofs << "int8 fixedexp footprint g2/g5_embed : "
        << full_stats->int8_fixedexp.footprint_g2_count << "/"
        << full_stats->int8_fixedexp.footprint_g5_embed_count << "\n";
    ofs << "int8 fixedexp first clamp block : "
        << (full_stats->int8_fixedexp.first_clamp_block.empty()
            ? "none" : full_stats->int8_fixedexp.first_clamp_block)
        << "\n";
    ofs << "fp16 roundtrip count      : " << full_stats->fp16.roundtrip_count << "\n";
    ofs << "fp16 nan in/out           : " << full_stats->fp16.nan_in_count
        << "/" << full_stats->fp16.nan_out_count << "\n";
    ofs << "fp16 inf in/out           : " << full_stats->fp16.inf_in_count
        << "/" << full_stats->fp16.inf_out_count << "\n";
    ofs << "fp16 underflow->zero      : " << full_stats->fp16.underflow_to_zero_count << "\n";
    ofs << "fp16 first nonfinite block: "
        << (full_stats->fp16.first_nonfinite_block.empty()
            ? "none" : full_stats->fp16.first_nonfinite_block)
        << "\n";
  }
  ofs << "\n";
  ofs << "per-pattern bit/frame summary:\n";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const EvalPatternRow& r = rows[i];
    ofs << "  pattern=" << r.pattern_index
        << " bits=" << r.evaluated_bits
        << " bit_errors=" << r.bit_errors
        << " frame_error=" << r.frame_error_flag
        << " xmatch=" << r.x_pred_match_count
        << "\n";
  }
  return ofs.good();
}

static bool write_eval_compare_summary_txt(
  const std::string& path,
  const EvalAggregateStats& baseline_stats,
  const EvalAggregateStats& experiment_stats,
  const std::vector<EvalComparePatternRow>& rows,
  const aecct_ref::RefFullQuantStats* experiment_full_stats = nullptr
) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
  if (!ofs.good()) {
    return false;
  }
  const double delta_ber = experiment_stats.ber - baseline_stats.ber;
  const double delta_fer = experiment_stats.fer - baseline_stats.fer;
  ofs.setf(std::ios::scientific);
  ofs << std::setprecision(9);
  ofs << "=== Evaluator Compare Summary ===\n";
  ofs << "total patterns             : " << baseline_stats.total_patterns << "\n";
  ofs << "total evaluated bits       : " << baseline_stats.total_bits << "\n";
  ofs << "baseline total bit errors  : " << baseline_stats.total_bit_errors << "\n";
  ofs << "baseline BER               : " << baseline_stats.ber << "\n";
  ofs << "baseline frame errors      : " << baseline_stats.frame_error_count << "\n";
  ofs << "baseline FER               : " << baseline_stats.fer << "\n";
  ofs << "experiment total bit errors: " << experiment_stats.total_bit_errors << "\n";
  ofs << "experiment BER             : " << experiment_stats.ber << "\n";
  ofs << "experiment frame errors    : " << experiment_stats.frame_error_count << "\n";
  ofs << "experiment FER             : " << experiment_stats.fer << "\n";
  ofs << "delta BER (exp-base)       : " << delta_ber << "\n";
  ofs << "delta FER (exp-base)       : " << delta_fer << "\n";
  ofs << "baseline x_pred match ratio: " << baseline_stats.x_pred_match_ratio << "\n";
  ofs << "experiment x_pred match ratio: " << experiment_stats.x_pred_match_ratio << "\n";
  if (experiment_full_stats != nullptr) {
    ofs << "\n";
    ofs << "=== Quant Stress Counters (experiment) ===\n";
    ofs << "int8 clamp count           : " << experiment_full_stats->int_linear.int8_clamp_count << "\n";
    ofs << "int16 overflow count       : " << experiment_full_stats->int_linear.int16_overflow_count << "\n";
    ofs << "dequant restore count      : " << experiment_full_stats->int_linear.dequant_restore_count << "\n";
    ofs << "e4m3 roundtrip count       : " << experiment_full_stats->e4m3.roundtrip_count << "\n";
    ofs << "e4m3 roundtrip g1/g2/g3/g4/g5 : "
        << experiment_full_stats->e4m3.roundtrip_g1_count << "/"
        << experiment_full_stats->e4m3.roundtrip_g2_count << "/"
        << experiment_full_stats->e4m3.roundtrip_g3_count << "/"
        << experiment_full_stats->e4m3.roundtrip_g4_count << "/"
        << experiment_full_stats->e4m3.roundtrip_g5_count << "\n";
    ofs << "e4m3 roundtrip g5 sub (embed/spe/assembly/prelayer) : "
        << experiment_full_stats->e4m3.roundtrip_g5_embed_count << "/"
        << experiment_full_stats->e4m3.roundtrip_g5_spe_count << "/"
        << experiment_full_stats->e4m3.roundtrip_g5_preproc_assembly_count << "/"
        << experiment_full_stats->e4m3.roundtrip_g5_prelayer_handoff_count << "\n";
    ofs << "e4m3 nan in/out            : " << experiment_full_stats->e4m3.nan_in_count
        << "/" << experiment_full_stats->e4m3.nan_out_count << "\n";
    ofs << "e4m3 inf in/out            : " << experiment_full_stats->e4m3.inf_in_count
        << "/" << experiment_full_stats->e4m3.inf_out_count << "\n";
    ofs << "first nonfinite block      : "
        << (experiment_full_stats->e4m3.first_nonfinite_block.empty()
            ? "none" : experiment_full_stats->e4m3.first_nonfinite_block)
        << "\n";
    ofs << "first int16 overflow block : "
        << (experiment_full_stats->int_linear.first_int16_overflow_block.empty()
            ? "none" : experiment_full_stats->int_linear.first_int16_overflow_block)
        << "\n";
    ofs << "int8 fixedexp roundtrip    : " << experiment_full_stats->int8_fixedexp.roundtrip_count << "\n";
    ofs << "int8 fixedexp clamp count  : " << experiment_full_stats->int8_fixedexp.clamp_count << "\n";
    ofs << "int8 fixedexp zone z1/z2/z3/z4 : "
        << experiment_full_stats->int8_fixedexp.zone1_count << "/"
        << experiment_full_stats->int8_fixedexp.zone2_count << "/"
        << experiment_full_stats->int8_fixedexp.zone3_count << "/"
        << experiment_full_stats->int8_fixedexp.zone4_count << "\n";
    ofs << "int8 fixedexp footprint g2/g5_embed : "
        << experiment_full_stats->int8_fixedexp.footprint_g2_count << "/"
        << experiment_full_stats->int8_fixedexp.footprint_g5_embed_count << "\n";
    ofs << "int8 fixedexp first clamp block : "
        << (experiment_full_stats->int8_fixedexp.first_clamp_block.empty()
            ? "none" : experiment_full_stats->int8_fixedexp.first_clamp_block)
        << "\n";
    ofs << "fp16 roundtrip count      : " << experiment_full_stats->fp16.roundtrip_count << "\n";
    ofs << "fp16 nan in/out           : " << experiment_full_stats->fp16.nan_in_count
        << "/" << experiment_full_stats->fp16.nan_out_count << "\n";
    ofs << "fp16 inf in/out           : " << experiment_full_stats->fp16.inf_in_count
        << "/" << experiment_full_stats->fp16.inf_out_count << "\n";
    ofs << "fp16 underflow->zero      : " << experiment_full_stats->fp16.underflow_to_zero_count << "\n";
    ofs << "fp16 first nonfinite block: "
        << (experiment_full_stats->fp16.first_nonfinite_block.empty()
            ? "none" : experiment_full_stats->fp16.first_nonfinite_block)
        << "\n";
  }
  ofs << "\n";
  ofs << "per-pattern bit/frame summary:\n";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const EvalComparePatternRow& r = rows[i];
    ofs << "  pattern=" << r.pattern_index
        << " bits=" << r.evaluated_bits
        << " baseline_bit_errors=" << r.baseline_bit_errors
        << " experiment_bit_errors=" << r.experiment_bit_errors
        << " baseline_frame_error=" << r.baseline_frame_error_flag
        << " experiment_frame_error=" << r.experiment_frame_error_flag
        << " baseline_xmatch=" << r.baseline_x_pred_match_count
        << " experiment_xmatch=" << r.experiment_x_pred_match_count
        << "\n";
  }
  return ofs.good();
}

static void init_batch_summary(BatchCompareSummary& s) {
  s.total_patterns_scanned = 0;
  s.patterns_with_xpred_flip = 0;
  s.patterns_with_sign_flip = 0;
  s.total_flipped_bits = 0;
  s.worst_logits_mse = -1.0;
  s.worst_logits_mse_pattern = -1;
  s.worst_max_abs_diff = -1.0;
  s.worst_max_abs_diff_pattern = -1;
  s.worst_min_margin = std::numeric_limits<double>::infinity();
  s.worst_min_margin_pattern = -1;
  s.max_sign_flip_count = 0;
  s.max_sign_flip_pattern = -1;
  s.total_sign_flip_count = 0;
  s.per_pattern_min_margin.clear();
  s.baseline_nonfinite_total = {};
  s.experiment_nonfinite_total = {};
}

static void update_batch_summary(BatchCompareSummary& s, const PerPatternCompareRow& row) {
  s.total_patterns_scanned += 1;
  s.total_flipped_bits += row.cmp.x_pred_mismatch_count;
  if (row.cmp.x_pred_mismatch_count > 0U) {
    s.patterns_with_xpred_flip += 1;
  }
  if (row.cmp.sign_flip_count > 0U) {
    s.patterns_with_sign_flip += 1;
  }
  if (row.cmp.logits_diff.mse > s.worst_logits_mse) {
    s.worst_logits_mse = row.cmp.logits_diff.mse;
    s.worst_logits_mse_pattern = row.pattern_index;
  }
  if (row.cmp.logits_diff.max_abs > s.worst_max_abs_diff) {
    s.worst_max_abs_diff = row.cmp.logits_diff.max_abs;
    s.worst_max_abs_diff_pattern = row.pattern_index;
  }

  const double row_min_margin =
    (row.cmp.baseline_min_abs_margin < row.cmp.experiment_min_abs_margin)
      ? row.cmp.baseline_min_abs_margin
      : row.cmp.experiment_min_abs_margin;
  if (row_min_margin < s.worst_min_margin) {
    s.worst_min_margin = row_min_margin;
    s.worst_min_margin_pattern = row.pattern_index;
  }
  s.per_pattern_min_margin.push_back(row_min_margin);

  if (row.cmp.sign_flip_count > s.max_sign_flip_count) {
    s.max_sign_flip_count = row.cmp.sign_flip_count;
    s.max_sign_flip_pattern = row.pattern_index;
  }
  s.total_sign_flip_count += row.cmp.sign_flip_count;

  s.baseline_nonfinite_total.nan_count += row.cmp.baseline_logits_nonfinite.nan_count;
  s.baseline_nonfinite_total.inf_count += row.cmp.baseline_logits_nonfinite.inf_count;
  s.experiment_nonfinite_total.nan_count += row.cmp.experiment_logits_nonfinite.nan_count;
  s.experiment_nonfinite_total.inf_count += row.cmp.experiment_logits_nonfinite.inf_count;
}

static void init_golden_aggregate(GoldenAggregateMetrics& g) {
  g.se_sum = 0.0;
  g.ae_sum = 0.0;
  g.max_abs = 0.0;
  g.logits_count = 0;
  g.x_pred_match_count = 0;
  g.x_pred_total_count = 0;
}

static void update_golden_aggregate(GoldenAggregateMetrics& g, const RefRunOutputs& run) {
  const std::size_t n = run.logits.size();
  g.se_sum += run.logits_vs_golden.mse * static_cast<double>(n);
  g.ae_sum += run.logits_vs_golden.mae * static_cast<double>(n);
  if (run.logits_vs_golden.max_abs > g.max_abs) {
    g.max_abs = run.logits_vs_golden.max_abs;
  }
  g.logits_count += n;
  g.x_pred_match_count += run.x_pred_match_count;
  g.x_pred_total_count += run.x_pred_total_count;
}

static aecct_ref::Metrics finalize_golden_aggregate(const GoldenAggregateMetrics& g) {
  aecct_ref::Metrics m{};
  if (g.logits_count > 0U) {
    m.mse = g.se_sum / static_cast<double>(g.logits_count);
    m.mae = g.ae_sum / static_cast<double>(g.logits_count);
    m.rmse = std::sqrt(m.mse);
    m.max_abs = g.max_abs;
  }
  return m;
}

static DistributionStats compute_distribution_stats(const std::vector<double>& values) {
  DistributionStats s{};
  s.min_v = 0.0;
  s.max_v = 0.0;
  s.mean_v = 0.0;
  s.median_v = 0.0;
  if (values.empty()) {
    return s;
  }

  s.min_v = values[0];
  s.max_v = values[0];
  double sum = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double v = values[i];
    if (v < s.min_v) s.min_v = v;
    if (v > s.max_v) s.max_v = v;
    sum += v;
  }
  s.mean_v = sum / static_cast<double>(values.size());

  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t mid = sorted.size() / 2U;
  if ((sorted.size() & 1U) != 0U) {
    s.median_v = sorted[mid];
  } else {
    s.median_v = 0.5 * (sorted[mid - 1U] + sorted[mid]);
  }
  return s;
}

static double pattern_risk_margin(const PerPatternCompareRow& row) {
  return (row.cmp.baseline_min_abs_margin < row.cmp.experiment_min_abs_margin)
    ? row.cmp.baseline_min_abs_margin
    : row.cmp.experiment_min_abs_margin;
}

static std::vector<std::size_t> collect_top_vulnerable_indices(
  const std::vector<PerPatternCompareRow>& rows,
  std::size_t topk
) {
  std::vector<std::size_t> idx;
  idx.reserve(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    idx.push_back(i);
  }
  std::sort(idx.begin(), idx.end(),
    [&rows](std::size_t a, std::size_t b) {
      const double ma = pattern_risk_margin(rows[a]);
      const double mb = pattern_risk_margin(rows[b]);
      if (ma != mb) {
        return ma < mb;
      }
      if (rows[a].cmp.logits_diff.mse != rows[b].cmp.logits_diff.mse) {
        return rows[a].cmp.logits_diff.mse > rows[b].cmp.logits_diff.mse;
      }
      return rows[a].pattern_index < rows[b].pattern_index;
    });
  if (idx.size() > topk) {
    idx.resize(topk);
  }
  return idx;
}

static std::string derive_summary_txt_path(const std::string& csv_path) {
  if (csv_path.size() >= 4U && csv_path.substr(csv_path.size() - 4U) == ".csv") {
    return csv_path.substr(0, csv_path.size() - 4U) + ".txt";
  }
  return csv_path + ".txt";
}

static bool write_batch_summary_txt(
  const std::string& path,
  const BatchCompareSummary& batch,
  const DistributionStats& margin_dist,
  const std::vector<PerPatternCompareRow>& rows,
  const std::vector<std::size_t>& vulnerable_idx,
  const aecct_ref::RefFullQuantStats* experiment_full_stats = nullptr
) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
  if (!ofs.good()) {
    return false;
  }

  ofs.setf(std::ios::scientific);
  ofs << std::setprecision(9);
  ofs << "=== Batch Compare Summary ===\n";
  ofs << "total patterns scanned        : " << batch.total_patterns_scanned << "\n";
  ofs << "patterns with x_pred flip     : " << batch.patterns_with_xpred_flip << "\n";
  ofs << "patterns with sign flip       : " << batch.patterns_with_sign_flip << "\n";
  ofs << "total flipped bits            : " << batch.total_flipped_bits << "\n";
  ofs << "worst logits MSE              : " << batch.worst_logits_mse
      << " (pattern " << batch.worst_logits_mse_pattern << ")\n";
  ofs << "worst max abs diff            : " << batch.worst_max_abs_diff
      << " (pattern " << batch.worst_max_abs_diff_pattern << ")\n";
  ofs << "worst min margin              : " << batch.worst_min_margin
      << " (pattern " << batch.worst_min_margin_pattern << ")\n";
  ofs << "max sign-flip count           : " << batch.max_sign_flip_count
      << " (pattern " << batch.max_sign_flip_pattern << ")\n";
  ofs << "total sign flips              : " << batch.total_sign_flip_count << "\n";
  ofs << "total baseline NaN / Inf      : "
      << batch.baseline_nonfinite_total.nan_count << " / "
      << batch.baseline_nonfinite_total.inf_count << "\n";
  ofs << "total experiment NaN / Inf    : "
      << batch.experiment_nonfinite_total.nan_count << " / "
      << batch.experiment_nonfinite_total.inf_count << "\n";
  if (experiment_full_stats != nullptr) {
    ofs << "\n";
    ofs << "=== Quant Stress Counters (experiment) ===\n";
    ofs << "int8 clamp count              : " << experiment_full_stats->int_linear.int8_clamp_count << "\n";
    ofs << "int16 overflow count          : " << experiment_full_stats->int_linear.int16_overflow_count << "\n";
    ofs << "dequant restore count         : " << experiment_full_stats->int_linear.dequant_restore_count << "\n";
    ofs << "e4m3 roundtrip count          : " << experiment_full_stats->e4m3.roundtrip_count << "\n";
    ofs << "e4m3 roundtrip g1/g2/g3/g4/g5 : "
        << experiment_full_stats->e4m3.roundtrip_g1_count << " / "
        << experiment_full_stats->e4m3.roundtrip_g2_count << " / "
        << experiment_full_stats->e4m3.roundtrip_g3_count << " / "
        << experiment_full_stats->e4m3.roundtrip_g4_count << " / "
        << experiment_full_stats->e4m3.roundtrip_g5_count << "\n";
    ofs << "e4m3 roundtrip g5 sub (embed/spe/assembly/prelayer) : "
        << experiment_full_stats->e4m3.roundtrip_g5_embed_count << " / "
        << experiment_full_stats->e4m3.roundtrip_g5_spe_count << " / "
        << experiment_full_stats->e4m3.roundtrip_g5_preproc_assembly_count << " / "
        << experiment_full_stats->e4m3.roundtrip_g5_prelayer_handoff_count << "\n";
    ofs << "e4m3 nan in/out               : " << experiment_full_stats->e4m3.nan_in_count
        << " / " << experiment_full_stats->e4m3.nan_out_count << "\n";
    ofs << "e4m3 inf in/out               : " << experiment_full_stats->e4m3.inf_in_count
        << " / " << experiment_full_stats->e4m3.inf_out_count << "\n";
    ofs << "first nonfinite block         : "
        << (experiment_full_stats->e4m3.first_nonfinite_block.empty()
            ? "none" : experiment_full_stats->e4m3.first_nonfinite_block)
        << "\n";
    ofs << "first int16 overflow block    : "
        << (experiment_full_stats->int_linear.first_int16_overflow_block.empty()
            ? "none" : experiment_full_stats->int_linear.first_int16_overflow_block)
        << "\n";
    ofs << "int8 fixedexp roundtrip       : " << experiment_full_stats->int8_fixedexp.roundtrip_count << "\n";
    ofs << "int8 fixedexp clamp count     : " << experiment_full_stats->int8_fixedexp.clamp_count << "\n";
    ofs << "int8 fixedexp zone z1/z2/z3/z4 : "
        << experiment_full_stats->int8_fixedexp.zone1_count << " / "
        << experiment_full_stats->int8_fixedexp.zone2_count << " / "
        << experiment_full_stats->int8_fixedexp.zone3_count << " / "
        << experiment_full_stats->int8_fixedexp.zone4_count << "\n";
    ofs << "int8 fixedexp footprint g2/g5_embed : "
        << experiment_full_stats->int8_fixedexp.footprint_g2_count << " / "
        << experiment_full_stats->int8_fixedexp.footprint_g5_embed_count << "\n";
    ofs << "int8 fixedexp first clamp block : "
        << (experiment_full_stats->int8_fixedexp.first_clamp_block.empty()
            ? "none" : experiment_full_stats->int8_fixedexp.first_clamp_block)
        << "\n";
    ofs << "fp16 roundtrip count        : " << experiment_full_stats->fp16.roundtrip_count << "\n";
    ofs << "fp16 nan in/out             : " << experiment_full_stats->fp16.nan_in_count
        << " / " << experiment_full_stats->fp16.nan_out_count << "\n";
    ofs << "fp16 inf in/out             : " << experiment_full_stats->fp16.inf_in_count
        << " / " << experiment_full_stats->fp16.inf_out_count << "\n";
    ofs << "fp16 underflow->zero        : " << experiment_full_stats->fp16.underflow_to_zero_count << "\n";
    ofs << "fp16 first nonfinite block  : "
        << (experiment_full_stats->fp16.first_nonfinite_block.empty()
            ? "none" : experiment_full_stats->fp16.first_nonfinite_block)
        << "\n";
  }
  ofs << "\n";

  ofs << "margin distribution (per-pattern min(abs(logit)), using min(baseline,experiment)):\n";
  ofs << "  min    : " << margin_dist.min_v << "\n";
  ofs << "  max    : " << margin_dist.max_v << "\n";
  ofs << "  mean   : " << margin_dist.mean_v << "\n";
  ofs << "  median : " << margin_dist.median_v << "\n";
  ofs << "\n";

  ofs << "top vulnerable patterns (sorted by smallest min margin):\n";
  for (std::size_t k = 0; k < vulnerable_idx.size(); ++k) {
    const PerPatternCompareRow& r = rows[vulnerable_idx[k]];
    ofs << "  rank " << (k + 1U)
        << " pattern=" << r.pattern_index
        << " min_margin_baseline=" << r.cmp.baseline_min_abs_margin
        << " min_margin_experiment=" << r.cmp.experiment_min_abs_margin
        << " logits_mse=" << r.cmp.logits_diff.mse
        << " logits_mae=" << r.cmp.logits_diff.mae
        << " max_abs_diff=" << r.cmp.logits_diff.max_abs
        << " xpred_mismatch=" << r.cmp.x_pred_mismatch_count
        << " sign_flip=" << r.cmp.sign_flip_count
        << "\n";
  }

  return ofs.good();
}

static std::string derive_timing_txt_path(const std::string& csv_path) {
  if (csv_path.size() >= 4U && csv_path.substr(csv_path.size() - 4U) == ".csv") {
    return csv_path.substr(0, csv_path.size() - 4U) + "_timing.txt";
  }
  return csv_path + "_timing.txt";
}

static bool write_timing_txt(const std::string& path, const PerfBreakdownSec& perf) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
  if (!ofs.good()) {
    return false;
  }
  ofs.setf(std::ios::scientific);
  ofs << std::setprecision(9);
  ofs << "startup_init_s       : " << perf.startup_init_s << "\n";
  ofs << "baseline_model_s     : " << perf.baseline_model_s << "\n";
  ofs << "experiment_path_s    : " << perf.experiment_path_s << "\n";
  ofs << "compare_aggregation_s: " << perf.compare_aggregation_s << "\n";
  ofs << "file_io_s            : " << perf.file_io_s << "\n";
  ofs << "total_s              : " << perf.total_s << "\n";
  return ofs.good();
}

static void print_eval_single_console_summary(const char* tag, const EvalAggregateStats& stats) {
  std::printf("=== Evaluator Summary (%s) ===\n", tag);
  std::printf("total patterns             : %d\n", stats.total_patterns);
  std::printf("total evaluated bits       : %zu\n", stats.total_bits);
  std::printf("total bit errors           : %zu\n", stats.total_bit_errors);
  std::printf("BER                        : %.9e\n", stats.ber);
  std::printf("frame error count          : %zu\n", stats.frame_error_count);
  std::printf("FER                        : %.9e\n", stats.fer);
  std::printf("x_pred/target match count  : %zu\n", stats.x_pred_match_count);
  std::printf("x_pred/target match ratio  : %.9e\n", stats.x_pred_match_ratio);
}

static void print_eval_compare_console_summary(
  const EvalAggregateStats& baseline_stats,
  const EvalAggregateStats& experiment_stats
) {
  const double delta_ber = experiment_stats.ber - baseline_stats.ber;
  const double delta_fer = experiment_stats.fer - baseline_stats.fer;
  std::printf("=== Evaluator Compare Summary ===\n");
  std::printf("total patterns             : %d\n", baseline_stats.total_patterns);
  std::printf("total evaluated bits       : %zu\n", baseline_stats.total_bits);
  std::printf("baseline total bit errors  : %zu\n", baseline_stats.total_bit_errors);
  std::printf("baseline BER               : %.9e\n", baseline_stats.ber);
  std::printf("baseline frame errors      : %zu\n", baseline_stats.frame_error_count);
  std::printf("baseline FER               : %.9e\n", baseline_stats.fer);
  std::printf("experiment total bit errors: %zu\n", experiment_stats.total_bit_errors);
  std::printf("experiment BER             : %.9e\n", experiment_stats.ber);
  std::printf("experiment frame errors    : %zu\n", experiment_stats.frame_error_count);
  std::printf("experiment FER             : %.9e\n", experiment_stats.fer);
  std::printf("delta BER (exp-base)       : %.9e\n", delta_ber);
  std::printf("delta FER (exp-base)       : %.9e\n", delta_fer);
  std::printf("baseline x_pred match ratio: %.9e\n", baseline_stats.x_pred_match_ratio);
  std::printf("experiment x_pred match ratio: %.9e\n", experiment_stats.x_pred_match_ratio);
}

static void print_full_stress_console_summary(const aecct_ref::RefFullQuantStats& stats) {
  std::printf("=== Quant Stress Counters (experiment) ===\n");
  std::printf("int8 clamp count           : %llu\n",
    static_cast<unsigned long long>(stats.int_linear.int8_clamp_count));
  std::printf("int16 overflow count       : %llu\n",
    static_cast<unsigned long long>(stats.int_linear.int16_overflow_count));
  std::printf("dequant restore count      : %llu\n",
    static_cast<unsigned long long>(stats.int_linear.dequant_restore_count));
  std::printf("e4m3 roundtrip count       : %llu\n",
    static_cast<unsigned long long>(stats.e4m3.roundtrip_count));
  std::printf("e4m3 roundtrip g1/g2/g3/g4/g5 : %llu / %llu / %llu / %llu / %llu\n",
    static_cast<unsigned long long>(stats.e4m3.roundtrip_g1_count),
    static_cast<unsigned long long>(stats.e4m3.roundtrip_g2_count),
    static_cast<unsigned long long>(stats.e4m3.roundtrip_g3_count),
    static_cast<unsigned long long>(stats.e4m3.roundtrip_g4_count),
    static_cast<unsigned long long>(stats.e4m3.roundtrip_g5_count));
  std::printf("e4m3 roundtrip g5 sub (embed/spe/assembly/prelayer) : %llu / %llu / %llu / %llu\n",
    static_cast<unsigned long long>(stats.e4m3.roundtrip_g5_embed_count),
    static_cast<unsigned long long>(stats.e4m3.roundtrip_g5_spe_count),
    static_cast<unsigned long long>(stats.e4m3.roundtrip_g5_preproc_assembly_count),
    static_cast<unsigned long long>(stats.e4m3.roundtrip_g5_prelayer_handoff_count));
  std::printf("e4m3 nan in/out            : %llu / %llu\n",
    static_cast<unsigned long long>(stats.e4m3.nan_in_count),
    static_cast<unsigned long long>(stats.e4m3.nan_out_count));
  std::printf("e4m3 inf in/out            : %llu / %llu\n",
    static_cast<unsigned long long>(stats.e4m3.inf_in_count),
    static_cast<unsigned long long>(stats.e4m3.inf_out_count));
  std::printf("first nonfinite block      : %s\n",
    stats.e4m3.first_nonfinite_block.empty() ? "none" : stats.e4m3.first_nonfinite_block.c_str());
  std::printf("first int16 overflow block : %s\n",
    stats.int_linear.first_int16_overflow_block.empty()
      ? "none" : stats.int_linear.first_int16_overflow_block.c_str());
  std::printf("int8 fixedexp roundtrip    : %llu\n",
    static_cast<unsigned long long>(stats.int8_fixedexp.roundtrip_count));
  std::printf("int8 fixedexp clamp count  : %llu\n",
    static_cast<unsigned long long>(stats.int8_fixedexp.clamp_count));
  std::printf("int8 fixedexp zone z1/z2/z3/z4 : %llu / %llu / %llu / %llu\n",
    static_cast<unsigned long long>(stats.int8_fixedexp.zone1_count),
    static_cast<unsigned long long>(stats.int8_fixedexp.zone2_count),
    static_cast<unsigned long long>(stats.int8_fixedexp.zone3_count),
    static_cast<unsigned long long>(stats.int8_fixedexp.zone4_count));
  std::printf("int8 fixedexp footprint g2/g5_embed : %llu / %llu\n",
    static_cast<unsigned long long>(stats.int8_fixedexp.footprint_g2_count),
    static_cast<unsigned long long>(stats.int8_fixedexp.footprint_g5_embed_count));
  std::printf("int8 fixedexp first clamp block : %s\n",
    stats.int8_fixedexp.first_clamp_block.empty()
      ? "none" : stats.int8_fixedexp.first_clamp_block.c_str());
  std::printf("fp16 roundtrip count      : %llu\n",
    static_cast<unsigned long long>(stats.fp16.roundtrip_count));
  std::printf("fp16 nan in/out           : %llu / %llu\n",
    static_cast<unsigned long long>(stats.fp16.nan_in_count),
    static_cast<unsigned long long>(stats.fp16.nan_out_count));
  std::printf("fp16 inf in/out           : %llu / %llu\n",
    static_cast<unsigned long long>(stats.fp16.inf_in_count),
    static_cast<unsigned long long>(stats.fp16.inf_out_count));
  std::printf("fp16 underflow->zero      : %llu\n",
    static_cast<unsigned long long>(stats.fp16.underflow_to_zero_count));
  std::printf("fp16 first nonfinite block: %s\n",
    stats.fp16.first_nonfinite_block.empty()
      ? "none" : stats.fp16.first_nonfinite_block.c_str());
}

struct StageCompareEvalSnapshot {
  aecct_ref::RefFinalHeadExploreStage stage;
  PatternRange range;
  BatchCompareSummary compare_batch;
  DistributionStats margin_dist;
  EvalAggregateStats baseline_eval;
  EvalAggregateStats experiment_eval;
  PerfBreakdownSec perf;
  std::vector<PerPatternCompareRow> compare_rows;
  std::vector<EvalComparePatternRow> eval_rows;
};

struct ExploreStageOutcome {
  bool attempted;
  bool pass;
  bool red;
  bool yellow;
  bool promoted;
  bool rolled_back;
  bool has_quick;
  bool has_eval32;
  std::string reason;
  StageCompareEvalSnapshot quick;
  StageCompareEvalSnapshot eval32;
};

static bool run_stage_compare_eval_snapshot(
  const CliOptions& opts,
  int n_vars,
  const PatternRange& range,
  aecct_ref::RefFinalHeadExploreStage stage,
  const std::string& file_prefix,
  bool summary_only,
  StageCompareEvalSnapshot& out
) {
  out = StageCompareEvalSnapshot{};
  out.stage = stage;
  out.range = range;
  init_batch_summary(out.compare_batch);
  init_eval_aggregate(out.baseline_eval);
  init_eval_aggregate(out.experiment_eval);
  out.compare_rows.reserve(static_cast<std::size_t>(range.count));
  out.eval_rows.reserve(static_cast<std::size_t>(range.count));

  aecct_ref::RefModel baseline_model;
  aecct_ref::RefRunConfig baseline_cfg = aecct_ref::make_fp32_baseline_run_config();
  baseline_cfg.algo_variant = opts.algo_variant;
  baseline_cfg.ln_mode = opts.ln_mode;
  baseline_cfg.finalhead_stage = stage;
  baseline_model.set_run_config(baseline_cfg);

  std::vector<double> baseline_logits_batch;
  std::vector<aecct_ref::bit1_t> baseline_x_pred_batch;
  std::vector<double> baseline_finalhead_s_t;
  baseline_finalhead_s_t.resize(static_cast<std::size_t>(range.count * STAGE_TOKENS));
  std::vector<double> experiment_logits_batch;
  std::vector<aecct_ref::bit1_t> experiment_x_pred_batch;

  const auto t0 = now_tp();
  run_ref_batch(
    baseline_model,
    range,
    n_vars,
    baseline_logits_batch,
    baseline_x_pred_batch,
    baseline_finalhead_s_t.data()
  );
  out.perf.baseline_model_s = elapsed_sec(t0, now_tp());

  const auto t1 = now_tp();
  run_experiment_from_baseline_finalhead(
    range,
    n_vars,
    stage,
    baseline_finalhead_s_t,
    experiment_logits_batch,
    experiment_x_pred_batch
  );
  out.perf.experiment_path_s = elapsed_sec(t1, now_tp());

  const auto t2 = now_tp();
  for (int off = 0; off < range.count; ++off) {
    const int pattern = range.begin + off;
    const std::size_t base_idx = static_cast<std::size_t>(off * n_vars);

    PerPatternCompareRow cmp_row{};
    cmp_row.pattern_index = pattern;
    cmp_row.baseline_vs_golden = compute_pattern_logits_vs_golden(baseline_logits_batch, off, pattern, n_vars);
    cmp_row.experiment_vs_golden = compute_pattern_logits_vs_golden(experiment_logits_batch, off, pattern, n_vars);
    cmp_row.cmp = aecct_ref::compute_experiment_compare_metrics(
      &baseline_logits_batch[base_idx],
      &experiment_logits_batch[base_idx],
      &baseline_x_pred_batch[base_idx],
      &experiment_x_pred_batch[base_idx],
      static_cast<std::size_t>(n_vars),
      static_cast<std::size_t>(n_vars),
      static_cast<std::size_t>(opts.topk)
    );
    out.compare_rows.push_back(cmp_row);
    update_batch_summary(out.compare_batch, cmp_row);

    const EvalPatternRow b_row = build_eval_pattern_row(
      &baseline_x_pred_batch[base_idx],
      pattern,
      n_vars
    );
    const EvalPatternRow e_row = build_eval_pattern_row(
      &experiment_x_pred_batch[base_idx],
      pattern,
      n_vars
    );
    update_eval_aggregate(out.baseline_eval, b_row);
    update_eval_aggregate(out.experiment_eval, e_row);

    EvalComparePatternRow eval_row{};
    eval_row.pattern_index = pattern;
    eval_row.evaluated_bits = b_row.evaluated_bits;
    eval_row.baseline_bit_errors = b_row.bit_errors;
    eval_row.experiment_bit_errors = e_row.bit_errors;
    eval_row.baseline_x_pred_match_count = b_row.x_pred_match_count;
    eval_row.experiment_x_pred_match_count = e_row.x_pred_match_count;
    eval_row.baseline_frame_error_flag = b_row.frame_error_flag;
    eval_row.experiment_frame_error_flag = e_row.frame_error_flag;
    out.eval_rows.push_back(eval_row);

    if (!summary_only) {
      std::printf("[stage %s][pattern %d] mse=%.9e mae=%.9e maxabs=%.9e xflip=%zu sflip=%zu min_margin=%.9e b_err=%zu e_err=%zu\n",
        aecct_ref::to_string(stage),
        pattern,
        cmp_row.cmp.logits_diff.mse,
        cmp_row.cmp.logits_diff.mae,
        cmp_row.cmp.logits_diff.max_abs,
        cmp_row.cmp.x_pred_mismatch_count,
        cmp_row.cmp.sign_flip_count,
        (cmp_row.cmp.baseline_min_abs_margin < cmp_row.cmp.experiment_min_abs_margin)
          ? cmp_row.cmp.baseline_min_abs_margin : cmp_row.cmp.experiment_min_abs_margin,
        eval_row.baseline_bit_errors,
        eval_row.experiment_bit_errors);
    }
  }
  out.perf.compare_aggregation_s = elapsed_sec(t2, now_tp());
  finalize_eval_aggregate(out.baseline_eval);
  finalize_eval_aggregate(out.experiment_eval);
  out.margin_dist = compute_distribution_stats(out.compare_batch.per_pattern_min_margin);

  const auto t3 = now_tp();
  const std::string compare_csv = file_prefix + "_compare.csv";
  const std::string compare_txt = file_prefix + "_compare.txt";
  const std::string eval_csv = file_prefix + "_eval.csv";
  const std::string eval_txt = file_prefix + "_eval.txt";
  const std::string timing_txt = file_prefix + "_timing.txt";
  const std::vector<std::size_t> vulnerable_idx = collect_top_vulnerable_indices(out.compare_rows, 5U);
  write_compare_csv(compare_csv, out.compare_rows);
  write_batch_summary_txt(compare_txt, out.compare_batch, out.margin_dist, out.compare_rows, vulnerable_idx);
  write_eval_compare_csv(eval_csv, out.eval_rows);
  write_eval_compare_summary_txt(eval_txt, out.baseline_eval, out.experiment_eval, out.eval_rows);
  out.perf.file_io_s = elapsed_sec(t3, now_tp());
  out.perf.total_s = out.perf.baseline_model_s + out.perf.experiment_path_s +
                     out.perf.compare_aggregation_s + out.perf.file_io_s;
  write_timing_txt(timing_txt, out.perf);
  return true;
}

static bool is_stage_red(const StageCompareEvalSnapshot& s, std::string& reason) {
  const double delta_ber = s.experiment_eval.ber - s.baseline_eval.ber;
  const double delta_fer = s.experiment_eval.fer - s.baseline_eval.fer;
  if (delta_ber > 0.0) {
    reason = "delta BER > 0";
    return true;
  }
  if (delta_fer > 0.0) {
    reason = "delta FER > 0";
    return true;
  }
  if (s.compare_batch.patterns_with_xpred_flip > 0U) {
    reason = "x_pred flips detected";
    return true;
  }
  if (s.compare_batch.patterns_with_sign_flip > 0U) {
    reason = "sign flips detected";
    return true;
  }
  if (s.compare_batch.baseline_nonfinite_total.nan_count > 0U ||
      s.compare_batch.baseline_nonfinite_total.inf_count > 0U ||
      s.compare_batch.experiment_nonfinite_total.nan_count > 0U ||
      s.compare_batch.experiment_nonfinite_total.inf_count > 0U) {
    reason = "NaN/Inf detected";
    return true;
  }
  reason.clear();
  return false;
}

static bool is_stage_yellow(const StageCompareEvalSnapshot& s) {
  return s.compare_batch.worst_min_margin < 2.0e-2;
}

static void print_stage_snapshot_summary(const char* tag, const StageCompareEvalSnapshot& s) {
  const double delta_ber = s.experiment_eval.ber - s.baseline_eval.ber;
  const double delta_fer = s.experiment_eval.fer - s.baseline_eval.fer;
  std::printf("[%s][%s] patterns=%d mse_worst=%.9e maxabs_worst=%.9e xflip_patterns=%zu signflip_patterns=%zu worst_min_margin=%.9e bBER=%.9e eBER=%.9e dBER=%.9e bFER=%.9e eFER=%.9e dFER=%.9e baseline_nan_inf=%zu/%zu experiment_nan_inf=%zu/%zu\n",
    aecct_ref::to_string(s.stage),
    tag,
    s.compare_batch.total_patterns_scanned,
    s.compare_batch.worst_logits_mse,
    s.compare_batch.worst_max_abs_diff,
    s.compare_batch.patterns_with_xpred_flip,
    s.compare_batch.patterns_with_sign_flip,
    s.compare_batch.worst_min_margin,
    s.baseline_eval.ber,
    s.experiment_eval.ber,
    delta_ber,
    s.baseline_eval.fer,
    s.experiment_eval.fer,
    delta_fer,
    s.compare_batch.baseline_nonfinite_total.nan_count,
    s.compare_batch.baseline_nonfinite_total.inf_count,
    s.compare_batch.experiment_nonfinite_total.nan_count,
    s.compare_batch.experiment_nonfinite_total.inf_count);
}

static inline uint32_t local_fp32_bits_from_double(double x) {
  const float xf = static_cast<float>(x);
  uint32_t bits = 0u;
  std::memcpy(&bits, &xf, sizeof(bits));
  return bits;
}

static inline double local_fp32_double_from_words16(uint16_t lo16, uint16_t hi16) {
  const uint32_t bits = static_cast<uint32_t>(lo16) | (static_cast<uint32_t>(hi16) << 16);
  float xf = 0.0f;
  std::memcpy(&xf, &bits, sizeof(xf));
  return static_cast<double>(xf);
}

static const char* io16_output_mode_to_string(aecct_ref::RefStep0OutputMode mode) {
  return (mode == aecct_ref::RefStep0OutputMode::LOGITS) ? "logits" : "xpred";
}

static bool run_io16_single_pattern_selfcheck(
  aecct_ref::RefModel& model,
  int pattern_index,
  int n_vars,
  const RefRunOutputs& run,
  aecct_ref::RefStep0OutputMode output_mode,
  const char* tag
) {
  const double* input_ptr = &trace_input_y_step0_tensor[pattern_index * n_vars];

  std::vector<double> image_logits(static_cast<std::size_t>(n_vars), 0.0);
  std::vector<aecct_ref::bit1_t> image_xpred(static_cast<std::size_t>(n_vars), aecct_ref::bit1_t(0));
  std::vector<double> image_final_s(static_cast<std::size_t>(STAGE_TOKENS), 0.0);

  aecct_ref::RefModelIO io{};
  io.input_y = nullptr;
  io.input_y_fp32 = input_ptr;
  io.out_logits = image_logits.data();
  io.out_x_pred = image_xpred.data();
  io.out_finalhead_s_t = image_final_s.data();
  io.B = 1;
  io.N = n_vars;

  aecct_ref::RefStep0Io16Image image{};
  const bool build_ok = model.build_step0_io16_image(io, output_mode, image);
  if (!build_ok) {
    std::printf("[io16-selfcheck][pattern %d][%s] build_ok=0 output_mode=%s\n",
      pattern_index, tag, io16_output_mode_to_string(output_mode));
    return false;
  }

  bool infer_copy_exact = (image_logits.size() == run.logits.size()) &&
                          (image_xpred.size() == run.x_pred.size());
  if (infer_copy_exact) {
    for (std::size_t i = 0; i < image_logits.size(); ++i) {
      if (local_fp32_bits_from_double(image_logits[i]) != local_fp32_bits_from_double(run.logits[i])) {
        infer_copy_exact = false;
        break;
      }
    }
  }
  if (infer_copy_exact) {
    for (std::size_t i = 0; i < image_xpred.size(); ++i) {
      if (image_xpred[i].to_uint() != run.x_pred[i].to_uint()) {
        infer_copy_exact = false;
        break;
      }
    }
  }

  std::vector<uint16_t> final_scalar_words16;
  const bool read_mem_ok = aecct_ref::RefModel::read_mem_words16(
    image,
    image.report.final_scalar_base_word16,
    image.report.final_scalar_words16,
    final_scalar_words16);

  bool read_mem_exact = false;
  bool final_scalar_decode_exact = false;
  if (read_mem_ok) {
    const std::size_t start = static_cast<std::size_t>(image.report.final_scalar_base_word16);
    const std::size_t count = static_cast<std::size_t>(image.report.final_scalar_words16);
    const auto begin_it = image.sram_words16.begin() + static_cast<std::ptrdiff_t>(start);
    const auto end_it = begin_it + static_cast<std::ptrdiff_t>(count);
    const std::vector<uint16_t> expect(begin_it, end_it);
    read_mem_exact = (final_scalar_words16 == expect);

    if (final_scalar_words16.size() >= (static_cast<std::size_t>(STAGE_TOKENS) * 2u)) {
      final_scalar_decode_exact = true;
      for (int i = 0; i < STAGE_TOKENS; ++i) {
        const double decoded = local_fp32_double_from_words16(
          final_scalar_words16[static_cast<std::size_t>(i) * 2u],
          final_scalar_words16[static_cast<std::size_t>(i) * 2u + 1u]);
        if (local_fp32_bits_from_double(decoded) !=
            local_fp32_bits_from_double(image_final_s[static_cast<std::size_t>(i)])) {
          final_scalar_decode_exact = false;
          break;
        }
      }
      if (final_scalar_decode_exact) {
        for (std::size_t i = static_cast<std::size_t>(STAGE_TOKENS) * 2u; i < final_scalar_words16.size(); ++i) {
          if (final_scalar_words16[i] != static_cast<uint16_t>(0u)) {
            final_scalar_decode_exact = false;
            break;
          }
        }
      }
    }
  }

  bool output_unpack_ok = false;
  bool output_exact = false;
  if (output_mode == aecct_ref::RefStep0OutputMode::LOGITS) {
    std::vector<double> unpacked_logits;
    output_unpack_ok = aecct_ref::RefModel::unpack_logits_from_io16(image.data_out_words16, unpacked_logits);
    output_exact = output_unpack_ok && (unpacked_logits.size() == run.logits.size());
    if (output_exact) {
      for (std::size_t i = 0; i < unpacked_logits.size(); ++i) {
        if (local_fp32_bits_from_double(unpacked_logits[i]) != local_fp32_bits_from_double(run.logits[i])) {
          output_exact = false;
          break;
        }
      }
    }
  } else {
    std::vector<uint8_t> unpacked_xpred;
    output_unpack_ok = aecct_ref::RefModel::unpack_xpred_from_io16(image.data_out_words16, n_vars, unpacked_xpred);
    output_exact = output_unpack_ok && (unpacked_xpred.size() == run.x_pred.size());
    if (output_exact) {
      for (std::size_t i = 0; i < unpacked_xpred.size(); ++i) {
        if (unpacked_xpred[i] != static_cast<uint8_t>(run.x_pred[i].to_uint())) {
          output_exact = false;
          break;
        }
      }
    }
  }

  std::printf("[io16-selfcheck][pattern %d][%s] output_mode=%s build_ok=1 infer_copy_exact=%d read_mem_ok=%d read_mem_exact=%d final_scalar_decode_exact=%d output_unpack_ok=%d output_exact=%d final_scalar_range_ok=%d final_scalar_capacity_ok=%d output_words16=%u final_scalar_base_word16=%u final_scalar_words16=%u scratch_base_word16=%u scratch_words16=%u\n",
    pattern_index,
    tag,
    io16_output_mode_to_string(output_mode),
    infer_copy_exact ? 1 : 0,
    read_mem_ok ? 1 : 0,
    read_mem_exact ? 1 : 0,
    final_scalar_decode_exact ? 1 : 0,
    output_unpack_ok ? 1 : 0,
    output_exact ? 1 : 0,
    image.report.final_scalar_range_ok ? 1 : 0,
    image.report.final_scalar_capacity_ok ? 1 : 0,
    image.report.output_words16,
    image.report.final_scalar_base_word16,
    image.report.final_scalar_words16,
    image.report.scratch_base_word16,
    image.report.scratch_words16);

  return infer_copy_exact && read_mem_ok && read_mem_exact && final_scalar_decode_exact &&
         output_unpack_ok && output_exact &&
         image.report.final_scalar_range_ok && image.report.final_scalar_capacity_ok;
}

static int run_single_mode(
  aecct_ref::RefPrecisionMode precision_mode,
  aecct_ref::RefLayerNormMode ln_mode,
  const char* tag,
  const PatternRange& range,
  int n_vars,
  const CliOptions& opts
) {
  auto append_bits_from_trace = [n_vars](int pattern, std::vector<int>& out_bits) {
    out_bits.clear();
    for (int i = 0; i < n_vars; ++i) {
      const bool t = trace_output_x_pred_step0_tensor[(pattern * n_vars) + i] != 0.0;
      if (t) out_bits.push_back(i);
    }
  };
  auto append_bits_from_pred = [n_vars](const std::vector<aecct_ref::bit1_t>& pred, std::vector<int>& out_bits) {
    out_bits.clear();
    const int lim = static_cast<int>(pred.size());
    const int n = (n_vars < lim) ? n_vars : lim;
    for (int i = 0; i < n; ++i) {
      if (static_cast<int>(pred[static_cast<std::size_t>(i)]) != 0) out_bits.push_back(i);
    }
  };
  auto append_mismatch_bits = [n_vars](int pattern, const std::vector<aecct_ref::bit1_t>& pred, std::vector<int>& out_bits) {
    out_bits.clear();
    const int lim = static_cast<int>(pred.size());
    const int n = (n_vars < lim) ? n_vars : lim;
    for (int i = 0; i < n; ++i) {
      const bool t = trace_output_x_pred_step0_tensor[(pattern * n_vars) + i] != 0.0;
      const bool p = static_cast<int>(pred[static_cast<std::size_t>(i)]) != 0;
      if (t != p) out_bits.push_back(i);
    }
  };
  auto format_bits = [](const std::vector<int>& bits) -> std::string {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < bits.size(); ++i) {
      if (i > 0U) oss << ",";
      oss << bits[i];
    }
    oss << "]";
    return oss.str();
  };

  aecct_ref::RefModel model;
  aecct_ref::RefRunConfig cfg = aecct_ref::make_fp32_baseline_run_config();
  cfg.precision_mode = precision_mode;
  cfg.algo_variant = opts.algo_variant;
  cfg.ln_mode = ln_mode;
  cfg.finalhead_stage = opts.finalhead_stage;
  cfg.frag_group = opts.frag_group;
  cfg.softmax_exp_mode = opts.softmax_exp_mode;
  model.set_run_config(cfg);

  GoldenAggregateMetrics agg{};
  init_golden_aggregate(agg);

  for (int off = 0; off < range.count; ++off) {
    const int p = range.begin + off;
    RefRunOutputs run{};
    run_ref_single_pattern(model, p, n_vars, run);
    if (range.count == 1) {
      print_vs_golden_summary(tag, run);
      if (opts.xpred_focused_inspect) {
        std::vector<int> trace_ones;
        std::vector<int> pred_ones;
        std::vector<int> mismatch_bits;
        std::vector<int> gt_error_bits;
        append_bits_from_trace(p, trace_ones);
        append_bits_from_pred(run.x_pred, pred_ones);
        append_mismatch_bits(p, run.x_pred, mismatch_bits);
        append_bits_from_pred(run.x_pred, gt_error_bits);
        std::printf("[xpred-focused][pattern %d][%s] trace_ones=%s\n",
          p, tag, format_bits(trace_ones).c_str());
        std::printf("[xpred-focused][pattern %d][%s] pred_ones=%s\n",
          p, tag, format_bits(pred_ones).c_str());
        std::printf("[xpred-focused][pattern %d][%s] mismatch_vs_trace_bits=%s\n",
          p, tag, format_bits(mismatch_bits).c_str());
        std::printf("[xpred-focused][pattern %d][%s] error_vs_all_zero_gt_bits=%s\n",
          p, tag, format_bits(gt_error_bits).c_str());
      }
      if (opts.io16_selfcheck) {
        const bool io16_ok = run_io16_single_pattern_selfcheck(
          model, p, n_vars, run, opts.io16_output_mode, tag);
        if (!io16_ok) {
          return 1;
        }
      }
    } else {
      std::printf("[pattern %d][%s] logits_mse=%.9e logits_maxabs=%.9e x_pred_match=%zu/%zu\n",
        p, tag, run.logits_vs_golden.mse, run.logits_vs_golden.max_abs,
        run.x_pred_match_count, run.x_pred_total_count);
    }
    update_golden_aggregate(agg, run);
  }

  if (range.count > 1) {
    const aecct_ref::Metrics m = finalize_golden_aggregate(agg);
    const double match_ratio = (agg.x_pred_total_count > 0U)
      ? (100.0 * static_cast<double>(agg.x_pred_match_count) / static_cast<double>(agg.x_pred_total_count))
      : 0.0;
    std::printf("=== %s aggregate vs golden ===\n", tag);
    std::printf("logits MSE    : %.9e\n", m.mse);
    std::printf("logits RMSE   : %.9e\n", m.rmse);
    std::printf("logits MAE    : %.9e\n", m.mae);
    std::printf("logits MaxAbs : %.9e\n", m.max_abs);
    std::printf("x_pred match  : %.2f%% (%zu / %zu)\n",
      match_ratio, agg.x_pred_match_count, agg.x_pred_total_count);
  }
  return 0;
}

} // anonymous namespace

int main(int argc, char** argv) {
  const auto t_program_start = now_tp();
  PerfBreakdownSec perf{};
  const int B = trace_input_y_step0_tensor_shape[0];
  const int N = trace_input_y_step0_tensor_shape[1];
  const int logits_B = trace_output_logits_step0_tensor_shape[0];
  const int logits_N = trace_output_logits_step0_tensor_shape[1];
  const int xpred_B = trace_output_x_pred_step0_tensor_shape[0];
  const int xpred_N = trace_output_x_pred_step0_tensor_shape[1];

  if (logits_B != B || xpred_B != B || logits_N != N || xpred_N != N) {
    std::printf("Trace shape mismatch between input/logits/x_pred headers.\n");
    return 2;
  }

  CliOptions opts{};
  const CliParseResult parse_result = parse_cli(argc, argv, opts);
  if (parse_result == CliParseResult::HELP) {
    print_usage();
    return 0;
  }
  if (parse_result != CliParseResult::OK) {
    return 1;
  }

  PatternRange range{};
  if (!resolve_pattern_range(opts, B, range)) {
    return 1;
  }
  if (precision_mode_requires_frag_group(opts.experiment_precision_mode) &&
      opts.frag_group == aecct_ref::RefFragGroup::NONE &&
      opts.run_mode != CliRunMode::BASELINE_ONLY) {
    std::printf("For generic_e4m3_frag_bisect, --frag-group must be one of G1..G5 or C1..C4\n");
    return 1;
  }
  if (opts.io16_selfcheck) {
    if (range.count != 1) {
      std::printf("--io16-selfcheck requires a single-pattern run (--pattern N or count=1)\n");
      return 1;
    }
    if (!(opts.run_mode == CliRunMode::BASELINE_ONLY || opts.run_mode == CliRunMode::EXPERIMENT_ONLY)) {
      std::printf("--io16-selfcheck only supports --mode baseline or --mode experiment\n");
      return 1;
    }
  }

  std::printf("Run config:\n");
  const bool anchor_baseline_for_compare =
    (opts.run_mode != CliRunMode::BASELINE_ONLY) &&
    precision_mode_anchors_to_finalhead_s0(opts.experiment_precision_mode);
  const aecct_ref::RefPrecisionMode effective_baseline_precision =
    anchor_baseline_for_compare
      ? aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD
      : aecct_ref::RefPrecisionMode::BASELINE_FP32;
  const bool use_experiment_banner_cfg = run_mode_uses_experiment_config(opts.run_mode);
  const bool run_has_dual_path = run_mode_has_dual_path(opts.run_mode);
  const aecct_ref::RefSoftmaxExpMode baseline_softmax_mode =
    run_has_dual_path ? aecct_ref::RefSoftmaxExpMode::BASELINE_NEAREST_LUT : opts.softmax_exp_mode;
  const aecct_ref::RefSoftmaxExpMode experiment_softmax_mode = opts.softmax_exp_mode;
  const aecct_ref::RefSoftmaxExpMode banner_softmax_mode =
    use_experiment_banner_cfg ? experiment_softmax_mode : baseline_softmax_mode;
  const aecct_ref::RefPrecisionMode banner_precision_mode =
    use_experiment_banner_cfg ? opts.experiment_precision_mode : effective_baseline_precision;
  const aecct_ref::RefLayerNormMode banner_ln_mode =
    use_experiment_banner_cfg ? opts.experiment_ln_mode : opts.ln_mode;
  std::printf("  mode           : %s\n", run_mode_to_string(opts.run_mode));
  std::printf("  precision_mode : %s%s\n",
              aecct_ref::to_string(banner_precision_mode),
              (use_experiment_banner_cfg && !opts.experiment_precision_explicit) ? " (default)" : "");
  std::printf("  algo_variant   : %s\n", aecct_ref::to_string(opts.algo_variant));
  std::printf("  softmax_exp_mode: %s\n", aecct_ref::to_string(banner_softmax_mode));
  std::printf("  ln_mode        : %s\n", aecct_ref::to_string(banner_ln_mode));
  std::printf("  precision(base): %s\n", aecct_ref::to_string(effective_baseline_precision));
  std::printf("  precision(exp) : %s\n", aecct_ref::to_string(opts.experiment_precision_mode));
  std::printf("  softmax_exp_mode(base): %s\n", aecct_ref::to_string(baseline_softmax_mode));
  std::printf("  softmax_exp_mode(exp) : %s\n", aecct_ref::to_string(experiment_softmax_mode));
  std::printf("  ln_mode(base)  : %s\n", aecct_ref::to_string(opts.ln_mode));
  std::printf("  ln_mode(exp)   : %s\n", aecct_ref::to_string(opts.experiment_ln_mode));
  std::printf("  finalhead_stage: %s\n", aecct_ref::to_string(opts.finalhead_stage));
  std::printf("  frag_group     : %s\n", aecct_ref::to_string(opts.frag_group));
  std::printf("  pattern_range  : begin=%d count=%d\n", range.begin, range.count);
  std::printf("  topk           : %d\n", opts.topk);
  std::printf("  summary_only   : %d\n", opts.summary_only ? 1 : 0);
  std::printf("  ln_debug       : %d\n", opts.ln_debug ? 1 : 0);
  std::printf("  ln_var_debug   : %d\n", opts.ln_var_debug ? 1 : 0);
  std::printf("  ln_input_debug : %d\n", opts.ln_input_debug ? 1 : 0);
  std::printf("  ln_upstream_debug: %d\n", opts.ln_upstream_debug ? 1 : 0);
  std::printf("  io16_selfcheck : %d\n", opts.io16_selfcheck ? 1 : 0);
  std::printf("  io16_output    : %s\n", io16_output_mode_to_string(opts.io16_output_mode));

  perf.startup_init_s = elapsed_sec(t_program_start, now_tp());

  if (opts.run_mode == CliRunMode::BASELINE_ONLY) {
    return run_single_mode(
      aecct_ref::RefPrecisionMode::BASELINE_FP32,
      opts.ln_mode,
      "baseline",
      range,
      N,
      opts);
  }
  if (opts.run_mode == CliRunMode::EXPERIMENT_ONLY) {
    return run_single_mode(
      opts.experiment_precision_mode,
      opts.experiment_ln_mode,
      "experiment",
      range,
      N,
      opts);
  }

  if (opts.run_mode == CliRunMode::EXPLORE) {
    if (range.count <= 0) {
      std::printf("[explore] Empty range.\n");
      return 1;
    }

    const PatternRange quick_range{range.begin, (range.count < 16) ? range.count : 16};
    const PatternRange eval32_range{range.begin, (range.count < 32) ? range.count : 32};
    const aecct_ref::RefFinalHeadExploreStage ladder[] = {
      aecct_ref::RefFinalHeadExploreStage::S1,
      aecct_ref::RefFinalHeadExploreStage::S2,
      aecct_ref::RefFinalHeadExploreStage::S3,
      aecct_ref::RefFinalHeadExploreStage::S4
    };

    std::vector<ExploreStageOutcome> outcomes(5);
    outcomes[0].attempted = true;
    outcomes[0].pass = true;
    outcomes[0].yellow = true;
    outcomes[0].reason = "Known stage from existing evidence (S0 margin tail below 2e-2).";

    aecct_ref::RefFinalHeadExploreStage best_stage = aecct_ref::RefFinalHeadExploreStage::S0;
    int first_fail_stage = -1;
    std::string stop_reason = "Reached end of FinalHead ladder";

    for (std::size_t si = 0; si < sizeof(ladder) / sizeof(ladder[0]); ++si) {
      const aecct_ref::RefFinalHeadExploreStage stage = ladder[si];
      ExploreStageOutcome& out = outcomes[static_cast<int>(stage)];
      out = ExploreStageOutcome{};
      out.attempted = true;

      const std::string base_prefix = "build/ref_eval/explore_" + std::string(aecct_ref::to_string(stage)) +
        "_begin" + std::to_string(range.begin) + "_count" + std::to_string(range.count);
      const std::string quick_prefix = base_prefix + "_quick16";
      const bool quick_ok = run_stage_compare_eval_snapshot(
        opts, N, quick_range, stage, quick_prefix, true, out.quick
      );
      out.has_quick = quick_ok;
      if (!quick_ok) {
        out.red = true;
        out.pass = false;
        out.rolled_back = true;
        out.reason = "RED: quick16 run failed";
        first_fail_stage = static_cast<int>(stage);
        stop_reason = out.reason;
        break;
      }
      print_stage_snapshot_summary("quick16", out.quick);

      std::string red_reason;
      if (is_stage_red(out.quick, red_reason)) {
        out.red = true;
        out.pass = false;
        out.rolled_back = true;
        out.reason = "RED at quick16: " + red_reason;
        first_fail_stage = static_cast<int>(stage);
        stop_reason = out.reason;
        break;
      }

      StageCompareEvalSnapshot gate_snapshot = out.quick;
      if (eval32_range.count > quick_range.count) {
        const std::string eval32_prefix = base_prefix + "_eval32";
        const bool eval32_ok = run_stage_compare_eval_snapshot(
          opts, N, eval32_range, stage, eval32_prefix, true, out.eval32
        );
        out.has_eval32 = eval32_ok;
        if (!eval32_ok) {
          out.red = true;
          out.pass = false;
          out.rolled_back = true;
          out.reason = "RED: eval32 run failed";
          first_fail_stage = static_cast<int>(stage);
          stop_reason = out.reason;
          break;
        }
        gate_snapshot = out.eval32;
        print_stage_snapshot_summary("eval32", out.eval32);
      }

      if (is_stage_red(gate_snapshot, red_reason)) {
        out.red = true;
        out.pass = false;
        out.rolled_back = true;
        out.reason = "RED at eval gate: " + red_reason;
        first_fail_stage = static_cast<int>(stage);
        stop_reason = out.reason;
        break;
      }

      out.yellow = is_stage_yellow(gate_snapshot);
      out.pass = true;
      out.promoted = (stage != aecct_ref::RefFinalHeadExploreStage::S4);
      out.reason = out.yellow
        ? "PASS YELLOW: worst min margin < 2e-2"
        : "PASS GREEN";
      best_stage = stage;
    }

    const std::string report_path = !opts.summary_csv_path.empty()
      ? derive_summary_txt_path(opts.summary_csv_path)
      : ("build/ref_eval/explore_report_begin" + std::to_string(range.begin) +
         "_count" + std::to_string(range.count) + ".txt");
    std::filesystem::path rp(report_path);
    if (rp.has_parent_path()) {
      std::filesystem::create_directories(rp.parent_path());
    }
    std::ofstream rofs(report_path.c_str(), std::ios::out | std::ios::trunc);
    if (rofs.good()) {
      rofs.setf(std::ios::scientific);
      rofs << std::setprecision(9);
      rofs << "=== Guardrailed Auto-Explore Report ===\n";
      rofs << "quick_range: begin=" << quick_range.begin << " count=" << quick_range.count << "\n";
      rofs << "eval32_range: begin=" << eval32_range.begin << " count=" << eval32_range.count << "\n";
      rofs << "best_stage: " << aecct_ref::to_string(best_stage) << "\n";
      rofs << "first_fail_stage: " << ((first_fail_stage >= 0) ? std::to_string(first_fail_stage) : std::string("none")) << "\n";
      rofs << "stop_reason: " << stop_reason << "\n";
      rofs << "\n";
      rofs << "stage table:\n";
      rofs << "S0: PASS (known baseline stage), class=YELLOW\n";
      const aecct_ref::RefFinalHeadExploreStage all_stages[] = {
        aecct_ref::RefFinalHeadExploreStage::S1,
        aecct_ref::RefFinalHeadExploreStage::S2,
        aecct_ref::RefFinalHeadExploreStage::S3,
        aecct_ref::RefFinalHeadExploreStage::S4
      };
      for (std::size_t i = 0; i < sizeof(all_stages) / sizeof(all_stages[0]); ++i) {
        const aecct_ref::RefFinalHeadExploreStage st = all_stages[i];
        const ExploreStageOutcome& o = outcomes[static_cast<int>(st)];
        rofs << aecct_ref::to_string(st) << ": ";
        if (!o.attempted) {
          rofs << "NOT_ATTEMPTED\n";
          continue;
        }
        rofs << (o.pass ? "PASS" : "FAIL")
             << " class=" << (o.red ? "RED" : (o.yellow ? "YELLOW" : "GREEN"))
             << " promoted=" << (o.promoted ? 1 : 0)
             << " rollback=" << (o.rolled_back ? 1 : 0)
             << " reason=\"" << o.reason << "\"\n";
        const StageCompareEvalSnapshot* snap = o.has_eval32 ? &o.eval32 : (o.has_quick ? &o.quick : nullptr);
        if (snap != nullptr) {
          const double delta_ber = snap->experiment_eval.ber - snap->baseline_eval.ber;
          const double delta_fer = snap->experiment_eval.fer - snap->baseline_eval.fer;
          rofs << "  patterns=" << snap->compare_batch.total_patterns_scanned
               << " worst_logits_mse=" << snap->compare_batch.worst_logits_mse
               << " worst_max_abs=" << snap->compare_batch.worst_max_abs_diff
               << " worst_min_margin=" << snap->compare_batch.worst_min_margin
               << " xflip_patterns=" << snap->compare_batch.patterns_with_xpred_flip
               << " signflip_patterns=" << snap->compare_batch.patterns_with_sign_flip
               << " baseline_nan_inf=" << snap->compare_batch.baseline_nonfinite_total.nan_count
               << "/" << snap->compare_batch.baseline_nonfinite_total.inf_count
               << " experiment_nan_inf=" << snap->compare_batch.experiment_nonfinite_total.nan_count
               << "/" << snap->compare_batch.experiment_nonfinite_total.inf_count
               << "\n";
          rofs << "  baseline_ber=" << snap->baseline_eval.ber
               << " experiment_ber=" << snap->experiment_eval.ber
               << " delta_ber=" << delta_ber
               << " baseline_fer=" << snap->baseline_eval.fer
               << " experiment_fer=" << snap->experiment_eval.fer
               << " delta_fer=" << delta_fer
               << "\n";
        }
      }
    }

    std::printf("=== Explore Summary ===\n");
    std::printf("S0 : PASS (known), class=YELLOW\n");
    for (int si = 1; si <= 4; ++si) {
      const ExploreStageOutcome& o = outcomes[si];
      std::printf("S%d : %s class=%s reason=%s\n",
        si,
        o.attempted ? (o.pass ? "PASS" : "FAIL") : "NOT_ATTEMPTED",
        o.attempted ? (o.red ? "RED" : (o.yellow ? "YELLOW" : "GREEN")) : "N/A",
        o.attempted ? o.reason.c_str() : "not attempted");
    }
    const std::string first_fail_text = (first_fail_stage >= 0)
      ? ("S" + std::to_string(first_fail_stage))
      : std::string("none");
    std::printf("highest safe stage : %s\n", aecct_ref::to_string(best_stage));
    std::printf("first failing stage: %s\n", first_fail_text.c_str());
    std::printf("explore report txt : %s\n", report_path.c_str());
    return 0;
  }

  if (opts.run_mode == CliRunMode::EVAL_BASELINE ||
      opts.run_mode == CliRunMode::EVAL_EXPERIMENT ||
      opts.run_mode == CliRunMode::EVAL_COMPARE) {
    const bool anchor_finalhead_s0 =
      precision_mode_anchors_to_finalhead_s0(opts.experiment_precision_mode);
    const bool use_frag_group_for_experiment =
      precision_mode_requires_frag_group(opts.experiment_precision_mode);
    const bool need_experiment_outputs =
      (opts.run_mode == CliRunMode::EVAL_EXPERIMENT || opts.run_mode == CliRunMode::EVAL_COMPARE);
    aecct_ref::RefModel baseline_model_eval;
    aecct_ref::RefRunConfig baseline_cfg_eval = aecct_ref::make_fp32_baseline_run_config();
    baseline_cfg_eval.precision_mode = anchor_finalhead_s0
      ? aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD
      : aecct_ref::RefPrecisionMode::BASELINE_FP32;
    baseline_cfg_eval.algo_variant = opts.algo_variant;
    baseline_cfg_eval.ln_mode = opts.ln_mode;
    baseline_cfg_eval.finalhead_stage = anchor_finalhead_s0
      ? aecct_ref::RefFinalHeadExploreStage::S0
      : opts.finalhead_stage;
    baseline_cfg_eval.frag_group = aecct_ref::RefFragGroup::NONE;
    baseline_cfg_eval.softmax_exp_mode =
      need_experiment_outputs ? aecct_ref::RefSoftmaxExpMode::BASELINE_NEAREST_LUT : opts.softmax_exp_mode;
    baseline_model_eval.set_run_config(baseline_cfg_eval);

    std::vector<double> baseline_logits_batch;
    std::vector<aecct_ref::bit1_t> baseline_x_pred_batch;
    std::vector<double> baseline_finalhead_s_t;
    std::vector<double> experiment_logits_batch;
    std::vector<aecct_ref::bit1_t> experiment_x_pred_batch;
    aecct_ref::RefFullQuantStats experiment_full_stats{};
    const bool experiment_use_reconstruct =
      need_experiment_outputs &&
      (opts.experiment_precision_mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD) &&
      (opts.experiment_ln_mode == opts.ln_mode) &&
      !opts.ln_debug &&
      !opts.ln_var_debug &&
      !opts.ln_input_debug &&
      !opts.ln_upstream_debug;
    if (experiment_use_reconstruct) {
      baseline_finalhead_s_t.resize(static_cast<std::size_t>(range.count * STAGE_TOKENS));
    }

    const auto t_baseline_start = now_tp();
    run_ref_batch(
      baseline_model_eval,
      range,
      N,
      baseline_logits_batch,
      baseline_x_pred_batch,
      experiment_use_reconstruct ? baseline_finalhead_s_t.data() : nullptr
    );
    perf.baseline_model_s = elapsed_sec(t_baseline_start, now_tp());

    if (need_experiment_outputs) {
      const auto t_experiment_start = now_tp();
      aecct_ref::reset_ref_full_quant_stats();
      if (experiment_use_reconstruct) {
        run_experiment_from_baseline_finalhead(
          range,
          N,
          opts.finalhead_stage,
          baseline_finalhead_s_t,
          experiment_logits_batch,
          experiment_x_pred_batch
        );
      } else {
        aecct_ref::RefModel experiment_model_eval;
        aecct_ref::RefRunConfig experiment_cfg_eval =
          aecct_ref::is_fp16_experiment_mode(opts.experiment_precision_mode)
            ? aecct_ref::make_fp16_experiment_run_config()
            : aecct_ref::make_fp32_baseline_run_config();
        experiment_cfg_eval.precision_mode = opts.experiment_precision_mode;
        experiment_cfg_eval.algo_variant = opts.algo_variant;
        experiment_cfg_eval.ln_mode = opts.experiment_ln_mode;
        experiment_cfg_eval.finalhead_stage = anchor_finalhead_s0
          ? aecct_ref::RefFinalHeadExploreStage::S0
          : opts.finalhead_stage;
        experiment_cfg_eval.frag_group = use_frag_group_for_experiment
          ? opts.frag_group
          : aecct_ref::RefFragGroup::NONE;
        experiment_cfg_eval.softmax_exp_mode = opts.softmax_exp_mode;
        experiment_model_eval.set_run_config(experiment_cfg_eval);
        run_ref_batch(
          experiment_model_eval,
          range,
          N,
          experiment_logits_batch,
          experiment_x_pred_batch,
          nullptr
        );
      }
      experiment_full_stats = aecct_ref::get_ref_full_quant_stats();
      perf.experiment_path_s = elapsed_sec(t_experiment_start, now_tp());
    } else {
      perf.experiment_path_s = 0.0;
    }

    std::string default_eval_name = "eval_compare";
    if (opts.run_mode == CliRunMode::EVAL_BASELINE) {
      default_eval_name = "eval_baseline";
    } else if (opts.run_mode == CliRunMode::EVAL_EXPERIMENT) {
      default_eval_name = "eval_experiment";
    }
    const std::string mode_suffix = precision_mode_output_suffix(
      opts.experiment_precision_mode,
      opts.frag_group
    );
    const std::string csv_path = !opts.summary_csv_path.empty()
      ? opts.summary_csv_path
      : ("build/ref_eval/" + default_eval_name + "_begin" + std::to_string(range.begin) +
         "_count" + std::to_string(range.count) + mode_suffix + ".csv");

    const auto t_eval_agg_start = now_tp();
    if (opts.run_mode == CliRunMode::EVAL_BASELINE || opts.run_mode == CliRunMode::EVAL_EXPERIMENT) {
      const std::vector<aecct_ref::bit1_t>& pred_batch =
        (opts.run_mode == CliRunMode::EVAL_BASELINE) ? baseline_x_pred_batch : experiment_x_pred_batch;
      std::vector<EvalPatternRow> rows;
      rows.reserve(static_cast<std::size_t>(range.count));
      EvalAggregateStats stats{};
      init_eval_aggregate(stats);
      for (int off = 0; off < range.count; ++off) {
        const int pattern = range.begin + off;
        const EvalPatternRow row = build_eval_pattern_row(
          &pred_batch[static_cast<std::size_t>(off * N)],
          pattern,
          N
        );
        rows.push_back(row);
        update_eval_aggregate(stats, row);
        if (!opts.summary_only) {
          const double ber = (row.evaluated_bits > 0U)
            ? (static_cast<double>(row.bit_errors) / static_cast<double>(row.evaluated_bits))
            : 0.0;
          std::printf("[pattern %d][%s] bit_errors=%zu bits=%zu ber=%.9e frame_error=%d xmatch=%zu/%zu\n",
            pattern,
            (opts.run_mode == CliRunMode::EVAL_BASELINE) ? "baseline" : "experiment",
            row.bit_errors,
            row.evaluated_bits,
            ber,
            row.frame_error_flag,
            row.x_pred_match_count,
            row.evaluated_bits);
        }
      }
      finalize_eval_aggregate(stats);
      perf.compare_aggregation_s = elapsed_sec(t_eval_agg_start, now_tp());

      const auto t_fileio_start = now_tp();
      if (write_eval_single_csv(csv_path, rows)) {
        std::printf("Per-pattern eval csv    : %s\n", csv_path.c_str());
      } else {
        std::printf("[warn] Failed to write per-pattern eval csv: %s\n", csv_path.c_str());
      }
      const std::string txt_path = derive_summary_txt_path(csv_path);
      const aecct_ref::RefFullQuantStats* single_eval_full_stats =
        (opts.run_mode == CliRunMode::EVAL_EXPERIMENT) ? &experiment_full_stats : nullptr;
      if (write_eval_single_summary_txt(
            txt_path,
            (opts.run_mode == CliRunMode::EVAL_BASELINE) ? "baseline" : "experiment",
            stats,
            rows,
            single_eval_full_stats)) {
        std::printf("Evaluator summary txt   : %s\n", txt_path.c_str());
      } else {
        std::printf("[warn] Failed to write evaluator summary txt: %s\n", txt_path.c_str());
      }
      const std::string timing_path = derive_timing_txt_path(csv_path);
      perf.file_io_s = elapsed_sec(t_fileio_start, now_tp());
      perf.total_s = elapsed_sec(t_program_start, now_tp());
      if (write_timing_txt(timing_path, perf)) {
        std::printf("Timing summary txt      : %s\n", timing_path.c_str());
      } else {
        std::printf("[warn] Failed to write timing summary txt: %s\n", timing_path.c_str());
      }

      print_eval_single_console_summary(
        (opts.run_mode == CliRunMode::EVAL_BASELINE) ? "baseline" : "experiment",
        stats
      );
      if (opts.run_mode == CliRunMode::EVAL_EXPERIMENT) {
        print_full_stress_console_summary(experiment_full_stats);
      }
      std::printf("=== Timing Breakdown (sec) ===\n");
      std::printf("startup/init             : %.6f\n", perf.startup_init_s);
      std::printf("baseline model run       : %.6f\n", perf.baseline_model_s);
      std::printf("experiment path run      : %.6f\n", perf.experiment_path_s);
      std::printf("eval aggregation         : %.6f\n", perf.compare_aggregation_s);
      std::printf("file I/O                 : %.6f\n", perf.file_io_s);
      std::printf("total runtime            : %.6f\n", perf.total_s);
      return 0;
    }

    std::vector<EvalComparePatternRow> rows;
    rows.reserve(static_cast<std::size_t>(range.count));
    EvalAggregateStats baseline_stats{};
    EvalAggregateStats experiment_stats{};
    init_eval_aggregate(baseline_stats);
    init_eval_aggregate(experiment_stats);
    for (int off = 0; off < range.count; ++off) {
      const int pattern = range.begin + off;
      const EvalPatternRow b_row = build_eval_pattern_row(
        &baseline_x_pred_batch[static_cast<std::size_t>(off * N)],
        pattern,
        N
      );
      const EvalPatternRow e_row = build_eval_pattern_row(
        &experiment_x_pred_batch[static_cast<std::size_t>(off * N)],
        pattern,
        N
      );
      update_eval_aggregate(baseline_stats, b_row);
      update_eval_aggregate(experiment_stats, e_row);

      EvalComparePatternRow row{};
      row.pattern_index = pattern;
      row.evaluated_bits = b_row.evaluated_bits;
      row.baseline_bit_errors = b_row.bit_errors;
      row.experiment_bit_errors = e_row.bit_errors;
      row.baseline_x_pred_match_count = b_row.x_pred_match_count;
      row.experiment_x_pred_match_count = e_row.x_pred_match_count;
      row.baseline_frame_error_flag = b_row.frame_error_flag;
      row.experiment_frame_error_flag = e_row.frame_error_flag;
      rows.push_back(row);

      if (!opts.summary_only) {
        const double b_ber = (b_row.evaluated_bits > 0U)
          ? (static_cast<double>(b_row.bit_errors) / static_cast<double>(b_row.evaluated_bits))
          : 0.0;
        const double e_ber = (e_row.evaluated_bits > 0U)
          ? (static_cast<double>(e_row.bit_errors) / static_cast<double>(e_row.evaluated_bits))
          : 0.0;
        std::printf("[pattern %d][eval-compare] b_err=%zu e_err=%zu b_ber=%.9e e_ber=%.9e b_fe=%d e_fe=%d\n",
          pattern,
          b_row.bit_errors,
          e_row.bit_errors,
          b_ber,
          e_ber,
          b_row.frame_error_flag,
          e_row.frame_error_flag);
      }
    }
    finalize_eval_aggregate(baseline_stats);
    finalize_eval_aggregate(experiment_stats);
    perf.compare_aggregation_s = elapsed_sec(t_eval_agg_start, now_tp());

    const auto t_fileio_start = now_tp();
    if (write_eval_compare_csv(csv_path, rows)) {
      std::printf("Per-pattern eval csv    : %s\n", csv_path.c_str());
    } else {
      std::printf("[warn] Failed to write per-pattern eval csv: %s\n", csv_path.c_str());
    }
    const std::string txt_path = derive_summary_txt_path(csv_path);
    if (write_eval_compare_summary_txt(txt_path, baseline_stats, experiment_stats, rows, &experiment_full_stats)) {
      std::printf("Evaluator summary txt   : %s\n", txt_path.c_str());
    } else {
      std::printf("[warn] Failed to write evaluator summary txt: %s\n", txt_path.c_str());
    }
    const std::string timing_path = derive_timing_txt_path(csv_path);
    perf.file_io_s = elapsed_sec(t_fileio_start, now_tp());
    perf.total_s = elapsed_sec(t_program_start, now_tp());
    if (write_timing_txt(timing_path, perf)) {
      std::printf("Timing summary txt      : %s\n", timing_path.c_str());
    } else {
      std::printf("[warn] Failed to write timing summary txt: %s\n", timing_path.c_str());
    }

    print_eval_compare_console_summary(baseline_stats, experiment_stats);
    print_full_stress_console_summary(experiment_full_stats);
    std::printf("=== Timing Breakdown (sec) ===\n");
    std::printf("startup/init             : %.6f\n", perf.startup_init_s);
    std::printf("baseline model run       : %.6f\n", perf.baseline_model_s);
    std::printf("experiment path run      : %.6f\n", perf.experiment_path_s);
    std::printf("eval aggregation         : %.6f\n", perf.compare_aggregation_s);
    std::printf("file I/O                 : %.6f\n", perf.file_io_s);
    std::printf("total runtime            : %.6f\n", perf.total_s);
    return 0;
  }

  const bool anchor_finalhead_s0 =
    precision_mode_anchors_to_finalhead_s0(opts.experiment_precision_mode);
  const bool use_frag_group_for_experiment =
    precision_mode_requires_frag_group(opts.experiment_precision_mode);
  const bool enable_fp16_stage_compare =
    aecct_ref::is_fp16_experiment_mode(opts.experiment_precision_mode);
  aecct_ref::RefModel baseline_model;
  aecct_ref::RefRunConfig baseline_cfg = aecct_ref::make_fp32_baseline_run_config();
  baseline_cfg.precision_mode = anchor_finalhead_s0
    ? aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD
    : aecct_ref::RefPrecisionMode::BASELINE_FP32;
  baseline_cfg.algo_variant = opts.algo_variant;
  baseline_cfg.ln_mode = opts.ln_mode;
  baseline_cfg.finalhead_stage = anchor_finalhead_s0
    ? aecct_ref::RefFinalHeadExploreStage::S0
    : opts.finalhead_stage;
  baseline_cfg.frag_group = aecct_ref::RefFragGroup::NONE;
  baseline_cfg.softmax_exp_mode = aecct_ref::RefSoftmaxExpMode::BASELINE_NEAREST_LUT;
  baseline_model.set_run_config(baseline_cfg);

  BatchCompareSummary batch{};
  init_batch_summary(batch);
  std::vector<PerPatternCompareRow> rows;
  rows.reserve(static_cast<std::size_t>(range.count));
  std::vector<double> baseline_logits_batch;
  std::vector<aecct_ref::bit1_t> baseline_x_pred_batch;
  std::vector<double> baseline_finalhead_s_t;
  std::vector<double> baseline_layer1_attn_input_batch;
  std::vector<double> baseline_end_norm_batch;
  std::vector<double> experiment_logits_batch;
  std::vector<aecct_ref::bit1_t> experiment_x_pred_batch;
  std::vector<double> experiment_finalhead_s_t;
  std::vector<double> experiment_layer1_attn_input_batch;
  std::vector<double> experiment_end_norm_batch;
  std::vector<LnDebugEntryRow> baseline_ln_debug_entries;
  std::vector<LnDebugEntryRow> experiment_ln_debug_entries;
  std::vector<LnUpstreamEntryRow> baseline_ln_upstream_entries;
  std::vector<LnUpstreamEntryRow> experiment_ln_upstream_entries;
  aecct_ref::RefFullQuantStats experiment_full_stats{};

  const bool experiment_use_reconstruct =
    (opts.experiment_precision_mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD) &&
    (opts.experiment_ln_mode == opts.ln_mode) &&
    !opts.ln_debug &&
    !opts.ln_var_debug &&
    !opts.ln_input_debug &&
    !opts.ln_upstream_debug;
  const bool capture_baseline_s_t = experiment_use_reconstruct || enable_fp16_stage_compare;
  if (capture_baseline_s_t) {
    baseline_finalhead_s_t.resize(static_cast<std::size_t>(range.count * STAGE_TOKENS));
  }
  if (enable_fp16_stage_compare) {
    const std::size_t stage_tensor_count =
      static_cast<std::size_t>(range.count * STAGE_TOKENS * STAGE_D_MODEL);
    baseline_layer1_attn_input_batch.resize(stage_tensor_count);
    baseline_end_norm_batch.resize(stage_tensor_count);
    experiment_finalhead_s_t.resize(static_cast<std::size_t>(range.count * STAGE_TOKENS));
    experiment_layer1_attn_input_batch.resize(stage_tensor_count);
    experiment_end_norm_batch.resize(stage_tensor_count);
  }

  const bool enable_ln_trace = opts.ln_debug || opts.ln_var_debug || opts.ln_input_debug;
  const bool enable_ln_upstream_trace = opts.ln_upstream_debug;
  aecct_ref::set_ref_ln_debug_enabled(enable_ln_trace);
  aecct_ref::set_ref_ln_upstream_debug_enabled(enable_ln_upstream_trace);
  if (enable_ln_trace) {
    aecct_ref::reset_ref_ln_debug_trace();
  }
  if (enable_ln_upstream_trace) {
    aecct_ref::reset_ref_ln_upstream_debug_trace();
  }
  const auto t_baseline_start = now_tp();
  run_ref_batch(
    baseline_model,
    range,
    N,
    baseline_logits_batch,
    baseline_x_pred_batch,
    capture_baseline_s_t ? baseline_finalhead_s_t.data() : nullptr,
    enable_fp16_stage_compare ? baseline_layer1_attn_input_batch.data() : nullptr,
    enable_fp16_stage_compare ? baseline_end_norm_batch.data() : nullptr
  );
  perf.baseline_model_s = elapsed_sec(t_baseline_start, now_tp());
  if (enable_ln_trace) {
    if (!collect_ln_debug_entries(baseline_ln_debug_entries)) {
      std::printf("[warn] Failed to collect baseline LN debug entries.\n");
    }
    aecct_ref::reset_ref_ln_debug_trace();
  }
  if (enable_ln_upstream_trace) {
    if (!collect_ln_upstream_debug_entries(baseline_ln_upstream_entries)) {
      std::printf("[warn] Failed to collect baseline LN upstream debug entries.\n");
    }
    aecct_ref::reset_ref_ln_upstream_debug_trace();
  }

  const auto t_experiment_start = now_tp();
  aecct_ref::reset_ref_full_quant_stats();
  if (experiment_use_reconstruct) {
    run_experiment_from_baseline_finalhead(
      range,
      N,
      opts.finalhead_stage,
      baseline_finalhead_s_t,
      experiment_logits_batch,
      experiment_x_pred_batch
    );
  } else {
    aecct_ref::RefModel experiment_model;
    aecct_ref::RefRunConfig experiment_cfg =
      aecct_ref::is_fp16_experiment_mode(opts.experiment_precision_mode)
        ? aecct_ref::make_fp16_experiment_run_config()
        : aecct_ref::make_fp32_baseline_run_config();
    experiment_cfg.precision_mode = opts.experiment_precision_mode;
    experiment_cfg.algo_variant = opts.algo_variant;
    experiment_cfg.ln_mode = opts.experiment_ln_mode;
    experiment_cfg.finalhead_stage = anchor_finalhead_s0
      ? aecct_ref::RefFinalHeadExploreStage::S0
      : opts.finalhead_stage;
    experiment_cfg.frag_group = use_frag_group_for_experiment
      ? opts.frag_group
      : aecct_ref::RefFragGroup::NONE;
    experiment_cfg.softmax_exp_mode = opts.softmax_exp_mode;
    experiment_model.set_run_config(experiment_cfg);
    run_ref_batch(
      experiment_model,
      range,
      N,
      experiment_logits_batch,
      experiment_x_pred_batch,
      enable_fp16_stage_compare ? experiment_finalhead_s_t.data() : nullptr,
      enable_fp16_stage_compare ? experiment_layer1_attn_input_batch.data() : nullptr,
      enable_fp16_stage_compare ? experiment_end_norm_batch.data() : nullptr
    );
  }
  experiment_full_stats = aecct_ref::get_ref_full_quant_stats();
  perf.experiment_path_s = elapsed_sec(t_experiment_start, now_tp());
  if (enable_ln_trace) {
    if (!collect_ln_debug_entries(experiment_ln_debug_entries)) {
      std::printf("[warn] Failed to collect experiment LN debug entries.\n");
    }
  }
  if (enable_ln_upstream_trace) {
    if (!collect_ln_upstream_debug_entries(experiment_ln_upstream_entries)) {
      std::printf("[warn] Failed to collect experiment LN upstream debug entries.\n");
    }
  }
  aecct_ref::set_ref_ln_debug_enabled(false);
  aecct_ref::set_ref_ln_upstream_debug_enabled(false);

  const auto t_compare_start = now_tp();
  for (int off = 0; off < range.count; ++off) {
    const int pattern = range.begin + off;
    const std::size_t base_idx = static_cast<std::size_t>(off * N);
    PerPatternCompareRow row{};
    row.pattern_index = pattern;
    row.baseline_vs_golden = compute_pattern_logits_vs_golden(baseline_logits_batch, off, pattern, N);
    row.experiment_vs_golden = compute_pattern_logits_vs_golden(experiment_logits_batch, off, pattern, N);
    row.cmp = aecct_ref::compute_experiment_compare_metrics(
      &baseline_logits_batch[base_idx],
      &experiment_logits_batch[base_idx],
      &baseline_x_pred_batch[base_idx],
      &experiment_x_pred_batch[base_idx],
      static_cast<std::size_t>(N),
      static_cast<std::size_t>(N),
      static_cast<std::size_t>(opts.topk)
    );

    rows.push_back(row);
    update_batch_summary(batch, row);

    if (!opts.summary_only) {
      std::printf("[pattern %d] mse=%.9e mae=%.9e maxabs=%.9e xpred_flip=%zu (%.9e) sign_flip=%zu margin(b/e)=%.9e/%.9e naninf(b)=%zu/%zu naninf(e)=%zu/%zu b_mse_g=%.9e e_mse_g=%.9e xmatch(b/e)=%zu/%zu,%zu/%zu\n",
        row.pattern_index,
        row.cmp.logits_diff.mse,
        row.cmp.logits_diff.mae,
        row.cmp.logits_diff.max_abs,
        row.cmp.x_pred_mismatch_count,
        row.cmp.x_pred_mismatch_ratio,
        row.cmp.sign_flip_count,
        row.cmp.baseline_min_abs_margin,
        row.cmp.experiment_min_abs_margin,
        row.cmp.baseline_logits_nonfinite.nan_count,
        row.cmp.baseline_logits_nonfinite.inf_count,
        row.cmp.experiment_logits_nonfinite.nan_count,
        row.cmp.experiment_logits_nonfinite.inf_count,
        row.baseline_vs_golden.mse,
        row.experiment_vs_golden.mse,
        compute_pattern_xpred_match_vs_golden(baseline_x_pred_batch, off, pattern, N),
        static_cast<std::size_t>(N),
        compute_pattern_xpred_match_vs_golden(experiment_x_pred_batch, off, pattern, N),
        static_cast<std::size_t>(N));
    }
  }
  perf.compare_aggregation_s = elapsed_sec(t_compare_start, now_tp());

  const DistributionStats margin_dist = compute_distribution_stats(batch.per_pattern_min_margin);
  const std::vector<std::size_t> vulnerable_idx = collect_top_vulnerable_indices(rows, 5U);
  StageTensorDiffSummary stage_mid_norm{};
  StageTensorDiffSummary stage_layer1_attn_input{};
  StageTensorDiffSummary stage_end_norm{};
  StageTensorDiffSummary stage_s_t{};
  StageTensorDiffSummary stage_logits{};
  StageBitDiffSummary stage_x_pred{};
  const bool has_fp16_stage_compare_data =
    enable_fp16_stage_compare &&
    !experiment_use_reconstruct &&
    baseline_layer1_attn_input_batch.size() == experiment_layer1_attn_input_batch.size() &&
    baseline_end_norm_batch.size() == experiment_end_norm_batch.size() &&
    baseline_finalhead_s_t.size() == experiment_finalhead_s_t.size() &&
    baseline_logits_batch.size() == experiment_logits_batch.size() &&
    baseline_x_pred_batch.size() == experiment_x_pred_batch.size();
  if (enable_fp16_stage_compare && !has_fp16_stage_compare_data) {
    std::printf("[warn] FP16 stage compare data incomplete, skip stage drift summary.\n");
  }
  if (has_fp16_stage_compare_data) {
    stage_mid_norm = compare_stage_tensor_diff(
      "mid_norm",
      baseline_layer1_attn_input_batch.data(),
      experiment_layer1_attn_input_batch.data(),
      range.begin,
      range.count,
      STAGE_TOKENS,
      STAGE_D_MODEL);
    stage_layer1_attn_input = stage_mid_norm;
    stage_layer1_attn_input.name = "layer1_attn_input";
    stage_end_norm = compare_stage_tensor_diff(
      "end_norm",
      baseline_end_norm_batch.data(),
      experiment_end_norm_batch.data(),
      range.begin,
      range.count,
      STAGE_TOKENS,
      STAGE_D_MODEL);
    stage_s_t = compare_stage_tensor_diff(
      "s_t",
      baseline_finalhead_s_t.data(),
      experiment_finalhead_s_t.data(),
      range.begin,
      range.count,
      STAGE_TOKENS,
      1);
    stage_logits = compare_stage_tensor_diff(
      "logits",
      baseline_logits_batch.data(),
      experiment_logits_batch.data(),
      range.begin,
      range.count,
      N,
      1);
    stage_x_pred = compare_xpred_diff(
      "x_pred",
      baseline_x_pred_batch,
      experiment_x_pred_batch,
      range.begin,
      range.count,
      N);
  }

  const std::string mode_suffix = precision_mode_output_suffix(
    opts.experiment_precision_mode,
    opts.frag_group
  );
  const std::string csv_path = !opts.summary_csv_path.empty()
    ? opts.summary_csv_path
    : ("build/ref_eval/compare_summary_begin" + std::to_string(range.begin) +
       "_count" + std::to_string(range.count) + mode_suffix + ".csv");
  const auto t_fileio_start = now_tp();
  if (write_compare_csv(csv_path, rows)) {
    std::printf("Per-pattern summary csv : %s\n", csv_path.c_str());
  } else {
    std::printf("[warn] Failed to write csv summary: %s\n", csv_path.c_str());
  }
  const std::string txt_path = derive_summary_txt_path(csv_path);
  if (write_batch_summary_txt(txt_path, batch, margin_dist, rows, vulnerable_idx, &experiment_full_stats)) {
    std::printf("Reviewer summary txt    : %s\n", txt_path.c_str());
  } else {
    std::printf("[warn] Failed to write reviewer summary txt: %s\n", txt_path.c_str());
  }
  if (has_fp16_stage_compare_data) {
    const std::string stage_compare_path = csv_path + ".stage_compare.txt";
    if (write_stage_compare_summary_txt(
          stage_compare_path,
          stage_mid_norm,
          stage_layer1_attn_input,
          stage_end_norm,
          stage_s_t,
          stage_logits,
          stage_x_pred)) {
      std::printf("Stage compare txt       : %s\n", stage_compare_path.c_str());
    } else {
      std::printf("[warn] Failed to write stage compare txt: %s\n", stage_compare_path.c_str());
    }
  }
  const std::string timing_path = derive_timing_txt_path(csv_path);
  if (opts.ln_debug) {
    const std::string ln_debug_path = csv_path + ".ln_debug.txt";
    dump_ln_debug_report(
      range.begin,
      range.count,
      baseline_ln_debug_entries,
      experiment_ln_debug_entries,
      ln_debug_path
    );
    std::printf("LN debug report txt     : %s\n", ln_debug_path.c_str());
  }
  if (opts.ln_var_debug) {
    const std::string ln_var_debug_path = csv_path + ".ln_var_debug.txt";
    dump_ln_var_debug_report(
      range.begin,
      range.count,
      baseline_ln_debug_entries,
      experiment_ln_debug_entries,
      ln_var_debug_path
    );
    std::printf("LN var debug report txt : %s\n", ln_var_debug_path.c_str());
  }
  if (opts.ln_input_debug) {
    const std::string ln_input_debug_path = csv_path + ".ln_input_debug.txt";
    dump_ln_input_debug_report(
      range.begin,
      range.count,
      baseline_ln_debug_entries,
      experiment_ln_debug_entries,
      ln_input_debug_path
    );
    std::printf("LN input debug report txt: %s\n", ln_input_debug_path.c_str());
  }
  if (opts.ln_upstream_debug) {
    const std::string ln_upstream_debug_path = csv_path + ".ln_upstream_debug.txt";
    dump_ln_upstream_debug_report(
      range.begin,
      range.count,
      baseline_ln_upstream_entries,
      experiment_ln_upstream_entries,
      ln_upstream_debug_path
    );
    std::printf("LN upstream debug report txt: %s\n", ln_upstream_debug_path.c_str());
  }
  perf.file_io_s = elapsed_sec(t_fileio_start, now_tp());
  perf.total_s = elapsed_sec(t_program_start, now_tp());
  if (write_timing_txt(timing_path, perf)) {
    std::printf("Timing summary txt      : %s\n", timing_path.c_str());
  } else {
    std::printf("[warn] Failed to write timing summary txt: %s\n", timing_path.c_str());
  }

  std::printf("=== Batch Compare Summary ===\n");
  std::printf("total patterns scanned        : %d\n", batch.total_patterns_scanned);
  std::printf("patterns with x_pred flip     : %zu\n", batch.patterns_with_xpred_flip);
  std::printf("patterns with sign flip       : %zu\n", batch.patterns_with_sign_flip);
  std::printf("total flipped bits            : %zu\n", batch.total_flipped_bits);
  std::printf("worst logits MSE              : %.9e (pattern %d)\n",
    batch.worst_logits_mse, batch.worst_logits_mse_pattern);
  std::printf("worst max abs diff            : %.9e (pattern %d)\n",
    batch.worst_max_abs_diff, batch.worst_max_abs_diff_pattern);
  std::printf("worst min margin              : %.9e (pattern %d)\n",
    batch.worst_min_margin, batch.worst_min_margin_pattern);
  std::printf("max sign-flip count           : %zu (pattern %d)\n",
    batch.max_sign_flip_count, batch.max_sign_flip_pattern);
  std::printf("total sign flips              : %zu\n", batch.total_sign_flip_count);
  std::printf("total baseline NaN / Inf      : %zu / %zu\n",
    batch.baseline_nonfinite_total.nan_count, batch.baseline_nonfinite_total.inf_count);
  std::printf("total experiment NaN / Inf    : %zu / %zu\n",
    batch.experiment_nonfinite_total.nan_count, batch.experiment_nonfinite_total.inf_count);
  print_full_stress_console_summary(experiment_full_stats);
  std::printf("margin dist (risk=min(b/e))   : min=%.9e max=%.9e mean=%.9e median=%.9e\n",
    margin_dist.min_v, margin_dist.max_v, margin_dist.mean_v, margin_dist.median_v);
  std::printf("top vulnerable patterns (by min margin):\n");
  for (std::size_t i = 0; i < vulnerable_idx.size(); ++i) {
    const PerPatternCompareRow& r = rows[vulnerable_idx[i]];
    std::printf("  rank%zu p=%d margin(b/e)=%.9e/%.9e mse=%.9e mae=%.9e maxabs=%.9e xflip=%zu sflip=%zu\n",
      i + 1U,
      r.pattern_index,
      r.cmp.baseline_min_abs_margin,
      r.cmp.experiment_min_abs_margin,
      r.cmp.logits_diff.mse,
      r.cmp.logits_diff.mae,
      r.cmp.logits_diff.max_abs,
      r.cmp.x_pred_mismatch_count,
      r.cmp.sign_flip_count);
  }
  if (has_fp16_stage_compare_data) {
    std::printf("=== FP16 Stage Drift Summary (Ref-vs-Ref) ===\n");
    print_stage_tensor_diff_summary(stage_mid_norm, "token");
    print_stage_tensor_diff_summary(stage_layer1_attn_input, "token");
    print_stage_tensor_diff_summary(stage_end_norm, "token");
    print_stage_tensor_diff_summary(stage_s_t, "token");
    print_stage_tensor_diff_summary(stage_logits, "index");
    print_stage_bit_diff_summary(stage_x_pred);
  }
  std::printf("=== Timing Breakdown (sec) ===\n");
  std::printf("startup/init             : %.6f\n", perf.startup_init_s);
  std::printf("baseline model run       : %.6f\n", perf.baseline_model_s);
  std::printf("experiment path run      : %.6f\n", perf.experiment_path_s);
  std::printf("compare aggregation      : %.6f\n", perf.compare_aggregation_s);
  std::printf("file I/O                 : %.6f\n", perf.file_io_s);
  std::printf("total runtime            : %.6f\n", perf.total_s);
  std::printf("BER/FER evaluator             : run with --mode eval-compare (target_x from output_x_pred_step0.h)\n");

  return 0;
}
