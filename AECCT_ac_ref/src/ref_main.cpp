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
  std::string summary_csv_path;
  aecct_ref::RefAlgoVariant algo_variant;
  aecct_ref::RefFinalHeadExploreStage finalhead_stage;
  aecct_ref::RefPrecisionMode experiment_precision_mode;
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

static void print_usage() {
  std::printf("Usage: ref_sim [pattern_index] [options]\n");
  std::printf("Options:\n");
  std::printf("  --mode compare|baseline|experiment|eval-baseline|eval-experiment|eval-compare|explore\n");
  std::printf("  --pattern N\n");
  std::printf("  --pattern-begin N --pattern-count M\n");
  std::printf("  --topk K\n");
  std::printf("  --stage S0|S1|S2|S3|S4\n");
  std::printf("  --precision-exp baseline_fp32|generic_e4m3_finalhead|full_e4m3_nonlinear_stress|generic_e4m3_frag_bisect|generic_e4m3_except_g5|generic_e4m3_g5_g4|generic_e4m3_g5_g1|generic_e4m3_g5_g3|generic_e4m3_g5_g2|generic_e4m3_g2_embed_only|generic_e4m3_g2_spe_only|generic_e4m3_g2_preproc_assembly|generic_e4m3_g2_prelayer_handoff\n");
  std::printf("  --frag-group NONE|G1|G2|G3|G4|G5|C1|C2|C3|C4\n");
  std::printf("  --summary-only\n");
  std::printf("  --quiet (alias of --summary-only)\n");
  std::printf("  --summary-csv PATH\n");
  std::printf("  --algo baseline_spec_flow|reserved_softmax_alt|reserved_finalhead_alt\n");
  std::printf("  --help\n");
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
  if (std::strcmp(text, "baseline_fp32") == 0) {
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

static CliParseResult parse_cli(int argc, char** argv, CliOptions& opts) {
  opts.run_mode = CliRunMode::COMPARE;
  opts.pattern_index = -1;
  opts.pattern_begin = -1;
  opts.pattern_count = -1;
  opts.topk = 5;
  opts.summary_only = false;
  opts.summary_csv_path.clear();
  opts.algo_variant = aecct_ref::RefAlgoVariant::BASELINE_SPEC_FLOW;
  opts.finalhead_stage = aecct_ref::RefFinalHeadExploreStage::S0;
  opts.experiment_precision_mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD;
  opts.frag_group = aecct_ref::RefFragGroup::NONE;

  bool positional_pattern_used = false;
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
         mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_G2_PRELAYER_HANDOFF;
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
  double* out_finalhead_s_t
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
    ofs << "=== Full E4M3 Stress Counters (experiment) ===\n";
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
    ofs << "=== Full E4M3 Stress Counters (experiment) ===\n";
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
    ofs << "=== Full E4M3 Stress Counters (experiment) ===\n";
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
  std::printf("=== Full E4M3 Stress Counters (experiment) ===\n");
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
  aecct_ref::RefRunConfig baseline_cfg{};
  baseline_cfg.precision_mode = aecct_ref::RefPrecisionMode::BASELINE_FP32;
  baseline_cfg.algo_variant = opts.algo_variant;
  baseline_cfg.finalhead_stage = stage;
  baseline_model.set_run_config(baseline_cfg);

  std::vector<double> baseline_logits_batch;
  std::vector<aecct_ref::bit1_t> baseline_x_pred_batch;
  std::vector<double> baseline_finalhead_s_t;
  baseline_finalhead_s_t.resize(static_cast<std::size_t>(range.count * 75));
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
      std::printf("[stage %s][pattern %d] mse=%.9e maxabs=%.9e xflip=%zu sflip=%zu min_margin=%.9e b_err=%zu e_err=%zu\n",
        aecct_ref::to_string(stage),
        pattern,
        cmp_row.cmp.logits_diff.mse,
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

static int run_single_mode(
  aecct_ref::RefPrecisionMode precision_mode,
  const char* tag,
  const PatternRange& range,
  int n_vars,
  const CliOptions& opts
) {
  aecct_ref::RefModel model;
  aecct_ref::RefRunConfig cfg{};
  cfg.precision_mode = precision_mode;
  cfg.algo_variant = opts.algo_variant;
  cfg.finalhead_stage = opts.finalhead_stage;
  cfg.frag_group = opts.frag_group;
  model.set_run_config(cfg);

  GoldenAggregateMetrics agg{};
  init_golden_aggregate(agg);

  for (int off = 0; off < range.count; ++off) {
    const int p = range.begin + off;
    RefRunOutputs run{};
    run_ref_single_pattern(model, p, n_vars, run);
    if (range.count == 1) {
      print_vs_golden_summary(tag, run);
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

  std::printf("Run config:\n");
  const aecct_ref::RefPrecisionMode effective_baseline_precision =
    precision_mode_anchors_to_finalhead_s0(opts.experiment_precision_mode)
      ? aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD
      : aecct_ref::RefPrecisionMode::BASELINE_FP32;
  std::printf("  mode           : %s\n", run_mode_to_string(opts.run_mode));
  std::printf("  precision(base): %s\n", aecct_ref::to_string(effective_baseline_precision));
  std::printf("  precision(exp) : %s\n", aecct_ref::to_string(opts.experiment_precision_mode));
  std::printf("  finalhead_stage: %s\n", aecct_ref::to_string(opts.finalhead_stage));
  std::printf("  frag_group     : %s\n", aecct_ref::to_string(opts.frag_group));
  std::printf("  algo_variant   : %s\n", aecct_ref::to_string(opts.algo_variant));
  std::printf("  pattern_range  : begin=%d count=%d\n", range.begin, range.count);
  std::printf("  topk           : %d\n", opts.topk);
  std::printf("  summary_only   : %d\n", opts.summary_only ? 1 : 0);

  perf.startup_init_s = elapsed_sec(t_program_start, now_tp());

  if (opts.run_mode == CliRunMode::BASELINE_ONLY) {
    return run_single_mode(aecct_ref::RefPrecisionMode::BASELINE_FP32, "baseline", range, N, opts);
  }
  if (opts.run_mode == CliRunMode::EXPERIMENT_ONLY) {
    return run_single_mode(opts.experiment_precision_mode, "experiment", range, N, opts);
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
    aecct_ref::RefModel baseline_model_eval;
    aecct_ref::RefRunConfig baseline_cfg_eval{};
    baseline_cfg_eval.precision_mode = anchor_finalhead_s0
      ? aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD
      : aecct_ref::RefPrecisionMode::BASELINE_FP32;
    baseline_cfg_eval.algo_variant = opts.algo_variant;
    baseline_cfg_eval.finalhead_stage = anchor_finalhead_s0
      ? aecct_ref::RefFinalHeadExploreStage::S0
      : opts.finalhead_stage;
    baseline_cfg_eval.frag_group = aecct_ref::RefFragGroup::NONE;
    baseline_model_eval.set_run_config(baseline_cfg_eval);

    std::vector<double> baseline_logits_batch;
    std::vector<aecct_ref::bit1_t> baseline_x_pred_batch;
    std::vector<double> baseline_finalhead_s_t;
    std::vector<double> experiment_logits_batch;
    std::vector<aecct_ref::bit1_t> experiment_x_pred_batch;
    aecct_ref::RefFullQuantStats experiment_full_stats{};

    const bool need_experiment_outputs =
      (opts.run_mode == CliRunMode::EVAL_EXPERIMENT || opts.run_mode == CliRunMode::EVAL_COMPARE);
    const bool experiment_use_reconstruct =
      need_experiment_outputs &&
      (opts.experiment_precision_mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD);
    if (experiment_use_reconstruct) {
      baseline_finalhead_s_t.resize(static_cast<std::size_t>(range.count * 75));
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
        aecct_ref::RefRunConfig experiment_cfg_eval{};
        experiment_cfg_eval.precision_mode = opts.experiment_precision_mode;
        experiment_cfg_eval.algo_variant = opts.algo_variant;
        experiment_cfg_eval.finalhead_stage = anchor_finalhead_s0
          ? aecct_ref::RefFinalHeadExploreStage::S0
          : opts.finalhead_stage;
        experiment_cfg_eval.frag_group = use_frag_group_for_experiment
          ? opts.frag_group
          : aecct_ref::RefFragGroup::NONE;
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
  aecct_ref::RefModel baseline_model;
  aecct_ref::RefRunConfig baseline_cfg{};
  baseline_cfg.precision_mode = anchor_finalhead_s0
    ? aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD
    : aecct_ref::RefPrecisionMode::BASELINE_FP32;
  baseline_cfg.algo_variant = opts.algo_variant;
  baseline_cfg.finalhead_stage = anchor_finalhead_s0
    ? aecct_ref::RefFinalHeadExploreStage::S0
    : opts.finalhead_stage;
  baseline_cfg.frag_group = aecct_ref::RefFragGroup::NONE;
  baseline_model.set_run_config(baseline_cfg);

  BatchCompareSummary batch{};
  init_batch_summary(batch);
  std::vector<PerPatternCompareRow> rows;
  rows.reserve(static_cast<std::size_t>(range.count));
  std::vector<double> baseline_logits_batch;
  std::vector<aecct_ref::bit1_t> baseline_x_pred_batch;
  std::vector<double> baseline_finalhead_s_t;
  std::vector<double> experiment_logits_batch;
  std::vector<aecct_ref::bit1_t> experiment_x_pred_batch;
  aecct_ref::RefFullQuantStats experiment_full_stats{};

  const bool experiment_use_reconstruct =
    (opts.experiment_precision_mode == aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD);
  if (experiment_use_reconstruct) {
    baseline_finalhead_s_t.resize(static_cast<std::size_t>(range.count * 75));
  }

  const auto t_baseline_start = now_tp();
  run_ref_batch(
    baseline_model,
    range,
    N,
    baseline_logits_batch,
    baseline_x_pred_batch,
    experiment_use_reconstruct ? baseline_finalhead_s_t.data() : nullptr
  );
  perf.baseline_model_s = elapsed_sec(t_baseline_start, now_tp());

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
    aecct_ref::RefRunConfig experiment_cfg{};
    experiment_cfg.precision_mode = opts.experiment_precision_mode;
    experiment_cfg.algo_variant = opts.algo_variant;
    experiment_cfg.finalhead_stage = anchor_finalhead_s0
      ? aecct_ref::RefFinalHeadExploreStage::S0
      : opts.finalhead_stage;
    experiment_cfg.frag_group = use_frag_group_for_experiment
      ? opts.frag_group
      : aecct_ref::RefFragGroup::NONE;
    experiment_model.set_run_config(experiment_cfg);
    run_ref_batch(
      experiment_model,
      range,
      N,
      experiment_logits_batch,
      experiment_x_pred_batch,
      nullptr
    );
  }
  experiment_full_stats = aecct_ref::get_ref_full_quant_stats();
  perf.experiment_path_s = elapsed_sec(t_experiment_start, now_tp());

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
      std::printf("[pattern %d] mse=%.9e maxabs=%.9e xpred_flip=%zu (%.9e) sign_flip=%zu margin(b/e)=%.9e/%.9e naninf(b)=%zu/%zu naninf(e)=%zu/%zu b_mse_g=%.9e e_mse_g=%.9e xmatch(b/e)=%zu/%zu,%zu/%zu\n",
        row.pattern_index,
        row.cmp.logits_diff.mse,
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
  const std::string timing_path = derive_timing_txt_path(csv_path);
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
    std::printf("  rank%zu p=%d margin(b/e)=%.9e/%.9e mse=%.9e maxabs=%.9e xflip=%zu sflip=%zu\n",
      i + 1U,
      r.pattern_index,
      r.cmp.baseline_min_abs_margin,
      r.cmp.experiment_min_abs_margin,
      r.cmp.logits_diff.mse,
      r.cmp.logits_diff.max_abs,
      r.cmp.x_pred_mismatch_count,
      r.cmp.sign_flip_count);
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
