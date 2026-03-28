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
  EXPERIMENT_ONLY = 2
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

static inline std::chrono::steady_clock::time_point now_tp() {
  return std::chrono::steady_clock::now();
}

static inline double elapsed_sec(
  const std::chrono::steady_clock::time_point& t0,
  const std::chrono::steady_clock::time_point& t1
) {
  return std::chrono::duration<double>(t1 - t0).count();
}

static void print_usage() {
  std::printf("Usage: ref_sim [pattern_index] [options]\n");
  std::printf("Options:\n");
  std::printf("  --mode compare|baseline|experiment\n");
  std::printf("  --pattern N\n");
  std::printf("  --pattern-begin N --pattern-count M\n");
  std::printf("  --topk K\n");
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

static CliParseResult parse_cli(int argc, char** argv, CliOptions& opts) {
  opts.run_mode = CliRunMode::COMPARE;
  opts.pattern_index = -1;
  opts.pattern_begin = -1;
  opts.pattern_count = -1;
  opts.topk = 5;
  opts.summary_only = false;
  opts.summary_csv_path.clear();
  opts.algo_variant = aecct_ref::RefAlgoVariant::BASELINE_SPEC_FLOW;

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
    if (std::strcmp(arg, "--summary-only") == 0 || std::strcmp(arg, "--quiet") == 0) {
      opts.summary_only = true;
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

  for (int b = 0; b < run_b; ++b) {
    aecct_ref::ref_fp32_t out_fc_in[kTokensT];
    for (int t = 0; t < kTokensT; ++t) {
      const double s_t = baseline_finalhead_s_t[static_cast<std::size_t>(b * kTokensT + t)];
      aecct_ref::ref_fp32_t x = aecct_ref::ref_fp32_t(static_cast<float>(s_t));
      out_fc_in[t] = aecct_ref::roundtrip_through_generic_e4m3(x);
    }

    const int src_pattern = range.begin + b;
    const double* input_y = &trace_input_y_step0_tensor[src_pattern * n_vars];
    for (int n = 0; n < kVars; ++n) {
      aecct_ref::ref_fp32_t acc = aecct_ref::ref_fp32_t(static_cast<float>(w_out_fc_bias[n]));
      for (int t = 0; t < kTokensT; ++t) {
        acc += aecct_ref::ref_fp32_t(static_cast<float>(w_out_fc_weight[n * kTokensT + t])) * out_fc_in[t];
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
  const std::vector<std::size_t>& vulnerable_idx
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

  std::printf("Run config:\n");
  std::printf("  mode           : %s\n",
    (opts.run_mode == CliRunMode::COMPARE) ? "compare"
      : ((opts.run_mode == CliRunMode::BASELINE_ONLY) ? "baseline" : "experiment"));
  std::printf("  precision(base): %s\n", aecct_ref::to_string(aecct_ref::RefPrecisionMode::BASELINE_FP32));
  std::printf("  precision(exp) : %s\n", aecct_ref::to_string(aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD));
  std::printf("  algo_variant   : %s\n", aecct_ref::to_string(opts.algo_variant));
  std::printf("  pattern_range  : begin=%d count=%d\n", range.begin, range.count);
  std::printf("  topk           : %d\n", opts.topk);
  std::printf("  summary_only   : %d\n", opts.summary_only ? 1 : 0);

  perf.startup_init_s = elapsed_sec(t_program_start, now_tp());

  if (opts.run_mode == CliRunMode::BASELINE_ONLY) {
    return run_single_mode(aecct_ref::RefPrecisionMode::BASELINE_FP32, "baseline", range, N, opts);
  }
  if (opts.run_mode == CliRunMode::EXPERIMENT_ONLY) {
    return run_single_mode(aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD, "experiment", range, N, opts);
  }

  aecct_ref::RefModel baseline_model;
  aecct_ref::RefRunConfig baseline_cfg{};
  baseline_cfg.precision_mode = aecct_ref::RefPrecisionMode::BASELINE_FP32;
  baseline_cfg.algo_variant = opts.algo_variant;
  baseline_model.set_run_config(baseline_cfg);

  BatchCompareSummary batch{};
  init_batch_summary(batch);
  std::vector<PerPatternCompareRow> rows;
  rows.reserve(static_cast<std::size_t>(range.count));
  std::vector<double> baseline_logits_batch;
  std::vector<aecct_ref::bit1_t> baseline_x_pred_batch;
  std::vector<double> baseline_finalhead_s_t;
  baseline_finalhead_s_t.resize(static_cast<std::size_t>(range.count * 75));
  std::vector<double> experiment_logits_batch;
  std::vector<aecct_ref::bit1_t> experiment_x_pred_batch;

  const auto t_baseline_start = now_tp();
  run_ref_batch(
    baseline_model,
    range,
    N,
    baseline_logits_batch,
    baseline_x_pred_batch,
    baseline_finalhead_s_t.data()
  );
  perf.baseline_model_s = elapsed_sec(t_baseline_start, now_tp());

  const auto t_experiment_start = now_tp();
  run_experiment_from_baseline_finalhead(
    range,
    N,
    baseline_finalhead_s_t,
    experiment_logits_batch,
    experiment_x_pred_batch
  );
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

  const std::string csv_path = !opts.summary_csv_path.empty()
    ? opts.summary_csv_path
    : ("build/ref_eval/compare_summary_begin" + std::to_string(range.begin) +
       "_count" + std::to_string(range.count) + ".csv");
  const auto t_fileio_start = now_tp();
  if (write_compare_csv(csv_path, rows)) {
    std::printf("Per-pattern summary csv : %s\n", csv_path.c_str());
  } else {
    std::printf("[warn] Failed to write csv summary: %s\n", csv_path.c_str());
  }
  const std::string txt_path = derive_summary_txt_path(csv_path);
  if (write_batch_summary_txt(txt_path, batch, margin_dist, rows, vulnerable_idx)) {
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
  std::printf("BER/FER evaluator             : unavailable in this AECCT_ac_ref flow (baseline-relative compare only)\n");

  return 0;
}
