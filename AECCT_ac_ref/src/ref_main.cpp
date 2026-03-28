#include <algorithm>
#include <cmath>
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
#include "../include/RefModel.h"
#include "../include/RefPrecisionMode.h"

#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"

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
  aecct_ref::RefNonFiniteCounters baseline_nonfinite_total;
  aecct_ref::RefNonFiniteCounters experiment_nonfinite_total;
};

struct GoldenAggregateMetrics {
  double se_sum;
  double ae_sum;
  double max_abs;
  std::size_t logits_count;
  std::size_t x_pred_match_count;
  std::size_t x_pred_total_count;
};

static void print_usage() {
  std::printf("Usage: ref_sim [pattern_index] [options]\n");
  std::printf("Options:\n");
  std::printf("  --mode compare|baseline|experiment\n");
  std::printf("  --pattern N\n");
  std::printf("  --pattern-begin N --pattern-count M\n");
  std::printf("  --topk K\n");
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
  io.B = 1;
  io.N = n_vars;
  model.infer_step0(io);

  out.logits_vs_golden = aecct_ref::compute_metrics(golden_logits, out.logits.data(), out.logits.size());
  out.x_pred_match_count = compute_x_pred_match_count(golden_x_pred, out.x_pred);
  out.x_pred_total_count = out.x_pred.size();
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

static void print_topk_entries(
  const char* tag,
  const std::vector<aecct_ref::RefExperimentCompareMetrics::TopKEntry>& entries
) {
  std::printf("%s", tag);
  for (std::size_t i = 0; i < entries.size(); ++i) {
    std::printf(" [%zu]idx=%zu,v=%.6e,|v|=%.6e",
      i, entries[i].index, entries[i].value, entries[i].abs_value);
  }
  std::printf("\n");
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
  s.baseline_nonfinite_total = {};
  s.experiment_nonfinite_total = {};
}

static void update_batch_summary(BatchCompareSummary& s, const PerPatternCompareRow& row) {
  s.total_patterns_scanned += 1;
  s.total_flipped_bits += row.cmp.x_pred_mismatch_count;
  if (row.cmp.x_pred_mismatch_count > 0U) {
    s.patterns_with_xpred_flip += 1;
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

  if (opts.run_mode == CliRunMode::BASELINE_ONLY) {
    return run_single_mode(aecct_ref::RefPrecisionMode::BASELINE_FP32, "baseline", range, N, opts);
  }
  if (opts.run_mode == CliRunMode::EXPERIMENT_ONLY) {
    return run_single_mode(aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD, "experiment", range, N, opts);
  }

  aecct_ref::RefModel baseline_model;
  aecct_ref::RefModel experiment_model;
  aecct_ref::RefRunConfig baseline_cfg{};
  baseline_cfg.precision_mode = aecct_ref::RefPrecisionMode::BASELINE_FP32;
  baseline_cfg.algo_variant = opts.algo_variant;
  baseline_model.set_run_config(baseline_cfg);

  aecct_ref::RefRunConfig experiment_cfg{};
  experiment_cfg.precision_mode = aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD;
  experiment_cfg.algo_variant = opts.algo_variant;
  experiment_model.set_run_config(experiment_cfg);

  BatchCompareSummary batch{};
  init_batch_summary(batch);
  std::vector<PerPatternCompareRow> rows;
  rows.reserve(static_cast<std::size_t>(range.count));

  for (int off = 0; off < range.count; ++off) {
    const int pattern = range.begin + off;
    RefRunOutputs baseline{};
    RefRunOutputs experiment{};
    run_ref_single_pattern(baseline_model, pattern, N, baseline);
    run_ref_single_pattern(experiment_model, pattern, N, experiment);

    PerPatternCompareRow row{};
    row.pattern_index = pattern;
    row.baseline_vs_golden = baseline.logits_vs_golden;
    row.experiment_vs_golden = experiment.logits_vs_golden;
    row.cmp = aecct_ref::compute_experiment_compare_metrics(
      baseline.logits.data(),
      experiment.logits.data(),
      baseline.x_pred.data(),
      experiment.x_pred.data(),
      baseline.logits.size(),
      baseline.x_pred.size(),
      static_cast<std::size_t>(opts.topk)
    );

    rows.push_back(row);
    update_batch_summary(batch, row);

    std::printf("[pattern %d] mse=%.9e maxabs=%.9e xpred_flip=%zu (%.9e) sign_flip=%zu margin(b/e)=%.9e/%.9e naninf(b)=%zu/%zu naninf(e)=%zu/%zu\n",
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
      row.cmp.experiment_logits_nonfinite.inf_count);
    print_topk_entries("  baseline topk smallest |logit|:", row.cmp.baseline_topk_smallest_abs);
    print_topk_entries("  experiment topk smallest |logit|:", row.cmp.experiment_topk_smallest_abs);
  }

  const std::string csv_path = !opts.summary_csv_path.empty()
    ? opts.summary_csv_path
    : ("build/ref_eval/compare_summary_begin" + std::to_string(range.begin) +
       "_count" + std::to_string(range.count) + ".csv");
  if (write_compare_csv(csv_path, rows)) {
    std::printf("Per-pattern summary csv : %s\n", csv_path.c_str());
  } else {
    std::printf("[warn] Failed to write csv summary: %s\n", csv_path.c_str());
  }

  std::printf("=== Batch Compare Summary ===\n");
  std::printf("total patterns scanned        : %d\n", batch.total_patterns_scanned);
  std::printf("patterns with x_pred flip     : %zu\n", batch.patterns_with_xpred_flip);
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
  std::printf("BER/FER evaluator             : unavailable in this AECCT_ac_ref flow (baseline-relative compare only)\n");

  return 0;
}
