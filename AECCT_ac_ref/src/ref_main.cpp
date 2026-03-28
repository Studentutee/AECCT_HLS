#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct CliOptions {
  CliRunMode run_mode;
  int pattern_index;
  aecct_ref::RefAlgoVariant algo_variant;
};

struct RefRunOutputs {
  std::vector<double> logits;
  std::vector<aecct_ref::bit1_t> x_pred;
  aecct_ref::Metrics logits_vs_golden;
  std::size_t x_pred_match_count;
  std::size_t x_pred_total_count;
};

static void print_usage() {
  std::printf("Usage: ref_sim [pattern_index] [--mode compare|baseline|experiment] [--pattern index]\n");
  std::printf("       ref_sim --help\n");
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

static bool parse_cli(int argc, char** argv, CliOptions& opts) {
  opts.run_mode = CliRunMode::COMPARE;
  opts.pattern_index = -1;
  opts.algo_variant = aecct_ref::RefAlgoVariant::BASELINE_SPEC_FLOW;

  bool positional_pattern_used = false;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      print_usage();
      return false;
    }
    if (std::strcmp(arg, "--mode") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --mode\n");
        return false;
      }
      if (!parse_run_mode(argv[++i], opts.run_mode)) {
        std::printf("Unsupported mode: %s\n", argv[i]);
        return false;
      }
      continue;
    }
    if (std::strcmp(arg, "--pattern") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --pattern\n");
        return false;
      }
      opts.pattern_index = std::atoi(argv[++i]);
      positional_pattern_used = true;
      continue;
    }
    if (std::strcmp(arg, "--algo") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value after --algo\n");
        return false;
      }
      if (!parse_algo_variant(argv[++i], opts.algo_variant)) {
        std::printf("Unsupported algo variant: %s\n", argv[i]);
        return false;
      }
      continue;
    }
    if (arg[0] == '-') {
      std::printf("Unknown flag: %s\n", arg);
      return false;
    }
    if (positional_pattern_used) {
      std::printf("Unexpected positional arg: %s\n", arg);
      return false;
    }
    opts.pattern_index = std::atoi(arg);
    positional_pattern_used = true;
  }
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

static void run_ref_mode(
  aecct_ref::RefPrecisionMode precision_mode,
  const CliOptions& opts,
  RefRunOutputs& out
) {
  const int B = trace_input_y_step0_tensor_shape[0];
  const int N = trace_input_y_step0_tensor_shape[1];

  const int run_B = (opts.pattern_index >= 0) ? 1 : B;
  const std::size_t logits_count = static_cast<std::size_t>(run_B * N);
  const std::size_t x_pred_count = logits_count;

  const double* input_ptr = (opts.pattern_index >= 0)
    ? &trace_input_y_step0_tensor[opts.pattern_index * N]
    : trace_input_y_step0_tensor;
  const double* golden_logits = (opts.pattern_index >= 0)
    ? &trace_output_logits_step0_tensor[opts.pattern_index * N]
    : trace_output_logits_step0_tensor;
  const double* golden_x_pred = (opts.pattern_index >= 0)
    ? &trace_output_x_pred_step0_tensor[opts.pattern_index * N]
    : trace_output_x_pred_step0_tensor;

  out.logits.assign(logits_count, 0.0);
  out.x_pred.assign(x_pred_count, aecct_ref::bit1_t(0));

  aecct_ref::RefModel model;
  aecct_ref::RefRunConfig cfg{};
  cfg.precision_mode = precision_mode;
  cfg.algo_variant = opts.algo_variant;
  model.set_run_config(cfg);

  aecct_ref::RefModelIO io{};
  io.input_y = nullptr;
  io.input_y_fp32 = input_ptr;
  io.out_logits = out.logits.data();
  io.out_x_pred = out.x_pred.data();
  io.B = run_B;
  io.N = N;
  model.infer_step0(io);

  out.logits_vs_golden = aecct_ref::compute_metrics(golden_logits, out.logits.data(), logits_count);
  out.x_pred_match_count = compute_x_pred_match_count(golden_x_pred, out.x_pred);
  out.x_pred_total_count = x_pred_count;
}

static void print_vs_golden_summary(
  const char* tag,
  const RefRunOutputs& run
) {
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

static void print_first_k_compare(
  const RefRunOutputs& baseline,
  const RefRunOutputs& experiment,
  int k
) {
  const int n = static_cast<int>(baseline.logits.size());
  const int kk = (k < n) ? k : n;
  std::printf("logits first %d (baseline / experiment / delta):\n", kk);
  for (int i = 0; i < kk; ++i) {
    const double b = baseline.logits[static_cast<std::size_t>(i)];
    const double e = experiment.logits[static_cast<std::size_t>(i)];
    std::printf("  [%d] %.9f / %.9f / %.9f\n", i, b, e, (e - b));
  }
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
  if (!parse_cli(argc, argv, opts)) {
    return 1;
  }
  if (opts.pattern_index >= B) {
    std::printf("pattern_index out of range: %d (valid [0, %d))\n", opts.pattern_index, B);
    return 1;
  }

  std::printf("Run config:\n");
  std::printf("  mode           : %s\n",
    (opts.run_mode == CliRunMode::COMPARE) ? "compare"
      : ((opts.run_mode == CliRunMode::BASELINE_ONLY) ? "baseline" : "experiment"));
  std::printf("  precision(base): %s\n", aecct_ref::to_string(aecct_ref::RefPrecisionMode::BASELINE_FP32));
  std::printf("  precision(exp) : %s\n", aecct_ref::to_string(aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD));
  std::printf("  algo_variant   : %s\n", aecct_ref::to_string(opts.algo_variant));
  if (opts.pattern_index >= 0) {
    std::printf("  pattern_index  : %d\n", opts.pattern_index);
  } else {
    std::printf("  pattern_index  : ALL (%d patterns)\n", B);
  }

  if (opts.run_mode == CliRunMode::BASELINE_ONLY) {
    RefRunOutputs baseline{};
    run_ref_mode(aecct_ref::RefPrecisionMode::BASELINE_FP32, opts, baseline);
    print_vs_golden_summary("baseline", baseline);
    return 0;
  }

  if (opts.run_mode == CliRunMode::EXPERIMENT_ONLY) {
    RefRunOutputs experiment{};
    run_ref_mode(aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD, opts, experiment);
    print_vs_golden_summary("experiment", experiment);
    return 0;
  }

  RefRunOutputs baseline{};
  RefRunOutputs experiment{};
  run_ref_mode(aecct_ref::RefPrecisionMode::BASELINE_FP32, opts, baseline);
  run_ref_mode(aecct_ref::RefPrecisionMode::GENERIC_E4M3_FINALHEAD, opts, experiment);

  print_vs_golden_summary("baseline", baseline);
  print_vs_golden_summary("experiment", experiment);

  const aecct_ref::RefExperimentCompareMetrics cmp = aecct_ref::compute_experiment_compare_metrics(
    baseline.logits.data(),
    experiment.logits.data(),
    baseline.x_pred.data(),
    experiment.x_pred.data(),
    baseline.logits.size(),
    baseline.x_pred.size()
  );

  std::printf("=== baseline vs experiment ===\n");
  std::printf("logits MSE              : %.9e\n", cmp.logits_diff.mse);
  std::printf("logits MaxAbs diff      : %.9e\n", cmp.logits_diff.max_abs);
  std::printf("x_pred mismatch count   : %zu\n", cmp.x_pred_mismatch_count);
  std::printf("x_pred mismatch ratio   : %.9e\n", cmp.x_pred_mismatch_ratio);
  std::printf("baseline logits NaN/Inf : %zu / %zu\n",
    cmp.baseline_logits_nonfinite.nan_count, cmp.baseline_logits_nonfinite.inf_count);
  std::printf("experiment logits NaN/Inf: %zu / %zu\n",
    cmp.experiment_logits_nonfinite.nan_count, cmp.experiment_logits_nonfinite.inf_count);
  std::printf("BER/FER evaluator       : unavailable in this AECCT_ac_ref flow (baseline-relative compare only)\n");

  print_first_k_compare(baseline, experiment, 8);
  return 0;
}
