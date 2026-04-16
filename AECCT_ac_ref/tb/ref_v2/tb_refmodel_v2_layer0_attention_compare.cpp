#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "AECCT_ac_ref/include/RefModel.h"
#include "AECCT_ac_ref/include/RefModelOptimized.h"
#include "AECCT_ac_ref/include/ref_v2/RefModel_v2.h"
#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"

namespace {
static constexpr int VAR_N = 63;
static constexpr double LOGIT_TOL = 1.0e-6;
static constexpr int kPatternCount = 8;
static constexpr std::array<int, kPatternCount> kPatternIndices = {906, 723, 849, 587, 217, 562, 222, 77};
static constexpr std::array<aecct_ref::RefLayerNormMode, 3> kLnModes = {
  aecct_ref::RefLayerNormMode::LN_BASELINE,
  aecct_ref::RefLayerNormMode::LN_SUM_SUMSQ_APPROX,
  aecct_ref::RefLayerNormMode::LN_EXACT_REFERENCE
};

struct TraceCompareResult {
  int logits_mismatch_count = 0;
  double logits_max_abs_diff = 0.0;
  int logits_first_mismatch_dim = -1;
  double logits_first_model = 0.0;
  double logits_first_trace = 0.0;
  int xpred_mismatch_count = 0;
  int xpred_first_mismatch_dim = -1;
  int xpred_first_model = 0;
  int xpred_first_trace = 0;
  bool pass = false;
};

static bool point_pass(const aecct_ref::ref_v2::RefV2ComparePoint& p) {
  return p.mismatch_count == 0;
}

static void print_compare_point(
  const char* ln_mode_name,
  int pattern_idx,
  const char* point_name,
  const aecct_ref::ref_v2::RefV2ComparePoint& p) {
  std::printf(
    "[tb_ref_v2_case_point] ln_mode=%s pattern_idx=%d point=%s mismatch_count=%d max_abs_diff=%.9e first_mismatch={token=%d,dim=%d,v2=%.9e,ref=%.9e}\n",
    ln_mode_name,
    pattern_idx,
    point_name,
    p.mismatch_count,
    p.max_abs_diff,
    p.first_mismatch_token,
    p.first_mismatch_dim,
    p.first_v2_value,
    p.first_ref_value);
}

static bool run_trace_golden_check(int pattern_idx, TraceCompareResult* out) {
  if (out == nullptr) {
    return false;
  }

  aecct_ref::RefModelOptimized model;
  model.set_run_config(aecct_ref::make_fp32_baseline_run_config());

  std::vector<double> logits(static_cast<std::size_t>(VAR_N), 0.0);
  std::vector<aecct_ref::bit1_t> xpred(static_cast<std::size_t>(VAR_N), aecct_ref::bit1_t(0));
  const int base = pattern_idx * VAR_N;

  aecct_ref::RefModelIO io{};
  io.input_y_fp32 = &trace_input_y_step0_tensor[base];
  io.out_logits = logits.data();
  io.out_x_pred = xpred.data();
  io.B = 1;
  io.N = VAR_N;
  model.infer_step0(io);

  out->logits_mismatch_count = 0;
  out->logits_max_abs_diff = 0.0;
  out->logits_first_mismatch_dim = -1;
  out->xpred_mismatch_count = 0;
  out->xpred_first_mismatch_dim = -1;

  for (int n = 0; n < VAR_N; ++n) {
    const double model_logit = logits[static_cast<std::size_t>(n)];
    const double golden_logit = trace_output_logits_step0_tensor[base + n];
    const double abs_diff = std::fabs(model_logit - golden_logit);
    if (abs_diff > out->logits_max_abs_diff) {
      out->logits_max_abs_diff = abs_diff;
    }
    if (abs_diff > LOGIT_TOL) {
      ++out->logits_mismatch_count;
      if (out->logits_first_mismatch_dim < 0) {
        out->logits_first_mismatch_dim = n;
        out->logits_first_model = model_logit;
        out->logits_first_trace = golden_logit;
      }
    }

    const int model_bit = xpred[static_cast<std::size_t>(n)].to_int();
    const int golden_bit = (trace_output_x_pred_step0_tensor[base + n] != 0.0) ? 1 : 0;
    if (model_bit != golden_bit) {
      ++out->xpred_mismatch_count;
      if (out->xpred_first_mismatch_dim < 0) {
        out->xpred_first_mismatch_dim = n;
        out->xpred_first_model = model_bit;
        out->xpred_first_trace = golden_bit;
      }
    }
  }

  out->pass = (out->logits_mismatch_count == 0) && (out->xpred_mismatch_count == 0);
  return true;
}
} // namespace

int main() {
  if (trace_input_y_step0_tensor_ndim != 2 || trace_output_logits_step0_tensor_ndim != 2 ||
      trace_output_x_pred_step0_tensor_ndim != 2) {
    std::printf("FAIL: unexpected trace ndim\n");
    return 1;
  }
  if (trace_input_y_step0_tensor_shape[1] != VAR_N ||
      trace_output_logits_step0_tensor_shape[1] != VAR_N ||
      trace_output_x_pred_step0_tensor_shape[1] != VAR_N) {
    std::printf("FAIL: unexpected trace VAR_N dimension\n");
    return 1;
  }
  if (trace_input_y_step0_tensor_shape[0] != trace_output_logits_step0_tensor_shape[0] ||
      trace_input_y_step0_tensor_shape[0] != trace_output_x_pred_step0_tensor_shape[0]) {
    std::printf("FAIL: trace batch dimension mismatch\n");
    return 1;
  }

  const int trace_batch = trace_input_y_step0_tensor_shape[0];
  for (int i = 0; i < kPatternCount; ++i) {
    if (kPatternIndices[static_cast<std::size_t>(i)] < 0 ||
        kPatternIndices[static_cast<std::size_t>(i)] >= trace_batch) {
      std::printf("FAIL: pattern index out of range: %d (trace_batch=%d)\n",
                  kPatternIndices[static_cast<std::size_t>(i)],
                  trace_batch);
      return 1;
    }
  }

  int trace_map_passed = 0;
  int trace_map_failed = 0;
  for (int i = 0; i < kPatternCount; ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    TraceCompareResult trace_result;
    if (!run_trace_golden_check(pattern_idx, &trace_result)) {
      std::printf("FAIL: trace mapping check execution failed at pattern_idx=%d\n", pattern_idx);
      return 1;
    }
    if (trace_result.pass) {
      ++trace_map_passed;
    } else {
      ++trace_map_failed;
    }
    std::printf(
      "[tb_ref_v2_trace_map] pattern_idx=%d logits_mismatch_count=%d logits_max_abs_diff=%.9e logits_first_mismatch={dim=%d,model=%.9e,trace=%.9e} "
      "xpred_mismatch_count=%d xpred_first_mismatch={dim=%d,model=%d,trace=%d} pass=%d\n",
      pattern_idx,
      trace_result.logits_mismatch_count,
      trace_result.logits_max_abs_diff,
      trace_result.logits_first_mismatch_dim,
      trace_result.logits_first_model,
      trace_result.logits_first_trace,
      trace_result.xpred_mismatch_count,
      trace_result.xpred_first_mismatch_dim,
      trace_result.xpred_first_model,
      trace_result.xpred_first_trace,
      trace_result.pass ? 1 : 0);
  }

  aecct_ref::ref_v2::RefModel_v2 model_v2;
  int passed_cases = 0;
  int failed_cases = 0;
  const int total_cases = static_cast<int>(kPatternCount * kLnModes.size());

  for (std::size_t mode_i = 0; mode_i < kLnModes.size(); ++mode_i) {
    const aecct_ref::RefLayerNormMode ln_mode = kLnModes[mode_i];
    const char* ln_mode_name = aecct_ref::to_string(ln_mode);
    aecct_ref::RefRunConfig cfg = aecct_ref::make_fp32_baseline_run_config();
    cfg.legacy.ln_mode = ln_mode;
    model_v2.set_run_config(cfg);

    for (int pat_i = 0; pat_i < kPatternCount; ++pat_i) {
      const int pattern_idx = kPatternIndices[static_cast<std::size_t>(pat_i)];
      aecct_ref::RefModelIO io{};
      io.input_y_fp32 = &trace_input_y_step0_tensor[pattern_idx * VAR_N];
      io.B = 1;
      io.N = VAR_N;

      const bool run_ok = model_v2.run_step0_layer0_attention_compare(io, 0);
      if (!run_ok) {
        ++failed_cases;
        std::printf(
          "[tb_ref_v2_case_summary] ln_mode=%s pattern_idx=%d run_ok=0 case_pass=FAIL\n",
          ln_mode_name,
          pattern_idx);
        continue;
      }

      const aecct_ref::ref_v2::RefV2CompareStats stats = model_v2.last_compare_stats();
      print_compare_point(ln_mode_name, pattern_idx, "preproc_output", stats.preproc_output);
      print_compare_point(ln_mode_name, pattern_idx, "attention_input", stats.attention_input);
      print_compare_point(ln_mode_name, pattern_idx, "SCR_K", stats.scr_k);
      print_compare_point(ln_mode_name, pattern_idx, "SCR_V", stats.scr_v);
      print_compare_point(ln_mode_name, pattern_idx, "x_work_writeback", stats.x_work_writeback);
      print_compare_point(ln_mode_name, pattern_idx, "layer0_ln_output", stats.layer0_ln_output);
      print_compare_point(ln_mode_name, pattern_idx, "x_work_after_layer0_ln", stats.x_work_after_layer0_ln);
      print_compare_point(ln_mode_name, pattern_idx, "layer0_ffn_output", stats.layer0_ffn_output);
      print_compare_point(ln_mode_name, pattern_idx, "x_work_after_layer0_ffn", stats.x_work_after_layer0_ffn);
      std::printf(
        "[tb_ref_v2_case_point] ln_mode=%s pattern_idx=%d point=next_stage_handoff mismatch_count=%d max_abs_diff=%.9e "
        "token_count=%d out_of_order=%d duplicate=%d missing=%d header_error=%d invalid_token=%d pass=%d first_mismatch={token=%d,dim=%d,v2=%.9e,ref=%.9e}\n",
        ln_mode_name,
        pattern_idx,
        stats.next_stage_handoff.mismatch_count,
        stats.next_stage_handoff.max_abs_diff,
        stats.next_stage_token_count,
        stats.next_stage_out_of_order_count,
        stats.next_stage_duplicate_count,
        stats.next_stage_missing_count,
        stats.next_stage_header_error_count,
        stats.next_stage_invalid_token_count,
        stats.next_stage_handoff_pass ? 1 : 0,
        stats.next_stage_handoff.first_mismatch_token,
        stats.next_stage_handoff.first_mismatch_dim,
        stats.next_stage_handoff.first_v2_value,
        stats.next_stage_handoff.first_ref_value);
      print_compare_point(ln_mode_name, pattern_idx, "final_passA_output", stats.final_passA_output);
      print_compare_point(ln_mode_name, pattern_idx, "final_logits", stats.final_logits);
      print_compare_point(ln_mode_name, pattern_idx, "final_x_pred", stats.final_x_pred);

      const bool preproc_pass = point_pass(stats.preproc_output);
      const bool attention_input_pass = point_pass(stats.attention_input);
      const bool scr_k_pass = point_pass(stats.scr_k);
      const bool scr_v_pass = point_pass(stats.scr_v);
      const bool x_work_writeback_pass = point_pass(stats.x_work_writeback);
      const bool next_stage_handoff_pass = stats.next_stage_handoff_pass;
      const bool layer0_ln_output_pass = point_pass(stats.layer0_ln_output);
      const bool x_work_after_layer0_ln_pass = point_pass(stats.x_work_after_layer0_ln);
      const bool layer0_ffn_output_pass = point_pass(stats.layer0_ffn_output);
      const bool x_work_after_layer0_ffn_pass = point_pass(stats.x_work_after_layer0_ffn);
      const bool final_passA_output_pass = point_pass(stats.final_passA_output);
      const bool final_logits_pass = point_pass(stats.final_logits);
      const bool final_x_pred_pass = point_pass(stats.final_x_pred);
      const bool case_pass = stats.all_match;

      if (case_pass) {
        ++passed_cases;
      } else {
        ++failed_cases;
      }

      std::printf(
        "[tb_ref_v2_case_summary] ln_mode=%s pattern_idx=%d preproc_output=%s attention_input=%s SCR_K=%s SCR_V=%s "
        "x_work_writeback=%s next_stage_handoff=%s layer0_ln_output=%s x_work_after_layer0_ln=%s layer0_ffn_output=%s "
        "x_work_after_layer0_ffn=%s final_passA_output=%s final_logits=%s final_x_pred=%s case_pass=%s\n",
        ln_mode_name,
        pattern_idx,
        preproc_pass ? "PASS" : "FAIL",
        attention_input_pass ? "PASS" : "FAIL",
        scr_k_pass ? "PASS" : "FAIL",
        scr_v_pass ? "PASS" : "FAIL",
        x_work_writeback_pass ? "PASS" : "FAIL",
        next_stage_handoff_pass ? "PASS" : "FAIL",
        layer0_ln_output_pass ? "PASS" : "FAIL",
        x_work_after_layer0_ln_pass ? "PASS" : "FAIL",
        layer0_ffn_output_pass ? "PASS" : "FAIL",
        x_work_after_layer0_ffn_pass ? "PASS" : "FAIL",
        final_passA_output_pass ? "PASS" : "FAIL",
        final_logits_pass ? "PASS" : "FAIL",
        final_x_pred_pass ? "PASS" : "FAIL",
        case_pass ? "PASS" : "FAIL");
    }
  }

  const bool all_cases_pass = (failed_cases == 0);
  const bool trace_mapping_pass = (trace_map_failed == 0);
  std::printf(
    "[tb_ref_v2_summary] total_cases=%d passed_cases=%d failed_cases=%d trace_mapping_passed=%d trace_mapping_failed=%d\n",
    total_cases,
    passed_cases,
    failed_cases,
    trace_map_passed,
    trace_map_failed);

  if (!trace_mapping_pass) {
    std::printf("WARN: trace pattern mapping compare has mismatches (non-gating)\n");
  }
  if (!all_cases_pass) {
    std::printf("FAIL: RefModel_v2 compare mismatch detected\n");
    return 2;
  }

  std::printf("PASS: tb_refmodel_v2_layer0_attention_compare\n");
  return 0;
}
