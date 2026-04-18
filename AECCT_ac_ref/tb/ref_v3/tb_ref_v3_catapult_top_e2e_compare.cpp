#include <array>
#include <cstdio>

#if defined(__SYNTHESIS__) || defined(REFV3_SYNTH_ONLY)
#error "tb_ref_v3_catapult_top_e2e_compare is host-only."
#endif

#include "AECCT_ac_ref/catapult/ref_v3/RefV3CatapultTop.h"
#include "input_y_step0.h"
#include "output_x_pred_step0.h"

namespace {

static constexpr int kVarN = 63;
static constexpr std::array<int, 8> kPatternIndices = {77, 116, 132, 179, 217, 265, 312, 572};

struct CaseResult {
  int pattern_idx = -1;
  bool run_ok = false;

  int payload_var_count_ok = 0;
  int dut_zero_count = 0;
  int dut_one_count = 0;
  int trace_zero_count = 0;
  int trace_one_count = 0;
  int match_trace_count = 0;
  int mismatch_trace_count = 0;
  int trace_non_binary_anomaly_count = 0;
  const char* winner_by_one_count = "TIE";
  int one_count_delta = 0;
  int dut_all_zero = 0;
  int trace_all_zero = 0;
};

static bool stream_preproc_input(
  int pattern_idx,
  ac_channel<aecct_ref::ref_v3::RefV3PreprocInputPayload>& preproc_in_ch) {
  const int base = pattern_idx * kVarN;

  aecct_ref::ref_v3::RefV3PreprocInputPayload input_payload;
  input_payload.var_count = ac_int<16, false>(kVarN);
  REFV3_CATAPULT_TOP_E2E_PREPROC_INPUT_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
    input_payload.input_y[n] = aecct_ref::ref_v3::refv3_fp_t(
      static_cast<float>(trace_input_y_step0_tensor[base + n]));
  }
  preproc_in_ch.write(input_payload);
  return true;
}

static bool run_case(int pattern_idx, CaseResult* out) {
  if (out == nullptr) {
    return false;
  }
  out->pattern_idx = pattern_idx;

  aecct_ref::ref_v3::RefV3CatapultTop dut;
  ac_channel<aecct_ref::ref_v3::RefV3PreprocInputPayload> preproc_in_ch;
  ac_channel<aecct_ref::ref_v3::RefV3FinalOutputPayload> dut_out_payload_ch;

  if (!stream_preproc_input(pattern_idx, preproc_in_ch)) {
    return false;
  }
  if (!dut.run(preproc_in_ch, dut_out_payload_ch)) {
    out->run_ok = false;
    return true;
  }

  const aecct_ref::ref_v3::RefV3FinalOutputPayload out_payload = dut_out_payload_ch.read();
  out->payload_var_count_ok =
    aecct_ref::ref_v3::REFV3_var_count_matches_shape(out_payload.var_count) ? 1 : 0;
  if (out->payload_var_count_ok != 1) {
    out->run_ok = false;
    return true;
  }

  const int base = pattern_idx * kVarN;
  REFV3_CATAPULT_TOP_E2E_XPRED_COMPARE_LOOP: for (int n = 0; n < kVarN; ++n) {
    const int dut_bit = (out_payload.x_pred[n].to_int() != 0) ? 1 : 0;
    if (dut_bit == 0) {
      ++out->dut_zero_count;
    } else {
      ++out->dut_one_count;
    }

    const double trace_raw = trace_output_x_pred_step0_tensor[base + n];
    int trace_bit = 0;
    bool trace_is_binary = true;
    if (trace_raw == 0.0) {
      trace_bit = 0;
      ++out->trace_zero_count;
    } else if (trace_raw == 1.0) {
      trace_bit = 1;
      ++out->trace_one_count;
    } else {
      trace_is_binary = false;
      ++out->trace_non_binary_anomaly_count;
      ++out->mismatch_trace_count;
    }

    if (trace_is_binary) {
      if (dut_bit == trace_bit) {
        ++out->match_trace_count;
      } else {
        ++out->mismatch_trace_count;
      }
    }
  }

  // one_count is the primary quality metric for all-zero codeword patterns.
  if (out->dut_one_count < out->trace_one_count) {
    out->winner_by_one_count = "DUT";
  } else if (out->dut_one_count > out->trace_one_count) {
    out->winner_by_one_count = "TRACE";
  } else {
    out->winner_by_one_count = "TIE";
  }
  out->one_count_delta = out->trace_one_count - out->dut_one_count;

  out->dut_all_zero = (out->dut_one_count == 0) ? 1 : 0;
  out->trace_all_zero =
    ((out->trace_one_count == 0) && (out->trace_non_binary_anomaly_count == 0)) ? 1 : 0;
  out->run_ok = true;
  return true;
}

} // namespace

int main() {
  if (trace_input_y_step0_tensor_ndim != 2 || trace_output_x_pred_step0_tensor_ndim != 2) {
    std::printf("FAIL: unexpected trace ndim\n");
    return 1;
  }
  if (trace_input_y_step0_tensor_shape[1] != kVarN || trace_output_x_pred_step0_tensor_shape[1] != kVarN) {
    std::printf("FAIL: unexpected trace tensor width\n");
    return 1;
  }
  if (trace_input_y_step0_tensor_shape[0] != trace_output_x_pred_step0_tensor_shape[0]) {
    std::printf("FAIL: trace batch dimension mismatch\n");
    return 1;
  }

  const int trace_batch = trace_input_y_step0_tensor_shape[0];
  REFV3_CATAPULT_TOP_E2E_PATTERN_RANGE_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    if (pattern_idx < 0 || pattern_idx >= trace_batch) {
      std::printf("FAIL: pattern index out of range: %d (trace_batch=%d)\n", pattern_idx, trace_batch);
      return 1;
    }
  }

  int run_ok_count = 0;
  int run_fail_count = 0;
  int dut_better_count = 0;
  int trace_better_count = 0;
  int tie_count = 0;
  int dut_all_zero_count = 0;
  int trace_all_zero_count = 0;
  int dut_total_one_count = 0;
  int trace_total_one_count = 0;
  int total_match_trace_count = 0;
  int total_mismatch_trace_count = 0;
  int trace_non_binary_anomaly_pattern_count = 0;
  int trace_non_binary_anomaly_total = 0;

  REFV3_CATAPULT_TOP_E2E_CASE_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    CaseResult result;
    const bool case_exec_ok = run_case(pattern_idx, &result);
    if (!case_exec_ok || !result.run_ok) {
      ++run_fail_count;
      std::printf(
        "[ref_v3_top_tail_xpred_onecount_compare] pattern_idx=%d dut_one_count=-1 dut_zero_count=-1 "
        "trace_one_count=-1 trace_zero_count=-1 winner_by_one_count=NA one_count_delta=0 "
        "dut_all_zero=0 trace_all_zero=0 match_trace_count=-1 mismatch_trace_count=-1 "
        "trace_non_binary_anomaly_count=-1 result=RUN_FAIL\n",
        pattern_idx);
      continue;
    }

    ++run_ok_count;
    dut_total_one_count += result.dut_one_count;
    trace_total_one_count += result.trace_one_count;
    total_match_trace_count += result.match_trace_count;
    total_mismatch_trace_count += result.mismatch_trace_count;
    if (result.trace_non_binary_anomaly_count > 0) {
      ++trace_non_binary_anomaly_pattern_count;
      trace_non_binary_anomaly_total += result.trace_non_binary_anomaly_count;
    }
    if (result.dut_all_zero == 1) {
      ++dut_all_zero_count;
    }
    if (result.trace_all_zero == 1) {
      ++trace_all_zero_count;
    }

    if (result.winner_by_one_count[0] == 'D') {
      ++dut_better_count;
    } else if (result.winner_by_one_count[0] == 'T' && result.winner_by_one_count[1] == 'R') {
      ++trace_better_count;
    } else {
      ++tie_count;
    }

    std::printf(
      "[ref_v3_top_tail_xpred_onecount_compare] pattern_idx=%d dut_one_count=%d dut_zero_count=%d "
      "trace_one_count=%d trace_zero_count=%d winner_by_one_count=%s one_count_delta=%d "
      "dut_all_zero=%d trace_all_zero=%d match_trace_count=%d mismatch_trace_count=%d "
      "trace_non_binary_anomaly_count=%d result=RUN_OK\n",
      result.pattern_idx,
      result.dut_one_count,
      result.dut_zero_count,
      result.trace_one_count,
      result.trace_zero_count,
      result.winner_by_one_count,
      result.one_count_delta,
      result.dut_all_zero,
      result.trace_all_zero,
      result.match_trace_count,
      result.mismatch_trace_count,
      result.trace_non_binary_anomaly_count);
  }

  const double dut_avg_one_count = (run_ok_count > 0)
    ? static_cast<double>(dut_total_one_count) / static_cast<double>(run_ok_count)
    : 0.0;
  const double trace_avg_one_count = (run_ok_count > 0)
    ? static_cast<double>(trace_total_one_count) / static_cast<double>(run_ok_count)
    : 0.0;
  const double avg_match_trace_count = (run_ok_count > 0)
    ? static_cast<double>(total_match_trace_count) / static_cast<double>(run_ok_count)
    : 0.0;
  const double avg_mismatch_trace_count = (run_ok_count > 0)
    ? static_cast<double>(total_mismatch_trace_count) / static_cast<double>(run_ok_count)
    : 0.0;

  std::printf(
    "[ref_v3_top_tail_xpred_onecount_compare_summary] total_patterns=%d run_ok=%d run_fail=%d "
    "dut_better_count_by_onecount=%d trace_better_count_by_onecount=%d tie_count=%d "
    "dut_all_zero_count=%d trace_all_zero_count=%d dut_avg_one_count=%.3f trace_avg_one_count=%.3f "
    "avg_match_trace_count=%.3f avg_mismatch_trace_count=%.3f "
    "trace_non_binary_anomaly_patterns=%d trace_non_binary_anomaly_total=%d\n",
    static_cast<int>(kPatternIndices.size()),
    run_ok_count,
    run_fail_count,
    dut_better_count,
    trace_better_count,
    tie_count,
    dut_all_zero_count,
    trace_all_zero_count,
    dut_avg_one_count,
    trace_avg_one_count,
    avg_match_trace_count,
    avg_mismatch_trace_count,
    trace_non_binary_anomaly_pattern_count,
    trace_non_binary_anomaly_total);

  if (run_fail_count != 0) {
    std::printf("FAIL: tb_ref_v3_catapult_top_e2e_compare\n");
    return 2;
  }

  std::printf("PASS: tb_ref_v3_catapult_top_e2e_compare\n");
  return 0;
}

