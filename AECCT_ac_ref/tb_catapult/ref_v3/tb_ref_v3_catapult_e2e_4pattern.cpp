#include <array>
#include <cstdio>

#if defined(__SYNTHESIS__) || defined(REFV3_SYNTH_ONLY)
#error "tb_ref_v3_catapult_e2e_4pattern is host-only."
#endif

#include "AECCT_ac_ref/catapult/ref_v3/RefV3CatapultTop.h"
#include "input_y_step0.h"

namespace {

static constexpr int kVarN = 63;
static constexpr std::array<int, 4> kPatternIndices = {77, 116, 132, 179};

struct CaseResult {
  int pattern_idx = -1;
  bool run_ok = false;
  int dut_one_count = 0;
  int dut_zero_count = 0;
  int dut_all_zero = 0;
};

static bool stream_preproc_input(
  int pattern_idx,
  ac_channel<aecct_ref::ref_v3::RefV3PreprocInputPayload>& preproc_in_ch) {
  const int base = pattern_idx * kVarN;

  aecct_ref::ref_v3::RefV3PreprocInputPayload input_payload;
  input_payload.var_count = ac_int<16, false>(kVarN);
  REFV3_E2E4_INPUT_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
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
  if (!aecct_ref::ref_v3::REFV3_var_count_matches_shape(out_payload.var_count)) {
    out->run_ok = false;
    return true;
  }

  REFV3_E2E4_XPRED_COUNT_LOOP: for (int n = 0; n < kVarN; ++n) {
    const int dut_bit = (out_payload.x_pred[n].to_int() != 0) ? 1 : 0;
    if (dut_bit == 0) {
      ++out->dut_zero_count;
    } else {
      ++out->dut_one_count;
    }
  }
  out->dut_all_zero = (out->dut_one_count == 0) ? 1 : 0;
  out->run_ok = true;
  return true;
}

} // namespace

int main() {
  if (trace_input_y_step0_tensor_ndim != 2) {
    std::printf("FAIL: unexpected input trace ndim\n");
    return 1;
  }
  if (trace_input_y_step0_tensor_shape[1] != kVarN) {
    std::printf("FAIL: unexpected input trace width\n");
    return 1;
  }

  const int trace_batch = trace_input_y_step0_tensor_shape[0];
  REFV3_E2E4_PATTERN_RANGE_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    if (pattern_idx < 0 || pattern_idx >= trace_batch) {
      std::printf("FAIL: pattern index out of range: %d (trace_batch=%d)\n", pattern_idx, trace_batch);
      return 1;
    }
  }

  int run_ok_count = 0;
  int run_fail_count = 0;
  int dut_all_zero_count = 0;
  int dut_total_one_count = 0;

  REFV3_E2E4_CASE_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    CaseResult result;
    const bool case_exec_ok = run_case(pattern_idx, &result);
    if (!case_exec_ok || !result.run_ok) {
      ++run_fail_count;
      std::printf(
        "[ref_v3_catapult_e2e_4pattern] pattern_idx=%d dut_one_count=-1 dut_zero_count=-1 "
        "dut_all_zero=0 result=RUN_FAIL\n",
        pattern_idx);
      continue;
    }

    ++run_ok_count;
    dut_total_one_count += result.dut_one_count;
    if (result.dut_all_zero == 1) {
      ++dut_all_zero_count;
    }

    std::printf(
      "[ref_v3_catapult_e2e_4pattern] pattern_idx=%d dut_one_count=%d dut_zero_count=%d "
      "dut_all_zero=%d result=RUN_OK\n",
      result.pattern_idx,
      result.dut_one_count,
      result.dut_zero_count,
      result.dut_all_zero);
  }

  const double dut_avg_one_count = (run_ok_count > 0)
    ? static_cast<double>(dut_total_one_count) / static_cast<double>(run_ok_count)
    : 0.0;
  std::printf(
    "[ref_v3_catapult_e2e_4pattern_summary] total_patterns=%d run_ok=%d run_fail=%d "
    "dut_all_zero_count=%d dut_avg_one_count=%.3f\n",
    static_cast<int>(kPatternIndices.size()),
    run_ok_count,
    run_fail_count,
    dut_all_zero_count,
    dut_avg_one_count);

  if (run_fail_count != 0) {
    std::printf("FAIL: tb_ref_v3_catapult_e2e_4pattern\n");
    return 2;
  }

  std::printf("PASS: tb_ref_v3_catapult_e2e_4pattern\n");
  return 0;
}
