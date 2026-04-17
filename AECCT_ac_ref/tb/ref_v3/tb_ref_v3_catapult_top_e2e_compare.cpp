#include <array>
#include <cstdio>

#if defined(__SYNTHESIS__) || defined(REFV3_SYNTH_ONLY)
#error "tb_ref_v3_catapult_top_e2e_compare is host-only."
#endif

#include "AECCT_ac_ref/catapult/ref_v3/RefV3CatapultTop.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3FinalPassABlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3FinalPassBBlock.h"
#include "input_y_step0.h"

namespace {

static constexpr int kVarN = 63;
static constexpr std::array<int, 16> kPatternIndices = {
  77, 116, 132, 179, 217, 265, 312, 572, 602, 689, 701, 759, 789, 849, 906, 976};

struct CaseResult {
  int pattern_idx = -1;
  bool run_ok = false;
  bool result_pass = false;
  int token_nan_count = 0;
  int token_header_error_count = 0;
  int token_duplicate_count = 0;
  int token_missing_count = 0;
  int final_var_count_ok = 0;
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

static bool run_final_head_from_tokens(
  int pattern_idx,
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload>& in_token_ch,
  CaseResult* case_result,
  aecct_ref::ref_v3::RefV3FinalOutputPayload* out_payload) {
  if (case_result == nullptr || out_payload == nullptr) {
    return false;
  }

  bool token_seen[aecct_ref::ref_v3::REFV3_TOKENS_T];
  REFV3_CATAPULT_TOP_E2E_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < aecct_ref::ref_v3::REFV3_TOKENS_T;
                                                      ++token) {
    token_seen[token] = false;
  }

  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> finala_in_token_ch;
  REFV3_CATAPULT_TOP_E2E_VALIDATE_TOKEN_LOOP: for (int token_rx = 0; token_rx < aecct_ref::ref_v3::REFV3_TOKENS_T;
                                                    ++token_rx) {
    const aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload token_payload = in_token_ch.read();
    if (!aecct_ref::ref_v3::REFV3_payload_header_matches_shape(token_payload.header) ||
        token_payload.header.layer_id.to_int() != aecct_ref::ref_v3::REFV3_LAYER1_ID) {
      ++case_result->token_header_error_count;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= aecct_ref::ref_v3::REFV3_TOKENS_T) {
      ++case_result->token_header_error_count;
      continue;
    }
    if (token_seen[token]) {
      ++case_result->token_duplicate_count;
      continue;
    }
    token_seen[token] = true;

    REFV3_CATAPULT_TOP_E2E_VALIDATE_DIM_LOOP: for (int dim = 0; dim < aecct_ref::ref_v3::REFV3_D_MODEL; ++dim) {
      const aecct_ref::ref_v3::refv3_fp_t v = token_payload.token_vec[dim];
      if (v != v) {
        ++case_result->token_nan_count;
      }
    }

    finala_in_token_ch.write(token_payload);
  }

  REFV3_CATAPULT_TOP_E2E_TOKEN_MISSING_LOOP: for (int token = 0; token < aecct_ref::ref_v3::REFV3_TOKENS_T;
                                                   ++token) {
    if (!token_seen[token]) {
      ++case_result->token_missing_count;
    }
  }

  const int base = pattern_idx * kVarN;
  aecct_ref::ref_v3::RefV3FinalPassABlock final_pass_a_block;
  aecct_ref::ref_v3::RefV3FinalPassBBlock final_pass_b_block;
  ac_channel<aecct_ref::ref_v3::RefV3FinalScalarTokenPayload> finala_out_scalar_ch;
  ac_channel<aecct_ref::ref_v3::RefV3FinalInputYPayload> finalb_in_input_y_ch;
  ac_channel<aecct_ref::ref_v3::RefV3FinalOutputPayload> finalb_out_payload_ch;

  if (!final_pass_a_block.run(finala_in_token_ch, finala_out_scalar_ch)) {
    return false;
  }

  aecct_ref::ref_v3::RefV3FinalInputYPayload input_y_payload;
  input_y_payload.var_count = ac_int<16, false>(kVarN);
  REFV3_CATAPULT_TOP_E2E_FINALB_INPUT_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
    input_y_payload.input_y[n] = aecct_ref::ref_v3::refv3_fp_t(
      static_cast<float>(trace_input_y_step0_tensor[base + n]));
  }
  finalb_in_input_y_ch.write(input_y_payload);

  if (!final_pass_b_block.run(finala_out_scalar_ch, finalb_in_input_y_ch, finalb_out_payload_ch)) {
    return false;
  }

  *out_payload = finalb_out_payload_ch.read();
  case_result->final_var_count_ok =
    aecct_ref::ref_v3::REFV3_var_count_matches_shape(out_payload->var_count) ? 1 : 0;
  return true;
}

static bool run_case(int pattern_idx, CaseResult* out) {
  if (out == nullptr) {
    return false;
  }
  out->pattern_idx = pattern_idx;

  aecct_ref::ref_v3::RefV3CatapultTop dut;
  ac_channel<aecct_ref::ref_v3::RefV3PreprocInputPayload> preproc_in_ch;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> dut_out_token_ch;

  if (!stream_preproc_input(pattern_idx, preproc_in_ch)) {
    return false;
  }
  if (!dut.run(preproc_in_ch, dut_out_token_ch)) {
    out->run_ok = false;
    out->result_pass = false;
    return true;
  }

  aecct_ref::ref_v3::RefV3FinalOutputPayload final_payload;
  if (!run_final_head_from_tokens(pattern_idx, dut_out_token_ch, out, &final_payload)) {
    out->run_ok = false;
    out->result_pass = false;
    return true;
  }

  out->run_ok = true;
  out->result_pass =
    (out->token_nan_count == 0) &&
    (out->token_header_error_count == 0) &&
    (out->token_duplicate_count == 0) &&
    (out->token_missing_count == 0) &&
    (out->final_var_count_ok == 1);
  return true;
}

} // namespace

int main() {
  if (trace_input_y_step0_tensor_ndim != 2) {
    std::printf("FAIL: unexpected trace_input_y_step0_tensor_ndim\n");
    return 1;
  }
  if (trace_input_y_step0_tensor_shape[1] != kVarN) {
    std::printf("FAIL: unexpected trace_input_y_step0_tensor width\n");
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
  int pass_count = 0;
  int fail_count = 0;

  REFV3_CATAPULT_TOP_E2E_CASE_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    CaseResult result;
    const bool case_exec_ok = run_case(pattern_idx, &result);
    if (!case_exec_ok) {
      ++run_fail_count;
      ++fail_count;
      std::printf(
        "[ref_v3_catapult_top_e2e_compare] pattern_idx=%d token_nan_count=-1 token_header_error_count=-1 "
        "token_duplicate_count=-1 token_missing_count=-1 final_var_count_ok=0 run_ok=0 result=FAIL\n",
        pattern_idx);
      continue;
    }

    if (result.run_ok) {
      ++run_ok_count;
    } else {
      ++run_fail_count;
    }
    if (result.result_pass) {
      ++pass_count;
    } else {
      ++fail_count;
    }

    std::printf(
      "[ref_v3_catapult_top_e2e_compare] pattern_idx=%d token_nan_count=%d token_header_error_count=%d "
      "token_duplicate_count=%d token_missing_count=%d final_var_count_ok=%d run_ok=%d result=%s\n",
      pattern_idx,
      result.token_nan_count,
      result.token_header_error_count,
      result.token_duplicate_count,
      result.token_missing_count,
      result.final_var_count_ok,
      result.run_ok ? 1 : 0,
      result.result_pass ? "PASS" : "FAIL");
  }

  std::printf(
    "[ref_v3_catapult_top_e2e_compare_summary] total=%d run_ok=%d run_fail=%d pass=%d fail=%d\n",
    static_cast<int>(kPatternIndices.size()),
    run_ok_count,
    run_fail_count,
    pass_count,
    fail_count);

  if (run_fail_count != 0 || fail_count != 0) {
    std::printf("FAIL: tb_ref_v3_catapult_top_e2e_compare\n");
    return 2;
  }

  std::printf("PASS: tb_ref_v3_catapult_top_e2e_compare\n");
  return 0;
}

