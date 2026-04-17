#include <array>
#include <cmath>
#include <cstdio>

#if defined(__SYNTHESIS__) || defined(REFV3_SYNTH_ONLY)
#error "tb_ref_v3_catapult_top_e2e_compare is host-only."
#endif

#include "AECCT_ac_ref/catapult/ref_v3/RefV3CatapultTop.h"
#include "AECCT_ac_ref/include/RefModel.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3FinalPassABlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3FinalPassBBlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer0AttnLnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3PreprocBlock.h"
#include "input_y_step0.h"

namespace {

static constexpr int kVarN = 63;
static constexpr double kLogitTol = 1.0e-6;
static constexpr std::array<int, 16> kPatternIndices = {
  77, 116, 132, 179, 217, 265, 312, 572, 602, 689, 701, 759, 789, 849, 906, 976};

struct CaseResult {
  int pattern_idx = -1;
  bool run_ok = false;
  bool result_pass = false;

  int xpred_mismatch_count = 0;
  int xpred_first_mismatch_idx = -1;
  int xpred_first_ref = -1;
  int xpred_first_wrapper = -1;

  int logits_mismatch_count = 0;
  double logits_max_abs_diff = 0.0;
  int logits_first_mismatch_idx = -1;
  double logits_first_ref = 0.0;
  double logits_first_wrapper = 0.0;
};

static bool stream_preproc_input(int pattern_idx,
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

static bool stream_preproc_output_to_dual_inputs(
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload>& preproc_out_token_ch,
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload>& ref_in_token_ch,
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload>& wrapper_in_token_ch) {
  REFV3_CATAPULT_TOP_E2E_PREPROC_SPLIT_LOOP: for (int token = 0; token < aecct_ref::ref_v3::REFV3_TOKENS_T; ++token) {
    const aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload token_payload = preproc_out_token_ch.read();
    if (!aecct_ref::ref_v3::REFV3_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != aecct_ref::ref_v3::REFV3_LAYER0_ID) {
      return false;
    }
    ref_in_token_ch.write(token_payload);
    wrapper_in_token_ch.write(token_payload);
  }
  return true;
}

static bool run_final_head_from_tokens(
  int pattern_idx,
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload>& in_token_ch,
  aecct_ref::ref_v3::RefV3FinalOutputPayload* out_payload) {
  if (out_payload == nullptr) {
    return false;
  }

  const int base = pattern_idx * kVarN;

  aecct_ref::ref_v3::RefV3FinalPassABlock final_pass_a_block;
  aecct_ref::ref_v3::RefV3FinalPassBBlock final_pass_b_block;
  ac_channel<aecct_ref::ref_v3::RefV3FinalScalarTokenPayload> finala_out_scalar_ch;
  ac_channel<aecct_ref::ref_v3::RefV3FinalInputYPayload> finalb_in_input_y_ch;
  ac_channel<aecct_ref::ref_v3::RefV3FinalOutputPayload> finalb_out_payload_ch;

  if (!final_pass_a_block.run(in_token_ch, finala_out_scalar_ch)) {
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
  return aecct_ref::ref_v3::REFV3_var_count_matches_shape(out_payload->var_count);
}

static bool run_case(int pattern_idx, CaseResult* out) {
  if (out == nullptr) {
    return false;
  }
  out->pattern_idx = pattern_idx;

  aecct_ref::RefRunConfig run_cfg = aecct_ref::make_fp32_baseline_run_config();
  run_cfg.legacy.ln_mode = aecct_ref::RefLayerNormMode::LN_BASELINE;

  aecct_ref::ref_v3::RefV3PreprocBlock preproc_block;
  aecct_ref::ref_v3::RefV3Layer0AttnLnPath ref_path_layer0;
  aecct_ref::ref_v3::RefV3CatapultTop wrapper_path_layer0;

  ac_channel<aecct_ref::ref_v3::RefV3PreprocInputPayload> preproc_in_ch;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> preproc_out_token_ch;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ref_in_token_ch;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> wrapper_in_token_ch;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ref_out_token_ch;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> wrapper_out_token_ch;

  if (!stream_preproc_input(pattern_idx, preproc_in_ch)) {
    return false;
  }
  if (!preproc_block.run(preproc_in_ch, preproc_out_token_ch)) {
    out->run_ok = false;
    out->result_pass = false;
    return true;
  }
  if (!stream_preproc_output_to_dual_inputs(preproc_out_token_ch, ref_in_token_ch, wrapper_in_token_ch)) {
    out->run_ok = false;
    out->result_pass = false;
    return true;
  }
  if (!ref_path_layer0.run(run_cfg, ref_in_token_ch, ref_out_token_ch)) {
    out->run_ok = false;
    out->result_pass = false;
    return true;
  }
  if (!wrapper_path_layer0.run(wrapper_in_token_ch, wrapper_out_token_ch)) {
    out->run_ok = false;
    out->result_pass = false;
    return true;
  }

  aecct_ref::ref_v3::RefV3FinalOutputPayload ref_final_payload;
  aecct_ref::ref_v3::RefV3FinalOutputPayload wrapper_final_payload;
  if (!run_final_head_from_tokens(pattern_idx, ref_out_token_ch, &ref_final_payload)) {
    out->run_ok = false;
    out->result_pass = false;
    return true;
  }
  if (!run_final_head_from_tokens(pattern_idx, wrapper_out_token_ch, &wrapper_final_payload)) {
    out->run_ok = false;
    out->result_pass = false;
    return true;
  }

  out->xpred_mismatch_count = 0;
  out->xpred_first_mismatch_idx = -1;
  out->logits_mismatch_count = 0;
  out->logits_max_abs_diff = 0.0;
  out->logits_first_mismatch_idx = -1;

  REFV3_CATAPULT_TOP_E2E_COMPARE_LOOP: for (int n = 0; n < kVarN; ++n) {
    const int ref_bit = ref_final_payload.x_pred[n].to_int();
    const int wrapper_bit = wrapper_final_payload.x_pred[n].to_int();
    if (ref_bit != wrapper_bit) {
      ++out->xpred_mismatch_count;
      if (out->xpred_first_mismatch_idx < 0) {
        out->xpred_first_mismatch_idx = n;
        out->xpred_first_ref = ref_bit;
        out->xpred_first_wrapper = wrapper_bit;
      }
    }

    const double ref_logit = static_cast<double>(ref_final_payload.logits[n].to_float());
    const double wrapper_logit = static_cast<double>(wrapper_final_payload.logits[n].to_float());
    const double abs_diff = std::fabs(ref_logit - wrapper_logit);
    if (abs_diff > out->logits_max_abs_diff) {
      out->logits_max_abs_diff = abs_diff;
    }
    if (abs_diff > kLogitTol) {
      ++out->logits_mismatch_count;
      if (out->logits_first_mismatch_idx < 0) {
        out->logits_first_mismatch_idx = n;
        out->logits_first_ref = ref_logit;
        out->logits_first_wrapper = wrapper_logit;
      }
    }
  }

  out->run_ok = true;
  out->result_pass = (out->xpred_mismatch_count == 0);
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
  for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    if (pattern_idx < 0 || pattern_idx >= trace_batch) {
      std::printf("FAIL: pattern index out of range: %d (trace_batch=%d)\n", pattern_idx, trace_batch);
      return 1;
    }
  }

  int run_ok_count = 0;
  int run_fail_count = 0;
  int xpred_exact_match_count = 0;
  int xpred_mismatch_patterns = 0;
  int logits_patterns_with_mismatch = 0;
  int first_mismatch_pattern_idx = -1;
  int first_mismatch_bit_idx = -1;
  int first_mismatch_ref_bit = -1;
  int first_mismatch_wrapper_bit = -1;

  for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    CaseResult result;
    const bool case_exec_ok = run_case(pattern_idx, &result);
    if (!case_exec_ok) {
      ++run_fail_count;
      ++xpred_mismatch_patterns;
      std::printf(
        "[ref_v3_catapult_top_e2e_compare] pattern_idx=%d xpred_mismatch_count=-1 "
        "first_xpred_mismatch={idx=-1,ref=-1,wrapper=-1} "
        "final_logits_cmp={informational_only=1,mismatch_count=-1,max_abs_diff=nan,first_mismatch={idx=-1,ref=nan,wrapper=nan}} "
        "run_ok=0 result=FAIL\n",
        pattern_idx);
      continue;
    }

    if (result.run_ok) {
      ++run_ok_count;
    } else {
      ++run_fail_count;
    }
    if (result.xpred_mismatch_count == 0) {
      ++xpred_exact_match_count;
    } else {
      ++xpred_mismatch_patterns;
      if (first_mismatch_pattern_idx < 0) {
        first_mismatch_pattern_idx = result.pattern_idx;
        first_mismatch_bit_idx = result.xpred_first_mismatch_idx;
        first_mismatch_ref_bit = result.xpred_first_ref;
        first_mismatch_wrapper_bit = result.xpred_first_wrapper;
      }
    }
    if (result.logits_mismatch_count > 0) {
      ++logits_patterns_with_mismatch;
    }

    std::printf(
      "[ref_v3_catapult_top_e2e_compare] pattern_idx=%d xpred_mismatch_count=%d "
      "first_xpred_mismatch={idx=%d,ref=%d,wrapper=%d} "
      "final_logits_cmp={informational_only=1,mismatch_count=%d,max_abs_diff=%.9e,first_mismatch={idx=%d,ref=%.9e,wrapper=%.9e}} "
      "run_ok=%d result=%s\n",
      pattern_idx,
      result.xpred_mismatch_count,
      result.xpred_first_mismatch_idx,
      result.xpred_first_ref,
      result.xpred_first_wrapper,
      result.logits_mismatch_count,
      result.logits_max_abs_diff,
      result.logits_first_mismatch_idx,
      result.logits_first_ref,
      result.logits_first_wrapper,
      result.run_ok ? 1 : 0,
      result.result_pass ? "PASS" : "FAIL");
  }

  std::printf(
    "[ref_v3_catapult_top_e2e_compare_summary] total=%d run_ok=%d run_fail=%d xpred_exact_match_count=%d "
    "xpred_mismatch_patterns=%d logits_patterns_with_mismatch=%d\n",
    static_cast<int>(kPatternIndices.size()),
    run_ok_count,
    run_fail_count,
    xpred_exact_match_count,
    xpred_mismatch_patterns,
    logits_patterns_with_mismatch);

  if (first_mismatch_pattern_idx >= 0) {
    std::printf(
      "[ref_v3_catapult_top_e2e_compare_first_mismatch] pattern_idx=%d first_xpred_mismatch={idx=%d,ref=%d,wrapper=%d}\n",
      first_mismatch_pattern_idx,
      first_mismatch_bit_idx,
      first_mismatch_ref_bit,
      first_mismatch_wrapper_bit);
  }

  if (run_fail_count != 0 || xpred_mismatch_patterns != 0) {
    std::printf("FAIL: tb_ref_v3_catapult_top_e2e_compare\n");
    return 2;
  }

  std::printf("PASS: tb_ref_v3_catapult_top_e2e_compare\n");
  return 0;
}
