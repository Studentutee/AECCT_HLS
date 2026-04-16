#include <array>
#include <cmath>
#include <cstdio>

#if defined(__SYNTHESIS__) || defined(REFV3_SYNTH_ONLY)
#error "tb_refmodel_v3_binary16_capability_compare is host-only."
#endif

#include "AECCT_ac_ref/include/RefModel.h"
#include "AECCT_ac_ref/include/ref_v3/RefModel_v3.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3FinalPassABlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3FinalPassBBlock.h"
#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"

namespace {

static constexpr int kVarN = 63;
static constexpr double kLogitTol = 1.0e-6;
static constexpr std::array<int, 16> kPatternIndices = {
  77, 116, 132, 179, 217, 265, 312, 572, 602, 689, 701, 759, 789, 849, 906, 976};

enum class CapabilityRelation {
  kDutBetter = 0,
  kEqual = 1,
  kTraceBetter = 2,
  kUnavailable = 3
};

struct CaseResult {
  int pattern_idx = -1;
  bool run_ok = false;
  bool result_pass = false;

  int trace_ones_count = -1;
  int dut_ones_count = -1;
  int dut_vs_trace_bit_errors = -1;
  CapabilityRelation relation = CapabilityRelation::kUnavailable;

  int final_logits_mismatch_count = -1;
  double final_logits_max_abs_diff = 0.0;
};

static const char* relation_to_ascii(CapabilityRelation relation) {
  switch (relation) {
    case CapabilityRelation::kDutBetter:
      return "DUT_BETTER";
    case CapabilityRelation::kEqual:
      return "EQUAL";
    case CapabilityRelation::kTraceBetter:
      return "TRACE_BETTER";
    default:
      return "NA";
  }
}

static bool run_case(int pattern_idx, CaseResult* out) {
  if (out == nullptr) {
    return false;
  }
  out->pattern_idx = pattern_idx;

  const int base = pattern_idx * kVarN;

  aecct_ref::ref_v3::RefModel_v3 model_v3;
  aecct_ref::RefRunConfig cfg = aecct_ref::make_fp32_baseline_run_config();
  cfg.legacy.ln_mode = aecct_ref::RefLayerNormMode::LN_BASELINE;
  model_v3.set_run_config(cfg);

  aecct_ref::RefModelIO io{};
  io.input_y_fp32 = &trace_input_y_step0_tensor[base];
  io.B = 1;
  io.N = kVarN;

  out->run_ok = model_v3.run_step0_layer0_attention_compare(io, 0);
  if (!out->run_ok) {
    out->result_pass = false;
    return true;
  }

  aecct_ref::ref_v3::RefV3FinalPassABlock final_pass_a_block;
  aecct_ref::ref_v3::RefV3FinalPassBBlock final_pass_b_block;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> finala_in_token_ch;
  ac_channel<aecct_ref::ref_v3::RefV3FinalScalarTokenPayload> finala_out_scalar_ch;
  ac_channel<aecct_ref::ref_v3::RefV3FinalInputYPayload> finalb_in_input_y_ch;
  ac_channel<aecct_ref::ref_v3::RefV3FinalOutputPayload> finalb_out_payload_ch;

  REFV3_CAP_TB_STREAM_FINALA_TOKEN_LOOP: for (int token = 0; token < aecct_ref::ref_v3::REFV3_TOKENS_T; ++token) {
    aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(aecct_ref::ref_v3::REFV3_LAYER1_ID);
    token_payload.header.token_rows = ac_int<16, false>(aecct_ref::ref_v3::REFV3_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(aecct_ref::ref_v3::REFV3_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV3_CAP_TB_STREAM_FINALA_DIM_LOOP: for (int dim = 0; dim < aecct_ref::ref_v3::REFV3_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = model_v3.x_work(token, dim);
    }
    finala_in_token_ch.write(token_payload);
  }
  if (!final_pass_a_block.run(finala_in_token_ch, finala_out_scalar_ch)) {
    return false;
  }

  aecct_ref::ref_v3::RefV3FinalInputYPayload input_y_payload;
  input_y_payload.var_count = ac_int<16, false>(kVarN);
  REFV3_CAP_TB_FINALB_INPUT_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
    input_y_payload.input_y[n] = aecct_ref::ref_v3::refv3_fp_t(
      static_cast<float>(trace_input_y_step0_tensor[base + n]));
  }
  finalb_in_input_y_ch.write(input_y_payload);

  if (!final_pass_b_block.run(finala_out_scalar_ch, finalb_in_input_y_ch, finalb_out_payload_ch)) {
    return false;
  }
  const aecct_ref::ref_v3::RefV3FinalOutputPayload dut_payload = finalb_out_payload_ch.read();
  if (!aecct_ref::ref_v3::REFV3_var_count_matches_shape(dut_payload.var_count)) {
    return false;
  }

  out->trace_ones_count = 0;
  out->dut_ones_count = 0;
  out->dut_vs_trace_bit_errors = 0;
  out->final_logits_mismatch_count = 0;
  out->final_logits_max_abs_diff = 0.0;

  REFV3_CAP_TB_COMPARE_LOOP: for (int n = 0; n < kVarN; ++n) {
    const int trace_bit = (trace_output_x_pred_step0_tensor[base + n] != 0.0) ? 1 : 0;
    const int dut_bit = dut_payload.x_pred[n].to_int();

    if (trace_bit != 0) {
      ++out->trace_ones_count;
    }
    if (dut_bit != 0) {
      ++out->dut_ones_count;
    }
    if (dut_bit != trace_bit) {
      ++out->dut_vs_trace_bit_errors;
    }

    const double dut_logit = static_cast<double>(dut_payload.logits[n].to_float());
    const double trace_logit = trace_output_logits_step0_tensor[base + n];
    const double abs_diff = std::fabs(dut_logit - trace_logit);
    if (abs_diff > out->final_logits_max_abs_diff) {
      out->final_logits_max_abs_diff = abs_diff;
    }
    if (abs_diff > kLogitTol) {
      ++out->final_logits_mismatch_count;
    }
  }

  if (out->dut_ones_count < out->trace_ones_count) {
    out->relation = CapabilityRelation::kDutBetter;
  } else if (out->dut_ones_count == out->trace_ones_count) {
    out->relation = CapabilityRelation::kEqual;
  } else {
    out->relation = CapabilityRelation::kTraceBetter;
  }

  out->result_pass = true;
  return true;
}

} // namespace

int main() {
  if (trace_input_y_step0_tensor_ndim != 2 || trace_output_x_pred_step0_tensor_ndim != 2 ||
      trace_output_logits_step0_tensor_ndim != 2) {
    std::printf("FAIL: unexpected trace ndim\n");
    return 1;
  }
  if (trace_input_y_step0_tensor_shape[1] != kVarN || trace_output_x_pred_step0_tensor_shape[1] != kVarN ||
      trace_output_logits_step0_tensor_shape[1] != kVarN) {
    std::printf("FAIL: unexpected trace tensor width\n");
    return 1;
  }
  if (trace_input_y_step0_tensor_shape[0] != trace_output_x_pred_step0_tensor_shape[0] ||
      trace_input_y_step0_tensor_shape[0] != trace_output_logits_step0_tensor_shape[0]) {
    std::printf("FAIL: trace batch dimension mismatch\n");
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
  int dut_better_count = 0;
  int equal_count = 0;
  int trace_better_count = 0;
  int dut_exact_zero_count = 0;
  int trace_exact_zero_count = 0;

  for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    CaseResult result;
    const bool case_exec_ok = run_case(pattern_idx, &result);
    if (!case_exec_ok) {
      ++run_fail_count;
      std::printf(
        "[ref_v3_binary16_capability_compare] pattern_idx=%d trace_ones_count=-1 dut_ones_count=-1 "
        "dut_vs_trace_bit_errors=-1 relation=NA run_ok=0 "
        "final_logits_mismatch_count=-1 final_logits_max_abs_diff=nan informational_only=1 result=FAIL\n",
        pattern_idx);
      continue;
    }

    if (!result.run_ok) {
      ++run_fail_count;
    } else {
      ++run_ok_count;
    }

    if (result.run_ok) {
      if (result.relation == CapabilityRelation::kDutBetter) {
        ++dut_better_count;
      } else if (result.relation == CapabilityRelation::kEqual) {
        ++equal_count;
      } else if (result.relation == CapabilityRelation::kTraceBetter) {
        ++trace_better_count;
      }

      if (result.dut_ones_count == 0) {
        ++dut_exact_zero_count;
      }
      if (result.trace_ones_count == 0) {
        ++trace_exact_zero_count;
      }
    }

    std::printf(
      "[ref_v3_binary16_capability_compare] pattern_idx=%d trace_ones_count=%d dut_ones_count=%d "
      "dut_vs_trace_bit_errors=%d relation=%s run_ok=%d "
      "final_logits_mismatch_count=%d final_logits_max_abs_diff=%.9e informational_only=1 result=%s\n",
      pattern_idx,
      result.trace_ones_count,
      result.dut_ones_count,
      result.dut_vs_trace_bit_errors,
      relation_to_ascii(result.relation),
      result.run_ok ? 1 : 0,
      result.final_logits_mismatch_count,
      result.final_logits_max_abs_diff,
      result.result_pass ? "PASS" : "FAIL");
  }

  std::printf(
    "[ref_v3_binary16_capability_compare_summary] total=%d run_ok=%d run_fail=%d dut_better_count=%d equal_count=%d trace_better_count=%d "
    "dut_exact_zero_count=%d trace_exact_zero_count=%d\n",
    static_cast<int>(kPatternIndices.size()),
    run_ok_count,
    run_fail_count,
    dut_better_count,
    equal_count,
    trace_better_count,
    dut_exact_zero_count,
    trace_exact_zero_count);

  if (run_fail_count != 0) {
    std::printf("FAIL: tb_refmodel_v3_binary16_capability_compare\n");
    return 2;
  }

  std::printf("PASS: tb_refmodel_v3_binary16_capability_compare\n");
  return 0;
}
