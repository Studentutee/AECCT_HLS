#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#if defined(__SYNTHESIS__) || defined(REFV3_SYNTH_ONLY)
#error "tb_refmodel_v3_binary16_e2e_smoke_compare is host-only."
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

struct CaseResult {
  int pattern_idx = -1;
  bool run_ok = false;
  bool result_pass = false;

  int xpred_mismatch_count = 0;
  int xpred_first_mismatch_idx = -1;
  int xpred_first_dut = -1;
  int xpred_first_trace = -1;
  int trace_ones_count = 0;
  int dut_ones_count = 0;
  bool degraded_vs_zero = false;

  int logits_mismatch_count = 0;
  double logits_max_abs_diff = 0.0;
  int logits_first_mismatch_idx = -1;
  double logits_first_dut = 0.0;
  double logits_first_trace = 0.0;

  std::array<int, kVarN> trace_xpred_bits{};
  std::array<int, kVarN> dut_xpred_bits{};
};

static std::string bits_to_ascii(const std::array<int, kVarN>& bits) {
  std::string s;
  s.reserve(static_cast<std::size_t>(kVarN));
  for (int n = 0; n < kVarN; ++n) {
    s.push_back(bits[static_cast<std::size_t>(n)] != 0 ? '1' : '0');
  }
  return s;
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

  REFV3_TB_STREAM_FINALA_TOKEN_LOOP: for (int token = 0; token < aecct_ref::ref_v3::REFV3_TOKENS_T; ++token) {
    aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(aecct_ref::ref_v3::REFV3_LAYER1_ID);
    token_payload.header.token_rows = ac_int<16, false>(aecct_ref::ref_v3::REFV3_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(aecct_ref::ref_v3::REFV3_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV3_TB_STREAM_FINALA_DIM_LOOP: for (int dim = 0; dim < aecct_ref::ref_v3::REFV3_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = model_v3.x_work(token, dim);
    }
    finala_in_token_ch.write(token_payload);
  }
  if (!final_pass_a_block.run(finala_in_token_ch, finala_out_scalar_ch)) {
    return false;
  }

  aecct_ref::ref_v3::RefV3FinalInputYPayload input_y_payload;
  input_y_payload.var_count = ac_int<16, false>(kVarN);
  REFV3_TB_FINALB_INPUT_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
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

  out->xpred_mismatch_count = 0;
  out->xpred_first_mismatch_idx = -1;
  out->trace_ones_count = 0;
  out->dut_ones_count = 0;
  out->logits_mismatch_count = 0;
  out->logits_max_abs_diff = 0.0;
  out->logits_first_mismatch_idx = -1;

  for (int n = 0; n < kVarN; ++n) {
    const int trace_bit = (trace_output_x_pred_step0_tensor[base + n] != 0.0) ? 1 : 0;
    const int dut_bit = dut_payload.x_pred[n].to_int();
    out->trace_xpred_bits[static_cast<std::size_t>(n)] = trace_bit;
    out->dut_xpred_bits[static_cast<std::size_t>(n)] = dut_bit;

    if (trace_bit != 0) {
      ++out->trace_ones_count;
    }
    if (dut_bit != 0) {
      ++out->dut_ones_count;
    }
    if (dut_bit != trace_bit) {
      ++out->xpred_mismatch_count;
      if (out->xpred_first_mismatch_idx < 0) {
        out->xpred_first_mismatch_idx = n;
        out->xpred_first_dut = dut_bit;
        out->xpred_first_trace = trace_bit;
      }
    }

    const double dut_logit = static_cast<double>(dut_payload.logits[n].to_float());
    const double trace_logit = trace_output_logits_step0_tensor[base + n];
    const double abs_diff = std::fabs(dut_logit - trace_logit);
    if (abs_diff > out->logits_max_abs_diff) {
      out->logits_max_abs_diff = abs_diff;
    }
    if (abs_diff > kLogitTol) {
      ++out->logits_mismatch_count;
      if (out->logits_first_mismatch_idx < 0) {
        out->logits_first_mismatch_idx = n;
        out->logits_first_dut = dut_logit;
        out->logits_first_trace = trace_logit;
      }
    }
  }

  out->degraded_vs_zero = (out->xpred_mismatch_count > out->trace_ones_count);
  out->result_pass = !out->degraded_vs_zero;
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
  int pass_count = 0;
  int fail_count = 0;
  int xpred_exact_match_count = 0;
  int degraded_count = 0;
  int worse_than_zero_count = 0;
  int logits_mismatch_pattern_count = 0;
  int logits_mismatch_total = 0;

  for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    CaseResult result;
    const bool case_exec_ok = run_case(pattern_idx, &result);
    if (!case_exec_ok) {
      ++fail_count;
      ++run_fail_count;
      std::printf(
        "[ref_v3_binary16_smoke] pattern_idx=%d trace_x_pred=NA dut_final_x_pred=NA "
        "xpred_mismatch_count=-1 xpred_first_mismatch={idx=-1,dut=-1,trace=-1} "
        "dut_vs_trace_bit_errors=-1 zero_codeword_vs_trace_bit_errors=-1 dut_ones_count=-1 degraded_vs_zero=1 "
        "final_logits_trace_cmp={non_gating=1,informational_only=1,mismatch_count=-1,max_abs_diff=nan,first_mismatch={idx=-1,dut=nan,trace=nan}} "
        "result=FAIL run_ok=0\n",
        pattern_idx);
      continue;
    }
    if (!result.run_ok) {
      ++fail_count;
      ++run_fail_count;
    } else {
      ++run_ok_count;
    }
    if (result.result_pass) {
      ++pass_count;
    } else {
      ++fail_count;
    }
    if (result.xpred_mismatch_count == 0) {
      ++xpred_exact_match_count;
    }
    if (result.degraded_vs_zero) {
      ++degraded_count;
      ++worse_than_zero_count;
    }
    if (result.logits_mismatch_count > 0) {
      ++logits_mismatch_pattern_count;
    }
    logits_mismatch_total += result.logits_mismatch_count;

    const std::string trace_bits_ascii = bits_to_ascii(result.trace_xpred_bits);
    const std::string dut_bits_ascii = bits_to_ascii(result.dut_xpred_bits);

    std::printf(
      "[ref_v3_binary16_smoke] pattern_idx=%d trace_x_pred=%s dut_final_x_pred=%s "
      "xpred_mismatch_count=%d xpred_first_mismatch={idx=%d,dut=%d,trace=%d} "
      "dut_vs_trace_bit_errors=%d zero_codeword_vs_trace_bit_errors=%d dut_ones_count=%d degraded_vs_zero=%d "
      "final_logits_trace_cmp={non_gating=1,informational_only=1,mismatch_count=%d,max_abs_diff=%.9e,first_mismatch={idx=%d,dut=%.9e,trace=%.9e}} "
      "result=%s run_ok=%d\n",
      pattern_idx,
      trace_bits_ascii.c_str(),
      dut_bits_ascii.c_str(),
      result.xpred_mismatch_count,
      result.xpred_first_mismatch_idx,
      result.xpred_first_dut,
      result.xpred_first_trace,
      result.xpred_mismatch_count,
      result.trace_ones_count,
      result.dut_ones_count,
      result.degraded_vs_zero ? 1 : 0,
      result.logits_mismatch_count,
      result.logits_max_abs_diff,
      result.logits_first_mismatch_idx,
      result.logits_first_dut,
      result.logits_first_trace,
      result.result_pass ? "PASS" : "FAIL",
      result.run_ok ? 1 : 0);
  }

  std::printf(
    "[ref_v3_binary16_smoke_summary] total=%d run_ok=%d run_fail=%d pass=%d fail=%d xpred_exact_match=%d non_degraded=%d degraded=%d worse_than_zero=%d\n",
    static_cast<int>(kPatternIndices.size()),
    run_ok_count,
    run_fail_count,
    pass_count,
    fail_count,
    xpred_exact_match_count,
    pass_count,
    degraded_count,
    worse_than_zero_count);

  std::printf(
    "[ref_v3_binary16_smoke_logits_info] non_gating=1 informational_only=1 patterns_with_mismatch=%d total_mismatch=%d\n",
    logits_mismatch_pattern_count,
    logits_mismatch_total);

  if (run_fail_count != 0 || degraded_count != 0) {
    std::printf("FAIL: tb_refmodel_v3_binary16_e2e_smoke_compare\n");
    return 2;
  }

  std::printf("PASS: tb_refmodel_v3_binary16_e2e_smoke_compare\n");
  return 0;
}
