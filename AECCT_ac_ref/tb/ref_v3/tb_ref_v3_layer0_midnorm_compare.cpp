#include <array>
#include <cmath>
#include <cstdio>

#if defined(__SYNTHESIS__) || defined(REFV3_SYNTH_ONLY)
#error "tb_ref_v3_layer0_midnorm_compare is host-only."
#endif

#include "AECCT_ac_ref/include/RefModel.h"
#include "AECCT_ac_ref/include/RefModelOptimized.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer0AttnLnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer0FfnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3MidNormPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3PreprocBlock.h"
#include "input_y_step0.h"

namespace {

static constexpr int kVarN = 63;
static constexpr std::array<int, 8> kPatternIndices = {77, 116, 132, 179, 217, 265, 312, 572};

struct CaseResult {
  int pattern_idx = -1;
  bool run_ok = false;
  int shape_ok = 0;
  double max_abs_err = 0.0;
  double mean_abs_err = 0.0;
  double mse = 0.0;
  double cosine_similarity = 0.0;
  int nan_count = 0;
};

static bool stream_preproc_input(
  int pattern_idx,
  ac_channel<aecct_ref::ref_v3::RefV3PreprocInputPayload>& preproc_in_ch) {
  const int base = pattern_idx * kVarN;
  aecct_ref::ref_v3::RefV3PreprocInputPayload input_payload;
  input_payload.var_count = ac_int<16, false>(kVarN);

  REFV3_LAYER0_MIDNORM_TB_INPUT_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
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

  aecct_ref::RefRunConfig run_cfg = aecct_ref::make_fp32_baseline_run_config();

  aecct_ref::RefModelOptimized reference_model;
  reference_model.set_run_config(run_cfg);

  const int base = pattern_idx * kVarN;
  aecct_ref::RefModelIO io{};
  io.input_y_fp32 = &trace_input_y_step0_tensor[base];
  io.B = 1;
  io.N = kVarN;

  if (!reference_model.stage_step0_phase_a(io, 0) ||
      !reference_model.run_step0_layer0_attention_writeback() ||
      !reference_model.run_step0_layer0_ln_writeback() ||
      !reference_model.run_step0_layer0_ffn_writeback() ||
      !reference_model.run_step0_mid_norm_writeback()) {
    out->run_ok = false;
    return true;
  }

  aecct_ref::ref_v3::RefV3PreprocBlock preproc_block;
  aecct_ref::ref_v3::RefV3Layer0AttnLnPath layer0_attn_ln_path;
  aecct_ref::ref_v3::RefV3Layer0FfnPath layer0_ffn_path;
  aecct_ref::ref_v3::RefV3MidNormPath mid_norm_path;

  ac_channel<aecct_ref::ref_v3::RefV3PreprocInputPayload> ch_preproc_in;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_preproc_to_l0_attn;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionInputPayload> ch_xwork0_side;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_l0_attn_to_ffn;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_l0_ffn_to_midnorm;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_midnorm_out_token;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionInputPayload> ch_midnorm_out_xwork;

  if (!stream_preproc_input(pattern_idx, ch_preproc_in)) {
    out->run_ok = false;
    return true;
  }
  if (!preproc_block.run(ch_preproc_in, ch_preproc_to_l0_attn, ch_xwork0_side)) {
    out->run_ok = false;
    return true;
  }
  if (!layer0_attn_ln_path.run(run_cfg, ch_preproc_to_l0_attn, ch_xwork0_side, ch_l0_attn_to_ffn)) {
    out->run_ok = false;
    return true;
  }
  if (!layer0_ffn_path.run(ch_l0_attn_to_ffn, ch_l0_ffn_to_midnorm)) {
    out->run_ok = false;
    return true;
  }
  if (!mid_norm_path.run(run_cfg, ch_l0_ffn_to_midnorm, ch_midnorm_out_token, ch_midnorm_out_xwork)) {
    out->run_ok = false;
    return true;
  }

  bool shape_ok = true;
  bool token_seen[aecct_ref::ref_v3::REFV3_TOKENS_T];
  aecct_ref::ref_v3::refv3_fp_t dut_midnorm_tokens[aecct_ref::ref_v3::REFV3_TOKENS_T][aecct_ref::ref_v3::REFV3_D_MODEL];

  REFV3_LAYER0_MIDNORM_TB_TOKEN_SEEN_INIT_LOOP: for (int token = 0;
                                                       token < aecct_ref::ref_v3::REFV3_TOKENS_T;
                                                       ++token) {
    token_seen[token] = false;
    REFV3_LAYER0_MIDNORM_TB_TOKEN_BUF_INIT_DIM_LOOP: for (int dim = 0;
                                                           dim < aecct_ref::ref_v3::REFV3_D_MODEL;
                                                           ++dim) {
      dut_midnorm_tokens[token][dim] = aecct_ref::ref_v3::refv3_fp_t(0.0f);
    }
  }

  REFV3_LAYER0_MIDNORM_TB_TOKEN_READ_LOOP: for (int token_rx = 0;
                                                 token_rx < aecct_ref::ref_v3::REFV3_TOKENS_T;
                                                 ++token_rx) {
    const aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload token_payload = ch_midnorm_out_token.read();
    if (!aecct_ref::ref_v3::REFV3_payload_header_matches_shape(token_payload.header)) {
      shape_ok = false;
      continue;
    }
    if (token_payload.header.layer_id.to_int() != aecct_ref::ref_v3::REFV3_LAYER1_ID) {
      shape_ok = false;
      continue;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= aecct_ref::ref_v3::REFV3_TOKENS_T) {
      shape_ok = false;
      continue;
    }
    if (token_seen[token]) {
      shape_ok = false;
      continue;
    }
    token_seen[token] = true;

    REFV3_LAYER0_MIDNORM_TB_TOKEN_COPY_DIM_LOOP: for (int dim = 0;
                                                       dim < aecct_ref::ref_v3::REFV3_D_MODEL;
                                                       ++dim) {
      dut_midnorm_tokens[token][dim] = token_payload.token_vec[dim];
    }
  }

  REFV3_LAYER0_MIDNORM_TB_TOKEN_MISSING_LOOP: for (int token = 0;
                                                    token < aecct_ref::ref_v3::REFV3_TOKENS_T;
                                                    ++token) {
    if (!token_seen[token]) {
      shape_ok = false;
    }
  }

  const aecct_ref::ref_v3::RefV3AttentionInputPayload xwork_payload = ch_midnorm_out_xwork.read();
  if (!aecct_ref::ref_v3::REFV3_payload_header_matches_shape(xwork_payload.header)) {
    shape_ok = false;
  }
  if (xwork_payload.header.layer_id.to_int() != aecct_ref::ref_v3::REFV3_LAYER1_ID) {
    shape_ok = false;
  }

  REFV3_LAYER0_MIDNORM_TB_XWORK_CROSSCHECK_LOOP: for (int token = 0;
                                                       token < aecct_ref::ref_v3::REFV3_TOKENS_T;
                                                       ++token) {
    REFV3_LAYER0_MIDNORM_TB_XWORK_CROSSCHECK_DIM_LOOP: for (int dim = 0;
                                                             dim < aecct_ref::ref_v3::REFV3_D_MODEL;
                                                             ++dim) {
      const int idx = aecct_ref::ref_v3::REFV3_flatten_row_major_index(token, dim);
      const float token_v = dut_midnorm_tokens[token][dim].to_float();
      const float xwork_v = xwork_payload.x_flat[idx].to_float();
      if (token_v != xwork_v) {
        shape_ok = false;
      }
    }
  }

  double sum_abs_err = 0.0;
  double sum_sq_err = 0.0;
  double dot = 0.0;
  double dut_norm_sq = 0.0;
  double ref_norm_sq = 0.0;
  int valid_count = 0;

  REFV3_LAYER0_MIDNORM_TB_METRIC_TOKEN_LOOP: for (int token = 0;
                                                   token < aecct_ref::ref_v3::REFV3_TOKENS_T;
                                                   ++token) {
    REFV3_LAYER0_MIDNORM_TB_METRIC_DIM_LOOP: for (int dim = 0;
                                                   dim < aecct_ref::ref_v3::REFV3_D_MODEL;
                                                   ++dim) {
      const double dut_v = static_cast<double>(dut_midnorm_tokens[token][dim].to_float());
      const double ref_v = static_cast<double>(reference_model.x_work(token, dim).to_float());

      const bool dut_nan = std::isnan(dut_v);
      const bool ref_nan = std::isnan(ref_v);
      if (dut_nan || ref_nan) {
        ++out->nan_count;
        continue;
      }

      const double err = dut_v - ref_v;
      const double abs_err = std::fabs(err);
      if (abs_err > out->max_abs_err) {
        out->max_abs_err = abs_err;
      }
      sum_abs_err += abs_err;
      sum_sq_err += (err * err);
      dot += (dut_v * ref_v);
      dut_norm_sq += (dut_v * dut_v);
      ref_norm_sq += (ref_v * ref_v);
      ++valid_count;
    }
  }

  if (valid_count > 0) {
    const double inv_count = 1.0 / static_cast<double>(valid_count);
    out->mean_abs_err = sum_abs_err * inv_count;
    out->mse = sum_sq_err * inv_count;

    const double norm_mul = std::sqrt(dut_norm_sq) * std::sqrt(ref_norm_sq);
    if (norm_mul > 0.0) {
      out->cosine_similarity = dot / norm_mul;
    } else {
      out->cosine_similarity = (dut_norm_sq == 0.0 && ref_norm_sq == 0.0) ? 1.0 : 0.0;
    }
  } else {
    out->mean_abs_err = 0.0;
    out->mse = 0.0;
    out->cosine_similarity = 0.0;
  }

  out->shape_ok = shape_ok ? 1 : 0;
  out->run_ok = shape_ok;
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
  REFV3_LAYER0_MIDNORM_TB_PATTERN_RANGE_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    if (pattern_idx < 0 || pattern_idx >= trace_batch) {
      std::printf("FAIL: pattern index out of range: %d (trace_batch=%d)\n", pattern_idx, trace_batch);
      return 1;
    }
  }

  int run_ok_count = 0;
  int run_fail_count = 0;
  double max_abs_err_worst = 0.0;
  double mean_abs_err_avg = 0.0;
  double mse_avg = 0.0;
  double cosine_similarity_avg = 0.0;
  int nan_count_total = 0;

  REFV3_LAYER0_MIDNORM_TB_CASE_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    CaseResult result;
    const bool case_exec_ok = run_case(pattern_idx, &result);
    if (!case_exec_ok || !result.run_ok) {
      ++run_fail_count;
      std::printf(
        "[ref_v3_layer0_midnorm_compare] pattern_idx=%d shape_ok=0 max_abs_err=nan mean_abs_err=nan mse=nan cosine_similarity=nan nan_count=-1 result=RUN_FAIL\n",
        pattern_idx);
      continue;
    }

    ++run_ok_count;
    if (result.max_abs_err > max_abs_err_worst) {
      max_abs_err_worst = result.max_abs_err;
    }
    mean_abs_err_avg += result.mean_abs_err;
    mse_avg += result.mse;
    cosine_similarity_avg += result.cosine_similarity;
    nan_count_total += result.nan_count;

    std::printf(
      "[ref_v3_layer0_midnorm_compare] pattern_idx=%d shape_ok=%d max_abs_err=%.9e mean_abs_err=%.9e mse=%.9e cosine_similarity=%.9e nan_count=%d result=RUN_OK\n",
      result.pattern_idx,
      result.shape_ok,
      result.max_abs_err,
      result.mean_abs_err,
      result.mse,
      result.cosine_similarity,
      result.nan_count);
  }

  if (run_ok_count > 0) {
    const double inv = 1.0 / static_cast<double>(run_ok_count);
    mean_abs_err_avg *= inv;
    mse_avg *= inv;
    cosine_similarity_avg *= inv;
  }

  std::printf(
    "[ref_v3_layer0_midnorm_compare_summary] total_patterns=%d run_ok=%d run_fail=%d max_abs_err_worst=%.9e mean_abs_err_avg=%.9e mse_avg=%.9e cosine_similarity_avg=%.9e nan_count_total=%d\n",
    static_cast<int>(kPatternIndices.size()),
    run_ok_count,
    run_fail_count,
    max_abs_err_worst,
    mean_abs_err_avg,
    mse_avg,
    cosine_similarity_avg,
    nan_count_total);

  if (run_fail_count != 0) {
    std::printf("FAIL: tb_ref_v3_layer0_midnorm_compare\n");
    return 2;
  }

  std::printf("PASS: tb_ref_v3_layer0_midnorm_compare\n");
  return 0;
}
