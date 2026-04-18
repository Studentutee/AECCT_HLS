#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

#if defined(__SYNTHESIS__) || defined(REFV3_SYNTH_ONLY)
#error "tb_ref_v3_tail_boundary_compare is host-only."
#endif

#include "AECCT_ac_ref/include/RefModel.h"
#include "AECCT_ac_ref/include/RefModelOptimized.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3FinalPassABlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer0AttnLnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer0FfnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer1AttnLnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer1FfnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3LayerNormBlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3MidNormPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3PreprocBlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3WeightsFp16LocalOnly.h"
#include "input_y_step0.h"

namespace {

static constexpr int kVarN = 63;
static constexpr int kTokens = aecct_ref::ref_v3::REFV3_TOKENS_T;
static constexpr int kDim = aecct_ref::ref_v3::REFV3_D_MODEL;
static constexpr std::array<int, 8> kPatternIndices = {77, 116, 132, 179, 217, 265, 312, 572};

struct StageMetrics {
  int shape_ok = 0;
  double max_abs_err = std::numeric_limits<double>::quiet_NaN();
  double mean_abs_err = std::numeric_limits<double>::quiet_NaN();
  double mse = std::numeric_limits<double>::quiet_NaN();
  double cosine_similarity = std::numeric_limits<double>::quiet_NaN();
  int nan_count = -1;
  bool run_ok = false;
};

struct CaseResult {
  int pattern_idx = -1;
  bool run_ok = false;
  StageMetrics endnorm;
  StageMetrics finalpassa;
};

static bool stream_preproc_input(
  int pattern_idx,
  ac_channel<aecct_ref::ref_v3::RefV3PreprocInputPayload>& preproc_in_ch) {
  const int base = pattern_idx * kVarN;

  aecct_ref::ref_v3::RefV3PreprocInputPayload input_payload;
  input_payload.var_count = ac_int<16, false>(kVarN);
  REFV3_TAIL_BOUNDARY_INPUT_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
    input_payload.input_y[n] = aecct_ref::ref_v3::refv3_fp_t(
      static_cast<float>(trace_input_y_step0_tensor[base + n]));
  }
  preproc_in_ch.write(input_payload);
  return true;
}

static void compute_metrics(
  const double* dut,
  const double* ref,
  int elems,
  StageMetrics* out) {
  if (out == nullptr) {
    return;
  }

  double sum_abs_err = 0.0;
  double sum_sq_err = 0.0;
  double dot = 0.0;
  double dut_norm_sq = 0.0;
  double ref_norm_sq = 0.0;
  int valid_count = 0;
  int nan_count = 0;
  double max_abs_err = 0.0;

  REFV3_TAIL_BOUNDARY_METRIC_ELEM_LOOP: for (int i = 0; i < elems; ++i) {
    const double dut_v = dut[i];
    const double ref_v = ref[i];
    if (std::isnan(dut_v) || std::isnan(ref_v)) {
      ++nan_count;
      continue;
    }

    const double err = dut_v - ref_v;
    const double abs_err = std::fabs(err);
    if (abs_err > max_abs_err) {
      max_abs_err = abs_err;
    }

    sum_abs_err += abs_err;
    sum_sq_err += (err * err);
    dot += (dut_v * ref_v);
    dut_norm_sq += (dut_v * dut_v);
    ref_norm_sq += (ref_v * ref_v);
    ++valid_count;
  }

  out->nan_count = nan_count;
  if (valid_count == 0) {
    out->max_abs_err = std::numeric_limits<double>::quiet_NaN();
    out->mean_abs_err = std::numeric_limits<double>::quiet_NaN();
    out->mse = std::numeric_limits<double>::quiet_NaN();
    out->cosine_similarity = std::numeric_limits<double>::quiet_NaN();
    out->run_ok = false;
    return;
  }

  out->max_abs_err = max_abs_err;
  out->mean_abs_err = sum_abs_err / static_cast<double>(valid_count);
  out->mse = sum_sq_err / static_cast<double>(valid_count);

  const double denom = std::sqrt(dut_norm_sq) * std::sqrt(ref_norm_sq);
  if (denom > 0.0) {
    out->cosine_similarity = dot / denom;
  } else {
    out->cosine_similarity = (dut_norm_sq == 0.0 && ref_norm_sq == 0.0) ? 1.0 : 0.0;
  }

  out->run_ok = (out->shape_ok == 1);
}

static bool run_case(int pattern_idx, CaseResult* out) {
  if (out == nullptr) {
    return false;
  }
  out->pattern_idx = pattern_idx;

  aecct_ref::RefRunConfig run_cfg = aecct_ref::make_fp32_baseline_run_config();

  aecct_ref::ref_v3::RefV3PreprocBlock preproc_block;
  aecct_ref::ref_v3::RefV3Layer0AttnLnPath layer0_attn_ln_path;
  aecct_ref::ref_v3::RefV3Layer0FfnPath layer0_ffn_path;
  aecct_ref::ref_v3::RefV3MidNormPath mid_norm_path;
  aecct_ref::ref_v3::RefV3Layer1AttnLnPath layer1_attn_ln_path;
  aecct_ref::ref_v3::RefV3Layer1FfnPath layer1_ffn_path;
  aecct_ref::ref_v3::RefV3LayerNormBlock end_norm_block;
  aecct_ref::ref_v3::RefV3FinalPassABlock final_pass_a_block;

  ac_channel<aecct_ref::ref_v3::RefV3PreprocInputPayload> ch_preproc_in;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_preproc_to_l0_attn;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionInputPayload> ch_xwork0_side;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_l0_attn_to_ffn;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_l0_ffn_to_midnorm;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_midnorm_to_l1_attn;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionInputPayload> ch_xwork1_side;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_l1_attn_to_ffn;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_l1_ffn_to_endnorm;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_endnorm_raw;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_endnorm_to_finala;
  ac_channel<aecct_ref::ref_v3::RefV3FinalScalarTokenPayload> ch_finala_out;

  if (!stream_preproc_input(pattern_idx, ch_preproc_in)) {
    return false;
  }
  if (!preproc_block.run(ch_preproc_in, ch_preproc_to_l0_attn, ch_xwork0_side)) {
    return false;
  }
  if (!layer0_attn_ln_path.run(run_cfg, ch_preproc_to_l0_attn, ch_xwork0_side, ch_l0_attn_to_ffn)) {
    return false;
  }
  if (!layer0_ffn_path.run(ch_l0_attn_to_ffn, ch_l0_ffn_to_midnorm)) {
    return false;
  }
  if (!mid_norm_path.run(run_cfg, ch_l0_ffn_to_midnorm, ch_midnorm_to_l1_attn, ch_xwork1_side)) {
    return false;
  }
  if (!layer1_attn_ln_path.run(run_cfg, ch_midnorm_to_l1_attn, ch_xwork1_side, ch_l1_attn_to_ffn)) {
    return false;
  }
  if (!layer1_ffn_path.run(ch_l1_attn_to_ffn, ch_l1_ffn_to_endnorm)) {
    return false;
  }

  const aecct_ref::ref_v3::RefV3TernaryLinearParams endnorm_params =
    aecct_ref::ref_v3::refv3_endnorm_params_fp_local_only();
  if (!end_norm_block.run_with_params(
        aecct_ref::ref_v3::REFV3_LAYER1_ID,
        endnorm_params,
        run_cfg,
        ch_l1_ffn_to_endnorm,
        ch_endnorm_raw)) {
    return false;
  }

  double dut_endnorm[kTokens * kDim];
  bool endnorm_token_seen[kTokens];
  out->endnorm.shape_ok = 1;
  out->endnorm.nan_count = 0;

  REFV3_TAIL_BOUNDARY_ENDNORM_SEEN_INIT_LOOP: for (int token = 0; token < kTokens; ++token) {
    endnorm_token_seen[token] = false;
    REFV3_TAIL_BOUNDARY_ENDNORM_BUF_INIT_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      const int idx = (token * kDim) + dim;
      dut_endnorm[idx] = 0.0;
    }
  }

  REFV3_TAIL_BOUNDARY_ENDNORM_CAPTURE_LOOP: for (int token_rx = 0; token_rx < kTokens; ++token_rx) {
    const aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload payload = ch_endnorm_raw.read();
    ch_endnorm_to_finala.write(payload);

    if (!aecct_ref::ref_v3::REFV3_payload_header_matches_shape(payload.header)) {
      out->endnorm.shape_ok = 0;
      continue;
    }
    if (payload.header.layer_id.to_int() != aecct_ref::ref_v3::REFV3_LAYER1_ID) {
      out->endnorm.shape_ok = 0;
      continue;
    }

    const int token = payload.token_row.to_int();
    if (token < 0 || token >= kTokens) {
      out->endnorm.shape_ok = 0;
      continue;
    }
    if (endnorm_token_seen[token]) {
      out->endnorm.shape_ok = 0;
      continue;
    }
    endnorm_token_seen[token] = true;

    REFV3_TAIL_BOUNDARY_ENDNORM_COPY_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      const int idx = (token * kDim) + dim;
      dut_endnorm[idx] = static_cast<double>(payload.token_vec[dim].to_float());
    }
  }

  REFV3_TAIL_BOUNDARY_ENDNORM_MISSING_LOOP: for (int token = 0; token < kTokens; ++token) {
    if (!endnorm_token_seen[token]) {
      out->endnorm.shape_ok = 0;
    }
  }

  if (!final_pass_a_block.run(ch_endnorm_to_finala, ch_finala_out)) {
    return false;
  }

  double dut_finalpassa[kTokens];
  bool finala_token_seen[kTokens];
  out->finalpassa.shape_ok = 1;
  out->finalpassa.nan_count = 0;

  REFV3_TAIL_BOUNDARY_FINALA_SEEN_INIT_LOOP: for (int token = 0; token < kTokens; ++token) {
    finala_token_seen[token] = false;
    dut_finalpassa[token] = 0.0;
  }

  REFV3_TAIL_BOUNDARY_FINALA_CAPTURE_LOOP: for (int token_rx = 0; token_rx < kTokens; ++token_rx) {
    const aecct_ref::ref_v3::RefV3FinalScalarTokenPayload payload = ch_finala_out.read();

    if (!aecct_ref::ref_v3::REFV3_payload_header_matches_shape(payload.header)) {
      out->finalpassa.shape_ok = 0;
      continue;
    }
    if (payload.header.layer_id.to_int() != aecct_ref::ref_v3::REFV3_LAYER1_ID) {
      out->finalpassa.shape_ok = 0;
      continue;
    }

    const int token = payload.token_row.to_int();
    if (token < 0 || token >= kTokens) {
      out->finalpassa.shape_ok = 0;
      continue;
    }
    if (finala_token_seen[token]) {
      out->finalpassa.shape_ok = 0;
      continue;
    }
    finala_token_seen[token] = true;
    dut_finalpassa[token] = static_cast<double>(payload.scalar.to_float());
  }

  REFV3_TAIL_BOUNDARY_FINALA_MISSING_LOOP: for (int token = 0; token < kTokens; ++token) {
    if (!finala_token_seen[token]) {
      out->finalpassa.shape_ok = 0;
    }
  }

  aecct_ref::RefModelOptimized ref_model;
  ref_model.set_run_config(run_cfg);

  const int base = pattern_idx * kVarN;
  aecct_ref::RefModelIO io{};
  io.input_y_fp32 = &trace_input_y_step0_tensor[base];
  io.B = 1;
  io.N = kVarN;

  if (!ref_model.stage_step0_phase_a(io, 0) ||
      !ref_model.run_step0_layer0_attention_writeback() ||
      !ref_model.run_step0_layer0_ln_writeback() ||
      !ref_model.run_step0_layer0_ffn_writeback() ||
      !ref_model.run_step0_mid_norm_writeback() ||
      !ref_model.run_step0_layer1_attn_input_handoff() ||
      !ref_model.run_step0_layer1_attention_writeback() ||
      !ref_model.run_step0_layer1_ln_writeback() ||
      !ref_model.run_step0_layer1_ffn_writeback() ||
      !ref_model.run_step0_end_norm_writeback() ||
      !ref_model.run_step0_final_head_pass_a_writeback()) {
    return false;
  }

  double ref_endnorm[kTokens * kDim];
  double ref_finalpassa[kTokens];

  REFV3_TAIL_BOUNDARY_REF_COPY_TOKEN_LOOP: for (int token = 0; token < kTokens; ++token) {
    REFV3_TAIL_BOUNDARY_REF_COPY_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      const int idx = (token * kDim) + dim;
      ref_endnorm[idx] = static_cast<double>(ref_model.x_work(token, dim).to_float());
    }
    ref_finalpassa[token] = static_cast<double>(ref_model.final_scalar_buf(token).to_float());
  }

  compute_metrics(dut_endnorm, ref_endnorm, kTokens * kDim, &out->endnorm);
  compute_metrics(dut_finalpassa, ref_finalpassa, kTokens, &out->finalpassa);

  out->run_ok = out->endnorm.run_ok && out->finalpassa.run_ok;
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
  REFV3_TAIL_BOUNDARY_PATTERN_RANGE_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    if (pattern_idx < 0 || pattern_idx >= trace_batch) {
      std::printf("FAIL: pattern index out of range: %d (trace_batch=%d)\n", pattern_idx, trace_batch);
      return 1;
    }
  }

  int run_ok_count = 0;
  int run_fail_count = 0;

  double endnorm_max_abs_err_worst = 0.0;
  double endnorm_mean_abs_err_sum = 0.0;
  double endnorm_mse_sum = 0.0;
  double endnorm_cosine_sum = 0.0;
  int endnorm_valid_count = 0;

  double finalpassa_max_abs_err_worst = 0.0;
  double finalpassa_mean_abs_err_sum = 0.0;
  double finalpassa_mse_sum = 0.0;
  double finalpassa_cosine_sum = 0.0;
  int finalpassa_valid_count = 0;

  int nan_count_total = 0;

  REFV3_TAIL_BOUNDARY_CASE_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    CaseResult result;
    const bool case_exec_ok = run_case(pattern_idx, &result);

    if (!case_exec_ok || !result.run_ok) {
      ++run_fail_count;
    } else {
      ++run_ok_count;
    }

    const bool endnorm_ok = case_exec_ok && result.endnorm.run_ok;
    const bool finalpassa_ok = case_exec_ok && result.finalpassa.run_ok;

    std::printf(
      "[ref_v3_tail_boundary_compare] pattern_idx=%d stage_name=ENDNORM shape_ok=%d max_abs_err=%.9e mean_abs_err=%.9e mse=%.9e cosine_similarity=%.9e nan_count=%d result=%s\n",
      pattern_idx,
      case_exec_ok ? result.endnorm.shape_ok : 0,
      endnorm_ok ? result.endnorm.max_abs_err : std::numeric_limits<double>::quiet_NaN(),
      endnorm_ok ? result.endnorm.mean_abs_err : std::numeric_limits<double>::quiet_NaN(),
      endnorm_ok ? result.endnorm.mse : std::numeric_limits<double>::quiet_NaN(),
      endnorm_ok ? result.endnorm.cosine_similarity : std::numeric_limits<double>::quiet_NaN(),
      case_exec_ok ? result.endnorm.nan_count : -1,
      endnorm_ok ? "RUN_OK" : "RUN_FAIL");

    std::printf(
      "[ref_v3_tail_boundary_compare] pattern_idx=%d stage_name=FINALPASSA shape_ok=%d max_abs_err=%.9e mean_abs_err=%.9e mse=%.9e cosine_similarity=%.9e nan_count=%d result=%s\n",
      pattern_idx,
      case_exec_ok ? result.finalpassa.shape_ok : 0,
      finalpassa_ok ? result.finalpassa.max_abs_err : std::numeric_limits<double>::quiet_NaN(),
      finalpassa_ok ? result.finalpassa.mean_abs_err : std::numeric_limits<double>::quiet_NaN(),
      finalpassa_ok ? result.finalpassa.mse : std::numeric_limits<double>::quiet_NaN(),
      finalpassa_ok ? result.finalpassa.cosine_similarity : std::numeric_limits<double>::quiet_NaN(),
      case_exec_ok ? result.finalpassa.nan_count : -1,
      finalpassa_ok ? "RUN_OK" : "RUN_FAIL");

    if (endnorm_ok) {
      if (result.endnorm.max_abs_err > endnorm_max_abs_err_worst) {
        endnorm_max_abs_err_worst = result.endnorm.max_abs_err;
      }
      endnorm_mean_abs_err_sum += result.endnorm.mean_abs_err;
      endnorm_mse_sum += result.endnorm.mse;
      endnorm_cosine_sum += result.endnorm.cosine_similarity;
      ++endnorm_valid_count;
      nan_count_total += result.endnorm.nan_count;
    }

    if (finalpassa_ok) {
      if (result.finalpassa.max_abs_err > finalpassa_max_abs_err_worst) {
        finalpassa_max_abs_err_worst = result.finalpassa.max_abs_err;
      }
      finalpassa_mean_abs_err_sum += result.finalpassa.mean_abs_err;
      finalpassa_mse_sum += result.finalpassa.mse;
      finalpassa_cosine_sum += result.finalpassa.cosine_similarity;
      ++finalpassa_valid_count;
      nan_count_total += result.finalpassa.nan_count;
    }
  }

  const double endnorm_mean_abs_err_avg =
    (endnorm_valid_count > 0) ? (endnorm_mean_abs_err_sum / static_cast<double>(endnorm_valid_count)) :
                                 std::numeric_limits<double>::quiet_NaN();
  const double endnorm_mse_avg =
    (endnorm_valid_count > 0) ? (endnorm_mse_sum / static_cast<double>(endnorm_valid_count)) :
                                 std::numeric_limits<double>::quiet_NaN();
  const double endnorm_cosine_similarity_avg =
    (endnorm_valid_count > 0) ? (endnorm_cosine_sum / static_cast<double>(endnorm_valid_count)) :
                                 std::numeric_limits<double>::quiet_NaN();

  const double finalpassa_mean_abs_err_avg =
    (finalpassa_valid_count > 0)
      ? (finalpassa_mean_abs_err_sum / static_cast<double>(finalpassa_valid_count))
      : std::numeric_limits<double>::quiet_NaN();
  const double finalpassa_mse_avg =
    (finalpassa_valid_count > 0) ? (finalpassa_mse_sum / static_cast<double>(finalpassa_valid_count))
                                 : std::numeric_limits<double>::quiet_NaN();
  const double finalpassa_cosine_similarity_avg =
    (finalpassa_valid_count > 0)
      ? (finalpassa_cosine_sum / static_cast<double>(finalpassa_valid_count))
      : std::numeric_limits<double>::quiet_NaN();

  std::printf(
    "[ref_v3_tail_boundary_compare_summary] total_patterns=%d run_ok=%d run_fail=%d "
    "endnorm_max_abs_err_worst=%.9e endnorm_mean_abs_err_avg=%.9e endnorm_mse_avg=%.9e endnorm_cosine_similarity_avg=%.9e "
    "finalpassa_max_abs_err_worst=%.9e finalpassa_mean_abs_err_avg=%.9e finalpassa_mse_avg=%.9e finalpassa_cosine_similarity_avg=%.9e "
    "nan_count_total=%d\n",
    static_cast<int>(kPatternIndices.size()),
    run_ok_count,
    run_fail_count,
    endnorm_max_abs_err_worst,
    endnorm_mean_abs_err_avg,
    endnorm_mse_avg,
    endnorm_cosine_similarity_avg,
    finalpassa_max_abs_err_worst,
    finalpassa_mean_abs_err_avg,
    finalpassa_mse_avg,
    finalpassa_cosine_similarity_avg,
    nan_count_total);

  if (run_fail_count != 0) {
    std::printf("FAIL: tb_ref_v3_tail_boundary_compare\n");
    return 2;
  }

  std::printf("PASS: tb_ref_v3_tail_boundary_compare\n");
  return 0;
}
