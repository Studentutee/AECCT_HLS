#include <array>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#if defined(__SYNTHESIS__) || defined(REFV3_SYNTH_ONLY)
#error "tb_ref_v3_tail_hybrid_onecount_compare is host-only."
#endif

#define private public
#include "AECCT_ac_ref/include/RefModelOptimized.h"
#undef private

#include "AECCT_ac_ref/include/ref_v3/RefV3FinalPassABlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3FinalPassBBlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer0AttnLnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer0FfnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer1AttnLnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3Layer1FfnPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3LayerNormBlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3MidNormPath.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3PreprocBlock.h"
#include "AECCT_ac_ref/include/ref_v3/RefV3WeightsFp16LocalOnly.h"
#include "input_y_step0.h"
#include "output_x_pred_step0.h"
#include "weights.h"

namespace {

static constexpr int kVarN = aecct_ref::ref_v3::REFV3_VAR_N;
static constexpr int kTokens = aecct_ref::ref_v3::REFV3_TOKENS_T;
static constexpr int kDim = aecct_ref::ref_v3::REFV3_D_MODEL;
static constexpr std::array<int, 8> kPatternIndices = {77, 116, 132, 179, 217, 265, 312, 572};

enum SplicePointId {
  L1_FFN_OUT = 0,
  ENDNORM_OUT = 1,
  FINALPASSA_OUT = 2
};

enum HybridModeId {
  DUT_PREFIX_REF_SUFFIX = 0,
  REF_PREFIX_DUT_SUFFIX = 1
};

struct PrefixArtifacts {
  bool run_ok = false;
  bool l1_ffn_ok = false;
  bool endnorm_ok = false;
  bool finalpassa_ok = false;
  double l1_ffn_xwork[kTokens * kDim]{};
  double endnorm_xwork[kTokens * kDim]{};
  double finalpassa_scalar[kTokens]{};
};

struct ScoreboardResult {
  int pattern_idx = -1;
  SplicePointId splice_point = L1_FFN_OUT;
  HybridModeId mode = DUT_PREFIX_REF_SUFFIX;
  bool run_ok = false;

  int hybrid_one_count = 0;
  int hybrid_zero_count = 0;
  int hybrid_all_zero = 0;

  int trace_one_count = 0;
  int trace_zero_count = 0;
  int trace_all_zero = 0;
  int trace_non_binary_anomaly_count = 0;

  const char* winner_by_one_count = "TIE";
  int one_count_delta = 0;
  int match_trace_count = 0;
  int mismatch_trace_count = 0;
};

struct ComboSummary {
  int run_ok = 0;
  int run_fail = 0;
  int hybrid_better_count_by_onecount = 0;
  int trace_better_count_by_onecount = 0;
  int tie_count = 0;
  int hybrid_all_zero_count = 0;
  int sum_one_count = 0;
  int sum_zero_count = 0;
  int sum_match_trace_count = 0;
  int sum_mismatch_trace_count = 0;
};

static const char* splice_point_name(SplicePointId splice_point) {
  switch (splice_point) {
    case L1_FFN_OUT:
      return "L1_FFN_OUT";
    case ENDNORM_OUT:
      return "ENDNORM_OUT";
    case FINALPASSA_OUT:
      return "FINALPASSA_OUT";
    default:
      return "UNKNOWN";
  }
}

static const char* mode_name(HybridModeId mode) {
  switch (mode) {
    case DUT_PREFIX_REF_SUFFIX:
      return "DUT_PREFIX_REF_SUFFIX";
    case REF_PREFIX_DUT_SUFFIX:
      return "REF_PREFIX_DUT_SUFFIX";
    default:
      return "UNKNOWN";
  }
}

static bool load_pattern_input_row(int pattern_idx, double out_input_row[kVarN]) {
  if (out_input_row == nullptr) {
    return false;
  }
  const int base = pattern_idx * kVarN;
  REFV3_HYBRID_INPUT_ROW_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
    out_input_row[n] = trace_input_y_step0_tensor[base + n];
  }
  return true;
}

static bool build_preproc_input_payload(
  int pattern_idx,
  aecct_ref::ref_v3::RefV3PreprocInputPayload* payload_out) {
  if (payload_out == nullptr) {
    return false;
  }
  const int base = pattern_idx * kVarN;
  payload_out->var_count = ac_int<16, false>(kVarN);
  REFV3_HYBRID_PREPROC_INPUT_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
    payload_out->input_y[n] = aecct_ref::ref_v3::refv3_fp_t(
      static_cast<float>(trace_input_y_step0_tensor[base + n]));
  }
  return true;
}

static bool build_final_input_y_payload(
  int pattern_idx,
  aecct_ref::ref_v3::RefV3FinalInputYPayload* payload_out) {
  if (payload_out == nullptr) {
    return false;
  }
  const int base = pattern_idx * kVarN;
  payload_out->var_count = ac_int<16, false>(kVarN);
  REFV3_HYBRID_FINAL_INPUT_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
    payload_out->input_y[n] = aecct_ref::ref_v3::refv3_fp_t(
      static_cast<float>(trace_input_y_step0_tensor[base + n]));
  }
  return true;
}

static bool build_xwork_payload_from_matrix(
  int layer_id,
  const double* matrix,
  aecct_ref::ref_v3::RefV3AttentionInputPayload* payload_out) {
  if (matrix == nullptr || payload_out == nullptr) {
    return false;
  }

  payload_out->header.layer_id = ac_int<8, false>(layer_id);
  payload_out->header.token_rows = ac_int<16, false>(kTokens);
  payload_out->header.dim_cols = ac_int<16, false>(kDim);

  REFV3_HYBRID_XWORK_MATRIX_COPY_TOKEN_LOOP: for (int token = 0; token < kTokens; ++token) {
    REFV3_HYBRID_XWORK_MATRIX_COPY_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      const int idx = (token * kDim) + dim;
      payload_out->x_flat[idx] = aecct_ref::ref_v3::refv3_fp_t(static_cast<float>(matrix[idx]));
    }
  }
  return true;
}

static bool emit_token_stream_from_matrix(
  int layer_id,
  const double* matrix,
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload>& out_token_ch) {
  if (matrix == nullptr) {
    return false;
  }

  aecct_ref::ref_v3::RefV3AttentionPayloadHeader header;
  header.layer_id = ac_int<8, false>(layer_id);
  header.token_rows = ac_int<16, false>(kTokens);
  header.dim_cols = ac_int<16, false>(kDim);

  REFV3_HYBRID_EMIT_TOKEN_FROM_MATRIX_TOKEN_LOOP: for (int token = 0; token < kTokens; ++token) {
    aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload token_payload;
    token_payload.header = header;
    token_payload.token_row = ac_int<16, false>(token);
    REFV3_HYBRID_EMIT_TOKEN_FROM_MATRIX_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      const int idx = (token * kDim) + dim;
      token_payload.token_vec[dim] = aecct_ref::ref_v3::refv3_fp_t(static_cast<float>(matrix[idx]));
    }
    out_token_ch.write(token_payload);
  }
  return true;
}

static bool emit_finala_stream_from_scalar(
  int layer_id,
  const double* final_scalar,
  ac_channel<aecct_ref::ref_v3::RefV3FinalScalarTokenPayload>& out_scalar_ch) {
  if (final_scalar == nullptr) {
    return false;
  }

  aecct_ref::ref_v3::RefV3AttentionPayloadHeader header;
  header.layer_id = ac_int<8, false>(layer_id);
  header.token_rows = ac_int<16, false>(kTokens);
  header.dim_cols = ac_int<16, false>(kDim);

  REFV3_HYBRID_EMIT_SCALAR_STREAM_TOKEN_LOOP: for (int token = 0; token < kTokens; ++token) {
    aecct_ref::ref_v3::RefV3FinalScalarTokenPayload scalar_payload;
    scalar_payload.header = header;
    scalar_payload.token_row = ac_int<16, false>(token);
    scalar_payload.scalar = aecct_ref::ref_v3::refv3_fp_t(static_cast<float>(final_scalar[token]));
    out_scalar_ch.write(scalar_payload);
  }
  return true;
}

static bool copy_xwork_payload_to_matrix(
  const aecct_ref::ref_v3::RefV3AttentionInputPayload& payload,
  int expected_layer_id,
  double out_matrix[kTokens * kDim]) {
  if (!aecct_ref::ref_v3::REFV3_payload_header_matches_shape(payload.header)) {
    return false;
  }
  if (payload.header.layer_id.to_int() != expected_layer_id) {
    return false;
  }

  REFV3_HYBRID_COPY_XWORK_TOKEN_LOOP: for (int token = 0; token < kTokens; ++token) {
    REFV3_HYBRID_COPY_XWORK_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      const int idx = (token * kDim) + dim;
      out_matrix[idx] = static_cast<double>(payload.x_flat[idx].to_float());
    }
  }
  return true;
}

static bool copy_token_stream_to_matrix(
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload>& in_token_ch,
  int expected_layer_id,
  double out_matrix[kTokens * kDim]) {
  bool token_seen[kTokens];
  REFV3_HYBRID_TOKEN_MATRIX_SEEN_INIT_LOOP: for (int token = 0; token < kTokens; ++token) {
    token_seen[token] = false;
    REFV3_HYBRID_TOKEN_MATRIX_INIT_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      out_matrix[(token * kDim) + dim] = 0.0;
    }
  }

  REFV3_HYBRID_TOKEN_MATRIX_CAPTURE_LOOP: for (int rx = 0; rx < kTokens; ++rx) {
    const aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload payload = in_token_ch.read();
    if (!aecct_ref::ref_v3::REFV3_payload_header_matches_shape(payload.header)) {
      return false;
    }
    if (payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }

    const int token = payload.token_row.to_int();
    if (token < 0 || token >= kTokens) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV3_HYBRID_TOKEN_MATRIX_CAPTURE_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      out_matrix[(token * kDim) + dim] = static_cast<double>(payload.token_vec[dim].to_float());
    }
  }

  REFV3_HYBRID_TOKEN_MATRIX_SEEN_CHECK_LOOP: for (int token = 0; token < kTokens; ++token) {
    if (!token_seen[token]) {
      return false;
    }
  }
  return true;
}

static bool copy_finala_stream_to_vector(
  ac_channel<aecct_ref::ref_v3::RefV3FinalScalarTokenPayload>& in_scalar_ch,
  int expected_layer_id,
  double out_final_scalar[kTokens]) {
  bool token_seen[kTokens];
  REFV3_HYBRID_SCALAR_SEEN_INIT_LOOP: for (int token = 0; token < kTokens; ++token) {
    token_seen[token] = false;
    out_final_scalar[token] = 0.0;
  }

  REFV3_HYBRID_SCALAR_CAPTURE_LOOP: for (int rx = 0; rx < kTokens; ++rx) {
    const aecct_ref::ref_v3::RefV3FinalScalarTokenPayload payload = in_scalar_ch.read();
    if (!aecct_ref::ref_v3::REFV3_payload_header_matches_shape(payload.header)) {
      return false;
    }
    if (payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }

    const int token = payload.token_row.to_int();
    if (token < 0 || token >= kTokens) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;
    out_final_scalar[token] = static_cast<double>(payload.scalar.to_float());
  }

  REFV3_HYBRID_SCALAR_SEEN_CHECK_LOOP: for (int token = 0; token < kTokens; ++token) {
    if (!token_seen[token]) {
      return false;
    }
  }
  return true;
}

static bool run_dut_prefix_artifacts(int pattern_idx, PrefixArtifacts* out) {
  if (out == nullptr) {
    return false;
  }

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
  ac_channel<aecct_ref::ref_v3::RefV3AttentionInputPayload> ch_xwork1_to_l1_attn;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_l1_attn_to_ffn;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_l1_ffn_raw;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_l1_ffn_to_endnorm;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_endnorm_raw;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_endnorm_for_capture;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_endnorm_to_finala;
  ac_channel<aecct_ref::ref_v3::RefV3FinalScalarTokenPayload> ch_finala_out;

  aecct_ref::ref_v3::RefV3PreprocInputPayload input_payload;
  if (!build_preproc_input_payload(pattern_idx, &input_payload)) {
    return false;
  }
  ch_preproc_in.write(input_payload);

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

  const aecct_ref::ref_v3::RefV3AttentionInputPayload layer1_xwork_payload = ch_xwork1_side.read();
  ch_xwork1_to_l1_attn.write(layer1_xwork_payload);

  if (!layer1_attn_ln_path.run(run_cfg, ch_midnorm_to_l1_attn, ch_xwork1_to_l1_attn, ch_l1_attn_to_ffn)) {
    return false;
  }
  if (!layer1_ffn_path.run(ch_l1_attn_to_ffn, ch_l1_ffn_raw)) {
    return false;
  }
  if (!copy_token_stream_to_matrix(ch_l1_ffn_raw, aecct_ref::ref_v3::REFV3_LAYER1_ID, out->l1_ffn_xwork)) {
    return false;
  }
  out->l1_ffn_ok = true;

  // Host-only splice adapter: re-emit captured L1_FFN boundary into downstream tail.
  if (!emit_token_stream_from_matrix(
        aecct_ref::ref_v3::REFV3_LAYER1_ID,
        out->l1_ffn_xwork,
        ch_l1_ffn_to_endnorm)) {
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

  REFV3_HYBRID_DUT_ENDNORM_CAPTURE_LOOP: for (int token_rx = 0; token_rx < kTokens; ++token_rx) {
    const aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload payload = ch_endnorm_raw.read();
    ch_endnorm_for_capture.write(payload);
  }
  if (!copy_token_stream_to_matrix(
        ch_endnorm_for_capture,
        aecct_ref::ref_v3::REFV3_LAYER1_ID,
        out->endnorm_xwork)) {
    return false;
  }
  out->endnorm_ok = true;

  if (!emit_token_stream_from_matrix(
        aecct_ref::ref_v3::REFV3_LAYER1_ID,
        out->endnorm_xwork,
        ch_endnorm_to_finala)) {
    return false;
  }
  if (!final_pass_a_block.run(ch_endnorm_to_finala, ch_finala_out)) {
    return false;
  }
  if (!copy_finala_stream_to_vector(ch_finala_out, aecct_ref::ref_v3::REFV3_LAYER1_ID, out->finalpassa_scalar)) {
    return false;
  }
  out->finalpassa_ok = true;

  out->run_ok = true;
  return true;
}

static bool run_ref_prefix_artifacts(int pattern_idx, PrefixArtifacts* out) {
  if (out == nullptr) {
    return false;
  }

  aecct_ref::RefRunConfig run_cfg = aecct_ref::make_fp32_baseline_run_config();
  aecct_ref::RefModelOptimized ref_model;
  ref_model.set_run_config(run_cfg);

  aecct_ref::RefOptimizedNumericConfig numeric_cfg;
  numeric_cfg.float_mode = aecct_ref::REF_OPT_FLOAT32;
  ref_model.set_numeric_config(numeric_cfg);

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
      !ref_model.run_step0_layer1_ffn_writeback()) {
    return false;
  }

  REFV3_HYBRID_REF_L1_FFN_COPY_TOKEN_LOOP: for (int token = 0; token < kTokens; ++token) {
    REFV3_HYBRID_REF_L1_FFN_COPY_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      const int idx = (token * kDim) + dim;
      out->l1_ffn_xwork[idx] = static_cast<double>(ref_model.x_work(token, dim).to_float());
    }
  }
  out->l1_ffn_ok = true;

  if (!ref_model.run_step0_end_norm_writeback()) {
    return false;
  }

  REFV3_HYBRID_REF_ENDNORM_COPY_TOKEN_LOOP: for (int token = 0; token < kTokens; ++token) {
    REFV3_HYBRID_REF_ENDNORM_COPY_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      const int idx = (token * kDim) + dim;
      out->endnorm_xwork[idx] = static_cast<double>(ref_model.x_work(token, dim).to_float());
    }
  }
  out->endnorm_ok = true;

  if (!ref_model.run_step0_final_head_pass_a_writeback()) {
    return false;
  }
  REFV3_HYBRID_REF_FINALA_COPY_LOOP: for (int token = 0; token < kTokens; ++token) {
    out->finalpassa_scalar[token] = static_cast<double>(ref_model.final_scalar_buf(token).to_float());
  }
  out->finalpassa_ok = true;

  out->run_ok = true;
  return true;
}

static void run_reference_pass_b_from_scalar(
  const double final_scalar[kTokens],
  const double input_y_row[kVarN],
  aecct_ref::bit1_t out_xpred[kVarN]) {
  REFV3_HYBRID_REF_PASSB_LOGITS_LOOP: for (int n = 0; n < kVarN; ++n) {
    double acc = w_out_fc_bias[n];
    REFV3_HYBRID_REF_PASSB_TOKEN_ACC_LOOP: for (int token = 0; token < kTokens; ++token) {
      acc += w_out_fc_weight[(n * kTokens) + token] * final_scalar[token];
    }

    const double y_n = input_y_row[n];
    const bool y_is_zero = (y_n == 0.0);
    const bool y_is_negative = (!y_is_zero) && std::signbit(y_n);
    const bool acc_is_negative = std::signbit(acc);
    const bool pred_bit = y_is_zero ? false : (acc_is_negative ^ y_is_negative);
    out_xpred[n] = aecct_ref::bit1_t(pred_bit ? 1 : 0);
  }
}

static bool run_ref_suffix_from_l1_ffn(
  const double l1_ffn_xwork[kTokens * kDim],
  const double input_y_row[kVarN],
  aecct_ref::bit1_t out_xpred[kVarN]) {
  aecct_ref::RefRunConfig run_cfg = aecct_ref::make_fp32_baseline_run_config();
  aecct_ref::RefModelOptimized ref_model;
  ref_model.set_run_config(run_cfg);

  aecct_ref::RefOptimizedNumericConfig numeric_cfg;
  numeric_cfg.float_mode = aecct_ref::REF_OPT_FLOAT32;
  ref_model.set_numeric_config(numeric_cfg);

  aecct_ref::RefModelIO io{};
  io.input_y_fp32 = input_y_row;
  io.B = 1;
  io.N = kVarN;
  if (!ref_model.stage_step0_phase_a(io, 0)) {
    return false;
  }

  // Host-only splice adapter: overwrite reference X_WORK at L1_FFN boundary.
  REFV3_HYBRID_REF_SPLICE_L1_FFN_COPY_TOKEN_LOOP: for (int token = 0; token < kTokens; ++token) {
    REFV3_HYBRID_REF_SPLICE_L1_FFN_COPY_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      const int idx = (token * kDim) + dim;
      ref_model.storage_fp32_.x_work[token][dim] = ac_ieee_float<binary32>(
        static_cast<float>(l1_ffn_xwork[idx]));
    }
  }

  ref_model.phase_a_valid_ = true;
  ref_model.layer0_attn_writeback_valid_ = false;
  ref_model.layer0_ln_writeback_valid_ = false;
  ref_model.layer0_ffn_writeback_valid_ = false;
  ref_model.mid_norm_writeback_valid_ = false;
  ref_model.layer1_attn_input_handoff_valid_ = false;
  ref_model.layer1_attn_writeback_valid_ = false;
  ref_model.layer1_ln_writeback_valid_ = false;
  ref_model.layer1_ffn_writeback_valid_ = true;
  ref_model.end_norm_writeback_valid_ = false;
  ref_model.final_head_pass_a_writeback_valid_ = false;

  if (!ref_model.run_step0_end_norm_writeback() ||
      !ref_model.run_step0_final_head_pass_a_writeback()) {
    return false;
  }

  double final_scalar[kTokens];
  REFV3_HYBRID_REF_SPLICE_L1_FFN_FINALA_COPY_LOOP: for (int token = 0; token < kTokens; ++token) {
    final_scalar[token] = static_cast<double>(ref_model.final_scalar_buf(token).to_float());
  }

  run_reference_pass_b_from_scalar(final_scalar, input_y_row, out_xpred);
  return true;
}

static bool run_ref_suffix_from_endnorm(
  const double endnorm_xwork[kTokens * kDim],
  const double input_y_row[kVarN],
  aecct_ref::bit1_t out_xpred[kVarN]) {
  aecct_ref::RefRunConfig run_cfg = aecct_ref::make_fp32_baseline_run_config();
  aecct_ref::RefModelOptimized ref_model;
  ref_model.set_run_config(run_cfg);

  aecct_ref::RefOptimizedNumericConfig numeric_cfg;
  numeric_cfg.float_mode = aecct_ref::REF_OPT_FLOAT32;
  ref_model.set_numeric_config(numeric_cfg);

  aecct_ref::RefModelIO io{};
  io.input_y_fp32 = input_y_row;
  io.B = 1;
  io.N = kVarN;
  if (!ref_model.stage_step0_phase_a(io, 0)) {
    return false;
  }

  // Host-only splice adapter: overwrite reference X_WORK at endnorm boundary.
  REFV3_HYBRID_REF_SPLICE_ENDNORM_COPY_TOKEN_LOOP: for (int token = 0; token < kTokens; ++token) {
    REFV3_HYBRID_REF_SPLICE_ENDNORM_COPY_DIM_LOOP: for (int dim = 0; dim < kDim; ++dim) {
      const int idx = (token * kDim) + dim;
      ref_model.storage_fp32_.x_work[token][dim] = ac_ieee_float<binary32>(
        static_cast<float>(endnorm_xwork[idx]));
    }
  }

  ref_model.phase_a_valid_ = true;
  ref_model.layer0_attn_writeback_valid_ = false;
  ref_model.layer0_ln_writeback_valid_ = false;
  ref_model.layer0_ffn_writeback_valid_ = false;
  ref_model.mid_norm_writeback_valid_ = false;
  ref_model.layer1_attn_input_handoff_valid_ = false;
  ref_model.layer1_attn_writeback_valid_ = false;
  ref_model.layer1_ln_writeback_valid_ = false;
  ref_model.layer1_ffn_writeback_valid_ = true;
  ref_model.end_norm_writeback_valid_ = true;
  ref_model.final_head_pass_a_writeback_valid_ = false;

  if (!ref_model.run_step0_final_head_pass_a_writeback()) {
    return false;
  }

  double final_scalar[kTokens];
  REFV3_HYBRID_REF_SPLICE_ENDNORM_FINALA_COPY_LOOP: for (int token = 0; token < kTokens; ++token) {
    final_scalar[token] = static_cast<double>(ref_model.final_scalar_buf(token).to_float());
  }

  run_reference_pass_b_from_scalar(final_scalar, input_y_row, out_xpred);
  return true;
}

static bool run_ref_suffix_from_finalpassa(
  const double finalpassa_scalar[kTokens],
  const double input_y_row[kVarN],
  aecct_ref::bit1_t out_xpred[kVarN]) {
  run_reference_pass_b_from_scalar(finalpassa_scalar, input_y_row, out_xpred);
  return true;
}

static bool run_dut_suffix_from_l1_ffn(
  const double l1_ffn_xwork[kTokens * kDim],
  int pattern_idx,
  aecct_ref::bit1_t out_xpred[kVarN]) {
  aecct_ref::RefRunConfig run_cfg = aecct_ref::make_fp32_baseline_run_config();

  aecct_ref::ref_v3::RefV3LayerNormBlock end_norm_block;
  aecct_ref::ref_v3::RefV3FinalPassABlock final_pass_a_block;
  aecct_ref::ref_v3::RefV3FinalPassBBlock final_pass_b_block;

  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_l1_ffn_in;
  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_endnorm_to_finala;
  ac_channel<aecct_ref::ref_v3::RefV3FinalScalarTokenPayload> ch_finala_to_finalb;
  ac_channel<aecct_ref::ref_v3::RefV3FinalInputYPayload> ch_final_input_y;
  ac_channel<aecct_ref::ref_v3::RefV3FinalOutputPayload> ch_final_output;

  if (!emit_token_stream_from_matrix(
        aecct_ref::ref_v3::REFV3_LAYER1_ID,
        l1_ffn_xwork,
        ch_l1_ffn_in)) {
    return false;
  }

  const aecct_ref::ref_v3::RefV3TernaryLinearParams endnorm_params =
    aecct_ref::ref_v3::refv3_endnorm_params_fp_local_only();
  if (!end_norm_block.run_with_params(
        aecct_ref::ref_v3::REFV3_LAYER1_ID,
        endnorm_params,
        run_cfg,
        ch_l1_ffn_in,
        ch_endnorm_to_finala)) {
    return false;
  }
  if (!final_pass_a_block.run(ch_endnorm_to_finala, ch_finala_to_finalb)) {
    return false;
  }

  aecct_ref::ref_v3::RefV3FinalInputYPayload final_input_payload;
  if (!build_final_input_y_payload(pattern_idx, &final_input_payload)) {
    return false;
  }
  ch_final_input_y.write(final_input_payload);

  if (!final_pass_b_block.run(ch_finala_to_finalb, ch_final_input_y, ch_final_output)) {
    return false;
  }
  const aecct_ref::ref_v3::RefV3FinalOutputPayload out_payload = ch_final_output.read();
  if (!aecct_ref::ref_v3::REFV3_var_count_matches_shape(out_payload.var_count)) {
    return false;
  }

  REFV3_HYBRID_DUT_SUFFIX_L1_FFN_XPRED_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
    out_xpred[n] = out_payload.x_pred[n];
  }
  return true;
}

static bool run_dut_suffix_from_endnorm(
  const double endnorm_xwork[kTokens * kDim],
  int pattern_idx,
  aecct_ref::bit1_t out_xpred[kVarN]) {
  aecct_ref::ref_v3::RefV3FinalPassABlock final_pass_a_block;
  aecct_ref::ref_v3::RefV3FinalPassBBlock final_pass_b_block;

  ac_channel<aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload> ch_endnorm_in;
  ac_channel<aecct_ref::ref_v3::RefV3FinalScalarTokenPayload> ch_finala_to_finalb;
  ac_channel<aecct_ref::ref_v3::RefV3FinalInputYPayload> ch_final_input_y;
  ac_channel<aecct_ref::ref_v3::RefV3FinalOutputPayload> ch_final_output;

  if (!emit_token_stream_from_matrix(
        aecct_ref::ref_v3::REFV3_LAYER1_ID,
        endnorm_xwork,
        ch_endnorm_in)) {
    return false;
  }

  if (!final_pass_a_block.run(ch_endnorm_in, ch_finala_to_finalb)) {
    return false;
  }

  aecct_ref::ref_v3::RefV3FinalInputYPayload final_input_payload;
  if (!build_final_input_y_payload(pattern_idx, &final_input_payload)) {
    return false;
  }
  ch_final_input_y.write(final_input_payload);

  if (!final_pass_b_block.run(ch_finala_to_finalb, ch_final_input_y, ch_final_output)) {
    return false;
  }

  const aecct_ref::ref_v3::RefV3FinalOutputPayload out_payload = ch_final_output.read();
  if (!aecct_ref::ref_v3::REFV3_var_count_matches_shape(out_payload.var_count)) {
    return false;
  }

  REFV3_HYBRID_DUT_SUFFIX_ENDNORM_XPRED_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
    out_xpred[n] = out_payload.x_pred[n];
  }
  return true;
}

static bool run_dut_suffix_from_finalpassa(
  const double finalpassa_scalar[kTokens],
  int pattern_idx,
  aecct_ref::bit1_t out_xpred[kVarN]) {
  aecct_ref::ref_v3::RefV3FinalPassBBlock final_pass_b_block;

  ac_channel<aecct_ref::ref_v3::RefV3FinalScalarTokenPayload> ch_finala_in;
  ac_channel<aecct_ref::ref_v3::RefV3FinalInputYPayload> ch_final_input_y;
  ac_channel<aecct_ref::ref_v3::RefV3FinalOutputPayload> ch_final_output;

  if (!emit_finala_stream_from_scalar(
        aecct_ref::ref_v3::REFV3_LAYER1_ID,
        finalpassa_scalar,
        ch_finala_in)) {
    return false;
  }

  aecct_ref::ref_v3::RefV3FinalInputYPayload final_input_payload;
  if (!build_final_input_y_payload(pattern_idx, &final_input_payload)) {
    return false;
  }
  ch_final_input_y.write(final_input_payload);

  if (!final_pass_b_block.run(ch_finala_in, ch_final_input_y, ch_final_output)) {
    return false;
  }

  const aecct_ref::ref_v3::RefV3FinalOutputPayload out_payload = ch_final_output.read();
  if (!aecct_ref::ref_v3::REFV3_var_count_matches_shape(out_payload.var_count)) {
    return false;
  }

  REFV3_HYBRID_DUT_SUFFIX_FINALA_XPRED_COPY_LOOP: for (int n = 0; n < kVarN; ++n) {
    out_xpred[n] = out_payload.x_pred[n];
  }
  return true;
}

static void score_against_trace(
  int pattern_idx,
  const aecct_ref::bit1_t hybrid_xpred[kVarN],
  ScoreboardResult* out) {
  if (out == nullptr) {
    return;
  }

  const int base = pattern_idx * kVarN;
  REFV3_HYBRID_SCORE_COMPARE_LOOP: for (int n = 0; n < kVarN; ++n) {
    const int hybrid_bit = (hybrid_xpred[n].to_int() != 0) ? 1 : 0;
    out->hybrid_one_count += hybrid_bit;
    out->hybrid_zero_count += (hybrid_bit == 0) ? 1 : 0;

    const double trace_raw = trace_output_x_pred_step0_tensor[base + n];
    int trace_bit = 0;
    bool trace_is_binary = true;

    if (trace_raw == 0.0) {
      trace_bit = 0;
      out->trace_zero_count += 1;
    } else if (trace_raw == 1.0) {
      trace_bit = 1;
      out->trace_one_count += 1;
    } else {
      trace_is_binary = false;
      ++out->trace_non_binary_anomaly_count;
      ++out->mismatch_trace_count;
    }

    if (trace_is_binary) {
      if (trace_bit == hybrid_bit) {
        ++out->match_trace_count;
      } else {
        ++out->mismatch_trace_count;
      }
    }
  }

  out->hybrid_all_zero = (out->hybrid_one_count == 0) ? 1 : 0;
  out->trace_all_zero = (out->trace_one_count == 0) ? 1 : 0;
  out->one_count_delta = out->trace_one_count - out->hybrid_one_count;

  if (out->hybrid_one_count < out->trace_one_count) {
    out->winner_by_one_count = "HYBRID";
  } else if (out->hybrid_one_count > out->trace_one_count) {
    out->winner_by_one_count = "TRACE";
  } else {
    out->winner_by_one_count = "TIE";
  }
}

static bool run_hybrid_case(
  int pattern_idx,
  SplicePointId splice_point,
  HybridModeId mode,
  const PrefixArtifacts& dut_prefix,
  const PrefixArtifacts& ref_prefix,
  ScoreboardResult* out) {
  if (out == nullptr) {
    return false;
  }

  out->pattern_idx = pattern_idx;
  out->splice_point = splice_point;
  out->mode = mode;

  aecct_ref::bit1_t hybrid_xpred[kVarN];
  REFV3_HYBRID_XPRED_INIT_LOOP: for (int n = 0; n < kVarN; ++n) {
    hybrid_xpred[n] = aecct_ref::bit1_t(0);
  }

  double input_y_row[kVarN];
  if (!load_pattern_input_row(pattern_idx, input_y_row)) {
    out->run_ok = false;
    return true;
  }

  bool exec_ok = false;
  if (splice_point == L1_FFN_OUT) {
    if (mode == DUT_PREFIX_REF_SUFFIX) {
      if (!dut_prefix.l1_ffn_ok) {
        out->run_ok = false;
        return true;
      }
      exec_ok = run_ref_suffix_from_l1_ffn(dut_prefix.l1_ffn_xwork, input_y_row, hybrid_xpred);
    } else {
      if (!ref_prefix.l1_ffn_ok) {
        out->run_ok = false;
        return true;
      }
      exec_ok = run_dut_suffix_from_l1_ffn(ref_prefix.l1_ffn_xwork, pattern_idx, hybrid_xpred);
    }
  } else if (splice_point == ENDNORM_OUT) {
    if (mode == DUT_PREFIX_REF_SUFFIX) {
      if (!dut_prefix.endnorm_ok) {
        out->run_ok = false;
        return true;
      }
      exec_ok = run_ref_suffix_from_endnorm(dut_prefix.endnorm_xwork, input_y_row, hybrid_xpred);
    } else {
      if (!ref_prefix.endnorm_ok) {
        out->run_ok = false;
        return true;
      }
      exec_ok = run_dut_suffix_from_endnorm(ref_prefix.endnorm_xwork, pattern_idx, hybrid_xpred);
    }
  } else if (splice_point == FINALPASSA_OUT) {
    if (mode == DUT_PREFIX_REF_SUFFIX) {
      if (!dut_prefix.finalpassa_ok) {
        out->run_ok = false;
        return true;
      }
      exec_ok = run_ref_suffix_from_finalpassa(dut_prefix.finalpassa_scalar, input_y_row, hybrid_xpred);
    } else {
      if (!ref_prefix.finalpassa_ok) {
        out->run_ok = false;
        return true;
      }
      exec_ok = run_dut_suffix_from_finalpassa(ref_prefix.finalpassa_scalar, pattern_idx, hybrid_xpred);
    }
  }

  if (!exec_ok) {
    out->run_ok = false;
    return true;
  }

  score_against_trace(pattern_idx, hybrid_xpred, out);
  out->run_ok = true;
  return true;
}

static void print_case_line(const ScoreboardResult& result) {
  std::printf(
    "[ref_v3_tail_hybrid_onecount_compare] pattern_idx=%d splice_point=%s mode=%s "
    "hybrid_one_count=%d hybrid_zero_count=%d hybrid_all_zero=%d trace_one_count=%d trace_zero_count=%d "
    "trace_all_zero=%d winner_by_one_count=%s one_count_delta=%d match_trace_count=%d mismatch_trace_count=%d "
    "trace_non_binary_anomaly_count=%d result=%s\n",
    result.pattern_idx,
    splice_point_name(result.splice_point),
    mode_name(result.mode),
    result.run_ok ? result.hybrid_one_count : -1,
    result.run_ok ? result.hybrid_zero_count : -1,
    result.run_ok ? result.hybrid_all_zero : 0,
    result.run_ok ? result.trace_one_count : -1,
    result.run_ok ? result.trace_zero_count : -1,
    result.run_ok ? result.trace_all_zero : 0,
    result.run_ok ? result.winner_by_one_count : "NA",
    result.run_ok ? result.one_count_delta : 0,
    result.run_ok ? result.match_trace_count : -1,
    result.run_ok ? result.mismatch_trace_count : -1,
    result.run_ok ? result.trace_non_binary_anomaly_count : -1,
    result.run_ok ? "RUN_OK" : "RUN_FAIL");
}

static void update_combo_summary(const ScoreboardResult& result, ComboSummary* summary) {
  if (summary == nullptr) {
    return;
  }

  if (!result.run_ok) {
    ++summary->run_fail;
    return;
  }

  ++summary->run_ok;
  summary->sum_one_count += result.hybrid_one_count;
  summary->sum_zero_count += result.hybrid_zero_count;
  summary->sum_match_trace_count += result.match_trace_count;
  summary->sum_mismatch_trace_count += result.mismatch_trace_count;

  if (result.hybrid_all_zero == 1) {
    ++summary->hybrid_all_zero_count;
  }

  if (result.winner_by_one_count[0] == 'H') {
    ++summary->hybrid_better_count_by_onecount;
  } else if (result.winner_by_one_count[0] == 'T' && result.winner_by_one_count[1] == 'R') {
    ++summary->trace_better_count_by_onecount;
  } else {
    ++summary->tie_count;
  }
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
    std::printf("FAIL: trace batch mismatch\n");
    return 1;
  }

  const int trace_batch = trace_input_y_step0_tensor_shape[0];
  REFV3_HYBRID_PATTERN_RANGE_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    if (pattern_idx < 0 || pattern_idx >= trace_batch) {
      std::printf("FAIL: pattern index out of range: %d (trace_batch=%d)\n", pattern_idx, trace_batch);
      return 1;
    }
  }

  ComboSummary combo_summary[3][2];
  int run_ok = 0;
  int run_fail = 0;

  REFV3_HYBRID_PATTERN_LOOP: for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];

    PrefixArtifacts dut_prefix;
    PrefixArtifacts ref_prefix;

    const bool dut_prefix_ok = run_dut_prefix_artifacts(pattern_idx, &dut_prefix);
    const bool ref_prefix_ok = run_ref_prefix_artifacts(pattern_idx, &ref_prefix);

  REFV3_HYBRID_SPLICE_LOOP: for (int splice = static_cast<int>(L1_FFN_OUT);
         splice <= static_cast<int>(FINALPASSA_OUT);
         ++splice) {
      REFV3_HYBRID_MODE_LOOP: for (int mode = static_cast<int>(DUT_PREFIX_REF_SUFFIX);
           mode <= static_cast<int>(REF_PREFIX_DUT_SUFFIX);
           ++mode) {
        ScoreboardResult result;
        result.pattern_idx = pattern_idx;
        result.splice_point = static_cast<SplicePointId>(splice);
        result.mode = static_cast<HybridModeId>(mode);

        bool case_exec_ok = false;
        if (dut_prefix_ok && ref_prefix_ok && dut_prefix.run_ok && ref_prefix.run_ok) {
          case_exec_ok = run_hybrid_case(
            pattern_idx,
            static_cast<SplicePointId>(splice),
            static_cast<HybridModeId>(mode),
            dut_prefix,
            ref_prefix,
            &result);
        }

        if (!case_exec_ok || !result.run_ok) {
          result.run_ok = false;
          ++run_fail;
        } else {
          ++run_ok;
        }

        print_case_line(result);
        update_combo_summary(result, &combo_summary[splice][mode]);
      }
    }
  }

  REFV3_HYBRID_SUMMARY_SPLICE_LOOP: for (int splice = static_cast<int>(L1_FFN_OUT);
       splice <= static_cast<int>(FINALPASSA_OUT);
       ++splice) {
    REFV3_HYBRID_SUMMARY_MODE_LOOP: for (int mode = static_cast<int>(DUT_PREFIX_REF_SUFFIX);
         mode <= static_cast<int>(REF_PREFIX_DUT_SUFFIX);
         ++mode) {
      const ComboSummary& summary = combo_summary[splice][mode];
      const double avg_one_count =
        (summary.run_ok > 0) ? (static_cast<double>(summary.sum_one_count) / static_cast<double>(summary.run_ok)) : -1.0;
      const double avg_zero_count =
        (summary.run_ok > 0) ? (static_cast<double>(summary.sum_zero_count) / static_cast<double>(summary.run_ok)) : -1.0;
      const double avg_match_trace_count =
        (summary.run_ok > 0)
          ? (static_cast<double>(summary.sum_match_trace_count) / static_cast<double>(summary.run_ok))
          : -1.0;
      const double avg_mismatch_trace_count =
        (summary.run_ok > 0)
          ? (static_cast<double>(summary.sum_mismatch_trace_count) / static_cast<double>(summary.run_ok))
          : -1.0;

      std::printf(
        "[ref_v3_tail_hybrid_onecount_compare_summary] splice_point=%s mode=%s "
        "hybrid_better_count_by_onecount=%d trace_better_count_by_onecount=%d tie_count=%d hybrid_all_zero_count=%d "
        "avg_one_count=%.6f avg_zero_count=%.6f avg_match_trace_count=%.6f avg_mismatch_trace_count=%.6f "
        "run_ok=%d run_fail=%d\n",
        splice_point_name(static_cast<SplicePointId>(splice)),
        mode_name(static_cast<HybridModeId>(mode)),
        summary.hybrid_better_count_by_onecount,
        summary.trace_better_count_by_onecount,
        summary.tie_count,
        summary.hybrid_all_zero_count,
        avg_one_count,
        avg_zero_count,
        avg_match_trace_count,
        avg_mismatch_trace_count,
        summary.run_ok,
        summary.run_fail);
    }
  }

  std::printf(
    "[ref_v3_tail_hybrid_onecount_compare_total] total_patterns=%d total_cases=%d run_ok=%d run_fail=%d\n",
    static_cast<int>(kPatternIndices.size()),
    static_cast<int>(kPatternIndices.size()) * 6,
    run_ok,
    run_fail);

  if (run_fail != 0) {
    std::printf("FAIL: tb_ref_v3_tail_hybrid_onecount_compare\n");
    return 2;
  }

  std::printf("PASS: tb_ref_v3_tail_hybrid_onecount_compare\n");
  return 0;
}
