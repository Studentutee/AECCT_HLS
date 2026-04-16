#include "../../include/ref_v2/RefModel_v2.h"

#include <cmath>
#include <cstdio>

#include "weights.h"

namespace aecct_ref {
namespace ref_v2 {
namespace {

static void reset_compare_point(RefV2ComparePoint* p) {
  p->mismatch_count = 0;
  p->max_abs_diff = 0.0;
  p->first_mismatch_token = -1;
  p->first_mismatch_dim = -1;
  p->first_v2_value = 0.0;
  p->first_ref_value = 0.0;
}

static void update_compare_point(
  RefV2ComparePoint* p,
  int token,
  int dim,
  double v2_v,
  double ref_v,
  double tol) {
  const double abs_diff = std::fabs(v2_v - ref_v);
  if (abs_diff > p->max_abs_diff) {
    p->max_abs_diff = abs_diff;
  }
  if (abs_diff > tol) {
    ++p->mismatch_count;
    if (p->first_mismatch_token < 0) {
      p->first_mismatch_token = token;
      p->first_mismatch_dim = dim;
      p->first_v2_value = v2_v;
      p->first_ref_value = ref_v;
    }
  }
}

static void print_compare_point(const char* name, const RefV2ComparePoint& p, double tol) {
  std::printf(
    "[ref_v2_compare] point=%s mismatch_count=%d max_abs_diff=%.9e tol=%.3e",
    name,
    p.mismatch_count,
    p.max_abs_diff,
    tol);
  if (p.first_mismatch_token >= 0) {
    std::printf(
      " first_mismatch={token=%d,dim=%d,v2=%.9e,ref=%.9e,abs_diff=%.9e}\n",
      p.first_mismatch_token,
      p.first_mismatch_dim,
      p.first_v2_value,
      p.first_ref_value,
      std::fabs(p.first_v2_value - p.first_ref_value));
  } else {
    std::printf(" first_mismatch={none}\n");
  }
}

static void print_next_stage_handoff(const RefV2CompareStats& stats, double tol) {
  const RefV2ComparePoint& p = stats.next_stage_handoff;
  std::printf(
    "[ref_v2_compare] point=next_stage_handoff mismatch_count=%d max_abs_diff=%.9e tol=%.3e "
    "token_count=%d out_of_order=%d duplicate=%d missing=%d header_error=%d invalid_token=%d pass=%d",
    p.mismatch_count,
    p.max_abs_diff,
    tol,
    stats.next_stage_token_count,
    stats.next_stage_out_of_order_count,
    stats.next_stage_duplicate_count,
    stats.next_stage_missing_count,
    stats.next_stage_header_error_count,
    stats.next_stage_invalid_token_count,
    stats.next_stage_handoff_pass ? 1 : 0);
  if (p.first_mismatch_token >= 0) {
    std::printf(
      " first_mismatch={token=%d,dim=%d,v2=%.9e,ref=%.9e,abs_diff=%.9e}\n",
      p.first_mismatch_token,
      p.first_mismatch_dim,
      p.first_v2_value,
      p.first_ref_value,
      std::fabs(p.first_v2_value - p.first_ref_value));
  } else {
    std::printf(" first_mismatch={none}\n");
  }
}

} // namespace

RefModel_v2::RefModel_v2()
  : phase_a_valid_(false),
    layer0_attention_valid_(false) {
  run_cfg_ = make_fp32_baseline_run_config();
  authoritative_model_.set_run_config(run_cfg_);
  clear_storage();
  reset_compare_stats();
}

void RefModel_v2::set_run_config(const RefRunConfig& cfg) {
  run_cfg_ = cfg;
  authoritative_model_.set_run_config(cfg);
}

RefRunConfig RefModel_v2::get_run_config() const {
  return run_cfg_;
}

bool RefModel_v2::stage_step0_phase_a_from_authoritative(const RefModelIO& io, int batch_index) {
  clear_storage();
  reset_compare_stats();

  if (!authoritative_model_.stage_step0_phase_a(io, batch_index)) {
    phase_a_valid_ = false;
    layer0_attention_valid_ = false;
    return false;
  }

  ac_channel<RefV2PreprocInputPayload> preproc_in_ch;
  ac_channel<RefV2AttentionTokenVectorPayload> preproc_out_token_ch;
  if (!stream_input_to_preproc_channel(io, batch_index, preproc_in_ch)) {
    phase_a_valid_ = false;
    layer0_attention_valid_ = false;
    return false;
  }
  if (!preproc_block_.run(preproc_in_ch, preproc_out_token_ch)) {
    phase_a_valid_ = false;
    layer0_attention_valid_ = false;
    return false;
  }
  if (!collect_preproc_output_stream_and_writeback(preproc_out_token_ch)) {
    phase_a_valid_ = false;
    layer0_attention_valid_ = false;
    return false;
  }

  phase_a_valid_ = true;
  layer0_attention_valid_ = false;
  return true;
}

bool RefModel_v2::run_layer0_attention_channel_transport() {
  return run_attention_layer_shared(REFV2_LAYER0_ID);
}

bool RefModel_v2::run_attention_layer_shared(int lid) {
  if (!phase_a_valid_) {
    return false;
  }

  ac_channel<RefV2AttentionTokenVectorPayload> kv_in_token_ch;
  ac_channel<RefV2AttentionTokenVectorPayload> query_token_ch;
  ac_channel<RefV2AttentionKPayload> kv_out_k_payload_ch;
  ac_channel<RefV2AttentionVPayload> kv_out_v_payload_ch;
  ac_channel<RefV2AttentionKPayload> qsoftres_in_k_payload_ch;
  ac_channel<RefV2AttentionVPayload> qsoftres_in_v_payload_ch;
  ac_channel<RefV2AttentionTokenVectorPayload> qsoftres_out_token_ch;

  if (!stream_x_work_to_attention_channels(lid, kv_in_token_ch, query_token_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!kv_block_.run(lid, kv_in_token_ch, kv_out_k_payload_ch, kv_out_v_payload_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  const RefV2AttentionKPayload k_payload = kv_out_k_payload_ch.read();
  const RefV2AttentionVPayload v_payload = kv_out_v_payload_ch.read();
  if (lid == REFV2_LAYER0_ID) {
    last_k_payload_ = k_payload;
    last_v_payload_ = v_payload;
    if (!collect_kv_payload_to_scratch(last_k_payload_, last_v_payload_)) {
      layer0_attention_valid_ = false;
      return false;
    }
  }
  qsoftres_in_k_payload_ch.write(k_payload);
  qsoftres_in_v_payload_ch.write(v_payload);
  if (!qsoftres_block_.run(
        lid,
        run_cfg_,
        query_token_ch,
        qsoftres_in_k_payload_ch,
        qsoftres_in_v_payload_ch,
        qsoftres_out_token_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!writeback_attention_output_stream_to_x_work(lid, qsoftres_out_token_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }

  if (lid == REFV2_LAYER0_ID) {
    layer0_attention_valid_ = true;
  }
  return true;
}

bool RefModel_v2::run_layer0_ln_channel_transport() {
  return run_ln_layer_shared(REFV2_LAYER0_ID);
}

bool RefModel_v2::run_layer0_ffn_channel_transport() {
  return run_ffn_layer_shared(REFV2_LAYER0_ID);
}

bool RefModel_v2::run_ln_layer_shared(int lid) {
  if (!phase_a_valid_) {
    return false;
  }
  if (lid == REFV2_LAYER0_ID && !layer0_attention_valid_) {
    return false;
  }

  ac_channel<RefV2AttentionTokenVectorPayload> ln_in_token_ch;
  ac_channel<RefV2AttentionTokenVectorPayload> ln_out_token_ch;

  if (!stream_x_work_to_ln_channel(lid, ln_in_token_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!ln_block_.run(lid, run_cfg_, ln_in_token_ch, ln_out_token_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!writeback_ln_output_stream_to_x_work(lid, ln_out_token_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }

  return true;
}

bool RefModel_v2::run_ffn_layer_shared(int lid) {
  if (!phase_a_valid_) {
    return false;
  }
  if (lid == REFV2_LAYER0_ID && !layer0_attention_valid_) {
    return false;
  }

  ac_channel<RefV2AttentionTokenVectorPayload> ffn_linear0_in_token_ch;
  ac_channel<RefV2AttentionTokenVectorPayload> ffn_residual_in_token_ch;
  ac_channel<RefV2FfnHiddenTokenPayload> ffn_hidden_token_ch;
  ac_channel<RefV2AttentionTokenVectorPayload> ffn_out_token_ch;

  if (!stream_x_work_to_ffn_channels(lid, ffn_linear0_in_token_ch, ffn_residual_in_token_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!ffn_linear0_relu_block_.run(lid, ffn_linear0_in_token_ch, ffn_hidden_token_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!ffn_linear1_residual_block_.run(lid, ffn_hidden_token_ch, ffn_residual_in_token_ch, ffn_out_token_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!writeback_ffn_output_stream_to_x_work(lid, ffn_out_token_ch)) {
    if (lid == REFV2_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }

  return true;
}

bool RefModel_v2::run_transformer_layer_shared(int lid) {
  if (!run_attention_layer_shared(lid)) {
    return false;
  }
  if (!run_ln_layer_shared(lid)) {
    return false;
  }
  if (!run_ffn_layer_shared(lid)) {
    return false;
  }
  return true;
}

bool RefModel_v2::run_step0_layer0_attention_compare(const RefModelIO& io, int batch_index) {
  if (!stage_step0_phase_a_from_authoritative(io, batch_index)) {
    return false;
  }

  REFV2_TRANSFORMER_LAYER_LOOP: for (int lid = 0; lid < 2; ++lid) {
    if (!run_transformer_layer_shared(lid)) {
      return false;
    }

    if (lid == REFV2_LAYER0_ID) {
      ac_channel<RefV2AttentionTokenVectorPayload> next_stage_token_ch;
      if (!stream_x_work_to_next_stage(lid, next_stage_token_ch)) {
        return false;
      }
      if (!consume_and_check_next_stage_stream(lid, next_stage_token_ch)) {
        return false;
      }
    }
  }

  if (!compare_against_authoritative_layer0()) {
    return false;
  }

  if (!authoritative_model_.run_step0_mid_norm_writeback()) return false;
  if (!authoritative_model_.run_step0_layer1_attn_input_handoff()) return false;
  if (!authoritative_model_.run_step0_layer1_attention_writeback()) return false;
  if (!authoritative_model_.run_step0_layer1_ln_writeback()) return false;
  if (!authoritative_model_.run_step0_layer1_ffn_writeback()) return false;
  if (!authoritative_model_.run_step0_end_norm_writeback()) return false;
  if (!authoritative_model_.run_step0_final_head_pass_a_writeback()) return false;

  if (!load_authoritative_end_norm_to_x_work()) {
    return false;
  }

  ac_channel<RefV2AttentionTokenVectorPayload> finala_in_token_ch;
  ac_channel<RefV2FinalScalarTokenPayload> finala_out_scalar_ch;
  ac_channel<RefV2FinalScalarTokenPayload> finalb_in_scalar_ch;
  ac_channel<RefV2FinalInputYPayload> finalb_in_input_y_ch;
  ac_channel<RefV2FinalOutputPayload> finalb_out_payload_ch;

  if (!stream_x_work_to_final_pass_a_channel(finala_in_token_ch)) {
    return false;
  }
  if (!final_pass_a_block_.run(finala_in_token_ch, finala_out_scalar_ch)) {
    return false;
  }
  if (!collect_final_pass_a_stream_and_forward(finala_out_scalar_ch, finalb_in_scalar_ch)) {
    return false;
  }
  if (!stream_input_to_final_pass_b_channel(io, batch_index, finalb_in_input_y_ch)) {
    return false;
  }
  if (!final_pass_b_block_.run(finalb_in_scalar_ch, finalb_in_input_y_ch, finalb_out_payload_ch)) {
    return false;
  }
  if (!collect_final_output_payload(finalb_out_payload_ch)) {
    return false;
  }
  if (!compare_final_against_authoritative(io, batch_index)) {
    return false;
  }

  return update_overall_match_status();
}

bool RefModel_v2::phase_a_valid() const {
  return phase_a_valid_;
}

bool RefModel_v2::layer0_attention_valid() const {
  return layer0_attention_valid_;
}

ref_fp32_t RefModel_v2::x_work(int token, int dim) const {
  if (token < 0 || token >= REFV2_TOKENS_T || dim < 0 || dim >= REFV2_D_MODEL) {
    return ref_fp32_t(0.0f);
  }
  return x_work_[token][dim];
}

ref_fp32_t RefModel_v2::scr_k(int token, int dim) const {
  if (token < 0 || token >= REFV2_TOKENS_T || dim < 0 || dim >= REFV2_D_MODEL) {
    return ref_fp32_t(0.0f);
  }
  return scr_k_[token][dim];
}

ref_fp32_t RefModel_v2::scr_v(int token, int dim) const {
  if (token < 0 || token >= REFV2_TOKENS_T || dim < 0 || dim >= REFV2_D_MODEL) {
    return ref_fp32_t(0.0f);
  }
  return scr_v_[token][dim];
}

RefV2CompareStats RefModel_v2::last_compare_stats() const {
  return last_compare_stats_;
}

void RefModel_v2::clear_storage() {
  const ref_fp32_t zero(0.0f);

  REFV2_CLEAR_XWORK_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_CLEAR_XWORK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      x_work_[token][dim] = zero;
      preproc_x_work_[token][dim] = zero;
      scr_k_[token][dim] = zero;
      scr_v_[token][dim] = zero;
      x_work_after_attention_[token][dim] = zero;
      layer0_ln_out_[token][dim] = zero;
      x_work_after_layer0_ln_[token][dim] = zero;
      layer0_ffn_out_[token][dim] = zero;
      x_work_after_layer0_ffn_[token][dim] = zero;
    }
    final_pass_a_observe_scalar_[token] = zero;
  }

  REFV2_CLEAR_PREPROC_INPUT_LOOP: for (int n = 0; n < REFV2_VAR_N; ++n) {
    last_preproc_input_payload_.input_y[n] = zero;
    final_logits_[n] = zero;
    final_x_pred_[n] = bit1_t(0);
    last_final_output_payload_.logits[n] = zero;
    last_final_output_payload_.x_pred[n] = bit1_t(0);
  }
  last_preproc_input_payload_.var_count = ac_int<16, false>(0);
  last_final_output_payload_.var_count = ac_int<16, false>(0);

  REFV2_CLEAR_PAYLOAD_LOOP: for (int i = 0; i < REFV2_ATTN_MATRIX_ELEMS; ++i) {
    last_attention_input_payload_.x_flat[i] = zero;
    last_k_payload_.k_flat[i] = zero;
    last_v_payload_.v_flat[i] = zero;
    last_out_payload_.out_flat[i] = zero;
  }
  last_attention_input_payload_.header.layer_id = 0;
  last_attention_input_payload_.header.token_rows = 0;
  last_attention_input_payload_.header.dim_cols = 0;
  last_k_payload_.header.layer_id = 0;
  last_k_payload_.header.token_rows = 0;
  last_k_payload_.header.dim_cols = 0;
  last_v_payload_.header.layer_id = 0;
  last_v_payload_.header.token_rows = 0;
  last_v_payload_.header.dim_cols = 0;
  last_out_payload_.header.layer_id = 0;
  last_out_payload_.header.token_rows = 0;
  last_out_payload_.header.dim_cols = 0;
}

void RefModel_v2::reset_compare_stats() {
  reset_compare_point(&last_compare_stats_.preproc_output);
  reset_compare_point(&last_compare_stats_.attention_input);
  reset_compare_point(&last_compare_stats_.scr_k);
  reset_compare_point(&last_compare_stats_.scr_v);
  reset_compare_point(&last_compare_stats_.x_work_writeback);
  reset_compare_point(&last_compare_stats_.layer0_ln_output);
  reset_compare_point(&last_compare_stats_.x_work_after_layer0_ln);
  reset_compare_point(&last_compare_stats_.layer0_ffn_output);
  reset_compare_point(&last_compare_stats_.x_work_after_layer0_ffn);
  reset_compare_point(&last_compare_stats_.next_stage_handoff);
  reset_compare_point(&last_compare_stats_.final_passA_output);
  reset_compare_point(&last_compare_stats_.final_logits);
  reset_compare_point(&last_compare_stats_.final_x_pred);
  last_compare_stats_.next_stage_token_count = 0;
  last_compare_stats_.next_stage_out_of_order_count = 0;
  last_compare_stats_.next_stage_duplicate_count = 0;
  last_compare_stats_.next_stage_missing_count = 0;
  last_compare_stats_.next_stage_header_error_count = 0;
  last_compare_stats_.next_stage_invalid_token_count = 0;
  last_compare_stats_.next_stage_handoff_pass = false;
  last_compare_stats_.tol = 1.0e-6;
  last_compare_stats_.all_match = false;
}

bool RefModel_v2::stream_input_to_preproc_channel(
  const RefModelIO& io,
  int batch_index,
  ac_channel<RefV2PreprocInputPayload>& preproc_in_ch) {
  if (io.input_y_fp32 == nullptr || io.N < REFV2_VAR_N || batch_index < 0 || batch_index >= io.B) {
    return false;
  }

  RefV2PreprocInputPayload payload;
  payload.var_count = ac_int<16, false>(REFV2_VAR_N);
  const int base = batch_index * io.N;
  REFV2_PREPROC_INPUT_COPY_LOOP: for (int n = 0; n < REFV2_VAR_N; ++n) {
    payload.input_y[n] = ref_fp32_t(static_cast<float>(io.input_y_fp32[base + n]));
  }
  last_preproc_input_payload_ = payload;
  preproc_in_ch.write(payload);
  return true;
}

bool RefModel_v2::collect_preproc_output_stream_and_writeback(
  ac_channel<RefV2AttentionTokenVectorPayload>& preproc_out_token_ch) {
  bool token_seen[REFV2_TOKENS_T];
  REFV2_PREPROC_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV2_PREPROC_WRITEBACK_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2AttentionTokenVectorPayload token_payload = preproc_out_token_ch.read();
    if (!refv2_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != REFV2_LAYER0_ID) {
      return false;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV2_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV2_PREPROC_WRITEBACK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const ref_fp32_t value = token_payload.token_vec[dim];
      preproc_x_work_[token][dim] = value;
      x_work_[token][dim] = value;
    }
  }

  return true;
}

bool RefModel_v2::stream_x_work_to_attention_channels(
  int lid,
  ac_channel<RefV2AttentionTokenVectorPayload>& kv_in_token_ch,
  ac_channel<RefV2AttentionTokenVectorPayload>& query_token_ch) {
  if (lid != REFV2_LAYER0_ID && lid != REFV2_LAYER1_ID) {
    return false;
  }

  RefV2AttentionPayloadHeader stream_header;
  stream_header.layer_id = ac_int<8, false>(lid);
  stream_header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
  stream_header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);
  if (lid == REFV2_LAYER0_ID) {
    last_attention_input_payload_.header = stream_header;
  }

  // Top owns X_WORK. Token is row, d is column, row-major flatten keeps d as inner-most.
  REFV2_STREAM_XWORK_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    RefV2AttentionTokenVectorPayload token_payload;
    token_payload.header = stream_header;
    token_payload.token_row = ac_int<16, false>(token);

    REFV2_STREAM_XWORK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      token_payload.token_vec[dim] = x_work_[token][dim];
      if (lid == REFV2_LAYER0_ID) {
        last_attention_input_payload_.x_flat[idx] = x_work_[token][dim];
      }
    }
    kv_in_token_ch.write(token_payload);
    query_token_ch.write(token_payload);
  }
  return true;
}

bool RefModel_v2::collect_kv_payload_to_scratch(const RefV2AttentionKPayload& k_payload,
                                                const RefV2AttentionVPayload& v_payload) {
  if (!refv2_payload_header_matches_shape(k_payload.header) ||
      !refv2_payload_header_matches_shape(v_payload.header)) {
    return false;
  }

  REFV2_COLLECT_KV_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COLLECT_KV_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      scr_k_[token][dim] = k_payload.k_flat[idx];
      scr_v_[token][dim] = v_payload.v_flat[idx];
    }
  }
  return true;
}

bool RefModel_v2::writeback_attention_output_stream_to_x_work(
  int lid,
  ac_channel<RefV2AttentionTokenVectorPayload>& out_token_ch) {
  if (lid != REFV2_LAYER0_ID && lid != REFV2_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  if (lid == REFV2_LAYER0_ID) {
    last_out_payload_.header.layer_id = ac_int<8, false>(REFV2_LAYER0_ID);
    last_out_payload_.header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
    last_out_payload_.header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);
  }

  bool token_seen[REFV2_TOKENS_T];
  REFV2_WRITEBACK_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  // Top receives token-vector stream and performs the only X_WORK write-back ownership action.
  REFV2_WRITEBACK_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2AttentionTokenVectorPayload token_payload = out_token_ch.read();
    if (!refv2_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV2_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV2_WRITEBACK_STREAM_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      if (lid == REFV2_LAYER0_ID) {
        last_out_payload_.out_flat[idx] = token_payload.token_vec[dim];
      }
      x_work_[token][dim] = token_payload.token_vec[dim];
      if (lid == REFV2_LAYER0_ID) {
        x_work_after_attention_[token][dim] = token_payload.token_vec[dim];
      }
    }
  }

  return true;
}

bool RefModel_v2::stream_x_work_to_ln_channel(
  int lid,
  ac_channel<RefV2AttentionTokenVectorPayload>& ln_in_token_ch) {
  if (lid != REFV2_LAYER0_ID && lid != REFV2_LAYER1_ID) {
    return false;
  }

  REFV2_STREAM_LN_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    RefV2AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(lid);
    token_payload.header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV2_STREAM_LN_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = x_work_[token][dim];
    }
    ln_in_token_ch.write(token_payload);
  }
  return true;
}

bool RefModel_v2::writeback_ln_output_stream_to_x_work(
  int lid,
  ac_channel<RefV2AttentionTokenVectorPayload>& ln_out_token_ch) {
  if (lid != REFV2_LAYER0_ID && lid != REFV2_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  bool token_seen[REFV2_TOKENS_T];
  REFV2_LN_WRITEBACK_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV2_LN_WRITEBACK_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2AttentionTokenVectorPayload token_payload = ln_out_token_ch.read();
    if (!refv2_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV2_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV2_LN_WRITEBACK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const ref_fp32_t value = token_payload.token_vec[dim];
      if (lid == REFV2_LAYER0_ID) {
        layer0_ln_out_[token][dim] = value;
      }
      x_work_[token][dim] = value;
      if (lid == REFV2_LAYER0_ID) {
        x_work_after_layer0_ln_[token][dim] = value;
      }
    }
  }

  return true;
}

bool RefModel_v2::stream_x_work_to_ffn_channels(
  int lid,
  ac_channel<RefV2AttentionTokenVectorPayload>& ffn_linear0_in_token_ch,
  ac_channel<RefV2AttentionTokenVectorPayload>& ffn_residual_in_token_ch) {
  if (lid != REFV2_LAYER0_ID && lid != REFV2_LAYER1_ID) {
    return false;
  }

  REFV2_STREAM_FFN_DUAL_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    RefV2AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(lid);
    token_payload.header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV2_STREAM_FFN_DUAL_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = x_work_[token][dim];
    }
    ffn_linear0_in_token_ch.write(token_payload);
    ffn_residual_in_token_ch.write(token_payload);
  }
  return true;
}

bool RefModel_v2::writeback_ffn_output_stream_to_x_work(
  int lid,
  ac_channel<RefV2AttentionTokenVectorPayload>& ffn_out_token_ch) {
  if (lid != REFV2_LAYER0_ID && lid != REFV2_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  bool token_seen[REFV2_TOKENS_T];
  REFV2_FFN_WRITEBACK_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV2_FFN_WRITEBACK_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2AttentionTokenVectorPayload token_payload = ffn_out_token_ch.read();
    if (!refv2_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV2_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV2_FFN_WRITEBACK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const ref_fp32_t value = token_payload.token_vec[dim];
      if (lid == REFV2_LAYER0_ID) {
        layer0_ffn_out_[token][dim] = value;
      }
      x_work_[token][dim] = value;
      if (lid == REFV2_LAYER0_ID) {
        x_work_after_layer0_ffn_[token][dim] = value;
      }
    }
  }

  return true;
}

bool RefModel_v2::stream_x_work_to_next_stage(
  int lid,
  ac_channel<RefV2AttentionTokenVectorPayload>& next_stage_token_ch) {
  if (lid != REFV2_LAYER0_ID && lid != REFV2_LAYER1_ID) {
    return false;
  }

  // Local-only handoff channel: demonstrate stage-to-stage token-vector transport after X_WORK overwrite.
  REFV2_NEXT_STAGE_STREAM_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    RefV2AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(lid);
    token_payload.header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV2_NEXT_STAGE_STREAM_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = x_work_[token][dim];
    }
    next_stage_token_ch.write(token_payload);
  }
  return true;
}

bool RefModel_v2::consume_and_check_next_stage_stream(
  int lid,
  ac_channel<RefV2AttentionTokenVectorPayload>& next_stage_token_ch) {
  if (lid != REFV2_LAYER0_ID && lid != REFV2_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  const double tol = last_compare_stats_.tol;

  bool token_seen[REFV2_TOKENS_T];
  REFV2_NEXT_STAGE_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV2_NEXT_STAGE_CHECK_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2AttentionTokenVectorPayload token_payload = next_stage_token_ch.read();
    ++last_compare_stats_.next_stage_token_count;

    if (!refv2_payload_header_matches_shape(token_payload.header) ||
        token_payload.header.layer_id.to_int() != expected_layer_id) {
      ++last_compare_stats_.next_stage_header_error_count;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV2_TOKENS_T) {
      ++last_compare_stats_.next_stage_invalid_token_count;
      continue;
    }
    if (token != token_rx) {
      ++last_compare_stats_.next_stage_out_of_order_count;
    }
    if (token_seen[token]) {
      ++last_compare_stats_.next_stage_duplicate_count;
    } else {
      token_seen[token] = true;
    }

    REFV2_NEXT_STAGE_CHECK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const double stream_v = static_cast<double>(token_payload.token_vec[dim].to_float());
      const double x_work_v = static_cast<double>(x_work_[token][dim].to_float());
      update_compare_point(
        &last_compare_stats_.next_stage_handoff,
        token,
        dim,
        stream_v,
        x_work_v,
        tol);
    }
  }

  REFV2_NEXT_STAGE_MISSING_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    if (!token_seen[token]) {
      ++last_compare_stats_.next_stage_missing_count;
    }
  }

  last_compare_stats_.next_stage_handoff_pass =
    (last_compare_stats_.next_stage_token_count == REFV2_TOKENS_T) &&
    (last_compare_stats_.next_stage_out_of_order_count == 0) &&
    (last_compare_stats_.next_stage_duplicate_count == 0) &&
    (last_compare_stats_.next_stage_missing_count == 0) &&
    (last_compare_stats_.next_stage_header_error_count == 0) &&
    (last_compare_stats_.next_stage_invalid_token_count == 0) &&
    (last_compare_stats_.next_stage_handoff.mismatch_count == 0);

  return true;
}

bool RefModel_v2::load_authoritative_end_norm_to_x_work() {
  REFV2_LOAD_END_NORM_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_LOAD_END_NORM_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      x_work_[token][dim] = authoritative_model_.x_work(token, dim);
    }
  }
  return true;
}

bool RefModel_v2::stream_x_work_to_final_pass_a_channel(
  ac_channel<RefV2AttentionTokenVectorPayload>& finala_in_token_ch) {
  REFV2_STREAM_FINALA_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    RefV2AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(REFV2_LAYER1_ID);
    token_payload.header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV2_STREAM_FINALA_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = x_work_[token][dim];
    }
    finala_in_token_ch.write(token_payload);
  }
  return true;
}

bool RefModel_v2::collect_final_pass_a_stream_and_forward(
  ac_channel<RefV2FinalScalarTokenPayload>& finala_out_scalar_ch,
  ac_channel<RefV2FinalScalarTokenPayload>& finalb_in_scalar_ch) {
  bool token_seen[REFV2_TOKENS_T];
  REFV2_COLLECT_FINALA_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV2_COLLECT_FINALA_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2FinalScalarTokenPayload scalar_payload = finala_out_scalar_ch.read();
    if (!refv2_payload_header_matches_shape(scalar_payload.header)) {
      return false;
    }

    const int token = scalar_payload.token_row.to_int();
    if (token < 0 || token >= REFV2_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    // Local-only observe buffer for compare/debug; functional path is direct stream forwarding.
    final_pass_a_observe_scalar_[token] = scalar_payload.scalar;
    finalb_in_scalar_ch.write(scalar_payload);
  }

  return true;
}

bool RefModel_v2::stream_input_to_final_pass_b_channel(
  const RefModelIO& io,
  int batch_index,
  ac_channel<RefV2FinalInputYPayload>& finalb_in_input_y_ch) {
  if (io.input_y_fp32 == nullptr || io.N < REFV2_VAR_N || batch_index < 0 || batch_index >= io.B) {
    return false;
  }

  RefV2FinalInputYPayload payload;
  payload.var_count = ac_int<16, false>(REFV2_VAR_N);
  const int base = batch_index * io.N;
  REFV2_FINALB_INPUT_COPY_LOOP: for (int n = 0; n < REFV2_VAR_N; ++n) {
    payload.input_y[n] = ref_fp32_t(static_cast<float>(io.input_y_fp32[base + n]));
  }
  finalb_in_input_y_ch.write(payload);
  return true;
}

bool RefModel_v2::collect_final_output_payload(
  ac_channel<RefV2FinalOutputPayload>& finalb_out_payload_ch) {
  last_final_output_payload_ = finalb_out_payload_ch.read();
  if (!refv2_var_count_matches_shape(last_final_output_payload_.var_count)) {
    return false;
  }

  REFV2_FINALB_OUTPUT_COPY_LOOP: for (int n = 0; n < REFV2_VAR_N; ++n) {
    final_logits_[n] = last_final_output_payload_.logits[n];
    final_x_pred_[n] = last_final_output_payload_.x_pred[n];
  }
  return true;
}

bool RefModel_v2::compare_against_authoritative_layer0() {
  const double tol = last_compare_stats_.tol;

  REFV2_COMPARE_PREPROC_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_PREPROC_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const double v2_v = static_cast<double>(preproc_x_work_[token][dim].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.x_work(token, dim).to_float());
      update_compare_point(&last_compare_stats_.preproc_output, token, dim, v2_v, ref_v, tol);
    }
  }

  REFV2_COMPARE_ATTN_INPUT_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_ATTN_INPUT_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      const double v2_v = static_cast<double>(last_attention_input_payload_.x_flat[idx].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.x_work(token, dim).to_float());
      update_compare_point(&last_compare_stats_.attention_input, token, dim, v2_v, ref_v, tol);
    }
  }

  REFV2_COMPARE_SCRK_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_SCRK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      const double v2_v = static_cast<double>(last_k_payload_.k_flat[idx].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.scr_k(token, dim).to_float());
      update_compare_point(&last_compare_stats_.scr_k, token, dim, v2_v, ref_v, tol);
    }
  }

  REFV2_COMPARE_SCRV_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_SCRV_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      const double v2_v = static_cast<double>(last_v_payload_.v_flat[idx].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.scr_v(token, dim).to_float());
      update_compare_point(&last_compare_stats_.scr_v, token, dim, v2_v, ref_v, tol);
    }
  }

  if (!authoritative_model_.run_step0_layer0_attention_writeback()) {
    return false;
  }

  REFV2_COMPARE_XWORK_ATTN_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_XWORK_ATTN_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const double v2_v = static_cast<double>(x_work_after_attention_[token][dim].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.x_work(token, dim).to_float());
      update_compare_point(&last_compare_stats_.x_work_writeback, token, dim, v2_v, ref_v, tol);
    }
  }

  if (!authoritative_model_.run_step0_layer0_ln_writeback()) {
    return false;
  }

  REFV2_COMPARE_LN_OUTPUT_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_LN_OUTPUT_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const double v2_ln_out = static_cast<double>(layer0_ln_out_[token][dim].to_float());
      const double v2_x_work = static_cast<double>(x_work_after_layer0_ln_[token][dim].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.x_work(token, dim).to_float());
      update_compare_point(&last_compare_stats_.layer0_ln_output, token, dim, v2_ln_out, ref_v, tol);
      update_compare_point(&last_compare_stats_.x_work_after_layer0_ln, token, dim, v2_x_work, ref_v, tol);
    }
  }

  if (!authoritative_model_.run_step0_layer0_ffn_writeback()) {
    return false;
  }

  REFV2_COMPARE_FFN_OUTPUT_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_FFN_OUTPUT_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const double v2_ffn_out = static_cast<double>(layer0_ffn_out_[token][dim].to_float());
      const double v2_x_work = static_cast<double>(x_work_after_layer0_ffn_[token][dim].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.x_work(token, dim).to_float());
      update_compare_point(&last_compare_stats_.layer0_ffn_output, token, dim, v2_ffn_out, ref_v, tol);
      update_compare_point(&last_compare_stats_.x_work_after_layer0_ffn, token, dim, v2_x_work, ref_v, tol);
    }
  }

  print_compare_point("preproc_output", last_compare_stats_.preproc_output, tol);
  print_compare_point("attention_input", last_compare_stats_.attention_input, tol);
  print_compare_point("SCR_K", last_compare_stats_.scr_k, tol);
  print_compare_point("SCR_V", last_compare_stats_.scr_v, tol);
  print_compare_point("x_work_writeback", last_compare_stats_.x_work_writeback, tol);
  print_compare_point("layer0_ln_output", last_compare_stats_.layer0_ln_output, tol);
  print_compare_point("x_work_after_layer0_ln", last_compare_stats_.x_work_after_layer0_ln, tol);
  print_compare_point("layer0_ffn_output", last_compare_stats_.layer0_ffn_output, tol);
  print_compare_point("x_work_after_layer0_ffn", last_compare_stats_.x_work_after_layer0_ffn, tol);
  print_next_stage_handoff(last_compare_stats_, tol);

  return true;
}

bool RefModel_v2::compare_final_against_authoritative(const RefModelIO& io, int batch_index) {
  const double tol = last_compare_stats_.tol;
  if (io.input_y_fp32 == nullptr || io.N < REFV2_VAR_N || batch_index < 0 || batch_index >= io.B) {
    return false;
  }

  const int input_base = batch_index * io.N;
  const ref_fp32_t zero(0.0f);

  REFV2_COMPARE_FINALA_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    const double v2_v = static_cast<double>(final_pass_a_observe_scalar_[token].to_float());
    const double ref_v = static_cast<double>(authoritative_model_.final_scalar_buf(token).to_float());
    update_compare_point(&last_compare_stats_.final_passA_output, token, 0, v2_v, ref_v, tol);
  }

  REFV2_COMPARE_FINALB_LOGITS_LOOP: for (int n = 0; n < REFV2_VAR_N; ++n) {
    ref_fp32_t ref_acc(static_cast<float>(w_out_fc_bias[n]));
    REFV2_COMPARE_FINALB_TOKEN_REDUCE_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
      const ref_fp32_t w_nt(static_cast<float>(w_out_fc_weight[n * REFV2_TOKENS_T + token]));
      ref_acc += (w_nt * authoritative_model_.final_scalar_buf(token));
    }

    const double v2_logit = static_cast<double>(final_logits_[n].to_float());
    const double ref_logit = static_cast<double>(ref_acc.to_float());
    update_compare_point(&last_compare_stats_.final_logits, n, 0, v2_logit, ref_logit, tol);

    const double y_n = io.input_y_fp32[input_base + n];
    const bool y_is_zero = (y_n == 0.0);
    const bool y_is_negative = (!y_is_zero) && std::signbit(y_n);
    const bool acc_is_negative = (ref_acc < zero);
    const int ref_bit = y_is_zero ? 0 : ((acc_is_negative ^ y_is_negative) ? 1 : 0);
    const int v2_bit = final_x_pred_[n].to_int();
    update_compare_point(
      &last_compare_stats_.final_x_pred,
      n,
      0,
      static_cast<double>(v2_bit),
      static_cast<double>(ref_bit),
      0.0);
  }

  print_compare_point("final_passA_output", last_compare_stats_.final_passA_output, tol);
  print_compare_point("final_logits", last_compare_stats_.final_logits, tol);
  print_compare_point("final_x_pred", last_compare_stats_.final_x_pred, 0.0);

  return true;
}

bool RefModel_v2::update_overall_match_status() {
  last_compare_stats_.all_match =
    (last_compare_stats_.preproc_output.mismatch_count == 0) &&
    (last_compare_stats_.attention_input.mismatch_count == 0) &&
    (last_compare_stats_.scr_k.mismatch_count == 0) &&
    (last_compare_stats_.scr_v.mismatch_count == 0) &&
    (last_compare_stats_.x_work_writeback.mismatch_count == 0) &&
    (last_compare_stats_.layer0_ln_output.mismatch_count == 0) &&
    (last_compare_stats_.x_work_after_layer0_ln.mismatch_count == 0) &&
    (last_compare_stats_.layer0_ffn_output.mismatch_count == 0) &&
    (last_compare_stats_.x_work_after_layer0_ffn.mismatch_count == 0) &&
    last_compare_stats_.next_stage_handoff_pass &&
    (last_compare_stats_.final_passA_output.mismatch_count == 0) &&
    (last_compare_stats_.final_logits.mismatch_count == 0) &&
    (last_compare_stats_.final_x_pred.mismatch_count == 0);
  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
