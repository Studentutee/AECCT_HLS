#include "../../include/ref_v3/RefModel_v3.h"

#include <cmath>
#include <cstdio>

#include "weights.h"

#if !REFV3_ENABLE_COMPARE
#error "RefModel_v3 compare path disabled. Exclude this TU in REFV3_SYNTH_ONLY builds."
#endif

namespace aecct_ref {
namespace ref_v3 {
namespace {

static void reset_compare_point(RefV3ComparePoint* p) {
  p->mismatch_count = 0;
  p->max_abs_diff = 0.0;
  p->first_mismatch_token = -1;
  p->first_mismatch_dim = -1;
  p->first_v2_value = 0.0;
  p->first_ref_value = 0.0;
}

static void update_compare_point(
  RefV3ComparePoint* p,
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

static void print_compare_point(const char* name, const RefV3ComparePoint& p, double tol) {
  std::printf(
    "[ref_v3_compare] point=%s mismatch_count=%d max_abs_diff=%.9e tol=%.3e",
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

static void print_next_stage_handoff(const RefV3CompareStats& stats, double tol) {
  const RefV3ComparePoint& p = stats.next_stage_handoff;
  std::printf(
    "[ref_v3_compare] point=next_stage_handoff mismatch_count=%d max_abs_diff=%.9e tol=%.3e "
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

RefModel_v3::RefModel_v3()
  : phase_a_valid_(false),
    layer0_attention_valid_(false) {
  run_cfg_ = make_fp32_baseline_run_config();
  authoritative_model_.set_run_config(run_cfg_);
  clear_storage();
  reset_compare_stats();
}

void RefModel_v3::set_run_config(const RefRunConfig& cfg) {
  run_cfg_ = cfg;
  authoritative_model_.set_run_config(cfg);
}

RefRunConfig RefModel_v3::get_run_config() const {
  return run_cfg_;
}

bool RefModel_v3::stage_step0_phase_a_from_authoritative(const RefModelIO& io, int batch_index) {
  clear_storage();
  reset_compare_stats();

  if (!authoritative_model_.stage_step0_phase_a(io, batch_index)) {
    phase_a_valid_ = false;
    layer0_attention_valid_ = false;
    return false;
  }

  ac_channel<RefV3PreprocInputPayload> preproc_in_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> preproc_out_token_ch;
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

bool RefModel_v3::run_layer0_attention_channel_transport() {
  return run_attention_layer_shared(REFV3_LAYER0_ID);
}

bool RefModel_v3::run_attention_layer_shared(int lid) {
  if (!phase_a_valid_) {
    return false;
  }

  ac_channel<RefV3AttentionTokenVectorPayload> kv_in_token_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> query_token_ch;
  ac_channel<RefV3AttentionKPayload> kv_out_k_payload_ch;
  ac_channel<RefV3AttentionVPayload> kv_out_v_payload_ch;
  ac_channel<RefV3AttentionKPayload> qsoftres_in_k_payload_ch;
  ac_channel<RefV3AttentionVPayload> qsoftres_in_v_payload_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> qsoftres_out_token_ch;

  if (!stream_x_work_to_attention_channels(lid, kv_in_token_ch, query_token_ch)) {
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!kv_block_.run(lid, kv_in_token_ch, kv_out_k_payload_ch, kv_out_v_payload_ch)) {
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  const RefV3AttentionKPayload k_payload = kv_out_k_payload_ch.read();
  const RefV3AttentionVPayload v_payload = kv_out_v_payload_ch.read();
  if (lid == REFV3_LAYER0_ID) {
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
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!writeback_attention_output_stream_to_x_work(lid, qsoftres_out_token_ch)) {
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }

  if (lid == REFV3_LAYER0_ID) {
    layer0_attention_valid_ = true;
  }
  return true;
}

bool RefModel_v3::run_layer0_ln_channel_transport() {
  return run_ln_layer_shared(REFV3_LAYER0_ID);
}

bool RefModel_v3::run_layer0_ffn_channel_transport() {
  return run_ffn_layer_shared(REFV3_LAYER0_ID);
}

bool RefModel_v3::run_ln_layer_shared(int lid) {
  if (!phase_a_valid_) {
    return false;
  }
  if (lid == REFV3_LAYER0_ID && !layer0_attention_valid_) {
    return false;
  }

  ac_channel<RefV3AttentionTokenVectorPayload> ln_in_token_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> ln_out_token_ch;

  if (!stream_x_work_to_ln_channel(lid, ln_in_token_ch)) {
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!ln_block_.run(lid, run_cfg_, ln_in_token_ch, ln_out_token_ch)) {
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!writeback_ln_output_stream_to_x_work(lid, ln_out_token_ch)) {
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }

  return true;
}

bool RefModel_v3::run_ffn_layer_shared(int lid) {
  if (!phase_a_valid_) {
    return false;
  }
  if (lid == REFV3_LAYER0_ID && !layer0_attention_valid_) {
    return false;
  }

  ac_channel<RefV3AttentionTokenVectorPayload> ffn_linear0_in_token_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> ffn_residual_in_token_ch;
  ac_channel<RefV3FfnHiddenTokenPayload> ffn_hidden_token_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> ffn_out_token_ch;

  if (!stream_x_work_to_ffn_channels(lid, ffn_linear0_in_token_ch, ffn_residual_in_token_ch)) {
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!ffn_linear0_relu_block_.run(lid, ffn_linear0_in_token_ch, ffn_hidden_token_ch)) {
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!ffn_linear1_residual_block_.run(lid, ffn_hidden_token_ch, ffn_residual_in_token_ch, ffn_out_token_ch)) {
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }
  if (!writeback_ffn_output_stream_to_x_work(lid, ffn_out_token_ch)) {
    if (lid == REFV3_LAYER0_ID) {
      layer0_attention_valid_ = false;
    }
    return false;
  }

  return true;
}

bool RefModel_v3::run_transformer_layer_shared(int lid) {
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

bool RefModel_v3::run_step0_layer0_attention_compare(const RefModelIO& io, int batch_index) {
  if (!stage_step0_phase_a_from_authoritative(io, batch_index)) {
    return false;
  }

  REFV3_TRANSFORMER_LAYER_LOOP: for (int lid = 0; lid < 2; ++lid) {
    if (!run_transformer_layer_shared(lid)) {
      return false;
    }

    if (lid == REFV3_LAYER0_ID) {
      ac_channel<RefV3AttentionTokenVectorPayload> next_stage_token_ch;
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

  ac_channel<RefV3AttentionTokenVectorPayload> finala_in_token_ch;
  ac_channel<RefV3FinalScalarTokenPayload> finala_out_scalar_ch;
  ac_channel<RefV3FinalScalarTokenPayload> finalb_in_scalar_ch;
  ac_channel<RefV3FinalInputYPayload> finalb_in_input_y_ch;
  ac_channel<RefV3FinalOutputPayload> finalb_out_payload_ch;

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

bool RefModel_v3::phase_a_valid() const {
  return phase_a_valid_;
}

bool RefModel_v3::layer0_attention_valid() const {
  return layer0_attention_valid_;
}

refv3_fp_t RefModel_v3::x_work(int token, int dim) const {
  if (token < 0 || token >= REFV3_TOKENS_T || dim < 0 || dim >= REFV3_D_MODEL) {
    return refv3_fp_t(0.0f);
  }
  return x_work_[token][dim];
}

refv3_fp_t RefModel_v3::scr_k(int token, int dim) const {
  if (token < 0 || token >= REFV3_TOKENS_T || dim < 0 || dim >= REFV3_D_MODEL) {
    return refv3_fp_t(0.0f);
  }
  return scr_k_[token][dim];
}

refv3_fp_t RefModel_v3::scr_v(int token, int dim) const {
  if (token < 0 || token >= REFV3_TOKENS_T || dim < 0 || dim >= REFV3_D_MODEL) {
    return refv3_fp_t(0.0f);
  }
  return scr_v_[token][dim];
}

RefV3CompareStats RefModel_v3::last_compare_stats() const {
  return last_compare_stats_;
}

void RefModel_v3::clear_storage() {
  const refv3_fp_t zero(0.0f);

  REFV3_CLEAR_XWORK_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_CLEAR_XWORK_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
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

  REFV3_CLEAR_PREPROC_INPUT_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    last_preproc_input_payload_.input_y[n] = zero;
    final_logits_[n] = zero;
    final_x_pred_[n] = bit1_t(0);
    last_final_output_payload_.logits[n] = zero;
    last_final_output_payload_.x_pred[n] = bit1_t(0);
  }
  last_preproc_input_payload_.var_count = ac_int<16, false>(0);
  last_final_output_payload_.var_count = ac_int<16, false>(0);

  REFV3_CLEAR_PAYLOAD_LOOP: for (int i = 0; i < REFV3_ATTN_MATRIX_ELEMS; ++i) {
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

void RefModel_v3::reset_compare_stats() {
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

bool RefModel_v3::stream_input_to_preproc_channel(
  const RefModelIO& io,
  int batch_index,
  ac_channel<RefV3PreprocInputPayload>& preproc_in_ch) {
  if (io.input_y_fp32 == nullptr || io.N < REFV3_VAR_N || batch_index < 0 || batch_index >= io.B) {
    return false;
  }

  RefV3PreprocInputPayload payload;
  payload.var_count = ac_int<16, false>(REFV3_VAR_N);
  const int base = batch_index * io.N;
  REFV3_PREPROC_INPUT_COPY_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    payload.input_y[n] = refv3_fp_t(static_cast<float>(io.input_y_fp32[base + n]));
  }
  last_preproc_input_payload_ = payload;
  preproc_in_ch.write(payload);
  return true;
}

bool RefModel_v3::collect_preproc_output_stream_and_writeback(
  ac_channel<RefV3AttentionTokenVectorPayload>& preproc_out_token_ch) {
  bool token_seen[REFV3_TOKENS_T];
  REFV3_PREPROC_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_PREPROC_WRITEBACK_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3AttentionTokenVectorPayload token_payload = preproc_out_token_ch.read();
    if (!REFV3_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != REFV3_LAYER0_ID) {
      return false;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV3_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV3_PREPROC_WRITEBACK_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const refv3_fp_t value = token_payload.token_vec[dim];
      preproc_x_work_[token][dim] = value;
      x_work_[token][dim] = value;
    }
  }

  return true;
}

bool RefModel_v3::stream_x_work_to_attention_channels(
  int lid,
  ac_channel<RefV3AttentionTokenVectorPayload>& kv_in_token_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& query_token_ch) {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  RefV3AttentionPayloadHeader stream_header;
  stream_header.layer_id = ac_int<8, false>(lid);
  stream_header.token_rows = ac_int<16, false>(REFV3_TOKENS_T);
  stream_header.dim_cols = ac_int<16, false>(REFV3_D_MODEL);
  if (lid == REFV3_LAYER0_ID) {
    last_attention_input_payload_.header = stream_header;
  }

  // Top owns X_WORK. Token is row, d is column, row-major flatten keeps d as inner-most.
  REFV3_STREAM_XWORK_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    RefV3AttentionTokenVectorPayload token_payload;
    token_payload.header = stream_header;
    token_payload.token_row = ac_int<16, false>(token);

    REFV3_STREAM_XWORK_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const int idx = REFV3_flatten_row_major_index(token, dim);
      token_payload.token_vec[dim] = x_work_[token][dim];
      if (lid == REFV3_LAYER0_ID) {
        last_attention_input_payload_.x_flat[idx] = x_work_[token][dim];
      }
    }
    kv_in_token_ch.write(token_payload);
    query_token_ch.write(token_payload);
  }
  return true;
}

bool RefModel_v3::collect_kv_payload_to_scratch(const RefV3AttentionKPayload& k_payload,
                                                const RefV3AttentionVPayload& v_payload) {
  if (!REFV3_payload_header_matches_shape(k_payload.header) ||
      !REFV3_payload_header_matches_shape(v_payload.header)) {
    return false;
  }

  REFV3_COLLECT_KV_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_COLLECT_KV_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const int idx = REFV3_flatten_row_major_index(token, dim);
      scr_k_[token][dim] = k_payload.k_flat[idx];
      scr_v_[token][dim] = v_payload.v_flat[idx];
    }
  }
  return true;
}

bool RefModel_v3::writeback_attention_output_stream_to_x_work(
  int lid,
  ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  if (lid == REFV3_LAYER0_ID) {
    last_out_payload_.header.layer_id = ac_int<8, false>(REFV3_LAYER0_ID);
    last_out_payload_.header.token_rows = ac_int<16, false>(REFV3_TOKENS_T);
    last_out_payload_.header.dim_cols = ac_int<16, false>(REFV3_D_MODEL);
  }

  bool token_seen[REFV3_TOKENS_T];
  REFV3_WRITEBACK_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  // Top receives token-vector stream and performs the only X_WORK write-back ownership action.
  REFV3_WRITEBACK_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3AttentionTokenVectorPayload token_payload = out_token_ch.read();
    if (!REFV3_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV3_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV3_WRITEBACK_STREAM_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const int idx = REFV3_flatten_row_major_index(token, dim);
      if (lid == REFV3_LAYER0_ID) {
        last_out_payload_.out_flat[idx] = token_payload.token_vec[dim];
      }
      x_work_[token][dim] = token_payload.token_vec[dim];
      if (lid == REFV3_LAYER0_ID) {
        x_work_after_attention_[token][dim] = token_payload.token_vec[dim];
      }
    }
  }

  return true;
}

bool RefModel_v3::stream_x_work_to_ln_channel(
  int lid,
  ac_channel<RefV3AttentionTokenVectorPayload>& ln_in_token_ch) {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  REFV3_STREAM_LN_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    RefV3AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(lid);
    token_payload.header.token_rows = ac_int<16, false>(REFV3_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(REFV3_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV3_STREAM_LN_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = x_work_[token][dim];
    }
    ln_in_token_ch.write(token_payload);
  }
  return true;
}

bool RefModel_v3::writeback_ln_output_stream_to_x_work(
  int lid,
  ac_channel<RefV3AttentionTokenVectorPayload>& ln_out_token_ch) {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  bool token_seen[REFV3_TOKENS_T];
  REFV3_LN_WRITEBACK_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_LN_WRITEBACK_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3AttentionTokenVectorPayload token_payload = ln_out_token_ch.read();
    if (!REFV3_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV3_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV3_LN_WRITEBACK_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const refv3_fp_t value = token_payload.token_vec[dim];
      if (lid == REFV3_LAYER0_ID) {
        layer0_ln_out_[token][dim] = value;
      }
      x_work_[token][dim] = value;
      if (lid == REFV3_LAYER0_ID) {
        x_work_after_layer0_ln_[token][dim] = value;
      }
    }
  }

  return true;
}

bool RefModel_v3::stream_x_work_to_ffn_channels(
  int lid,
  ac_channel<RefV3AttentionTokenVectorPayload>& ffn_linear0_in_token_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& ffn_residual_in_token_ch) {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  REFV3_STREAM_FFN_DUAL_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    RefV3AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(lid);
    token_payload.header.token_rows = ac_int<16, false>(REFV3_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(REFV3_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV3_STREAM_FFN_DUAL_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = x_work_[token][dim];
    }
    ffn_linear0_in_token_ch.write(token_payload);
    ffn_residual_in_token_ch.write(token_payload);
  }
  return true;
}

bool RefModel_v3::writeback_ffn_output_stream_to_x_work(
  int lid,
  ac_channel<RefV3AttentionTokenVectorPayload>& ffn_out_token_ch) {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  bool token_seen[REFV3_TOKENS_T];
  REFV3_FFN_WRITEBACK_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_FFN_WRITEBACK_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3AttentionTokenVectorPayload token_payload = ffn_out_token_ch.read();
    if (!REFV3_payload_header_matches_shape(token_payload.header)) {
      return false;
    }
    if (token_payload.header.layer_id.to_int() != expected_layer_id) {
      return false;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV3_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV3_FFN_WRITEBACK_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const refv3_fp_t value = token_payload.token_vec[dim];
      if (lid == REFV3_LAYER0_ID) {
        layer0_ffn_out_[token][dim] = value;
      }
      x_work_[token][dim] = value;
      if (lid == REFV3_LAYER0_ID) {
        x_work_after_layer0_ffn_[token][dim] = value;
      }
    }
  }

  return true;
}

bool RefModel_v3::stream_x_work_to_next_stage(
  int lid,
  ac_channel<RefV3AttentionTokenVectorPayload>& next_stage_token_ch) {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  // Local-only handoff channel: demonstrate stage-to-stage token-vector transport after X_WORK overwrite.
  REFV3_NEXT_STAGE_STREAM_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    RefV3AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(lid);
    token_payload.header.token_rows = ac_int<16, false>(REFV3_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(REFV3_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV3_NEXT_STAGE_STREAM_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = x_work_[token][dim];
    }
    next_stage_token_ch.write(token_payload);
  }
  return true;
}

bool RefModel_v3::consume_and_check_next_stage_stream(
  int lid,
  ac_channel<RefV3AttentionTokenVectorPayload>& next_stage_token_ch) {
  if (lid != REFV3_LAYER0_ID && lid != REFV3_LAYER1_ID) {
    return false;
  }

  const int expected_layer_id = lid;
  const double tol = last_compare_stats_.tol;

  bool token_seen[REFV3_TOKENS_T];
  REFV3_NEXT_STAGE_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_NEXT_STAGE_CHECK_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3AttentionTokenVectorPayload token_payload = next_stage_token_ch.read();
    ++last_compare_stats_.next_stage_token_count;

    if (!REFV3_payload_header_matches_shape(token_payload.header) ||
        token_payload.header.layer_id.to_int() != expected_layer_id) {
      ++last_compare_stats_.next_stage_header_error_count;
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV3_TOKENS_T) {
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

    REFV3_NEXT_STAGE_CHECK_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
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

  REFV3_NEXT_STAGE_MISSING_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    if (!token_seen[token]) {
      ++last_compare_stats_.next_stage_missing_count;
    }
  }

  last_compare_stats_.next_stage_handoff_pass =
    (last_compare_stats_.next_stage_token_count == REFV3_TOKENS_T) &&
    (last_compare_stats_.next_stage_out_of_order_count == 0) &&
    (last_compare_stats_.next_stage_duplicate_count == 0) &&
    (last_compare_stats_.next_stage_missing_count == 0) &&
    (last_compare_stats_.next_stage_header_error_count == 0) &&
    (last_compare_stats_.next_stage_invalid_token_count == 0) &&
    (last_compare_stats_.next_stage_handoff.mismatch_count == 0);

  return true;
}

bool RefModel_v3::load_authoritative_end_norm_to_x_work() {
  REFV3_LOAD_END_NORM_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_LOAD_END_NORM_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      x_work_[token][dim] = authoritative_model_.x_work(token, dim);
    }
  }
  return true;
}

bool RefModel_v3::stream_x_work_to_final_pass_a_channel(
  ac_channel<RefV3AttentionTokenVectorPayload>& finala_in_token_ch) {
  REFV3_STREAM_FINALA_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    RefV3AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(REFV3_LAYER1_ID);
    token_payload.header.token_rows = ac_int<16, false>(REFV3_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(REFV3_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV3_STREAM_FINALA_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      token_payload.token_vec[dim] = x_work_[token][dim];
    }
    finala_in_token_ch.write(token_payload);
  }
  return true;
}

bool RefModel_v3::collect_final_pass_a_stream_and_forward(
  ac_channel<RefV3FinalScalarTokenPayload>& finala_out_scalar_ch,
  ac_channel<RefV3FinalScalarTokenPayload>& finalb_in_scalar_ch) {
  bool token_seen[REFV3_TOKENS_T];
  REFV3_COLLECT_FINALA_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_COLLECT_FINALA_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3FinalScalarTokenPayload scalar_payload = finala_out_scalar_ch.read();
    if (!REFV3_payload_header_matches_shape(scalar_payload.header)) {
      return false;
    }

    const int token = scalar_payload.token_row.to_int();
    if (token < 0 || token >= REFV3_TOKENS_T) {
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

bool RefModel_v3::stream_input_to_final_pass_b_channel(
  const RefModelIO& io,
  int batch_index,
  ac_channel<RefV3FinalInputYPayload>& finalb_in_input_y_ch) {
  if (io.input_y_fp32 == nullptr || io.N < REFV3_VAR_N || batch_index < 0 || batch_index >= io.B) {
    return false;
  }

  RefV3FinalInputYPayload payload;
  payload.var_count = ac_int<16, false>(REFV3_VAR_N);
  const int base = batch_index * io.N;
  REFV3_FINALB_INPUT_COPY_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    payload.input_y[n] = refv3_fp_t(static_cast<float>(io.input_y_fp32[base + n]));
  }
  finalb_in_input_y_ch.write(payload);
  return true;
}

bool RefModel_v3::collect_final_output_payload(
  ac_channel<RefV3FinalOutputPayload>& finalb_out_payload_ch) {
  last_final_output_payload_ = finalb_out_payload_ch.read();
  if (!REFV3_var_count_matches_shape(last_final_output_payload_.var_count)) {
    return false;
  }

  REFV3_FINALB_OUTPUT_COPY_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    final_logits_[n] = last_final_output_payload_.logits[n];
    final_x_pred_[n] = last_final_output_payload_.x_pred[n];
  }
  return true;
}

bool RefModel_v3::compare_against_authoritative_layer0() {
  const double tol = last_compare_stats_.tol;

  REFV3_COMPARE_PREPROC_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_COMPARE_PREPROC_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const double v2_v = static_cast<double>(preproc_x_work_[token][dim].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.x_work(token, dim).to_float());
      update_compare_point(&last_compare_stats_.preproc_output, token, dim, v2_v, ref_v, tol);
    }
  }

  REFV3_COMPARE_ATTN_INPUT_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_COMPARE_ATTN_INPUT_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const int idx = REFV3_flatten_row_major_index(token, dim);
      const double v2_v = static_cast<double>(last_attention_input_payload_.x_flat[idx].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.x_work(token, dim).to_float());
      update_compare_point(&last_compare_stats_.attention_input, token, dim, v2_v, ref_v, tol);
    }
  }

  REFV3_COMPARE_SCRK_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_COMPARE_SCRK_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const int idx = REFV3_flatten_row_major_index(token, dim);
      const double v2_v = static_cast<double>(last_k_payload_.k_flat[idx].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.scr_k(token, dim).to_float());
      update_compare_point(&last_compare_stats_.scr_k, token, dim, v2_v, ref_v, tol);
    }
  }

  REFV3_COMPARE_SCRV_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_COMPARE_SCRV_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const int idx = REFV3_flatten_row_major_index(token, dim);
      const double v2_v = static_cast<double>(last_v_payload_.v_flat[idx].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.scr_v(token, dim).to_float());
      update_compare_point(&last_compare_stats_.scr_v, token, dim, v2_v, ref_v, tol);
    }
  }

  if (!authoritative_model_.run_step0_layer0_attention_writeback()) {
    return false;
  }

  REFV3_COMPARE_XWORK_ATTN_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_COMPARE_XWORK_ATTN_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const double v2_v = static_cast<double>(x_work_after_attention_[token][dim].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.x_work(token, dim).to_float());
      update_compare_point(&last_compare_stats_.x_work_writeback, token, dim, v2_v, ref_v, tol);
    }
  }

  if (!authoritative_model_.run_step0_layer0_ln_writeback()) {
    return false;
  }

  REFV3_COMPARE_LN_OUTPUT_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_COMPARE_LN_OUTPUT_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
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

  REFV3_COMPARE_FFN_OUTPUT_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_COMPARE_FFN_OUTPUT_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
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

bool RefModel_v3::compare_final_against_authoritative(const RefModelIO& io, int batch_index) {
  const double tol = last_compare_stats_.tol;
  if (io.input_y_fp32 == nullptr || io.N < REFV3_VAR_N || batch_index < 0 || batch_index >= io.B) {
    return false;
  }

  const int input_base = batch_index * io.N;
  const refv3_fp_t zero(0.0f);

  REFV3_COMPARE_FINALA_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    const double v2_v = static_cast<double>(final_pass_a_observe_scalar_[token].to_float());
    const double ref_v = static_cast<double>(authoritative_model_.final_scalar_buf(token).to_float());
    update_compare_point(&last_compare_stats_.final_passA_output, token, 0, v2_v, ref_v, tol);
  }

  REFV3_COMPARE_FINALB_LOGITS_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    refv3_fp_t ref_acc(static_cast<float>(w_out_fc_bias[n]));
    REFV3_COMPARE_FINALB_TOKEN_REDUCE_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      const refv3_fp_t w_nt(static_cast<float>(w_out_fc_weight[n * REFV3_TOKENS_T + token]));
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

bool RefModel_v3::update_overall_match_status() {
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

} // namespace ref_v3
} // namespace aecct_ref
