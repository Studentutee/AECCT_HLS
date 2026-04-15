#include "../../include/ref_v2/RefModel_v2.h"

#include <cmath>
#include <cstdio>

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

  REFV2_STAGE_COPY_XWORK_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_STAGE_COPY_XWORK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      x_work_[token][dim] = authoritative_model_.x_work(token, dim);
    }
  }

  phase_a_valid_ = true;
  layer0_attention_valid_ = false;
  return true;
}

bool RefModel_v2::run_layer0_attention_channel_transport() {
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
  ac_channel<RefV2AttentionTokenVectorPayload> next_stage_token_ch;

  if (!stream_x_work_to_attention_channels(kv_in_token_ch, query_token_ch)) {
    layer0_attention_valid_ = false;
    return false;
  }
  if (!kv_block_.run(kv_in_token_ch, kv_out_k_payload_ch, kv_out_v_payload_ch)) {
    layer0_attention_valid_ = false;
    return false;
  }
  last_k_payload_ = kv_out_k_payload_ch.read();
  last_v_payload_ = kv_out_v_payload_ch.read();
  if (!collect_kv_payload_to_scratch(last_k_payload_, last_v_payload_)) {
    layer0_attention_valid_ = false;
    return false;
  }
  qsoftres_in_k_payload_ch.write(last_k_payload_);
  qsoftres_in_v_payload_ch.write(last_v_payload_);
  if (!qsoftres_block_.run(
        run_cfg_,
        query_token_ch,
        qsoftres_in_k_payload_ch,
        qsoftres_in_v_payload_ch,
        qsoftres_out_token_ch)) {
    layer0_attention_valid_ = false;
    return false;
  }
  if (!writeback_attention_output_stream_to_x_work(qsoftres_out_token_ch)) {
    layer0_attention_valid_ = false;
    return false;
  }
  if (!stream_x_work_to_next_stage(next_stage_token_ch)) {
    layer0_attention_valid_ = false;
    return false;
  }
  if (!consume_and_check_next_stage_stream(next_stage_token_ch)) {
    layer0_attention_valid_ = false;
    return false;
  }

  layer0_attention_valid_ = true;
  return true;
}

bool RefModel_v2::run_step0_layer0_attention_compare(const RefModelIO& io, int batch_index) {
  if (!stage_step0_phase_a_from_authoritative(io, batch_index)) {
    return false;
  }
  if (!run_layer0_attention_channel_transport()) {
    return false;
  }
  if (!compare_against_authoritative_layer0()) {
    return false;
  }
  return true;
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
      scr_k_[token][dim] = zero;
      scr_v_[token][dim] = zero;
    }
  }
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
  reset_compare_point(&last_compare_stats_.attention_input);
  reset_compare_point(&last_compare_stats_.scr_k);
  reset_compare_point(&last_compare_stats_.scr_v);
  reset_compare_point(&last_compare_stats_.x_work_writeback);
  reset_compare_point(&last_compare_stats_.next_stage_handoff);
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

bool RefModel_v2::stream_x_work_to_attention_channels(
  ac_channel<RefV2AttentionTokenVectorPayload>& kv_in_token_ch,
  ac_channel<RefV2AttentionTokenVectorPayload>& query_token_ch) {
  last_attention_input_payload_.header.layer_id = ac_int<8, false>(REFV2_LAYER0_ID);
  last_attention_input_payload_.header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
  last_attention_input_payload_.header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);

  // Top owns X_WORK. Token is row, d is column, row-major flatten keeps d as inner-most.
  REFV2_STREAM_XWORK_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    RefV2AttentionTokenVectorPayload token_payload;
    token_payload.header = last_attention_input_payload_.header;
    token_payload.token_row = ac_int<16, false>(token);

    REFV2_STREAM_XWORK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      token_payload.token_vec[dim] = x_work_[token][dim];
      last_attention_input_payload_.x_flat[idx] = x_work_[token][dim];
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
  ac_channel<RefV2AttentionTokenVectorPayload>& out_token_ch) {
  last_out_payload_.header.layer_id = ac_int<8, false>(REFV2_LAYER0_ID);
  last_out_payload_.header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
  last_out_payload_.header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);

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

    REFV2_WRITEBACK_STREAM_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      last_out_payload_.out_flat[idx] = token_payload.token_vec[dim];
      x_work_[token][dim] = token_payload.token_vec[dim];
    }
  }

  return true;
}

bool RefModel_v2::stream_x_work_to_next_stage(ac_channel<RefV2AttentionTokenVectorPayload>& next_stage_token_ch) {
  // Local-only handoff channel: demonstrate stage-to-stage token-vector transport after X_WORK overwrite.
  REFV2_NEXT_STAGE_STREAM_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    RefV2AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(REFV2_LAYER0_ID);
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
  ac_channel<RefV2AttentionTokenVectorPayload>& next_stage_token_ch) {
  const double tol = last_compare_stats_.tol;

  bool token_seen[REFV2_TOKENS_T];
  REFV2_NEXT_STAGE_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV2_NEXT_STAGE_CHECK_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2AttentionTokenVectorPayload token_payload = next_stage_token_ch.read();
    ++last_compare_stats_.next_stage_token_count;

    if (!refv2_payload_header_matches_shape(token_payload.header) ||
        token_payload.header.layer_id.to_int() != REFV2_LAYER0_ID) {
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
      const double abs_diff = std::fabs(stream_v - x_work_v);
      if (abs_diff > last_compare_stats_.next_stage_handoff.max_abs_diff) {
        last_compare_stats_.next_stage_handoff.max_abs_diff = abs_diff;
      }
      if (abs_diff > tol) {
        ++last_compare_stats_.next_stage_handoff.mismatch_count;
        if (last_compare_stats_.next_stage_handoff.first_mismatch_token < 0) {
          last_compare_stats_.next_stage_handoff.first_mismatch_token = token;
          last_compare_stats_.next_stage_handoff.first_mismatch_dim = dim;
          last_compare_stats_.next_stage_handoff.first_v2_value = stream_v;
          last_compare_stats_.next_stage_handoff.first_ref_value = x_work_v;
        }
      }
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

bool RefModel_v2::compare_against_authoritative_layer0() {
  const double tol = last_compare_stats_.tol;

  REFV2_COMPARE_ATTN_INPUT_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_ATTN_INPUT_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      const double v2_v = static_cast<double>(last_attention_input_payload_.x_flat[idx].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.x_work(token, dim).to_float());
      const double abs_diff = std::fabs(v2_v - ref_v);
      if (abs_diff > last_compare_stats_.attention_input.max_abs_diff) {
        last_compare_stats_.attention_input.max_abs_diff = abs_diff;
      }
      if (abs_diff > tol) {
        ++last_compare_stats_.attention_input.mismatch_count;
        if (last_compare_stats_.attention_input.first_mismatch_token < 0) {
          last_compare_stats_.attention_input.first_mismatch_token = token;
          last_compare_stats_.attention_input.first_mismatch_dim = dim;
          last_compare_stats_.attention_input.first_v2_value = v2_v;
          last_compare_stats_.attention_input.first_ref_value = ref_v;
        }
      }
    }
  }

  REFV2_COMPARE_SCRK_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_SCRK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      const double v2_v = static_cast<double>(last_k_payload_.k_flat[idx].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.scr_k(token, dim).to_float());
      const double abs_diff = std::fabs(v2_v - ref_v);
      if (abs_diff > last_compare_stats_.scr_k.max_abs_diff) {
        last_compare_stats_.scr_k.max_abs_diff = abs_diff;
      }
      if (abs_diff > tol) {
        ++last_compare_stats_.scr_k.mismatch_count;
        if (last_compare_stats_.scr_k.first_mismatch_token < 0) {
          last_compare_stats_.scr_k.first_mismatch_token = token;
          last_compare_stats_.scr_k.first_mismatch_dim = dim;
          last_compare_stats_.scr_k.first_v2_value = v2_v;
          last_compare_stats_.scr_k.first_ref_value = ref_v;
        }
      }
    }
  }

  REFV2_COMPARE_SCRV_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_SCRV_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      const double v2_v = static_cast<double>(last_v_payload_.v_flat[idx].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.scr_v(token, dim).to_float());
      const double abs_diff = std::fabs(v2_v - ref_v);
      if (abs_diff > last_compare_stats_.scr_v.max_abs_diff) {
        last_compare_stats_.scr_v.max_abs_diff = abs_diff;
      }
      if (abs_diff > tol) {
        ++last_compare_stats_.scr_v.mismatch_count;
        if (last_compare_stats_.scr_v.first_mismatch_token < 0) {
          last_compare_stats_.scr_v.first_mismatch_token = token;
          last_compare_stats_.scr_v.first_mismatch_dim = dim;
          last_compare_stats_.scr_v.first_v2_value = v2_v;
          last_compare_stats_.scr_v.first_ref_value = ref_v;
        }
      }
    }
  }

  if (!authoritative_model_.run_step0_layer0_attention_writeback()) {
    return false;
  }

  REFV2_COMPARE_XWORK_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_COMPARE_XWORK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const double v2_v = static_cast<double>(x_work_[token][dim].to_float());
      const double ref_v = static_cast<double>(authoritative_model_.x_work(token, dim).to_float());
      const double abs_diff = std::fabs(v2_v - ref_v);
      if (abs_diff > last_compare_stats_.x_work_writeback.max_abs_diff) {
        last_compare_stats_.x_work_writeback.max_abs_diff = abs_diff;
      }
      if (abs_diff > tol) {
        ++last_compare_stats_.x_work_writeback.mismatch_count;
        if (last_compare_stats_.x_work_writeback.first_mismatch_token < 0) {
          last_compare_stats_.x_work_writeback.first_mismatch_token = token;
          last_compare_stats_.x_work_writeback.first_mismatch_dim = dim;
          last_compare_stats_.x_work_writeback.first_v2_value = v2_v;
          last_compare_stats_.x_work_writeback.first_ref_value = ref_v;
        }
      }
    }
  }

  last_compare_stats_.all_match =
    (last_compare_stats_.attention_input.mismatch_count == 0) &&
    (last_compare_stats_.scr_k.mismatch_count == 0) &&
    (last_compare_stats_.scr_v.mismatch_count == 0) &&
    (last_compare_stats_.x_work_writeback.mismatch_count == 0) &&
    last_compare_stats_.next_stage_handoff_pass;

  print_compare_point("attention_input", last_compare_stats_.attention_input, tol);
  print_compare_point("SCR_K", last_compare_stats_.scr_k, tol);
  print_compare_point("SCR_V", last_compare_stats_.scr_v, tol);
  print_compare_point("x_work_writeback", last_compare_stats_.x_work_writeback, tol);
  print_next_stage_handoff(last_compare_stats_, tol);

  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
