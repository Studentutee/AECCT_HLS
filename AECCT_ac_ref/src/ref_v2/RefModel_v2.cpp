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

bool RefModel_v2::run_layer0_attention_direct_call() {
  if (!phase_a_valid_) {
    return false;
  }

  pack_attention_input_payload_from_x_work(&last_attention_input_payload_);

  if (!kv_block_.run(last_attention_input_payload_, &last_k_payload_, &last_v_payload_)) {
    layer0_attention_valid_ = false;
    return false;
  }
  if (!collect_kv_payload_to_scratch(last_k_payload_, last_v_payload_)) {
    layer0_attention_valid_ = false;
    return false;
  }

  if (!qsoftres_block_.run(
        run_cfg_,
        last_attention_input_payload_,
        last_k_payload_,
        last_v_payload_,
        &last_out_payload_)) {
    layer0_attention_valid_ = false;
    return false;
  }
  if (!writeback_attention_output_to_x_work(last_out_payload_)) {
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
  if (!run_layer0_attention_direct_call()) {
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
  last_compare_stats_.tol = 1.0e-6;
  last_compare_stats_.all_match = false;
}

void RefModel_v2::pack_attention_input_payload_from_x_work(RefV2AttentionInputPayload* payload) const {
  payload->header.layer_id = ac_int<8, false>(REFV2_LAYER0_ID);
  payload->header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
  payload->header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);

  REFV2_PACK_INPUT_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_PACK_INPUT_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      payload->x_flat[idx] = x_work_[token][dim];
    }
  }
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

bool RefModel_v2::writeback_attention_output_to_x_work(const RefV2AttentionOutputPayload& out_payload) {
  if (!refv2_payload_header_matches_shape(out_payload.header)) {
    return false;
  }

  REFV2_WRITEBACK_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    REFV2_WRITEBACK_DIM_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const int idx = refv2_flatten_row_major_index(token, dim);
      x_work_[token][dim] = out_payload.out_flat[idx];
    }
  }
  return true;
}

bool RefModel_v2::compare_against_authoritative_layer0() {
  reset_compare_stats();
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
    (last_compare_stats_.x_work_writeback.mismatch_count == 0);

  print_compare_point("attention_input", last_compare_stats_.attention_input, tol);
  print_compare_point("SCR_K", last_compare_stats_.scr_k, tol);
  print_compare_point("SCR_V", last_compare_stats_.scr_v, tol);
  print_compare_point("x_work_writeback", last_compare_stats_.x_work_writeback, tol);

  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
