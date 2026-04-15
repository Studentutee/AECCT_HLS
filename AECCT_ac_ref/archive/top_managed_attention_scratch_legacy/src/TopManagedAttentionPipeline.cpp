#include "../include/top_managed_attention/TopManagedAttentionPipeline.h"

#include <algorithm>

namespace aecct_ref {
namespace top_managed_attention {

TopManagedAttentionPipeline::TopManagedAttentionPipeline()
  : top_payload_emit_count_(0),
    top_writeback_count_(0),
    last_contract_ok_(false),
    last_writeback_row_major_ok_(false) {
  clear_x_work();
}

void TopManagedAttentionPipeline::clear_x_work() {
  TOP_CLEAR_X_WORK_TOKEN_LOOP: for (int token = 0; token < TMATTN_TOKENS_T; ++token) {
    TOP_CLEAR_X_WORK_DIM_LOOP: for (int dim = 0; dim < TMATTN_D_MODEL; ++dim) {
      x_work_[token][dim] = ref_fp32_t(0.0f);
    }
  }
  TOP_CLEAR_X_PRED_LOOP: for (int i = 0; i < ModelShapes::N_VARS; ++i) {
    x_pred_[i] = bit1_t(0);
  }
  top_payload_emit_count_ = 0;
  top_writeback_count_ = 0;
  last_contract_ok_ = false;
  last_writeback_row_major_ok_ = false;
}

void TopManagedAttentionPipeline::load_input_y_into_top_x_work(const double* input_y_fp32, int input_len) {
  TOP_LOAD_INPUT_TOKEN_LOOP: for (int token = 0; token < TMATTN_TOKENS_T; ++token) {
    TOP_LOAD_INPUT_DIM_LOOP: for (int dim = 0; dim < TMATTN_D_MODEL; ++dim) {
      const int flat_idx = flatten_row_major_index(token, dim);
      if (input_y_fp32 != 0 && flat_idx < input_len) {
        x_work_[token][dim] = ref_fp32_t(static_cast<float>(input_y_fp32[flat_idx]));
      } else {
        // Deterministic local-only seed for skeleton bring-up.
        const int pattern = ((token + 1) * 13) - ((dim + 3) * 7);
        x_work_[token][dim] = ref_fp32_t(static_cast<float>(pattern) * 0.03125f);
      }
    }
  }
  update_top_output_x_pred();
}

bool TopManagedAttentionPipeline::run_layer0_attention_skeleton() {
  AttentionInputMatrixPayload attention_input_payload;
  AttentionQueryMatrixPayload attention_query_payload;

  build_attention_input_payload_from_x_work(attention_input_payload);
  build_attention_query_payload_from_x_work(attention_query_payload);

  top_to_kv_input_ch_.write(attention_input_payload);
  top_to_qsoft_query_ch_.write(attention_query_payload);
  top_payload_emit_count_ += 2;

  kv_block_.run(top_to_kv_input_ch_, kv_to_qsoft_k_ch_, kv_to_qsoft_v_ch_);
  qsoftres_block_.run(top_to_qsoft_query_ch_, kv_to_qsoft_k_ch_, kv_to_qsoft_v_ch_, qsoft_to_top_out_ch_);

  const AttentionQSoftResOutputMatrixPayload out_payload = qsoft_to_top_out_ch_.read();

  last_contract_ok_ = payload_header_matches_shape(out_payload.header) &&
                      (out_payload.header.layer_id.to_int() == TMATTN_LAYER0_ID) &&
                      (kv_block_.last_layer_id() == TMATTN_LAYER0_ID) &&
                      (qsoftres_block_.last_layer_id() == TMATTN_LAYER0_ID);

  if (!last_contract_ok_) {
    return false;
  }

  if (!writeback_output_payload_to_x_work(out_payload)) {
    return false;
  }

  update_top_output_x_pred();
  return true;
}

void TopManagedAttentionPipeline::export_x_pred_from_top(bit1_t* x_pred_out, int x_pred_len) const {
  if (x_pred_out == 0 || x_pred_len <= 0) {
    return;
  }
  const int copy_n = std::min(x_pred_len, ModelShapes::N_VARS);
  TOP_EXPORT_X_PRED_COPY_LOOP: for (int i = 0; i < copy_n; ++i) {
    x_pred_out[i] = x_pred_[i];
  }
  TOP_EXPORT_X_PRED_PAD_LOOP: for (int i = copy_n; i < x_pred_len; ++i) {
    x_pred_out[i] = bit1_t(0);
  }
}

ref_fp32_t TopManagedAttentionPipeline::x_work(int token, int dim) const {
  if (token < 0 || token >= TMATTN_TOKENS_T || dim < 0 || dim >= TMATTN_D_MODEL) {
    return ref_fp32_t(0.0f);
  }
  return x_work_[token][dim];
}

int TopManagedAttentionPipeline::top_payload_emit_count() const { return top_payload_emit_count_; }
int TopManagedAttentionPipeline::top_writeback_count() const { return top_writeback_count_; }
bool TopManagedAttentionPipeline::last_contract_ok() const { return last_contract_ok_; }
bool TopManagedAttentionPipeline::last_writeback_row_major_ok() const { return last_writeback_row_major_ok_; }
int TopManagedAttentionPipeline::kv_block_run_count() const { return kv_block_.run_count(); }
int TopManagedAttentionPipeline::qsoftres_block_run_count() const { return qsoftres_block_.run_count(); }

void TopManagedAttentionPipeline::build_attention_input_payload_from_x_work(
  AttentionInputMatrixPayload& payload) const {
  payload.header.layer_id = ac_int<8, false>(TMATTN_LAYER0_ID);
  payload.header.token_rows = ac_int<16, false>(TMATTN_TOKENS_T);
  payload.header.dim_cols = ac_int<16, false>(TMATTN_D_MODEL);

  TOP_BUILD_INPUT_TOKEN_LOOP: for (int token = 0; token < TMATTN_TOKENS_T; ++token) {
    TOP_BUILD_INPUT_DIM_LOOP: for (int dim = 0; dim < TMATTN_D_MODEL; ++dim) {
      payload.matrix[token][dim] = x_work_[token][dim];
    }
  }
}

void TopManagedAttentionPipeline::build_attention_query_payload_from_x_work(
  AttentionQueryMatrixPayload& payload) const {
  payload.header.layer_id = ac_int<8, false>(TMATTN_LAYER0_ID);
  payload.header.token_rows = ac_int<16, false>(TMATTN_TOKENS_T);
  payload.header.dim_cols = ac_int<16, false>(TMATTN_D_MODEL);

  TOP_BUILD_QUERY_TOKEN_LOOP: for (int token = 0; token < TMATTN_TOKENS_T; ++token) {
    TOP_BUILD_QUERY_DIM_LOOP: for (int dim = 0; dim < TMATTN_D_MODEL; ++dim) {
      payload.matrix[token][dim] = x_work_[token][dim];
    }
  }
}

bool TopManagedAttentionPipeline::writeback_output_payload_to_x_work(
  const AttentionQSoftResOutputMatrixPayload& payload) {
  if (!payload_header_matches_shape(payload.header)) {
    last_writeback_row_major_ok_ = false;
    return false;
  }

  int write_seq = 0;
  bool row_major_ok = true;
  TOP_WRITEBACK_TOKEN_LOOP: for (int token = 0; token < TMATTN_TOKENS_T; ++token) {
    TOP_WRITEBACK_DIM_LOOP: for (int dim = 0; dim < TMATTN_D_MODEL; ++dim) {
      const int expected_flat = flatten_row_major_index(token, dim);
      if (write_seq != expected_flat) {
        row_major_ok = false;
      }
      x_work_[token][dim] = payload.matrix[token][dim];
      ++write_seq;
    }
  }

  top_writeback_count_ = write_seq;
  last_writeback_row_major_ok_ = row_major_ok;
  return row_major_ok;
}

void TopManagedAttentionPipeline::update_top_output_x_pred() {
  TOP_UPDATE_XPRED_LOOP: for (int i = 0; i < ModelShapes::N_VARS; ++i) {
    const int token = i / TMATTN_D_MODEL;
    const int dim = i % TMATTN_D_MODEL;
    const ref_fp32_t value = x_work_[token][dim];
    const ref_fp32_t zero(0.0f);
    x_pred_[i] = bit1_t((value < zero) ? 1 : 0);
  }
}

} // namespace top_managed_attention
} // namespace aecct_ref
