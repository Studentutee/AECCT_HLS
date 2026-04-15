#pragma once

#include "AttenKvBlock.h"
#include "AttenQSoftResBlock.h"

namespace aecct_ref {
namespace top_managed_attention {

// Top-managed skeleton for layer0 attention:
// - Top owns X_WORK lifecycle and all write-back.
// - Blocks only consume/produce matrix payloads through ac_channel.
// - This is a parallel skeleton track for block-ready architecture bring-up.
class TopManagedAttentionPipeline {
public:
  TopManagedAttentionPipeline();

  void clear_x_work();
  void load_input_y_into_top_x_work(const double* input_y_fp32, int input_len);
  bool run_layer0_attention_skeleton();
  void export_x_pred_from_top(bit1_t* x_pred_out, int x_pred_len) const;

  ref_fp32_t x_work(int token, int dim) const;

  int top_payload_emit_count() const;
  int top_writeback_count() const;
  bool last_contract_ok() const;
  bool last_writeback_row_major_ok() const;
  int kv_block_run_count() const;
  int qsoftres_block_run_count() const;

private:
  void build_attention_input_payload_from_x_work(AttentionInputMatrixPayload& payload) const;
  void build_attention_query_payload_from_x_work(AttentionQueryMatrixPayload& payload) const;
  bool writeback_output_payload_to_x_work(const AttentionQSoftResOutputMatrixPayload& payload);
  void update_top_output_x_pred();

  ref_fp32_t x_work_[TMATTN_TOKENS_T][TMATTN_D_MODEL];
  bit1_t x_pred_[ModelShapes::N_VARS];

  AttenKvBlock kv_block_;
  AttenQSoftResBlock qsoftres_block_;

  attention_input_payload_ch_t top_to_kv_input_ch_;
  attention_query_payload_ch_t top_to_qsoft_query_ch_;
  attention_k_payload_ch_t kv_to_qsoft_k_ch_;
  attention_v_payload_ch_t kv_to_qsoft_v_ch_;
  attention_qsoftres_output_payload_ch_t qsoft_to_top_out_ch_;

  int top_payload_emit_count_;
  int top_writeback_count_;
  bool last_contract_ok_;
  bool last_writeback_row_major_ok_;
};

} // namespace top_managed_attention
} // namespace aecct_ref
