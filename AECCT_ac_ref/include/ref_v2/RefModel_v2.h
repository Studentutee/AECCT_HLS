#pragma once

#include "RefModel.h"
#include "RefModelOptimized.h"
#include "ac_channel.h"
#include "ref_v2/RefV2AttenKvBlock.h"
#include "ref_v2/RefV2AttenQSoftResBlock.h"
#include "ref_v2/RefV2FfnBlock.h"
#include "ref_v2/RefV2FinalPassABlock.h"
#include "ref_v2/RefV2FinalPassBBlock.h"
#include "ref_v2/RefV2LayerNormBlock.h"
#include "ref_v2/RefV2PreprocBlock.h"

namespace aecct_ref {
namespace ref_v2 {

struct RefV2ComparePoint {
  int mismatch_count;
  double max_abs_diff;
  int first_mismatch_token;
  int first_mismatch_dim;
  double first_v2_value;
  double first_ref_value;
};

struct RefV2CompareStats {
  RefV2ComparePoint preproc_output;
  RefV2ComparePoint attention_input;
  RefV2ComparePoint scr_k;
  RefV2ComparePoint scr_v;
  RefV2ComparePoint x_work_writeback;
  RefV2ComparePoint layer0_ln_output;
  RefV2ComparePoint x_work_after_layer0_ln;
  RefV2ComparePoint layer0_ffn_output;
  RefV2ComparePoint x_work_after_layer0_ffn;
  RefV2ComparePoint next_stage_handoff;
  RefV2ComparePoint final_passA_output;
  RefV2ComparePoint final_logits;
  RefV2ComparePoint final_x_pred;
  int next_stage_token_count;
  int next_stage_out_of_order_count;
  int next_stage_duplicate_count;
  int next_stage_missing_count;
  int next_stage_header_error_count;
  int next_stage_invalid_token_count;
  bool next_stage_handoff_pass;
  double tol;
  bool all_match;
};

class RefModel_v2 {
public:
  RefModel_v2();

  void set_run_config(const RefRunConfig& cfg);
  RefRunConfig get_run_config() const;

  bool stage_step0_phase_a_from_authoritative(const RefModelIO& io, int batch_index = 0);
  bool run_layer0_attention_channel_transport();
  bool run_layer0_ln_channel_transport();
  bool run_layer0_ffn_channel_transport();
  bool run_step0_layer0_attention_compare(const RefModelIO& io, int batch_index = 0);

  bool phase_a_valid() const;
  bool layer0_attention_valid() const;

  ref_fp32_t x_work(int token, int dim) const;
  ref_fp32_t scr_k(int token, int dim) const;
  ref_fp32_t scr_v(int token, int dim) const;

  RefV2CompareStats last_compare_stats() const;

private:
  void clear_storage();
  void reset_compare_stats();
  bool stream_input_to_preproc_channel(const RefModelIO& io, int batch_index,
                                       ac_channel<RefV2PreprocInputPayload>& preproc_in_ch);
  bool collect_preproc_output_stream_and_writeback(
    ac_channel<RefV2AttentionTokenVectorPayload>& preproc_out_token_ch);
  bool stream_x_work_to_attention_channels(
    ac_channel<RefV2AttentionTokenVectorPayload>& kv_in_token_ch,
    ac_channel<RefV2AttentionTokenVectorPayload>& query_token_ch);
  bool collect_kv_payload_to_scratch(const RefV2AttentionKPayload& k_payload,
                                     const RefV2AttentionVPayload& v_payload);
  bool writeback_attention_output_stream_to_x_work(
    ac_channel<RefV2AttentionTokenVectorPayload>& out_token_ch);
  bool stream_x_work_to_layer0_ln_channel(ac_channel<RefV2AttentionTokenVectorPayload>& ln_in_token_ch);
  bool writeback_layer0_ln_output_stream_to_x_work(
    ac_channel<RefV2AttentionTokenVectorPayload>& ln_out_token_ch);
  bool stream_x_work_to_layer0_ffn_channel(ac_channel<RefV2AttentionTokenVectorPayload>& ffn_in_token_ch);
  bool writeback_layer0_ffn_output_stream_to_x_work(
    ac_channel<RefV2AttentionTokenVectorPayload>& ffn_out_token_ch);
  bool stream_x_work_to_next_stage(ac_channel<RefV2AttentionTokenVectorPayload>& next_stage_token_ch);
  bool consume_and_check_next_stage_stream(
    ac_channel<RefV2AttentionTokenVectorPayload>& next_stage_token_ch);
  bool load_authoritative_end_norm_to_x_work();
  bool stream_x_work_to_final_pass_a_channel(
    ac_channel<RefV2AttentionTokenVectorPayload>& finala_in_token_ch);
  bool collect_final_pass_a_stream_and_forward(
    ac_channel<RefV2FinalScalarTokenPayload>& finala_out_scalar_ch,
    ac_channel<RefV2FinalScalarTokenPayload>& finalb_in_scalar_ch);
  bool stream_input_to_final_pass_b_channel(const RefModelIO& io, int batch_index,
                                            ac_channel<RefV2FinalInputYPayload>& finalb_in_input_y_ch);
  bool collect_final_output_payload(ac_channel<RefV2FinalOutputPayload>& finalb_out_payload_ch);
  bool compare_against_authoritative_layer0();
  bool compare_final_against_authoritative(const RefModelIO& io, int batch_index);
  bool update_overall_match_status();

private:
  RefRunConfig run_cfg_;
  RefModelOptimized authoritative_model_;
  RefV2PreprocBlock preproc_block_;
  RefV2AttenKvBlock kv_block_;
  RefV2AttenQSoftResBlock qsoftres_block_;
  RefV2LayerNormBlock layer0_ln_block_;
  RefV2FfnBlock layer0_ffn_block_;
  RefV2FinalPassABlock final_pass_a_block_;
  RefV2FinalPassBBlock final_pass_b_block_;

  ref_fp32_t x_work_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t preproc_x_work_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t scr_k_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t scr_v_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t x_work_after_attention_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t layer0_ln_out_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t x_work_after_layer0_ln_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t layer0_ffn_out_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t x_work_after_layer0_ffn_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t final_scalar_buf_[REFV2_TOKENS_T];
  ref_fp32_t final_logits_[REFV2_VAR_N];
  bit1_t final_x_pred_[REFV2_VAR_N];

  RefV2PreprocInputPayload last_preproc_input_payload_;
  RefV2AttentionInputPayload last_attention_input_payload_;
  RefV2AttentionKPayload last_k_payload_;
  RefV2AttentionVPayload last_v_payload_;
  RefV2AttentionOutputPayload last_out_payload_;
  RefV2FinalOutputPayload last_final_output_payload_;

  RefV2CompareStats last_compare_stats_;
  bool phase_a_valid_;
  bool layer0_attention_valid_;
};

} // namespace ref_v2
} // namespace aecct_ref
