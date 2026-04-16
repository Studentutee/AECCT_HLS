#pragma once

#ifndef REFV3_ENABLE_COMPARE
#define REFV3_ENABLE_COMPARE 1
#endif

#include "RefModel.h"
#if REFV3_ENABLE_COMPARE
#include "RefModelOptimized.h"
#endif
#include "ac_channel.h"
#include "ref_v3/RefV3AttenKvBlock.h"
#include "ref_v3/RefV3AttenQSoftResBlock.h"
#include "ref_v3/RefV3FfnLinear0ReluBlock.h"
#include "ref_v3/RefV3FfnLinear1ResidualBlock.h"
#include "ref_v3/RefV3FinalPassABlock.h"
#include "ref_v3/RefV3FinalPassBBlock.h"
#include "ref_v3/RefV3LayerNormBlock.h"
#include "ref_v3/RefV3PreprocBlock.h"

namespace aecct_ref {
namespace ref_v3 {

struct RefV3ComparePoint {
  int mismatch_count;
  double max_abs_diff;
  int first_mismatch_token;
  int first_mismatch_dim;
  double first_v2_value;
  double first_ref_value;
};

struct RefV3CompareStats {
  RefV3ComparePoint preproc_output;
  RefV3ComparePoint attention_input;
  RefV3ComparePoint scr_k;
  RefV3ComparePoint scr_v;
  RefV3ComparePoint x_work_writeback;
  RefV3ComparePoint layer0_ln_output;
  RefV3ComparePoint x_work_after_layer0_ln;
  RefV3ComparePoint layer0_ffn_output;
  RefV3ComparePoint x_work_after_layer0_ffn;
  RefV3ComparePoint next_stage_handoff;
  RefV3ComparePoint final_passA_output;
  RefV3ComparePoint final_logits;
  RefV3ComparePoint final_x_pred;
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

class RefModel_v3 {
public:
  RefModel_v3();

  void set_run_config(const RefRunConfig& cfg);
  RefRunConfig get_run_config() const;

  bool stage_step0_phase_a_from_authoritative(const RefModelIO& io, int batch_index = 0);
  bool run_layer0_attention_channel_transport();
  bool run_layer0_ln_channel_transport();
  bool run_layer0_ffn_channel_transport();
#if REFV3_ENABLE_COMPARE
  bool run_step0_layer0_attention_compare(const RefModelIO& io, int batch_index = 0);
#endif

  bool phase_a_valid() const;
  bool layer0_attention_valid() const;

  refv3_fp_t x_work(int token, int dim) const;
  refv3_fp_t scr_k(int token, int dim) const;
  refv3_fp_t scr_v(int token, int dim) const;

#if REFV3_ENABLE_COMPARE
  RefV3CompareStats last_compare_stats() const;
#endif

private:
  void clear_storage();
#if REFV3_ENABLE_COMPARE
  void reset_compare_stats();
#endif
  bool run_attention_layer_shared(int lid);
  bool run_ln_layer_shared(int lid);
  bool run_ffn_layer_shared(int lid);
  bool run_transformer_layer_shared(int lid);
  bool stream_input_to_preproc_channel(const RefModelIO& io, int batch_index,
                                       ac_channel<RefV3PreprocInputPayload>& preproc_in_ch);
  bool collect_preproc_output_stream_and_writeback(
    ac_channel<RefV3AttentionTokenVectorPayload>& preproc_out_token_ch);
  bool stream_x_work_to_attention_channels(
    int lid,
    ac_channel<RefV3AttentionTokenVectorPayload>& kv_in_token_ch,
    ac_channel<RefV3AttentionTokenVectorPayload>& query_token_ch);
  bool collect_kv_payload_to_scratch(const RefV3AttentionKPayload& k_payload,
                                     const RefV3AttentionVPayload& v_payload);
  bool writeback_attention_output_stream_to_x_work(
    int lid,
    ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch);
  bool stream_x_work_to_ln_channel(int lid,
                                   ac_channel<RefV3AttentionTokenVectorPayload>& ln_in_token_ch);
  bool writeback_ln_output_stream_to_x_work(
    int lid,
    ac_channel<RefV3AttentionTokenVectorPayload>& ln_out_token_ch);
  bool stream_x_work_to_ffn_channels(
    int lid,
    ac_channel<RefV3AttentionTokenVectorPayload>& ffn_linear0_in_token_ch,
    ac_channel<RefV3AttentionTokenVectorPayload>& ffn_residual_in_token_ch);
  bool writeback_ffn_output_stream_to_x_work(
    int lid,
    ac_channel<RefV3AttentionTokenVectorPayload>& ffn_out_token_ch);
  bool stream_x_work_to_next_stage(int lid,
                                   ac_channel<RefV3AttentionTokenVectorPayload>& next_stage_token_ch);
  bool consume_and_check_next_stage_stream(
    int lid,
    ac_channel<RefV3AttentionTokenVectorPayload>& next_stage_token_ch);
  bool load_authoritative_end_norm_to_x_work();
  bool stream_x_work_to_final_pass_a_channel(
    ac_channel<RefV3AttentionTokenVectorPayload>& finala_in_token_ch);
  bool collect_final_pass_a_stream_and_forward(
    ac_channel<RefV3FinalScalarTokenPayload>& finala_out_scalar_ch,
    ac_channel<RefV3FinalScalarTokenPayload>& finalb_in_scalar_ch);
  bool stream_input_to_final_pass_b_channel(const RefModelIO& io, int batch_index,
                                            ac_channel<RefV3FinalInputYPayload>& finalb_in_input_y_ch);
  bool collect_final_output_payload(ac_channel<RefV3FinalOutputPayload>& finalb_out_payload_ch);
#if REFV3_ENABLE_COMPARE
  bool compare_against_authoritative_layer0();
  bool compare_final_against_authoritative(const RefModelIO& io, int batch_index);
  bool update_overall_match_status();
#endif

private:
  RefRunConfig run_cfg_;
#if REFV3_ENABLE_COMPARE
  RefModelOptimized authoritative_model_;
#endif
  RefV3PreprocBlock preproc_block_;
  RefV3AttenKvBlock kv_block_;
  RefV3AttenQSoftResBlock qsoftres_block_;
  RefV3LayerNormBlock ln_block_;
  RefV3FfnLinear0ReluBlock ffn_linear0_relu_block_;
  RefV3FfnLinear1ResidualBlock ffn_linear1_residual_block_;
  RefV3FinalPassABlock final_pass_a_block_;
  RefV3FinalPassBBlock final_pass_b_block_;

  refv3_fp_t x_work_[REFV3_TOKENS_T][REFV3_D_MODEL];
  refv3_fp_t preproc_x_work_[REFV3_TOKENS_T][REFV3_D_MODEL];
  refv3_fp_t scr_k_[REFV3_TOKENS_T][REFV3_D_MODEL];
  refv3_fp_t scr_v_[REFV3_TOKENS_T][REFV3_D_MODEL];
  refv3_fp_t x_work_after_attention_[REFV3_TOKENS_T][REFV3_D_MODEL];
  refv3_fp_t layer0_ln_out_[REFV3_TOKENS_T][REFV3_D_MODEL];
  refv3_fp_t x_work_after_layer0_ln_[REFV3_TOKENS_T][REFV3_D_MODEL];
  refv3_fp_t layer0_ffn_out_[REFV3_TOKENS_T][REFV3_D_MODEL];
  refv3_fp_t x_work_after_layer0_ffn_[REFV3_TOKENS_T][REFV3_D_MODEL];
  // Local-only compare evidence; not part of production dataflow state.
  refv3_fp_t final_pass_a_observe_scalar_[REFV3_TOKENS_T];
  refv3_fp_t final_logits_[REFV3_VAR_N];
  bit1_t final_x_pred_[REFV3_VAR_N];

  RefV3PreprocInputPayload last_preproc_input_payload_;
  RefV3AttentionInputPayload last_attention_input_payload_;
  RefV3AttentionKPayload last_k_payload_;
  RefV3AttentionVPayload last_v_payload_;
  RefV3AttentionOutputPayload last_out_payload_;
  RefV3FinalOutputPayload last_final_output_payload_;

#if REFV3_ENABLE_COMPARE
  RefV3CompareStats last_compare_stats_;
#endif
  bool phase_a_valid_;
  bool layer0_attention_valid_;
};

} // namespace ref_v3
} // namespace aecct_ref
