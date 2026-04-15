#pragma once

#include "RefModel.h"
#include "RefModelOptimized.h"
#include "ref_v2/RefV2AttenKvBlock.h"
#include "ref_v2/RefV2AttenQSoftResBlock.h"

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
  RefV2ComparePoint attention_input;
  RefV2ComparePoint scr_k;
  RefV2ComparePoint scr_v;
  RefV2ComparePoint x_work_writeback;
  double tol;
  bool all_match;
};

class RefModel_v2 {
public:
  RefModel_v2();

  void set_run_config(const RefRunConfig& cfg);
  RefRunConfig get_run_config() const;

  bool stage_step0_phase_a_from_authoritative(const RefModelIO& io, int batch_index = 0);
  bool run_layer0_attention_direct_call();
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
  void pack_attention_input_payload_from_x_work(RefV2AttentionInputPayload* payload) const;
  bool collect_kv_payload_to_scratch(const RefV2AttentionKPayload& k_payload,
                                     const RefV2AttentionVPayload& v_payload);
  bool writeback_attention_output_to_x_work(const RefV2AttentionOutputPayload& out_payload);
  bool compare_against_authoritative_layer0();

private:
  RefRunConfig run_cfg_;
  RefModelOptimized authoritative_model_;
  RefV2AttenKvBlock kv_block_;
  RefV2AttenQSoftResBlock qsoftres_block_;

  ref_fp32_t x_work_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t scr_k_[REFV2_TOKENS_T][REFV2_D_MODEL];
  ref_fp32_t scr_v_[REFV2_TOKENS_T][REFV2_D_MODEL];

  RefV2AttentionInputPayload last_attention_input_payload_;
  RefV2AttentionKPayload last_k_payload_;
  RefV2AttentionVPayload last_v_payload_;
  RefV2AttentionOutputPayload last_out_payload_;

  RefV2CompareStats last_compare_stats_;
  bool phase_a_valid_;
  bool layer0_attention_valid_;
};

} // namespace ref_v2
} // namespace aecct_ref
