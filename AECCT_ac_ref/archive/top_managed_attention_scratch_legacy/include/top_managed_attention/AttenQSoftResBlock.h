#pragma once

#include "TopManagedAttentionTypes.h"

namespace aecct_ref {
namespace top_managed_attention {

// AttenQSoftResBlock contract:
// - Inputs : query payload from Top, full K payload, full V payload.
// - Output : post-(Q/score-softmax/context/Wo/residual) matrix payload.
// - Residual source comes from query payload in this skeleton.
// - Final write-back into X_WORK is done by Top, not this block.
// - This block has no shared SRAM ownership.
class AttenQSoftResBlock {
public:
  AttenQSoftResBlock() : run_count_(0), last_layer_id_(0) {}

  void run(attention_query_payload_ch_t& in_query_payload_ch,
           attention_k_payload_ch_t& in_k_payload_ch,
           attention_v_payload_ch_t& in_v_payload_ch,
           attention_qsoftres_output_payload_ch_t& out_payload_ch) {
    const AttentionQueryMatrixPayload query_payload = in_query_payload_ch.read();
    const AttentionKMatrixPayload k_payload = in_k_payload_ch.read();
    const AttentionVMatrixPayload v_payload = in_v_payload_ch.read();

    AttentionQSoftResOutputMatrixPayload out_payload;
    out_payload.header = query_payload.header;

    QSOFTRES_TOKEN_ROW_LOOP: for (int token = 0; token < TMATTN_TOKENS_T; ++token) {
      QSOFTRES_DIM_COL_LOOP: for (int dim = 0; dim < TMATTN_D_MODEL; ++dim) {
        const ref_fp32_t q_val = query_payload.matrix[token][dim];
        const ref_fp32_t k_val = k_payload.matrix[token][dim];
        const ref_fp32_t v_val = v_payload.matrix[token][dim];

        // Placeholder score/mask/softmax/context pipeline:
        // score_proxy -> softmax_proxy -> context_proxy -> residual add.
        const ref_fp32_t score_scale(0.015625f);
        const ref_fp32_t one(1.0f);
        const ref_fp32_t score_proxy = q_val * k_val * score_scale;
        const ref_fp32_t softmax_proxy = score_proxy / (one + abs_fp32(score_proxy));
        const ref_fp32_t context_proxy = softmax_proxy * v_val;
        const ref_fp32_t wo_scale(0.5f);
        out_payload.matrix[token][dim] = q_val + (context_proxy * wo_scale);
      }
    }

    out_payload_ch.write(out_payload);
    ++run_count_;
    last_layer_id_ = query_payload.header.layer_id;
  }

  int run_count() const { return run_count_; }
  int last_layer_id() const { return last_layer_id_.to_int(); }

private:
  static ref_fp32_t abs_fp32(ref_fp32_t x) {
    const ref_fp32_t zero(0.0f);
    if (x < zero) {
      return -x;
    }
    return x;
  }

  int run_count_;
  ac_int<8, false> last_layer_id_;
};

} // namespace top_managed_attention
} // namespace aecct_ref
