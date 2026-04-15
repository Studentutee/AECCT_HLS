#pragma once

#include "TopManagedAttentionTypes.h"

namespace aecct_ref {
namespace top_managed_attention {

// AttenKvBlock contract:
// - Input : full attention source matrix payload from Top (layer-scoped).
// - Output: full K matrix payload + full V matrix payload over ac_channel.
// - Not responsible for softmax, residual, or X_WORK write-back.
// - Does not own shared SRAM; Top remains the sole shared-storage owner.
class AttenKvBlock {
public:
  AttenKvBlock() : run_count_(0), last_layer_id_(0) {}

  void run(attention_input_payload_ch_t& in_attention_payload_ch,
           attention_k_payload_ch_t& out_k_payload_ch,
           attention_v_payload_ch_t& out_v_payload_ch) {
    const AttentionInputMatrixPayload in_payload = in_attention_payload_ch.read();

    AttentionKMatrixPayload k_payload;
    AttentionVMatrixPayload v_payload;
    k_payload.header = in_payload.header;
    v_payload.header = in_payload.header;

    // Placeholder bounded compute body:
    // K/V are generated as deterministic transforms of Top-fed input payload.
    KV_BLOCK_TOKEN_ROW_LOOP: for (int token = 0; token < TMATTN_TOKENS_T; ++token) {
      KV_BLOCK_DIM_COL_LOOP: for (int dim = 0; dim < TMATTN_D_MODEL; ++dim) {
        const ref_fp32_t x = in_payload.matrix[token][dim];
        const ref_fp32_t k_bias(0.25f);
        const ref_fp32_t v_bias(-0.25f);
        k_payload.matrix[token][dim] = x + k_bias;
        v_payload.matrix[token][dim] = x + v_bias;
      }
    }

    out_k_payload_ch.write(k_payload);
    out_v_payload_ch.write(v_payload);

    ++run_count_;
    last_layer_id_ = in_payload.header.layer_id;
  }

  int run_count() const { return run_count_; }
  int last_layer_id() const { return last_layer_id_.to_int(); }

private:
  int run_count_;
  ac_int<8, false> last_layer_id_;
};

} // namespace top_managed_attention
} // namespace aecct_ref
