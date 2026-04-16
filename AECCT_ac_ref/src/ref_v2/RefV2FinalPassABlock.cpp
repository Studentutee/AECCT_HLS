#include "../../include/ref_v2/RefV2FinalPassABlock.h"

#include "weights.h"

namespace aecct_ref {
namespace ref_v2 {

RefV2FinalPassABlock::RefV2FinalPassABlock() {}

bool RefV2FinalPassABlock::run(ac_channel<RefV2AttentionTokenVectorPayload>& in_token_ch,
                               ac_channel<RefV2FinalScalarTokenPayload>& out_scalar_ch) const {
  RefV2AttentionPayloadHeader header_ref;
  bool header_init = false;
  bool token_seen[REFV2_TOKENS_T];

  const ref_fp32_t bias(static_cast<float>(w_oned_final_embed_0_bias[0]));

  REFV2_FINALA_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV2_FINALA_TOKEN_STREAM_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2AttentionTokenVectorPayload token_payload = in_token_ch.read();
    if (!refv2_payload_header_matches_shape(token_payload.header)) {
      return false;
    }

    if (!header_init) {
      header_ref = token_payload.header;
      header_init = true;
    } else {
      if (token_payload.header.layer_id != header_ref.layer_id ||
          token_payload.header.token_rows != header_ref.token_rows ||
          token_payload.header.dim_cols != header_ref.dim_cols) {
        return false;
      }
    }

    const int token = token_payload.token_row.to_int();
    if (token < 0 || token >= REFV2_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    RefV2FinalScalarTokenPayload scalar_payload;
    scalar_payload.header = token_payload.header;
    scalar_payload.token_row = token_payload.token_row;
    scalar_payload.scalar = bias;

    REFV2_FINALA_DIM_MAC_LOOP: for (int dim = 0; dim < REFV2_D_MODEL; ++dim) {
      const ref_fp32_t w_d(static_cast<float>(w_oned_final_embed_0_weight[dim]));
      scalar_payload.scalar += token_payload.token_vec[dim] * w_d;
    }
    out_scalar_ch.write(scalar_payload);
  }

  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
