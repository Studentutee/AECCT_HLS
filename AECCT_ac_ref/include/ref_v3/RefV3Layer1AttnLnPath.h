#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3AttenKvBlock.h"
#include "ref_v3/RefV3AttenQSoftResBlock.h"
#include "ref_v3/RefV3Config.h"
#include "ref_v3/RefV3LayerNormBlock.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3Layer1AttnLnPath {
public:
  RefV3Layer1AttnLnPath() {}

  // Major-stage boundary: consume layer1 full-matrix payload and stream tokens into layer1 attn+ln path.
  bool run(const RefRunConfig& run_cfg,
           ac_channel<RefV3AttentionInputPayload>& in_xwork_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    const RefV3AttentionInputPayload xwork_payload = in_xwork_ch.read();
    if (!REFV3_payload_header_matches_shape(xwork_payload.header)) {
      return false;
    }
    if (xwork_payload.header.layer_id.to_int() != REFV3_LAYER1_ID) {
      return false;
    }

    ac_channel<RefV3AttentionTokenVectorPayload> xwork_to_attn_token_ch;
    REFV3_LAYER1_ATTLN_XWORK_TO_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      RefV3AttentionTokenVectorPayload token_payload;
      token_payload.header = xwork_payload.header;
      token_payload.token_row = ac_int<16, false>(token);

      REFV3_LAYER1_ATTLN_XWORK_TO_TOKEN_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
        const int idx = REFV3_flatten_row_major_index(token, dim);
        token_payload.token_vec[dim] = xwork_payload.x_flat[idx];
      }
      xwork_to_attn_token_ch.write(token_payload);
    }

    return run(run_cfg, xwork_to_attn_token_ch, out_token_ch);
  }

  bool run(const RefRunConfig& run_cfg,
           ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    ac_channel<RefV3AttentionTokenVectorPayload> kv_in_token_ch;
    ac_channel<RefV3AttentionTokenVectorPayload> query_token_ch;
    ac_channel<RefV3AttentionKPayload> kv_out_k_payload_ch;
    ac_channel<RefV3AttentionVPayload> kv_out_v_payload_ch;
    ac_channel<RefV3AttentionKPayload> qsoftres_in_k_payload_ch;
    ac_channel<RefV3AttentionVPayload> qsoftres_in_v_payload_ch;
    ac_channel<RefV3AttentionTokenVectorPayload> qsoftres_out_token_ch;

    REFV3_LAYER1_ATTLN_SPLIT_INPUT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      const RefV3AttentionTokenVectorPayload token_payload = in_token_ch.read();
      kv_in_token_ch.write(token_payload);
      query_token_ch.write(token_payload);
    }

    if (!kv_block_.run(
          REFV3_LAYER1_ID,
          kv_in_token_ch,
          kv_out_k_payload_ch,
          kv_out_v_payload_ch)) {
      return false;
    }

    qsoftres_in_k_payload_ch.write(kv_out_k_payload_ch.read());
    qsoftres_in_v_payload_ch.write(kv_out_v_payload_ch.read());

    if (!qsoftres_block_.run(
          REFV3_LAYER1_ID,
          run_cfg,
          query_token_ch,
          qsoftres_in_k_payload_ch,
          qsoftres_in_v_payload_ch,
          qsoftres_out_token_ch)) {
      return false;
    }

    if (!ln_block_.run(
          REFV3_LAYER1_ID,
          run_cfg,
          qsoftres_out_token_ch,
          out_token_ch)) {
      return false;
    }

    return true;
  }

private:
  RefV3AttenKvBlock kv_block_;
  RefV3AttenQSoftResBlock qsoftres_block_;
  RefV3LayerNormBlock ln_block_;
};

} // namespace ref_v3
} // namespace aecct_ref

