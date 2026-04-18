#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Config.h"
#include "ref_v3/RefV3LayerNormBlock.h"
#include "ref_v3/RefV3Payload.h"
#include "ref_v3/RefV3WeightsFp16LocalOnly.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3MidNormPath {
public:
  RefV3MidNormPath() {}

  // Main path: preserve token stream while emitting a side full-matrix snapshot for layer1 ownership boundary.
  bool run(const RefRunConfig& run_cfg,
           ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch,
           ac_channel<RefV3AttentionInputPayload>& out_xwork_ch) {
    ac_channel<RefV3AttentionTokenVectorPayload> ch_midnorm_in;
    ac_channel<RefV3AttentionTokenVectorPayload> ch_midnorm_out;

    // Ownership boundary retag: layer0 FFN output becomes layer1 stage input carrier before mid norm.
    REFV3_MIDNORM_RETARGET_LAYER_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      const RefV3AttentionTokenVectorPayload in_payload = in_token_ch.read();
      if (!REFV3_payload_header_matches_shape(in_payload.header)) {
        return false;
      }
      if (in_payload.header.layer_id.to_int() != REFV3_LAYER0_ID) {
        return false;
      }

      RefV3AttentionTokenVectorPayload retag_payload = in_payload;
      retag_payload.header.layer_id = ac_int<8, false>(REFV3_LAYER1_ID);
      ch_midnorm_in.write(retag_payload);
    }

    // Mid norm must use decoder.norm2 parameters, not layer1 sublayer norm parameters.
    const RefV3TernaryLinearParams midnorm_params = refv3_midnorm_params_fp_local_only();
    if (!mid_norm_block_.run_with_params(
          REFV3_LAYER1_ID,
          midnorm_params,
          run_cfg,
          ch_midnorm_in,
          ch_midnorm_out)) {
      return false;
    }

    RefV3AttentionInputPayload xwork_payload;
    xwork_payload.header.layer_id = ac_int<8, false>(REFV3_LAYER1_ID);
    xwork_payload.header.token_rows = ac_int<16, false>(REFV3_TOKENS_T);
    xwork_payload.header.dim_cols = ac_int<16, false>(REFV3_D_MODEL);

    bool token_seen[REFV3_TOKENS_T];
    REFV3_MIDNORM_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      token_seen[token] = false;
    }

    REFV3_MIDNORM_PACK_XWORK_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
      const RefV3AttentionTokenVectorPayload out_payload = ch_midnorm_out.read();
      if (!REFV3_payload_header_matches_shape(out_payload.header)) {
        return false;
      }
      if (out_payload.header.layer_id.to_int() != REFV3_LAYER1_ID) {
        return false;
      }

      const int token = out_payload.token_row.to_int();
      if (token < 0 || token >= REFV3_TOKENS_T) {
        return false;
      }
      if (token_seen[token]) {
        return false;
      }
      token_seen[token] = true;

      out_token_ch.write(out_payload);

      REFV3_MIDNORM_PACK_XWORK_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
        const int idx = REFV3_flatten_row_major_index(token, dim);
        xwork_payload.x_flat[idx] = out_payload.token_vec[dim];
      }
    }

    out_xwork_ch.write(xwork_payload);
    return true;
  }

  // Compatibility wrapper: keep xwork-only callers available.
  bool run(const RefRunConfig& run_cfg,
           ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV3AttentionInputPayload>& out_xwork_ch) {
    ac_channel<RefV3AttentionTokenVectorPayload> ch_token_local_only;
    if (!run(run_cfg, in_token_ch, ch_token_local_only, out_xwork_ch)) {
      return false;
    }
    REFV3_MIDNORM_SIDE_TOKEN_DRAIN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      (void)ch_token_local_only.read();
    }
    return true;
  }

  // Compatibility wrapper: keep token-only callers available.
  bool run(const RefRunConfig& run_cfg,
           ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    ac_channel<RefV3AttentionInputPayload> ch_xwork_local_only;
    if (!run(run_cfg, in_token_ch, out_token_ch, ch_xwork_local_only)) {
      return false;
    }
    (void)ch_xwork_local_only.read();
    return true;
  }

private:
  RefV3LayerNormBlock mid_norm_block_;
};

} // namespace ref_v3
} // namespace aecct_ref
