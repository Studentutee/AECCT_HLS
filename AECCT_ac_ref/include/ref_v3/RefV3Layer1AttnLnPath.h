#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3AttenKvBlock.h"
#include "ref_v3/RefV3AttenQSoftResBlock.h"
#include "ref_v3/RefV3Config.h"
#include "ref_v3/RefV3LayerNormBlock.h"

#if defined(__has_include)
#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#endif
#endif

#ifndef CCS_BLOCK
#define CCS_BLOCK(name) name
#endif

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

    ac_channel<RefV3AttentionTokenVectorPayload> xwork_to_kv_token_ch;
    ac_channel<RefV3AttentionInputPayload> xwork_to_qsoftres_ch;
    xwork_to_qsoftres_ch.write(xwork_payload);

    REFV3_LAYER1_ATTLN_XWORK_TO_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      RefV3AttentionTokenVectorPayload token_payload;
      token_payload.header = xwork_payload.header;
      token_payload.token_row = ac_int<16, false>(token);

      REFV3_LAYER1_ATTLN_XWORK_TO_TOKEN_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
        const int idx = REFV3_flatten_row_major_index(token, dim);
        token_payload.token_vec[dim] = xwork_payload.x_flat[idx];
      }
      xwork_to_kv_token_ch.write(token_payload);
    }

    return run(run_cfg, xwork_to_kv_token_ch, xwork_to_qsoftres_ch, out_token_ch);
  }

  // Mainline path: token stream feeds KV only, and QSoftRes consumes X_WORK for query/residual base.
  // Catapult class-based interface entry for hierarchical block.
  // CCS_BLOCK added for SCVerify/Catapult hierarchy friendliness.
#pragma hls_design interface
  bool CCS_BLOCK(run)(const RefRunConfig& run_cfg,
                      ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                      ac_channel<RefV3AttentionInputPayload>& in_xwork_ch,
                      ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    ac_channel<RefV3AttentionKPayload> kv_out_k_payload_ch;
    ac_channel<RefV3AttentionVPayload> kv_out_v_payload_ch;
    ac_channel<RefV3AttentionTokenVectorPayload> qsoftres_out_token_ch;

    if (!kv_block_.run(
          REFV3_LAYER1_ID,
          in_token_ch,
          kv_out_k_payload_ch,
          kv_out_v_payload_ch)) {
      return false;
    }

    if (!qsoftres_block_.run(
          REFV3_LAYER1_ID,
          run_cfg,
          in_xwork_ch,
          kv_out_k_payload_ch,
          kv_out_v_payload_ch,
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

  // Compatibility wrapper for token-only callers; reconstructs X_WORK locally.
  bool run(const RefRunConfig& run_cfg,
           ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    ac_channel<RefV3AttentionTokenVectorPayload> kv_in_token_ch;
    ac_channel<RefV3AttentionInputPayload> qsoftres_in_xwork_ch;
    RefV3AttentionInputPayload xwork_payload;
    bool header_init = false;
    bool token_seen[REFV3_TOKENS_T];

    REFV3_LAYER1_ATTLN_COMPAT_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      token_seen[token] = false;
    }

    REFV3_LAYER1_ATTLN_COMPAT_READ_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      const RefV3AttentionTokenVectorPayload token_payload = in_token_ch.read();
      if (!REFV3_payload_header_matches_shape(token_payload.header)) {
        return false;
      }
      if (token_payload.header.layer_id.to_int() != REFV3_LAYER1_ID) {
        return false;
      }
      if (!header_init) {
        xwork_payload.header = token_payload.header;
        header_init = true;
      } else {
        if (token_payload.header.layer_id != xwork_payload.header.layer_id ||
            token_payload.header.token_rows != xwork_payload.header.token_rows ||
            token_payload.header.dim_cols != xwork_payload.header.dim_cols) {
          return false;
        }
      }

      const int q_token = token_payload.token_row.to_int();
      if (q_token < 0 || q_token >= REFV3_TOKENS_T) {
        return false;
      }
      if (token_seen[q_token]) {
        return false;
      }
      token_seen[q_token] = true;

      kv_in_token_ch.write(token_payload);
      REFV3_LAYER1_ATTLN_COMPAT_COPY_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
        const int idx = REFV3_flatten_row_major_index(q_token, dim);
        xwork_payload.x_flat[idx] = token_payload.token_vec[dim];
      }
    }

    if (!header_init) {
      return false;
    }

    qsoftres_in_xwork_ch.write(xwork_payload);
    return run(run_cfg, kv_in_token_ch, qsoftres_in_xwork_ch, out_token_ch);
  }

private:
  RefV3AttenKvBlock kv_block_;
  RefV3AttenQSoftResBlock qsoftres_block_;
  RefV3LayerNormBlock ln_block_;
};

} // namespace ref_v3
} // namespace aecct_ref
