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

#pragma hls_design top
class RefV3CatapultTop {
public:
  RefV3CatapultTop() {}

#pragma hls_design interface
  bool CCS_BLOCK(run)(
    ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
    ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    ac_channel<RefV3AttentionTokenVectorPayload> kv_in_token_ch;
    ac_channel<RefV3AttentionTokenVectorPayload> query_token_ch;
    ac_channel<RefV3AttentionKPayload> kv_out_k_payload_ch;
    ac_channel<RefV3AttentionVPayload> kv_out_v_payload_ch;
    ac_channel<RefV3AttentionKPayload> qsoftres_in_k_payload_ch;
    ac_channel<RefV3AttentionVPayload> qsoftres_in_v_payload_ch;
    ac_channel<RefV3AttentionTokenVectorPayload> qsoftres_out_token_ch;

    REFV3_CATAPULT_SPLIT_INPUT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      const RefV3AttentionTokenVectorPayload token_payload = in_token_ch.read();
      kv_in_token_ch.write(token_payload);
      query_token_ch.write(token_payload);
    }

    if (!kv_block_.run(
          REFV3_LAYER0_ID,
          kv_in_token_ch,
          kv_out_k_payload_ch,
          kv_out_v_payload_ch)) {
      return false;
    }
    qsoftres_in_k_payload_ch.write(kv_out_k_payload_ch.read());
    qsoftres_in_v_payload_ch.write(kv_out_v_payload_ch.read());

    RefRunConfig run_cfg{};
    if (!qsoftres_block_.run(
          REFV3_LAYER0_ID,
          run_cfg,
          query_token_ch,
          qsoftres_in_k_payload_ch,
          qsoftres_in_v_payload_ch,
          qsoftres_out_token_ch)) {
      return false;
    }

    if (!ln_block_.run(
          REFV3_LAYER0_ID,
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
