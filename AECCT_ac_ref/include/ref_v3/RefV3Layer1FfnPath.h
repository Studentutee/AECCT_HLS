#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Config.h"
#include "ref_v3/RefV3FfnLinear0ReluBlock.h"
#include "ref_v3/RefV3FfnLinear1ResidualBlock.h"

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

class RefV3Layer1FfnPath {
public:
  RefV3Layer1FfnPath() {}

  // Layer1 FFN path: token stream in -> split -> linear0/relu + residual FIFO -> linear1+residual out.
  // Catapult class-based interface entry for hierarchical block.
  // CCS_BLOCK added for SCVerify/Catapult hierarchy friendliness.
#pragma hls_design interface
  bool CCS_BLOCK(run)(ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                      ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    ac_channel<RefV3AttentionTokenVectorPayload> ch_l1_ffn_linear0_in;
    ac_channel<RefV3AttentionTokenVectorPayload> residual_fifo_l1;
    ac_channel<RefV3FfnHiddenTokenPayload> ch_l1_ffn_linear0_to_linear1;

    // Dedicated long residual FIFO structure for layer1 FFN; intended depth target is >=32 via directives.
    REFV3_LAYER1_FFN_SPLIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
      const RefV3AttentionTokenVectorPayload token_payload = in_token_ch.read();
      if (!REFV3_payload_header_matches_shape(token_payload.header)) {
        return false;
      }
      if (token_payload.header.layer_id.to_int() != REFV3_LAYER1_ID) {
        return false;
      }
      ch_l1_ffn_linear0_in.write(token_payload);
      residual_fifo_l1.write(token_payload);
    }

    if (!linear0_relu_block_.run(
          REFV3_LAYER1_ID,
          ch_l1_ffn_linear0_in,
          ch_l1_ffn_linear0_to_linear1)) {
      return false;
    }

    if (!linear1_residual_block_.run(
          REFV3_LAYER1_ID,
          ch_l1_ffn_linear0_to_linear1,
          residual_fifo_l1,
          out_token_ch)) {
      return false;
    }

    return true;
  }

private:
  RefV3FfnLinear0ReluBlock linear0_relu_block_;
  RefV3FfnLinear1ResidualBlock linear1_residual_block_;
};

} // namespace ref_v3
} // namespace aecct_ref
