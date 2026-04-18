#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Payload.h"

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

class RefV3AttenKvBlock {
public:
  RefV3AttenKvBlock();

  // Catapult class-based interface entry for hierarchical block.
  // CCS_BLOCK added for SCVerify/Catapult hierarchy friendliness.
#pragma hls_design interface
  bool CCS_BLOCK(run)(int lid,
                      ac_channel<RefV3AttentionTokenVectorPayload>& in_x_token_ch,
                      ac_channel<RefV3AttentionKPayload>& out_k_payload_ch,
                      ac_channel<RefV3AttentionVPayload>& out_v_payload_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
