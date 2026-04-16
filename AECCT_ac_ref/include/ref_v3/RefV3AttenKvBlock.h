#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Payload.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3AttenKvBlock {
public:
  RefV3AttenKvBlock();

  bool run(int lid,
           ac_channel<RefV3AttentionTokenVectorPayload>& in_x_token_ch,
           ac_channel<RefV3AttentionKPayload>& out_k_payload_ch,
           ac_channel<RefV3AttentionVPayload>& out_v_payload_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
