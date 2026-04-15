#pragma once

#include "ac_channel.h"
#include "ref_v2/RefV2Payload.h"

namespace aecct_ref {
namespace ref_v2 {

class RefV2AttenKvBlock {
public:
  RefV2AttenKvBlock();

  bool run(ac_channel<RefV2AttentionTokenVectorPayload>& in_x_token_ch,
           ac_channel<RefV2AttentionKPayload>& out_k_payload_ch,
           ac_channel<RefV2AttentionVPayload>& out_v_payload_ch) const;
};

} // namespace ref_v2
} // namespace aecct_ref
