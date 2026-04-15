#pragma once

#include "ac_channel.h"
#include "ref_v2/RefV2Payload.h"

namespace aecct_ref {
namespace ref_v2 {

class RefV2FfnBlock {
public:
  RefV2FfnBlock();

  bool run(ac_channel<RefV2AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV2AttentionTokenVectorPayload>& out_token_ch) const;
};

} // namespace ref_v2
} // namespace aecct_ref
