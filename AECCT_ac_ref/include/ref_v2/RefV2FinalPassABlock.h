#pragma once

#include "ac_channel.h"
#include "ref_v2/RefV2Payload.h"

namespace aecct_ref {
namespace ref_v2 {

class RefV2FinalPassABlock {
public:
  RefV2FinalPassABlock();

  bool run(ac_channel<RefV2AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV2FinalScalarTokenPayload>& out_scalar_ch) const;
};

} // namespace ref_v2
} // namespace aecct_ref
