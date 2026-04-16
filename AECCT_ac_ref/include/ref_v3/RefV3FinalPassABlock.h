#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Payload.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3FinalPassABlock {
public:
  RefV3FinalPassABlock();

  bool run(ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV3FinalScalarTokenPayload>& out_scalar_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
