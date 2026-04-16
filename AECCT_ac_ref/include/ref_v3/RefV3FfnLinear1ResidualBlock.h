#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3FfnLinear0ReluBlock.h"
#include "ref_v3/RefV3Payload.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3FfnLinear1ResidualBlock {
public:
  RefV3FfnLinear1ResidualBlock();

  bool run(int lid,
           ac_channel<RefV3FfnHiddenTokenPayload>& in_hidden_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& in_residual_token_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
