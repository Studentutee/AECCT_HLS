#pragma once

#include "ac_channel.h"
#include "ref_v2/RefV2FfnLinear0ReluBlock.h"
#include "ref_v2/RefV2Payload.h"

namespace aecct_ref {
namespace ref_v2 {

class RefV2FfnLinear1ResidualBlock {
public:
  RefV2FfnLinear1ResidualBlock();

  bool run(ac_channel<RefV2FfnHiddenTokenPayload>& in_hidden_ch,
           ac_channel<RefV2AttentionTokenVectorPayload>& in_residual_token_ch,
           ac_channel<RefV2AttentionTokenVectorPayload>& out_token_ch) const;
};

} // namespace ref_v2
} // namespace aecct_ref
