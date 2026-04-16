#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Payload.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3PreprocBlock {
public:
  RefV3PreprocBlock();

  bool run(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
