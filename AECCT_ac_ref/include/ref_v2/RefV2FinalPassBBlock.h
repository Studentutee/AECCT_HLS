#pragma once

#include "ac_channel.h"
#include "ref_v2/RefV2Payload.h"

namespace aecct_ref {
namespace ref_v2 {

class RefV2FinalPassBBlock {
public:
  RefV2FinalPassBBlock();

  bool run(ac_channel<RefV2FinalScalarTokenPayload>& in_scalar_ch,
           ac_channel<RefV2FinalInputYPayload>& in_input_y_ch,
           ac_channel<RefV2FinalOutputPayload>& out_payload_ch) const;
};

} // namespace ref_v2
} // namespace aecct_ref
