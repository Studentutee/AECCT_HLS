#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Payload.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3FinalPassBBlock {
public:
  RefV3FinalPassBBlock();

  bool run(ac_channel<RefV3FinalScalarTokenPayload>& in_scalar_ch,
           ac_channel<RefV3FinalInputYPayload>& in_input_y_ch,
           ac_channel<RefV3FinalOutputPayload>& out_payload_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
