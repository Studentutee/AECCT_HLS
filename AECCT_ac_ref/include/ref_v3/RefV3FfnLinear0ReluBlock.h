#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Payload.h"

namespace aecct_ref {
namespace ref_v3 {

struct RefV3FfnHiddenTokenPayload {
  RefV3AttentionPayloadHeader header;
  ac_int<16, false> token_row;
  refv3_fp_t hidden_vec[REFV3_FF_DIM];
};

class RefV3FfnLinear0ReluBlock {
public:
  RefV3FfnLinear0ReluBlock();

  bool run(int lid,
           ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV3FfnHiddenTokenPayload>& out_hidden_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
