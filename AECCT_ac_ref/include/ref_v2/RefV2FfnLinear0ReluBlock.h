#pragma once

#include "ac_channel.h"
#include "ref_v2/RefV2Payload.h"

namespace aecct_ref {
namespace ref_v2 {

struct RefV2FfnHiddenTokenPayload {
  RefV2AttentionPayloadHeader header;
  ac_int<16, false> token_row;
  ref_fp32_t hidden_vec[REFV2_FF_DIM];
};

class RefV2FfnLinear0ReluBlock {
public:
  RefV2FfnLinear0ReluBlock();

  bool run(ac_channel<RefV2AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV2FfnHiddenTokenPayload>& out_hidden_ch) const;
};

} // namespace ref_v2
} // namespace aecct_ref
