#pragma once

#include "RefModel.h"
#include "ac_channel.h"
#include "ref_v2/RefV2Payload.h"

namespace aecct_ref {
namespace ref_v2 {

class RefV2AttenQSoftResBlock {
public:
  RefV2AttenQSoftResBlock();

  bool run(int lid,
           const RefRunConfig& run_cfg,
           ac_channel<RefV2AttentionTokenVectorPayload>& query_token_ch,
           ac_channel<RefV2AttentionKPayload>& in_k_payload_ch,
           ac_channel<RefV2AttentionVPayload>& in_v_payload_ch,
           ac_channel<RefV2AttentionTokenVectorPayload>& out_token_ch) const;
};

} // namespace ref_v2
} // namespace aecct_ref
