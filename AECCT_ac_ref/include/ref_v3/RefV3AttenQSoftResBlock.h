#pragma once

#include "RefModel.h"
#include "ac_channel.h"
#include "ref_v3/RefV3Payload.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3AttenQSoftResBlock {
public:
  RefV3AttenQSoftResBlock();

  bool run(int lid,
           const RefRunConfig& run_cfg,
           ac_channel<RefV3AttentionTokenVectorPayload>& query_token_ch,
           ac_channel<RefV3AttentionKPayload>& in_k_payload_ch,
           ac_channel<RefV3AttentionVPayload>& in_v_payload_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
