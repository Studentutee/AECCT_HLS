#pragma once

#include "RefModel.h"
#include "ref_v2/RefV2Payload.h"

namespace aecct_ref {
namespace ref_v2 {

class RefV2AttenQSoftResBlock {
public:
  RefV2AttenQSoftResBlock();

  bool run(const RefRunConfig& run_cfg,
           const RefV2AttentionInputPayload& query_payload,
           const RefV2AttentionKPayload& in_k_payload,
           const RefV2AttentionVPayload& in_v_payload,
           RefV2AttentionOutputPayload* out_payload) const;
};

} // namespace ref_v2
} // namespace aecct_ref
