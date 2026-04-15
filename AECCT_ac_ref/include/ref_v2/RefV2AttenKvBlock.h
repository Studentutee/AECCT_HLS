#pragma once

#include "ref_v2/RefV2Payload.h"

namespace aecct_ref {
namespace ref_v2 {

class RefV2AttenKvBlock {
public:
  RefV2AttenKvBlock();

  bool run(const RefV2AttentionInputPayload& in_payload,
           RefV2AttentionKPayload* out_k_payload,
           RefV2AttentionVPayload* out_v_payload) const;
};

} // namespace ref_v2
} // namespace aecct_ref
