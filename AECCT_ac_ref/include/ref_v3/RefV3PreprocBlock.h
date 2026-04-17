#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Payload.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3PreprocBlock {
public:
  RefV3PreprocBlock();

  // Preproc stage emits one full-matrix X_WORK payload for the next major stage boundary.
  bool run(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
           ac_channel<RefV3AttentionInputPayload>& out_xwork_ch) const;

  // Compatibility bridge: keeps legacy token-stream callers working.
  bool run(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
