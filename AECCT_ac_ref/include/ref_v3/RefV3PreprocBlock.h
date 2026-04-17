#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Payload.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3PreprocBlock {
public:
  RefV3PreprocBlock();

  // Main path: emit token stream immediately while packing side X_WORK payload locally.
  bool run(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch,
           ac_channel<RefV3AttentionInputPayload>& out_xwork_ch) const;

  // Compatibility wrapper: keep token-only callers available.
  bool run(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const;

  // Compatibility wrapper: keep xwork-only callers available.
  bool run(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
           ac_channel<RefV3AttentionInputPayload>& out_xwork_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
