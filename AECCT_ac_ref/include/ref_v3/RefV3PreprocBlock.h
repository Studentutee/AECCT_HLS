#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Payload.h"

#if defined(__has_include)
#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#endif
#endif

#ifndef CCS_BLOCK
#define CCS_BLOCK(name) name
#endif

namespace aecct_ref {
namespace ref_v3 {

class RefV3PreprocBlock {
public:
  RefV3PreprocBlock();

  // Main path: emit token stream immediately while packing side X_WORK payload locally.
  // Catapult class-based interface entry for hierarchical block.
  // CCS_BLOCK added for SCVerify/Catapult hierarchy friendliness.
#pragma hls_design interface
  bool CCS_BLOCK(run)(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
                      ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch,
                      ac_channel<RefV3AttentionInputPayload>& out_xwork_ch) const;
};

// Compatibility wrappers are kept as namespace-level helpers to avoid adding
// extra public hierarchy methods on RefV3PreprocBlock.
bool refv3_preproc_run_token_only(
  const RefV3PreprocBlock& block,
  ac_channel<RefV3PreprocInputPayload>& in_input_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch);

bool refv3_preproc_run_xwork_only(
  const RefV3PreprocBlock& block,
  ac_channel<RefV3PreprocInputPayload>& in_input_ch,
  ac_channel<RefV3AttentionInputPayload>& out_xwork_ch);

} // namespace ref_v3
} // namespace aecct_ref
