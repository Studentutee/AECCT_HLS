#pragma once

#include "RefModel.h"
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

class RefV3AttenQSoftResBlock {
public:
  RefV3AttenQSoftResBlock();

  // Catapult class-based interface entry for hierarchical block.
  // CCS_BLOCK added for SCVerify/Catapult hierarchy friendliness.
#pragma hls_design interface
  bool CCS_BLOCK(run)(int lid,
                      const RefRunConfig& run_cfg,
                      ac_channel<RefV3AttentionInputPayload>& in_xwork_ch,
                      ac_channel<RefV3AttentionKPayload>& in_k_payload_ch,
                      ac_channel<RefV3AttentionVPayload>& in_v_payload_ch,
                      ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const;

  // Compatibility wrapper for token-stream callers; mainline should pass X_WORK payload.
  bool run(int lid,
           const RefRunConfig& run_cfg,
           ac_channel<RefV3AttentionTokenVectorPayload>& query_token_ch,
           ac_channel<RefV3AttentionKPayload>& in_k_payload_ch,
           ac_channel<RefV3AttentionVPayload>& in_v_payload_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
