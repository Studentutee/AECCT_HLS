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

class RefV3LayerNormBlock {
public:
  RefV3LayerNormBlock();

  // Catapult class-based interface entry for hierarchical block.
  // CCS_BLOCK added for SCVerify/Catapult hierarchy friendliness.
#pragma hls_design interface
  bool CCS_BLOCK(run)(int lid,
                      const RefRunConfig& run_cfg,
                      ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                      ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const;

  // Direct-parameter entry for non-layer-local norm sites (for example decoder.norm2 mid norm).
  bool run_with_params(int expected_layer_id,
                       const RefV3TernaryLinearParams& ln_params,
                       const RefRunConfig& run_cfg,
                       ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                       ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
