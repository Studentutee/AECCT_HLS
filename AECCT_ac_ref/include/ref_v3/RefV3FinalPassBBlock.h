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

class RefV3FinalPassBBlock {
public:
  RefV3FinalPassBBlock();

  // Catapult class-based interface entry for hierarchical block.
  // CCS_BLOCK added for SCVerify/Catapult hierarchy friendliness.
#pragma hls_design interface
  bool CCS_BLOCK(run)(ac_channel<RefV3FinalScalarTokenPayload>& in_scalar_ch,
                      ac_channel<RefV3FinalInputYPayload>& in_input_y_ch,
                      ac_channel<RefV3FinalOutputPayload>& out_payload_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
