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

struct RefV3FfnHiddenTokenPayload {
  RefV3AttentionPayloadHeader header;
  ac_int<16, false> token_row;
  refv3_fp_t hidden_vec[REFV3_FF_DIM];
};

class RefV3FfnLinear0ReluBlock {
public:
  RefV3FfnLinear0ReluBlock();

  // Catapult class-based interface entry for hierarchical block.
  // CCS_BLOCK added for SCVerify/Catapult hierarchy friendliness.
#pragma hls_design interface
  bool CCS_BLOCK(run)(int lid,
                      ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                      ac_channel<RefV3FfnHiddenTokenPayload>& out_hidden_ch) const;
};

} // namespace ref_v3
} // namespace aecct_ref
