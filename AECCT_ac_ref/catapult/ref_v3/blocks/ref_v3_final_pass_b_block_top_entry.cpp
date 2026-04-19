// Deprecated block-level Catapult free-function entry.
// Kept for rollback/reference only; removed from ref_v3 block project filelists.
#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3FinalPassBBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
bool ref_v3_final_pass_b_block_top(ac_channel<RefV3FinalScalarTokenPayload>& in_scalar_ch,
                                   ac_channel<RefV3FinalInputYPayload>& in_input_y_ch,
                                   ac_channel<RefV3FinalOutputPayload>& out_payload_ch) {
  RefV3FinalPassBBlock block;
  return block.run(in_scalar_ch, in_input_y_ch, out_payload_ch);
}

} // namespace ref_v3
} // namespace aecct_ref

