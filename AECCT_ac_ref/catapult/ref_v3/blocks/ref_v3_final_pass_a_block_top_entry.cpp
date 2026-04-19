// Deprecated block-level Catapult free-function entry.
// Kept for rollback/reference only; removed from ref_v3 block project filelists.
#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3FinalPassABlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
bool ref_v3_final_pass_a_block_top(ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                                   ac_channel<RefV3FinalScalarTokenPayload>& out_scalar_ch) {
  RefV3FinalPassABlock block;
  return block.run(in_token_ch, out_scalar_ch);
}

} // namespace ref_v3
} // namespace aecct_ref

