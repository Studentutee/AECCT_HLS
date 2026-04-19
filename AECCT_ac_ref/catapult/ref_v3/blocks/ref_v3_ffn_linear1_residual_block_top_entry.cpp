// Deprecated block-level Catapult free-function entry.
// Kept for rollback/reference only; removed from ref_v3 block project filelists.
#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3FfnLinear1ResidualBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
bool ref_v3_ffn_linear1_residual_block_top(
  int lid,
  ac_channel<RefV3FfnHiddenTokenPayload>& in_hidden_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& in_residual_token_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
  RefV3FfnLinear1ResidualBlock block;
  return block.run(lid, in_hidden_ch, in_residual_token_ch, out_token_ch);
}

} // namespace ref_v3
} // namespace aecct_ref

