// Deprecated block-level Catapult free-function entry.
// Kept for rollback/reference only; removed from ref_v3 block project filelists.
#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3AttenKvBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
bool ref_v3_atten_kv_block_top(int lid,
                               ac_channel<RefV3AttentionTokenVectorPayload>& in_x_token_ch,
                               ac_channel<RefV3AttentionKPayload>& out_k_payload_ch,
                               ac_channel<RefV3AttentionVPayload>& out_v_payload_ch) {
  RefV3AttenKvBlock block;
  return block.run(lid, in_x_token_ch, out_k_payload_ch, out_v_payload_ch);
}

} // namespace ref_v3
} // namespace aecct_ref

