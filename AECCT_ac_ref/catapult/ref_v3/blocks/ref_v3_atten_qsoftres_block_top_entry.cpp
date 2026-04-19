#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "RefModel.h"
#include "ref_v3/RefV3AttenQSoftResBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
bool ref_v3_atten_qsoftres_block_top(int lid,
                                     ac_channel<RefV3AttentionInputPayload>& in_xwork_ch,
                                     ac_channel<RefV3AttentionKPayload>& in_k_payload_ch,
                                     ac_channel<RefV3AttentionVPayload>& in_v_payload_ch,
                                     ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
  RefRunConfig run_cfg{};
  RefV3AttenQSoftResBlock block;
  return block.run(lid, run_cfg, in_xwork_ch, in_k_payload_ch, in_v_payload_ch, out_token_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
