#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "RefModel.h"
#include "ref_v3/RefV3LayerNormBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
bool ref_v3_layernorm_block_top(int lid,
                                ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                                ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
  RefRunConfig run_cfg{};
  RefV3LayerNormBlock block;
  return block.run(lid, run_cfg, in_token_ch, out_token_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
