#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3PreprocBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
bool ref_v3_preproc_block_top(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
                              ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch,
                              ac_channel<RefV3AttentionInputPayload>& out_xwork_ch) {
  RefV3PreprocBlock block;
  return block.run(in_input_ch, out_token_ch, out_xwork_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
