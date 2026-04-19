#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "RefModel.h"
#include "ref_v3/RefV3LayerNormBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
class RefV3LayerNormBlockTop {
public:
  RefV3LayerNormBlockTop() {}

#pragma hls_design interface
  bool CCS_BLOCK(run)(int lid,
                      ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                      ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    return block_.run(lid, run_cfg_, in_token_ch, out_token_ch);
  }

private:
  RefRunConfig run_cfg_;
  RefV3LayerNormBlock block_;
};

// Deprecated compatibility wrapper for non-Catapult smoke callers.
bool ref_v3_layernorm_block_top(int lid,
                                ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                                ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
  static RefV3LayerNormBlockTop top;
  return top.run(lid, in_token_ch, out_token_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
