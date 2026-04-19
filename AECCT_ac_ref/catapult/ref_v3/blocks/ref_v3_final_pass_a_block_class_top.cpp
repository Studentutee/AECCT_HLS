#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3FinalPassABlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
class RefV3FinalPassABlockTop {
public:
  RefV3FinalPassABlockTop() {}

#pragma hls_design interface
  bool CCS_BLOCK(run)(ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                      ac_channel<RefV3FinalScalarTokenPayload>& out_scalar_ch) {
    return block_.run(in_token_ch, out_scalar_ch);
  }

private:
  RefV3FinalPassABlock block_;
};

// Deprecated compatibility wrapper for non-Catapult smoke callers.
bool ref_v3_final_pass_a_block_top(ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                                   ac_channel<RefV3FinalScalarTokenPayload>& out_scalar_ch) {
  static RefV3FinalPassABlockTop top;
  return top.run(in_token_ch, out_scalar_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
