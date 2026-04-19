#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3FfnLinear0ReluBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
class RefV3FfnLinear0ReluBlockTop {
public:
  RefV3FfnLinear0ReluBlockTop() {}

#pragma hls_design interface
  bool CCS_BLOCK(run)(int lid,
                      ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                      ac_channel<RefV3FfnHiddenTokenPayload>& out_hidden_ch) {
    return block_.run(lid, in_token_ch, out_hidden_ch);
  }

private:
  RefV3FfnLinear0ReluBlock block_;
};

// Deprecated compatibility wrapper for non-Catapult smoke callers.
bool ref_v3_ffn_linear0_relu_block_top(int lid,
                                        ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                                        ac_channel<RefV3FfnHiddenTokenPayload>& out_hidden_ch) {
  static RefV3FfnLinear0ReluBlockTop top;
  return top.run(lid, in_token_ch, out_hidden_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
