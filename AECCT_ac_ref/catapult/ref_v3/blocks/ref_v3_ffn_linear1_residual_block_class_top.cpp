#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3FfnLinear1ResidualBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
class RefV3FfnLinear1ResidualBlockTop {
public:
  RefV3FfnLinear1ResidualBlockTop() {}

#pragma hls_design interface
  bool CCS_BLOCK(run)(int lid,
                      ac_channel<RefV3FfnHiddenTokenPayload>& in_hidden_ch,
                      ac_channel<RefV3AttentionTokenVectorPayload>& in_residual_token_ch,
                      ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    return block_.run(lid, in_hidden_ch, in_residual_token_ch, out_token_ch);
  }

private:
  RefV3FfnLinear1ResidualBlock block_;
};

// Deprecated compatibility wrapper for non-Catapult smoke callers.
bool ref_v3_ffn_linear1_residual_block_top(
  int lid,
  ac_channel<RefV3FfnHiddenTokenPayload>& in_hidden_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& in_residual_token_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
  static RefV3FfnLinear1ResidualBlockTop top;
  return top.run(lid, in_hidden_ch, in_residual_token_ch, out_token_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
