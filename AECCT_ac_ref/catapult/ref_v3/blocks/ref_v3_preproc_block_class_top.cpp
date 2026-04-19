#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3PreprocBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
class RefV3PreprocBlockTop {
public:
  RefV3PreprocBlockTop() {}

#pragma hls_design interface
  bool CCS_BLOCK(run)(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
                      ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch,
                      ac_channel<RefV3AttentionInputPayload>& out_xwork_ch) {
    return block_.run(in_input_ch, out_token_ch, out_xwork_ch);
  }

private:
  RefV3PreprocBlock block_;
};

// Deprecated compatibility wrapper for non-Catapult smoke callers.
bool ref_v3_preproc_block_top(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
                              ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch,
                              ac_channel<RefV3AttentionInputPayload>& out_xwork_ch) {
  static RefV3PreprocBlockTop top;
  return top.run(in_input_ch, out_token_ch, out_xwork_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
