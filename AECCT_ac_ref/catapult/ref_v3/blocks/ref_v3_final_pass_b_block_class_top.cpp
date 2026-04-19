#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3FinalPassBBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
class RefV3FinalPassBBlockTop {
public:
  RefV3FinalPassBBlockTop() {}

#pragma hls_design interface
  bool CCS_BLOCK(run)(ac_channel<RefV3FinalScalarTokenPayload>& in_scalar_ch,
                      ac_channel<RefV3FinalInputYPayload>& in_input_y_ch,
                      ac_channel<RefV3FinalOutputPayload>& out_payload_ch) {
    return block_.run(in_scalar_ch, in_input_y_ch, out_payload_ch);
  }

private:
  RefV3FinalPassBBlock block_;
};

// Deprecated compatibility wrapper for non-Catapult smoke callers.
bool ref_v3_final_pass_b_block_top(ac_channel<RefV3FinalScalarTokenPayload>& in_scalar_ch,
                                   ac_channel<RefV3FinalInputYPayload>& in_input_y_ch,
                                   ac_channel<RefV3FinalOutputPayload>& out_payload_ch) {
  static RefV3FinalPassBBlockTop top;
  return top.run(in_scalar_ch, in_input_y_ch, out_payload_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
