#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3AttenKvBlock.h"

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
class RefV3AttenKvBlockTop {
public:
  RefV3AttenKvBlockTop() {}

#pragma hls_design interface
  bool CCS_BLOCK(run)(int lid,
                      ac_channel<RefV3AttentionTokenVectorPayload>& in_x_token_ch,
                      ac_channel<RefV3AttentionKPayload>& out_k_payload_ch,
                      ac_channel<RefV3AttentionVPayload>& out_v_payload_ch) {
    return block_.run(lid, in_x_token_ch, out_k_payload_ch, out_v_payload_ch);
  }

private:
  RefV3AttenKvBlock block_;
};

// Deprecated compatibility wrapper for non-Catapult smoke callers.
bool ref_v3_atten_kv_block_top(int lid,
                               ac_channel<RefV3AttentionTokenVectorPayload>& in_x_token_ch,
                               ac_channel<RefV3AttentionKPayload>& out_k_payload_ch,
                               ac_channel<RefV3AttentionVPayload>& out_v_payload_ch) {
  static RefV3AttenKvBlockTop top;
  return top.run(lid, in_x_token_ch, out_k_payload_ch, out_v_payload_ch);
}

} // namespace ref_v3
} // namespace aecct_ref
