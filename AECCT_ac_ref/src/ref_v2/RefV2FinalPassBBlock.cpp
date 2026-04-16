#include "../../include/ref_v2/RefV2FinalPassBBlock.h"

#include <cmath>

#include "weights.h"

namespace aecct_ref {
namespace ref_v2 {

RefV2FinalPassBBlock::RefV2FinalPassBBlock() {}

bool RefV2FinalPassBBlock::run(ac_channel<RefV2FinalScalarTokenPayload>& in_scalar_ch,
                               ac_channel<RefV2FinalInputYPayload>& in_input_y_ch,
                               ac_channel<RefV2FinalOutputPayload>& out_payload_ch) const {
  const RefV2FinalInputYPayload input_y_payload = in_input_y_ch.read();
  if (!refv2_var_count_matches_shape(input_y_payload.var_count)) {
    return false;
  }

  ref_fp32_t logits_acc[REFV2_VAR_N];
  bool token_seen[REFV2_TOKENS_T];
  bool header_init = false;
  RefV2AttentionPayloadHeader header_ref;

  REFV2_FINALB_ACC_INIT_LOOP: for (int n = 0; n < REFV2_VAR_N; ++n) {
    logits_acc[n] = ref_fp32_t(static_cast<float>(w_out_fc_bias[n]));
  }

  REFV2_FINALB_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV2_FINALB_TOKEN_ACCUM_LOOP: for (int token_rx = 0; token_rx < REFV2_TOKENS_T; ++token_rx) {
    const RefV2FinalScalarTokenPayload scalar_payload = in_scalar_ch.read();
    if (!refv2_payload_header_matches_shape(scalar_payload.header)) {
      return false;
    }

    if (!header_init) {
      header_ref = scalar_payload.header;
      header_init = true;
    } else {
      if (scalar_payload.header.layer_id != header_ref.layer_id ||
          scalar_payload.header.token_rows != header_ref.token_rows ||
          scalar_payload.header.dim_cols != header_ref.dim_cols) {
        return false;
      }
    }

    const int token = scalar_payload.token_row.to_int();
    if (token < 0 || token >= REFV2_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV2_FINALB_LOGITS_ACCUM_LOOP: for (int n = 0; n < REFV2_VAR_N; ++n) {
      const ref_fp32_t w_nt(static_cast<float>(w_out_fc_weight[n * REFV2_TOKENS_T + token]));
      logits_acc[n] += (w_nt * scalar_payload.scalar);
    }
  }

  RefV2FinalOutputPayload out_payload;
  out_payload.var_count = ac_int<16, false>(REFV2_VAR_N);
  const ref_fp32_t zero(0.0f);

  REFV2_FINALB_LOGITS_LOOP: for (int n = 0; n < REFV2_VAR_N; ++n) {
    const ref_fp32_t acc = logits_acc[n];
    out_payload.logits[n] = acc;

    const float y_n = input_y_payload.input_y[n].to_float();
    const bool y_is_zero = (y_n == 0.0f);
    const bool y_is_negative = (!y_is_zero) && std::signbit(y_n);
    const bool acc_is_negative = (acc < zero);
    const bool pred_bit = y_is_zero ? false : (acc_is_negative ^ y_is_negative);
    out_payload.x_pred[n] = bit1_t(pred_bit ? 1 : 0);
  }

  out_payload_ch.write(out_payload);
  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
