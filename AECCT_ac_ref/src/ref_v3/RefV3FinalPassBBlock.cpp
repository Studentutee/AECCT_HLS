#include "../../include/ref_v3/RefV3FinalPassBBlock.h"

#include <cmath>

#include "weights.h"

namespace aecct_ref {
namespace ref_v3 {

RefV3FinalPassBBlock::RefV3FinalPassBBlock() {}

bool RefV3FinalPassBBlock::run(ac_channel<RefV3FinalScalarTokenPayload>& in_scalar_ch,
                               ac_channel<RefV3FinalInputYPayload>& in_input_y_ch,
                               ac_channel<RefV3FinalOutputPayload>& out_payload_ch) const {
  const RefV3FinalInputYPayload input_y_payload = in_input_y_ch.read();
  if (!REFV3_var_count_matches_shape(input_y_payload.var_count)) {
    return false;
  }

  refv3_fp_t logits_acc[REFV3_VAR_N];
  bool token_seen[REFV3_TOKENS_T];
  bool header_init = false;
  RefV3AttentionPayloadHeader header_ref;

  REFV3_FINALB_ACC_INIT_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    logits_acc[n] = refv3_fp_t(static_cast<float>(w_out_fc_bias[n]));
  }

  REFV3_FINALB_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_FINALB_TOKEN_ACCUM_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3FinalScalarTokenPayload scalar_payload = in_scalar_ch.read();
    if (!REFV3_payload_header_matches_shape(scalar_payload.header)) {
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
    if (token < 0 || token >= REFV3_TOKENS_T) {
      return false;
    }
    if (token_seen[token]) {
      return false;
    }
    token_seen[token] = true;

    REFV3_FINALB_LOGITS_ACCUM_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
      const refv3_fp_t w_nt(static_cast<float>(w_out_fc_weight[n * REFV3_TOKENS_T + token]));
      logits_acc[n] += (w_nt * scalar_payload.scalar);
    }
  }

  RefV3FinalOutputPayload out_payload;
  out_payload.var_count = ac_int<16, false>(REFV3_VAR_N);
  const refv3_fp_t zero(0.0f);

  REFV3_FINALB_LOGITS_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    const refv3_fp_t acc = logits_acc[n];
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

} // namespace ref_v3
} // namespace aecct_ref
