#include "tb_catapult/ref_v3/blocks/RefV3BlockSmokeCommon.h"

CCS_MAIN(int argc, char** argv) {
  (void)argc;
  (void)argv;

  using namespace aecct_ref::ref_v3;
  using namespace aecct_ref::ref_v3::tb_block_smoke;

  ac_channel<RefV3AttentionTokenVectorPayload> in_token_ch;
  ac_channel<RefV3FfnHiddenTokenPayload> out_hidden_ch;

  write_attention_token_stream(REFV3_LAYER0_ID, 19, in_token_ch);

  if (!ref_v3_ffn_linear0_relu_block_top(REFV3_LAYER0_ID, in_token_ch, out_hidden_ch)) {
    CCS_RETURN(fail("tb_ref_v3_ffn_linear0_relu_block_smoke", "top returned false"));
  }

  int nonzero_acc = 0;
  const refv3_fp_t zero(0.0f);
  REFV3_FFNL0_SMOKE_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    const RefV3FfnHiddenTokenPayload hidden_payload = out_hidden_ch.read();
    if (!header_matches_shape_and_layer(hidden_payload.header, REFV3_LAYER0_ID)) {
      CCS_RETURN(fail("tb_ref_v3_ffn_linear0_relu_block_smoke", "header mismatch"));
    }
    if (hidden_payload.token_row.to_int() != token) {
      CCS_RETURN(fail("tb_ref_v3_ffn_linear0_relu_block_smoke", "token row mismatch"));
    }
    REFV3_FFNL0_SMOKE_DIM_LOOP: for (int idx = 0; idx < REFV3_FF_DIM; ++idx) {
      if (fp_is_nan(hidden_payload.hidden_vec[idx])) {
        CCS_RETURN(fail("tb_ref_v3_ffn_linear0_relu_block_smoke", "NaN observed"));
      }
      if (hidden_payload.hidden_vec[idx] < zero) {
        CCS_RETURN(fail("tb_ref_v3_ffn_linear0_relu_block_smoke", "ReLU contract violated"));
      }
    }
    nonzero_acc += count_nonzero_hidden_values(hidden_payload);
  }

  if (nonzero_acc == 0) {
    CCS_RETURN(fail("tb_ref_v3_ffn_linear0_relu_block_smoke", "all-zero output observed"));
  }

  std::printf("PASS: tb_ref_v3_ffn_linear0_relu_block_smoke nonzero_acc=%d\n", nonzero_acc);
  CCS_RETURN(0);
}
