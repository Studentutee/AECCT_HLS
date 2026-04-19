#include "tb_catapult/ref_v3/blocks/RefV3BlockSmokeCommon.h"

CCS_MAIN(int argc, char** argv) {
  (void)argc;
  (void)argv;

  using namespace aecct_ref::ref_v3;
  using namespace aecct_ref::ref_v3::tb_block_smoke;

  ac_channel<RefV3FfnHiddenTokenPayload> in_hidden_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> in_residual_token_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> out_token_ch;

  write_hidden_stream(REFV3_LAYER1_ID, 23, in_hidden_ch);
  write_attention_token_stream(REFV3_LAYER1_ID, 29, in_residual_token_ch);

  if (!ref_v3_ffn_linear1_residual_block_top(
        REFV3_LAYER1_ID, in_hidden_ch, in_residual_token_ch, out_token_ch)) {
    CCS_RETURN(fail("tb_ref_v3_ffn_linear1_residual_block_smoke", "top returned false"));
  }

  int nonzero_acc = 0;
  REFV3_FFNL1_SMOKE_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    const RefV3AttentionTokenVectorPayload token_payload = out_token_ch.read();
    if (!header_matches_shape_and_layer(token_payload.header, REFV3_LAYER1_ID)) {
      CCS_RETURN(fail("tb_ref_v3_ffn_linear1_residual_block_smoke", "header mismatch"));
    }
    if (token_payload.token_row.to_int() != token) {
      CCS_RETURN(fail("tb_ref_v3_ffn_linear1_residual_block_smoke", "token row mismatch"));
    }
    REFV3_FFNL1_SMOKE_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      if (fp_is_nan(token_payload.token_vec[dim])) {
        CCS_RETURN(fail("tb_ref_v3_ffn_linear1_residual_block_smoke", "NaN observed"));
      }
    }
    nonzero_acc += count_nonzero_token_values(token_payload);
  }

  if (nonzero_acc == 0) {
    CCS_RETURN(fail("tb_ref_v3_ffn_linear1_residual_block_smoke", "all-zero output observed"));
  }

  std::printf("PASS: tb_ref_v3_ffn_linear1_residual_block_smoke nonzero_acc=%d\n", nonzero_acc);
  CCS_RETURN(0);
}
