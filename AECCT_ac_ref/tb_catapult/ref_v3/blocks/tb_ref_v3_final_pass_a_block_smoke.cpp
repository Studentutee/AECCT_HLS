#include "tb_catapult/ref_v3/blocks/RefV3BlockSmokeCommon.h"

CCS_MAIN(int argc, char** argv) {
  (void)argc;
  (void)argv;

  using namespace aecct_ref::ref_v3;
  using namespace aecct_ref::ref_v3::tb_block_smoke;

  ac_channel<RefV3AttentionTokenVectorPayload> in_token_ch;
  ac_channel<RefV3FinalScalarTokenPayload> out_scalar_ch;

  write_attention_token_stream(REFV3_LAYER1_ID, 31, in_token_ch);

  if (!ref_v3_final_pass_a_block_top(in_token_ch, out_scalar_ch)) {
    CCS_RETURN(fail("tb_ref_v3_final_pass_a_block_smoke", "top returned false"));
  }

  int nonzero_count = 0;
  const refv3_fp_t zero(0.0f);
  REFV3_FINALA_SMOKE_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    const RefV3FinalScalarTokenPayload scalar_payload = out_scalar_ch.read();
    if (!header_matches_shape_and_layer(scalar_payload.header, REFV3_LAYER1_ID)) {
      CCS_RETURN(fail("tb_ref_v3_final_pass_a_block_smoke", "header mismatch"));
    }
    if (scalar_payload.token_row.to_int() != token) {
      CCS_RETURN(fail("tb_ref_v3_final_pass_a_block_smoke", "token row mismatch"));
    }
    if (fp_is_nan(scalar_payload.scalar)) {
      CCS_RETURN(fail("tb_ref_v3_final_pass_a_block_smoke", "NaN observed"));
    }
    if (scalar_payload.scalar != zero) {
      ++nonzero_count;
    }
  }

  if (nonzero_count == 0) {
    CCS_RETURN(fail("tb_ref_v3_final_pass_a_block_smoke", "all-zero scalar output observed"));
  }

  std::printf("PASS: tb_ref_v3_final_pass_a_block_smoke nonzero_count=%d\n", nonzero_count);
  CCS_RETURN(0);
}
