#include "tb_catapult/ref_v3/blocks/RefV3BlockSmokeCommon.h"

CCS_MAIN(int argc, char** argv) {
  (void)argc;
  (void)argv;

  using namespace aecct_ref::ref_v3;
  using namespace aecct_ref::ref_v3::tb_block_smoke;

  ac_channel<RefV3PreprocInputPayload> in_input_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> out_token_ch;
  ac_channel<RefV3AttentionInputPayload> out_xwork_ch;

  in_input_ch.write(make_preproc_input_payload(3));

  if (!ref_v3_preproc_block_top(in_input_ch, out_token_ch, out_xwork_ch)) {
    CCS_RETURN(fail("tb_ref_v3_preproc_block_smoke", "top returned false"));
  }

  const RefV3AttentionInputPayload xwork_payload = out_xwork_ch.read();
  if (!header_matches_shape_and_layer(xwork_payload.header, REFV3_LAYER0_ID)) {
    CCS_RETURN(fail("tb_ref_v3_preproc_block_smoke", "xwork header mismatch"));
  }

  int token_nonzero_acc = 0;
  REFV3_PREPROC_SMOKE_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    const RefV3AttentionTokenVectorPayload token_payload = out_token_ch.read();
    if (!header_matches_shape_and_layer(token_payload.header, REFV3_LAYER0_ID)) {
      CCS_RETURN(fail("tb_ref_v3_preproc_block_smoke", "token header mismatch"));
    }
    if (token_payload.token_row.to_int() != token) {
      CCS_RETURN(fail("tb_ref_v3_preproc_block_smoke", "token row mismatch"));
    }
    token_nonzero_acc += count_nonzero_token_values(token_payload);
  }

  int xwork_nonzero_acc = 0;
  const refv3_fp_t zero(0.0f);
  REFV3_PREPROC_SMOKE_XWORK_LOOP: for (int idx = 0; idx < REFV3_ATTN_MATRIX_ELEMS; ++idx) {
    if (xwork_payload.x_flat[idx] != zero) {
      ++xwork_nonzero_acc;
    }
  }

  if (token_nonzero_acc == 0 || xwork_nonzero_acc == 0) {
    CCS_RETURN(fail("tb_ref_v3_preproc_block_smoke", "all-zero output observed"));
  }

  std::printf("PASS: tb_ref_v3_preproc_block_smoke token_nonzero_acc=%d xwork_nonzero=%d\n",
              token_nonzero_acc,
              xwork_nonzero_acc);
  CCS_RETURN(0);
}
