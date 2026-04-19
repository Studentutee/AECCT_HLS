#include "tb_catapult/ref_v3/blocks/RefV3BlockSmokeCommon.h"

CCS_MAIN(int argc, char** argv) {
  (void)argc;
  (void)argv;

  using namespace aecct_ref::ref_v3;
  using namespace aecct_ref::ref_v3::tb_block_smoke;

  ac_channel<RefV3FinalScalarTokenPayload> in_scalar_ch;
  ac_channel<RefV3FinalInputYPayload> in_input_y_ch;
  ac_channel<RefV3FinalOutputPayload> out_payload_ch;

  write_scalar_stream(REFV3_LAYER1_ID, 37, in_scalar_ch);
  in_input_y_ch.write(make_final_input_y_payload(41));

  if (!ref_v3_final_pass_b_block_top(in_scalar_ch, in_input_y_ch, out_payload_ch)) {
    CCS_RETURN(fail("tb_ref_v3_final_pass_b_block_smoke", "top returned false"));
  }

  const RefV3FinalOutputPayload out_payload = out_payload_ch.read();
  if (!REFV3_var_count_matches_shape(out_payload.var_count)) {
    CCS_RETURN(fail("tb_ref_v3_final_pass_b_block_smoke", "var_count mismatch"));
  }

  int nonzero_logits = 0;
  int ones_count = 0;
  const refv3_fp_t zero(0.0f);
  REFV3_FINALB_SMOKE_VAR_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    if (fp_is_nan(out_payload.logits[n])) {
      CCS_RETURN(fail("tb_ref_v3_final_pass_b_block_smoke", "NaN observed"));
    }
    if (out_payload.logits[n] != zero) {
      ++nonzero_logits;
    }
    const int pred = out_payload.x_pred[n].to_int();
    if (pred != 0 && pred != 1) {
      CCS_RETURN(fail("tb_ref_v3_final_pass_b_block_smoke", "x_pred out of range"));
    }
    if (pred == 1) {
      ++ones_count;
    }
  }

  if (nonzero_logits == 0) {
    CCS_RETURN(fail("tb_ref_v3_final_pass_b_block_smoke", "all-zero logits observed"));
  }

  std::printf(
    "PASS: tb_ref_v3_final_pass_b_block_smoke nonzero_logits=%d xpred_one_count=%d\n",
    nonzero_logits,
    ones_count);
  CCS_RETURN(0);
}
