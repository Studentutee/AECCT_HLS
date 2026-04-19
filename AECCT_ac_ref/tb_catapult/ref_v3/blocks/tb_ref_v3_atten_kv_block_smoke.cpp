#include "tb_catapult/ref_v3/blocks/RefV3BlockSmokeCommon.h"

CCS_MAIN(int argc, char** argv) {
  (void)argc;
  (void)argv;

  using namespace aecct_ref::ref_v3;
  using namespace aecct_ref::ref_v3::tb_block_smoke;

  ac_channel<RefV3AttentionTokenVectorPayload> in_token_ch;
  ac_channel<RefV3AttentionKPayload> out_k_payload_ch;
  ac_channel<RefV3AttentionVPayload> out_v_payload_ch;

  write_attention_token_stream(REFV3_LAYER0_ID, 5, in_token_ch);

  if (!ref_v3_atten_kv_block_top(REFV3_LAYER0_ID, in_token_ch, out_k_payload_ch, out_v_payload_ch)) {
    CCS_RETURN(fail("tb_ref_v3_atten_kv_block_smoke", "top returned false"));
  }

  const RefV3AttentionKPayload k_payload = out_k_payload_ch.read();
  const RefV3AttentionVPayload v_payload = out_v_payload_ch.read();
  if (!header_matches_shape_and_layer(k_payload.header, REFV3_LAYER0_ID)) {
    CCS_RETURN(fail("tb_ref_v3_atten_kv_block_smoke", "k header mismatch"));
  }
  if (!header_matches_shape_and_layer(v_payload.header, REFV3_LAYER0_ID)) {
    CCS_RETURN(fail("tb_ref_v3_atten_kv_block_smoke", "v header mismatch"));
  }

  int nonzero_k = 0;
  int nonzero_v = 0;
  const refv3_fp_t zero(0.0f);
  REFV3_ATTKV_SMOKE_FLAT_LOOP: for (int idx = 0; idx < REFV3_ATTN_MATRIX_ELEMS; ++idx) {
    if (fp_is_nan(k_payload.k_flat[idx]) || fp_is_nan(v_payload.v_flat[idx])) {
      CCS_RETURN(fail("tb_ref_v3_atten_kv_block_smoke", "NaN observed"));
    }
    if (k_payload.k_flat[idx] != zero) {
      ++nonzero_k;
    }
    if (v_payload.v_flat[idx] != zero) {
      ++nonzero_v;
    }
  }

  if (nonzero_k == 0 || nonzero_v == 0) {
    CCS_RETURN(fail("tb_ref_v3_atten_kv_block_smoke", "all-zero output observed"));
  }

  std::printf("PASS: tb_ref_v3_atten_kv_block_smoke nonzero_k=%d nonzero_v=%d\n", nonzero_k, nonzero_v);
  CCS_RETURN(0);
}
