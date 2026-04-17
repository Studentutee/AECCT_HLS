#include <cstdio>

#include "ac_channel.h"
#include "catapult/ref_v3/RefV3CatapultTop.h"
#include "ref_v3/RefV3Config.h"
#include "ref_v3/RefV3Payload.h"

#if defined(__has_include)
#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#define REFV3_TB_HAS_SCVERIFY 1
#else
#define REFV3_TB_HAS_SCVERIFY 0
#endif
#else
#define REFV3_TB_HAS_SCVERIFY 0
#endif

#if !REFV3_TB_HAS_SCVERIFY
#ifndef CCS_MAIN
#define CCS_MAIN(...) int main(__VA_ARGS__)
#endif
#ifndef CCS_RETURN
#define CCS_RETURN(x) return (x)
#endif
#endif

namespace {

static int fail(const char* msg) {
  std::printf("[ref_v3_catapult_tb][FAIL] %s\n", msg);
  return 1;
}

static aecct_ref::ref_v3::RefV3AttentionTokenVectorPayload make_input_token_payload(int token) {
  using namespace aecct_ref::ref_v3;

  RefV3AttentionTokenVectorPayload payload;
  payload.header.layer_id = ac_int<8, false>(REFV3_LAYER0_ID);
  payload.header.token_rows = ac_int<16, false>(REFV3_TOKENS_T);
  payload.header.dim_cols = ac_int<16, false>(REFV3_D_MODEL);
  payload.token_row = ac_int<16, false>(token);

  REFV3_TB_INPUT_DIM_INIT_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
    const float seed = static_cast<float>((token * REFV3_D_MODEL) + dim);
    payload.token_vec[dim] = refv3_fp_t(seed * 0.001953125f);
  }
  return payload;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
  (void)argc;
  (void)argv;

  using namespace aecct_ref::ref_v3;

  ac_channel<RefV3AttentionTokenVectorPayload> in_token_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> out_token_ch;

  REFV3_TB_WRITE_INPUT_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    in_token_ch.write(make_input_token_payload(token));
  }

  RefV3CatapultTop dut;
  if (!dut.run(in_token_ch, out_token_ch)) {
    CCS_RETURN(fail("dut.run returned false"));
  }

  bool token_seen[REFV3_TOKENS_T];
  REFV3_TB_TOKEN_SEEN_INIT_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_seen[token] = false;
  }

  REFV3_TB_READ_OUTPUT_TOKEN_LOOP: for (int token_rx = 0; token_rx < REFV3_TOKENS_T; ++token_rx) {
    const RefV3AttentionTokenVectorPayload out_payload = out_token_ch.read();
    if (!REFV3_payload_header_matches_shape(out_payload.header)) {
      CCS_RETURN(fail("output header shape mismatch"));
    }
    if (out_payload.header.layer_id.to_int() != REFV3_LAYER0_ID) {
      CCS_RETURN(fail("output layer_id mismatch"));
    }

    const int token = out_payload.token_row.to_int();
    if (token < 0 || token >= REFV3_TOKENS_T) {
      CCS_RETURN(fail("output token index out of range"));
    }
    if (token_seen[token]) {
      CCS_RETURN(fail("duplicate output token"));
    }
    token_seen[token] = true;

    REFV3_TB_VALIDATE_VALUE_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const refv3_fp_t v = out_payload.token_vec[dim];
      if (v != v) {
        CCS_RETURN(fail("output token contains NaN"));
      }
    }
  }

  REFV3_TB_VALIDATE_TOKEN_SEEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    if (!token_seen[token]) {
      CCS_RETURN(fail("missing output token"));
    }
  }

  std::printf("REFV3_CATAPULT_TB_SANITY PASS\n");
  std::printf("REFV3_CATAPULT_MODE=%d REFV3_HAS_MC_SCVERIFY_HEADER=%d\n",
              REFV3_CATAPULT_MODE,
              REFV3_HAS_MC_SCVERIFY_HEADER);
  std::printf("PASS: tb_ref_v3_catapult_scverify\n");
  CCS_RETURN(0);
}
