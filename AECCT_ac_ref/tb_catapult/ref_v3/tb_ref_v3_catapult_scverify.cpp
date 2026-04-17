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

static aecct_ref::ref_v3::RefV3PreprocInputPayload make_preproc_input_payload() {
  using namespace aecct_ref::ref_v3;

  RefV3PreprocInputPayload payload;
  payload.var_count = ac_int<16, false>(REFV3_VAR_N);

  REFV3_TB_INPUT_VAR_INIT_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    const float seed = static_cast<float>((n % 9) - 4);
    payload.input_y[n] = refv3_fp_t(seed * 0.125f);
  }
  return payload;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
  (void)argc;
  (void)argv;

  using namespace aecct_ref::ref_v3;

  ac_channel<RefV3PreprocInputPayload> in_preproc_ch;
  ac_channel<RefV3AttentionTokenVectorPayload> out_token_ch;

  in_preproc_ch.write(make_preproc_input_payload());

  RefV3CatapultTop dut;
  if (!dut.run(in_preproc_ch, out_token_ch)) {
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
    if (out_payload.header.layer_id.to_int() != REFV3_LAYER1_ID) {
      CCS_RETURN(fail("output layer_id mismatch (expected layer1)"));
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
