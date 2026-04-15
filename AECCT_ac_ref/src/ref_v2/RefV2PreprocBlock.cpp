#include "../../include/ref_v2/RefV2PreprocBlock.h"

#include <cmath>

#include "weights.h"

namespace aecct_ref {
namespace ref_v2 {
namespace {

static inline ref_fp32_t abs_ref_fp32(ref_fp32_t x) {
  return ref_fp32_t(std::fabs(x.to_float()));
}

} // namespace

RefV2PreprocBlock::RefV2PreprocBlock() {}

bool RefV2PreprocBlock::run(ac_channel<RefV2PreprocInputPayload>& in_input_ch,
                            ac_channel<RefV2AttentionTokenVectorPayload>& out_token_ch) const {
  const RefV2PreprocInputPayload input_payload = in_input_ch.read();
  if (!refv2_var_count_matches_shape(input_payload.var_count)) {
    return false;
  }

  static const int REFV2_CHECK_N = REFV2_TOKENS_T - REFV2_VAR_N;
  static const int REFV2_EMBED_D = 24;
  static const int REFV2_LPE_D = REFV2_D_MODEL - REFV2_EMBED_D;

  ref_fp32_t node_feature[REFV2_TOKENS_T];
  ac_int<1, false> y_hard[REFV2_VAR_N];

  REFV2_PREPROC_VAR_FEATURE_LOOP: for (int i = 0; i < REFV2_VAR_N; ++i) {
    const ref_fp32_t y = input_payload.input_y[i];
    node_feature[i] = abs_ref_fp32(y);
    y_hard[i] = (y < ref_fp32_t(0.0f)) ? ac_int<1, false>(1) : ac_int<1, false>(0);
  }

  REFV2_PREPROC_CHECK_FEATURE_LOOP: for (int c = 0; c < REFV2_CHECK_N; ++c) {
    ac_int<1, false> parity = 0;
    REFV2_PREPROC_CHECK_PARITY_LOOP: for (int v = 0; v < REFV2_VAR_N; ++v) {
      if (h_H[c * REFV2_VAR_N + v].to_int() != 0) {
        parity = ac_int<1, false>(parity ^ y_hard[v]);
      }
    }
    node_feature[REFV2_VAR_N + c] = (parity == 0) ? ref_fp32_t(1.0f) : ref_fp32_t(-1.0f);
  }

  REFV2_PREPROC_OUT_TOKEN_LOOP: for (int token = 0; token < REFV2_TOKENS_T; ++token) {
    RefV2AttentionTokenVectorPayload token_payload;
    token_payload.header.layer_id = ac_int<8, false>(REFV2_LAYER0_ID);
    token_payload.header.token_rows = ac_int<16, false>(REFV2_TOKENS_T);
    token_payload.header.dim_cols = ac_int<16, false>(REFV2_D_MODEL);
    token_payload.token_row = ac_int<16, false>(token);

    REFV2_PREPROC_EMBED_DIM_LOOP: for (int dim = 0; dim < REFV2_EMBED_D; ++dim) {
      const float w = static_cast<float>(w_src_embed[token * REFV2_EMBED_D + dim]);
      token_payload.token_vec[dim] = node_feature[token] * ref_fp32_t(w);
    }
    REFV2_PREPROC_LPE_DIM_LOOP: for (int dim = 0; dim < REFV2_LPE_D; ++dim) {
      const float lpe = static_cast<float>(w_lpe_token[token * REFV2_LPE_D + dim]);
      token_payload.token_vec[REFV2_EMBED_D + dim] = ref_fp32_t(lpe);
    }

    out_token_ch.write(token_payload);
  }

  return true;
}

} // namespace ref_v2
} // namespace aecct_ref
