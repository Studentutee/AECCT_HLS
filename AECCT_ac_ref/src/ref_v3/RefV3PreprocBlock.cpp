#include "../../include/ref_v3/RefV3PreprocBlock.h"
#include "../../include/ref_v3/RefV3WeightsFp16LocalOnly.h"

#include "weights.h"

namespace aecct_ref {
namespace ref_v3 {
namespace {

static inline refv3_fp_t abs_ref_fp32(refv3_fp_t x) {
  return refv3_abs_fp(x);
}

} // namespace

RefV3PreprocBlock::RefV3PreprocBlock() {}

bool RefV3PreprocBlock::run(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
                            ac_channel<RefV3AttentionInputPayload>& out_xwork_ch) const {
  const RefV3PreprocInputPayload input_payload = in_input_ch.read();
  if (!REFV3_var_count_matches_shape(input_payload.var_count)) {
    return false;
  }

  static const int REFV3_CHECK_N = REFV3_TOKENS_T - REFV3_VAR_N;
  static const int REFV3_EMBED_D = 24;
  static const int REFV3_LPE_D = REFV3_D_MODEL - REFV3_EMBED_D;
  const refv3_fp_t* src_embed = refv3_preproc_src_embed_fp_local_only();
  const refv3_fp_t* lpe_token = refv3_preproc_lpe_token_fp_local_only();

  refv3_fp_t node_feature[REFV3_TOKENS_T];
  ac_int<1, false> y_hard[REFV3_VAR_N];
  RefV3AttentionInputPayload xwork_payload;
  xwork_payload.header.layer_id = ac_int<8, false>(REFV3_LAYER0_ID);
  xwork_payload.header.token_rows = ac_int<16, false>(REFV3_TOKENS_T);
  xwork_payload.header.dim_cols = ac_int<16, false>(REFV3_D_MODEL);

  REFV3_PREPROC_VAR_FEATURE_LOOP: for (int i = 0; i < REFV3_VAR_N; ++i) {
    const refv3_fp_t y = input_payload.input_y[i];
    node_feature[i] = abs_ref_fp32(y);
    y_hard[i] = (y < refv3_fp_t(0.0f)) ? ac_int<1, false>(1) : ac_int<1, false>(0);
  }

  REFV3_PREPROC_CHECK_FEATURE_LOOP: for (int c = 0; c < REFV3_CHECK_N; ++c) {
    ac_int<1, false> parity = 0;
    REFV3_PREPROC_CHECK_PARITY_LOOP: for (int v = 0; v < REFV3_VAR_N; ++v) {
      if (h_H[c * REFV3_VAR_N + v].to_int() != 0) {
        parity = ac_int<1, false>(parity ^ y_hard[v]);
      }
    }
    node_feature[REFV3_VAR_N + c] = (parity == 0) ? refv3_fp_t(1.0f) : refv3_fp_t(-1.0f);
  }

  REFV3_PREPROC_OUT_XWORK_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_PREPROC_EMBED_DIM_LOOP: for (int dim = 0; dim < REFV3_EMBED_D; ++dim) {
      const refv3_fp_t w = src_embed[token * REFV3_EMBED_D + dim];
      const int idx = REFV3_flatten_row_major_index(token, dim);
      xwork_payload.x_flat[idx] = node_feature[token] * w;
    }
    REFV3_PREPROC_LPE_DIM_LOOP: for (int dim = 0; dim < REFV3_LPE_D; ++dim) {
      const refv3_fp_t lpe = lpe_token[token * REFV3_LPE_D + dim];
      const int out_dim = REFV3_EMBED_D + dim;
      const int idx = REFV3_flatten_row_major_index(token, out_dim);
      xwork_payload.x_flat[idx] = lpe;
    }
  }

  out_xwork_ch.write(xwork_payload);
  return true;
}

bool RefV3PreprocBlock::run(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
                            ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) const {
  ac_channel<RefV3AttentionInputPayload> ch_xwork_local_only;
  if (!run(in_input_ch, ch_xwork_local_only)) {
    return false;
  }

  const RefV3AttentionInputPayload xwork_payload = ch_xwork_local_only.read();
  REFV3_PREPROC_BRIDGE_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    RefV3AttentionTokenVectorPayload token_payload;
    token_payload.header = xwork_payload.header;
    token_payload.token_row = ac_int<16, false>(token);

    REFV3_PREPROC_BRIDGE_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const int idx = REFV3_flatten_row_major_index(token, dim);
      token_payload.token_vec[dim] = xwork_payload.x_flat[idx];
    }
    out_token_ch.write(token_payload);
  }
  return true;
}

} // namespace ref_v3
} // namespace aecct_ref
