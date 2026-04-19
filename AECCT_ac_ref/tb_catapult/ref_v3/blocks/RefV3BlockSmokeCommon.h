#pragma once

#include <cstdio>

#include "ac_channel.h"
#include "catapult/ref_v3/blocks/RefV3BlockTops.h"
#include "ref_v3/RefV3Config.h"
#include "ref_v3/RefV3Payload.h"

#if defined(__has_include)
#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#define REFV3_BLOCK_TB_HAS_SCVERIFY 1
#else
#define REFV3_BLOCK_TB_HAS_SCVERIFY 0
#endif
#else
#define REFV3_BLOCK_TB_HAS_SCVERIFY 0
#endif

#if !REFV3_BLOCK_TB_HAS_SCVERIFY
#ifndef CCS_MAIN
#define CCS_MAIN(...) int main(__VA_ARGS__)
#endif
#ifndef CCS_RETURN
#define CCS_RETURN(x) return (x)
#endif
#endif

namespace aecct_ref {
namespace ref_v3 {
namespace tb_block_smoke {

inline int fail(const char* tb_name, const char* msg) {
  std::printf("[%s][FAIL] %s\n", tb_name, msg);
  return 1;
}

inline refv3_fp_t make_seeded_fp(int primary, int secondary, int salt) {
  const int raw = ((primary * 17) + (secondary * 13) + (salt * 7)) % 29;
  const float centered = static_cast<float>(raw - 14) * 0.0625f;
  return refv3_fp_t(centered);
}

inline void init_attention_header(int layer_id, RefV3AttentionPayloadHeader* out_header) {
  out_header->layer_id = ac_int<8, false>(layer_id);
  out_header->token_rows = ac_int<16, false>(REFV3_TOKENS_T);
  out_header->dim_cols = ac_int<16, false>(REFV3_D_MODEL);
}

inline bool header_matches_shape_and_layer(const RefV3AttentionPayloadHeader& header, int layer_id) {
  return REFV3_payload_header_matches_shape(header) && (header.layer_id.to_int() == layer_id);
}

inline RefV3PreprocInputPayload make_preproc_input_payload(int salt) {
  RefV3PreprocInputPayload payload = {};
  payload.var_count = ac_int<16, false>(REFV3_VAR_N);
  REFV3_BLOCK_TB_PREPROC_INPUT_INIT_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    payload.input_y[n] = make_seeded_fp(n, 0, salt);
  }
  return payload;
}

inline RefV3AttentionTokenVectorPayload make_attention_token_payload(int layer_id, int token_row, int salt) {
  RefV3AttentionTokenVectorPayload payload = {};
  init_attention_header(layer_id, &payload.header);
  payload.token_row = ac_int<16, false>(token_row);
  REFV3_BLOCK_TB_TOKEN_DIM_INIT_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
    payload.token_vec[dim] = make_seeded_fp(token_row, dim, salt);
  }
  return payload;
}

inline void write_attention_token_stream(int layer_id,
                                         int salt,
                                         ac_channel<RefV3AttentionTokenVectorPayload>& token_ch) {
  REFV3_BLOCK_TB_TOKEN_STREAM_WRITE_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    token_ch.write(make_attention_token_payload(layer_id, token, salt));
  }
}

inline RefV3AttentionInputPayload make_attention_input_payload(int layer_id, int salt) {
  RefV3AttentionInputPayload payload = {};
  init_attention_header(layer_id, &payload.header);
  REFV3_BLOCK_TB_XWORK_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_BLOCK_TB_XWORK_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const int idx = REFV3_flatten_row_major_index(token, dim);
      payload.x_flat[idx] = make_seeded_fp(token, dim, salt);
    }
  }
  return payload;
}

inline RefV3AttentionKPayload make_attention_k_payload(int layer_id, int salt) {
  RefV3AttentionKPayload payload = {};
  init_attention_header(layer_id, &payload.header);
  REFV3_BLOCK_TB_K_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_BLOCK_TB_K_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const int idx = REFV3_flatten_row_major_index(token, dim);
      payload.k_flat[idx] = make_seeded_fp(token, dim, salt);
    }
  }
  return payload;
}

inline RefV3AttentionVPayload make_attention_v_payload(int layer_id, int salt) {
  RefV3AttentionVPayload payload = {};
  init_attention_header(layer_id, &payload.header);
  REFV3_BLOCK_TB_V_TOKEN_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    REFV3_BLOCK_TB_V_DIM_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
      const int idx = REFV3_flatten_row_major_index(token, dim);
      payload.v_flat[idx] = make_seeded_fp(token, dim, salt);
    }
  }
  return payload;
}

inline RefV3FfnHiddenTokenPayload make_hidden_payload(int layer_id, int token_row, int salt) {
  RefV3FfnHiddenTokenPayload payload = {};
  init_attention_header(layer_id, &payload.header);
  payload.token_row = ac_int<16, false>(token_row);
  REFV3_BLOCK_TB_HIDDEN_DIM_INIT_LOOP: for (int idx = 0; idx < REFV3_FF_DIM; ++idx) {
    payload.hidden_vec[idx] = make_seeded_fp(token_row, idx, salt);
  }
  return payload;
}

inline void write_hidden_stream(int layer_id, int salt, ac_channel<RefV3FfnHiddenTokenPayload>& hidden_ch) {
  REFV3_BLOCK_TB_HIDDEN_STREAM_WRITE_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    hidden_ch.write(make_hidden_payload(layer_id, token, salt));
  }
}

inline RefV3FinalScalarTokenPayload make_scalar_payload(int layer_id, int token_row, int salt) {
  RefV3FinalScalarTokenPayload payload = {};
  init_attention_header(layer_id, &payload.header);
  payload.token_row = ac_int<16, false>(token_row);
  payload.scalar = make_seeded_fp(token_row, 0, salt);
  return payload;
}

inline void write_scalar_stream(int layer_id,
                                int salt,
                                ac_channel<RefV3FinalScalarTokenPayload>& scalar_ch) {
  REFV3_BLOCK_TB_SCALAR_STREAM_WRITE_LOOP: for (int token = 0; token < REFV3_TOKENS_T; ++token) {
    scalar_ch.write(make_scalar_payload(layer_id, token, salt));
  }
}

inline RefV3FinalInputYPayload make_final_input_y_payload(int salt) {
  RefV3FinalInputYPayload payload = {};
  payload.var_count = ac_int<16, false>(REFV3_VAR_N);
  REFV3_BLOCK_TB_FINAL_INPUT_Y_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
    payload.input_y[n] = make_seeded_fp(n, 0, salt);
  }
  return payload;
}

inline bool fp_is_nan(refv3_fp_t v) {
  return !(v == v);
}

inline int count_nonzero_token_values(const RefV3AttentionTokenVectorPayload& payload) {
  int nonzero_count = 0;
  const refv3_fp_t zero(0.0f);
  REFV3_BLOCK_TB_TOKEN_NONZERO_LOOP: for (int dim = 0; dim < REFV3_D_MODEL; ++dim) {
    if (payload.token_vec[dim] != zero) {
      ++nonzero_count;
    }
  }
  return nonzero_count;
}

inline int count_nonzero_hidden_values(const RefV3FfnHiddenTokenPayload& payload) {
  int nonzero_count = 0;
  const refv3_fp_t zero(0.0f);
  REFV3_BLOCK_TB_HIDDEN_NONZERO_LOOP: for (int idx = 0; idx < REFV3_FF_DIM; ++idx) {
    if (payload.hidden_vec[idx] != zero) {
      ++nonzero_count;
    }
  }
  return nonzero_count;
}

} // namespace tb_block_smoke
} // namespace ref_v3
} // namespace aecct_ref
