#pragma once

#include "ref_v3/RefV3Types.h"
#include "ac_int.h"

#include "ref_v3/RefV3Config.h"

namespace aecct_ref {
namespace ref_v3 {

struct RefV3AttentionPayloadHeader {
  ac_int<8, false> layer_id;
  ac_int<16, false> token_rows;
  ac_int<16, false> dim_cols;
};

// Token is row, d is column. token_vec[d] keeps row-major inner-most contiguous d.
struct RefV3AttentionTokenVectorPayload {
  RefV3AttentionPayloadHeader header;
  ac_int<16, false> token_row;
  refv3_fp_t token_vec[REFV3_D_MODEL];
};

// Full-matrix payload: flatten index = token_row * D + dim_col (row-major).
struct RefV3AttentionInputPayload {
  RefV3AttentionPayloadHeader header;
  refv3_fp_t x_flat[REFV3_ATTN_MATRIX_ELEMS];
};

struct RefV3AttentionKPayload {
  RefV3AttentionPayloadHeader header;
  refv3_fp_t k_flat[REFV3_ATTN_MATRIX_ELEMS];
};

struct RefV3AttentionVPayload {
  RefV3AttentionPayloadHeader header;
  refv3_fp_t v_flat[REFV3_ATTN_MATRIX_ELEMS];
};

struct RefV3AttentionOutputPayload {
  RefV3AttentionPayloadHeader header;
  refv3_fp_t out_flat[REFV3_ATTN_MATRIX_ELEMS];
};

struct RefV3PreprocInputPayload {
  ac_int<16, false> var_count;
  refv3_fp_t input_y[REFV3_VAR_N];
};

struct RefV3FinalScalarTokenPayload {
  RefV3AttentionPayloadHeader header;
  ac_int<16, false> token_row;
  refv3_fp_t scalar;
};

struct RefV3FinalInputYPayload {
  ac_int<16, false> var_count;
  refv3_fp_t input_y[REFV3_VAR_N];
};

struct RefV3FinalOutputPayload {
  ac_int<16, false> var_count;
  refv3_fp_t logits[REFV3_VAR_N];
  bit1_t x_pred[REFV3_VAR_N];
};

static inline bool REFV3_payload_header_matches_shape(const RefV3AttentionPayloadHeader& header) {
  return (header.token_rows.to_int() == REFV3_TOKENS_T) &&
         (header.dim_cols.to_int() == REFV3_D_MODEL);
}

static inline bool REFV3_var_count_matches_shape(ac_int<16, false> var_count) {
  return (var_count.to_int() == REFV3_VAR_N);
}

} // namespace ref_v3
} // namespace aecct_ref
