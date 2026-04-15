#pragma once

#include "RefTypes.h"
#include "ac_int.h"

#include "ref_v2/RefV2Config.h"

namespace aecct_ref {
namespace ref_v2 {

struct RefV2AttentionPayloadHeader {
  ac_int<8, false> layer_id;
  ac_int<16, false> token_rows;
  ac_int<16, false> dim_cols;
};

// Token is row, d is column. token_vec[d] keeps row-major inner-most contiguous d.
struct RefV2AttentionTokenVectorPayload {
  RefV2AttentionPayloadHeader header;
  ac_int<16, false> token_row;
  ref_fp32_t token_vec[REFV2_D_MODEL];
};

// Full-matrix payload: flatten index = token_row * D + dim_col (row-major).
struct RefV2AttentionInputPayload {
  RefV2AttentionPayloadHeader header;
  ref_fp32_t x_flat[REFV2_ATTN_MATRIX_ELEMS];
};

struct RefV2AttentionKPayload {
  RefV2AttentionPayloadHeader header;
  ref_fp32_t k_flat[REFV2_ATTN_MATRIX_ELEMS];
};

struct RefV2AttentionVPayload {
  RefV2AttentionPayloadHeader header;
  ref_fp32_t v_flat[REFV2_ATTN_MATRIX_ELEMS];
};

struct RefV2AttentionOutputPayload {
  RefV2AttentionPayloadHeader header;
  ref_fp32_t out_flat[REFV2_ATTN_MATRIX_ELEMS];
};

static inline bool refv2_payload_header_matches_shape(const RefV2AttentionPayloadHeader& header) {
  return (header.token_rows.to_int() == REFV2_TOKENS_T) &&
         (header.dim_cols.to_int() == REFV2_D_MODEL);
}

} // namespace ref_v2
} // namespace aecct_ref
