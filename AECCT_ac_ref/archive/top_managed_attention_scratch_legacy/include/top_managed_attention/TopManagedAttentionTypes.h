#pragma once

#include "RefStep0ShapeBridge.h"
#include "RefTypes.h"
#include "ac_channel.h"
#include "ac_int.h"

namespace aecct_ref {
namespace top_managed_attention {

// Shape SSOT bridge for this future blockized track.
// token = Row, dim = Column, flatten = row-major: idx = token * D_MODEL + dim.
static const int TMATTN_TOKENS_T = ModelShapes::T_TOKENS;
static const int TMATTN_D_MODEL = ModelShapes::D_MODEL;
static const int TMATTN_HEADS = ModelShapes::N_HEADS;
static const int TMATTN_D_HEAD = ModelShapes::D_HEAD;
static const int TMATTN_LAYER0_ID = 0;
static const int TMATTN_LAYER1_ID = 1;

static_assert(TMATTN_TOKENS_T > 0, "TMATTN_TOKENS_T must be positive");
static_assert(TMATTN_D_MODEL > 0, "TMATTN_D_MODEL must be positive");
static_assert(TMATTN_HEADS > 0, "TMATTN_HEADS must be positive");
static_assert(TMATTN_D_HEAD > 0, "TMATTN_D_HEAD must be positive");
static_assert(TMATTN_HEADS * TMATTN_D_HEAD == TMATTN_D_MODEL,
              "heads * d_head must match d_model");

struct AttentionMatrixHeader {
  ac_int<8, false> layer_id;
  ac_int<16, false> token_rows;
  ac_int<16, false> dim_cols;
};

struct AttentionInputMatrixPayload {
  AttentionMatrixHeader header;
  ref_fp32_t matrix[TMATTN_TOKENS_T][TMATTN_D_MODEL];
};

struct AttentionQueryMatrixPayload {
  AttentionMatrixHeader header;
  ref_fp32_t matrix[TMATTN_TOKENS_T][TMATTN_D_MODEL];
};

struct AttentionKMatrixPayload {
  AttentionMatrixHeader header;
  ref_fp32_t matrix[TMATTN_TOKENS_T][TMATTN_D_MODEL];
};

struct AttentionVMatrixPayload {
  AttentionMatrixHeader header;
  ref_fp32_t matrix[TMATTN_TOKENS_T][TMATTN_D_MODEL];
};

struct AttentionQSoftResOutputMatrixPayload {
  AttentionMatrixHeader header;
  ref_fp32_t matrix[TMATTN_TOKENS_T][TMATTN_D_MODEL];
};

typedef ac_channel<AttentionInputMatrixPayload> attention_input_payload_ch_t;
typedef ac_channel<AttentionQueryMatrixPayload> attention_query_payload_ch_t;
typedef ac_channel<AttentionKMatrixPayload> attention_k_payload_ch_t;
typedef ac_channel<AttentionVMatrixPayload> attention_v_payload_ch_t;
typedef ac_channel<AttentionQSoftResOutputMatrixPayload> attention_qsoftres_output_payload_ch_t;

static inline int flatten_row_major_index(int token_row, int dim_col) {
  return (token_row * TMATTN_D_MODEL) + dim_col;
}

static inline bool payload_header_matches_shape(const AttentionMatrixHeader& header) {
  return (header.token_rows.to_int() == TMATTN_TOKENS_T) &&
         (header.dim_cols.to_int() == TMATTN_D_MODEL);
}

} // namespace top_managed_attention
} // namespace aecct_ref
