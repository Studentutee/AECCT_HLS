#include "../../include/ref_v3/RefV3WeightsFp16LocalOnly.h"
#include "../../include/ref_v3/RefV3Config.h"

#if !defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
#include "weights.h"
#endif

namespace aecct_ref {
namespace ref_v3 {
namespace {

enum : int {
  REFV3_PREPROC_EMBED_D_LOCAL_ONLY = 24,
  REFV3_PREPROC_LPE_D_LOCAL_ONLY = REFV3_D_MODEL - REFV3_PREPROC_EMBED_D_LOCAL_ONLY,
  REFV3_SRC_EMBED_NUMEL_LOCAL_ONLY = REFV3_TOKENS_T * REFV3_PREPROC_EMBED_D_LOCAL_ONLY,
  REFV3_LPE_TOKEN_NUMEL_LOCAL_ONLY = REFV3_TOKENS_T * REFV3_PREPROC_LPE_D_LOCAL_ONLY,
  REFV3_ATTN_WEIGHT_NUMEL_LOCAL_ONLY = REFV3_D_MODEL * REFV3_D_MODEL,
  REFV3_ATTN_BIAS_NUMEL_LOCAL_ONLY = REFV3_D_MODEL,
  REFV3_FFN_W1_WEIGHT_NUMEL_LOCAL_ONLY = REFV3_FF_DIM * REFV3_D_MODEL,
  REFV3_FFN_W1_BIAS_NUMEL_LOCAL_ONLY = REFV3_FF_DIM,
  REFV3_FFN_W2_WEIGHT_NUMEL_LOCAL_ONLY = REFV3_D_MODEL * REFV3_FF_DIM,
  REFV3_FFN_W2_BIAS_NUMEL_LOCAL_ONLY = REFV3_D_MODEL,
  REFV3_LN_AFFINE_NUMEL_LOCAL_ONLY = REFV3_D_MODEL,
  REFV3_MID_END_NORM_NUMEL_LOCAL_ONLY = REFV3_D_MODEL,
  REFV3_FINAL_EMBED_NUMEL_LOCAL_ONLY = REFV3_D_MODEL,
  REFV3_OUT_FC_WEIGHT_NUMEL_LOCAL_ONLY = REFV3_VAR_N * REFV3_TOKENS_T,
  REFV3_OUT_FC_BIAS_NUMEL_LOCAL_ONLY = REFV3_VAR_N,
  REFV3_SRC_MASK_NUMEL_LOCAL_ONLY = REFV3_TOKENS_T * REFV3_TOKENS_T,
  REFV3_H_PARITY_CHECK_N_LOCAL_ONLY = REFV3_TOKENS_T - REFV3_VAR_N,
  REFV3_H_PARITY_NUMEL_LOCAL_ONLY = REFV3_H_PARITY_CHECK_N_LOCAL_ONLY * REFV3_VAR_N
};

static_assert(REFV3_PREPROC_LPE_D_LOCAL_ONLY > 0, "REFV3_PREPROC_LPE_D_LOCAL_ONLY must be positive");

#if !defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
static_assert(
  REFV3_SRC_EMBED_NUMEL_LOCAL_ONLY == w_src_embed_numel,
  "REFV3_SRC_EMBED_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_LPE_TOKEN_NUMEL_LOCAL_ONLY == w_lpe_token_numel,
  "REFV3_LPE_TOKEN_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_ATTN_WEIGHT_NUMEL_LOCAL_ONLY == w_decoder_layers_0_self_attn_linears_0_weight_numel,
  "REFV3_ATTN_WEIGHT_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_ATTN_BIAS_NUMEL_LOCAL_ONLY == w_decoder_layers_0_self_attn_linears_0_bias_numel,
  "REFV3_ATTN_BIAS_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_FFN_W1_WEIGHT_NUMEL_LOCAL_ONLY == w_decoder_layers_0_feed_forward_w_1_weight_numel,
  "REFV3_FFN_W1_WEIGHT_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_FFN_W1_BIAS_NUMEL_LOCAL_ONLY == w_decoder_layers_0_feed_forward_w_1_bias_numel,
  "REFV3_FFN_W1_BIAS_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_FFN_W2_WEIGHT_NUMEL_LOCAL_ONLY == w_decoder_layers_0_feed_forward_w_2_weight_numel,
  "REFV3_FFN_W2_WEIGHT_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_FFN_W2_BIAS_NUMEL_LOCAL_ONLY == w_decoder_layers_0_feed_forward_w_2_bias_numel,
  "REFV3_FFN_W2_BIAS_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_LN_AFFINE_NUMEL_LOCAL_ONLY == w_decoder_layers_0_sublayer_0_norm_weight_numel,
  "REFV3_LN_AFFINE_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_MID_END_NORM_NUMEL_LOCAL_ONLY == w_decoder_norm2_weight_numel,
  "REFV3_MID_END_NORM_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_FINAL_EMBED_NUMEL_LOCAL_ONLY == w_oned_final_embed_0_weight_numel,
  "REFV3_FINAL_EMBED_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_OUT_FC_WEIGHT_NUMEL_LOCAL_ONLY == w_out_fc_weight_numel,
  "REFV3_OUT_FC_WEIGHT_NUMEL_LOCAL_ONLY mismatch");
static_assert(
  REFV3_OUT_FC_BIAS_NUMEL_LOCAL_ONLY == w_out_fc_bias_numel,
  "REFV3_OUT_FC_BIAS_NUMEL_LOCAL_ONLY mismatch");
#endif

template <int N, typename SrcElemT>
static inline void copy_array_to_fp_local_only(
  const SrcElemT (&src)[N],
  refv3_fp_t (&dst)[N]) {
  REFV3_LOCAL_ONLY_COPY_LOOP: for (int i = 0; i < N; ++i) {
    dst[i] = refv3_fp_from_scalar(src[i]);
  }
}

template <typename SrcScalarT>
static inline refv3_fp_t copy_scalar_to_fp_local_only(const SrcScalarT& src) {
  return refv3_fp_from_scalar(src);
}

#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
// Deterministic compile-only stub values; this path is for compile surface only.
static inline refv3_fp_t refv3_stub_weight_value_local_only(int idx, int salt) {
  const int bucket = (idx + salt) % 3;
  if (bucket == 0) return refv3_fp_t(-1.0f);
  if (bucket == 1) return refv3_fp_t(0.0f);
  return refv3_fp_t(1.0f);
}

static inline refv3_fp_t refv3_stub_bias_value_local_only(int idx, int salt) {
  const int raw = ((idx + salt) % 7) - 3;
  return refv3_fp_t(static_cast<float>(raw) * 0.03125f);
}

static inline refv3_fp_t refv3_stub_scale_value_local_only(int idx, int salt) {
  const int raw = ((idx + salt) % 5);
  return refv3_fp_t(0.75f + (static_cast<float>(raw) * 0.0625f));
}
#endif

struct RefV3EmbedGraphSubsetCacheLocalOnly {
  refv3_fp_t preproc_src_embed[REFV3_SRC_EMBED_NUMEL_LOCAL_ONLY];
  refv3_fp_t preproc_lpe_token[REFV3_LPE_TOKEN_NUMEL_LOCAL_ONLY];

  RefV3EmbedGraphSubsetCacheLocalOnly() {
#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
    REFV3_STUB_COPY_SRC_EMBED_LOOP: for (int i = 0; i < REFV3_SRC_EMBED_NUMEL_LOCAL_ONLY; ++i) {
      preproc_src_embed[i] = refv3_stub_weight_value_local_only(i, 11);
    }
    REFV3_STUB_COPY_LPE_TOKEN_LOOP: for (int i = 0; i < REFV3_LPE_TOKEN_NUMEL_LOCAL_ONLY; ++i) {
      preproc_lpe_token[i] = refv3_stub_weight_value_local_only(i, 17);
    }
#else
    copy_array_to_fp_local_only(w_src_embed, preproc_src_embed);
    copy_array_to_fp_local_only(w_lpe_token, preproc_lpe_token);
#endif
  }
};

struct RefV3AttnSubsetCacheLocalOnly {
  refv3_fp_t attn_in_s_x[2];
  refv3_fp_t attn_o_s_x[2];
  refv3_fp_t attn_weight[2][4][REFV3_ATTN_WEIGHT_NUMEL_LOCAL_ONLY];
  refv3_fp_t attn_bias[2][4][REFV3_ATTN_BIAS_NUMEL_LOCAL_ONLY];

  RefV3AttnSubsetCacheLocalOnly() {
#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
    REFV3_STUB_COPY_ATTN_SX_LOOP: for (int lid = 0; lid < 2; ++lid) {
      attn_in_s_x[lid] = refv3_stub_scale_value_local_only(lid, 23);
      attn_o_s_x[lid] = refv3_stub_scale_value_local_only(lid, 29);
    }
    REFV3_STUB_COPY_ATTN_WEIGHT_LOOP: for (int lid = 0; lid < 2; ++lid) {
      for (int linear = 0; linear < 4; ++linear) {
        for (int i = 0; i < REFV3_ATTN_WEIGHT_NUMEL_LOCAL_ONLY; ++i) {
          attn_weight[lid][linear][i] = refv3_stub_weight_value_local_only(i, 31 + (lid * 4) + linear);
        }
      }
    }
    REFV3_STUB_COPY_ATTN_BIAS_LOOP: for (int lid = 0; lid < 2; ++lid) {
      for (int linear = 0; linear < 4; ++linear) {
        for (int i = 0; i < REFV3_ATTN_BIAS_NUMEL_LOCAL_ONLY; ++i) {
          attn_bias[lid][linear][i] = refv3_stub_bias_value_local_only(i, 41 + (lid * 4) + linear);
        }
      }
    }
#else
    attn_in_s_x[0] = copy_scalar_to_fp_local_only(l0_in_s_x);
    attn_in_s_x[1] = copy_scalar_to_fp_local_only(l1_in_s_x);
    attn_o_s_x[0] = copy_scalar_to_fp_local_only(l0_o_s_x);
    attn_o_s_x[1] = copy_scalar_to_fp_local_only(l1_o_s_x);

    copy_array_to_fp_local_only(w_decoder_layers_0_self_attn_linears_0_weight, attn_weight[0][0]);
    copy_array_to_fp_local_only(w_decoder_layers_0_self_attn_linears_1_weight, attn_weight[0][1]);
    copy_array_to_fp_local_only(w_decoder_layers_0_self_attn_linears_2_weight, attn_weight[0][2]);
    copy_array_to_fp_local_only(w_decoder_layers_0_self_attn_linears_3_weight, attn_weight[0][3]);
    copy_array_to_fp_local_only(w_decoder_layers_1_self_attn_linears_0_weight, attn_weight[1][0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_self_attn_linears_1_weight, attn_weight[1][1]);
    copy_array_to_fp_local_only(w_decoder_layers_1_self_attn_linears_2_weight, attn_weight[1][2]);
    copy_array_to_fp_local_only(w_decoder_layers_1_self_attn_linears_3_weight, attn_weight[1][3]);

    copy_array_to_fp_local_only(w_decoder_layers_0_self_attn_linears_0_bias, attn_bias[0][0]);
    copy_array_to_fp_local_only(w_decoder_layers_0_self_attn_linears_1_bias, attn_bias[0][1]);
    copy_array_to_fp_local_only(w_decoder_layers_0_self_attn_linears_2_bias, attn_bias[0][2]);
    copy_array_to_fp_local_only(w_decoder_layers_0_self_attn_linears_3_bias, attn_bias[0][3]);
    copy_array_to_fp_local_only(w_decoder_layers_1_self_attn_linears_0_bias, attn_bias[1][0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_self_attn_linears_1_bias, attn_bias[1][1]);
    copy_array_to_fp_local_only(w_decoder_layers_1_self_attn_linears_2_bias, attn_bias[1][2]);
    copy_array_to_fp_local_only(w_decoder_layers_1_self_attn_linears_3_bias, attn_bias[1][3]);
#endif
  }
};

struct RefV3FfnW1FamilyCacheLocalOnly {
  refv3_fp_t ffn_w1_s_x[2];
  refv3_fp_t ffn_w1_weight[2][REFV3_FFN_W1_WEIGHT_NUMEL_LOCAL_ONLY];
  refv3_fp_t ffn_w1_bias[2][REFV3_FFN_W1_BIAS_NUMEL_LOCAL_ONLY];

  RefV3FfnW1FamilyCacheLocalOnly() {
#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
    REFV3_STUB_COPY_FFN_W1_SX_LOOP: for (int lid = 0; lid < 2; ++lid) {
      ffn_w1_s_x[lid] = refv3_stub_scale_value_local_only(lid, 53);
    }
    REFV3_STUB_COPY_FFN_W1_WEIGHT_LOOP: for (int lid = 0; lid < 2; ++lid) {
      for (int i = 0; i < REFV3_FFN_W1_WEIGHT_NUMEL_LOCAL_ONLY; ++i) {
        ffn_w1_weight[lid][i] = refv3_stub_weight_value_local_only(i, 61 + lid);
      }
    }
    REFV3_STUB_COPY_FFN_W1_BIAS_LOOP: for (int lid = 0; lid < 2; ++lid) {
      for (int i = 0; i < REFV3_FFN_W1_BIAS_NUMEL_LOCAL_ONLY; ++i) {
        ffn_w1_bias[lid][i] = refv3_stub_bias_value_local_only(i, 67 + lid);
      }
    }
#else
    ffn_w1_s_x[0] = copy_scalar_to_fp_local_only(l0_ff1_s_x);
    ffn_w1_s_x[1] = copy_scalar_to_fp_local_only(l1_ff1_s_x);

    copy_array_to_fp_local_only(w_decoder_layers_0_feed_forward_w_1_weight, ffn_w1_weight[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_feed_forward_w_1_weight, ffn_w1_weight[1]);
    copy_array_to_fp_local_only(w_decoder_layers_0_feed_forward_w_1_bias, ffn_w1_bias[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_feed_forward_w_1_bias, ffn_w1_bias[1]);
#endif
  }
};

struct RefV3FfnW2FamilyCacheLocalOnly {
  refv3_fp_t ffn_w2_s_x[2];
  refv3_fp_t ffn_w2_weight[2][REFV3_FFN_W2_WEIGHT_NUMEL_LOCAL_ONLY];
  refv3_fp_t ffn_w2_bias[2][REFV3_FFN_W2_BIAS_NUMEL_LOCAL_ONLY];

  RefV3FfnW2FamilyCacheLocalOnly() {
#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
    REFV3_STUB_COPY_FFN_W2_SX_LOOP: for (int lid = 0; lid < 2; ++lid) {
      ffn_w2_s_x[lid] = refv3_stub_scale_value_local_only(lid, 71);
    }
    REFV3_STUB_COPY_FFN_W2_WEIGHT_LOOP: for (int lid = 0; lid < 2; ++lid) {
      for (int i = 0; i < REFV3_FFN_W2_WEIGHT_NUMEL_LOCAL_ONLY; ++i) {
        ffn_w2_weight[lid][i] = refv3_stub_weight_value_local_only(i, 79 + lid);
      }
    }
    REFV3_STUB_COPY_FFN_W2_BIAS_LOOP: for (int lid = 0; lid < 2; ++lid) {
      for (int i = 0; i < REFV3_FFN_W2_BIAS_NUMEL_LOCAL_ONLY; ++i) {
        ffn_w2_bias[lid][i] = refv3_stub_bias_value_local_only(i, 83 + lid);
      }
    }
#else
    ffn_w2_s_x[0] = copy_scalar_to_fp_local_only(l0_ff2_s_x);
    ffn_w2_s_x[1] = copy_scalar_to_fp_local_only(l1_ff2_s_x);

    copy_array_to_fp_local_only(w_decoder_layers_0_feed_forward_w_2_weight, ffn_w2_weight[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_feed_forward_w_2_weight, ffn_w2_weight[1]);
    copy_array_to_fp_local_only(w_decoder_layers_0_feed_forward_w_2_bias, ffn_w2_bias[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_feed_forward_w_2_bias, ffn_w2_bias[1]);
#endif
  }
};

struct RefV3LayerNormAffineFamilyCacheLocalOnly {
  refv3_fp_t ln0_weight[2][REFV3_LN_AFFINE_NUMEL_LOCAL_ONLY];
  refv3_fp_t ln0_bias[2][REFV3_LN_AFFINE_NUMEL_LOCAL_ONLY];
  refv3_fp_t ln1_weight[2][REFV3_LN_AFFINE_NUMEL_LOCAL_ONLY];
  refv3_fp_t ln1_bias[2][REFV3_LN_AFFINE_NUMEL_LOCAL_ONLY];

  RefV3LayerNormAffineFamilyCacheLocalOnly() {
#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
    REFV3_STUB_COPY_LN_AFFINE_LOOP: for (int lid = 0; lid < 2; ++lid) {
      for (int i = 0; i < REFV3_LN_AFFINE_NUMEL_LOCAL_ONLY; ++i) {
        ln0_weight[lid][i] = refv3_stub_weight_value_local_only(i, 89 + lid);
        ln0_bias[lid][i] = refv3_stub_bias_value_local_only(i, 97 + lid);
        ln1_weight[lid][i] = refv3_stub_weight_value_local_only(i, 101 + lid);
        ln1_bias[lid][i] = refv3_stub_bias_value_local_only(i, 103 + lid);
      }
    }
#else
    copy_array_to_fp_local_only(w_decoder_layers_0_sublayer_0_norm_weight, ln0_weight[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_sublayer_0_norm_weight, ln0_weight[1]);
    copy_array_to_fp_local_only(w_decoder_layers_0_sublayer_0_norm_bias, ln0_bias[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_sublayer_0_norm_bias, ln0_bias[1]);
    copy_array_to_fp_local_only(w_decoder_layers_0_sublayer_1_norm_weight, ln1_weight[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_sublayer_1_norm_weight, ln1_weight[1]);
    copy_array_to_fp_local_only(w_decoder_layers_0_sublayer_1_norm_bias, ln1_bias[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_sublayer_1_norm_bias, ln1_bias[1]);
#endif
  }
};

struct RefV3MidEndNormFamilyCacheLocalOnly {
  refv3_fp_t midnorm_weight[REFV3_MID_END_NORM_NUMEL_LOCAL_ONLY];
  refv3_fp_t midnorm_bias[REFV3_MID_END_NORM_NUMEL_LOCAL_ONLY];
  refv3_fp_t endnorm_weight[REFV3_MID_END_NORM_NUMEL_LOCAL_ONLY];
  refv3_fp_t endnorm_bias[REFV3_MID_END_NORM_NUMEL_LOCAL_ONLY];

  RefV3MidEndNormFamilyCacheLocalOnly() {
#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
    REFV3_STUB_COPY_MID_END_NORM_LOOP: for (int i = 0; i < REFV3_MID_END_NORM_NUMEL_LOCAL_ONLY; ++i) {
      midnorm_weight[i] = refv3_stub_weight_value_local_only(i, 107);
      midnorm_bias[i] = refv3_stub_bias_value_local_only(i, 109);
      endnorm_weight[i] = refv3_stub_weight_value_local_only(i, 113);
      endnorm_bias[i] = refv3_stub_bias_value_local_only(i, 127);
    }
#else
    copy_array_to_fp_local_only(w_decoder_norm2_weight, midnorm_weight);
    copy_array_to_fp_local_only(w_decoder_norm2_bias, midnorm_bias);
    copy_array_to_fp_local_only(w_decoder_norm_weight, endnorm_weight);
    copy_array_to_fp_local_only(w_decoder_norm_bias, endnorm_bias);
#endif
  }
};

struct RefV3FinalEmbedFamilyCacheLocalOnly {
  refv3_fp_t final_embed_weight[REFV3_FINAL_EMBED_NUMEL_LOCAL_ONLY];
  refv3_fp_t final_embed_bias = refv3_fp_t(0.0f);

  RefV3FinalEmbedFamilyCacheLocalOnly() {
#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
    REFV3_STUB_COPY_FINAL_EMBED_WEIGHT_LOOP: for (int i = 0; i < REFV3_FINAL_EMBED_NUMEL_LOCAL_ONLY; ++i) {
      final_embed_weight[i] = refv3_stub_weight_value_local_only(i, 131);
    }
    final_embed_bias = refv3_stub_bias_value_local_only(0, 137);
#else
    copy_array_to_fp_local_only(w_oned_final_embed_0_weight, final_embed_weight);
    final_embed_bias = copy_scalar_to_fp_local_only(w_oned_final_embed_0_bias[0]);
#endif
  }
};

struct RefV3OutFcFamilyCacheLocalOnly {
  refv3_fp_t out_fc_weight[REFV3_OUT_FC_WEIGHT_NUMEL_LOCAL_ONLY];
  refv3_fp_t out_fc_bias[REFV3_OUT_FC_BIAS_NUMEL_LOCAL_ONLY];

  RefV3OutFcFamilyCacheLocalOnly() {
#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
    REFV3_STUB_COPY_OUT_FC_WEIGHT_LOOP: for (int i = 0; i < REFV3_OUT_FC_WEIGHT_NUMEL_LOCAL_ONLY; ++i) {
      out_fc_weight[i] = refv3_stub_weight_value_local_only(i, 139);
    }
    REFV3_STUB_COPY_OUT_FC_BIAS_LOOP: for (int i = 0; i < REFV3_OUT_FC_BIAS_NUMEL_LOCAL_ONLY; ++i) {
      out_fc_bias[i] = refv3_stub_bias_value_local_only(i, 149);
    }
#else
    copy_array_to_fp_local_only(w_out_fc_weight, out_fc_weight);
    copy_array_to_fp_local_only(w_out_fc_bias, out_fc_bias);
#endif
  }
};

static inline const RefV3EmbedGraphSubsetCacheLocalOnly& refv3_embed_graph_subset_cache_fp_local_only() {
  static const RefV3EmbedGraphSubsetCacheLocalOnly cache;
  return cache;
}

static inline const RefV3AttnSubsetCacheLocalOnly& refv3_attn_subset_cache_fp_local_only() {
  static const RefV3AttnSubsetCacheLocalOnly cache;
  return cache;
}

static inline const RefV3FfnW1FamilyCacheLocalOnly& refv3_ffn_w1_family_cache_fp_local_only() {
  static const RefV3FfnW1FamilyCacheLocalOnly cache;
  return cache;
}

static inline const RefV3FfnW2FamilyCacheLocalOnly& refv3_ffn_w2_family_cache_fp_local_only() {
  static const RefV3FfnW2FamilyCacheLocalOnly cache;
  return cache;
}

static inline const RefV3LayerNormAffineFamilyCacheLocalOnly& refv3_layernorm_affine_family_cache_fp_local_only() {
  static const RefV3LayerNormAffineFamilyCacheLocalOnly cache;
  return cache;
}

static inline const RefV3MidEndNormFamilyCacheLocalOnly& refv3_mid_end_norm_family_cache_fp_local_only() {
  static const RefV3MidEndNormFamilyCacheLocalOnly cache;
  return cache;
}

static inline const RefV3FinalEmbedFamilyCacheLocalOnly& refv3_final_embed_family_cache_fp_local_only() {
  static const RefV3FinalEmbedFamilyCacheLocalOnly cache;
  return cache;
}

static inline const RefV3OutFcFamilyCacheLocalOnly& refv3_out_fc_family_cache_fp_local_only() {
  static const RefV3OutFcFamilyCacheLocalOnly cache;
  return cache;
}

static inline int refv3_layer_idx_local_only(int lid) {
  return (lid == REFV3_LAYER1_ID) ? 1 : 0;
}

static inline int refv3_attn_linear_idx_local_only(int linear_id) {
  if (linear_id <= 0) return 0;
  if (linear_id >= 3) return 3;
  return linear_id;
}

} // namespace

const refv3_fp_t* refv3_preproc_src_embed_fp_local_only() {
  const RefV3EmbedGraphSubsetCacheLocalOnly& cache = refv3_embed_graph_subset_cache_fp_local_only();
  return cache.preproc_src_embed;
}

const refv3_fp_t* refv3_preproc_lpe_token_fp_local_only() {
  const RefV3EmbedGraphSubsetCacheLocalOnly& cache = refv3_embed_graph_subset_cache_fp_local_only();
  return cache.preproc_lpe_token;
}

refv3_fp_t refv3_attn_input_s_x_fp_local_only(int lid) {
  const RefV3AttnSubsetCacheLocalOnly& cache = refv3_attn_subset_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return cache.attn_in_s_x[layer_idx];
}

refv3_fp_t refv3_attn_output_s_x_fp_local_only(int lid) {
  const RefV3AttnSubsetCacheLocalOnly& cache = refv3_attn_subset_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return cache.attn_o_s_x[layer_idx];
}

refv3_fp_t refv3_ffn_w1_s_x_fp_local_only(int lid) {
  const RefV3FfnW1FamilyCacheLocalOnly& cache = refv3_ffn_w1_family_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return cache.ffn_w1_s_x[layer_idx];
}

refv3_fp_t refv3_ffn_w2_s_x_fp_local_only(int lid) {
  const RefV3FfnW2FamilyCacheLocalOnly& cache = refv3_ffn_w2_family_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return cache.ffn_w2_s_x[layer_idx];
}

RefV3TernaryLinearParams refv3_attn_linear_params_fp_local_only(int lid, int linear_id) {
  const RefV3AttnSubsetCacheLocalOnly& cache = refv3_attn_subset_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  const int linear_idx = refv3_attn_linear_idx_local_only(linear_id);
  return refv3_make_ternary_linear_params(
    cache.attn_weight[layer_idx][linear_idx],
    cache.attn_bias[layer_idx][linear_idx]);
}

RefV3TernaryLinearParams refv3_layernorm0_params_fp_local_only(int lid) {
  const RefV3LayerNormAffineFamilyCacheLocalOnly& cache = refv3_layernorm_affine_family_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return refv3_make_ternary_linear_params(cache.ln0_weight[layer_idx], cache.ln0_bias[layer_idx]);
}

RefV3TernaryLinearParams refv3_layernorm1_params_fp_local_only(int lid) {
  const RefV3LayerNormAffineFamilyCacheLocalOnly& cache = refv3_layernorm_affine_family_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return refv3_make_ternary_linear_params(cache.ln1_weight[layer_idx], cache.ln1_bias[layer_idx]);
}

RefV3TernaryLinearParams refv3_midnorm_params_fp_local_only() {
  const RefV3MidEndNormFamilyCacheLocalOnly& cache = refv3_mid_end_norm_family_cache_fp_local_only();
  return refv3_make_ternary_linear_params(cache.midnorm_weight, cache.midnorm_bias);
}

RefV3TernaryLinearParams refv3_endnorm_params_fp_local_only() {
  const RefV3MidEndNormFamilyCacheLocalOnly& cache = refv3_mid_end_norm_family_cache_fp_local_only();
  return refv3_make_ternary_linear_params(cache.endnorm_weight, cache.endnorm_bias);
}

RefV3TernaryLinearParams refv3_ffn_w1_params_fp_local_only(int lid) {
  const RefV3FfnW1FamilyCacheLocalOnly& cache = refv3_ffn_w1_family_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return refv3_make_ternary_linear_params(
    cache.ffn_w1_weight[layer_idx],
    cache.ffn_w1_bias[layer_idx]);
}

RefV3TernaryLinearParams refv3_ffn_w2_params_fp_local_only(int lid) {
  const RefV3FfnW2FamilyCacheLocalOnly& cache = refv3_ffn_w2_family_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return refv3_make_ternary_linear_params(
    cache.ffn_w2_weight[layer_idx],
    cache.ffn_w2_bias[layer_idx]);
}

const refv3_fp_t* refv3_final_embed_weight_fp_local_only() {
  const RefV3FinalEmbedFamilyCacheLocalOnly& cache = refv3_final_embed_family_cache_fp_local_only();
  return cache.final_embed_weight;
}

refv3_fp_t refv3_final_embed_bias_fp_local_only() {
  const RefV3FinalEmbedFamilyCacheLocalOnly& cache = refv3_final_embed_family_cache_fp_local_only();
  return cache.final_embed_bias;
}

const refv3_fp_t* refv3_out_fc_weight_fp_local_only() {
  const RefV3OutFcFamilyCacheLocalOnly& cache = refv3_out_fc_family_cache_fp_local_only();
  return cache.out_fc_weight;
}

const refv3_fp_t* refv3_out_fc_bias_fp_local_only() {
  const RefV3OutFcFamilyCacheLocalOnly& cache = refv3_out_fc_family_cache_fp_local_only();
  return cache.out_fc_bias;
}

bool refv3_src_mask_bit_local_only(int q_token, int k_token) {
  if (q_token < 0 || q_token >= REFV3_TOKENS_T || k_token < 0 || k_token >= REFV3_TOKENS_T) {
    return false;
  }
#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
  return (((q_token * REFV3_TOKENS_T) + k_token + 7) % 3) == 0;
#else
  return (w_src_mask[(q_token * REFV3_TOKENS_T) + k_token].to_int() != 0);
#endif
}

bool refv3_h_parity_edge_local_only(int check_idx, int var_idx) {
  const int check_n = REFV3_TOKENS_T - REFV3_VAR_N;
  if (check_idx < 0 || check_idx >= check_n || var_idx < 0 || var_idx >= REFV3_VAR_N) {
    return false;
  }
#if defined(AECCT_REFV3_CATAPULT_COMPILE_STUB)
  return (((check_idx * REFV3_VAR_N) + var_idx + 5) % 2) != 0;
#else
  return (h_H[(check_idx * REFV3_VAR_N) + var_idx].to_int() != 0);
#endif
}

} // namespace ref_v3
} // namespace aecct_ref
