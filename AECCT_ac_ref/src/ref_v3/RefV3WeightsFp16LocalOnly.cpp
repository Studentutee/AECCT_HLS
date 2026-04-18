#include "../../include/ref_v3/RefV3WeightsFp16LocalOnly.h"
#include "../../include/ref_v3/RefV3Config.h"

#include "weights.h"

namespace aecct_ref {
namespace ref_v3 {
namespace {

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

struct RefV3WeightsFp16CacheLocalOnly {
  refv3_fp_t preproc_src_embed[w_src_embed_numel];
  refv3_fp_t preproc_lpe_token[w_lpe_token_numel];

  refv3_fp_t attn_in_s_x[2];
  refv3_fp_t attn_o_s_x[2];
  refv3_fp_t ffn_w1_s_x[2];
  refv3_fp_t ffn_w2_s_x[2];

  refv3_fp_t attn_weight[2][4][w_decoder_layers_0_self_attn_linears_0_weight_numel];
  refv3_fp_t attn_bias[2][4][w_decoder_layers_0_self_attn_linears_0_bias_numel];
  refv3_fp_t attn_s_w[2][4];

  refv3_fp_t ffn_w1_weight[2][w_decoder_layers_0_feed_forward_w_1_weight_numel];
  refv3_fp_t ffn_w1_bias[2][w_decoder_layers_0_feed_forward_w_1_bias_numel];
  refv3_fp_t ffn_w1_s_w[2];

  refv3_fp_t ffn_w2_weight[2][w_decoder_layers_0_feed_forward_w_2_weight_numel];
  refv3_fp_t ffn_w2_bias[2][w_decoder_layers_0_feed_forward_w_2_bias_numel];
  refv3_fp_t ffn_w2_s_w[2];

  refv3_fp_t ln0_weight[2][w_decoder_layers_0_sublayer_0_norm_weight_numel];
  refv3_fp_t ln0_bias[2][w_decoder_layers_0_sublayer_0_norm_bias_numel];
  refv3_fp_t midnorm_weight[w_decoder_norm2_weight_numel];
  refv3_fp_t midnorm_bias[w_decoder_norm2_bias_numel];

  refv3_fp_t final_embed_weight[w_oned_final_embed_0_weight_numel];
  refv3_fp_t final_embed_bias;

  refv3_fp_t out_fc_weight[w_out_fc_weight_numel];
  refv3_fp_t out_fc_bias[w_out_fc_bias_numel];

  RefV3WeightsFp16CacheLocalOnly() {
    copy_array_to_fp_local_only(w_src_embed, preproc_src_embed);
    copy_array_to_fp_local_only(w_lpe_token, preproc_lpe_token);

    attn_in_s_x[0] = copy_scalar_to_fp_local_only(l0_in_s_x);
    attn_in_s_x[1] = copy_scalar_to_fp_local_only(l1_in_s_x);
    attn_o_s_x[0] = copy_scalar_to_fp_local_only(l0_o_s_x);
    attn_o_s_x[1] = copy_scalar_to_fp_local_only(l1_o_s_x);
    ffn_w1_s_x[0] = copy_scalar_to_fp_local_only(l0_ff1_s_x);
    ffn_w1_s_x[1] = copy_scalar_to_fp_local_only(l1_ff1_s_x);
    ffn_w2_s_x[0] = copy_scalar_to_fp_local_only(l0_ff2_s_x);
    ffn_w2_s_x[1] = copy_scalar_to_fp_local_only(l1_ff2_s_x);

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

    attn_s_w[0][0] = copy_scalar_to_fp_local_only(w_decoder_layers_0_self_attn_linears_0_s_w[0]);
    attn_s_w[0][1] = copy_scalar_to_fp_local_only(w_decoder_layers_0_self_attn_linears_1_s_w[0]);
    attn_s_w[0][2] = copy_scalar_to_fp_local_only(w_decoder_layers_0_self_attn_linears_2_s_w[0]);
    attn_s_w[0][3] = copy_scalar_to_fp_local_only(w_decoder_layers_0_self_attn_linears_3_s_w[0]);
    attn_s_w[1][0] = copy_scalar_to_fp_local_only(w_decoder_layers_1_self_attn_linears_0_s_w[0]);
    attn_s_w[1][1] = copy_scalar_to_fp_local_only(w_decoder_layers_1_self_attn_linears_1_s_w[0]);
    attn_s_w[1][2] = copy_scalar_to_fp_local_only(w_decoder_layers_1_self_attn_linears_2_s_w[0]);
    attn_s_w[1][3] = copy_scalar_to_fp_local_only(w_decoder_layers_1_self_attn_linears_3_s_w[0]);

    copy_array_to_fp_local_only(w_decoder_layers_0_feed_forward_w_1_weight, ffn_w1_weight[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_feed_forward_w_1_weight, ffn_w1_weight[1]);
    copy_array_to_fp_local_only(w_decoder_layers_0_feed_forward_w_1_bias, ffn_w1_bias[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_feed_forward_w_1_bias, ffn_w1_bias[1]);
    ffn_w1_s_w[0] = copy_scalar_to_fp_local_only(w_decoder_layers_0_feed_forward_w_1_s_w[0]);
    ffn_w1_s_w[1] = copy_scalar_to_fp_local_only(w_decoder_layers_1_feed_forward_w_1_s_w[0]);

    copy_array_to_fp_local_only(w_decoder_layers_0_feed_forward_w_2_weight, ffn_w2_weight[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_feed_forward_w_2_weight, ffn_w2_weight[1]);
    copy_array_to_fp_local_only(w_decoder_layers_0_feed_forward_w_2_bias, ffn_w2_bias[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_feed_forward_w_2_bias, ffn_w2_bias[1]);
    ffn_w2_s_w[0] = copy_scalar_to_fp_local_only(w_decoder_layers_0_feed_forward_w_2_s_w[0]);
    ffn_w2_s_w[1] = copy_scalar_to_fp_local_only(w_decoder_layers_1_feed_forward_w_2_s_w[0]);

    copy_array_to_fp_local_only(w_decoder_layers_0_sublayer_0_norm_weight, ln0_weight[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_sublayer_0_norm_weight, ln0_weight[1]);
    copy_array_to_fp_local_only(w_decoder_layers_0_sublayer_0_norm_bias, ln0_bias[0]);
    copy_array_to_fp_local_only(w_decoder_layers_1_sublayer_0_norm_bias, ln0_bias[1]);
    copy_array_to_fp_local_only(w_decoder_norm2_weight, midnorm_weight);
    copy_array_to_fp_local_only(w_decoder_norm2_bias, midnorm_bias);

    copy_array_to_fp_local_only(w_oned_final_embed_0_weight, final_embed_weight);
    final_embed_bias = copy_scalar_to_fp_local_only(w_oned_final_embed_0_bias[0]);

    copy_array_to_fp_local_only(w_out_fc_weight, out_fc_weight);
    copy_array_to_fp_local_only(w_out_fc_bias, out_fc_bias);
  }
};

static inline const RefV3WeightsFp16CacheLocalOnly& refv3_weight_cache_fp_local_only() {
  static const RefV3WeightsFp16CacheLocalOnly cache;
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
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  return cache.preproc_src_embed;
}

const refv3_fp_t* refv3_preproc_lpe_token_fp_local_only() {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  return cache.preproc_lpe_token;
}

refv3_fp_t refv3_attn_input_s_x_fp_local_only(int lid) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return cache.attn_in_s_x[layer_idx];
}

refv3_fp_t refv3_attn_output_s_x_fp_local_only(int lid) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return cache.attn_o_s_x[layer_idx];
}

refv3_fp_t refv3_ffn_w1_s_x_fp_local_only(int lid) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return cache.ffn_w1_s_x[layer_idx];
}

refv3_fp_t refv3_ffn_w2_s_x_fp_local_only(int lid) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return cache.ffn_w2_s_x[layer_idx];
}

RefV3TernaryLinearParams refv3_attn_linear_params_fp_local_only(int lid, int linear_id) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  const int linear_idx = refv3_attn_linear_idx_local_only(linear_id);
  return refv3_make_ternary_linear_params(
    cache.attn_weight[layer_idx][linear_idx],
    cache.attn_bias[layer_idx][linear_idx]);
}

refv3_fp_t refv3_attn_linear_s_w_fp_local_only(int lid, int linear_id) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  const int linear_idx = refv3_attn_linear_idx_local_only(linear_id);
  return cache.attn_s_w[layer_idx][linear_idx];
}

RefV3TernaryLinearParams refv3_layernorm0_params_fp_local_only(int lid) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return refv3_make_ternary_linear_params(cache.ln0_weight[layer_idx], cache.ln0_bias[layer_idx]);
}

RefV3TernaryLinearParams refv3_midnorm_params_fp_local_only() {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  return refv3_make_ternary_linear_params(cache.midnorm_weight, cache.midnorm_bias);
}

RefV3TernaryLinearParams refv3_ffn_w1_params_fp_local_only(int lid) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return refv3_make_ternary_linear_params(
    cache.ffn_w1_weight[layer_idx],
    cache.ffn_w1_bias[layer_idx]);
}

RefV3TernaryLinearParams refv3_ffn_w2_params_fp_local_only(int lid) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return refv3_make_ternary_linear_params(
    cache.ffn_w2_weight[layer_idx],
    cache.ffn_w2_bias[layer_idx]);
}

const refv3_fp_t* refv3_final_embed_weight_fp_local_only() {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  return cache.final_embed_weight;
}

refv3_fp_t refv3_final_embed_bias_fp_local_only() {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  return cache.final_embed_bias;
}

const refv3_fp_t* refv3_out_fc_weight_fp_local_only() {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  return cache.out_fc_weight;
}

const refv3_fp_t* refv3_out_fc_bias_fp_local_only() {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  return cache.out_fc_bias;
}

refv3_fp_t refv3_ffn_w1_s_w_fp_local_only(int lid) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return cache.ffn_w1_s_w[layer_idx];
}

refv3_fp_t refv3_ffn_w2_s_w_fp_local_only(int lid) {
  const RefV3WeightsFp16CacheLocalOnly& cache = refv3_weight_cache_fp_local_only();
  const int layer_idx = refv3_layer_idx_local_only(lid);
  return cache.ffn_w2_s_w[layer_idx];
}

} // namespace ref_v3
} // namespace aecct_ref
