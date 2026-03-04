#include "RefStep0Synth.h"

#include "../include/InvSqrtApprox.h"
#include "../include/SoftmaxApprox.h"
#include "weights.h"

namespace aecct_ref {
namespace {

static const int TOKENS_T = 75;
static const int VAR_N = 63;
static const int CHECK_N = 12;
static const int D_MODEL = 32;
static const int HEADS = 8;
static const int D_HEAD = 4;
static const int FF_DIM = 128;
static const int OUT_TILE = 8;
static const int FIFO_DEPTH = 2;

static const fp32_ref_t LN_EPS = fp32_ref_t(1.0e-5f);
static const fp32_ref_t INV_D_MODEL = fp32_ref_t(0.03125f);
static const fp32_ref_t INV_SQRT_D_HEAD = fp32_ref_t(0.5f);
static const fp32_ref_t NEG_LARGE = fp32_ref_t(-1.0e30f);

static const fp32_ref_t INV_SCALE_L0_Q = fp32_ref_t(0.0071290909f);
static const fp32_ref_t INV_SCALE_L0_K = fp32_ref_t(0.0077177854f);
static const fp32_ref_t INV_SCALE_L0_V = fp32_ref_t(0.0030414062f);
static const fp32_ref_t INV_SCALE_L0_O = fp32_ref_t(0.00089474069f);
static const fp32_ref_t INV_SCALE_L0_FF1 = fp32_ref_t(0.0026840530f);
static const fp32_ref_t INV_SCALE_L0_FF2 = fp32_ref_t(0.0036128608f);

static const fp32_ref_t INV_SCALE_L1_Q = fp32_ref_t(0.0043993234f);
static const fp32_ref_t INV_SCALE_L1_K = fp32_ref_t(0.0071688984f);
static const fp32_ref_t INV_SCALE_L1_V = fp32_ref_t(0.0049903886f);
static const fp32_ref_t INV_SCALE_L1_O = fp32_ref_t(0.0043067653f);
static const fp32_ref_t INV_SCALE_L1_FF1 = fp32_ref_t(0.0024541419f);
static const fp32_ref_t INV_SCALE_L1_FF2 = fp32_ref_t(0.0055658957f);

struct LayerConfig {
  const double *w_q;
  const double *b_q;
  const double *w_k;
  const double *b_k;
  const double *w_v;
  const double *b_v;
  const double *w_o;
  const double *b_o;
  const double *w_ff1;
  const double *b_ff1;
  const double *w_ff2;
  const double *b_ff2;
  const double *ln0_w;
  const double *ln0_b;
  const double *ln1_w;
  const double *ln1_b;
  fp32_ref_t s_x_in;
  fp32_ref_t s_x_o;
  fp32_ref_t s_x_ff1;
  fp32_ref_t s_x_ff2;
  fp32_ref_t inv_q;
  fp32_ref_t inv_k;
  fp32_ref_t inv_v;
  fp32_ref_t inv_o;
  fp32_ref_t inv_ff1;
  fp32_ref_t inv_ff2;
};

static inline fp32_ref_t fp32_abs(fp32_ref_t x) {
  return (x < fp32_ref_t(0.0f)) ? (fp32_ref_t(0.0f) - x) : x;
}

static inline fp32_ref_t sign_fp32(fp32_ref_t x) {
  if (x > fp32_ref_t(0.0f)) return fp32_ref_t(1.0f);
  if (x < fp32_ref_t(0.0f)) return fp32_ref_t(-1.0f);
  return fp32_ref_t(0.0f);
}

static inline fp32_ref_t fp32_round(fp32_ref_t x) {
  return x.round();
}

static inline fp32_ref_t fp32_relu(fp32_ref_t x) {
  return (x > fp32_ref_t(0.0f)) ? x : fp32_ref_t(0.0f);
}

template <int OUT_DIM, int IN_DIM>
static inline void prefetch_quant_weight_tile(
  const double w[OUT_DIM * IN_DIM],
  fp32_ref_t inv_scale,
  int out_base,
  fp32_ref_t wbuf[OUT_TILE][IN_DIM]
) {
  for (int tile_o = 0; tile_o < OUT_TILE; ++tile_o) {
    const int out_idx = out_base + tile_o;
    for (int in_idx = 0; in_idx < IN_DIM; ++in_idx) {
      if (out_idx < OUT_DIM) {
        const float wf = static_cast<float>(w[out_idx * IN_DIM + in_idx]);
        wbuf[tile_o][in_idx] = fp32_ref_t(wf) * inv_scale;
      } else {
        wbuf[tile_o][in_idx] = fp32_ref_t(0.0f);
      }
    }
  }
}

template <int OUT_DIM, int IN_DIM>
static inline void prefetch_weight_tile(
  const double w[OUT_DIM * IN_DIM],
  int out_base,
  fp32_ref_t wbuf[OUT_TILE][IN_DIM]
) {
  for (int tile_o = 0; tile_o < OUT_TILE; ++tile_o) {
    const int out_idx = out_base + tile_o;
    for (int in_idx = 0; in_idx < IN_DIM; ++in_idx) {
      if (out_idx < OUT_DIM) {
        wbuf[tile_o][in_idx] = fp32_ref_t(static_cast<float>(w[out_idx * IN_DIM + in_idx]));
      } else {
        wbuf[tile_o][in_idx] = fp32_ref_t(0.0f);
      }
    }
  }
}

template <int OUT_DIM, int IN_DIM>
static inline void quant_linear_vec_tiled(
  const fp32_ref_t x[IN_DIM],
  const double w[OUT_DIM * IN_DIM],
  const double b[OUT_DIM],
  fp32_ref_t s_x,
  fp32_ref_t inv_scale,
  fp32_ref_t y[OUT_DIM]
) {
  fp32_ref_t qx[IN_DIM];
  for (int i = 0; i < IN_DIM; ++i) {
    qx[i] = fp32_round(x[i] * s_x);
  }

  fp32_ref_t wbuf_ping[OUT_TILE][IN_DIM];
  fp32_ref_t wbuf_pong[OUT_TILE][IN_DIM];

  int out_base = 0;
  bool use_ping = true;
  prefetch_quant_weight_tile<OUT_DIM, IN_DIM>(w, inv_scale, out_base, wbuf_ping);

  while (out_base < OUT_DIM) {
    const int next_out_base = out_base + OUT_TILE;
    if (next_out_base < OUT_DIM) {
      if (use_ping) {
        prefetch_quant_weight_tile<OUT_DIM, IN_DIM>(w, inv_scale, next_out_base, wbuf_pong);
      } else {
        prefetch_quant_weight_tile<OUT_DIM, IN_DIM>(w, inv_scale, next_out_base, wbuf_ping);
      }
    }

    fp32_ref_t (*cur)[IN_DIM] = use_ping ? wbuf_ping : wbuf_pong;
    for (int tile_o = 0; tile_o < OUT_TILE; ++tile_o) {
      const int out_idx = out_base + tile_o;
      if (out_idx >= OUT_DIM) {
        continue;
      }
      fp32_ref_t acc = fp32_ref_t(static_cast<float>(b[out_idx]));
      for (int in_idx = 0; in_idx < IN_DIM; ++in_idx) {
        acc += qx[in_idx] * cur[tile_o][in_idx];
      }
      y[out_idx] = acc;
    }

    out_base = next_out_base;
    use_ping = !use_ping;
  }
}

template <int OUT_DIM, int IN_DIM>
static inline void dense_vec_tiled(
  const fp32_ref_t x[IN_DIM],
  const double w[OUT_DIM * IN_DIM],
  const double b[OUT_DIM],
  fp32_ref_t y[OUT_DIM]
) {
  fp32_ref_t wbuf_ping[OUT_TILE][IN_DIM];
  fp32_ref_t wbuf_pong[OUT_TILE][IN_DIM];

  int out_base = 0;
  bool use_ping = true;
  prefetch_weight_tile<OUT_DIM, IN_DIM>(w, out_base, wbuf_ping);

  while (out_base < OUT_DIM) {
    const int next_out_base = out_base + OUT_TILE;
    if (next_out_base < OUT_DIM) {
      if (use_ping) {
        prefetch_weight_tile<OUT_DIM, IN_DIM>(w, next_out_base, wbuf_pong);
      } else {
        prefetch_weight_tile<OUT_DIM, IN_DIM>(w, next_out_base, wbuf_ping);
      }
    }

    fp32_ref_t (*cur)[IN_DIM] = use_ping ? wbuf_ping : wbuf_pong;
    for (int tile_o = 0; tile_o < OUT_TILE; ++tile_o) {
      const int out_idx = out_base + tile_o;
      if (out_idx >= OUT_DIM) {
        continue;
      }
      fp32_ref_t acc = fp32_ref_t(static_cast<float>(b[out_idx]));
      for (int in_idx = 0; in_idx < IN_DIM; ++in_idx) {
        acc += x[in_idx] * cur[tile_o][in_idx];
      }
      y[out_idx] = acc;
    }

    out_base = next_out_base;
    use_ping = !use_ping;
  }
}

static inline fp32_ref_t dot_head(
  const fp32_ref_t q_vec[D_MODEL],
  const fp32_ref_t k_vec[D_MODEL],
  int head_idx
) {
  const int base = head_idx * D_HEAD;
  fp32_ref_t dot = fp32_ref_t(0.0f);
  for (int dh = 0; dh < D_HEAD; ++dh) {
    dot += q_vec[base + dh] * k_vec[base + dh];
  }
  return dot;
}

static inline void layernorm_token(
  const fp32_ref_t x[D_MODEL],
  const double gamma[D_MODEL],
  const double beta[D_MODEL],
  fp32_ref_t y[D_MODEL]
) {
  fp32_ref_t sum = fp32_ref_t(0.0f);
  for (int i = 0; i < D_MODEL; ++i) {
    sum += x[i];
  }
  const fp32_ref_t mean = sum * INV_D_MODEL;

  fp32_ref_t var_acc = fp32_ref_t(0.0f);
  for (int i = 0; i < D_MODEL; ++i) {
    const fp32_ref_t d = x[i] - mean;
    var_acc += d * d;
  }
  const fp32_ref_t var = var_acc * INV_D_MODEL;
  const fp32_ref_t inv_std = ref_inv_sqrt_approx(var + LN_EPS);

  for (int i = 0; i < D_MODEL; ++i) {
    const fp32_ref_t xn = (x[i] - mean) * inv_std;
    const fp32_ref_t g = fp32_ref_t(static_cast<float>(gamma[i]));
    const fp32_ref_t b = fp32_ref_t(static_cast<float>(beta[i]));
    y[i] = xn * g + b;
  }
}

static inline void build_masks(
  bool one_ring[TOKENS_T][TOKENS_T],
  bool second_ring[TOKENS_T][TOKENS_T]
) {
  bool src[TOKENS_T][TOKENS_T];
  for (int i = 0; i < TOKENS_T; ++i) {
    for (int j = 0; j < TOKENS_T; ++j) {
      src[i][j] = (w_src_mask[i * TOKENS_T + j].to_int() != 0);
    }
  }

  for (int i = 0; i < TOKENS_T; ++i) {
    for (int j = 0; j < TOKENS_T; ++j) {
      const bool i_is_var = (i < VAR_N);
      const bool j_is_var = (j < VAR_N);

      if (i_is_var && j_is_var) {
        one_ring[i][j] = true;
        second_ring[i][j] = src[i][j];
      } else if (i_is_var && (!j_is_var)) {
        one_ring[i][j] = src[i][j];
        second_ring[i][j] = true;
      } else if ((!i_is_var) && j_is_var) {
        one_ring[i][j] = src[i][j];
        second_ring[i][j] = true;
      } else {
        one_ring[i][j] = true;
        second_ring[i][j] = src[i][j];
      }
    }
  }
}

static inline void get_layer_config(int layer_idx, LayerConfig &cfg) {
  if (layer_idx == 0) {
    cfg.w_q = w_decoder_layers_0_self_attn_linears_0_weight;
    cfg.b_q = w_decoder_layers_0_self_attn_linears_0_bias;
    cfg.w_k = w_decoder_layers_0_self_attn_linears_1_weight;
    cfg.b_k = w_decoder_layers_0_self_attn_linears_1_bias;
    cfg.w_v = w_decoder_layers_0_self_attn_linears_2_weight;
    cfg.b_v = w_decoder_layers_0_self_attn_linears_2_bias;
    cfg.w_o = w_decoder_layers_0_self_attn_linears_3_weight;
    cfg.b_o = w_decoder_layers_0_self_attn_linears_3_bias;
    cfg.w_ff1 = w_decoder_layers_0_feed_forward_w_1_weight;
    cfg.b_ff1 = w_decoder_layers_0_feed_forward_w_1_bias;
    cfg.w_ff2 = w_decoder_layers_0_feed_forward_w_2_weight;
    cfg.b_ff2 = w_decoder_layers_0_feed_forward_w_2_bias;
    cfg.ln0_w = w_decoder_layers_0_sublayer_0_norm_weight;
    cfg.ln0_b = w_decoder_layers_0_sublayer_0_norm_bias;
    cfg.ln1_w = w_decoder_layers_0_sublayer_1_norm_weight;
    cfg.ln1_b = w_decoder_layers_0_sublayer_1_norm_bias;
    cfg.s_x_in = fp32_ref_t(static_cast<float>(l0_in_s_x));
    cfg.s_x_o = fp32_ref_t(static_cast<float>(l0_o_s_x));
    cfg.s_x_ff1 = fp32_ref_t(static_cast<float>(l0_ff1_s_x));
    cfg.s_x_ff2 = fp32_ref_t(static_cast<float>(l0_ff2_s_x));
    cfg.inv_q = INV_SCALE_L0_Q;
    cfg.inv_k = INV_SCALE_L0_K;
    cfg.inv_v = INV_SCALE_L0_V;
    cfg.inv_o = INV_SCALE_L0_O;
    cfg.inv_ff1 = INV_SCALE_L0_FF1;
    cfg.inv_ff2 = INV_SCALE_L0_FF2;
  } else {
    cfg.w_q = w_decoder_layers_1_self_attn_linears_0_weight;
    cfg.b_q = w_decoder_layers_1_self_attn_linears_0_bias;
    cfg.w_k = w_decoder_layers_1_self_attn_linears_1_weight;
    cfg.b_k = w_decoder_layers_1_self_attn_linears_1_bias;
    cfg.w_v = w_decoder_layers_1_self_attn_linears_2_weight;
    cfg.b_v = w_decoder_layers_1_self_attn_linears_2_bias;
    cfg.w_o = w_decoder_layers_1_self_attn_linears_3_weight;
    cfg.b_o = w_decoder_layers_1_self_attn_linears_3_bias;
    cfg.w_ff1 = w_decoder_layers_1_feed_forward_w_1_weight;
    cfg.b_ff1 = w_decoder_layers_1_feed_forward_w_1_bias;
    cfg.w_ff2 = w_decoder_layers_1_feed_forward_w_2_weight;
    cfg.b_ff2 = w_decoder_layers_1_feed_forward_w_2_bias;
    cfg.ln0_w = w_decoder_layers_1_sublayer_0_norm_weight;
    cfg.ln0_b = w_decoder_layers_1_sublayer_0_norm_bias;
    cfg.ln1_w = w_decoder_layers_1_sublayer_1_norm_weight;
    cfg.ln1_b = w_decoder_layers_1_sublayer_1_norm_bias;
    cfg.s_x_in = fp32_ref_t(static_cast<float>(l1_in_s_x));
    cfg.s_x_o = fp32_ref_t(static_cast<float>(l1_o_s_x));
    cfg.s_x_ff1 = fp32_ref_t(static_cast<float>(l1_ff1_s_x));
    cfg.s_x_ff2 = fp32_ref_t(static_cast<float>(l1_ff2_s_x));
    cfg.inv_q = INV_SCALE_L1_Q;
    cfg.inv_k = INV_SCALE_L1_K;
    cfg.inv_v = INV_SCALE_L1_V;
    cfg.inv_o = INV_SCALE_L1_O;
    cfg.inv_ff1 = INV_SCALE_L1_FF1;
    cfg.inv_ff2 = INV_SCALE_L1_FF2;
  }
}

static void run_layer(
  const LayerConfig &cfg,
  const fp32_ref_t x_in[TOKENS_T][D_MODEL],
  const bool one_ring[TOKENS_T][TOKENS_T],
  const bool second_ring[TOKENS_T][TOKENS_T],
  fp32_ref_t scr_k[TOKENS_T][D_MODEL],
  fp32_ref_t scr_v[TOKENS_T][D_MODEL],
  fp32_ref_t x_out[TOKENS_T][D_MODEL]
) {
  for (int n = 0; n < TOKENS_T; ++n) {
    quant_linear_vec_tiled<D_MODEL, D_MODEL>(
      x_in[n],
      cfg.w_k,
      cfg.b_k,
      cfg.s_x_in,
      cfg.inv_k,
      scr_k[n]
    );
    quant_linear_vec_tiled<D_MODEL, D_MODEL>(
      x_in[n],
      cfg.w_v,
      cfg.b_v,
      cfg.s_x_in,
      cfg.inv_v,
      scr_v[n]
    );
  }

  fp32_ref_t attn_fifo[FIFO_DEPTH][D_MODEL];
  fp32_ref_t ffn1_fifo[FIFO_DEPTH][FF_DIM];

  for (int q_idx = 0; q_idx < TOKENS_T; ++q_idx) {
    const int slot = q_idx & (FIFO_DEPTH - 1);
    fp32_ref_t q_vec[D_MODEL];
    fp32_ref_t post_concat[D_MODEL];
    fp32_ref_t ln0_in[D_MODEL];
    fp32_ref_t ln0_out[D_MODEL];
    fp32_ref_t ffn2_out[D_MODEL];
    fp32_ref_t ln1_in[D_MODEL];

    quant_linear_vec_tiled<D_MODEL, D_MODEL>(
      x_in[q_idx],
      cfg.w_q,
      cfg.b_q,
      cfg.s_x_in,
      cfg.inv_q,
      q_vec
    );

    for (int h = 0; h < HEADS; ++h) {
      const bool (*mask)[TOKENS_T] = (h < 4) ? one_ring : second_ring;
      bool has_valid = false;
      fp32_ref_t max_score = NEG_LARGE;

      for (int k_idx = 0; k_idx < TOKENS_T; ++k_idx) {
        if (mask[q_idx][k_idx]) {
          continue;
        }
        has_valid = true;
        const fp32_ref_t score = dot_head(q_vec, scr_k[k_idx], h) * INV_SQRT_D_HEAD;
        if (score > max_score) {
          max_score = score;
        }
      }

      fp32_ref_t acc_vec[D_HEAD];
      for (int dh = 0; dh < D_HEAD; ++dh) {
        acc_vec[dh] = fp32_ref_t(0.0f);
      }

      if (!has_valid) {
        const int base = h * D_HEAD;
        for (int dh = 0; dh < D_HEAD; ++dh) {
          post_concat[base + dh] = fp32_ref_t(0.0f);
        }
        continue;
      }

      fp32_ref_t sumexp = fp32_ref_t(0.0f);
      for (int k_idx = 0; k_idx < TOKENS_T; ++k_idx) {
        if (mask[q_idx][k_idx]) {
          continue;
        }
        const fp32_ref_t score = dot_head(q_vec, scr_k[k_idx], h) * INV_SQRT_D_HEAD;
        const fp32_ref_t delta = ref_softmax_clamp_x(score - max_score);
        const fp32_ref_t w = ref_softmax_exp_lut(delta);
        sumexp += w;

        const int base = h * D_HEAD;
        for (int dh = 0; dh < D_HEAD; ++dh) {
          acc_vec[dh] += w * scr_v[k_idx][base + dh];
        }
      }

      const fp32_ref_t inv_sumexp = ref_softmax_rcp_lut(sumexp);
      const int base = h * D_HEAD;
      for (int dh = 0; dh < D_HEAD; ++dh) {
        post_concat[base + dh] = acc_vec[dh] * inv_sumexp;
      }
    }

    quant_linear_vec_tiled<D_MODEL, D_MODEL>(
      post_concat,
      cfg.w_o,
      cfg.b_o,
      cfg.s_x_o,
      cfg.inv_o,
      attn_fifo[slot]
    );

    for (int d = 0; d < D_MODEL; ++d) {
      ln0_in[d] = attn_fifo[slot][d] + x_in[q_idx][d];
    }
    layernorm_token(ln0_in, cfg.ln0_w, cfg.ln0_b, ln0_out);

    quant_linear_vec_tiled<FF_DIM, D_MODEL>(
      ln0_out,
      cfg.w_ff1,
      cfg.b_ff1,
      cfg.s_x_ff1,
      cfg.inv_ff1,
      ffn1_fifo[slot]
    );

    for (int i = 0; i < FF_DIM; ++i) {
      ffn1_fifo[slot][i] = fp32_relu(ffn1_fifo[slot][i]);
    }

    quant_linear_vec_tiled<D_MODEL, FF_DIM>(
      ffn1_fifo[slot],
      cfg.w_ff2,
      cfg.b_ff2,
      cfg.s_x_ff2,
      cfg.inv_ff2,
      ffn2_out
    );

    for (int d = 0; d < D_MODEL; ++d) {
      ln1_in[d] = ffn2_out[d] + ln0_out[d];
    }
    layernorm_token(ln1_in, cfg.ln1_w, cfg.ln1_b, x_out[q_idx]);
  }
}

} // namespace

void ref_step0_synth(
  ac_channel<fp32_ref_t> &in_y_ch,
  ac_channel<fp32_ref_t> &out_logits_ch,
  ac_channel<bit1_t> &out_xpred_ch
) {
  bool one_ring[TOKENS_T][TOKENS_T];
  bool second_ring[TOKENS_T][TOKENS_T];
  build_masks(one_ring, second_ring);

  fp32_ref_t y_var[VAR_N];
  int y_hard[VAR_N];
  for (int i = 0; i < VAR_N; ++i) {
    const fp32_ref_t y = in_y_ch.read();
    y_var[i] = y;
    y_hard[i] = (y < fp32_ref_t(0.0f)) ? 1 : 0;
  }

  fp32_ref_t node_feature[TOKENS_T];
  for (int i = 0; i < VAR_N; ++i) {
    node_feature[i] = fp32_abs(y_var[i]);
  }
  for (int c = 0; c < CHECK_N; ++c) {
    int parity = 0;
    for (int v = 0; v < VAR_N; ++v) {
      if (h_H[c * VAR_N + v].to_int() != 0) {
        parity ^= y_hard[v];
      }
    }
    node_feature[VAR_N + c] = (parity == 0) ? fp32_ref_t(1.0f) : fp32_ref_t(-1.0f);
  }

  static fp32_ref_t x_page0[TOKENS_T][D_MODEL];
  static fp32_ref_t x_page1[TOKENS_T][D_MODEL];
  static fp32_ref_t scr_k[TOKENS_T][D_MODEL];
  static fp32_ref_t scr_v[TOKENS_T][D_MODEL];

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int k = 0; k < 24; ++k) {
      x_page0[t][k] = node_feature[t] * fp32_ref_t(static_cast<float>(w_src_embed[t * 24 + k]));
    }
    for (int k = 0; k < 8; ++k) {
      x_page0[t][24 + k] = fp32_ref_t(static_cast<float>(w_lpe_token[t * 8 + k]));
    }
  }

  LayerConfig layer0_cfg;
  LayerConfig layer1_cfg;
  get_layer_config(0, layer0_cfg);
  get_layer_config(1, layer1_cfg);

  run_layer(layer0_cfg, x_page0, one_ring, second_ring, scr_k, scr_v, x_page1);

  for (int t = 0; t < TOKENS_T; ++t) {
    layernorm_token(x_page1[t], w_decoder_norm2_weight, w_decoder_norm2_bias, x_page0[t]);
  }

  run_layer(layer1_cfg, x_page0, one_ring, second_ring, scr_k, scr_v, x_page1);

  fp32_ref_t final_node_logits[TOKENS_T];
  for (int t = 0; t < TOKENS_T; ++t) {
    fp32_ref_t token_norm[D_MODEL];
    fp32_ref_t token_logit[1];
    layernorm_token(x_page1[t], w_decoder_norm_weight, w_decoder_norm_bias, token_norm);
    dense_vec_tiled<1, D_MODEL>(
      token_norm,
      w_oned_final_embed_0_weight,
      w_oned_final_embed_0_bias,
      token_logit
    );
    final_node_logits[t] = token_logit[0];
  }

  fp32_ref_t logits[VAR_N];
  dense_vec_tiled<VAR_N, TOKENS_T>(
    final_node_logits,
    w_out_fc_weight,
    w_out_fc_bias,
    logits
  );

  for (int n = 0; n < VAR_N; ++n) {
    out_logits_ch.write(logits[n]);
    const fp32_ref_t decision = logits[n] * sign_fp32(y_var[n]);
    const bit1_t xp = (decision < fp32_ref_t(0.0f)) ? bit1_t(1) : bit1_t(0);
    out_xpred_ch.write(xp);
  }
}

} // namespace aecct_ref
