#pragma once

// ============================================================
// Auto-merged header: Layer0_QKVProj.h + Layer0_Rest.h
// ============================================================

#include "ac_channel.h"
#include "ac_fixed.h"
#include "ac_int.h"
#include "weights.h"
#include "PreprocEmbedSPE.h"
#include "LayerNormUtils.h"
#include "ExpApproxUtils.h"

// ------------------------------------------------------------
// From: Layer0_QKVProj.h
// ------------------------------------------------------------
// ============================================================
// Layer0 self-attention Q/K/V projection (quantized activation, int8 weights)
//
// Python reference (from your notebook):
//   s_x = 127 / max(abs(x_layer0_in))   // max over trace set (constant)
//   x_q = round(x * s_x)
//   out = (x_q @ W.T) / (s_x * s_w) + b
//
// To avoid dynamic max(|x|) scan in hardware, we use a fixed constant scale.
// ============================================================

#ifndef FIXED_L0_IN_S_X
#define FIXED_L0_IN_S_X (40.336464)
#endif

class Layer0_QKVProj {
public:

  static constexpr int N_NODES = PreprocEmbedSPE::N_NODES; // 75
  static constexpr int D_MODEL = PreprocEmbedSPE::D_MODEL; // 32

  void run(ac_channel<ac_fixed<64,32,true>> &x_in_ch,
           ac_channel<ac_fixed<64,32,true>> &q_out_ch,
           ac_channel<ac_fixed<64,32,true>> &k_out_ch,
           ac_channel<ac_fixed<64,32,true>> &v_out_ch) {

    const ac_fixed<64,32,true> s_x = ac_fixed<64,32,true>(FIXED_L0_IN_S_X);

    const ac_fixed<64,32,true> s_w_q = w_decoder_layers_0_self_attn_linears_0_s_w[0];
    const ac_fixed<64,32,true> s_w_k = w_decoder_layers_0_self_attn_linears_1_s_w[0];
    const ac_fixed<64,32,true> s_w_v = w_decoder_layers_0_self_attn_linears_2_s_w[0];

    const ac_fixed<64,32, true, AC_RND_CONV, AC_SAT_SYM> inv_q = ac_fixed<64,32,true,AC_RND_CONV,AC_SAT_SYM>(1) / (s_x * s_w_q);
    const ac_fixed<64,32, true, AC_RND_CONV, AC_SAT_SYM> inv_k = ac_fixed<64,32,true,AC_RND_CONV,AC_SAT_SYM>(1) / (s_x * s_w_k);
    const ac_fixed<64,32, true, AC_RND_CONV, AC_SAT_SYM> inv_v = ac_fixed<64,32,true,AC_RND_CONV,AC_SAT_SYM>(1) / (s_x * s_w_v);

    NODE_LOOP: for (int node_idx = 0; node_idx < N_NODES; ++node_idx) {

      ac_int<8, true> x_q[D_MODEL];

      // quantize x
      XQ_LOOP: for (int in_idx = 0; in_idx < D_MODEL; ++in_idx) {
        ac_fixed<64,32,true> x_val = x_in_ch.read();
        ac_fixed<64,32,true, AC_RND_CONV, AC_SAT_SYM> qfx = ac_fixed<64,32,true ,AC_RND_CONV,AC_SAT_SYM>(x_val * s_x);
        ac_fixed<8,8,true, AC_RND_CONV, AC_SAT_SYM> q8 = qfx;
        x_q[in_idx] = ac_int<8,true>(q8.to_int());
      }

      // Q
      for (int out_idx = 0; out_idx < D_MODEL; ++out_idx) {
        ac_int<24, true> acc = 0;
        for (int in_idx = 0; in_idx < D_MODEL; ++in_idx) {
          ac_int<8,true> w = ac_int<8,true>(w_decoder_layers_0_self_attn_linears_0_weight[out_idx * D_MODEL + in_idx].to_int());
          acc += ac_int<24,true>(x_q[in_idx]) * ac_int<24,true>(w);
        }
        ac_fixed<64,32,true> y = ac_fixed<64,32,true>(ac_fixed<40,16,true,AC_RND_CONV,AC_SAT_SYM>(acc) * inv_q + w_decoder_layers_0_self_attn_linears_0_bias[out_idx]);
        q_out_ch.write(y);
      }

      // K
      for (int out_idx = 0; out_idx < D_MODEL; ++out_idx) {
        ac_int<24, true> acc = 0;
        for (int in_idx = 0; in_idx < D_MODEL; ++in_idx) {
          ac_int<8,true> w = ac_int<8,true>(w_decoder_layers_0_self_attn_linears_1_weight[out_idx * D_MODEL + in_idx].to_int());
          acc += ac_int<24,true>(x_q[in_idx]) * ac_int<24,true>(w);
        }
        ac_fixed<64,32,true> y = ac_fixed<64,32,true>(ac_fixed<40,16,true,AC_RND_CONV,AC_SAT_SYM>(acc) * inv_k + w_decoder_layers_0_self_attn_linears_1_bias[out_idx]);
        k_out_ch.write(y);
      }

      // V
      for (int out_idx = 0; out_idx < D_MODEL; ++out_idx) {
        ac_int<24, true> acc = 0;
        for (int in_idx = 0; in_idx < D_MODEL; ++in_idx) {
          ac_int<8,true> w = ac_int<8,true>(w_decoder_layers_0_self_attn_linears_2_weight[out_idx * D_MODEL + in_idx].to_int());
          acc += ac_int<24,true>(x_q[in_idx]) * ac_int<24,true>(w);
        }
        ac_fixed<64,32,true> y = ac_fixed<64,32,true>(ac_fixed<40,16,true,AC_RND_CONV,AC_SAT_SYM>(acc) * inv_v + w_decoder_layers_0_self_attn_linears_2_bias[out_idx]);
        v_out_ch.write(y);
      }
    }
  }
};

// ------------------------------------------------------------
// From: Layer0_Rest.h
// ------------------------------------------------------------
// ============================================================
// Layer0 block-level implementation (for block pipeline)
// All blocks use ac_datatype (ac_fixed/ac_int). No double.
//
// Blocks provided:
//  - StreamFork<N,T>: fork stream
//  - StreamAdd<N>: elementwise add
//  - StreamLayerNorm<N_NODES,D_MODEL>: LayerNorm per node
//  - L0_AttnCore: Q,K,V + mask + LPE -> concat head output (32)
//  - L0_OutProj: 32->32 quant linear (int8 weights)
//  - L0_FFN1: 32->128 quant linear
//  - StreamReLU<N>: ReLU
//  - L0_FFN2: 128->32 quant linear
//
// NOTE: scaling factors s_x use macros (default 1.0). You can replace with trace-fixed constants.
// ============================================================

#ifndef FIXED_L0_ATTN_OUT_S_X
#define FIXED_L0_ATTN_OUT_S_X (110.631622)
#endif
#ifndef FIXED_L0_FFN1_S_X
#define FIXED_L0_FFN1_S_X (42.850277)
#endif
#ifndef FIXED_L0_FFN2_S_X
#define FIXED_L0_FFN2_S_X (57.637299)
#endif

// --------------------------------------------
// Stream fork: read N items, write to two outputs
// --------------------------------------------
template<int N, typename T>
class StreamFork {
public:
  void run(ac_channel<T> &in, ac_channel<T> &o0, ac_channel<T> &o1) {
    for (int i=0;i<N;++i) {
      T v = in.read();
      o0.write(v);
      o1.write(v);
    }
  }
};

// --------------------------------------------
// Add two streams element-wise
// --------------------------------------------
template<int N>
class StreamAdd {
public:
  void run(ac_channel<fx_utils::fx_t> &a, ac_channel<fx_utils::fx_t> &b, ac_channel<fx_utils::fx_t> &y) {
    for (int i=0;i<N;++i) {
      y.write(fx_utils::fx_t(a.read() + b.read()));
    }
  }
};

// --------------------------------------------
// LayerNorm per node
// --------------------------------------------
template<int N_NODES, int D_MODEL>
class StreamLayerNorm {
public:
  void run(ac_channel<fx_utils::fx_t> &in, const fx_utils::fx_t gamma[D_MODEL], const fx_utils::fx_t beta[D_MODEL], ac_channel<fx_utils::fx_t> &out) {
    fx_utils::fx_t x[D_MODEL];
    fx_utils::fx_t y[D_MODEL];
    for (int n=0;n<N_NODES;++n) {
      for (int d=0; d<D_MODEL; ++d) x[d] = in.read();
      fx_utils::layer_norm<D_MODEL>(x, gamma, beta, y);
      for (int d=0; d<D_MODEL; ++d) out.write(y[d]);
    }
  }
};

// --------------------------------------------
// Attention core (masked + LPE) -> concat heads (32)
// - score = (Q·K)/sqrt(dh) + LPE
// - masked score => very negative
// - softmax uses exp_neg_approx()
// --------------------------------------------
class L0_AttnCore {
public:
  static constexpr int N_NODES = PreprocEmbedSPE::N_NODES; // 75
  static constexpr int D_MODEL = PreprocEmbedSPE::D_MODEL; // 32
  static constexpr int CODE_N  = PreprocEmbedSPE::CODE_N;
  static constexpr int CODE_C  = PreprocEmbedSPE::CODE_C;
  static constexpr int N_HEADS = 8;
  static constexpr int D_H     = D_MODEL / N_HEADS; // 4

  void run(ac_channel<fx_utils::fx_t> &q_in,
           ac_channel<fx_utils::fx_t> &k_in,
           ac_channel<fx_utils::fx_t> &v_in,
           ac_channel<fx_utils::fx_t> &attn_concat_out) {

    fx_utils::fx_t Q[N_NODES][D_MODEL];
    fx_utils::fx_t K[N_NODES][D_MODEL];
    fx_utils::fx_t V[N_NODES][D_MODEL];

    for (int n=0;n<N_NODES;++n) {
      for (int d=0; d<D_MODEL; ++d) {
        Q[n][d] = q_in.read();
        K[n][d] = k_in.read();
        V[n][d] = v_in.read();
      }
    }

    const fx_utils::fx_t inv_sqrt_dh = fx_utils::fx_t(0.5); // 1/sqrt(4)

    for (int i=0;i<N_NODES;++i) {
      fx_utils::fx_t out_vec[D_MODEL];

      for (int h=0; h<N_HEADS; ++h) {
        fx_utils::fx_t score[N_NODES];
        fx_utils::fx_t row_max = fx_utils::fx_t(-64);

        for (int j=0;j<N_NODES;++j) {

          ac_int<1,false> src_m = (w_src_mask[i*N_NODES + j] != 0) ? ac_int<1,false>(1) : ac_int<1,false>(0);

          ac_int<1,false> isVV = ((i < CODE_N) && (j < CODE_N)) ? ac_int<1,false>(1) : ac_int<1,false>(0);
          ac_int<1,false> isCC = ((i >= CODE_N) && (j >= CODE_N)) ? ac_int<1,false>(1) : ac_int<1,false>(0);
          ac_int<1,false> isVC = ((i < CODE_N) && (j >= CODE_N)) ? ac_int<1,false>(1) : ac_int<1,false>(0);
          ac_int<1,false> isCV = ((i >= CODE_N) && (j < CODE_N)) ? ac_int<1,false>(1) : ac_int<1,false>(0);

          ac_int<1,false> m_one    = ((isVV != 0) || (isCC != 0)) ? ac_int<1,false>(1) : src_m;
          ac_int<1,false> m_second = ((isVC != 0) || (isCV != 0)) ? ac_int<1,false>(1) : src_m;
          ac_int<1,false> masked   = (h < 4) ? m_one : m_second;

          if (masked != 0) {
            score[j] = fx_utils::fx_t(-64);
          } else {
            fx_utils::acc_t dot = 0;
            for (int dh=0; dh<D_H; ++dh) {
              int d = h*D_H + dh;
              dot += fx_utils::acc_t(Q[i][d]) * fx_utils::acc_t(K[j][d]);
            }
            fx_utils::fx_t s = fx_utils::fx_t(dot) * inv_sqrt_dh;

            int lpe_c = (h < 4) ? 0 : 1;
            fx_utils::fx_t lpe = w_lpe[(i*N_NODES + j)*2 + lpe_c];
            s = fx_utils::fx_t(s + lpe);

            score[j] = s;
            if (s > row_max) row_max = s;
          }
        }

        fx_exp::ufx_t expv[N_NODES];
        fx_exp::denom_t denom = 0;
        for (int j=0;j<N_NODES;++j) {
          fx_utils::fx_t diff = fx_utils::fx_t(score[j] - row_max);
          fx_exp::ufx_t e = fx_exp::exp_neg_approx(diff);
          expv[j] = e;
          denom += fx_exp::denom_t(e);
        }

        fx_utils::fx_t inv_denom;

        if (denom == fx_exp::denom_t(0)) {
            inv_denom = fx_utils::fx_t(0);
        } else {
            // 注意：/ 的回傳型別很寬，最後要明確轉回 fx_t
            inv_denom = fx_utils::fx_t( fx_utils::fx_t(1) / fx_utils::fx_t(denom) );
        }

        for (int dh=0; dh<D_H; ++dh) {
          fx_utils::acc_t acc = 0;
          for (int j=0;j<N_NODES;++j) {
            fx_utils::fx_t w = fx_utils::fx_t(expv[j]) * inv_denom;
            int d = h*D_H + dh;
            acc += fx_utils::acc_t(fx_utils::fx_t(w)) * fx_utils::acc_t(V[j][d]);
          }
          out_vec[h*D_H + dh] = fx_utils::fx_t(acc);
        }
      }

      for (int d=0; d<D_MODEL; ++d) attn_concat_out.write(out_vec[d]);
    }
  }
};

// --------------------------------------------
// Quantized linear: (32 -> 32) int8 weights (out proj)
// --------------------------------------------
class L0_OutProj {
public:
  static constexpr int N_NODES = PreprocEmbedSPE::N_NODES;
  static constexpr int D_MODEL = PreprocEmbedSPE::D_MODEL;

  void run(ac_channel<fx_utils::fx_t> &in, ac_channel<fx_utils::fx_t> &out) {
    const fx_utils::fx_t s_x = fx_utils::fx_t(FIXED_L0_ATTN_OUT_S_X);
    const fx_utils::fx_t s_w = w_decoder_layers_0_self_attn_linears_3_s_w[0];
    const ac_fixed<32, 6, true, AC_RND_CONV, AC_SAT_SYM> inv = ac_fixed<32,6,true,AC_RND_CONV,AC_SAT_SYM>(1) / (s_x * s_w);

    for (int n=0;n<N_NODES;++n) {
      ac_int<8,true> xq[D_MODEL];
      for (int i=0;i<D_MODEL;++i) {
        fx_utils::fx_t x = in.read();
        ac_fixed<16, 8, true, AC_RND_CONV, AC_SAT_SYM> qfx = ac_fixed<16,8,true,AC_RND_CONV,AC_SAT_SYM>(x * s_x);
        ac_fixed<8, 8, true, AC_RND_CONV, AC_SAT_SYM> q8 = qfx;
        xq[i] = ac_int<8,true>(q8.to_int());
      }

      for (int o=0;o<D_MODEL;++o) {
        ac_int<24,true> acc = 0;
        for (int i=0;i<D_MODEL;++i) {
          ac_int<8,true> w = ac_int<8,true>(w_decoder_layers_0_self_attn_linears_3_weight[o*D_MODEL + i].to_int());
          acc += ac_int<24,true>(xq[i]) * ac_int<24,true>(w);
        }
        fx_utils::fx_t y = fx_utils::fx_t(ac_fixed<40,16,true,AC_RND_CONV,AC_SAT_SYM>(acc) * inv + w_decoder_layers_0_self_attn_linears_3_bias[o]);
        out.write(y);
      }
    }
  }
};

// --------------------------------------------
// FFN1: (32 -> 128) int8 weights
// --------------------------------------------
class L0_FFN1 {
public:
  static constexpr int N_NODES = PreprocEmbedSPE::N_NODES;
  static constexpr int D_MODEL = PreprocEmbedSPE::D_MODEL;
  static constexpr int D_HID   = 128;

  void run(ac_channel<fx_utils::fx_t> &in, ac_channel<fx_utils::fx_t> &out) {
    const fx_utils::fx_t s_x = fx_utils::fx_t(FIXED_L0_FFN1_S_X);
    const fx_utils::fx_t s_w = w_decoder_layers_0_feed_forward_w_1_s_w[0];
    const ac_fixed<32, 6, true, AC_RND_CONV, AC_SAT_SYM> inv = ac_fixed<32,6,true,AC_RND_CONV,AC_SAT_SYM>(1) / (s_x * s_w);

    for (int n=0;n<N_NODES;++n) {
      ac_int<8,true> xq[D_MODEL];
      for (int i=0;i<D_MODEL;++i) {
        fx_utils::fx_t x = in.read();
        ac_fixed<16, 8, true, AC_RND_CONV, AC_SAT_SYM> qfx = ac_fixed<16,8,true,AC_RND_CONV,AC_SAT_SYM>(x * s_x);
        ac_fixed<8, 8, true, AC_RND_CONV, AC_SAT_SYM> q8 = qfx;
        xq[i] = ac_int<8,true>(q8.to_int());
      }

      for (int o=0;o<D_HID;++o) {
        ac_int<24,true> acc = 0;
        for (int i=0;i<D_MODEL;++i) {
          ac_int<8,true> w = ac_int<8,true>(w_decoder_layers_0_feed_forward_w_1_weight[o*D_MODEL + i].to_int());
          acc += ac_int<24,true>(xq[i]) * ac_int<24,true>(w);
        }
        fx_utils::fx_t y = fx_utils::fx_t(ac_fixed<40,16,true,AC_RND_CONV,AC_SAT_SYM>(acc) * inv + w_decoder_layers_0_feed_forward_w_1_bias[o]);
        out.write(y);
      }
    }
  }
};

// --------------------------------------------
// ReLU stream
// --------------------------------------------
template<int N>
class StreamReLU {
public:
  void run(ac_channel<fx_utils::fx_t> &in, ac_channel<fx_utils::fx_t> &out) {
    for (int i=0;i<N;++i) {
      fx_utils::fx_t x = in.read();
      out.write(fx_utils::relu(x));
    }
  }
};

// --------------------------------------------
// FFN2: (128 -> 32) int8 weights
// --------------------------------------------
class L0_FFN2 {
public:
  static constexpr int N_NODES = PreprocEmbedSPE::N_NODES;
  static constexpr int D_MODEL = PreprocEmbedSPE::D_MODEL;
  static constexpr int D_HID   = 128;

  void run(ac_channel<fx_utils::fx_t> &in, ac_channel<fx_utils::fx_t> &out) {
    const fx_utils::fx_t s_x = fx_utils::fx_t(FIXED_L0_FFN2_S_X);
    const fx_utils::fx_t s_w = w_decoder_layers_0_feed_forward_w_2_s_w[0];
    const ac_fixed<32, 6, true, AC_RND_CONV, AC_SAT_SYM> inv = ac_fixed<32,6,true,AC_RND_CONV,AC_SAT_SYM>(1) / (s_x * s_w);

    for (int n=0;n<N_NODES;++n) {
      ac_int<8,true> xq[D_HID];
      for (int i=0;i<D_HID;++i) {
        fx_utils::fx_t x = in.read();
        ac_fixed<16, 8, true, AC_RND_CONV, AC_SAT_SYM> qfx = ac_fixed<16,8,true,AC_RND_CONV,AC_SAT_SYM>(x * s_x);
        ac_fixed<8, 8, true, AC_RND_CONV, AC_SAT_SYM> q8 = qfx;
        xq[i] = ac_int<8,true>(q8.to_int());
      }

      for (int o=0;o<D_MODEL;++o) {
        ac_int<24,true> acc = 0;
        for (int i=0;i<D_HID;++i) {
          ac_int<8,true> w = ac_int<8,true>(w_decoder_layers_0_feed_forward_w_2_weight[o*D_HID + i].to_int());
          acc += ac_int<24,true>(xq[i]) * ac_int<24,true>(w);
        }
        fx_utils::fx_t y = fx_utils::fx_t(ac_fixed<40,16,true,AC_RND_CONV,AC_SAT_SYM>(acc) * inv + w_decoder_layers_0_feed_forward_w_2_bias[o]);
        out.write(y);
      }
    }
  }
};
