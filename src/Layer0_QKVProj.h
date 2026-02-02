#pragma once

#include "ac_channel.h"
#include "ac_int.h"

#include "weights.h"
#include "PreprocEmbedSPE.h"

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
