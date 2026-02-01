#pragma once

#include "ac_channel.h"
#include "ac_int.h"

#include "weights.h"

// Pre-processing + Embedding + SPE token
// Input : stream of CODE_N y values
// Output: stream of N_NODES * D_MODEL values (node-major, then feature-major)
class PreprocEmbedSPE {
public:

  // Sizes from weights.h
  static constexpr int CODE_N  = h_H_shape[1];                 // e.g. 63
  static constexpr int CODE_C  = h_H_shape[0];                 // e.g. 12
  static constexpr int N_NODES = CODE_N + CODE_C;              // e.g. 75

  static constexpr int D_EMBED = w_src_embed_shape[1];         // e.g. 24
  static constexpr int D_SPE   = w_lpe_token_shape[1];         // e.g. 8
  static constexpr int D_MODEL = D_EMBED + D_SPE;              // e.g. 32

private:
  static inline ac_fixed<64,32,true> abs_fx(const ac_fixed<64,32,true> &x) {
    return (x < ac_fixed<64,32,true>(0)) ? ac_fixed<64,32,true>(-x) : ac_fixed<64,32,true>(x);
  }

public:
  void run(ac_channel<ac_fixed<64,32,true>> &y_in_ch,
           ac_channel<ac_fixed<64,32,true>> &out_ch) {

    // Hard-bit pack of y (1 => negative)
    ac_int<1, false> hb_pack[CODE_N];

    // ------------------------------------------------------------
    // Nodes [0 .. CODE_N-1]: abs(y) * embed + SPE token
    // ------------------------------------------------------------
    NODE_Y_LOOP: for (int node_idx = 0; node_idx < CODE_N; ++node_idx) {
      ac_fixed<64,32,true> y_val = y_in_ch.read();
      hb_pack[node_idx] = (y_val < ac_fixed<64,32,true>(0)) ? ac_int<1,false>(1) : ac_int<1,false>(0);

      ac_fixed<64,32,true> abs_y = ac_fixed<64,32,true>(abs_fx(y_val));

      FEAT_Y_LOOP: for (int feat_idx = 0; feat_idx < D_MODEL; ++feat_idx) {
        ac_fixed<64,32,true> out_val;
        if (feat_idx < D_EMBED) {
          out_val = ac_fixed<64,32,true>(abs_y * w_src_embed[node_idx * D_EMBED + feat_idx]);
        } else {
          const int spe_idx = feat_idx - D_EMBED;
          out_val = ac_fixed<64,32,true>(w_lpe_token[node_idx * D_SPE + spe_idx]);
        }
        out_ch.write(out_val);
      }
    }

    // ------------------------------------------------------------
    // Nodes [CODE_N .. N_NODES-1]: syndrome (+1/-1) * embed + SPE token
    // ------------------------------------------------------------
    NODE_SYN_LOOP: for (int chk_idx = 0; chk_idx < CODE_C; ++chk_idx) {
      ac_int<1,false> parity = 0;
      SYN_COL_LOOP: for (int col_idx = 0; col_idx < CODE_N; ++col_idx) {
        const ac_int<1,false> h_bit = h_H[chk_idx * CODE_N + col_idx];
        parity ^= (h_bit & hb_pack[col_idx]);
      }

      ac_fixed<64,32,true> syn_val = (parity == 0) ? ac_fixed<64,32,true>(1) : ac_fixed<64,32,true>(-1);
      const int node_idx = CODE_N + chk_idx;

      FEAT_SYN_LOOP: for (int feat_idx = 0; feat_idx < D_MODEL; ++feat_idx) {
        ac_fixed<64,32,true> out_val;
        if (feat_idx < D_EMBED) {
          out_val = ac_fixed<64,32,true>(syn_val * w_src_embed[node_idx * D_EMBED + feat_idx]);
        } else {
          const int spe_idx = feat_idx - D_EMBED;
          out_val = ac_fixed<64,32,true>(w_lpe_token[node_idx * D_SPE + spe_idx]);
        }
        out_ch.write(out_val);
      }
    }
  }
};
