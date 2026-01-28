#pragma once
#include "ac_fixed.h"
#include "ac_int.h"
#include "weights.h"

class PreprocEmbedSPE {
public:
    // ---- sizes from weights.h ----
    static constexpr int CODE_N = h_H_shape[1];          // 63
    static constexpr int CODE_C = h_H_shape[0];          // 12
    static constexpr int N_NODES = CODE_N + CODE_C;       // 75

    static constexpr int D_EMBED = w_src_embed_shape[1];  // 24
    static constexpr int D_SPE = w_lpe_token_shape[1];  // 8
    static constexpr int D_MODEL = D_EMBED + D_SPE;       // 32

    // 綜合友善 abs：直接針對輸入型別寫死
    static inline ac_fixed<32, 16, false> abs_fx(const ac_fixed<32, 16, true>& x) {
        return (x < 0) ? ac_fixed<32, 16, false>(-x) : ac_fixed<32, 16, false>(x);
    }

    // ============================================================
    // input : y_in[63]          
    // output: out_layer0_in[]   , len=75*32
    // ============================================================
	void run(const ac_fixed<32, 16, true> y_in[CODE_N], //Q3.5
        ac_fixed<32, 16, true> out_layer0_in[N_NODES * D_MODEL]) //Q3.5
    {
        // node_features_in: 前63=|y|, 後12=syndrome_pm1
        ac_fixed<32, 16, true> node_features_in[N_NODES];

        // hard bits / parity: 0/1
        ac_int<1, false> hard_bits[CODE_N];

        // ---- Step1: y_abs + hard bits ----
        Yabs_HardBits : for (int i = 0; i < CODE_N; ++i) {
			ac_fixed<32, 16, true> yv = y_in[i]; // tmp Q3.5
			node_features_in[i] = abs_fx(yv); // unsign Q2.6 如果需要這裡或許還能摳出一點面積
			// hard_bits: 0->正, 1->負，因為後面要用XOR計算parity
            hard_bits[i] = (yv < 0) ? ac_int<1, false>(1) : ac_int<1, false>(0);
        }

        // ---- Step1: syndrome_bits = (H @ hard_bits) % 2 ----
        syndrome_bits_ROW : for (int r = 0; r < CODE_C; ++r) {
            ac_int<1, false> parity = 0;
            syndrome_bits_COL : for (int c = 0; c < CODE_N; ++c) {
                //ac_int<1, false> h = ac_int<1, false>(h_H[r * CODE_N + c] & 1);
                ac_int<1, false> h = h_H[r * CODE_N + c]; // 0/1
                parity ^= (h & hard_bits[c]);
            }

            // syndrome_pm1: 0->+1, 1->-1  (pm1_t 也直接寫死在這裡)
            node_features_in[CODE_N + r] = (parity == 0) ? ac_int<2, true>(1) : ac_int<2, true>(-1);
        }

        // ---- Step2: x_layer0_in = [x_embed_y, lpe_token] ----
            x_layer0_in : for (int i = 0; i < N_NODES; ++i) {
			ac_fixed<32, 16, true> s = node_features_in[i]; // Q2.6

            // left 24 dims: x_embed_y(i,d) = s * src_embed(i,d)
            x_embed_y : for (int d = 0; d < D_EMBED; ++d) {
				ac_fixed<32, 16, true> w = w_src_embed[i * D_EMBED + d]; // float -> fixed
                out_layer0_in[i * D_MODEL + d] = s * w;
            }

            // right 8 dims: SPE token
            SPE_token : for (int k = 0; k < D_SPE; ++k) {
				ac_fixed<32, 16, true> t = w_lpe_token[i * D_SPE + k];  // float -> fixed
                out_layer0_in[i * D_MODEL + (D_EMBED + k)] = t;
            }
        }
    }
};
