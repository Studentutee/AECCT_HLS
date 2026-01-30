#pragma once
#include "ac_channel.h"
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

    // 建議把型別集中管理（更好維護/換量化）
    using y_t = ac_fixed<32, 16, true>;   // 你現在用的型別
    using feat_t = ac_fixed<32, 16, true>;   // feature 建議 signed（因為 syndrome 需要 -1）
    using out_t = ac_fixed<32, 16, true>;
    using bit_t = ac_int<1, false>;
    using hb_t = ac_int<1, false>;

    static inline ac_fixed<32, 16, false> abs_fx(const y_t& x) {
        return (x < 0) ? ac_fixed<32, 16, false>(-x) : ac_fixed<32, 16, false>(x);
    }

    // ============================================================
    // Channel-based DUT I/O
    //  - y_in_ch  : stream CODE_N samples (order i=0..CODE_N-1)
    //  - out_ch   : stream N_NODES*D_MODEL features in the same
    //               flattened order as the original out_layer0_in[]
    //               (node-major, then feature-major)
    // ============================================================
    void run(ac_channel<y_t>& y_in_ch, ac_channel<out_t>& out_ch) {
        hb_t hb_pack[CODE_N];

        // ---- Step1: first CODE_N nodes: abs(y) + hardbits + embedding ----
        Yabs_Embed: for (int i = 0; i < CODE_N; ++i) {
            y_t yv = y_in_ch.read();
            hb_pack[i] = (yv < 0) ? bit_t(1) : bit_t(0);

            feat_t s = feat_t(abs_fx(yv));

            // left D_EMBED dims: x_embed_y = s * src_embed
            // right D_SPE  dims: SPE token
            // 合併成 single loop：每個 node(i) 連續輸出 D_MODEL = D_EMBED + D_SPE 個元素
            XembedYSPE: for (int d = 0; d < D_MODEL; ++d) {
                out_t v;

                if (d < D_EMBED) {
                    // 前半段：s * src_embed
                    v = out_t(s * w_src_embed[i * D_EMBED + d]);
                }
                else {
                    // 後半段：SPE token
                    v = out_t(w_lpe_token[i * D_SPE + (d - D_EMBED)]);
                }
                out_ch.write(v);
            }
        }

        // ---- Step2: remaining CODE_C nodes: syndrome(Row=列 r) + embedding ----
        Syndrome_Embed: for (int r = 0; r < CODE_C; ++r) {
            bit_t parity = 0;

            // XOR across columns(Colume=行 c) of H row r
            Syndrome_COL: for (int c = 0; c < CODE_N; ++c) {
                bit_t h = h_H[r * CODE_N + c];
                parity ^= (h & bit_t(hb_pack[c]));
            }

            // syndrome_pm1: 0 -> +1, 1 -> -1
            feat_t s = (parity == 0) ? feat_t(1) : feat_t(-1);
            const int i = CODE_N + r;

            Xembed_all: for (int d = 0; d < D_MODEL; ++d) {
                out_t v;

                if (d < D_EMBED) {
                    // 前半段：src_embed * s
                    v = out_t(s * w_src_embed[i * D_EMBED + d]);
                }
                else {
                    // 後半段：lpe_token
                    v = out_t(w_lpe_token[i * D_SPE + (d - D_EMBED)]);
                }

                out_ch.write(v);
            }
        }
    }
};
