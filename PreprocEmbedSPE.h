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

    // 建議把型別集中管理（更好維護/換量化）
    using y_t = ac_fixed<32, 16, true>;   // 你現在用的型別
    using feat_t = ac_fixed<32, 16, true>;   // feature 建議 signed（因為 syndrome 需要 -1）
    using out_t = ac_fixed<32, 16, true>;
    using bit_t = ac_int<1, false>;
    using hb_t = ac_int<1, false>;

    static inline ac_fixed<32, 16, false> abs_fx(const y_t& x) {
        return (x < 0) ? ac_fixed<32, 16, false>(-x) : ac_fixed<32, 16, false>(x);
    }

    void run(const y_t y_in[CODE_N], out_t out_layer0_in[N_NODES * D_MODEL]){
        // hard bits 打包（第 i bit 代表 y_in[i] 是否為負）
        hb_t hb_pack[CODE_N];

        // ---- Step1: 前 63 個 nodes：abs(y) + hardbits + embedding ----
        Yabs_Embed: for (int i = 0; i < CODE_N; ++i) {
            y_t yv = y_in[i];
            hb_pack[i] = (yv < 0) ? bit_t(1) : bit_t(0);

            feat_t s = feat_t(abs_fx(yv));  // abs 是 unsigned，但 feature 用 signed 承接也OK（值是正的）

            // left 24 dims: x_embed_y = s * src_embed
            XembedY: for (int d = 0; d < D_EMBED; ++d) {
                // ★強烈建議：w_src_embed 在 weights.h 內就用 ac_fixed 常數存，避免 float->fixed 在綜合路徑出現
                out_layer0_in[i * D_MODEL + d] = s * w_src_embed[i * D_EMBED + d];
            }

            // right 8 dims: SPE token
            SPETwitter: for (int k = 0; k < D_SPE; ++k) {
                out_layer0_in[i * D_MODEL + (D_EMBED + k)] = w_lpe_token[i * D_SPE + k];
            }
        }

        // ---- Step2: 後 12 個 nodes：syndrome(列(Row) r) + embedding ----
        Syndrome_Embed: for (int r = 0; r < CODE_C; ++r) {
            bit_t parity = 0;
            // H 的第 r 列(Row) 與 hardbits 做 XOR parity
            Syndrome_COL: for (int c = 0; c < CODE_N; ++c) {
                bit_t h = h_H[r * CODE_N + c];       // 0/1
                parity ^= (h & bit_t(hb_pack[c]));   // hb_pack[c] 取出 bit
            }
            // syndrome_pm1: 0 -> +1, 1 -> -1
            feat_t s = (parity == 0) ? feat_t(1) : feat_t(-1);

            const int i = CODE_N + r;

            XembedS: for (int d = 0; d < D_EMBED; ++d) {
                out_layer0_in[i * D_MODEL + d] = s * w_src_embed[i * D_EMBED + d];
            }
            SPETwitter2: for (int k = 0; k < D_SPE; ++k) {
                out_layer0_in[i * D_MODEL + (D_EMBED + k)] = w_lpe_token[i * D_SPE + k];
            }
        }
    }
};
