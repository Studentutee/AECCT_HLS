#include <cstdio>
#include <cmath>
#include <algorithm>

#include "ac_channel.h"
#include "ac_std_float.h"
#include "ac_int.h"

#include "Top_PreprocEmbedSPE.h"

// 你的 weights.h（double arrays + shape arrays）
// - w_src_embed, w_lpe_token, h_H, and *_shape
#include "weights.h"

// 你的 trace headers（請替換成你目前的檔名）
// input y : [NUM_SAMPLES, CODE_N]
// expected out: [NUM_SAMPLES, N_NODES * D_MODEL]  (node-major then feat-major)
// 例如你常用：input_y_step0.h / embed_plus_SPE_step0.h
#include "input_y_step0.h"
#include "embed_plus_SPE_step0.h"

static inline double to_double(const ac_ieee_float<binary32> &x) {
    return (double)x.to_float();
}

int main() {
    Top_PreprocEmbedSPE dut;

    ac_channel<Top_PreprocEmbedSPE::CtrlPkt> ctrl_ch;
    ac_channel<Top_PreprocEmbedSPE::DataPkt> data_ch;
    ac_channel<ac_ieee_float<binary32>> out_ch;

    // -----------------------------
    // 1) CFG
    // -----------------------------
    {
        Top_PreprocEmbedSPE::CtrlPkt c;
        c.op      = ac_int<2,false>(Top_PreprocEmbedSPE::CTRL_CFG);
        c.code_n  = ac_int<16,false>((int)h_H_shape[1]);
        c.code_c  = ac_int<16,false>((int)h_H_shape[0]);
        c.d_embed = ac_int<16,false>((int)w_src_embed_shape[1]);
        c.d_spe   = ac_int<16,false>((int)w_lpe_token_shape[1]);
        ctrl_ch.write(c);
    }

    // push dut until it consumes cfg
    for (int k = 0; k < 8; ++k) dut.run(ctrl_ch, data_ch, out_ch);

    const int CODE_N  = (int)h_H_shape[1];
    const int CODE_C  = (int)h_H_shape[0];
    const int N_NODES = CODE_N + CODE_C;
    const int D_EMBED = (int)w_src_embed_shape[1];
    const int D_SPE   = (int)w_lpe_token_shape[1];
    const int D_MODEL = D_EMBED + D_SPE;

    const int SRC_WORDS = N_NODES * D_EMBED;
    const int LPE_WORDS = N_NODES * D_SPE;
    const int H_BITS    = CODE_C * CODE_N;

    // -----------------------------
    // 2) WLOAD command
    // -----------------------------
    {
        Top_PreprocEmbedSPE::CtrlPkt c;
        c.op = ac_int<2,false>(Top_PreprocEmbedSPE::CTRL_WLOAD);
        c.code_n = 0; c.code_c = 0; c.d_embed = 0; c.d_spe = 0;
        ctrl_ch.write(c);
    }

    // let dut enter ST_WLOAD
    for (int k = 0; k < 8; ++k) dut.run(ctrl_ch, data_ch, out_ch);

    // -----------------------------
    // 3) Stream weights via shared data channel
    //    - TB converts double -> ac_ieee_float<binary32>
    //    - order: src_embed floats, lpe_token floats, H bits
    // -----------------------------
    for (int i = 0; i < SRC_WORDS; ++i) {
        Top_PreprocEmbedSPE::DataPkt d;
        d.kind = ac_int<2,false>(0);
        d.f    = ac_ieee_float<binary32>((float)w_src_embed[i]);
        d.b    = 0;
        data_ch.write(d);
        dut.run(ctrl_ch, data_ch, out_ch);
    }

    for (int i = 0; i < LPE_WORDS; ++i) {
        Top_PreprocEmbedSPE::DataPkt d;
        d.kind = ac_int<2,false>(0);
        d.f    = ac_ieee_float<binary32>((float)w_lpe_token[i]);
        d.b    = 0;
        data_ch.write(d);
        dut.run(ctrl_ch, data_ch, out_ch);
    }

    for (int i = 0; i < H_BITS; ++i) {
        Top_PreprocEmbedSPE::DataPkt d;
        d.kind = ac_int<2,false>(1);
        d.f    = ac_ieee_float<binary32>(0.0f);
        // weights.h 的 h_H 這裡假設是 0/1 (double 或 int 都可)
        d.b    = (h_H[i] != 0) ? ac_int<1,false>(1) : ac_int<1,false>(0);
        data_ch.write(d);
        dut.run(ctrl_ch, data_ch, out_ch);
    }

    // flush a bit
    for (int k = 0; k < 32; ++k) dut.run(ctrl_ch, data_ch, out_ch);

    // -----------------------------
    // 4) INFER command
    // -----------------------------
    {
        Top_PreprocEmbedSPE::CtrlPkt c;
        c.op = ac_int<2,false>(Top_PreprocEmbedSPE::CTRL_INFER);
        c.code_n = 0; c.code_c = 0; c.d_embed = 0; c.d_spe = 0;
        ctrl_ch.write(c);
    }

    // let dut enter ST_INFER
    for (int k = 0; k < 8; ++k) dut.run(ctrl_ch, data_ch, out_ch);

    // -----------------------------
    // 5) Feed y from trace, compare output
    // -----------------------------
    const int NUM_SAMPLES = (int)trace_input_y_step0_tensor_shape[0]; // 你 trace header 若沒有 shape，請改成常數
	const int NUM_IN = (int)trace_input_y_step0_tensor_shape[1];     
    const double TH = 1e-5;

    int total_fail = 0;
    double max_err = 0.0;

    for (int s = 0; s < NUM_SAMPLES; ++s) {
        // feed CODE_N y
        for (int i = 0; i < CODE_N; ++i) {
            Top_PreprocEmbedSPE::DataPkt d;
            d.kind = ac_int<2,false>(2);
            d.f    = ac_ieee_float<binary32>((float)trace_input_y_step0_tensor[(s * NUM_IN)+i]);
            d.b    = 0;
            data_ch.write(d);
        }

        // run once (this consumes all y and produces full out)
        dut.run(ctrl_ch, data_ch, out_ch);

		const int NUM_OUT = N_NODES * D_MODEL;
        // read out and compare
        for (int idx = 0; idx < (N_NODES * D_MODEL); ++idx) {
            if (!out_ch.available(1)) {
                // keep ticking until data appears (simple TB)
                dut.run(ctrl_ch, data_ch, out_ch);
            }
            ac_ieee_float<binary32> got = out_ch.read();

            const double ref = (double)trace_embed_plus_SPE_step0_tensor[(s * NUM_OUT) + idx];
            const double g   = to_double(got);
            const double err = std::fabs(g - ref);

            if (err > TH) total_fail++;
            if (err > max_err) max_err = err;
        }

        // re-arm next inference: send CTRL_INFER again (因為 TOP 做成「一次推論→回 IDLE」)
        Top_PreprocEmbedSPE::CtrlPkt c;
        c.op = ac_int<2,false>(Top_PreprocEmbedSPE::CTRL_INFER);
        c.code_n = 0; c.code_c = 0; c.d_embed = 0; c.d_spe = 0;
        ctrl_ch.write(c);

        // tick to enter ST_INFER
        for (int k = 0; k < 8; ++k) dut.run(ctrl_ch, data_ch, out_ch);
    }

    std::printf("TH = %.1e\n", TH);
    std::printf("total_fail = %d\n", total_fail);
    std::printf("max_err    = %.12e\n", max_err);

    return (total_fail == 0) ? 0 : 1;
}
