#pragma once

#include "ac_channel.h"
#include "ac_std_float.h"
#include "ac_int.h"

#ifndef PREPROC_MAX_CODE_N
#define PREPROC_MAX_CODE_N  128
#endif
#ifndef PREPROC_MAX_CODE_C
#define PREPROC_MAX_CODE_C  128
#endif
#ifndef PREPROC_MAX_D_EMBED
#define PREPROC_MAX_D_EMBED 64
#endif
#ifndef PREPROC_MAX_D_SPE
#define PREPROC_MAX_D_SPE   64
#endif

#include "PreprocEmbedSPE_ext.h"

// ============================================================
// TOP control + shared SRAM
// - ctrl channel: commands
// - data channel: shared for weights + inference y
// - FSM: CFG -> WLOAD -> IDLE -> INFER
// - "ctrl 不接收新訊號直到 weights 完成"：在 WLOAD 狀態不 read ctrl
// ============================================================
class Top_PreprocEmbedSPE {
public:

    // SRAM sizing (words)
    // - src_embed : N_NODES * D_EMBED (float)
    // - lpe_token : N_NODES * D_SPE   (float)
    // - H bits    : CODE_C * CODE_N   (bit)
    static constexpr int MAX_CODE_N  = PREPROC_MAX_CODE_N;
    static constexpr int MAX_CODE_C  = PREPROC_MAX_CODE_C;
    static constexpr int MAX_N_NODES = PREPROC_MAX_CODE_N + PREPROC_MAX_CODE_C;
    static constexpr int MAX_D_EMBED = PREPROC_MAX_D_EMBED;
    static constexpr int MAX_D_SPE   = PREPROC_MAX_D_SPE;
    static constexpr int MAX_D_MODEL = PREPROC_MAX_D_EMBED + PREPROC_MAX_D_SPE;
    static constexpr int MAX_SRC_WORDS = MAX_N_NODES * MAX_D_EMBED;
    static constexpr int MAX_LPE_WORDS = MAX_N_NODES * MAX_D_SPE;
    static constexpr int MAX_H_BITS    = MAX_CODE_C  * MAX_CODE_N;

    // ----------------------------
    // Ctrl protocol
    // ----------------------------
    enum CtrlOpEnum {
        CTRL_NOP   = 0,
        CTRL_CFG   = 1,
        CTRL_WLOAD = 2,
        CTRL_INFER = 3
    };

    struct CtrlPkt {
        ac_int<2, false> op;       // CtrlOpEnum
        ac_int<16,false> code_n;   // for CFG
        ac_int<16,false> code_c;   // for CFG
        ac_int<16,false> d_embed;  // for CFG
        ac_int<16,false> d_spe;    // for CFG
    };

    // ----------------------------
    // Data protocol (shared data channel)
    // - kind=0: float weight (src/lpe)
    // - kind=1: H bit (b1)
    // - kind=2: inference y float
    // ----------------------------
    struct DataPkt {
        ac_int<2,false> kind;
        ac_ieee_float<binary32> f;
        ac_int<1,false> b;
    };

private:
    // ----------------------------
    // FSM states
    // ----------------------------
    enum StateEnum {
        ST_WAIT_CFG  = 0,
        ST_WLOAD     = 1,
        ST_IDLE      = 2,
        ST_INFER     = 3
    };

    ac_int<2,false> st;

    // runtime cfg latched
    PreprocEmbedSPE_ext::Cfg cfg_reg;

    // base offsets within SRAM spaces
    ac_int<32,false> base_src;
    ac_int<32,false> base_lpe;
    ac_int<32,false> base_h;

    // ----------------------------
    // SRAM storage (集中在 TOP)
    // ----------------------------
    ac_ieee_float<binary32> sram_f32[MAX_SRC_WORDS + MAX_LPE_WORDS];
    ac_int<1,false>         sram_b1 [MAX_H_BITS];

    // write pointers during WLOAD
    ac_int<32,false> wptr_src;
    ac_int<32,false> wptr_lpe;
    ac_int<32,false> wptr_h;

    // cached counts (for WLOAD)
    ac_int<32,false> need_src;
    ac_int<32,false> need_lpe;
    ac_int<32,false> need_h;

    // block instance
    PreprocEmbedSPE_ext preproc;

    // helper: safe SRAM read
    ac_ieee_float<binary32> mem_read_f32(const ac_int<32,false> addr) const {
        // addr maps to sram_f32 index
        return sram_f32[(int)addr];
    }
    ac_int<1,false> mem_read_b1(const ac_int<32,false> addr) const {
        return sram_b1[(int)addr];
    }

public:
    Top_PreprocEmbedSPE() {
        st = ST_WAIT_CFG;

        cfg_reg.code_n  = 0;
        cfg_reg.code_c  = 0;
        cfg_reg.d_embed = 0;
        cfg_reg.d_spe   = 0;

        base_src = 0;
        base_lpe = 0;
        base_h   = 0;

        wptr_src = 0;
        wptr_lpe = 0;
        wptr_h   = 0;

        need_src = 0;
        need_lpe = 0;
        need_h   = 0;
    }

    // ------------------------------------------------------------
    // TOP run
    // ------------------------------------------------------------
    void run(ac_channel<CtrlPkt> &ctrl_ch,
        ac_channel<DataPkt> &data_ch,
        ac_channel<ac_ieee_float<binary32>> &out_ch) {

        // ----------------------------------------------------------
        // ST_WAIT_CFG: only accept CTRL_CFG
        // ----------------------------------------------------------
        if (st == ST_WAIT_CFG) {
            if (!ctrl_ch.available(1)) return;
            CtrlPkt c = ctrl_ch.read();

            if (c.op == ac_int<2,false>(CTRL_CFG)) {
                cfg_reg.code_n  = c.code_n;
                cfg_reg.code_c  = c.code_c;
                cfg_reg.d_embed = c.d_embed;
                cfg_reg.d_spe   = c.d_spe;

                // base offsets (bias) inside TOP SRAM spaces
                base_src = 0;
                base_lpe = (ac_int<32,false>)(MAX_SRC_WORDS); // place lpe after src region (fixed partition)
                base_h   = 0;

                const int code_n  = (int)cfg_reg.code_n;
                const int code_c  = (int)cfg_reg.code_c;
                const int n_nodes = code_n + code_c;
                const int d_embed = (int)cfg_reg.d_embed;
                const int d_spe   = (int)cfg_reg.d_spe;

                need_src = (ac_int<32,false>)(n_nodes * d_embed);
                need_lpe = (ac_int<32,false>)(n_nodes * d_spe);
                need_h   = (ac_int<32,false>)(code_c * code_n);

                // reset write pointers
                wptr_src = 0;
                wptr_lpe = 0;
                wptr_h   = 0;

                st = ST_IDLE;
            }
            return;
        }

        // ----------------------------------------------------------
        // ST_IDLE: accept CTRL_WLOAD / CTRL_INFER
        // ----------------------------------------------------------
        if (st == ST_IDLE) {
            if (!ctrl_ch.available(1)) return;
            CtrlPkt c = ctrl_ch.read();

            if (c.op == ac_int<2,false>(CTRL_WLOAD)) {
                // enter weight load mode; during this mode, ctrl won't be read
                st = ST_WLOAD;
            } else if (c.op == ac_int<2,false>(CTRL_INFER)) {
                // configure block with bases
                preproc.configure(cfg_reg, base_src, base_lpe, base_h);
                st = ST_INFER;
            }
            return;
        }

        // ----------------------------------------------------------
        // ST_WLOAD: stream weights through data_ch
        // - ctrl is NOT read in this state (as you requested)
        // - order: src_embed (float) -> lpe_token (float) -> H bits (b1)
        // ----------------------------------------------------------
        if (st == ST_WLOAD) {
            // src embed
            if (wptr_src < need_src) {
                if (!data_ch.available(1)) return;
                DataPkt d = data_ch.read();
                // expect kind=0
                sram_f32[(int)(base_src + wptr_src)] = d.f;
                wptr_src++;
                return;
            }

            // lpe token
            if (wptr_lpe < need_lpe) {
                if (!data_ch.available(1)) return;
                DataPkt d = data_ch.read();
                sram_f32[(int)(base_lpe + wptr_lpe)] = d.f;
                wptr_lpe++;
                return;
            }

            // H bits
            if (wptr_h < need_h) {
                if (!data_ch.available(1)) return;
                DataPkt d = data_ch.read();
                // expect kind=1
                sram_b1[(int)(base_h + wptr_h)] = d.b;
                wptr_h++;
                return;
            }

            // done
            st = ST_IDLE;
            return;
        }

        // ----------------------------------------------------------
        // ST_INFER:
        // - read CODE_N y samples from data_ch (kind=2)
        // - feed to PreprocEmbedSPE_ext via internal y channel
        // - forward its output to out_ch
        // ----------------------------------------------------------
        if (st == ST_INFER) {
            ac_channel<ac_ieee_float<binary32>> y_in_ch;

            const int code_n  = (int)cfg_reg.code_n;
            const int code_c  = (int)cfg_reg.code_c;
            const int n_nodes = code_n + code_c;
            const int d_model = (int)cfg_reg.d_embed + (int)cfg_reg.d_spe;
            const int out_words = n_nodes * d_model;

            // collect y
        READ_Y_LOOP: for (int i = 0; i < code_n; ++i) {
            // blocking-style: if not available, just return and keep state
            if (!data_ch.available(1)) return;
            DataPkt d = data_ch.read();
            y_in_ch.write(d.f);
        }

        // run block (reads TOP SRAM through lambdas)
        preproc.run(
            y_in_ch, out_ch,
            [&](const ac_int<32,false> a) -> ac_ieee_float<binary32> { return mem_read_f32(a); },
            [&](const ac_int<32,false> a) -> ac_int<1,false> { return mem_read_b1(a); }
        );

        // one inference done; go back idle
        // (如果你要支援連續多筆 inference，可改成 CTRL_INFER_START/STOP 或加計數器)
        st = ST_IDLE;
        (void)out_words; // avoid unused warning in some toolchains
        return;
        }
    }
};
