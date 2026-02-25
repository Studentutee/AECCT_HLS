#pragma once
// Top.h（header-only）
// M3：Top-FSM + 單一 SRAM + CFG_RX expected_len 嚴格驗證
// 1) one-command-per-call：每次呼叫最多處理 1 筆 ctrl_cmd
// 2) ST_CFG_RX 在無 ctrl_cmd 時會接收 1 個 cfg word（from data_in）
// 3) READ_MEM 維持 M2 行為（IDLE/HALTED 可讀）

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "SramMapBringup.h"
#include "ModelDescBringup.h"
#include <cstdint>

namespace aecct {

    enum RegionId : unsigned {
        REG_X0 = 0,
        REG_X1 = 1,
        REG_SCR = 2,
        REG_W = 3,
        REG_OOR = 255
    };

    // M1/M2/M3 內部暫存器（internal regs）
    struct TopRegs {
        TopState state;
        bool w_base_set;
        u32_t w_base_word;
        u32_t outmode;

        // M3：cfg 接收與落地
        u32_t cfg_words[CFG_WORDS_EXPECTED];
        u32_t cfg_count;
        bool cfg_ready;

        u32_t cfg_magic;
        u32_t cfg_code_n;
        u32_t cfg_code_c;
        u32_t cfg_d_model;
        u32_t cfg_n_heads;
        u32_t cfg_d_head;
        u32_t cfg_d_ffn;
        u32_t cfg_d_lpe;
        u32_t cfg_n_layers;
        u32_t cfg_out_len_x_pred;
        u32_t cfg_out_len_logits;

        void clear() {
            state = ST_IDLE;
            w_base_set = false;
            w_base_word = 0;
            outmode = 0;

            cfg_count = 0;
            cfg_ready = false;
            for (unsigned i = 0; i < CFG_WORDS_EXPECTED; ++i) {
                cfg_words[i] = 0;
            }

            cfg_magic = 0;
            cfg_code_n = 0;
            cfg_code_c = 0;
            cfg_d_model = 0;
            cfg_n_heads = 0;
            cfg_d_head = 0;
            cfg_d_ffn = 0;
            cfg_d_lpe = 0;
            cfg_n_layers = 0;
            cfg_out_len_x_pred = 0;
            cfg_out_len_logits = 0;
        }
    };

    static inline TopRegs& top_regs() {
        static TopRegs regs;
        return regs;
    }

    // M2：單一實體 SRAM（single physical SRAM）
    static inline u32_t* top_sram() {
        static u32_t sram[SRAM_TOTAL_WORDS];
        return sram;
    }

    // TB 用 debug helper（不影響 top 介面）
    static inline TopState top_peek_state() { return top_regs().state; }
    static inline unsigned top_peek_cfg_count() { return (unsigned)top_regs().cfg_count.to_uint(); }
    static inline bool top_peek_cfg_ready() { return top_regs().cfg_ready; }
    static inline u32_t top_peek_cfg_word(unsigned idx) {
        if (idx >= CFG_WORDS_EXPECTED) { return (u32_t)0; }
        return top_regs().cfg_words[idx];
    }
    static inline u32_t top_peek_cfg_code_n() { return top_regs().cfg_code_n; }
    static inline u32_t top_peek_cfg_d_model() { return top_regs().cfg_d_model; }
    static inline u32_t top_peek_cfg_n_heads() { return top_regs().cfg_n_heads; }
    static inline u32_t top_peek_cfg_n_layers() { return top_regs().cfg_n_layers; }

    static inline RegionId decode_region(const u32_t& addr_word) {
        unsigned a = (unsigned)addr_word.to_uint();
        if (a >= X0_BASE_WORD && a < (X0_BASE_WORD + X0_WORDS)) { return REG_X0; }
        if (a >= X1_BASE_WORD && a < (X1_BASE_WORD + X1_WORDS)) { return REG_X1; }
        if (a >= SCR_BASE_WORD && a < (SCR_BASE_WORD + SCR_WORDS)) { return REG_SCR; }
        if (a >= W_BASE_WORD && a < (W_BASE_WORD + W_WORDS)) { return REG_W; }
        return REG_OOR;
    }

    static inline u32_t pack_init_pattern(unsigned region_id, unsigned local_idx) {
        uint32_t v = ((region_id & 0xFFu) << 24) | (local_idx & 0x00FFFFFFu);
        return (u32_t)v;
    }

    static inline unsigned min_u(unsigned a, unsigned b) {
        return (a < b) ? a : b;
    }

    static inline void init_region_prefix(
        u32_t* sram,
        unsigned base_word,
        unsigned region_words,
        unsigned region_id
    ) {
        unsigned n = min_u(INIT_WORDS, region_words);
        for (unsigned i = 0; i < n; ++i) {
            sram[base_word + i] = pack_init_pattern(region_id, i);
        }
    }

    static inline void cfg_session_clear(TopRegs& regs) {
        regs.cfg_count = 0;
        regs.cfg_ready = false;
        for (unsigned i = 0; i < CFG_WORDS_EXPECTED; ++i) {
            regs.cfg_words[i] = 0;
        }
    }

    static inline void soft_reset_all(TopRegs& regs, u32_t* sram) {
        regs.clear();
        init_region_prefix(sram, X0_BASE_WORD, X0_WORDS, (unsigned)REG_X0);
        init_region_prefix(sram, X1_BASE_WORD, X1_WORDS, (unsigned)REG_X1);
        init_region_prefix(sram, SCR_BASE_WORD, SCR_WORDS, (unsigned)REG_SCR);
        init_region_prefix(sram, W_BASE_WORD, W_WORDS, (unsigned)REG_W);
    }

    static inline bool cfg_validate_minimal(const TopRegs& regs) {
        uint32_t code_n = (uint32_t)regs.cfg_words[CFG_IDX_CODE_N].to_uint();
        uint32_t code_c = (uint32_t)regs.cfg_words[CFG_IDX_CODE_C].to_uint();
        uint32_t d_model = (uint32_t)regs.cfg_words[CFG_IDX_D_MODEL].to_uint();
        uint32_t n_heads = (uint32_t)regs.cfg_words[CFG_IDX_N_HEADS].to_uint();
        uint32_t d_head = (uint32_t)regs.cfg_words[CFG_IDX_D_HEAD].to_uint();
        uint32_t d_ffn = (uint32_t)regs.cfg_words[CFG_IDX_D_FFN].to_uint();
        uint32_t n_layers = (uint32_t)regs.cfg_words[CFG_IDX_N_LAYERS].to_uint();

        if (code_n == 0u) { return false; }
        if (code_c == 0u) { return false; }
        if (code_c > code_n) { return false; }

        if (d_model == 0u) { return false; }
        if (n_heads == 0u) { return false; }
        if ((d_model % n_heads) != 0u) { return false; }
        if (d_head == 0u) { return false; }
        if (d_head != (d_model / n_heads)) { return false; }

        if (d_ffn == 0u) { return false; }
        if (n_layers == 0u) { return false; }

        return true;
    }

    static inline void cfg_apply_to_regs(TopRegs& regs) {
        regs.cfg_magic = regs.cfg_words[CFG_IDX_MAGIC];
        regs.cfg_code_n = regs.cfg_words[CFG_IDX_CODE_N];
        regs.cfg_code_c = regs.cfg_words[CFG_IDX_CODE_C];
        regs.cfg_d_model = regs.cfg_words[CFG_IDX_D_MODEL];
        regs.cfg_n_heads = regs.cfg_words[CFG_IDX_N_HEADS];
        regs.cfg_d_head = regs.cfg_words[CFG_IDX_D_HEAD];
        regs.cfg_d_ffn = regs.cfg_words[CFG_IDX_D_FFN];
        regs.cfg_d_lpe = regs.cfg_words[CFG_IDX_D_LPE];
        regs.cfg_n_layers = regs.cfg_words[CFG_IDX_N_LAYERS];
        regs.cfg_out_len_x_pred = regs.cfg_words[CFG_IDX_OUT_LEN_X_PRED];
        regs.cfg_out_len_logits = regs.cfg_words[CFG_IDX_OUT_LEN_LOGITS];
    }

    static inline void cfg_ingest_one_word(TopRegs& regs, ac_channel<ac_int<32, false> >& data_in) {
        if (regs.cfg_ready) { return; }
        u32_t w;
        if (!data_in.nb_read(w)) { return; }

        unsigned idx = (unsigned)regs.cfg_count.to_uint();
        if (idx < CFG_WORDS_EXPECTED) {
            regs.cfg_words[idx] = w;
            regs.cfg_count = regs.cfg_count + 1;
            if ((unsigned)regs.cfg_count.to_uint() == CFG_WORDS_EXPECTED) {
                regs.cfg_ready = true;
            }
        }
    }

    static inline void handle_read_mem(
        ac_channel<ac_int<16, false> >& ctrl_rsp,
        ac_channel<ac_int<32, false> >& data_in,
        ac_channel<ac_int<32, false> >& data_out,
        u32_t* sram
    ) {
        // READ_MEM 參數：addr_word、len_words（依序）
        u32_t addr_word_in = data_in.read();
        u32_t len_words_in = data_in.read();

        unsigned long long addr_word = (unsigned long long)(uint32_t)addr_word_in.to_uint();
        unsigned long long len_words = (unsigned long long)(uint32_t)len_words_in.to_uint();

        if (len_words == 0ull) {
            ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_READ_MEM));
            return;
        }

        if (addr_word >= (unsigned long long)SRAM_TOTAL_WORDS ||
            (addr_word + len_words) > (unsigned long long)SRAM_TOTAL_WORDS) {
            ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_MEM_RANGE));
            return;
        }

        for (unsigned long long i = 0ull; i < len_words; ++i) {
            unsigned idx = (unsigned)(addr_word + i);
            data_out.write(sram[idx]);
        }
        ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_READ_MEM));
    }

    // Top 函式介面（固定，不可擅改）
    static inline void top(
        ac_channel<ac_int<16, false> >& ctrl_cmd,
        ac_channel<ac_int<16, false> >& ctrl_rsp,
        ac_channel<ac_int<32, false> >& data_in,
        ac_channel<ac_int<32, false> >& data_out
    ) {
#pragma HLS inline // Catapult-friendly: allow inlining

        TopRegs& regs = top_regs();
        u32_t* sram = top_sram();

        ac_int<16, false> cmdw;
        bool has_cmd = ctrl_cmd.nb_read(cmdw);

        if (has_cmd) {
            uint8_t op = unpack_ctrl_cmd_opcode(cmdw);

            if (regs.state == ST_IDLE) {
                if (op == (uint8_t)OP_NOOP) {
                    ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_NOOP));
                }
                else if (op == (uint8_t)OP_SOFT_RESET) {
                    soft_reset_all(regs, sram);
                    ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_SOFT_RESET));
                }
                else if (op == (uint8_t)OP_CFG_BEGIN) {
                    regs.state = ST_CFG_RX;
                    cfg_session_clear(regs);
                    ctrl_rsp.write(pack_ctrl_rsp_ok((uint8_t)OP_CFG_BEGIN));
                }
                else if (op == (uint8_t)OP_CFG_COMMIT) {
                    ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_STATE));
                }
                else if (op == (uint8_t)OP_SET_W_BASE) {
                    regs.w_base_set = true;
                    regs.w_base_word = 0; // M3 仍不讀 data_in arg
                    ctrl_rsp.write(pack_ctrl_rsp_ok((uint8_t)OP_SET_W_BASE));
                }
                else if (op == (uint8_t)OP_LOAD_W) {
                    if (!regs.w_base_set) {
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_STATE));
                    }
                    else {
                        regs.state = ST_PARAM_RX;
                        ctrl_rsp.write(pack_ctrl_rsp_ok((uint8_t)OP_LOAD_W));
                    }
                }
                else if (op == (uint8_t)OP_SET_OUTMODE) {
                    regs.outmode = 0; // M3 仍不讀 data_in arg，先固定 0
                    ctrl_rsp.write(pack_ctrl_rsp_ok((uint8_t)OP_SET_OUTMODE));
                }
                else if (op == (uint8_t)OP_INFER) {
                    regs.state = ST_INFER_RX;
                    ctrl_rsp.write(pack_ctrl_rsp_ok((uint8_t)OP_INFER));
                }
                else if (op == (uint8_t)OP_READ_MEM) {
                    handle_read_mem(ctrl_rsp, data_in, data_out, sram);
                }
                else if (op == (uint8_t)OP_DEBUG_CFG) {
                    ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_UNIMPL));
                }
                else {
                    ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_UNIMPL));
                }
            }
            else if (regs.state == ST_CFG_RX) {
                if (op == (uint8_t)OP_CFG_COMMIT) {
                    if (!regs.cfg_ready) {
                        // M3 釘死語意：長度不足報錯，留在 ST_CFG_RX 允許補送
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_CFG_LEN_MISMATCH));
                    }
                    else if (!cfg_validate_minimal(regs)) {
                        // M3 釘死語意：非法 cfg 回 IDLE，需重新 begin
                        regs.state = ST_IDLE;
                        cfg_session_clear(regs);
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_CFG_ILLEGAL));
                    }
                    else {
                        cfg_apply_to_regs(regs);
                        regs.state = ST_IDLE;
                        ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_CFG_COMMIT));
                    }
                }
                else if (op == (uint8_t)OP_SOFT_RESET) {
                    soft_reset_all(regs, sram);
                    ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_SOFT_RESET));
                }
                else if (op == (uint8_t)OP_NOOP) {
                    ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_NOOP));
                }
                else {
                    ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_STATE));
                }
            }
            else if (regs.state == ST_PARAM_RX) {
                if (op == (uint8_t)OP_SOFT_RESET) {
                    soft_reset_all(regs, sram);
                    ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_SOFT_RESET));
                }
                else if (op == (uint8_t)OP_NOOP) {
                    ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_NOOP));
                }
                else {
                    ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_STATE));
                }
            }
            else if (regs.state == ST_INFER_RX) {
                if (op == (uint8_t)OP_SOFT_RESET) {
                    soft_reset_all(regs, sram);
                    ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_SOFT_RESET));
                }
                else if (op == (uint8_t)OP_NOOP) {
                    ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_NOOP));
                }
                else {
                    ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_STATE));
                }
            }
            else if (regs.state == ST_HALTED) {
                if (op == (uint8_t)OP_READ_MEM) {
                    handle_read_mem(ctrl_rsp, data_in, data_out, sram);
                }
                else if (op == (uint8_t)OP_DEBUG_CFG) {
                    ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_UNIMPL));
                }
                else if (op == (uint8_t)OP_SOFT_RESET) {
                    soft_reset_all(regs, sram);
                    ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_SOFT_RESET));
                }
                else if (op == (uint8_t)OP_NOOP) {
                    ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_NOOP));
                }
                else {
                    ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_STATE));
                }
            }
            else {
                ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_INTERNAL));
            }
        }
        else {
            // 無控制命令時，僅在 CFG_RX 且未收滿時接收 1 個 cfg word
            if (regs.state == ST_CFG_RX && !regs.cfg_ready) {
                cfg_ingest_one_word(regs, data_in);
            }
        }
    }

} // namespace aecct
