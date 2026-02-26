#pragma once
// Top.h（header-only）
// M6：Top-FSM + CFG/PARAM/INFER + DEBUG_CFG/HALTED/RESUME
// 1) one-command-per-call：每次呼叫最多處理 1 筆 ctrl_cmd
// 2) ST_CFG_RX 在無 ctrl_cmd 時會接收 1 個 cfg word（from data_in）
// 3) ST_PARAM_RX 在無 ctrl_cmd 時會接收 1 個 param word（from data_in）
// 4) ST_INFER_RX 在無 ctrl_cmd 時會接收 1 個 input word（from data_in）
// 5) HALTED：先回 ctrl ERR，再吐 meta0/meta1 到 data_out
// 6) READ_MEM 維持 M2 行為（IDLE/HALTED 可讀）

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "SramMap.h"
#include "ModelDesc.h"
#include "PreprocDescBringup.h"
#include "LayerNormDesc.h"
#include "AttnDescBringup.h"
#include "FfnDescBringup.h"
#include "WeightStreamOrder.h"
#include "blocks/PreprocEmbedSPE.h"
#include "blocks/LayerNormBlock.h"
#include "blocks/AttnLayer0.h"
#include "blocks/FFNLayer0.h"
#include "weights.h"
#include <cstdint>

namespace aecct {

    static const unsigned CFG_WORDS_EXPECTED = (unsigned)EXP_LEN_CFG_WORDS;
    static const unsigned PARAM_WORDS_EXPECTED = (unsigned)EXP_LEN_PARAM_WORDS;
    static const unsigned PARAM_ALIGN_WORDS = (unsigned)W_LANES;
    static const unsigned INFER_IN_WORDS_EXPECTED = (unsigned)PREPROC_IN_WORDS_EXPECTED;
    static const unsigned X_OUT_WORDS_EXPECTED = (unsigned)PREPROC_X_OUT_WORDS_EXPECTED;
    static const unsigned OUT_WORDS_X_PRED = (unsigned)EXP_LEN_OUT_XPRED_WORDS;
    static const unsigned OUT_WORDS_LOGITS = (unsigned)EXP_LEN_OUT_LOGITS_WORDS;
    static const unsigned IN_BASE_WORD = (unsigned)PREPROC_IN_BASE_WORD_DEFAULT;
    static const unsigned X_OUT_BASE_WORD = (unsigned)PREPROC_X_OUT_BASE_WORD_DEFAULT;
    static const unsigned LN_X_IN_BASE_WORD = (unsigned)LN_X_IN_BASE_WORD_DEFAULT;
    static const unsigned LN_X_OUT_BASE_WORD = (unsigned)LN_X_OUT_BASE_WORD_DEFAULT;
    static const unsigned LN_GAMMA_BASE_WORD = (unsigned)LN_GAMMA_BASE_WORD_DEFAULT;
    static const unsigned LN_BETA_BASE_WORD = (unsigned)LN_BETA_BASE_WORD_DEFAULT;
    static const unsigned ATTN_X_IN_BASE_WORD = (unsigned)ATTN_X_IN_BASE_WORD_DEFAULT;
    static const unsigned ATTN_OUT_BASE_WORD = (unsigned)ATTN_OUT_BASE_WORD_DEFAULT;
    static const unsigned FFN_X_IN_BASE_WORD = (unsigned)FFN_X_IN_BASE_WORD_DEFAULT;
    static const unsigned INIT_WORDS = 64;
    static const unsigned DBG_META1_LEN_WORDS = 16u;

    enum DebugAction : unsigned {
        DBG_ACTION_CLEAR = 0u,
        DBG_ACTION_ARM = 1u,
        DBG_ACTION_RESUME = 2u
    };

    enum DebugTriggerSel : unsigned {
        DBG_TRIGGER_DISABLED = 0u,
        DBG_TRIGGER_ON_LOADW_COUNT = 1u
    };

    enum CfgIndexCompat : unsigned {
        CFG_IDX_CODE_N = (unsigned)CFG_CODE_N,
        CFG_IDX_CODE_K = (unsigned)CFG_CODE_K,
        CFG_IDX_CODE_C = (unsigned)CFG_CODE_C,
        CFG_IDX_N_NODES = (unsigned)CFG_N_NODES,
        CFG_IDX_D_MODEL = (unsigned)CFG_D_MODEL,
        CFG_IDX_N_HEAD = (unsigned)CFG_N_HEAD,
        CFG_IDX_N_LAYERS = (unsigned)CFG_N_LAYERS,
        CFG_IDX_D_FFN = (unsigned)CFG_D_FFN,
        CFG_IDX_ENABLE_LPE = (unsigned)CFG_ENABLE_LPE,
        CFG_IDX_ENABLE_LPE_TOKEN = (unsigned)CFG_ENABLE_LPE_TOKEN,
        CFG_IDX_OUT_MODE = (unsigned)CFG_OUT_MODE,
        CFG_IDX_RESERVED0 = (unsigned)CFG_RESERVED0
    };

    enum RegionId : unsigned {
        REG_X0 = 0,
        REG_X1 = 1,
        REG_SCR = 2,
        REG_W = 3,
        REG_OOR = 255
    };

    struct HaltInfo {
        bool valid;
        u32_t halt_reason;
        TopState prev_state;
        u32_t meta0_word_addr;
        u32_t meta1_len_words;

        void clear() {
            valid = false;
            halt_reason = 0;
            prev_state = ST_IDLE;
            meta0_word_addr = 0;
            meta1_len_words = 0;
        }
    };

    // M1/M2/M3/M4/M5 內部暫存器（internal regs）
    struct TopRegs {
        TopState state;
        bool w_base_set;
        u32_t w_base_word;
        u32_t param_count;
        u32_t input_count;
        u32_t outmode;

        // M5：debug/halt 控制
        bool debug_armed;
        u32_t dbg_trigger_sel;
        u32_t dbg_k_value;
        bool halt_active;
        HaltInfo halt_info;

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
            param_count = 0;
            input_count = 0;
            outmode = 0;

            debug_armed = false;
            dbg_trigger_sel = 0;
            dbg_k_value = 0;
            halt_active = false;
            halt_info.clear();

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
        static u32_t sram[sram_map::SRAM_WORDS_TOTAL];
        return sram;
    }

    // TB 用 debug helper（不影響 top 介面）
    static inline TopState top_peek_state() { return top_regs().state; }
    static inline unsigned top_peek_cfg_count() { return (unsigned)top_regs().cfg_count.to_uint(); }
    static inline bool top_peek_cfg_ready() { return top_regs().cfg_ready; }
    static inline unsigned top_peek_param_count() { return (unsigned)top_regs().param_count.to_uint(); }
    static inline unsigned top_peek_input_count() { return (unsigned)top_regs().input_count.to_uint(); }
    static inline u32_t top_peek_w_base_word() { return top_regs().w_base_word; }
    static inline bool top_peek_halt_active() { return top_regs().halt_active; }
    static inline u32_t top_peek_dbg_k_value() { return top_regs().dbg_k_value; }
    static inline u32_t top_peek_outmode() { return top_regs().outmode; }
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
        if (a >= sram_map::X_PAGE0_BASE_W && a < (sram_map::X_PAGE0_BASE_W + sram_map::X_PAGE0_WORDS)) { return REG_X0; }
        if (a >= sram_map::X_PAGE1_BASE_W && a < (sram_map::X_PAGE1_BASE_W + sram_map::X_PAGE1_WORDS)) { return REG_X1; }
        if (a >= sram_map::BASE_SCRATCH_W && a < (sram_map::BASE_SCRATCH_W + sram_map::SIZE_SCRATCH_W)) { return REG_SCR; }
        if (a >= sram_map::W_REGION_BASE && a < (sram_map::W_REGION_BASE + sram_map::W_REGION_WORDS)) { return REG_W; }
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

    static inline void param_session_clear(TopRegs& regs) {
        regs.param_count = 0;
    }

    static inline void infer_session_clear(TopRegs& regs) {
        regs.input_count = 0;
    }

    static inline bool is_param_base_in_w_region(uint32_t w_base_word) {
        return (w_base_word >= sram_map::W_REGION_BASE) &&
            (w_base_word < (sram_map::W_REGION_BASE + sram_map::W_REGION_WORDS));
    }

    static inline bool is_param_base_aligned(uint32_t w_base_word) {
        return ((w_base_word % PARAM_ALIGN_WORDS) == 0u);
    }

    static inline bool is_param_span_in_w_region(uint32_t w_base_word) {
        unsigned long long begin = (unsigned long long)w_base_word;
        unsigned long long end_excl = begin + (unsigned long long)PARAM_WORDS_EXPECTED;
        unsigned long long region_begin = (unsigned long long)sram_map::W_REGION_BASE;
        unsigned long long region_end = region_begin + (unsigned long long)sram_map::W_REGION_WORDS;
        return (begin >= region_begin) && (end_excl <= region_end);
    }

    static inline bool is_valid_outmode(uint32_t outmode) {
        return (outmode <= 2u);
    }

    static inline uint32_t dbg_get_action(uint32_t dbg_word) {
        return (dbg_word & 0x3u);
    }

    static inline uint32_t dbg_get_trigger_sel(uint32_t dbg_word) {
        return ((dbg_word >> 8) & 0xFFu);
    }

    static inline uint32_t dbg_get_k_value(uint32_t dbg_word) {
        return ((dbg_word >> 16) & 0xFFFFu);
    }

    static inline void debug_clear(TopRegs& regs) {
        regs.debug_armed = false;
        regs.dbg_trigger_sel = (u32_t)DBG_TRIGGER_DISABLED;
        regs.dbg_k_value = 0;
    }

    static inline void debug_arm(TopRegs& regs, uint32_t trigger_sel, uint32_t k_value) {
        regs.debug_armed = true;
        regs.dbg_trigger_sel = (u32_t)trigger_sel;
        regs.dbg_k_value = (u32_t)k_value;
    }

    static inline void enter_halted_and_emit(
        TopRegs& regs,
        ac_channel<ac_int<16, false> >& ctrl_rsp,
        ac_channel<ac_int<32, false> >& data_out
    ) {
        regs.halt_active = true;
        regs.halt_info.valid = true;
        regs.halt_info.halt_reason = (u32_t)ERR_DBG_HALT;
        regs.halt_info.prev_state = ST_PARAM_RX;
        regs.halt_info.meta0_word_addr = regs.w_base_word;
        regs.halt_info.meta1_len_words = (u32_t)DBG_META1_LEN_WORDS;
        regs.state = ST_HALTED;

        // M5 規格：先 ctrl_rsp ERR，再吐 meta0/meta1。
        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_DBG_HALT));
        data_out.write(regs.halt_info.meta0_word_addr);
        data_out.write(regs.halt_info.meta1_len_words);
    }

    static inline bool handle_debug_cfg_idle(
        TopRegs& regs,
        ac_channel<ac_int<16, false> >& ctrl_rsp,
        ac_channel<ac_int<32, false> >& data_in
    ) {
        u32_t dbg_word_in = data_in.read();
        uint32_t dbg_word = (uint32_t)dbg_word_in.to_uint();
        uint32_t action = dbg_get_action(dbg_word);
        uint32_t trigger_sel = dbg_get_trigger_sel(dbg_word);
        uint32_t k_value = dbg_get_k_value(dbg_word);

        if (action == DBG_ACTION_CLEAR) {
            debug_clear(regs);
            ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_DEBUG_CFG));
            return true;
        }
        if (action == DBG_ACTION_ARM) {
            if (trigger_sel == DBG_TRIGGER_DISABLED) {
                debug_clear(regs);
                ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_DEBUG_CFG));
                return true;
            }
            if (trigger_sel == DBG_TRIGGER_ON_LOADW_COUNT) {
                debug_arm(regs, trigger_sel, k_value);
                ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_DEBUG_CFG));
                return true;
            }
            ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_ARG));
            return true;
        }
        if (action == DBG_ACTION_RESUME) {
            ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_ARG));
            return true;
        }

        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_ARG));
        return true;
    }

    static inline bool handle_debug_cfg_halted(
        TopRegs& regs,
        ac_channel<ac_int<16, false> >& ctrl_rsp,
        ac_channel<ac_int<32, false> >& data_in
    ) {
        u32_t dbg_word_in = data_in.read();
        uint32_t dbg_word = (uint32_t)dbg_word_in.to_uint();
        uint32_t action = dbg_get_action(dbg_word);

        if (action == DBG_ACTION_CLEAR) {
            debug_clear(regs);
            ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_DEBUG_CFG));
            return true;
        }
        if (action == DBG_ACTION_RESUME) {
            regs.halt_active = false;
            regs.state = regs.halt_info.prev_state;
            ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_DEBUG_CFG));
            return true;
        }
        if (action == DBG_ACTION_ARM) {
            ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_ARG));
            return true;
        }

        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_ARG));
        return true;
    }

    static inline void soft_reset_all(TopRegs& regs, u32_t* sram) {
        regs.clear();
        init_region_prefix(sram, sram_map::X_PAGE0_BASE_W, sram_map::X_PAGE0_WORDS, (unsigned)REG_X0);
        init_region_prefix(sram, sram_map::X_PAGE1_BASE_W, sram_map::X_PAGE1_WORDS, (unsigned)REG_X1);
        init_region_prefix(sram, sram_map::BASE_SCRATCH_W, sram_map::SIZE_SCRATCH_W, (unsigned)REG_SCR);
        init_region_prefix(sram, sram_map::W_REGION_BASE, sram_map::W_REGION_WORDS, (unsigned)REG_W);
    }

    static inline bool cfg_validate_minimal(const TopRegs& regs) {
        uint32_t code_n = (uint32_t)regs.cfg_words[CFG_IDX_CODE_N].to_uint();
        uint32_t code_k = (uint32_t)regs.cfg_words[CFG_IDX_CODE_K].to_uint();
        uint32_t code_c = (uint32_t)regs.cfg_words[CFG_IDX_CODE_C].to_uint();
        uint32_t d_model = (uint32_t)regs.cfg_words[CFG_IDX_D_MODEL].to_uint();
        uint32_t n_heads = (uint32_t)regs.cfg_words[CFG_IDX_N_HEAD].to_uint();
        uint32_t d_ffn = (uint32_t)regs.cfg_words[CFG_IDX_D_FFN].to_uint();
        uint32_t n_layers = (uint32_t)regs.cfg_words[CFG_IDX_N_LAYERS].to_uint();

        if (code_n == 0u) { return false; }
        if (code_k == 0u) { return false; }
        if (code_k > code_n) { return false; }
        if (code_c == 0u) { return false; }
        if (code_c > code_n) { return false; }
        if ((code_k + code_c) != code_n) { return false; }

        if (d_model == 0u) { return false; }
        if (n_heads == 0u) { return false; }
        if ((d_model % n_heads) != 0u) { return false; }

        if (d_ffn == 0u) { return false; }
        if (n_layers == 0u) { return false; }

        return true;
    }

    static inline void cfg_apply_to_regs(TopRegs& regs) {
        regs.cfg_magic = 0;
        regs.cfg_code_n = regs.cfg_words[CFG_IDX_CODE_N];
        regs.cfg_code_c = regs.cfg_words[CFG_IDX_CODE_C];
        regs.cfg_d_model = regs.cfg_words[CFG_IDX_D_MODEL];
        regs.cfg_n_heads = regs.cfg_words[CFG_IDX_N_HEAD];
        regs.cfg_d_head = regs.cfg_words[CFG_IDX_D_MODEL] / regs.cfg_words[CFG_IDX_N_HEAD];
        regs.cfg_d_ffn = regs.cfg_words[CFG_IDX_D_FFN];
        regs.cfg_d_lpe = regs.cfg_words[CFG_IDX_ENABLE_LPE];
        regs.cfg_n_layers = regs.cfg_words[CFG_IDX_N_LAYERS];
        regs.cfg_out_len_x_pred = regs.cfg_words[CFG_IDX_OUT_MODE];
        regs.cfg_out_len_logits = regs.cfg_words[CFG_IDX_RESERVED0];
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

    static inline void param_ingest_one_word(
        TopRegs& regs,
        ac_channel<ac_int<32, false> >& data_in,
        ac_channel<ac_int<16, false> >& ctrl_rsp,
        ac_channel<ac_int<32, false> >& data_out,
        u32_t* sram
    ) {
        unsigned idx = (unsigned)regs.param_count.to_uint();
        if (idx >= PARAM_WORDS_EXPECTED) {
            regs.state = ST_IDLE;
            ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_LOAD_W));
            return;
        }

        u32_t w;
        if (!data_in.nb_read(w)) { return; }

        uint32_t base = (uint32_t)regs.w_base_word.to_uint();
        uint32_t addr = base + idx;
        sram[addr] = w;
        regs.param_count = regs.param_count + 1;

        // M5：當收到第 k 個 LOAD_W word（k 從 0 起算）時觸發 HALTED。
        if (regs.debug_armed &&
            ((uint32_t)regs.dbg_trigger_sel.to_uint() == (uint32_t)DBG_TRIGGER_ON_LOADW_COUNT) &&
            (idx == (unsigned)regs.dbg_k_value.to_uint())) {
            regs.debug_armed = false; // 觸發一次後自動解除，避免 RESUME 後立即再停。
            enter_halted_and_emit(regs, ctrl_rsp, data_out);
            return;
        }

        if ((unsigned)regs.param_count.to_uint() == PARAM_WORDS_EXPECTED) {
            // M4 bring-up：bitpack padding 檢查延後到 M4.1（此處先直接完成）
            regs.state = ST_IDLE;
            ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_LOAD_W));
        }
    }

    static inline void run_preproc_block(u32_t* sram) {
        PreprocCfg cfg;
        cfg.infer_in_words = (u32_t)INFER_IN_WORDS_EXPECTED;
        cfg.x_out_words = (u32_t)X_OUT_WORDS_EXPECTED;
        PreprocEmbedSPE(
            sram,
            cfg,
            (u32_t)IN_BASE_WORD,
            (u32_t)X_OUT_BASE_WORD
        );
    }

    static inline void run_layernorm_block(u32_t* sram) {
        LayerNormCfg cfg;
        cfg.token_count = (u32_t)LN_TOKEN_COUNT;
        cfg.d_model = (u32_t)LN_D_MODEL;
        cfg.eps = LN_EPS;

        LayerNormBlock(
            sram,
            cfg,
            (u32_t)LN_X_IN_BASE_WORD,
            (u32_t)LN_X_OUT_BASE_WORD,
            (u32_t)LN_GAMMA_BASE_WORD,
            (u32_t)LN_BETA_BASE_WORD
        );
    }

    static inline void run_attn_layer0_block(u32_t* sram) {
        AttnCfg cfg;
        cfg.token_count = (u32_t)ATTN_TOKEN_COUNT;
        cfg.d_model = (u32_t)ATTN_D_MODEL;
        cfg.n_heads = (u32_t)ATTN_N_HEADS;
        cfg.d_head = (u32_t)ATTN_D_HEAD;

        AttnScratch sc = default_attn_scratch();
        AttnLayer0<ATTN_STAGE_FULL>(
            sram,
            cfg,
            (u32_t)ATTN_X_IN_BASE_WORD,
            (u32_t)ATTN_OUT_BASE_WORD,
            sc
        );
    }

    static inline void run_ffn_layer0_block(u32_t* sram) {
        FfnCfg cfg;
        cfg.token_count = (u32_t)FFN_TOKEN_COUNT;
        cfg.d_model = (u32_t)FFN_D_MODEL;
        cfg.d_ffn = (u32_t)FFN_D_FFN;

        FfnScratch sc = default_ffn_scratch();
        FFNLayer0<FFN_STAGE_FULL>(
            sram,
            cfg,
            (u32_t)FFN_X_IN_BASE_WORD,
            sc
        );
    }

    static inline void run_ffn_add2_and_norm_block(u32_t* sram) {
        FfnScratch sc = default_ffn_scratch();
        uint32_t x_in_base = (uint32_t)FFN_X_IN_BASE_WORD;
        uint32_t w2_base = (uint32_t)sc.w2_out_base_word.to_uint();
        uint32_t add2_base = (uint32_t)sc.add2_base_word.to_uint();
        uint32_t gamma_base = (uint32_t)sc.ln_gamma_base_word.to_uint();
        uint32_t beta_base = (uint32_t)sc.ln_beta_base_word.to_uint();
        uint32_t ln_out_base = (uint32_t)sc.ln_out_base_word.to_uint();

        for (uint32_t i = 0; i < (uint32_t)FFN_X_WORDS; ++i) {
            union {
                uint32_t u;
                float f;
            } x_cvt, w2_cvt, y_cvt;
            x_cvt.u = (uint32_t)sram[x_in_base + i].to_uint();
            w2_cvt.u = (uint32_t)sram[w2_base + i].to_uint();
            y_cvt.f = x_cvt.f + w2_cvt.f;
            sram[add2_base + i] = (u32_t)y_cvt.u;
        }

        for (uint32_t c = 0; c < (uint32_t)FFN_D_MODEL; ++c) {
            union {
                uint32_t u;
                float f;
            } g_cvt, b_cvt;
            g_cvt.f = (float)w_decoder_layers_0_sublayer_1_norm_weight[c];
            b_cvt.f = (float)w_decoder_layers_0_sublayer_1_norm_bias[c];
            sram[gamma_base + c] = (u32_t)g_cvt.u;
            sram[beta_base + c] = (u32_t)b_cvt.u;
        }

        LayerNormCfg cfg;
        cfg.token_count = (u32_t)FFN_TOKEN_COUNT;
        cfg.d_model = (u32_t)FFN_D_MODEL;
        cfg.eps = LN_EPS;

        LayerNormBlock(
            sram,
            cfg,
            (u32_t)add2_base,
            (u32_t)ln_out_base,
            (u32_t)gamma_base,
            (u32_t)beta_base
        );
    }

    static inline void infer_emit_stub(
        const TopRegs& regs,
        ac_channel<ac_int<32, false> >& data_out,
        const u32_t* sram
    ) {
        uint32_t mode = (uint32_t)regs.outmode.to_uint();
        if (mode == 2u) {
            return;
        }
        if (mode == 0u) {
            for (unsigned i = 0; i < OUT_WORDS_X_PRED; ++i) {
                uint32_t inw = (uint32_t)sram[IN_BASE_WORD + (i % INFER_IN_WORDS_EXPECTED)].to_uint();
                uint32_t outw = inw ^ 0x5A5A5A5Au;
                data_out.write((u32_t)outw);
            }
            return;
        }
        if (mode == 1u) {
            for (unsigned i = 0; i < OUT_WORDS_LOGITS; ++i) {
                uint32_t outw = 0xC0000000u | i;
                data_out.write((u32_t)outw);
            }
            return;
        }
    }

    static inline void infer_ingest_one_word(
        TopRegs& regs,
        ac_channel<ac_int<32, false> >& data_in,
        ac_channel<ac_int<16, false> >& ctrl_rsp,
        ac_channel<ac_int<32, false> >& data_out,
        u32_t* sram
    ) {
        unsigned idx = (unsigned)regs.input_count.to_uint();
        if (idx >= INFER_IN_WORDS_EXPECTED) {
            run_preproc_block(sram);
            run_layernorm_block(sram);
            run_attn_layer0_block(sram);
            run_ffn_layer0_block(sram);
            run_ffn_add2_and_norm_block(sram);
            infer_emit_stub(regs, data_out, sram);
            regs.state = ST_IDLE;
            ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_INFER));
            return;
        }

        u32_t w;
        if (!data_in.nb_read(w)) { return; }

        sram[IN_BASE_WORD + idx] = w;
        regs.input_count = regs.input_count + 1;

        if ((unsigned)regs.input_count.to_uint() == INFER_IN_WORDS_EXPECTED) {
            run_preproc_block(sram);
            run_layernorm_block(sram);
            run_attn_layer0_block(sram);
            run_ffn_layer0_block(sram);
            run_ffn_add2_and_norm_block(sram);
            infer_emit_stub(regs, data_out, sram);
            regs.state = ST_IDLE;
            ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_INFER));
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

        if (addr_word >= (unsigned long long)sram_map::SRAM_WORDS_TOTAL ||
            (addr_word + len_words) > (unsigned long long)sram_map::SRAM_WORDS_TOTAL) {
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
                    u32_t w_base_in = data_in.read();
                    uint32_t w_base_word = (uint32_t)w_base_in.to_uint();

                    if (!is_param_base_in_w_region(w_base_word)) {
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_PARAM_BASE_RANGE));
                    }
                    else if (!is_param_base_aligned(w_base_word)) {
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_PARAM_BASE_ALIGN));
                    }
                    else {
                        regs.w_base_set = true;
                        regs.w_base_word = w_base_in;
                        ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_SET_W_BASE));
                    }
                }
                else if (op == (uint8_t)OP_LOAD_W) {
                    if (!regs.w_base_set) {
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_STATE));
                    }
                    else if (!is_param_span_in_w_region((uint32_t)regs.w_base_word.to_uint())) {
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_MEM_RANGE));
                    }
                    else {
                        regs.state = ST_PARAM_RX;
                        param_session_clear(regs);
                        ctrl_rsp.write(pack_ctrl_rsp_ok((uint8_t)OP_LOAD_W));
                    }
                }
                else if (op == (uint8_t)OP_SET_OUTMODE) {
                    u32_t outmode_in = data_in.read();
                    uint32_t outmode = (uint32_t)outmode_in.to_uint();
                    if (!is_valid_outmode(outmode)) {
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_ARG));
                    }
                    else {
                        regs.outmode = outmode_in;
                        ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_SET_OUTMODE));
                    }
                }
                else if (op == (uint8_t)OP_INFER) {
                    if (!regs.cfg_ready) {
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_BAD_STATE));
                    }
                    else {
                        regs.state = ST_INFER_RX;
                        infer_session_clear(regs);
                        ctrl_rsp.write(pack_ctrl_rsp_ok((uint8_t)OP_INFER));
                    }
                }
                else if (op == (uint8_t)OP_READ_MEM) {
                    handle_read_mem(ctrl_rsp, data_in, data_out, sram);
                }
                else if (op == (uint8_t)OP_DEBUG_CFG) {
                    handle_debug_cfg_idle(regs, ctrl_rsp, data_in);
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
                    handle_debug_cfg_halted(regs, ctrl_rsp, data_in);
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
            // 無控制命令時，在接收態各收 1 個資料字
            if (regs.state == ST_CFG_RX && !regs.cfg_ready) {
                cfg_ingest_one_word(regs, data_in);
            }
            else if (regs.state == ST_PARAM_RX) {
                param_ingest_one_word(regs, data_in, ctrl_rsp, data_out, sram);
            }
            else if (regs.state == ST_INFER_RX) {
                infer_ingest_one_word(regs, data_in, ctrl_rsp, data_out, sram);
            }
        }
    }

} // namespace aecct
