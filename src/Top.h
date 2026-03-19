#pragma once
// SSOT: src/Top.h is the only Top contract definition.
// design/AecctTop.h is wrapper/adapter only.
// Header-only Top integration contract for FSM dispatch and runtime paths.
// One command is consumed per top() call from ctrl_cmd.
// CFG/PARAM/INFER payload words are consumed from data_in in their RX states.
// HALTED emits ERR + metadata to data_out; READ_MEM is gated by legal states.
#include "AecctTypes.h"
#include "AecctUtil.h"
#include "AecctProtocol.h"
#include "AecctRanges.h"
#include "AecctMemReq.h"
#include "gen/SramMap.h"
#include "gen/ModelDesc.h"
#include "PreprocDescBringup.h"
#include "LayerNormDesc.h"
#include "AttnDescBringup.h"
#include "FfnDescBringup.h"
#include "LayerScratchDesc.h"
#include "LayerParamBringup.h"
#include "gen/WeightStreamOrder.h"
#include "blocks/PreprocEmbedSPE.h"
#include "blocks/LayerNormBlock.h"
#include "blocks/AttnPhaseATopManagedKv.h"
#include "blocks/AttnPhaseATopManagedQ.h"
#include "blocks/TransformerLayer.h"
#include "blocks/FinalHead.h"
#include <cstdint>

namespace aecct {
    // Top is the integration owner for this design boundary:
    // - owns the external 4-channel contract and state machine dispatch
    // - owns shared SRAM lifetime/arbitration semantics
    // - dispatches block calls with explicit base/range ownership boundaries
    // Sub-blocks consume the ranges passed by Top and do not own global SRAM policy.

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
    static const unsigned FINAL_LOGITS_BASE_WORD = (unsigned)sram_map::BASE_SCRATCH_W;
    static const unsigned FINAL_XPRED_BASE_WORD = (unsigned)(sram_map::BASE_SCRATCH_W + OUT_WORDS_LOGITS);
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

    enum ReceiverState : unsigned {
        RX_NONE = 0,
        RX_CFG = 1,
        RX_PARAM = 2,
        RX_INFER = 3
    };

    static const unsigned MEM_REQ_SLOTS = (unsigned)REQ_ID_COUNT;

    struct MemArbRegs {
        MemReq pending[MEM_REQ_SLOTS];
        bool pending_valid[MEM_REQ_SLOTS];
        u16_t rr_cursor[PRIO_CLASS_COUNT];
        MemGrant grant_latched;
        bool grant_valid;

        void clear() {
            for (unsigned i = 0; i < MEM_REQ_SLOTS; ++i) {
                pending[i] = make_empty_mem_req();
                pending_valid[i] = false;
            }
            for (unsigned p = 0; p < (unsigned)PRIO_CLASS_COUNT; ++p) {
                rr_cursor[p] = 0;
            }
            grant_latched = make_empty_mem_grant();
            grant_valid = false;
        }
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

    // Persistent top-level internal registers.
    // These registers are the single source of truth for Top-owned runtime state.
    struct TopRegs {
        TopState state;
        ReceiverState rx_state;
        bool w_base_set;
        u32_t w_base_word;
        u32_t param_count;
        u32_t input_count;
        u32_t outmode;
        u32_t infer_final_x_base_word;
        u32_t infer_mid_dump_base_word;
        bool infer_mid_valid;
        u32_t infer_logits_base_word;
        u32_t infer_xpred_base_word;
        u32_t infer_input_shadow[INFER_IN_WORDS_EXPECTED];
        bool p11ac_mainline_path_taken;
        bool p11ac_fallback_taken;
        bool p11ad_mainline_q_path_taken;
        bool p11ad_q_fallback_taken;

        // Top-controlled block contract placeholders (skeleton only).
        PreprocBlockContract preproc_contract;
        TransformerLayerContract transformer_contract;
        LayerNormBlockContract layernorm_contract;
        FinalHeadContract final_head_contract;

        // Shared SRAM arbitration stub owned by Top.
        MemArbRegs mem_arb;

        // Debug/HALT control and sticky halt metadata owned by Top.
        bool debug_armed;
        u32_t dbg_trigger_sel;
        u32_t dbg_k_value;
        bool halt_active;
        HaltInfo halt_info;

        // CFG ingest shadow words and decoded runtime CFG registers.
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
            rx_state = RX_NONE;
            w_base_set = false;
            w_base_word = 0;
            param_count = 0;
            input_count = 0;
            outmode = 0;
            infer_final_x_base_word = 0;
            infer_mid_dump_base_word = 0;
            infer_mid_valid = false;
            infer_logits_base_word = (u32_t)FINAL_LOGITS_BASE_WORD;
            infer_xpred_base_word = (u32_t)FINAL_XPRED_BASE_WORD;
            for (unsigned i = 0; i < INFER_IN_WORDS_EXPECTED; ++i) {
                infer_input_shadow[i] = 0;
            }
            p11ac_mainline_path_taken = false;
            p11ac_fallback_taken = false;
            p11ad_mainline_q_path_taken = false;
            p11ad_q_fallback_taken = false;
            clear_preproc_contract(preproc_contract);
            clear_transformer_layer_contract(transformer_contract);
            clear_layernorm_contract(layernorm_contract);
            clear_final_head_contract(final_head_contract);
            mem_arb.clear();

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

    // Single physical SRAM backing store.
    static inline u32_t* top_sram() {
        static u32_t sram[sram_map::SRAM_WORDS_TOTAL];
        return sram;
    }

    // Internal input staging FIFO.
    // Functional C++ model does not rely on finite depth; concrete depth is
    // configured later by Catapult/HLS constraints.
    static inline data_ch_t& top_in_fifo() {
        static data_ch_t in_fifo;
        return in_fifo;
    }

    static inline bool top_data_nb_read(data_ch_t& data_in, u32_t& word) {
        if (top_in_fifo().nb_read(word)) {
            return true;
        }
        u32_t staged;
        if (!data_in.nb_read(staged)) {
            return false;
        }
        top_in_fifo().write(staged);
        return top_in_fifo().nb_read(word);
    }

    static inline u32_t top_data_read(data_ch_t& data_in) {
        u32_t word;
        if (top_in_fifo().nb_read(word)) {
            return word;
        }
        top_in_fifo().write(data_in.read());
        return top_in_fifo().read();
    }

    // TB debug peek helpers for observing Top-owned runtime state.
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
    static inline u32_t top_peek_infer_final_x_base_word() { return top_regs().infer_final_x_base_word; }
    static inline u32_t top_peek_infer_mid_dump_base_word() { return top_regs().infer_mid_dump_base_word; }
    static inline bool top_peek_infer_mid_valid() { return top_regs().infer_mid_valid; }
    static inline u32_t top_peek_infer_logits_base_word() { return top_regs().infer_logits_base_word; }
    static inline u32_t top_peek_infer_xpred_base_word() { return top_regs().infer_xpred_base_word; }
    static inline bool top_peek_p11ac_mainline_path_taken() { return top_regs().p11ac_mainline_path_taken; }
    static inline bool top_peek_p11ac_fallback_taken() { return top_regs().p11ac_fallback_taken; }
    static inline bool top_peek_p11ad_mainline_q_path_taken() { return top_regs().p11ad_mainline_q_path_taken; }
    static inline bool top_peek_p11ad_q_fallback_taken() { return top_regs().p11ad_q_fallback_taken; }

    static inline RegionId decode_region(const u32_t& addr_word) {
        unsigned a = (unsigned)addr_word.to_uint();
        if (a >= sram_map::X_PAGE0_BASE_W && a < (sram_map::X_PAGE0_BASE_W + sram_map::X_PAGE0_WORDS)) { return REG_X0; }
        if (a >= sram_map::X_PAGE1_BASE_W && a < (sram_map::X_PAGE1_BASE_W + sram_map::X_PAGE1_WORDS)) { return REG_X1; }
        if (a >= sram_map::BASE_SCRATCH_W && a < (sram_map::BASE_SCRATCH_W + sram_map::SIZE_SCRATCH_W)) { return REG_SCR; }
        if (a >= sram_map::W_REGION_BASE && a < (sram_map::W_REGION_BASE + sram_map::W_REGION_WORDS)) { return REG_W; }
        return REG_OOR;
    }

    static inline ReceiverState receiver_state_of(TopState state) {
        if (state == ST_CFG_RX) { return RX_CFG; }
        if (state == ST_PARAM_RX) { return RX_PARAM; }
        if (state == ST_INFER_RX) { return RX_INFER; }
        return RX_NONE;
    }

    static inline void refresh_receiver_state(TopRegs& regs) {
        regs.rx_state = receiver_state_of(regs.state);
    }

    static inline void mem_arb_submit(TopRegs& regs, const MemReq& req) {
        if (!req.valid) { return; }
        unsigned slot = (unsigned)req.requester;
        if (slot >= MEM_REQ_SLOTS) { return; }
        regs.mem_arb.pending[slot] = req;
        regs.mem_arb.pending_valid[slot] = true;
    }

    static inline bool mem_arb_grant_one(TopRegs& regs) {
        regs.mem_arb.grant_latched = make_empty_mem_grant();
        regs.mem_arb.grant_valid = false;

        for (unsigned p = 0; p < (unsigned)PRIO_CLASS_COUNT; ++p) {
            unsigned start = (unsigned)regs.mem_arb.rr_cursor[p].to_uint();
            for (unsigned off = 0; off < MEM_REQ_SLOTS; ++off) {
                unsigned slot = (start + off) % MEM_REQ_SLOTS;
                if (!regs.mem_arb.pending_valid[slot]) { continue; }
                const MemReq& req = regs.mem_arb.pending[slot];
                if ((unsigned)req.prio != p) { continue; }

                regs.mem_arb.grant_valid = true;
                regs.mem_arb.grant_latched.valid = true;
                regs.mem_arb.grant_latched.accept = true;
                regs.mem_arb.grant_latched.requester = req.requester;
                regs.mem_arb.grant_latched.granted_addr_word = req.addr_word;
                regs.mem_arb.grant_latched.granted_len_words = req.len_words;
                regs.mem_arb.grant_latched.reason = 0;

                regs.mem_arb.pending_valid[slot] = false;
                regs.mem_arb.rr_cursor[p] = (u16_t)((slot + 1u) % MEM_REQ_SLOTS);
                return true;
            }
        }
        return false;
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

        // Emit ERR response and metadata words for HALTED debug state.
        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_DBG_HALT));
        data_out.write(regs.halt_info.meta0_word_addr);
        data_out.write(regs.halt_info.meta1_len_words);
    }

    static inline bool handle_debug_cfg_idle(
        TopRegs& regs,
        ac_channel<ac_int<16, false> >& ctrl_rsp,
        ac_channel<ac_int<32, false> >& data_in
    ) {
        u32_t dbg_word_in = top_data_read(data_in);
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
        u32_t dbg_word_in = top_data_read(data_in);
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
        if (!top_data_nb_read(data_in, w)) { return; }

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
        if (!top_data_nb_read(data_in, w)) { return; }

        uint32_t base = (uint32_t)regs.w_base_word.to_uint();
        uint32_t addr = base + idx;
        sram[addr] = w;
        regs.param_count = regs.param_count + 1;

        // HALT when debug trigger matches the k-th LOAD_W word.
        if (regs.debug_armed &&
            ((uint32_t)regs.dbg_trigger_sel.to_uint() == (uint32_t)DBG_TRIGGER_ON_LOADW_COUNT) &&
            (idx == (unsigned)regs.dbg_k_value.to_uint())) {
            regs.debug_armed = false; // One-shot trigger; re-arm via DEBUG_CFG.
            enter_halted_and_emit(regs, ctrl_rsp, data_out);
            return;
        }

        if ((unsigned)regs.param_count.to_uint() == PARAM_WORDS_EXPECTED) {
            // LOAD_W transaction complete.
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
        cfg.eps_bits = LN_EPS_BITS;

        LayerNormBlock(
            sram,
            cfg,
            (u32_t)LN_X_IN_BASE_WORD,
            (u32_t)LN_X_OUT_BASE_WORD,
            (u32_t)LN_GAMMA_BASE_WORD,
            (u32_t)LN_BETA_BASE_WORD
        );
    }

    static inline CfgRegs build_layer_cfg(const TopRegs& regs) {
        CfgRegs cfg;
        cfg.d_model = regs.cfg_d_model;
        cfg.n_heads = regs.cfg_n_heads;
        cfg.d_ffn = regs.cfg_d_ffn;
        cfg.n_layers = regs.cfg_n_layers;

        if ((uint32_t)cfg.d_model.to_uint() == 0u) { cfg.d_model = (u32_t)D_MODEL; }
        if ((uint32_t)cfg.n_heads.to_uint() == 0u) { cfg.n_heads = (u32_t)N_HEAD; }
        if ((uint32_t)cfg.d_ffn.to_uint() == 0u) { cfg.d_ffn = (u32_t)D_FFN; }
        if ((uint32_t)cfg.n_layers.to_uint() == 0u) { cfg.n_layers = (u32_t)N_LAYERS; }
        return cfg;
    }

    static inline u32_t alternate_x_page(u32_t x_base_word) {
        uint32_t x = (uint32_t)x_base_word.to_uint();
        if (x == (uint32_t)sram_map::X_PAGE0_BASE_W) {
            return (u32_t)sram_map::X_PAGE1_BASE_W;
        }
        return (u32_t)sram_map::X_PAGE0_BASE_W;
    }

    static inline void copy_x_words(u32_t* dst, const u32_t* src, uint32_t words) {
        TOP_COPY_X_WORDS_LOOP: for (uint32_t i = 0; i < words; ++i) {
            dst[i] = src[i];
        }
    }

    static inline bool run_p11ac_layer0_top_managed_kv(
        u32_t* sram,
        const CfgRegs& cfg,
        u32_t x_in_base_word,
        const LayerScratch& sc,
        const LayerParamBase& pb,
        bool& fallback_taken
    ) {
        AttnCfg attn_cfg;
        attn_cfg.token_count = (u32_t)ATTN_TOKEN_COUNT;
        attn_cfg.d_model = cfg.d_model;
        attn_cfg.n_heads = cfg.n_heads;
        uint32_t d_model = (uint32_t)attn_cfg.d_model.to_uint();
        uint32_t n_heads = (uint32_t)attn_cfg.n_heads.to_uint();
        if (d_model == 0u) { d_model = (uint32_t)ATTN_D_MODEL; }
        if (n_heads == 0u) { n_heads = (uint32_t)ATTN_N_HEADS; }
        if (n_heads == 0u) { n_heads = 1u; }
        if ((d_model % n_heads) != 0u) { n_heads = 1u; }
        attn_cfg.d_model = (u32_t)d_model;
        attn_cfg.n_heads = (u32_t)n_heads;
        attn_cfg.d_head = (u32_t)(d_model / n_heads);

        // P11AC_MAINLINE_TOP_CALLSITE
        return attn_phasea_top_managed_kv_mainline(
            sram,
            pb.param_base_word,
            x_in_base_word,
            attn_cfg,
            sc.attn,
            fallback_taken
        );
    }

    static inline bool run_p11ad_layer0_top_managed_q(
        u32_t* sram,
        const CfgRegs& cfg,
        u32_t x_in_base_word,
        const LayerScratch& sc,
        const LayerParamBase& pb,
        bool& fallback_taken
    ) {
        AttnCfg attn_cfg;
        attn_cfg.token_count = (u32_t)ATTN_TOKEN_COUNT;
        attn_cfg.d_model = cfg.d_model;
        attn_cfg.n_heads = cfg.n_heads;
        uint32_t d_model = (uint32_t)attn_cfg.d_model.to_uint();
        uint32_t n_heads = (uint32_t)attn_cfg.n_heads.to_uint();
        if (d_model == 0u) { d_model = (uint32_t)ATTN_D_MODEL; }
        if (n_heads == 0u) { n_heads = (uint32_t)ATTN_N_HEADS; }
        if (n_heads == 0u) { n_heads = 1u; }
        if ((d_model % n_heads) != 0u) { n_heads = 1u; }
        attn_cfg.d_model = (u32_t)d_model;
        attn_cfg.n_heads = (u32_t)n_heads;
        attn_cfg.d_head = (u32_t)(d_model / n_heads);

        // P11AD_MAINLINE_TOP_Q_CALLSITE
        return attn_phasea_top_managed_q_mainline(
            sram,
            pb.param_base_word,
            x_in_base_word,
            attn_cfg,
            sc.attn,
            fallback_taken
        );
    }

    static inline void load_mid_or_end_norm_params(
        bool is_mid_norm,
        u32_t* sram,
        uint32_t param_base_word,
        uint32_t gamma_base,
        uint32_t beta_base,
        uint32_t d_model
    ) {
        const uint32_t norm_w_id = is_mid_norm ? 65u : 64u;
        const uint32_t norm_b_id = is_mid_norm ? 17u : 16u;
        const uint32_t norm_w_base = param_base_word + kParamMeta[norm_w_id].offset_w;
        const uint32_t norm_b_base = param_base_word + kParamMeta[norm_b_id].offset_w;

        TOP_NORM_PARAM_COPY_LOOP: for (uint32_t c = 0; c < d_model; ++c) {
            sram[gamma_base + c] = sram[norm_w_base + c];
            sram[beta_base + c] = sram[norm_b_base + c];
        }
    }

    static inline void run_mid_or_end_layernorm(
        bool is_mid_norm,
        const CfgRegs& cfg_regs,
        u32_t* sram,
        u32_t param_base_word,
        u32_t x_in_base_word,
        u32_t x_out_base_word
    ) {
        uint32_t d_model = (uint32_t)cfg_regs.d_model.to_uint();
        if (d_model == 0u) { d_model = (uint32_t)LN_D_MODEL; }

        uint32_t gamma_base = (uint32_t)LN_GAMMA_BASE_WORD;
        uint32_t beta_base = (uint32_t)LN_BETA_BASE_WORD;
        load_mid_or_end_norm_params(is_mid_norm, sram, (uint32_t)param_base_word.to_uint(), gamma_base, beta_base, d_model);

        LayerNormCfg ln_cfg;
        ln_cfg.token_count = (u32_t)LN_TOKEN_COUNT;
        ln_cfg.d_model = (u32_t)d_model;
        ln_cfg.eps_bits = LN_EPS_BITS;

        LayerNormBlock(
            sram,
            ln_cfg,
            x_in_base_word,
            x_out_base_word,
            (u32_t)gamma_base,
            (u32_t)beta_base
        );
    }

    // Top-level layer orchestration boundary.
    // Top owns: layer loop scheduling, X_WORK page alternation, mid/end LN insertion,
    // and latching "mainline taken" vs "fallback taken" status for reviewer-visible checks.
    // TransformerLayer owns: per-layer compute under the base words handed in by Top.
    static inline void run_transformer_layer_loop(TopRegs& regs, u32_t* sram) {
        CfgRegs cfg = build_layer_cfg(regs);
        uint32_t n_layers = (uint32_t)cfg.n_layers.to_uint();
        int mid_index = (int)(n_layers / 2u) - 1;

        u32_t x_in_base = (u32_t)LN_X_OUT_BASE_WORD;
        u32_t x_out_base = alternate_x_page(x_in_base);
        bool mid_valid = false;
        static u32_t mid_snapshot[LN_X_TOTAL_WORDS];
        regs.p11ac_mainline_path_taken = false;
        regs.p11ac_fallback_taken = false;
        regs.p11ad_mainline_q_path_taken = false;
        regs.p11ad_q_fallback_taken = false;

        TOP_LAYER_ORCHESTRATION_LOOP: for (uint32_t lid = 0; lid < n_layers; ++lid) {
            LayerScratch sc = make_layer_scratch(x_in_base);
            LayerParamBase pb = make_layer_param_base(regs.w_base_word, (u32_t)lid);
            bool q_prebuilt_from_top_managed = false;
            bool kv_prebuilt_from_top_managed = false;

            // P11AC mainline wiring is intentionally scoped to the current
            // local-only target layer path to keep integration additive.
            if (lid == 0u) {
                // Layer-0 is the only place where Top prebuilds Q/KV in this local-only path.
                // Fallback meaning is established and latched here for reviewer evidence.
                bool q_fallback_taken = true;
                q_prebuilt_from_top_managed = run_p11ad_layer0_top_managed_q(
                    sram,
                    cfg,
                    x_in_base,
                    sc,
                    pb,
                    q_fallback_taken
                );
                regs.p11ad_mainline_q_path_taken = q_prebuilt_from_top_managed;
                regs.p11ad_q_fallback_taken = q_fallback_taken;

                bool fallback_taken = true;
                kv_prebuilt_from_top_managed = run_p11ac_layer0_top_managed_kv(
                    sram,
                    cfg,
                    x_in_base,
                    sc,
                    pb,
                    fallback_taken
                );
                regs.p11ac_mainline_path_taken = kv_prebuilt_from_top_managed;
                regs.p11ac_fallback_taken = fallback_taken;
            }

            // Dispatch one logical layer with explicit X_WORK/SCRATCH/W_REGION boundaries.
            TransformerLayer(
                sram,
                cfg,
                (u32_t)lid,
                x_in_base,
                x_out_base,
                sc,
                pb,
                kv_prebuilt_from_top_managed,
                q_prebuilt_from_top_managed
            );

            x_in_base = x_out_base;
            x_out_base = alternate_x_page(x_in_base);

            if ((int)lid == mid_index) {
                // mid LN must be out-of-place: current_x -> other_x
                run_mid_or_end_layernorm(true, cfg, sram, regs.w_base_word, x_in_base, x_out_base);
                x_in_base = x_out_base;
                x_out_base = alternate_x_page(x_in_base);

                copy_x_words(mid_snapshot, &sram[(uint32_t)x_in_base.to_uint()], (uint32_t)LN_X_TOTAL_WORDS);
                mid_valid = true;
            }
        }

        // end LN must be out-of-place and always runs before FinalHead.
        run_mid_or_end_layernorm(false, cfg, sram, regs.w_base_word, x_in_base, x_out_base);
        x_in_base = x_out_base;
        x_out_base = alternate_x_page(x_in_base);

        regs.infer_final_x_base_word = x_in_base;
        if (mid_valid) {
            regs.infer_mid_valid = true;
            regs.infer_mid_dump_base_word = x_out_base;
            copy_x_words(&sram[(uint32_t)x_out_base.to_uint()], mid_snapshot, (uint32_t)LN_X_TOTAL_WORDS);
        }
        else {
            regs.infer_mid_valid = false;
            regs.infer_mid_dump_base_word = 0;
        }
    }

    static inline void run_infer_pipeline(TopRegs& regs, u32_t* sram) {
        run_preproc_block(sram);
        run_layernorm_block(sram);
        run_transformer_layer_loop(regs, sram);

        HeadParamBase hp = make_head_param_base(regs.w_base_word);
        FinalHead(
            sram,
            build_layer_cfg(regs),
            regs.infer_final_x_base_word,
            regs.infer_input_shadow,
            regs.infer_logits_base_word,
            regs.infer_xpred_base_word,
            hp
        );
    }

    static inline void infer_emit_outmode_payload(
        const TopRegs& regs,
        ac_channel<ac_int<32, false> >& data_out,
        const u32_t* sram
    ) {
        // Output/write-back boundary owned by Top:
        // Top selects which finalized output region is streamed to data_out.
        uint32_t mode = (uint32_t)regs.outmode.to_uint();
        if (mode == 2u) {
            return;
        }
        if (mode == 0u) {
            uint32_t base = (uint32_t)regs.infer_xpred_base_word.to_uint();
            TOP_OUTMODE_XPRED_WRITEBACK_LOOP: for (unsigned i = 0; i < OUT_WORDS_X_PRED; ++i) {
                data_out.write(sram[base + i]);
            }
            return;
        }
        if (mode == 1u) {
            uint32_t base = (uint32_t)regs.infer_logits_base_word.to_uint();
            TOP_OUTMODE_LOGITS_WRITEBACK_LOOP: for (unsigned i = 0; i < OUT_WORDS_LOGITS; ++i) {
                data_out.write(sram[base + i]);
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
            run_infer_pipeline(regs, sram);
            infer_emit_outmode_payload(regs, data_out, sram);
            regs.state = ST_IDLE;
            ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_INFER));
            return;
        }

        u32_t w;
        if (!top_data_nb_read(data_in, w)) { return; }

        sram[IN_BASE_WORD + idx] = w;
        regs.infer_input_shadow[idx] = w;
        regs.input_count = regs.input_count + 1;

        if ((unsigned)regs.input_count.to_uint() == INFER_IN_WORDS_EXPECTED) {
            run_infer_pipeline(regs, sram);
            infer_emit_outmode_payload(regs, data_out, sram);
            regs.state = ST_IDLE;
            ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_INFER));
        }
    }

    static inline void handle_read_mem(
        TopRegs& regs,
        ac_channel<ac_int<16, false> >& ctrl_rsp,
        ac_channel<ac_int<32, false> >& data_in,
        ac_channel<ac_int<32, false> >& data_out,
        u32_t* sram
    ) {
        // Debug read-back path owned by Top. This does not transfer SRAM ownership.
        // READ_MEM payload: addr_word then len_words, both in u32 words.
        u32_t addr_word_in = top_data_read(data_in);
        u32_t len_words_in = top_data_read(data_in);

        MemReq req = make_empty_mem_req();
        req.valid = true;
        req.requester = REQ_DEBUG_READ_MEM;
        req.prio = PRIO_DEBUG_READ_MEM;
        req.is_write = false;
        req.addr_word = addr_word_in;
        req.len_words = len_words_in;
        req.tag = 0;
        mem_arb_submit(regs, req);
        (void)mem_arb_grant_one(regs);

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

        TOP_READ_MEM_STREAM_LOOP: for (unsigned long long i = 0ull; i < len_words; ++i) {
            unsigned idx = (unsigned)(addr_word + i);
            data_out.write(sram[idx]);
        }
        ctrl_rsp.write(pack_ctrl_rsp_done((uint8_t)OP_READ_MEM));
    }

    // Top dispatch entrypoint for the external 4-channel contract.
    // Top accepts commands, validates state/range constraints, and dispatches block execution.
    // Top functional entrypoint for command dispatch and RX-state servicing.
    static inline void top(
        ac_channel<ac_int<16, false> >& ctrl_cmd,
        ac_channel<ac_int<16, false> >& ctrl_rsp,
        ac_channel<ac_int<32, false> >& data_in,
        ac_channel<ac_int<32, false> >& data_out
    ) {
        TopRegs& regs = top_regs();
        u32_t* sram = top_sram();
        (void)top_in_fifo(); // Ensure in_fifo exists in Top contract.
        (void)mem_arb_grant_one(regs); // Deterministic arbiter stub step.
        refresh_receiver_state(regs);

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
                    u32_t w_base_in = top_data_read(data_in);
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
                        ctrl_rsp.write(pack_ctrl_rsp_ok((uint8_t)OP_SET_W_BASE));
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
                    u32_t outmode_in = top_data_read(data_in);
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
                    handle_read_mem(regs, ctrl_rsp, data_in, data_out, sram);
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
                        // CFG_COMMIT before full CFG payload is a length mismatch error.
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_CFG_LEN_MISMATCH));
                    }
                    else if (!cfg_validate_minimal(regs)) {
                        // Illegal CFG resets session state back to IDLE and clears CFG words.
                        regs.state = ST_IDLE;
                        cfg_session_clear(regs);
                        ctrl_rsp.write(pack_ctrl_rsp_err((uint8_t)ERR_CFG_ILLEGAL));
                    }
                    else {
                        cfg_apply_to_regs(regs);
                        regs.state = ST_IDLE;
                        ctrl_rsp.write(pack_ctrl_rsp_ok((uint8_t)OP_CFG_COMMIT));
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
                    handle_read_mem(regs, ctrl_rsp, data_in, data_out, sram);
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
            // No command this cycle: service one payload word for the active RX state.
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
        refresh_receiver_state(regs);
    }

} // namespace aecct


