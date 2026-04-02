// P00-011AH: full-loop local end-to-end bring-up for run_transformer_layer_loop (local-only).

#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <vector>

#include "AecctProtocol.h"
#include "tb_p11aeaf_common.h"

#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#define AECCT_HAS_SCVERIFY 1
#else
#define AECCT_HAS_SCVERIFY 0
#endif

#if !AECCT_HAS_SCVERIFY
#ifndef CCS_MAIN
#define CCS_MAIN(...) int main(__VA_ARGS__)
#endif
#ifndef CCS_RETURN
#define CCS_RETURN(x) return (x)
#endif
#endif

namespace {

class TbP11ahFullLoopLocalE2e {
public:
    int run_all() {
        if (!init()) {
            return 1;
        }
        if (!run_top_infer_marker_checker()) {
            return 1;
        }
        if (!run_full_loop_mainline()) {
            return 1;
        }
        if (!run_full_loop_reference()) {
            return 1;
        }
        if (!compare_final_x_deterministic()) {
            return 1;
        }
        if (!scan_no_nonfinite_in_key_spans()) {
            return 1;
        }

        std::printf("PASS: tb_full_loop_local_e2e_p11ah\n");
        return 0;
    }

private:
    struct TopIo {
        aecct::ctrl_ch_t ctrl_cmd;
        aecct::ctrl_ch_t ctrl_rsp;
        aecct::data_ch_t data_in;
        aecct::data_ch_t data_out;
    };

    std::vector<aecct::u32_t> sram_mainline_;
    std::vector<aecct::u32_t> sram_ref_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::CfgRegs cfg_;
    aecct::LayerScratch sc_;
    aecct::TopRegs regs_;
    aecct::TopRegs regs_ref_;

    static uint32_t f32_to_bits(float f) {
        union {
            float f;
            uint32_t u;
        } cvt;
        cvt.f = f;
        return cvt.u;
    }

    static void init_full_x_rows(std::vector<aecct::u32_t>& sram) {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t x_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
        for (uint32_t t = 0u; t < token_count; ++t) {
            const uint32_t row_base = x_base + t * d_model;
            for (uint32_t i = 0u; i < d_model; ++i) {
                const int32_t v = (int32_t)((t + 3u) * 17u + (i + 5u) * 11u) - 211;
                const float f = ((float)v) * 0.015625f;
                sram[row_base + i] = (aecct::u32_t)f32_to_bits(f);
            }
        }
    }

    static bool is_nonfinite_bits(uint32_t bits) {
        return ((bits & 0x7F800000u) == 0x7F800000u);
    }

    static void top_tick(TopIo& io) {
        aecct::top(io.ctrl_cmd, io.ctrl_rsp, io.data_in, io.data_out);
    }

    static void top_send_cmd(TopIo& io, uint8_t op) {
        io.ctrl_cmd.write(aecct::pack_ctrl_cmd(op));
        top_tick(io);
    }

    static bool top_expect_no_rsp(TopIo& io, const char* tag) {
        aecct::u16_t w;
        if (io.ctrl_rsp.nb_read(w)) {
            std::printf("[p11ah][FAIL] %s unexpected rsp kind=%u payload=%u\n",
                tag,
                (unsigned)aecct::unpack_ctrl_rsp_kind(w),
                (unsigned)aecct::unpack_ctrl_rsp_payload(w));
            return false;
        }
        return true;
    }

    static bool top_expect_rsp_exact(
        TopIo& io,
        uint8_t kind_exp,
        uint8_t payload_exp,
        const char* tag
    ) {
        aecct::u16_t w;
        if (!io.ctrl_rsp.nb_read(w)) {
            std::printf("[p11ah][FAIL] %s missing ctrl_rsp\n", tag);
            return false;
        }
        const uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
        const uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);
        if (kind != kind_exp || payload != payload_exp) {
            std::printf("[p11ah][FAIL] %s rsp mismatch kind=%u payload=%u expect_kind=%u expect_payload=%u\n",
                tag,
                (unsigned)kind,
                (unsigned)payload,
                (unsigned)kind_exp,
                (unsigned)payload_exp);
            return false;
        }
        return true;
    }

    static bool top_expect_rsp_kind_either(
        TopIo& io,
        uint8_t kind_exp0,
        uint8_t kind_exp1,
        uint8_t payload_exp,
        const char* tag
    ) {
        aecct::u16_t w;
        if (!io.ctrl_rsp.nb_read(w)) {
            std::printf("[p11ah][FAIL] %s missing ctrl_rsp\n", tag);
            return false;
        }
        const uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
        const uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);
        if ((kind != kind_exp0 && kind != kind_exp1) || payload != payload_exp) {
            std::printf(
                "[p11ah][FAIL] %s rsp mismatch kind=%u payload=%u expect_kind=%u|%u expect_payload=%u\n",
                tag,
                (unsigned)kind,
                (unsigned)payload,
                (unsigned)kind_exp0,
                (unsigned)kind_exp1,
                (unsigned)payload_exp);
            return false;
        }
        return true;
    }

    static void dump_top_infer_marker_snapshot(
        const char* banner,
        bool ac_mainline,
        bool ad_mainline,
        bool ae_mainline,
        bool af_mainline,
        bool ac_fallback,
        bool ad_fallback,
        bool ae_fallback,
        bool af_fallback
    ) {
        std::printf("%s\n", banner);
        std::printf("  AC(KV stage): mainline=%u fallback=%u\n",
            (unsigned)(ac_mainline ? 1u : 0u),
            (unsigned)(ac_fallback ? 1u : 0u));
        std::printf("  AD(Q stage): mainline=%u fallback=%u\n",
            (unsigned)(ad_mainline ? 1u : 0u),
            (unsigned)(ad_fallback ? 1u : 0u));
        std::printf("  AE(Score stage): mainline=%u fallback=%u\n",
            (unsigned)(ae_mainline ? 1u : 0u),
            (unsigned)(ae_fallback ? 1u : 0u));
        std::printf("  AF(SoftmaxOut stage): mainline=%u fallback=%u\n",
            (unsigned)(af_mainline ? 1u : 0u),
            (unsigned)(af_fallback ? 1u : 0u));
    }

    bool run_top_infer_marker_checker() {
        TopIo io;

        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        std::vector<aecct::u32_t> sram_seed((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_seed, payloads_, param_base);

        std::vector<aecct::u32_t> param_words((uint32_t)EXP_LEN_PARAM_WORDS, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_PARAM_WORDS; ++i) {
            param_words[i] = sram_seed[param_base + i];
        }

        top_send_cmd(io, (uint8_t)aecct::OP_SOFT_RESET);
        if (!top_expect_rsp_exact(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET, "top_soft_reset")) {
            return false;
        }

        uint32_t cfg_words[EXP_LEN_CFG_WORDS];
        for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_CFG_WORDS; ++i) {
            cfg_words[i] = 0u;
        }
        cfg_words[CFG_CODE_N] = CODE_N;
        cfg_words[CFG_CODE_K] = CODE_K;
        cfg_words[CFG_CODE_C] = CODE_C;
        cfg_words[CFG_N_NODES] = N_NODES;
        cfg_words[CFG_D_MODEL] = (uint32_t)p11aeaf_tb::kTileWords;
        cfg_words[CFG_N_HEAD] = 8u;
        cfg_words[CFG_N_LAYERS] = 1u;
        cfg_words[CFG_D_FFN] = (uint32_t)p11aeaf_tb::kTileWords;
        cfg_words[CFG_ENABLE_LPE] = 1u;
        cfg_words[CFG_ENABLE_LPE_TOKEN] = 1u;
        cfg_words[CFG_OUT_MODE] = 2u;
        cfg_words[CFG_RESERVED0] = 0u;

        top_send_cmd(io, (uint8_t)aecct::OP_CFG_BEGIN);
        if (!top_expect_rsp_exact(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN, "top_cfg_begin")) {
            return false;
        }
        for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_CFG_WORDS; ++i) {
            io.data_in.write((aecct::u32_t)cfg_words[i]);
            top_tick(io);
            if (!top_expect_no_rsp(io, "top_cfg_ingest")) {
                return false;
            }
        }
        top_send_cmd(io, (uint8_t)aecct::OP_CFG_COMMIT);
        if (!top_expect_rsp_kind_either(
            io,
            (uint8_t)aecct::RSP_OK,
            (uint8_t)aecct::RSP_DONE,
            (uint8_t)aecct::OP_CFG_COMMIT,
            "top_cfg_commit")) {
            return false;
        }

        io.data_in.write((aecct::u32_t)param_base);
        top_send_cmd(io, (uint8_t)aecct::OP_SET_W_BASE);
        if (!top_expect_rsp_kind_either(
            io,
            (uint8_t)aecct::RSP_OK,
            (uint8_t)aecct::RSP_DONE,
            (uint8_t)aecct::OP_SET_W_BASE,
            "top_set_w_base")) {
            return false;
        }

        top_send_cmd(io, (uint8_t)aecct::OP_LOAD_W);
        if (!top_expect_rsp_exact(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W, "top_load_w_begin")) {
            return false;
        }
        for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_PARAM_WORDS; ++i) {
            io.data_in.write(param_words[i]);
            top_tick(io);
            if (i + 1u < (uint32_t)EXP_LEN_PARAM_WORDS) {
                if (!top_expect_no_rsp(io, "top_load_w_ingest")) {
                    return false;
                }
            } else {
                if (!top_expect_rsp_exact(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W, "top_load_w_done")) {
                    return false;
                }
            }
        }

        io.data_in.write((aecct::u32_t)2u);
        top_send_cmd(io, (uint8_t)aecct::OP_SET_OUTMODE);
        if (!top_expect_rsp_exact(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE, "top_set_outmode")) {
            return false;
        }

        top_send_cmd(io, (uint8_t)aecct::OP_INFER);
        if (!top_expect_rsp_exact(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_INFER, "top_infer_begin")) {
            return false;
        }
        for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_INFER_IN_WORDS; ++i) {
            const int32_t sv = (int32_t)(i & 31u) - 16;
            const float fv = ((float)sv) * 0.03125f;
            io.data_in.write((aecct::u32_t)f32_to_bits(fv));
            top_tick(io);
            if (i + 1u < (uint32_t)EXP_LEN_INFER_IN_WORDS) {
                if (!top_expect_no_rsp(io, "top_infer_ingest")) {
                    return false;
                }
            } else {
                if (!top_expect_rsp_exact(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_INFER, "top_infer_done")) {
                    return false;
                }
            }
        }

        const bool ac_mainline = aecct::top_peek_p11ac_mainline_path_taken();
        const bool ad_mainline = aecct::top_peek_p11ad_mainline_q_path_taken();
        const bool ae_mainline = aecct::top_peek_p11ae_mainline_score_path_taken();
        const bool af_mainline = aecct::top_peek_p11af_mainline_softmax_output_path_taken();
        const bool ac_fallback = aecct::top_peek_p11ac_fallback_taken();
        const bool ad_fallback = aecct::top_peek_p11ad_q_fallback_taken();
        const bool ae_fallback = aecct::top_peek_p11ae_score_fallback_taken();
        const bool af_fallback = aecct::top_peek_p11af_softmax_output_fallback_taken();

        const bool pass =
            ac_mainline &&
            ad_mainline &&
            ae_mainline &&
            af_mainline &&
            !ac_fallback &&
            !ad_fallback &&
            !ae_fallback &&
            !af_fallback;

        if (!pass) {
            dump_top_infer_marker_snapshot(
                "P11AH_TOP_INFER_LID0_ATTN_MARKER_CHECKER FAIL",
                ac_mainline,
                ad_mainline,
                ae_mainline,
                af_mainline,
                ac_fallback,
                ad_fallback,
                ae_fallback,
                af_fallback);
            if (!ac_mainline || ac_fallback) {
                std::printf("  FAIL_STAGE: AC = KV stage\n");
            }
            if (!ad_mainline || ad_fallback) {
                std::printf("  FAIL_STAGE: AD = Q stage\n");
            }
            if (!ae_mainline || ae_fallback) {
                std::printf("  FAIL_STAGE: AE = Score stage\n");
            }
            if (!af_mainline || af_fallback) {
                std::printf("  FAIL_STAGE: AF = SoftmaxOut stage\n");
            }
            return false;
        }

        dump_top_infer_marker_snapshot(
            "P11AH_TOP_INFER_LID0_ATTN_MARKER_CHECKER PASS",
            ac_mainline,
            ad_mainline,
            ae_mainline,
            af_mainline,
            ac_fallback,
            ad_fallback,
            ae_fallback,
            af_fallback);
        return true;
    }

    bool init() {
        sram_mainline_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        sram_ref_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        init_full_x_rows(sram_mainline_);
        init_full_x_rows(sram_ref_);

        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads_)) {
            std::printf("[p11ah][FAIL] payload preparation failed\n");
            return false;
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_mainline_, payloads_, param_base);
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_ref_, payloads_, param_base);

        cfg_ = p11aeaf_tb::build_cfg();
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);

        regs_.clear();
        regs_.w_base_set = true;
        regs_.w_base_word = (aecct::u32_t)param_base;
        regs_.cfg_d_model = cfg_.d_model;
        regs_.cfg_n_heads = cfg_.n_heads;
        regs_.cfg_d_ffn = cfg_.d_ffn;
        regs_.cfg_n_layers = (aecct::u32_t)1u;
        regs_.cfg_ready = true;

        regs_ref_.clear();
        regs_ref_.w_base_set = true;
        regs_ref_.w_base_word = (aecct::u32_t)param_base;
        regs_ref_.cfg_d_model = cfg_.d_model;
        regs_ref_.cfg_n_heads = cfg_.n_heads;
        regs_ref_.cfg_d_ffn = cfg_.d_ffn;
        regs_ref_.cfg_n_layers = (aecct::u32_t)1u;
        regs_ref_.cfg_ready = true;
        return true;
    }

    bool run_full_loop_mainline() {
        aecct::run_transformer_layer_loop(regs_, sram_mainline_.data());

        const bool mainline_all_taken =
            regs_.p11ac_mainline_path_taken &&
            regs_.p11ad_mainline_q_path_taken &&
            regs_.p11ae_mainline_score_path_taken &&
            regs_.p11af_mainline_softmax_output_path_taken;
        if (!mainline_all_taken) {
            std::printf("[p11ah][FAIL] full-loop mainline flags not all true (ac=%d ad=%d ae=%d af=%d)\n",
                regs_.p11ac_mainline_path_taken ? 1 : 0,
                regs_.p11ad_mainline_q_path_taken ? 1 : 0,
                regs_.p11ae_mainline_score_path_taken ? 1 : 0,
                regs_.p11af_mainline_softmax_output_path_taken ? 1 : 0);
            return false;
        }

        const bool fallback_taken =
            regs_.p11ac_fallback_taken ||
            regs_.p11ad_q_fallback_taken ||
            regs_.p11ae_score_fallback_taken ||
            regs_.p11af_softmax_output_fallback_taken;
        if (fallback_taken) {
            std::printf("[p11ah][FAIL] fallback detected in full-loop path (ac=%d ad=%d ae=%d af=%d)\n",
                regs_.p11ac_fallback_taken ? 1 : 0,
                regs_.p11ad_q_fallback_taken ? 1 : 0,
                regs_.p11ae_score_fallback_taken ? 1 : 0,
                regs_.p11af_softmax_output_fallback_taken ? 1 : 0);
            return false;
        }

        std::printf("FULL_LOOP_MAINLINE_PATH_TAKEN PASS\n");
        std::printf("fallback_taken = false\n");
        std::printf("FULL_LOOP_FALLBACK_NOT_TAKEN PASS\n");
        return true;
    }

    bool run_full_loop_reference() {
        aecct::run_transformer_layer_loop(regs_ref_, sram_ref_.data());
        return true;
    }

    bool compare_final_x_deterministic() {
        const uint32_t words = (uint32_t)aecct::LN_X_TOTAL_WORDS;
        const uint32_t base = (uint32_t)regs_.infer_final_x_base_word.to_uint();
        const uint32_t ref_base = (uint32_t)regs_ref_.infer_final_x_base_word.to_uint();
        if (base != ref_base) {
            std::printf("[p11ah][FAIL] final_x base mismatch mainline=%u ref=%u\n",
                (unsigned)base, (unsigned)ref_base);
            return false;
        }
        for (uint32_t i = 0u; i < words; ++i) {
            const uint32_t got = (uint32_t)sram_mainline_[base + i].to_uint();
            const uint32_t ref = (uint32_t)sram_ref_[base + i].to_uint();
            if (got != ref) {
                std::printf("[p11ah][FAIL] final_x deterministic compare mismatch idx=%u got=0x%08X ref=0x%08X\n",
                    (unsigned)i, (unsigned)got, (unsigned)ref);
                return false;
            }
        }
        std::printf("FULL_LOOP_FINAL_X_DETERMINISTIC_COMPARE PASS\n");
        return true;
    }

    bool scan_no_nonfinite_in_key_spans() {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t words = token_count * d_model;
        const uint32_t score_words = (uint32_t)aecct::ATTN_N_HEADS * token_count;
        const uint32_t attn_out_base = (uint32_t)sc_.attn_out_base_word.to_uint();
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        const uint32_t final_x_base = (uint32_t)regs_.infer_final_x_base_word.to_uint();

        for (uint32_t i = 0u; i < words; ++i) {
            const uint32_t b = (uint32_t)sram_mainline_[attn_out_base + i].to_uint();
            if (is_nonfinite_bits(b)) {
                std::printf("[p11ah][FAIL] non-finite in attn_out addr=%u bits=0x%08X\n",
                    (unsigned)(attn_out_base + i), (unsigned)b);
                return false;
            }
        }
        for (uint32_t i = 0u; i < score_words; ++i) {
            const uint32_t b = (uint32_t)sram_mainline_[score_base + i].to_uint();
            if (is_nonfinite_bits(b)) {
                std::printf("[p11ah][FAIL] non-finite in score addr=%u bits=0x%08X\n",
                    (unsigned)(score_base + i), (unsigned)b);
                return false;
            }
        }
        for (uint32_t i = 0u; i < words; ++i) {
            const uint32_t b = (uint32_t)sram_mainline_[final_x_base + i].to_uint();
            if (is_nonfinite_bits(b)) {
                std::printf("[p11ah][FAIL] non-finite in final_x addr=%u bits=0x%08X\n",
                    (unsigned)(final_x_base + i), (unsigned)b);
                return false;
            }
        }

        std::printf("FULL_LOOP_FINITE_SCAN PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbP11ahFullLoopLocalE2e tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__
