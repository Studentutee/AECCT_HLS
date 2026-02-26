#define AECCT_FINAL_TRACE_MODE 1

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "AecctProtocol.h"
#include "AecctTypes.h"
#include "gen/ModelDesc.h"
#include "gen/ModelShapes.h"
#include "gen/SramMap.h"
#include "Top.h"
#include "VerifyTolerance.h"
#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"

struct TopIo {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;
};

static uint32_t f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = f;
    return cvt.u;
}

static float bits_to_f32(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = u;
    return cvt.f;
}

static void fail_now(const char* msg) {
    std::printf("ERROR: %s\n", msg);
    std::exit(1);
}

static void tick(TopIo& io) {
    aecct::top(io.ctrl_cmd, io.ctrl_rsp, io.data_in, io.data_out);
}

static void send_cmd(TopIo& io, uint8_t opcode) {
    io.ctrl_cmd.write(aecct::pack_ctrl_cmd(opcode));
    tick(io);
}

static void send_u32(aecct::data_ch_t& data_in, uint32_t w) {
    data_in.write((aecct::u32_t)w);
}

static void send_data_and_tick(TopIo& io, uint32_t w) {
    send_u32(io.data_in, w);
    tick(io);
}

static void recv_rsp_expect(TopIo& io, uint8_t kind_exp, uint8_t payload_exp, const char* tag) {
    aecct::u16_t w;
    if (!io.ctrl_rsp.nb_read(w)) {
        std::printf("ERROR(%s): missing ctrl_rsp\n", tag);
        std::exit(1);
    }
    uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
    uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);
    if (kind != kind_exp || payload != payload_exp) {
        std::printf("ERROR(%s): ctrl_rsp kind=%u payload=%u, expect kind=%u payload=%u\n",
            tag,
            (unsigned)kind,
            (unsigned)payload,
            (unsigned)kind_exp,
            (unsigned)payload_exp);
        std::exit(1);
    }
}

static void recv_no_rsp(TopIo& io, const char* tag) {
    aecct::u16_t w;
    if (io.ctrl_rsp.nb_read(w)) {
        std::printf("ERROR(%s): unexpected ctrl_rsp kind=%u payload=%u\n",
            tag,
            (unsigned)aecct::unpack_ctrl_rsp_kind(w),
            (unsigned)aecct::unpack_ctrl_rsp_payload(w));
        std::exit(1);
    }
}

static uint32_t recv_data_word(TopIo& io, const char* tag) {
    aecct::u32_t w;
    if (!io.data_out.nb_read(w)) {
        std::printf("ERROR(%s): missing data_out word\n", tag);
        std::exit(1);
    }
    return (uint32_t)w.to_uint();
}

static void recv_data_n(TopIo& io, uint32_t n, uint32_t* out, const char* tag) {
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = recv_data_word(io, tag);
    }
}

static void drain_no_data_out(TopIo& io, const char* tag) {
    aecct::u32_t w;
    if (io.data_out.nb_read(w)) {
        std::printf("ERROR(%s): unexpected data_out word=0x%08X\n", tag, (unsigned)w.to_uint());
        std::exit(1);
    }
}

static uint32_t fold_hash(const uint32_t* words, uint32_t n) {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < n; ++i) {
        h ^= words[i];
        h *= 16777619u;
        h ^= (i * 0x9E3779B9u);
    }
    return h;
}

static uint32_t pattern_word(uint32_t i) {
    return (0xA5000000u | (i & 0x00FFFFFFu));
}

static uint32_t make_debug_word(uint32_t action, uint32_t trigger_sel, uint32_t k_value) {
    return ((action & 0x3u) | ((trigger_sel & 0xFFu) << 8) | ((k_value & 0xFFFFu) << 16));
}

static void build_cfg_words_default(uint32_t cfg_words[EXP_LEN_CFG_WORDS]) {
    for (uint32_t i = 0; i < (uint32_t)EXP_LEN_CFG_WORDS; ++i) {
        cfg_words[i] = 0u;
    }

    cfg_words[CFG_CODE_N] = CODE_N;
    cfg_words[CFG_CODE_K] = CODE_K;
    cfg_words[CFG_CODE_C] = CODE_C;
    cfg_words[CFG_N_NODES] = N_NODES;
    cfg_words[CFG_D_MODEL] = D_MODEL;
    cfg_words[CFG_N_HEAD] = N_HEAD;
    cfg_words[CFG_N_LAYERS] = N_LAYERS;
    cfg_words[CFG_D_FFN] = D_FFN;
    cfg_words[CFG_ENABLE_LPE] = 1u;
    cfg_words[CFG_ENABLE_LPE_TOKEN] = 1u;
    cfg_words[CFG_OUT_MODE] = 2u;
    cfg_words[CFG_RESERVED0] = 0u;
}

static void do_soft_reset(TopIo& io) {
    send_cmd(io, (uint8_t)aecct::OP_SOFT_RESET);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET, "SOFT_RESET");
}

static void do_cfg_commit_with_words(
    TopIo& io,
    const uint32_t* cfg_words,
    uint32_t words_to_send,
    uint8_t rsp_kind_exp,
    uint8_t rsp_payload_exp,
    const char* tag
) {
    send_cmd(io, (uint8_t)aecct::OP_CFG_BEGIN);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN, tag);

    for (uint32_t i = 0; i < words_to_send; ++i) {
        send_data_and_tick(io, cfg_words[i]);
        recv_no_rsp(io, tag);
    }

    send_cmd(io, (uint8_t)aecct::OP_CFG_COMMIT);
    recv_rsp_expect(io, rsp_kind_exp, rsp_payload_exp, tag);
}

static void do_cfg_commit_ok(TopIo& io, const char* tag) {
    uint32_t cfg_words[EXP_LEN_CFG_WORDS];
    build_cfg_words_default(cfg_words);
    do_cfg_commit_with_words(
        io,
        cfg_words,
        (uint32_t)EXP_LEN_CFG_WORDS,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_CFG_COMMIT,
        tag
    );
}

static void do_set_w_base_expect(TopIo& io, uint32_t w_base_word, uint8_t rsp_kind_exp, uint8_t rsp_payload_exp, const char* tag) {
    send_u32(io.data_in, w_base_word);
    send_cmd(io, (uint8_t)aecct::OP_SET_W_BASE);
    recv_rsp_expect(io, rsp_kind_exp, rsp_payload_exp, tag);
}

static void do_set_outmode_expect(TopIo& io, uint32_t outmode, uint8_t rsp_kind_exp, uint8_t rsp_payload_exp, const char* tag) {
    send_u32(io.data_in, outmode);
    send_cmd(io, (uint8_t)aecct::OP_SET_OUTMODE);
    recv_rsp_expect(io, rsp_kind_exp, rsp_payload_exp, tag);
}

static void do_debug_cfg_expect(
    TopIo& io,
    uint32_t dbg_word,
    uint8_t rsp_kind_exp,
    uint8_t rsp_payload_exp,
    const char* tag
) {
    send_u32(io.data_in, dbg_word);
    send_cmd(io, (uint8_t)aecct::OP_DEBUG_CFG);
    recv_rsp_expect(io, rsp_kind_exp, rsp_payload_exp, tag);
}

static void do_read_mem_expect(
    TopIo& io,
    uint32_t addr_word,
    uint32_t len_words,
    const char* tag
) {
    (void)tag;
    send_u32(io.data_in, addr_word);
    send_u32(io.data_in, len_words);
    send_cmd(io, (uint8_t)aecct::OP_READ_MEM);
}

static void do_load_w_full_pattern(TopIo& io, const char* tag) {
    send_cmd(io, (uint8_t)aecct::OP_LOAD_W);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W, tag);

    for (uint32_t i = 0; i < (uint32_t)EXP_LEN_PARAM_WORDS; ++i) {
        send_data_and_tick(io, pattern_word(i));
        if (i + 1u < (uint32_t)EXP_LEN_PARAM_WORDS) {
            recv_no_rsp(io, tag);
        }
        else {
            recv_rsp_expect(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W, tag);
        }
    }
}

static void do_infer_sample0(TopIo& io, const char* tag) {
    send_cmd(io, (uint8_t)aecct::OP_INFER);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_INFER, tag);

    const uint32_t in_words = (uint32_t)EXP_LEN_INFER_IN_WORDS;
    for (uint32_t i = 0; i < in_words; ++i) {
        float v = (float)trace_input_y_step0_tensor[i];
        send_data_and_tick(io, f32_to_bits(v));
        if (i + 1u < in_words) {
            recv_no_rsp(io, tag);
        }
        else {
            recv_rsp_expect(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_INFER, tag);
        }
    }
}

static int compare_output(
    const char* name,
    const uint32_t* got_words,
    const double* ref_tensor,
    uint32_t sample_idx,
    uint32_t words,
    double tol
) {
    bool exact_ok = true;
    double max_abs_err = 0.0;
    uint32_t max_idx = 0u;
    uint32_t max_got = 0u;
    uint32_t max_ref = 0u;

    for (uint32_t i = 0; i < words; ++i) {
        float ref_f = (float)ref_tensor[sample_idx * words + i];
        uint32_t ref_bits = f32_to_bits(ref_f);
        uint32_t got_bits = got_words[i];
        if (got_bits != ref_bits) {
            exact_ok = false;
            double err = std::fabs((double)bits_to_f32(got_bits) - (double)ref_f);
            if (err > max_abs_err) {
                max_abs_err = err;
                max_idx = i;
                max_got = got_bits;
                max_ref = ref_bits;
            }
        }
    }

    if (exact_ok) {
        std::printf("PASS: %s exact-bit match\n", name);
        return 0;
    }

    std::printf("INFO: %s exact mismatch, max_abs_err=%.9g idx=%u\n",
        name, max_abs_err, (unsigned)max_idx);
    if (max_abs_err <= tol) {
        std::printf("PASS: %s abs_err<=%.1e\n", name, tol);
        return 0;
    }

    std::printf("ERROR: %s mismatch idx=%u got=0x%08X ref=0x%08X\n",
        name, (unsigned)max_idx, (unsigned)max_got, (unsigned)max_ref);
    return 1;
}

static void case1_cfg_len_mismatch() {
    TopIo io;
    do_soft_reset(io);

    uint32_t cfg_words[EXP_LEN_CFG_WORDS];
    build_cfg_words_default(cfg_words);
    do_cfg_commit_with_words(
        io,
        cfg_words,
        (uint32_t)EXP_LEN_CFG_WORDS - 1u,
        (uint8_t)aecct::RSP_ERR,
        (uint8_t)aecct::ERR_CFG_LEN_MISMATCH,
        "case1_cfg_len_mismatch"
    );
    std::printf("PASS: case1_cfg_len_mismatch\n");
}

static void case2_cfg_illegal() {
    TopIo io;
    do_soft_reset(io);

    uint32_t cfg_words[EXP_LEN_CFG_WORDS];
    build_cfg_words_default(cfg_words);
    cfg_words[CFG_D_MODEL] = D_MODEL + 1u; // d_model % n_heads != 0

    do_cfg_commit_with_words(
        io,
        cfg_words,
        (uint32_t)EXP_LEN_CFG_WORDS,
        (uint8_t)aecct::RSP_ERR,
        (uint8_t)aecct::ERR_CFG_ILLEGAL,
        "case2_cfg_illegal"
    );
    std::printf("PASS: case2_cfg_illegal\n");
}

static void case3_set_outmode_bad_arg() {
    TopIo io;
    do_soft_reset(io);
    do_cfg_commit_ok(io, "case3_cfg_ok");

    do_set_outmode_expect(
        io,
        3u,
        (uint8_t)aecct::RSP_ERR,
        (uint8_t)aecct::ERR_BAD_ARG,
        "case3_set_outmode_bad_arg"
    );
    std::printf("PASS: case3_set_outmode_bad_arg\n");
}

static void case4_read_mem_range_check() {
    TopIo io;
    do_soft_reset(io);

    do_read_mem_expect(
        io,
        (uint32_t)sram_map::SRAM_WORDS_TOTAL - 4u,
        8u,
        "case4_read_mem_range_check"
    );
    recv_rsp_expect(io, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_MEM_RANGE, "case4_read_mem_rsp");
    drain_no_data_out(io, "case4_read_mem_no_data");
    std::printf("PASS: case4_read_mem_range_check\n");
}

static void case5_set_w_base_range_align() {
    TopIo io;
    do_soft_reset(io);

    uint32_t out_of_range_base = (uint32_t)sram_map::W_REGION_BASE + (uint32_t)sram_map::W_REGION_WORDS;
    do_set_w_base_expect(
        io,
        out_of_range_base,
        (uint8_t)aecct::RSP_ERR,
        (uint8_t)aecct::ERR_PARAM_BASE_RANGE,
        "case5_set_w_base_range"
    );

    if ((uint32_t)aecct::PARAM_ALIGN_WORDS > 1u) {
        uint32_t misaligned_base = (uint32_t)sram_map::W_REGION_BASE + 1u;
        do_set_w_base_expect(
            io,
            misaligned_base,
            (uint8_t)aecct::RSP_ERR,
            (uint8_t)aecct::ERR_PARAM_BASE_ALIGN,
            "case5_set_w_base_align"
        );
    }
    std::printf("PASS: case5_set_w_base_range_align\n");
}

static void case6_load_w_bad_state() {
    TopIo io;
    do_soft_reset(io);

    send_cmd(io, (uint8_t)aecct::OP_LOAD_W);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE, "case6_load_w_bad_state");
    std::printf("PASS: case6_load_w_bad_state\n");
}

static void case7_load_w_mem_range() {
    TopIo io;
    do_soft_reset(io);

    uint32_t base = (uint32_t)sram_map::W_REGION_BASE +
        (uint32_t)sram_map::W_REGION_WORDS -
        ((uint32_t)EXP_LEN_PARAM_WORDS / 2u);
    uint32_t align = (uint32_t)aecct::PARAM_ALIGN_WORDS;
    if (align == 0u) {
        fail_now("case7 align is zero");
    }
    base = base - (base % align);

    do_set_w_base_expect(
        io,
        base,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_W_BASE,
        "case7_set_w_base_done"
    );

    send_cmd(io, (uint8_t)aecct::OP_LOAD_W);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_MEM_RANGE, "case7_load_w_mem_range");
    std::printf("PASS: case7_load_w_mem_range\n");
}

static void case8_infer_bad_state() {
    TopIo io;
    do_soft_reset(io);

    send_cmd(io, (uint8_t)aecct::OP_INFER);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE, "case8_infer_bad_state");
    std::printf("PASS: case8_infer_bad_state\n");
}

static void case9_end_to_end_all_outmodes() {
    TopIo io;
    static uint32_t logits_got[EXP_LEN_OUT_LOGITS_WORDS];
    static uint32_t xpred_got[EXP_LEN_OUT_XPRED_WORDS];

    do_soft_reset(io);
    do_cfg_commit_ok(io, "case9_cfg_ok");
    do_set_w_base_expect(
        io,
        (uint32_t)sram_map::PARAM_BASE_DEFAULT,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_W_BASE,
        "case9_set_w_base"
    );
    do_load_w_full_pattern(io, "case9_load_w");

    do_set_outmode_expect(
        io,
        1u,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_OUTMODE,
        "case9_set_outmode_logits"
    );
    do_infer_sample0(io, "case9_infer_logits");
    recv_data_n(io, (uint32_t)EXP_LEN_OUT_LOGITS_WORDS, logits_got, "case9_logits_data");
    if (compare_output(
        "case9_logits",
        logits_got,
        trace_output_logits_step0_tensor,
        0u,
        (uint32_t)EXP_LEN_OUT_LOGITS_WORDS,
        EPS_LOGITS
    ) != 0) {
        std::exit(1);
    }
    drain_no_data_out(io, "case9_logits_no_extra");

    do_set_outmode_expect(
        io,
        0u,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_OUTMODE,
        "case9_set_outmode_xpred"
    );
    do_infer_sample0(io, "case9_infer_xpred");
    recv_data_n(io, (uint32_t)EXP_LEN_OUT_XPRED_WORDS, xpred_got, "case9_xpred_data");
    if (compare_output(
        "case9_xpred",
        xpred_got,
        trace_output_x_pred_step0_tensor,
        0u,
        (uint32_t)EXP_LEN_OUT_XPRED_WORDS,
        EPS_LOGITS
    ) != 0) {
        std::exit(1);
    }
    drain_no_data_out(io, "case9_xpred_no_extra");

    do_set_outmode_expect(
        io,
        2u,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_OUTMODE,
        "case9_set_outmode_none"
    );
    do_infer_sample0(io, "case9_infer_none");
    drain_no_data_out(io, "case9_none_no_data");

    std::printf("PASS: case9_end_to_end_all_outmodes\n");
}

static void case10_determinism_endurance() {
    TopIo io;
    static uint32_t logits_got[EXP_LEN_OUT_LOGITS_WORDS];
    const uint32_t runs = 10u;
    bool first = true;
    uint32_t ref_hash = 0u;

    do_soft_reset(io);
    do_cfg_commit_ok(io, "case10_cfg_ok");
    do_set_w_base_expect(
        io,
        (uint32_t)sram_map::PARAM_BASE_DEFAULT,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_W_BASE,
        "case10_set_w_base"
    );
    do_load_w_full_pattern(io, "case10_load_w");
    do_set_outmode_expect(
        io,
        1u,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_OUTMODE,
        "case10_set_outmode_logits"
    );

    for (uint32_t r = 0; r < runs; ++r) {
        do_infer_sample0(io, "case10_infer");
        recv_data_n(io, (uint32_t)EXP_LEN_OUT_LOGITS_WORDS, logits_got, "case10_logits_data");
        drain_no_data_out(io, "case10_logits_no_extra");

        uint32_t h = fold_hash(logits_got, (uint32_t)EXP_LEN_OUT_LOGITS_WORDS);
        if (first) {
            ref_hash = h;
            first = false;
        }
        else if (h != ref_hash) {
            std::printf("ERROR: case10 hash mismatch at run=%u got=0x%08X ref=0x%08X\n",
                (unsigned)r, (unsigned)h, (unsigned)ref_hash);
            std::exit(1);
        }
    }

    std::printf("PASS: case10_determinism_endurance (runs=%u hash=0x%08X)\n",
        (unsigned)runs, (unsigned)ref_hash);
}

static void case11_debug_halt_read_resume() {
    TopIo io;
    const uint32_t halt_k = 4u;
    uint32_t meta0 = 0u;
    uint32_t meta1 = 0u;
    static uint32_t read_back[256];

    do_soft_reset(io);
    do_cfg_commit_ok(io, "case11_cfg_ok");
    do_set_w_base_expect(
        io,
        (uint32_t)sram_map::PARAM_BASE_DEFAULT,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_W_BASE,
        "case11_set_w_base"
    );

    do_debug_cfg_expect(
        io,
        make_debug_word(1u, 1u, halt_k),
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_DEBUG_CFG,
        "case11_debug_arm"
    );

    send_cmd(io, (uint8_t)aecct::OP_LOAD_W);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W, "case11_load_w_ok");

    for (uint32_t i = 0; i <= halt_k; ++i) {
        send_data_and_tick(io, pattern_word(i));
        if (i < halt_k) {
            recv_no_rsp(io, "case11_pre_halt_no_rsp");
        }
        else {
            recv_rsp_expect(io, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_DBG_HALT, "case11_halt_rsp");
            meta0 = recv_data_word(io, "case11_meta0");
            meta1 = recv_data_word(io, "case11_meta1");
        }
    }

    if (meta0 != (uint32_t)sram_map::PARAM_BASE_DEFAULT) {
        std::printf("ERROR: case11 meta0 mismatch got=0x%08X expect=0x%08X\n",
            (unsigned)meta0, (unsigned)sram_map::PARAM_BASE_DEFAULT);
        std::exit(1);
    }
    if (meta1 != (uint32_t)aecct::DBG_META1_LEN_WORDS) {
        std::printf("ERROR: case11 meta1 mismatch got=%u expect=%u\n",
            (unsigned)meta1, (unsigned)aecct::DBG_META1_LEN_WORDS);
        std::exit(1);
    }
    if (meta1 > 256u) {
        fail_now("case11 meta1 too large");
    }

    do_read_mem_expect(
        io,
        meta0,
        meta1,
        "case11_read_mem"
    );
    recv_data_n(io, meta1, read_back, "case11_read_mem_data");
    recv_rsp_expect(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM, "case11_read_mem_done");

    for (uint32_t i = 0; i <= halt_k; ++i) {
        uint32_t exp_w = pattern_word(i);
        if (read_back[i] != exp_w) {
            std::printf("ERROR: case11 read_back mismatch idx=%u got=0x%08X expect=0x%08X\n",
                (unsigned)i, (unsigned)read_back[i], (unsigned)exp_w);
            std::exit(1);
        }
    }

    do_debug_cfg_expect(
        io,
        make_debug_word(2u, 0u, 0u),
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_DEBUG_CFG,
        "case11_debug_resume"
    );

    for (uint32_t i = halt_k + 1u; i < (uint32_t)EXP_LEN_PARAM_WORDS; ++i) {
        send_data_and_tick(io, pattern_word(i));
        if (i + 1u < (uint32_t)EXP_LEN_PARAM_WORDS) {
            recv_no_rsp(io, "case11_load_continue_no_rsp");
        }
        else {
            recv_rsp_expect(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W, "case11_load_done");
        }
    }

    std::printf("PASS: case11_debug_halt_read_resume\n");
}

static void case12_halted_illegal_cmd() {
    TopIo io;
    const uint32_t halt_k = 2u;

    do_soft_reset(io);
    do_cfg_commit_ok(io, "case12_cfg_ok");
    do_set_w_base_expect(
        io,
        (uint32_t)sram_map::PARAM_BASE_DEFAULT,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_W_BASE,
        "case12_set_w_base"
    );
    do_debug_cfg_expect(
        io,
        make_debug_word(1u, 1u, halt_k),
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_DEBUG_CFG,
        "case12_debug_arm"
    );

    send_cmd(io, (uint8_t)aecct::OP_LOAD_W);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W, "case12_load_ok");

    for (uint32_t i = 0; i <= halt_k; ++i) {
        send_data_and_tick(io, pattern_word(i));
        if (i < halt_k) {
            recv_no_rsp(io, "case12_no_rsp_before_halt");
        }
        else {
            recv_rsp_expect(io, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_DBG_HALT, "case12_halt_rsp");
            (void)recv_data_word(io, "case12_meta0");
            (void)recv_data_word(io, "case12_meta1");
        }
    }

    send_cmd(io, (uint8_t)aecct::OP_CFG_BEGIN);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE, "case12_cfg_begin_bad_state");

    send_cmd(io, (uint8_t)aecct::OP_INFER);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE, "case12_infer_bad_state");

    do_debug_cfg_expect(
        io,
        make_debug_word(2u, 0u, 0u),
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_DEBUG_CFG,
        "case12_resume"
    );
    std::printf("PASS: case12_halted_illegal_cmd\n");
}

static void case13_resume_non_halted() {
    TopIo io;
    do_soft_reset(io);

    do_debug_cfg_expect(
        io,
        make_debug_word(2u, 0u, 0u),
        (uint8_t)aecct::RSP_ERR,
        (uint8_t)aecct::ERR_BAD_ARG,
        "case13_resume_non_halted"
    );
    std::printf("PASS: case13_resume_non_halted\n");
}

int main() {
    case1_cfg_len_mismatch();
    case2_cfg_illegal();
    case3_set_outmode_bad_arg();
    case4_read_mem_range_check();
    case5_set_w_base_range_align();
    case6_load_w_bad_state();
    case7_load_w_mem_range();
    case8_infer_bad_state();
    case9_end_to_end_all_outmodes();
    case10_determinism_endurance();
    case11_debug_halt_read_resume();
    case12_halted_illegal_cmd();
    case13_resume_non_halted();

    std::printf("PASS: tb_regress_m14\n");
    return 0;
}

