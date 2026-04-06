// tb_backup_io8_loadw_infer_smoke.cpp
// Backup profile next-step smoke:
// - Top external io8 boundary
// - minimal CFG + SET_W_BASE + LOAD_W + INFER narrow path
// - output non-empty / byte-count check / deterministic rerun

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "AecctProtocol.h"
#include "AecctUtil.h"
#include "Top.h"

namespace {

struct Io8Top {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data8_ch_t data_in;
    aecct::data8_ch_t data_out;
};

static void fail(const char* msg) {
    std::printf("[backup_io8][FAIL] %s\n", msg);
    std::exit(1);
}

static void top_tick(Io8Top& io) {
    aecct::top(io.ctrl_cmd, io.ctrl_rsp, io.data_in, io.data_out);
}

static void push_u32_le(Io8Top& io, uint32_t w) {
    io.data_in.write((aecct::u8_t)(w & 0xFFu));
    io.data_in.write((aecct::u8_t)((w >> 8) & 0xFFu));
    io.data_in.write((aecct::u8_t)((w >> 16) & 0xFFu));
    io.data_in.write((aecct::u8_t)((w >> 24) & 0xFFu));
}

static void send_cmd(Io8Top& io, uint8_t op) {
    io.ctrl_cmd.write(aecct::pack_ctrl_cmd(op));
    top_tick(io);
}

static bool nb_read_rsp(Io8Top& io, uint8_t& out_kind, uint8_t& out_payload) {
    aecct::u16_t rspw;
    if (!io.ctrl_rsp.nb_read(rspw)) {
        return false;
    }
    out_kind = aecct::unpack_ctrl_rsp_kind(rspw);
    out_payload = aecct::unpack_ctrl_rsp_payload(rspw);
    return true;
}

static void expect_no_rsp(Io8Top& io, const char* tag) {
    uint8_t kind = 0u;
    uint8_t payload = 0u;
    if (nb_read_rsp(io, kind, payload)) {
        std::printf(
            "[backup_io8][FAIL] %s unexpected rsp kind=%u payload=%u\n",
            tag,
            (unsigned)kind,
            (unsigned)payload);
        std::exit(1);
    }
}

static void expect_rsp(Io8Top& io, uint8_t kind_exp, uint8_t payload_exp, const char* tag) {
    uint8_t kind = 0u;
    uint8_t payload = 0u;
    if (!nb_read_rsp(io, kind, payload)) {
        std::printf("[backup_io8][FAIL] %s missing ctrl_rsp\n", tag);
        std::exit(1);
    }
    if (kind != kind_exp || payload != payload_exp) {
        std::printf(
            "[backup_io8][FAIL] %s rsp mismatch kind=%u payload=%u expect_kind=%u expect_payload=%u\n",
            tag,
            (unsigned)kind,
            (unsigned)payload,
            (unsigned)kind_exp,
            (unsigned)payload_exp);
        std::exit(1);
    }
}

static void expect_rsp_kind_either(
    Io8Top& io,
    uint8_t kind_exp0,
    uint8_t kind_exp1,
    uint8_t payload_exp,
    const char* tag
) {
    uint8_t kind = 0u;
    uint8_t payload = 0u;
    if (!nb_read_rsp(io, kind, payload)) {
        std::printf("[backup_io8][FAIL] %s missing ctrl_rsp\n", tag);
        std::exit(1);
    }
    if ((kind != kind_exp0 && kind != kind_exp1) || payload != payload_exp) {
        std::printf(
            "[backup_io8][FAIL] %s rsp mismatch kind=%u payload=%u expect_kind=%u|%u expect_payload=%u\n",
            tag,
            (unsigned)kind,
            (unsigned)payload,
            (unsigned)kind_exp0,
            (unsigned)kind_exp1,
            (unsigned)payload_exp);
        std::exit(1);
    }
}

static uint32_t f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = f;
    return cvt.u;
}

static void build_cfg_words(uint32_t cfg_words[EXP_LEN_CFG_WORDS]) {
    CFG_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_CFG_WORDS; ++i) {
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
    cfg_words[CFG_OUT_MODE] = 0u; // stream XPRED words
    cfg_words[CFG_RESERVED0] = 0u;
}

static void build_min_param_words(std::vector<uint32_t>& param_words) {
    param_words.assign((uint32_t)EXP_LEN_PARAM_WORDS, 0u);

    // Keep quant inv_s_w slots non-zero so backup INT8xternary path avoids divide-by-zero style guards.
    PARAM_INVSW_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)QUANT_LINEAR_MATRIX_COUNT; ++i) {
        const QuantLinearMeta meta = kQuantLinearMeta[i];
        const ParamMeta inv_meta = kParamMeta[meta.inv_sw_param_id];
        if (inv_meta.offset_w < (uint32_t)param_words.size()) {
            param_words[inv_meta.offset_w] = (uint32_t)aecct::fp32_bits_from_double(1.0).to_uint();
        }
    }
}

static void build_infer_words(std::vector<uint32_t>& infer_words) {
    infer_words.assign((uint32_t)EXP_LEN_INFER_IN_WORDS, 0u);
    INFER_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_INFER_IN_WORDS; ++i) {
        const int32_t sv = (int32_t)(i & 31u) - 16;
        const float fv = ((float)sv) * 0.03125f;
        infer_words[i] = f32_to_bits(fv);
    }
}

static void collect_out_bytes(
    Io8Top& io,
    uint32_t expected_bytes,
    std::vector<uint8_t>& out_bytes
) {
    out_bytes.clear();
    uint32_t guard = 0u;
    while ((uint32_t)out_bytes.size() < expected_bytes) {
        aecct::u8_t b;
        if (io.data_out.nb_read(b)) {
            out_bytes.push_back((uint8_t)b.to_uint());
            continue;
        }
        top_tick(io);
        ++guard;
        if (guard > (expected_bytes * 8u + 1024u)) {
            fail("output byte collection timeout");
        }
    }
}

static void run_single_session(std::vector<uint8_t>& out_bytes) {
    Io8Top io;

    std::vector<uint32_t> param_words;
    std::vector<uint32_t> infer_words;
    build_min_param_words(param_words);
    build_infer_words(infer_words);
    uint32_t cfg_words[EXP_LEN_CFG_WORDS];
    build_cfg_words(cfg_words);

    send_cmd(io, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET, "soft_reset");

    send_cmd(io, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN, "cfg_begin");
    CFG_INGEST_LOOP: for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_CFG_WORDS; ++i) {
        push_u32_le(io, cfg_words[i]);
        top_tick(io);
        expect_no_rsp(io, "cfg_ingest");
    }
    send_cmd(io, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp_kind_either(
        io,
        (uint8_t)aecct::RSP_OK,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_CFG_COMMIT,
        "cfg_commit");

    push_u32_le(io, (uint32_t)sram_map::PARAM_BASE_DEFAULT);
    io.ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_SET_W_BASE));
    top_tick(io);
    expect_rsp_kind_either(
        io,
        (uint8_t)aecct::RSP_OK,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_W_BASE,
        "set_w_base");

    send_cmd(io, (uint8_t)aecct::OP_LOAD_W);
    expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W, "load_w_begin");
    LOADW_INGEST_LOOP: for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_PARAM_WORDS; ++i) {
        push_u32_le(io, param_words[i]);
        top_tick(io);
        if (i + 1u < (uint32_t)EXP_LEN_PARAM_WORDS) {
            expect_no_rsp(io, "load_w_ingest");
        } else {
            expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W, "load_w_done");
        }
    }

    push_u32_le(io, 0u);
    io.ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_SET_OUTMODE));
    top_tick(io);
    expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE, "set_outmode");

    send_cmd(io, (uint8_t)aecct::OP_INFER);
    expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_INFER, "infer_begin");
    INFER_INGEST_LOOP: for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_INFER_IN_WORDS; ++i) {
        push_u32_le(io, infer_words[i]);
        top_tick(io);
        if (i + 1u < (uint32_t)EXP_LEN_INFER_IN_WORDS) {
            expect_no_rsp(io, "infer_ingest");
        } else {
            expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_INFER, "infer_done");
        }
    }

    const uint32_t expected_out_bytes = (uint32_t)EXP_LEN_OUT_XPRED_WORDS * 4u;
    collect_out_bytes(io, expected_out_bytes, out_bytes);
    if (out_bytes.empty()) {
        fail("output bytes unexpectedly empty");
    }
    if ((uint32_t)out_bytes.size() != expected_out_bytes) {
        fail("output byte count mismatch");
    }
}

} // namespace

int main() {
    std::vector<uint8_t> out_a;
    std::vector<uint8_t> out_b;

    run_single_session(out_a);
    run_single_session(out_b);

    if (out_a != out_b) {
        std::printf(
            "[backup_io8][FAIL] deterministic rerun mismatch: size_a=%u size_b=%u\n",
            (unsigned)out_a.size(),
            (unsigned)out_b.size());
        return 1;
    }

    std::printf(
        "[backup_io8] output_bytes=%u first4=%02X %02X %02X %02X\n",
        (unsigned)out_a.size(),
        (unsigned)out_a[0],
        (unsigned)out_a[1],
        (unsigned)out_a[2],
        (unsigned)out_a[3]);
    std::printf("PASS: tb_backup_io8_loadw_infer_smoke\n");
    return 0;
}
