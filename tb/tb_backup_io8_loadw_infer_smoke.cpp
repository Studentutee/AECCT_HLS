// tb_backup_io8_loadw_infer_smoke.cpp
// Backup profile next-step smoke:
// - Top external io8 boundary
// - minimal CFG + SET_W_BASE + LOAD_W + INFER narrow path
// - output non-empty / byte-count check / deterministic rerun

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
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

static uint32_t fnv1a_u32(const std::vector<uint8_t>& bytes) {
    uint32_t h = 2166136261u;
    HASH_LOOP: for (uint32_t i = 0u; i < (uint32_t)bytes.size(); ++i) {
        h ^= (uint32_t)bytes[i];
        h *= 16777619u;
    }
    return h;
}

static bool parse_hex_byte_token(const std::string& tok, uint8_t& out_byte) {
    std::string t = tok;
    if (t.size() >= 2u && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
        t = t.substr(2u);
    }
    if (t.empty() || t.size() > 2u) {
        return false;
    }
    uint32_t v = 0u;
    HEX_PARSE_LOOP: for (uint32_t i = 0u; i < (uint32_t)t.size(); ++i) {
        const char c = t[(size_t)i];
        uint32_t d = 0u;
        if (c >= '0' && c <= '9') {
            d = (uint32_t)(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            d = (uint32_t)(10 + (c - 'a'));
        } else if (c >= 'A' && c <= 'F') {
            d = (uint32_t)(10 + (c - 'A'));
        } else {
            return false;
        }
        v = (v << 4) | d;
    }
    out_byte = (uint8_t)v;
    return true;
}

static bool load_external_golden_bytes(
    const char* path,
    std::vector<uint8_t>& out_expected_bytes
) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::printf("[backup_io8][FAIL] cannot open external golden: %s\n", path);
        return false;
    }

    out_expected_bytes.clear();
    std::string line;
    uint32_t line_no = 0u;
    while (std::getline(ifs, line)) {
        ++line_no;
        const std::size_t cpos = line.find('#');
        if (cpos != std::string::npos) {
            line.erase(cpos);
        }
        std::stringstream ss(line);
        std::string tok0;
        if (!(ss >> tok0)) {
            continue;
        }

        if (tok0 == "repeat") {
            std::string count_tok;
            std::string byte_tok;
            if (!(ss >> count_tok >> byte_tok)) {
                std::printf(
                    "[backup_io8][FAIL] external golden parse error at line %u: malformed repeat\n",
                    (unsigned)line_no);
                return false;
            }

            char* endp = nullptr;
            const unsigned long count_ul = std::strtoul(count_tok.c_str(), &endp, 10);
            if (endp == nullptr || *endp != '\0') {
                std::printf(
                    "[backup_io8][FAIL] external golden parse error at line %u: bad repeat count\n",
                    (unsigned)line_no);
                return false;
            }
            uint8_t b = 0u;
            if (!parse_hex_byte_token(byte_tok, b)) {
                std::printf(
                    "[backup_io8][FAIL] external golden parse error at line %u: bad repeat byte token '%s'\n",
                    (unsigned)line_no,
                    byte_tok.c_str());
                return false;
            }

            REPEAT_APPEND_LOOP: for (unsigned long i = 0ul; i < count_ul; ++i) {
                out_expected_bytes.push_back(b);
            }
            continue;
        }

        uint8_t b0 = 0u;
        if (!parse_hex_byte_token(tok0, b0)) {
            std::printf(
                "[backup_io8][FAIL] external golden parse error at line %u: bad byte token '%s'\n",
                (unsigned)line_no,
                tok0.c_str());
            return false;
        }
        out_expected_bytes.push_back(b0);

        std::string tok;
        while (ss >> tok) {
            uint8_t b = 0u;
            if (!parse_hex_byte_token(tok, b)) {
                std::printf(
                    "[backup_io8][FAIL] external golden parse error at line %u: bad byte token '%s'\n",
                    (unsigned)line_no,
                    tok.c_str());
                return false;
            }
            out_expected_bytes.push_back(b);
        }
    }

    if (out_expected_bytes.empty()) {
        std::printf("[backup_io8][FAIL] external golden file has no bytes: %s\n", path);
        return false;
    }
    return true;
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

    const uint32_t out_hash = fnv1a_u32(out_a);
    static const char* kExternalGoldenPath =
        "tb/golden/backup_io8_loadw_infer_fixed_case_xpred.hex";
    std::vector<uint8_t> expected_bytes;
    if (!load_external_golden_bytes(kExternalGoldenPath, expected_bytes)) {
        return 1;
    }

    if (out_a.size() != expected_bytes.size()) {
        std::printf(
            "[backup_io8][FAIL] external golden byte-count mismatch got=%u expect=%u\n",
            (unsigned)out_a.size(),
            (unsigned)expected_bytes.size());
        return 1;
    }

    uint32_t first_mismatch_idx = 0u;
    bool has_mismatch = false;
    COMPARE_LOOP: for (uint32_t i = 0u; i < (uint32_t)out_a.size(); ++i) {
        if (out_a[i] != expected_bytes[i]) {
            first_mismatch_idx = i;
            has_mismatch = true;
            break;
        }
    }

    if (has_mismatch) {
        const uint32_t i = first_mismatch_idx;
        std::printf(
            "[backup_io8][FAIL] external golden byte mismatch idx=%u got=0x%02X expect=0x%02X\n",
            (unsigned)i,
            (unsigned)out_a[i],
            (unsigned)expected_bytes[i]);
        return 1;
    }

    const uint32_t exp_hash = fnv1a_u32(expected_bytes);
    if (out_hash != exp_hash) {
            std::printf(
            "[backup_io8][FAIL] external golden hash mismatch got=0x%08X expect=0x%08X\n",
            (unsigned)out_hash,
            (unsigned)exp_hash);
        return 1;
    }
    std::printf(
        "PASS: tb_backup_io8_loadw_infer_external_golden_compare path=%s bytes=%u hash=0x%08X\n",
        kExternalGoldenPath,
        (unsigned)out_a.size(),
        (unsigned)out_hash);

    std::printf(
        "[backup_io8] output_bytes=%u first4=%02X %02X %02X %02X hash=0x%08X\n",
        (unsigned)out_a.size(),
        (unsigned)out_a[0],
        (unsigned)out_a[1],
        (unsigned)out_a[2],
        (unsigned)out_a[3],
        (unsigned)out_hash);
    std::printf("PASS: tb_backup_io8_loadw_infer_smoke\n");
    return 0;
}
