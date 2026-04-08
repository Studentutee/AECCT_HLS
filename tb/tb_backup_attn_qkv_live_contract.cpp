// tb_backup_attn_qkv_live_contract.cpp
// Isolated layer0 attention contract smoke:
// - LOAD_W through existing Top io8 path
// - RefModel sample=5 reference for layer0 attention checkpoints
// - Q/K/V consume-path exactness and anti-bypass checks

#ifndef __SYNTHESIS__

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "AecctProtocol.h"
#include "AecctUtil.h"
#include "Top.h"
#include "weights_streamer.h"
#include "input_y_step0.h"
#include "RefModel.h"
#include "blocks/AttnLayer0.h"

namespace {

struct Io8Top {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data8_ch_t data_in;
    aecct::data8_ch_t data_out;
};

struct CompareOutcome {
    bool exact;
    bool has_mismatch;
    uint32_t token;
    uint32_t dim;
    uint32_t dut_bits;
    uint32_t ref_bits;
};

struct MaskedSemanticsOutcome {
    uint32_t masked_count;
    bool has_first_masked;
    uint32_t first_masked_score_idx;
    uint32_t first_dut_score_bits;
    uint32_t first_ref_score_bits;
    uint32_t first_dut_prob_bits;
    uint32_t first_ref_prob_bits;
    uint32_t masked_prob_nonzero;
    bool masked_score_semantics_ok;
};

static void fail(const char* msg) {
    std::printf("[backup_attn_qkv][FAIL] %s\n", msg);
    std::exit(1);
}

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
        float f;
        uint32_t u;
    } cvt;
    cvt.u = u;
    return cvt.f;
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
            "[backup_attn_qkv][FAIL] %s unexpected rsp kind=%u payload=%u\n",
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
        std::printf("[backup_attn_qkv][FAIL] %s missing ctrl_rsp\n", tag);
        std::exit(1);
    }
    if (kind != kind_exp || payload != payload_exp) {
        std::printf(
            "[backup_attn_qkv][FAIL] %s rsp mismatch kind=%u payload=%u expect_kind=%u expect_payload=%u\n",
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
        std::printf("[backup_attn_qkv][FAIL] %s missing ctrl_rsp\n", tag);
        std::exit(1);
    }
    if ((kind != kind_exp0 && kind != kind_exp1) || payload != payload_exp) {
        std::printf(
            "[backup_attn_qkv][FAIL] %s rsp mismatch kind=%u payload=%u expect_kind=%u|%u expect_payload=%u\n",
            tag,
            (unsigned)kind,
            (unsigned)payload,
            (unsigned)kind_exp0,
            (unsigned)kind_exp1,
            (unsigned)payload_exp);
        std::exit(1);
    }
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
    cfg_words[CFG_OUT_MODE] = 0u;
    cfg_words[CFG_RESERVED0] = 0u;
}

static void build_trace_infer_words(uint32_t sample_idx, std::vector<uint32_t>& infer_words) {
    infer_words.assign((uint32_t)EXP_LEN_INFER_IN_WORDS, 0u);
    const uint32_t stride = (uint32_t)EXP_LEN_INFER_IN_WORDS;
    const uint32_t base = sample_idx * stride;
    TRACE_INFER_WORD_LOOP: for (uint32_t i = 0u; i < stride; ++i) {
        infer_words[i] = f32_to_bits((float)trace_input_y_step0_tensor[base + i]);
    }
}

static bool build_param_words_from_repo_reference(
    std::vector<uint32_t>& param_words,
    std::string& error
) {
    error.clear();
    aecct::data_ch_t stream_words;

    PARAM_BIAS_STREAM_LOOP: for (uint32_t i = 0u; i < (uint32_t)BIAS_COUNT; ++i) {
        const BiasId bid = (BiasId)i;
        const TensorMeta meta = kBiasMeta[i];
        uint32_t numel = 0u;
        const double* ptr = tb_lookup_bias_fp64(bid, numel);
        if (ptr == 0 || numel == 0u) {
            tb_emit_padding_zeros(stream_words, meta.len_w);
        } else {
            tb_emit_fp32_words_from_fp64(stream_words, ptr, numel, meta.len_w);
        }
    }

    PARAM_WEIGHT_STREAM_LOOP: for (uint32_t i = 0u; i < (uint32_t)WEIGHT_COUNT; ++i) {
        const WeightId wid = (WeightId)i;
        const TensorMeta meta = kWeightMeta[i];
        if (meta.dtype == 0u) {
            uint32_t numel = 0u;
            const double* ptr = tb_lookup_weight_fp64(wid, numel);
            if (ptr == 0 || numel == 0u) {
                tb_emit_padding_zeros(stream_words, meta.len_w);
            } else if (is_quant_linear_inv_sw_weight_slot(wid)) {
                if (!tb_emit_inv_sw_words_from_fp64(stream_words, ptr, numel, meta.len_w, wid)) {
                    error = "inv_s_w conversion failed";
                    return false;
                }
            } else {
                tb_emit_fp32_words_from_fp64(stream_words, ptr, numel, meta.len_w);
            }
        } else {
            uint32_t num_bits = 0u;
            const ac_int<1, false>* bits = tb_lookup_weight_bits(wid, num_bits);
            if (bits == 0 || num_bits == 0u) {
                tb_emit_padding_zeros(stream_words, meta.len_w);
            } else {
                tb_emit_bitpack_words(stream_words, bits, num_bits, meta.len_w);
            }
        }
    }

    param_words.clear();
    param_words.reserve((uint32_t)EXP_LEN_PARAM_WORDS);
    aecct::u32_t w;
    PARAM_WORD_COLLECT_LOOP: while (stream_words.nb_read(w)) {
        param_words.push_back((uint32_t)w.to_uint());
    }

    if ((uint32_t)param_words.size() != (uint32_t)EXP_LEN_PARAM_WORDS) {
        error = "PARAM word count mismatch";
        return false;
    }
    return true;
}

static void run_setup_cfg_loadw(
    Io8Top& io,
    const std::vector<uint32_t>& param_words
) {
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
    LOADW_INGEST_LOOP: for (uint32_t i = 0u; i < (uint32_t)param_words.size(); ++i) {
        push_u32_le(io, param_words[i]);
        top_tick(io);
        if (i + 1u < (uint32_t)param_words.size()) {
            expect_no_rsp(io, "load_w_ingest");
        } else {
            expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W, "load_w_done");
        }
    }
}

static std::string trim_copy(const std::string& s) {
    std::size_t b = 0u;
    while (b < s.size() && std::isspace((unsigned char)s[b])) { ++b; }
    std::size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1u])) { --e; }
    return s.substr(b, e - b);
}

static bool parse_shape_2d(const std::string& header, uint32_t& rows, uint32_t& cols) {
    rows = 0u;
    cols = 0u;
    const std::size_t shape_pos = header.find("shape");
    if (shape_pos == std::string::npos) {
        return false;
    }
    const std::size_t lp = header.find('(', shape_pos);
    const std::size_t rp = header.find(')', lp);
    if (lp == std::string::npos || rp == std::string::npos || rp <= lp) {
        return false;
    }
    const std::string shape_csv = header.substr(lp + 1u, rp - lp - 1u);
    std::stringstream ss(shape_csv);
    std::string tok;
    std::vector<uint32_t> dims;
    SHAPE_TOKEN_LOOP: while (std::getline(ss, tok, ',')) {
        const std::string t = trim_copy(tok);
        if (t.empty()) {
            continue;
        }
        dims.push_back((uint32_t)std::strtoul(t.c_str(), 0, 10));
    }
    if (dims.size() != 2u) {
        return false;
    }
    rows = dims[0];
    cols = dims[1];
    return true;
}

static bool parse_shape_3d(const std::string& header, uint32_t& d0, uint32_t& d1, uint32_t& d2) {
    d0 = 0u;
    d1 = 0u;
    d2 = 0u;
    const std::size_t shape_pos = header.find("shape");
    if (shape_pos == std::string::npos) {
        return false;
    }
    const std::size_t lp = header.find('(', shape_pos);
    const std::size_t rp = header.find(')', lp);
    if (lp == std::string::npos || rp == std::string::npos || rp <= lp) {
        return false;
    }
    const std::string shape_csv = header.substr(lp + 1u, rp - lp - 1u);
    std::stringstream ss(shape_csv);
    std::string tok;
    std::vector<uint32_t> dims;
    SHAPE3_TOKEN_LOOP: while (std::getline(ss, tok, ',')) {
        const std::string t = trim_copy(tok);
        if (t.empty()) {
            continue;
        }
        dims.push_back((uint32_t)std::strtoul(t.c_str(), 0, 10));
    }
    if (dims.size() != 3u) {
        return false;
    }
    d0 = dims[0];
    d1 = dims[1];
    d2 = dims[2];
    return true;
}

static bool load_npy_f32_2d(
    const std::string& path,
    uint32_t expect_rows,
    uint32_t expect_cols,
    std::vector<float>& out,
    std::string& err
) {
    err.clear();
    out.clear();

    std::ifstream ifs(path.c_str(), std::ios::binary);
    if (!ifs.good()) {
        err = "cannot open npy file: " + path;
        return false;
    }

    char magic[6];
    ifs.read(magic, 6);
    if (ifs.gcount() != 6 || !(magic[0] == char(0x93) && magic[1] == 'N' && magic[2] == 'U' &&
                               magic[3] == 'M' && magic[4] == 'P' && magic[5] == 'Y')) {
        err = "bad npy magic: " + path;
        return false;
    }

    unsigned char ver_major = 0u;
    unsigned char ver_minor = 0u;
    ifs.read((char*)&ver_major, 1);
    ifs.read((char*)&ver_minor, 1);
    (void)ver_minor;
    if (!ifs.good()) {
        err = "bad npy version bytes: " + path;
        return false;
    }

    uint32_t header_len = 0u;
    if (ver_major == 1u) {
        unsigned char h0 = 0u;
        unsigned char h1 = 0u;
        ifs.read((char*)&h0, 1);
        ifs.read((char*)&h1, 1);
        if (!ifs.good()) {
            err = "bad npy header length (v1): " + path;
            return false;
        }
        header_len = (uint32_t)h0 | ((uint32_t)h1 << 8);
    } else {
        unsigned char h[4] = {0u, 0u, 0u, 0u};
        ifs.read((char*)h, 4);
        if (!ifs.good()) {
            err = "bad npy header length (v2+): " + path;
            return false;
        }
        header_len = ((uint32_t)h[0]) | ((uint32_t)h[1] << 8) | ((uint32_t)h[2] << 16) | ((uint32_t)h[3] << 24);
    }

    std::string header;
    header.resize(header_len);
    ifs.read(&header[0], (std::streamsize)header_len);
    if (!ifs.good()) {
        err = "bad npy header read: " + path;
        return false;
    }
    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos) {
        err = "npy descr is not <f4: " + path;
        return false;
    }
    if (header.find("False") == std::string::npos) {
        err = "npy fortran_order is not False: " + path;
        return false;
    }

    uint32_t rows = 0u;
    uint32_t cols = 0u;
    if (!parse_shape_2d(header, rows, cols)) {
        err = "npy shape parse failed: " + path;
        return false;
    }
    if (rows != expect_rows || cols != expect_cols) {
        std::ostringstream oss;
        oss << "npy shape mismatch: got=(" << rows << "," << cols << "), expect=("
            << expect_rows << "," << expect_cols << ")";
        err = oss.str();
        return false;
    }

    const uint32_t count = rows * cols;
    out.resize(count);
    ifs.read((char*)out.data(), (std::streamsize)(count * sizeof(float)));
    if (!ifs.good()) {
        err = "npy payload read failed: " + path;
        return false;
    }
    return true;
}

static bool load_npy_f32_3d(
    const std::string& path,
    uint32_t expect_d0,
    uint32_t expect_d1,
    uint32_t expect_d2,
    std::vector<float>& out,
    std::string& err
) {
    err.clear();
    out.clear();

    std::ifstream ifs(path.c_str(), std::ios::binary);
    if (!ifs.good()) {
        err = "cannot open npy file: " + path;
        return false;
    }

    char magic[6];
    ifs.read(magic, 6);
    if (ifs.gcount() != 6 || !(magic[0] == char(0x93) && magic[1] == 'N' && magic[2] == 'U' &&
                               magic[3] == 'M' && magic[4] == 'P' && magic[5] == 'Y')) {
        err = "bad npy magic: " + path;
        return false;
    }

    unsigned char ver_major = 0u;
    unsigned char ver_minor = 0u;
    ifs.read((char*)&ver_major, 1);
    ifs.read((char*)&ver_minor, 1);
    (void)ver_minor;
    if (!ifs.good()) {
        err = "bad npy version bytes: " + path;
        return false;
    }

    uint32_t header_len = 0u;
    if (ver_major == 1u) {
        unsigned char h0 = 0u;
        unsigned char h1 = 0u;
        ifs.read((char*)&h0, 1);
        ifs.read((char*)&h1, 1);
        if (!ifs.good()) {
            err = "bad npy header length (v1): " + path;
            return false;
        }
        header_len = (uint32_t)h0 | ((uint32_t)h1 << 8);
    } else {
        unsigned char h[4] = {0u, 0u, 0u, 0u};
        ifs.read((char*)h, 4);
        if (!ifs.good()) {
            err = "bad npy header length (v2+): " + path;
            return false;
        }
        header_len = ((uint32_t)h[0]) | ((uint32_t)h[1] << 8) | ((uint32_t)h[2] << 16) | ((uint32_t)h[3] << 24);
    }

    std::string header;
    header.resize(header_len);
    ifs.read(&header[0], (std::streamsize)header_len);
    if (!ifs.good()) {
        err = "bad npy header read: " + path;
        return false;
    }
    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos) {
        err = "npy descr is not <f4: " + path;
        return false;
    }
    if (header.find("False") == std::string::npos) {
        err = "npy fortran_order is not False: " + path;
        return false;
    }

    uint32_t d0 = 0u;
    uint32_t d1 = 0u;
    uint32_t d2 = 0u;
    if (!parse_shape_3d(header, d0, d1, d2)) {
        err = "npy shape parse failed: " + path;
        return false;
    }
    if (d0 != expect_d0 || d1 != expect_d1 || d2 != expect_d2) {
        std::ostringstream oss;
        oss << "npy shape mismatch: got=(" << d0 << "," << d1 << "," << d2 << "), expect=("
            << expect_d0 << "," << expect_d1 << "," << expect_d2 << ")";
        err = oss.str();
        return false;
    }

    const uint32_t count = d0 * d1 * d2;
    out.resize(count);
    ifs.read((char*)out.data(), (std::streamsize)(count * sizeof(float)));
    if (!ifs.good()) {
        err = "npy payload read failed: " + path;
        return false;
    }
    return true;
}

static CompareOutcome compare_sram_with_ref_fp32(
    const aecct::u32_t* sram,
    uint32_t base_word,
    const std::vector<float>& ref_tensor,
    uint32_t tokens,
    uint32_t d_model
) {
    CompareOutcome out = {true, false, 0u, 0u, 0u, 0u};
    COMPARE_REF_TOKEN_LOOP: for (uint32_t t = 0u; t < tokens; ++t) {
        COMPARE_REF_DIM_LOOP: for (uint32_t d = 0u; d < d_model; ++d) {
            const uint32_t flat = t * d_model + d;
            const uint32_t dut_bits = (uint32_t)sram[base_word + flat].to_uint();
            const uint32_t ref_bits = f32_to_bits(ref_tensor[flat]);
            if (dut_bits != ref_bits) {
                out.exact = false;
                out.has_mismatch = true;
                out.token = t;
                out.dim = d;
                out.dut_bits = dut_bits;
                out.ref_bits = ref_bits;
                return out;
            }
        }
    }
    return out;
}

static CompareOutcome compare_sram_with_ref_double(
    const aecct::u32_t* sram,
    uint32_t base_word,
    const std::vector<double>& ref_tensor,
    uint32_t tokens,
    uint32_t d_model
) {
    CompareOutcome out = {true, false, 0u, 0u, 0u, 0u};
    COMPARE_REF_DBL_TOKEN_LOOP: for (uint32_t t = 0u; t < tokens; ++t) {
        COMPARE_REF_DBL_DIM_LOOP: for (uint32_t d = 0u; d < d_model; ++d) {
            const uint32_t flat = t * d_model + d;
            const uint32_t dut_bits = (uint32_t)sram[base_word + flat].to_uint();
            const uint32_t ref_bits = f32_to_bits((float)ref_tensor[flat]);
            if (dut_bits != ref_bits) {
                out.exact = false;
                out.has_mismatch = true;
                out.token = t;
                out.dim = d;
                out.dut_bits = dut_bits;
                out.ref_bits = ref_bits;
                return out;
            }
        }
    }
    return out;
}

static CompareOutcome compare_sram_with_input_bits(
    const aecct::u32_t* sram,
    uint32_t base_word,
    const std::vector<uint32_t>& input_bits,
    uint32_t tokens,
    uint32_t d_model
) {
    CompareOutcome out = {true, false, 0u, 0u, 0u, 0u};
    COMPARE_IN_TOKEN_LOOP: for (uint32_t t = 0u; t < tokens; ++t) {
        COMPARE_IN_DIM_LOOP: for (uint32_t d = 0u; d < d_model; ++d) {
            const uint32_t flat = t * d_model + d;
            const uint32_t dut_bits = (uint32_t)sram[base_word + flat].to_uint();
            const uint32_t ref_bits = input_bits[flat];
            if (dut_bits != ref_bits) {
                out.exact = false;
                out.has_mismatch = true;
                out.token = t;
                out.dim = d;
                out.dut_bits = dut_bits;
                out.ref_bits = ref_bits;
                return out;
            }
        }
    }
    return out;
}

static void print_first_mismatch(const char* name, const CompareOutcome& r) {
    if (r.has_mismatch) {
        std::printf(
            "%s_first_mismatch(token,dim,dut,ref)=(%u,%u,0x%08X,0x%08X)\n",
            name,
            (unsigned)r.token,
            (unsigned)r.dim,
            (unsigned)r.dut_bits,
            (unsigned)r.ref_bits);
    } else {
        std::printf("%s_first_mismatch(token,dim,dut,ref)=(none)\n", name);
    }
}

static inline uint32_t idx_3d(
    uint32_t i0,
    uint32_t i1,
    uint32_t i2,
    uint32_t dim1,
    uint32_t dim2
) {
    return (i0 * dim1 + i1) * dim2 + i2;
}

static MaskedSemanticsOutcome evaluate_masked_semantics(
    const aecct::u32_t* sram,
    uint32_t score_base_word,
    uint32_t prob_base_word,
    const std::vector<float>& ref_scores,
    const std::vector<float>& ref_probs,
    uint32_t n_heads,
    uint32_t token_count,
    uint32_t head_idx,
    uint32_t token_idx
) {
    (void)n_heads;
    const uint32_t neg_inf_bits = f32_to_bits(-std::numeric_limits<float>::infinity());
    MaskedSemanticsOutcome out = {0u, false, 0u, 0u, 0u, 0u, 0u, 0u, true};
    MASKED_SEMANTICS_SCAN_LOOP: for (uint32_t j = 0u; j < token_count; ++j) {
        const uint32_t ref_flat = idx_3d(head_idx, token_idx, j, token_count, token_count);
        const float ref_score_fp = ref_scores[ref_flat];
        const bool ref_masked = std::isinf(ref_score_fp) && (ref_score_fp < 0.0f);
        if (!ref_masked) {
            continue;
        }

        const uint32_t dut_score_bits = (uint32_t)sram[score_base_word + j].to_uint();
        const uint32_t dut_prob_bits = (uint32_t)sram[prob_base_word + j].to_uint();
        const uint32_t ref_score_bits = f32_to_bits(ref_score_fp);
        const uint32_t ref_prob_bits = f32_to_bits(ref_probs[ref_flat]);

        out.masked_count += 1u;
        if (!out.has_first_masked) {
            out.has_first_masked = true;
            out.first_masked_score_idx = j;
            out.first_dut_score_bits = dut_score_bits;
            out.first_ref_score_bits = ref_score_bits;
            out.first_dut_prob_bits = dut_prob_bits;
            out.first_ref_prob_bits = ref_prob_bits;
        }
        if (dut_prob_bits != 0u) {
            out.masked_prob_nonzero += 1u;
        }
        if (dut_score_bits != neg_inf_bits) {
            out.masked_score_semantics_ok = false;
        }
    }
    return out;
}

} // namespace

int main() {
    const uint32_t sample_idx = 5u;
    const uint32_t token_count = (uint32_t)N_NODES;
    const uint32_t d_model = (uint32_t)D_MODEL;
    const uint32_t tensor_words = token_count * d_model;

    std::vector<uint32_t> param_words;
    std::string param_error;
    if (!build_param_words_from_repo_reference(param_words, param_error)) {
        fail(param_error.c_str());
    }

    Io8Top io;
    run_setup_cfg_loadw(io, param_words);

    std::vector<uint32_t> infer_words;
    build_trace_infer_words(sample_idx, infer_words);
    std::vector<double> ref_input_fp32((uint32_t)EXP_LEN_INFER_IN_WORDS, 0.0);
    REF_INPUT_CONVERT_LOOP: for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_INFER_IN_WORDS; ++i) {
        ref_input_fp32[i] = (double)bits_to_f32(infer_words[i]);
    }

    std::vector<double> ref_logits((uint32_t)EXP_LEN_OUT_LOGITS_WORDS, 0.0);
    std::vector<aecct_ref::bit1_t> ref_xpred((uint32_t)EXP_LEN_OUT_XPRED_WORDS);
    std::vector<double> ref_layer0_attn_input(tensor_words, 0.0);
    std::vector<double> ref_layer0_post_concat(tensor_words, 0.0);

    const std::string dump_dir = "build/backup_attn_qkv_live_contract/refdump_sample5";
    std::filesystem::create_directories(dump_dir);

    aecct_ref::RefModel ref_model;
    aecct_ref::RefDumpConfig dump_cfg;
    dump_cfg.enabled = true;
    dump_cfg.dump_dir = dump_dir.c_str();
    dump_cfg.pattern_index = (int)sample_idx;
    ref_model.set_dump_config(dump_cfg);

    aecct_ref::RefModelIO ref_io;
    ref_io.input_y = 0;
    ref_io.input_y_fp32 = ref_input_fp32.data();
    ref_io.out_logits = ref_logits.data();
    ref_io.out_x_pred = ref_xpred.data();
    ref_io.out_layer0_attn_input = ref_layer0_attn_input.data();
    ref_io.out_layer0_post_concat = ref_layer0_post_concat.data();
    ref_io.B = 1;
    ref_io.N = (int)EXP_LEN_OUT_XPRED_WORDS;
    ref_model.infer_step0(ref_io);

    std::vector<float> ref_q;
    std::vector<float> ref_k;
    std::vector<float> ref_v;
    std::vector<float> ref_attn_scores;
    std::vector<float> ref_attn_probs;
    std::string npy_error;
    if (!load_npy_f32_2d(dump_dir + "/layer0_q.npy", token_count, d_model, ref_q, npy_error)) {
        fail(npy_error.c_str());
    }
    if (!load_npy_f32_2d(dump_dir + "/layer0_k.npy", token_count, d_model, ref_k, npy_error)) {
        fail(npy_error.c_str());
    }
    if (!load_npy_f32_2d(dump_dir + "/layer0_v.npy", token_count, d_model, ref_v, npy_error)) {
        fail(npy_error.c_str());
    }
    if (!load_npy_f32_3d(
            dump_dir + "/layer0_attn_scores.npy",
            (uint32_t)N_HEAD,
            token_count,
            token_count,
            ref_attn_scores,
            npy_error)) {
        fail(npy_error.c_str());
    }
    if (!load_npy_f32_3d(
            dump_dir + "/layer0_attn_probs.npy",
            (uint32_t)N_HEAD,
            token_count,
            token_count,
            ref_attn_probs,
            npy_error)) {
        fail(npy_error.c_str());
    }

    std::vector<uint32_t> attn_input_bits(tensor_words, 0u);
    ATTN_INPUT_PACK_LOOP: for (uint32_t i = 0u; i < tensor_words; ++i) {
        attn_input_bits[i] = f32_to_bits((float)ref_layer0_attn_input[i]);
    }

    aecct::u32_t* sram = aecct::top_sram();
    const uint32_t x_base = (uint32_t)aecct::ATTN_X_IN_BASE_WORD_DEFAULT;
    ATTN_INPUT_WRITE_LOOP: for (uint32_t i = 0u; i < tensor_words; ++i) {
        sram[x_base + i] = (aecct::u32_t)attn_input_bits[i];
    }

    aecct::AttnCfg cfg;
    cfg.token_count = (aecct::u32_t)token_count;
    cfg.d_model = (aecct::u32_t)d_model;
    cfg.n_heads = (aecct::u32_t)N_HEAD;
    cfg.d_head = (aecct::u32_t)(d_model / (uint32_t)N_HEAD);
    const aecct::AttnScratch sc = aecct::default_attn_scratch();

    aecct::AttnLayer0<aecct::ATTN_STAGE_FULL>(
        sram,
        cfg,
        (aecct::u32_t)x_base,
        (aecct::u32_t)aecct::ATTN_OUT_BASE_WORD_DEFAULT,
        sc,
        (aecct::u32_t)sram_map::PARAM_BASE_DEFAULT,
        aecct::make_attn_layer0_prebuilt_handoff_desc(),
        (aecct::u32_t)0u);

    const CompareOutcome q_cmp = compare_sram_with_ref_fp32(
        sram,
        (uint32_t)sc.q_base_word.to_uint(),
        ref_q,
        token_count,
        d_model);
    const CompareOutcome k_cmp = compare_sram_with_ref_fp32(
        sram,
        (uint32_t)sc.k_base_word.to_uint(),
        ref_k,
        token_count,
        d_model);
    const CompareOutcome v_cmp = compare_sram_with_ref_fp32(
        sram,
        (uint32_t)sc.v_base_word.to_uint(),
        ref_v,
        token_count,
        d_model);

    const CompareOutcome q_eq_in = compare_sram_with_input_bits(
        sram,
        (uint32_t)sc.q_base_word.to_uint(),
        attn_input_bits,
        token_count,
        d_model);
    const CompareOutcome k_eq_in = compare_sram_with_input_bits(
        sram,
        (uint32_t)sc.k_base_word.to_uint(),
        attn_input_bits,
        token_count,
        d_model);
    const CompareOutcome v_eq_in = compare_sram_with_input_bits(
        sram,
        (uint32_t)sc.v_base_word.to_uint(),
        attn_input_bits,
        token_count,
        d_model);

    const CompareOutcome post_cmp = compare_sram_with_ref_double(
        sram,
        (uint32_t)sc.post_concat_base_word.to_uint(),
        ref_layer0_post_concat,
        token_count,
        d_model);
    std::vector<float> ref_score_row(token_count, 0.0f);
    std::vector<float> ref_prob_row(token_count, 0.0f);
    REF_SCOREPROB_ROW_EXTRACT_LOOP: for (uint32_t j = 0u; j < token_count; ++j) {
        const uint32_t flat = idx_3d(0u, 0u, j, token_count, token_count);
        ref_score_row[j] = ref_attn_scores[flat];
        ref_prob_row[j] = ref_attn_probs[flat];
    }
    const CompareOutcome score_cmp = compare_sram_with_ref_fp32(
        sram,
        (uint32_t)sc.score_base_word.to_uint(),
        ref_score_row,
        1u,
        token_count);
    const CompareOutcome prob_cmp = compare_sram_with_ref_fp32(
        sram,
        (uint32_t)sc.softmax_base_word.to_uint(),
        ref_prob_row,
        1u,
        token_count);
    const MaskedSemanticsOutcome masked_sem = evaluate_masked_semantics(
        sram,
        (uint32_t)sc.score_base_word.to_uint(),
        (uint32_t)sc.softmax_base_word.to_uint(),
        ref_attn_scores,
        ref_attn_probs,
        (uint32_t)N_HEAD,
        token_count,
        0u,
        0u);

    std::printf("Q_exact=%u\n", (unsigned)(q_cmp.exact ? 1u : 0u));
    std::printf("K_exact=%u\n", (unsigned)(k_cmp.exact ? 1u : 0u));
    std::printf("V_exact=%u\n", (unsigned)(v_cmp.exact ? 1u : 0u));
    std::printf("Q_equals_input_exact=%u\n", (unsigned)(q_eq_in.exact ? 1u : 0u));
    std::printf("K_equals_input_exact=%u\n", (unsigned)(k_eq_in.exact ? 1u : 0u));
    std::printf("V_equals_input_exact=%u\n", (unsigned)(v_eq_in.exact ? 1u : 0u));
    std::printf("SCORE_row_head0_token0_exact=%u\n", (unsigned)(score_cmp.exact ? 1u : 0u));
    std::printf("PROB_row_head0_token0_exact=%u\n", (unsigned)(prob_cmp.exact ? 1u : 0u));
    std::printf("masked_count=%u\n", (unsigned)masked_sem.masked_count);
    if (masked_sem.has_first_masked) {
        std::printf("first_masked_score_idx=%u\n", (unsigned)masked_sem.first_masked_score_idx);
        std::printf("first_masked_dut_score=0x%08X\n", (unsigned)masked_sem.first_dut_score_bits);
        std::printf("first_masked_ref_score=0x%08X\n", (unsigned)masked_sem.first_ref_score_bits);
        std::printf("first_masked_dut_prob=0x%08X\n", (unsigned)masked_sem.first_dut_prob_bits);
        std::printf("first_masked_ref_prob=0x%08X\n", (unsigned)masked_sem.first_ref_prob_bits);
    } else {
        std::printf("first_masked_score_idx=(none)\n");
        std::printf("first_masked_dut_score=(none)\n");
        std::printf("first_masked_ref_score=(none)\n");
        std::printf("first_masked_dut_prob=(none)\n");
        std::printf("first_masked_ref_prob=(none)\n");
    }
    std::printf("MASKED_SCORE_SEMANTICS_OK=%u\n", (unsigned)(masked_sem.masked_score_semantics_ok ? 1u : 0u));
    std::printf("MASKED_PROB_NONZERO=%u\n", (unsigned)masked_sem.masked_prob_nonzero);
    std::printf("POST_exact=%u\n", (unsigned)(post_cmp.exact ? 1u : 0u));

    print_first_mismatch("Q", q_cmp);
    print_first_mismatch("K", k_cmp);
    print_first_mismatch("V", v_cmp);
    print_first_mismatch("Q_equals_input", q_eq_in);
    print_first_mismatch("K_equals_input", k_eq_in);
    print_first_mismatch("V_equals_input", v_eq_in);
    print_first_mismatch("SCORE_row_head0_token0", score_cmp);
    print_first_mismatch("PROB_row_head0_token0", prob_cmp);
    print_first_mismatch("POST", post_cmp);

    const bool qkv_exact = q_cmp.exact && k_cmp.exact && v_cmp.exact;
    const bool qkv_not_input = (!q_eq_in.exact) && (!k_eq_in.exact) && (!v_eq_in.exact);
    if (!qkv_exact || !qkv_not_input) {
        std::printf("[backup_attn_qkv][FAIL] Q/K/V acceptance gate failed\n");
        return 1;
    }
    if (masked_sem.masked_prob_nonzero != 0u || !masked_sem.masked_score_semantics_ok) {
        std::printf("[backup_attn_qkv][FAIL] masked score/prob semantics gate failed\n");
        return 1;
    }

    if (!post_cmp.exact) {
        std::printf("[backup_attn_qkv][INFO] post_concat remains mismatch (accepted for this wave)\n");
    }

    std::printf("PASS: tb_backup_attn_qkv_live_contract\n");
    return 0;
}

#endif // __SYNTHESIS__
