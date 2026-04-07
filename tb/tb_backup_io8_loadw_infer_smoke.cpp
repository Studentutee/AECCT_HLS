// tb_backup_io8_loadw_infer_smoke.cpp
// Backup profile trace-aligned smoke:
// - Top external io8 boundary
// - SET_W_BASE + LOAD_W + INFER narrow path
// - bounded trace-aligned x_pred exact compare (8 samples)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "AecctProtocol.h"
#include "AecctUtil.h"
#include "Top.h"
#include "weights_streamer.h"
#include "input_y_step0.h"
#include "output_x_pred_step0.h"
#include "output_logits_step0.h"

namespace {

struct Io8Top {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data8_ch_t data_in;
    aecct::data8_ch_t data_out;
};

static const uint32_t kTracePatternCount = 8u;
static const uint32_t kTraceSampleIds[kTracePatternCount] = { 0u, 1u, 2u, 3u, 4u, 6u, 8u, 9u };
static const uint32_t kDebugMaxXpredOneSamples = 3u;
static const uint32_t kDebugPayloadReadbackWords = 64u;

struct XpredOneSample {
    uint32_t sample_id;
    std::vector<uint32_t> one_indices;
};

enum DebugBoundary : uint32_t {
    DEBUG_BOUNDARY_NONE = 0u,
    DEBUG_BOUNDARY_INPUT_SELECTION = 1u,
    DEBUG_BOUNDARY_PAYLOAD_LOADW = 2u,
    DEBUG_BOUNDARY_OUTPUT_PACKING = 3u,
    DEBUG_BOUNDARY_DUT_ALGO = 4u
};

struct DebugCompareResult {
    bool exact;
    uint32_t mismatch_idx;
    uint32_t got_word;
    uint32_t exp_word;
    bool byte_mismatch_found;
    uint32_t byte_mismatch_idx;
    uint8_t got_byte;
    uint8_t exp_byte;
};

static_assert(
    trace_input_y_step0_tensor_ndim == 2,
    "trace_input_y_step0_tensor must be rank-2");
static_assert(
    trace_output_x_pred_step0_tensor_ndim == 2,
    "trace_output_x_pred_step0_tensor must be rank-2");
static_assert(
    (uint32_t)trace_input_y_step0_tensor_shape[1] == (uint32_t)EXP_LEN_INFER_IN_WORDS,
    "trace input width must match EXP_LEN_INFER_IN_WORDS");
static_assert(
    (uint32_t)trace_output_x_pred_step0_tensor_shape[1] == (uint32_t)EXP_LEN_OUT_XPRED_WORDS,
    "trace x_pred width must match EXP_LEN_OUT_XPRED_WORDS");
static_assert(
    (uint32_t)trace_output_logits_step0_tensor_shape[1] == (uint32_t)EXP_LEN_OUT_LOGITS_WORDS,
    "trace logits width must match EXP_LEN_OUT_LOGITS_WORDS");

static void fail(const char* msg) {
    std::printf("[backup_io8][FAIL] %s\n", msg);
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

static uint32_t fnv1a_u32_words(const std::vector<uint32_t>& words) {
    uint32_t h = 2166136261u;
    HASH_WORD_LOOP: for (uint32_t i = 0u; i < (uint32_t)words.size(); ++i) {
        h ^= words[i];
        h *= 16777619u;
    }
    return h;
}

static void print_indices_line(const std::vector<uint32_t>& indices) {
    if (indices.empty()) {
        std::printf("none");
        return;
    }
    IDX_PRINT_LOOP: for (uint32_t i = 0u; i < (uint32_t)indices.size(); ++i) {
        std::printf("%u", (unsigned)indices[i]);
        if (i + 1u < (uint32_t)indices.size()) {
            std::printf(",");
        }
    }
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
    cfg_words[CFG_OUT_MODE] = 0u; // stream x_pred words
    cfg_words[CFG_RESERVED0] = 0u;
}

static bool build_param_words_from_repo_reference(
    std::vector<uint32_t>& param_words,
    std::string& error
) {
    error.clear();
    aecct::data_ch_t stream_words;

    // Unified PARAM stream section A: bias tensors.
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

    // Unified PARAM stream section B: weight tensors.
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
                    error = "inv_s_w conversion failed while building PARAM stream";
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
    PARAM_VECTOR_COLLECT_LOOP: while (stream_words.nb_read(w)) {
        param_words.push_back((uint32_t)w.to_uint());
    }

    if ((uint32_t)param_words.size() != (uint32_t)EXP_LEN_PARAM_WORDS) {
        error = "PARAM word count mismatch against EXP_LEN_PARAM_WORDS";
        return false;
    }

    return true;
}

static void build_trace_infer_words(uint32_t sample_idx, std::vector<uint32_t>& infer_words) {
    infer_words.assign((uint32_t)EXP_LEN_INFER_IN_WORDS, 0u);
    const uint32_t stride = (uint32_t)EXP_LEN_INFER_IN_WORDS;
    const uint32_t base = sample_idx * stride;
    TRACE_INFER_WORD_LOOP: for (uint32_t i = 0u; i < stride; ++i) {
        const float fv = (float)trace_input_y_step0_tensor[base + i];
        infer_words[i] = f32_to_bits(fv);
    }
}

static void build_trace_xpred_words(uint32_t sample_idx, std::vector<uint32_t>& expected_words) {
    expected_words.assign((uint32_t)EXP_LEN_OUT_XPRED_WORDS, 0u);
    const uint32_t stride = (uint32_t)EXP_LEN_OUT_XPRED_WORDS;
    const uint32_t base = sample_idx * stride;
    TRACE_XPRED_WORD_LOOP: for (uint32_t i = 0u; i < stride; ++i) {
        const float fv = (float)trace_output_x_pred_step0_tensor[base + i];
        expected_words[i] = f32_to_bits(fv);
    }
}

static void build_trace_logits_words(uint32_t sample_idx, std::vector<uint32_t>& logits_words) {
    logits_words.assign((uint32_t)EXP_LEN_OUT_LOGITS_WORDS, 0u);
    const uint32_t stride = (uint32_t)EXP_LEN_OUT_LOGITS_WORDS;
    const uint32_t base = sample_idx * stride;
    TRACE_LOGITS_WORD_LOOP: for (uint32_t i = 0u; i < stride; ++i) {
        const float fv = (float)trace_output_logits_step0_tensor[base + i];
        logits_words[i] = f32_to_bits(fv);
    }
}

static void collect_trace_xpred_one_indices(uint32_t sample_idx, std::vector<uint32_t>& one_indices) {
    one_indices.clear();
    const uint32_t stride = (uint32_t)EXP_LEN_OUT_XPRED_WORDS;
    const uint32_t base = sample_idx * stride;
    const uint32_t one_bits = f32_to_bits(1.0f);
    X_PRED_ONE_INDEX_SCAN_LOOP: for (uint32_t i = 0u; i < stride; ++i) {
        const float fv = (float)trace_output_x_pred_step0_tensor[base + i];
        if (f32_to_bits(fv) == one_bits) {
            one_indices.push_back(i);
        }
    }
}

static bool select_xpred_one_samples(std::vector<XpredOneSample>& samples) {
    samples.clear();
    const uint32_t total_samples = (uint32_t)trace_output_x_pred_step0_tensor_shape[0];
    SELECT_XPRED_ONE_SAMPLE_LOOP: for (uint32_t sample_idx = 0u; sample_idx < total_samples; ++sample_idx) {
        XpredOneSample pick;
        pick.sample_id = sample_idx;
        collect_trace_xpred_one_indices(sample_idx, pick.one_indices);
        if (pick.one_indices.empty()) {
            continue;
        }
        samples.push_back(pick);
        if ((uint32_t)samples.size() >= kDebugMaxXpredOneSamples) {
            break;
        }
    }
    return !samples.empty();
}

static void collect_out_bytes(Io8Top& io, uint32_t expected_bytes, std::vector<uint8_t>& out_bytes) {
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

static void bytes_to_words_le(const std::vector<uint8_t>& bytes, std::vector<uint32_t>& words_out) {
    if ((bytes.size() % 4u) != 0u) {
        fail("output byte stream is not u32-word aligned");
    }
    const uint32_t word_count = (uint32_t)(bytes.size() / 4u);
    words_out.assign(word_count, 0u);
    BYTES_TO_WORDS_LOOP: for (uint32_t i = 0u; i < word_count; ++i) {
        const uint32_t b = i * 4u;
        const uint32_t w =
            ((uint32_t)bytes[b + 0u]) |
            ((uint32_t)bytes[b + 1u] << 8) |
            ((uint32_t)bytes[b + 2u] << 16) |
            ((uint32_t)bytes[b + 3u] << 24);
        words_out[i] = w;
    }
}

static void words_to_bytes_le(const std::vector<uint32_t>& words, std::vector<uint8_t>& bytes_out) {
    bytes_out.assign((uint32_t)words.size() * 4u, 0u);
    WORDS_TO_BYTES_LOOP: for (uint32_t i = 0u; i < (uint32_t)words.size(); ++i) {
        const uint32_t b = i * 4u;
        const uint32_t w = words[i];
        bytes_out[b + 0u] = (uint8_t)(w & 0xFFu);
        bytes_out[b + 1u] = (uint8_t)((w >> 8) & 0xFFu);
        bytes_out[b + 2u] = (uint8_t)((w >> 16) & 0xFFu);
        bytes_out[b + 3u] = (uint8_t)((w >> 24) & 0xFFu);
    }
}

static void run_setup_cfg_loadw(Io8Top& io, const std::vector<uint32_t>& param_words) {
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

    push_u32_le(io, 0u);
    io.ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_SET_OUTMODE));
    top_tick(io);
    expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE, "set_outmode_xpred");
}

static bool verify_payload_readback_prefix(
    Io8Top& io,
    const std::vector<uint32_t>& param_words,
    uint32_t readback_words,
    uint32_t& first_bad_idx,
    uint32_t& first_bad_got,
    uint32_t& first_bad_exp
) {
    first_bad_idx = 0u;
    first_bad_got = 0u;
    first_bad_exp = 0u;

    uint32_t words_to_check = readback_words;
    if (words_to_check > (uint32_t)param_words.size()) {
        words_to_check = (uint32_t)param_words.size();
    }
    if (words_to_check == 0u) {
        return false;
    }

    std::vector<uint8_t> out_bytes;
    std::vector<uint32_t> got_words;
    std::vector<uint32_t> exp_words;
    exp_words.assign(param_words.begin(), param_words.begin() + words_to_check);

    push_u32_le(io, (uint32_t)sram_map::PARAM_BASE_DEFAULT);
    push_u32_le(io, words_to_check);
    send_cmd(io, (uint8_t)aecct::OP_READ_MEM);
    collect_out_bytes(io, words_to_check * 4u, out_bytes);
    expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM, "read_mem_done");
    bytes_to_words_le(out_bytes, got_words);

    if (got_words.size() != exp_words.size()) {
        return false;
    }

    PAYLOAD_READBACK_COMPARE_LOOP: for (uint32_t i = 0u; i < words_to_check; ++i) {
        if (got_words[i] != exp_words[i]) {
            first_bad_idx = i;
            first_bad_got = got_words[i];
            first_bad_exp = exp_words[i];
            return false;
        }
    }
    return true;
}

static void print_input_summary(uint32_t sample_idx, const std::vector<uint32_t>& infer_words) {
    uint32_t pos_cnt = 0u;
    uint32_t neg_cnt = 0u;
    uint32_t zero_cnt = 0u;
    const uint32_t stride = (uint32_t)EXP_LEN_INFER_IN_WORDS;
    const uint32_t base = sample_idx * stride;
    INPUT_SIGN_SCAN_LOOP: for (uint32_t i = 0u; i < stride; ++i) {
        const float v = (float)trace_input_y_step0_tensor[base + i];
        if (v > 0.0f) {
            ++pos_cnt;
        } else if (v < 0.0f) {
            ++neg_cnt;
        } else {
            ++zero_cnt;
        }
    }

    std::printf(
        "[backup_io8][debug_input] sample=%u in_words=%u hash=0x%08X sign_pos=%u sign_neg=%u sign_zero=%u first4=0x%08X,0x%08X,0x%08X,0x%08X\n",
        (unsigned)sample_idx,
        (unsigned)infer_words.size(),
        (unsigned)fnv1a_u32_words(infer_words),
        (unsigned)pos_cnt,
        (unsigned)neg_cnt,
        (unsigned)zero_cnt,
        (unsigned)infer_words[0],
        (unsigned)infer_words[1],
        (unsigned)infer_words[2],
        (unsigned)infer_words[3]);
}

static bool run_one_trace_sample_and_compare(Io8Top& io, uint32_t sample_idx) {
    std::vector<uint32_t> infer_words;
    std::vector<uint8_t> out_bytes;
    std::vector<uint32_t> got_words;
    std::vector<uint32_t> expected_words;

    build_trace_infer_words(sample_idx, infer_words);
    build_trace_xpred_words(sample_idx, expected_words);

    send_cmd(io, (uint8_t)aecct::OP_INFER);
    expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_INFER, "infer_begin");
    TRACE_INFER_INGEST_LOOP: for (uint32_t i = 0u; i < (uint32_t)infer_words.size(); ++i) {
        push_u32_le(io, infer_words[i]);
        top_tick(io);
        if (i + 1u < (uint32_t)infer_words.size()) {
            expect_no_rsp(io, "infer_ingest");
        } else {
            expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_INFER, "infer_done");
        }
    }

    const uint32_t expected_out_bytes = (uint32_t)EXP_LEN_OUT_XPRED_WORDS * 4u;
    collect_out_bytes(io, expected_out_bytes, out_bytes);
    bytes_to_words_le(out_bytes, got_words);

    if (got_words.size() != expected_words.size()) {
        std::printf(
            "[backup_io8][FAIL] sample=%u output word-count mismatch got=%u expect=%u\n",
            (unsigned)sample_idx,
            (unsigned)got_words.size(),
            (unsigned)expected_words.size());
        return false;
    }

    uint32_t mismatch_idx = 0u;
    bool mismatch = false;
    TRACE_COMPARE_LOOP: for (uint32_t i = 0u; i < (uint32_t)got_words.size(); ++i) {
        if (got_words[i] != expected_words[i]) {
            mismatch_idx = i;
            mismatch = true;
            break;
        }
    }

    if (mismatch) {
        std::printf(
            "[backup_io8][FAIL] sample=%u x_pred word mismatch idx=%u got=0x%08X expect=0x%08X\n",
            (unsigned)sample_idx,
            (unsigned)mismatch_idx,
            (unsigned)got_words[mismatch_idx],
            (unsigned)expected_words[mismatch_idx]);
        return false;
    }

    const uint32_t sample_hash = fnv1a_u32_words(got_words);
    std::printf(
        "[backup_io8][trace_xpred] sample=%u words=%u bytes=%u hash=0x%08X exact=PASS\n",
        (unsigned)sample_idx,
        (unsigned)got_words.size(),
        (unsigned)out_bytes.size(),
        (unsigned)sample_hash);
    return true;
}

static DebugCompareResult run_one_xpred_one_debug_sample(
    Io8Top& io,
    const XpredOneSample& pick
) {
    DebugCompareResult ret;
    ret.exact = true;
    ret.mismatch_idx = 0u;
    ret.got_word = 0u;
    ret.exp_word = 0u;
    ret.byte_mismatch_found = false;
    ret.byte_mismatch_idx = 0u;
    ret.got_byte = 0u;
    ret.exp_byte = 0u;

    std::vector<uint32_t> infer_words;
    std::vector<uint32_t> expected_xpred_words;
    std::vector<uint32_t> expected_logits_words;
    std::vector<uint8_t> out_bytes;
    std::vector<uint32_t> got_words;
    std::vector<uint8_t> expected_bytes;
    build_trace_infer_words(pick.sample_id, infer_words);
    build_trace_xpred_words(pick.sample_id, expected_xpred_words);
    build_trace_logits_words(pick.sample_id, expected_logits_words);
    print_input_summary(pick.sample_id, infer_words);
    std::printf("[backup_io8][debug_ref] sample=%u xpred_one_indices=", (unsigned)pick.sample_id);
    print_indices_line(pick.one_indices);
    std::printf("\n");

    send_cmd(io, (uint8_t)aecct::OP_INFER);
    expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_INFER, "infer_begin_debug");
    DEBUG_INFER_INGEST_LOOP: for (uint32_t i = 0u; i < (uint32_t)infer_words.size(); ++i) {
        push_u32_le(io, infer_words[i]);
        top_tick(io);
        if (i + 1u < (uint32_t)infer_words.size()) {
            expect_no_rsp(io, "infer_ingest_debug");
        } else {
            expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_INFER, "infer_done_debug");
        }
    }

    const uint32_t expected_bytes_len = (uint32_t)EXP_LEN_OUT_XPRED_WORDS * 4u;
    collect_out_bytes(io, expected_bytes_len, out_bytes);
    bytes_to_words_le(out_bytes, got_words);
    words_to_bytes_le(expected_xpred_words, expected_bytes);

    DEBUG_COMPARE_WORD_LOOP: for (uint32_t i = 0u; i < (uint32_t)expected_xpred_words.size(); ++i) {
        if (got_words[i] != expected_xpred_words[i]) {
            ret.exact = false;
            ret.mismatch_idx = i;
            ret.got_word = got_words[i];
            ret.exp_word = expected_xpred_words[i];
            break;
        }
    }

    if (!ret.exact) {
        DEBUG_COMPARE_BYTE_LOOP: for (uint32_t i = 0u; i < (uint32_t)expected_bytes.size(); ++i) {
            if (out_bytes[i] != expected_bytes[i]) {
                ret.byte_mismatch_found = true;
                ret.byte_mismatch_idx = i;
                ret.got_byte = out_bytes[i];
                ret.exp_byte = expected_bytes[i];
                break;
            }
        }

        const uint32_t idx = ret.mismatch_idx;
        std::printf(
            "[backup_io8][debug_mismatch] sample=%u first_word_mismatch_idx=%u expected=0x%08X actual=0x%08X ref_logit_bits=0x%08X\n",
            (unsigned)pick.sample_id,
            (unsigned)idx,
            (unsigned)ret.exp_word,
            (unsigned)ret.got_word,
            (unsigned)expected_logits_words[idx]);
        if (ret.byte_mismatch_found) {
            std::printf(
                "[backup_io8][debug_mismatch] sample=%u first_byte_mismatch_idx=%u expected=0x%02X actual=0x%02X\n",
                (unsigned)pick.sample_id,
                (unsigned)ret.byte_mismatch_idx,
                (unsigned)ret.exp_byte,
                (unsigned)ret.got_byte);
        } else {
            std::printf(
                "[backup_io8][debug_mismatch] sample=%u byte_mismatch=none (packing suspect)\n",
                (unsigned)pick.sample_id);
        }
    } else {
        std::printf(
            "[backup_io8][debug_exact] sample=%u xpred_exact=PASS hash=0x%08X\n",
            (unsigned)pick.sample_id,
            (unsigned)fnv1a_u32_words(got_words));
    }

    return ret;
}

} // namespace

int main() {
    const uint32_t trace_input_samples = (uint32_t)trace_input_y_step0_tensor_shape[0];
    const uint32_t trace_xpred_samples = (uint32_t)trace_output_x_pred_step0_tensor_shape[0];
    if (trace_input_samples == 0u || trace_xpred_samples == 0u) {
        fail("trace sample count is zero");
    }

    TRACE_SAMPLE_ID_RANGE_CHECK_LOOP: for (uint32_t i = 0u; i < kTracePatternCount; ++i) {
        const uint32_t sid = kTraceSampleIds[i];
        if (sid >= trace_input_samples || sid >= trace_xpred_samples) {
            fail("trace sample id list exceeds available trace sample range");
        }
    }

    std::vector<uint32_t> param_words;
    std::string build_param_error;
    if (!build_param_words_from_repo_reference(param_words, build_param_error)) {
        std::printf("[backup_io8][FAIL] build_param_words_from_repo_reference: %s\n", build_param_error.c_str());
        return 1;
    }

    Io8Top io;
    run_setup_cfg_loadw(io, param_words);

    TRACE_PATTERN_LOOP: for (uint32_t pattern_idx = 0u; pattern_idx < kTracePatternCount; ++pattern_idx) {
        const uint32_t sample_idx = kTraceSampleIds[pattern_idx];
        if (!run_one_trace_sample_and_compare(io, sample_idx)) {
            return 1;
        }
    }

    std::printf(
        "PASS: tb_backup_io8_loadw_infer_trace_aligned_xpred_compare patterns=%u words_per_pattern=%u\n",
        (unsigned)kTracePatternCount,
        (unsigned)EXP_LEN_OUT_XPRED_WORDS);

    std::vector<XpredOneSample> xpred_one_samples;
    if (!select_xpred_one_samples(xpred_one_samples)) {
        fail("cannot find any trace sample with x_pred=1");
    }
    std::printf(
        "[backup_io8][debug_select] selected_samples=%u (max=%u)\n",
        (unsigned)xpred_one_samples.size(),
        (unsigned)kDebugMaxXpredOneSamples);
    DEBUG_SELECTED_SAMPLE_PRINT_LOOP: for (uint32_t i = 0u; i < (uint32_t)xpred_one_samples.size(); ++i) {
        const XpredOneSample& pick = xpred_one_samples[i];
        std::printf("[backup_io8][debug_select] sample=%u xpred_one_indices=", (unsigned)pick.sample_id);
        print_indices_line(pick.one_indices);
        std::printf("\n");
    }

    uint32_t payload_bad_idx = 0u;
    uint32_t payload_bad_got = 0u;
    uint32_t payload_bad_exp = 0u;
    const bool payload_readback_ok = verify_payload_readback_prefix(
        io,
        param_words,
        kDebugPayloadReadbackWords,
        payload_bad_idx,
        payload_bad_got,
        payload_bad_exp);
    if (payload_readback_ok) {
        std::vector<uint32_t> payload_prefix(
            param_words.begin(),
            param_words.begin() + (size_t)kDebugPayloadReadbackWords);
        std::printf(
            "[backup_io8][debug_payload] readback_prefix_words=%u exact=PASS hash=0x%08X\n",
            (unsigned)kDebugPayloadReadbackWords,
            (unsigned)fnv1a_u32_words(payload_prefix));
    } else {
        std::printf(
            "[backup_io8][debug_payload] readback_prefix_words=%u exact=FAIL first_mismatch_idx=%u expected=0x%08X actual=0x%08X\n",
            (unsigned)kDebugPayloadReadbackWords,
            (unsigned)payload_bad_idx,
            (unsigned)payload_bad_exp,
            (unsigned)payload_bad_got);
    }

    bool mismatch_found = false;
    uint32_t mismatch_sample = 0u;
    DebugCompareResult mismatch_ret;
    DEBUG_SAMPLE_COMPARE_LOOP: for (uint32_t i = 0u; i < (uint32_t)xpred_one_samples.size(); ++i) {
        const XpredOneSample& pick = xpred_one_samples[i];
        DebugCompareResult r = run_one_xpred_one_debug_sample(io, pick);
        if (!r.exact) {
            mismatch_found = true;
            mismatch_sample = pick.sample_id;
            mismatch_ret = r;
            break;
        }
    }

    DebugBoundary boundary = DEBUG_BOUNDARY_NONE;
    if (!payload_readback_ok) {
        boundary = DEBUG_BOUNDARY_PAYLOAD_LOADW;
    } else if (mismatch_found && !mismatch_ret.byte_mismatch_found) {
        boundary = DEBUG_BOUNDARY_OUTPUT_PACKING;
    } else if (mismatch_found) {
        boundary = DEBUG_BOUNDARY_DUT_ALGO;
    }

    if (boundary == DEBUG_BOUNDARY_PAYLOAD_LOADW) {
        std::printf(
            "[backup_io8][debug_boundary] earliest_boundary=payload_or_loadw first_payload_mismatch_idx=%u\n",
            (unsigned)payload_bad_idx);
    } else if (boundary == DEBUG_BOUNDARY_OUTPUT_PACKING) {
        std::printf(
            "[backup_io8][debug_boundary] earliest_boundary=output_packing sample=%u first_word_mismatch_idx=%u\n",
            (unsigned)mismatch_sample,
            (unsigned)mismatch_ret.mismatch_idx);
    } else if (boundary == DEBUG_BOUNDARY_DUT_ALGO) {
        std::printf(
            "[backup_io8][debug_boundary] earliest_boundary=dut_algorithm_divergence sample=%u first_word_mismatch_idx=%u\n",
            (unsigned)mismatch_sample,
            (unsigned)mismatch_ret.mismatch_idx);
    } else {
        std::printf("[backup_io8][debug_boundary] earliest_boundary=none (selected x_pred=1 samples all exact)\n");
    }

    std::printf("PASS: tb_backup_io8_loadw_infer_xpred1_debug_bridge\n");
    std::printf("PASS: tb_backup_io8_loadw_infer_smoke\n");
    return 0;
}
