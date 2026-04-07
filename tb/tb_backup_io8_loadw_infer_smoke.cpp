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
#include "RefModel.h"

namespace {

struct Io8Top {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data8_ch_t data_in;
    aecct::data8_ch_t data_out;
};

static const uint32_t kTracePatternCount = 8u;
static const uint32_t kTraceSampleIds[kTracePatternCount] = { 0u, 1u, 2u, 3u, 4u, 6u, 8u, 9u };
static const uint32_t kDebugMaxXpredOneSamples = 1u;
static const uint32_t kDebugPreferredSampleId = 5u;
static const uint32_t kDebugFocusedIdx = 31u;
static const uint32_t kContractSample0 = 0u;
static const uint32_t kContractIdx0 = 6u;
static const uint32_t kDebugPayloadReadbackWords = 64u;
static const uint32_t kDebugPayloadWindowWords = 16u;

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

enum PayloadResponsibilityBoundary : uint32_t {
    PAYLOAD_BOUNDARY_UNKNOWN = 0u,
    PAYLOAD_BOUNDARY_A_EXPECTED_STREAM = 1u,
    PAYLOAD_BOUNDARY_B_PARAM_ASSEMBLY = 2u,
    PAYLOAD_BOUNDARY_C_LOADW_WRITE = 3u,
    PAYLOAD_BOUNDARY_D_READ_MEM = 4u
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
    uint32_t focused_idx;
    bool st31_exact;
    bool logit31_exact;
    bool xpred31_exact;
    bool st_any_mismatch;
    uint32_t st_first_mismatch_idx;
    uint32_t dut_st31_bits;
    uint32_t ref_st31_bits;
    uint32_t dut_logit31_bits;
    uint32_t ref_logit31_bits;
    uint32_t trace_logit31_bits;
    uint32_t dut_xpred31_bits;
    uint32_t ref_xpred31_bits;
    uint32_t y31_bits;
    uint32_t boundary_bucket; // A=1, B=2, C=3, D=4
};

struct LocalContractResult {
    uint32_t sample_idx;
    uint32_t focused_idx;
    uint32_t dut_st_bits;
    uint32_t ref_st_bits;
    uint32_t dut_logit_bits;
    uint32_t ref_logit_bits;
    uint32_t trace_logit_bits;
    uint32_t dut_xpred_bits;
    uint32_t trace_xpred_bits;
    uint32_t y_bits_raw;
    uint32_t rule_xpred_bits;
    bool st_match;
    bool logit_match;
    bool xpred_rule_match;
    bool local_ref_ok;
    bool trace_mismatch;
    bool dut_bug_candidate;
};

struct RefModelStageCompareResult {
    uint32_t sample_idx;
    uint32_t focused_idx;
    bool st_exact;
    bool logit_exact;
    bool xpred_exact;
    bool all_exact;
    uint32_t st_first_mismatch_idx;
    uint32_t st_dut_bits;
    uint32_t st_ref_bits;
    uint32_t logit_first_mismatch_idx;
    uint32_t logit_dut_bits;
    uint32_t logit_ref_bits;
    uint32_t xpred_first_mismatch_idx;
    uint32_t xpred_dut_bits;
    uint32_t xpred_ref_bits;
    uint32_t boundary_bucket; // A=1, B=2, C=3, D=4
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

static float bits_to_f32(uint32_t u) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.u = u;
    return cvt.f;
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
    if (kDebugPreferredSampleId < total_samples) {
        XpredOneSample pick;
        pick.sample_id = kDebugPreferredSampleId;
        collect_trace_xpred_one_indices(kDebugPreferredSampleId, pick.one_indices);
        if (!pick.one_indices.empty()) {
            samples.push_back(pick);
            return true;
        }
    }

    SELECT_XPRED_ONE_SAMPLE_LOOP: for (uint32_t sample_idx = 0u; sample_idx < total_samples; ++sample_idx) {
        if (sample_idx == kDebugPreferredSampleId) {
            continue;
        }
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

static void run_setup_cfg_loadw(
    Io8Top& io,
    const std::vector<uint32_t>& param_words,
    const char* session_tag
) {
    const char* tag = (session_tag != 0) ? session_tag : "session";
    std::printf("[backup_io8][session] %s session begin\n", tag);
    uint32_t cfg_words[EXP_LEN_CFG_WORDS];
    build_cfg_words(cfg_words);

    send_cmd(io, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET, "soft_reset");
    std::printf("[backup_io8][session] %s setup_soft_reset_done=1\n", tag);

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
    std::printf("[backup_io8][session] %s setup_cfg_done=1\n", tag);

    push_u32_le(io, (uint32_t)sram_map::PARAM_BASE_DEFAULT);
    io.ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_SET_W_BASE));
    top_tick(io);
    expect_rsp_kind_either(
        io,
        (uint8_t)aecct::RSP_OK,
        (uint8_t)aecct::RSP_DONE,
        (uint8_t)aecct::OP_SET_W_BASE,
        "set_w_base");
    std::printf("[backup_io8][session] %s setup_set_w_base_done=1\n", tag);

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
    std::printf("[backup_io8][session] %s setup_load_w_done=1\n", tag);

    push_u32_le(io, 0u);
    io.ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_SET_OUTMODE));
    top_tick(io);
    expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE, "set_outmode_xpred");
    std::printf("[backup_io8][session] %s setup_set_outmode_done=1\n", tag);
    std::printf(
        "[backup_io8][session] %s setup_complete soft_reset=1 cfg=1 load_w=1 set_outmode=1\n",
        tag);
}

static uint32_t clip_words_to_check(
    const std::vector<uint32_t>& param_words,
    uint32_t requested_words
) {
    uint32_t words_to_check = requested_words;
    if (words_to_check > (uint32_t)param_words.size()) {
        words_to_check = (uint32_t)param_words.size();
    }
    return words_to_check;
}

static void read_mem_words(
    Io8Top& io,
    uint32_t base_word,
    uint32_t readback_words,
    std::vector<uint32_t>& out_words
) {
    out_words.clear();
    if (readback_words == 0u) {
        return;
    }

    std::vector<uint8_t> out_bytes;
    push_u32_le(io, base_word);
    push_u32_le(io, readback_words);
    send_cmd(io, (uint8_t)aecct::OP_READ_MEM);
    collect_out_bytes(io, readback_words * 4u, out_bytes);
    expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM, "read_mem_done");
    bytes_to_words_le(out_bytes, out_words);
}

static void read_sram_words_direct(
    uint32_t base_word,
    uint32_t readback_words,
    std::vector<uint32_t>& out_words
) {
    out_words.assign(readback_words, 0u);
    const aecct::u32_t* sram = aecct::top_sram();
    SRAM_DIRECT_READ_LOOP: for (uint32_t i = 0u; i < readback_words; ++i) {
        out_words[i] = (uint32_t)sram[base_word + i].to_uint();
    }
}

static bool compare_word_vectors_exact(
    const std::vector<uint32_t>& expected_words,
    const std::vector<uint32_t>& actual_words,
    uint32_t& first_bad_idx,
    uint32_t& first_bad_got,
    uint32_t& first_bad_exp
) {
    first_bad_idx = 0u;
    first_bad_got = 0u;
    first_bad_exp = 0u;
    if (expected_words.size() != actual_words.size()) {
        return false;
    }
    PAYLOAD_COMPARE_LOOP: for (uint32_t i = 0u; i < (uint32_t)expected_words.size(); ++i) {
        if (actual_words[i] != expected_words[i]) {
            first_bad_idx = i;
            first_bad_got = actual_words[i];
            first_bad_exp = expected_words[i];
            return false;
        }
    }
    return true;
}

static bool map_param_word_index(
    uint32_t word_idx,
    uint32_t& out_param_id,
    uint32_t& out_local_word_idx
) {
    MAP_PARAM_WORD_LOOP: for (uint32_t pid = 0u; pid < (uint32_t)PARAM_COUNT; ++pid) {
        const uint32_t begin = kParamMeta[pid].offset_w;
        const uint32_t len_w = kParamMeta[pid].len_w;
        const uint32_t end_excl = begin + len_w;
        if (word_idx >= begin && word_idx < end_excl) {
            out_param_id = pid;
            out_local_word_idx = word_idx - begin;
            return true;
        }
    }
    return false;
}

static const char* payload_section_name(uint32_t param_id) {
    return (param_id < (uint32_t)BIAS_COUNT) ? "bias_section_a" : "weight_section_b";
}

static const char* payload_emit_helper_name(uint32_t param_id, uint32_t dtype) {
    if (param_id < (uint32_t)BIAS_COUNT) {
        return "tb_emit_fp32_words_from_fp64(bias)";
    }
    const uint32_t weight_slot = param_id - (uint32_t)BIAS_COUNT;
    if (weight_slot >= (uint32_t)WEIGHT_COUNT) {
        return "unknown_weight_slot";
    }
    const WeightId wid = (WeightId)weight_slot;
    if (dtype == 0u) {
        if (is_quant_linear_inv_sw_weight_slot(wid)) {
            return "tb_emit_inv_sw_words_from_fp64";
        }
        return "tb_emit_fp32_words_from_fp64(weight)";
    }
    return "tb_emit_bitpack_words";
}

static bool try_get_source_word_for_expected(
    uint32_t param_id,
    uint32_t local_word_idx,
    uint32_t& out_word
) {
    if (param_id < (uint32_t)BIAS_COUNT) {
        uint32_t numel = 0u;
        const double* ptr = tb_lookup_bias_fp64((BiasId)param_id, numel);
        if (ptr == 0) {
            return false;
        }
        if (local_word_idx < numel) {
            out_word = f32_to_bits((float)ptr[local_word_idx]);
        } else {
            out_word = 0u;
        }
        return true;
    }

    const uint32_t weight_slot = param_id - (uint32_t)BIAS_COUNT;
    if (weight_slot >= (uint32_t)WEIGHT_COUNT) {
        return false;
    }

    const WeightId wid = (WeightId)weight_slot;
    const TensorMeta meta = kWeightMeta[weight_slot];
    if (meta.dtype != 0u || is_quant_linear_inv_sw_weight_slot(wid)) {
        return false;
    }

    uint32_t numel = 0u;
    const double* ptr = tb_lookup_weight_fp64(wid, numel);
    if (ptr == 0) {
        return false;
    }
    if (local_word_idx < numel) {
        out_word = f32_to_bits((float)ptr[local_word_idx]);
    } else {
        out_word = 0u;
    }
    return true;
}

static void print_payload_stream_order_summary() {
    std::printf(
        "[backup_io8][debug_payload_expected] stream_order=sectionA_bias(param_id=0..%u) + sectionB_weight(param_id=%u..%u)\n",
        (unsigned)((uint32_t)BIAS_COUNT - 1u),
        (unsigned)(uint32_t)BIAS_COUNT,
        (unsigned)((uint32_t)PARAM_COUNT - 1u));
}

static void print_expected_payload_prefix_with_source(
    const std::vector<uint32_t>& param_words,
    uint32_t window_words
) {
    const uint32_t words_to_print = clip_words_to_check(param_words, window_words);
    print_payload_stream_order_summary();
    EXPECTED_PREFIX_DUMP_LOOP: for (uint32_t i = 0u; i < words_to_print; ++i) {
        uint32_t param_id = 0u;
        uint32_t local_idx = 0u;
        if (!map_param_word_index(i, param_id, local_idx)) {
            std::printf(
                "[backup_io8][debug_payload_expected] idx=%u word=0x%08X source=unmapped\n",
                (unsigned)i,
                (unsigned)param_words[i]);
            continue;
        }

        const ParamMeta meta = kParamMeta[param_id];
        uint32_t source_word = 0u;
        const bool source_known = try_get_source_word_for_expected(param_id, local_idx, source_word);
        const bool source_match = source_known && (source_word == param_words[i]);

        std::printf(
            "[backup_io8][debug_payload_expected] idx=%u word=0x%08X param_id=%u key=%s section=%s dtype=%u local_word=%u helper=%s source_known=%u source_word=0x%08X source_match=%u\n",
            (unsigned)i,
            (unsigned)param_words[i],
            (unsigned)param_id,
            kParamKey[param_id],
            payload_section_name(param_id),
            (unsigned)meta.dtype,
            (unsigned)local_idx,
            payload_emit_helper_name(param_id, meta.dtype),
            (unsigned)(source_known ? 1u : 0u),
            (unsigned)source_word,
            (unsigned)(source_match ? 1u : 0u));
    }
}

static void print_payload_compare_window(
    const char* tag,
    const std::vector<uint32_t>& expected_words,
    const std::vector<uint32_t>& actual_words,
    uint32_t window_words
) {
    uint32_t words_to_print = window_words;
    if (words_to_print > (uint32_t)expected_words.size()) {
        words_to_print = (uint32_t)expected_words.size();
    }
    if (words_to_print > (uint32_t)actual_words.size()) {
        words_to_print = (uint32_t)actual_words.size();
    }

    PAYLOAD_WINDOW_DUMP_LOOP: for (uint32_t i = 0u; i < words_to_print; ++i) {
        uint32_t param_id = 0u;
        uint32_t local_idx = 0u;
        const bool mapped = map_param_word_index(i, param_id, local_idx);
        const char* key = mapped ? kParamKey[param_id] : "unmapped";
        std::printf(
            "[backup_io8][%s] idx=%u expected=0x%08X actual=0x%08X exact=%s param_id=%u key=%s local_word=%u\n",
            tag,
            (unsigned)i,
            (unsigned)expected_words[i],
            (unsigned)actual_words[i],
            (expected_words[i] == actual_words[i]) ? "PASS" : "FAIL",
            (unsigned)param_id,
            key,
            (unsigned)local_idx);
    }
}

static bool find_first_occurrence(
    const std::vector<uint32_t>& words,
    uint32_t needle,
    uint32_t& out_idx
) {
    FIND_WORD_OCC_LOOP: for (uint32_t i = 0u; i < (uint32_t)words.size(); ++i) {
        if (words[i] == needle) {
            out_idx = i;
            return true;
        }
    }
    return false;
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

static bool run_one_trace_sample_and_compare(
    Io8Top& io,
    uint32_t sample_idx,
    uint32_t& mismatch_idx_out,
    uint32_t& got_word_out,
    uint32_t& exp_word_out
) {
    mismatch_idx_out = 0u;
    got_word_out = 0u;
    exp_word_out = 0u;
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
            "[backup_io8][trace_diag] sample=%u output word-count mismatch got=%u expect=%u\n",
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
        mismatch_idx_out = mismatch_idx;
        got_word_out = got_words[mismatch_idx];
        exp_word_out = expected_words[mismatch_idx];
        std::printf(
            "[backup_io8][trace_diag] sample=%u x_pred word mismatch idx=%u got=0x%08X expect=0x%08X\n",
            (unsigned)sample_idx,
            (unsigned)mismatch_idx,
            (unsigned)got_word_out,
            (unsigned)exp_word_out);
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

static LocalContractResult run_one_local_contract_focus_sample(
    Io8Top& io,
    uint32_t sample_idx,
    uint32_t focused_idx
) {
    LocalContractResult r;
    r.sample_idx = sample_idx;
    r.focused_idx = focused_idx;
    r.dut_st_bits = 0u;
    r.ref_st_bits = 0u;
    r.dut_logit_bits = 0u;
    r.ref_logit_bits = 0u;
    r.trace_logit_bits = 0u;
    r.dut_xpred_bits = 0u;
    r.trace_xpred_bits = 0u;
    r.y_bits_raw = 0u;
    r.rule_xpred_bits = 0u;
    r.st_match = false;
    r.logit_match = false;
    r.xpred_rule_match = false;
    r.local_ref_ok = false;
    r.trace_mismatch = false;
    r.dut_bug_candidate = true;

    std::vector<uint32_t> infer_words;
    std::vector<uint32_t> expected_xpred_words;
    std::vector<uint32_t> expected_logits_words;
    std::vector<uint8_t> out_bytes;
    std::vector<uint32_t> got_words;
    build_trace_infer_words(sample_idx, infer_words);
    build_trace_xpred_words(sample_idx, expected_xpred_words);
    build_trace_logits_words(sample_idx, expected_logits_words);

    if (focused_idx >= (uint32_t)EXP_LEN_OUT_XPRED_WORDS ||
        focused_idx >= (uint32_t)EXP_LEN_OUT_LOGITS_WORDS ||
        focused_idx >= (uint32_t)N_NODES ||
        focused_idx >= (uint32_t)infer_words.size()) {
        fail("local contract focused idx out of range");
    }

    send_cmd(io, (uint8_t)aecct::OP_INFER);
    expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_INFER, "infer_begin_local_contract");
    LOCAL_CONTRACT_INGEST_LOOP: for (uint32_t i = 0u; i < (uint32_t)infer_words.size(); ++i) {
        push_u32_le(io, infer_words[i]);
        top_tick(io);
        if (i + 1u < (uint32_t)infer_words.size()) {
            expect_no_rsp(io, "infer_ingest_local_contract");
        } else {
            expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_INFER, "infer_done_local_contract");
        }
    }

    const uint32_t expected_bytes_len = (uint32_t)EXP_LEN_OUT_XPRED_WORDS * 4u;
    collect_out_bytes(io, expected_bytes_len, out_bytes);
    bytes_to_words_le(out_bytes, got_words);
    if ((uint32_t)got_words.size() != (uint32_t)EXP_LEN_OUT_XPRED_WORDS) {
        fail("local contract output words mismatch EXP_LEN_OUT_XPRED_WORDS");
    }

    const aecct::u32_t* sram = aecct::top_sram();
    const uint32_t x_end_base = (uint32_t)aecct::top_peek_infer_final_x_base_word().to_uint();
    const uint32_t logits_base = (uint32_t)aecct::top_peek_infer_logits_base_word().to_uint();
    const uint32_t final_scalar_base = (uint32_t)sram_map::SCR_FINAL_SCALAR_BASE_W;
    const aecct::HeadParamBase hp = aecct::make_head_param_base(aecct::top_peek_w_base_word());
    const uint32_t ffn1_w_base = (uint32_t)hp.ffn1_w_base_word.to_uint();
    const uint32_t ffn1_b_base = (uint32_t)hp.ffn1_b_base_word.to_uint();
    const uint32_t out_fc_w_base = (uint32_t)hp.out_fc_w_base_word.to_uint();
    const uint32_t out_fc_b_base = (uint32_t)hp.out_fc_b_base_word.to_uint();
    const uint32_t token_count = (uint32_t)N_NODES;
    const uint32_t d_model = (uint32_t)D_MODEL;

    std::vector<uint32_t> ref_st_words(token_count, 0u);
    LOCAL_CONTRACT_REF_ST_LOOP: for (uint32_t t = 0u; t < token_count; ++t) {
        const uint32_t x_row_base = x_end_base + t * d_model;
        float acc = bits_to_f32((uint32_t)sram[ffn1_b_base + 0u].to_uint());
        LOCAL_CONTRACT_REF_ST_DOT_LOOP: for (uint32_t i = 0u; i < d_model; ++i) {
            const float xv = bits_to_f32((uint32_t)sram[x_row_base + i].to_uint());
            const float wv = bits_to_f32((uint32_t)sram[ffn1_w_base + i].to_uint());
            acc += (xv * wv);
        }
        ref_st_words[t] = f32_to_bits(acc);
    }

    r.dut_st_bits = (uint32_t)sram[final_scalar_base + focused_idx].to_uint();
    r.ref_st_bits = ref_st_words[focused_idx];
    r.st_match = (r.dut_st_bits == r.ref_st_bits);

    float ref_logit = bits_to_f32((uint32_t)sram[out_fc_b_base + focused_idx].to_uint());
    LOCAL_CONTRACT_REF_LOGIT_LOOP: for (uint32_t t = 0u; t < token_count; ++t) {
        const float stv = bits_to_f32(ref_st_words[t]);
        const float wv = bits_to_f32((uint32_t)sram[out_fc_w_base + focused_idx * token_count + t].to_uint());
        ref_logit += (stv * wv);
    }
    r.ref_logit_bits = f32_to_bits(ref_logit);
    r.dut_logit_bits = (uint32_t)sram[logits_base + focused_idx].to_uint();
    r.trace_logit_bits = expected_logits_words[focused_idx];
    r.logit_match = (r.dut_logit_bits == r.ref_logit_bits);

    r.dut_xpred_bits = got_words[focused_idx];
    r.trace_xpred_bits = expected_xpred_words[focused_idx];
    r.y_bits_raw = infer_words[focused_idx];
    const float y = bits_to_f32(r.y_bits_raw);
    const float sign_y = (y > 0.0f) ? 1.0f : ((y < 0.0f) ? -1.0f : 0.0f);
    const float dut_logit = bits_to_f32(r.dut_logit_bits);
    r.rule_xpred_bits = f32_to_bits(((dut_logit * sign_y) < 0.0f) ? 1.0f : 0.0f);
    r.xpred_rule_match = (r.dut_xpred_bits == r.rule_xpred_bits);

    r.local_ref_ok = r.st_match && r.logit_match && r.xpred_rule_match;
    r.trace_mismatch = (r.dut_logit_bits != r.trace_logit_bits) || (r.dut_xpred_bits != r.trace_xpred_bits);
    r.dut_bug_candidate = !r.local_ref_ok;

    std::printf(
        "[backup_io8][contract_triage] sample=%u idx=%u dut_st=0x%08X ref_st=0x%08X dut_logit=0x%08X ref_logit=0x%08X trace_logit=0x%08X dut_xpred=0x%08X trace_xpred=0x%08X y_bits_raw=0x%08X rule_xpred=0x%08X\n",
        (unsigned)sample_idx,
        (unsigned)focused_idx,
        (unsigned)r.dut_st_bits,
        (unsigned)r.ref_st_bits,
        (unsigned)r.dut_logit_bits,
        (unsigned)r.ref_logit_bits,
        (unsigned)r.trace_logit_bits,
        (unsigned)r.dut_xpred_bits,
        (unsigned)r.trace_xpred_bits,
        (unsigned)r.y_bits_raw,
        (unsigned)r.rule_xpred_bits);
    std::printf(
        "[backup_io8][contract_triage] sample=%u idx=%u st_match=%u logit_match=%u xpred_rule_match=%u local_ref_ok=%u trace_mismatch=%u verdict=%s\n",
        (unsigned)sample_idx,
        (unsigned)focused_idx,
        (unsigned)(r.st_match ? 1u : 0u),
        (unsigned)(r.logit_match ? 1u : 0u),
        (unsigned)(r.xpred_rule_match ? 1u : 0u),
        (unsigned)(r.local_ref_ok ? 1u : 0u),
        (unsigned)(r.trace_mismatch ? 1u : 0u),
        r.local_ref_ok ? (r.trace_mismatch ? "trace_contract_mismatch" : "local_ref_aligned") : "dut_bug_candidate");
    return r;
}

static uint32_t ref_xpred_bit_to_word_bits(const aecct_ref::bit1_t& bit) {
    return (bit.to_uint() != 0u) ? f32_to_bits(1.0f) : f32_to_bits(0.0f);
}

static RefModelStageCompareResult run_one_ref_model_stage_probe(
    Io8Top& io,
    uint32_t sample_idx,
    uint32_t focused_idx
) {
    RefModelStageCompareResult r;
    r.sample_idx = sample_idx;
    r.focused_idx = focused_idx;
    r.st_exact = true;
    r.logit_exact = true;
    r.xpred_exact = true;
    r.all_exact = true;
    r.st_first_mismatch_idx = 0u;
    r.st_dut_bits = 0u;
    r.st_ref_bits = 0u;
    r.logit_first_mismatch_idx = 0u;
    r.logit_dut_bits = 0u;
    r.logit_ref_bits = 0u;
    r.xpred_first_mismatch_idx = 0u;
    r.xpred_dut_bits = 0u;
    r.xpred_ref_bits = 0u;
    r.boundary_bucket = 4u;

    std::vector<uint32_t> infer_words;
    std::vector<uint32_t> trace_logits_words;
    std::vector<uint32_t> trace_xpred_words;
    std::vector<uint8_t> out_bytes;
    std::vector<uint32_t> got_words;
    build_trace_infer_words(sample_idx, infer_words);
    build_trace_logits_words(sample_idx, trace_logits_words);
    build_trace_xpred_words(sample_idx, trace_xpred_words);

    if (focused_idx >= (uint32_t)EXP_LEN_OUT_XPRED_WORDS ||
        focused_idx >= (uint32_t)EXP_LEN_OUT_LOGITS_WORDS ||
        focused_idx >= (uint32_t)N_NODES ||
        focused_idx >= (uint32_t)infer_words.size()) {
        fail("ref model probe focused idx out of range");
    }

    send_cmd(io, (uint8_t)aecct::OP_INFER);
    expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_INFER, "infer_begin_ref_model_probe");
    REF_MODEL_PROBE_INGEST_LOOP: for (uint32_t i = 0u; i < (uint32_t)infer_words.size(); ++i) {
        push_u32_le(io, infer_words[i]);
        top_tick(io);
        if (i + 1u < (uint32_t)infer_words.size()) {
            expect_no_rsp(io, "infer_ingest_ref_model_probe");
        } else {
            expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_INFER, "infer_done_ref_model_probe");
        }
    }

    const uint32_t expected_bytes_len = (uint32_t)EXP_LEN_OUT_XPRED_WORDS * 4u;
    collect_out_bytes(io, expected_bytes_len, out_bytes);
    bytes_to_words_le(out_bytes, got_words);
    if ((uint32_t)got_words.size() != (uint32_t)EXP_LEN_OUT_XPRED_WORDS) {
        fail("ref model probe output words mismatch EXP_LEN_OUT_XPRED_WORDS");
    }

    std::vector<double> ref_input_fp32((uint32_t)EXP_LEN_INFER_IN_WORDS, 0.0);
    REF_INPUT_CONVERT_LOOP: for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_INFER_IN_WORDS; ++i) {
        ref_input_fp32[i] = (double)bits_to_f32(infer_words[i]);
    }
    std::vector<double> ref_logits((uint32_t)EXP_LEN_OUT_LOGITS_WORDS, 0.0);
    std::vector<aecct_ref::bit1_t> ref_xpred((uint32_t)EXP_LEN_OUT_XPRED_WORDS);
    std::vector<double> ref_st((uint32_t)N_NODES, 0.0);

    aecct_ref::RefModel ref_model;
    aecct_ref::RefModelIO ref_io;
    ref_io.input_y = 0;
    ref_io.input_y_fp32 = ref_input_fp32.data();
    ref_io.out_logits = ref_logits.data();
    ref_io.out_x_pred = ref_xpred.data();
    ref_io.out_finalhead_s_t = ref_st.data();
    ref_io.B = 1;
    ref_io.N = (int)EXP_LEN_OUT_XPRED_WORDS;
    ref_model.infer_step0(ref_io);

    const aecct::u32_t* sram = aecct::top_sram();
    const uint32_t final_scalar_base = (uint32_t)sram_map::SCR_FINAL_SCALAR_BASE_W;
    const uint32_t logits_base = (uint32_t)aecct::top_peek_infer_logits_base_word().to_uint();

    REF_ST_COMPARE_LOOP: for (uint32_t t = 0u; t < (uint32_t)N_NODES; ++t) {
        const uint32_t dut_bits = (uint32_t)sram[final_scalar_base + t].to_uint();
        const uint32_t ref_bits = f32_to_bits((float)ref_st[t]);
        if (dut_bits != ref_bits) {
            r.st_exact = false;
            r.st_first_mismatch_idx = t;
            r.st_dut_bits = dut_bits;
            r.st_ref_bits = ref_bits;
            break;
        }
    }
    if (r.st_exact) {
        r.st_dut_bits = (uint32_t)sram[final_scalar_base + focused_idx].to_uint();
        r.st_ref_bits = f32_to_bits((float)ref_st[focused_idx]);
    }

    REF_LOGIT_COMPARE_LOOP: for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_OUT_LOGITS_WORDS; ++i) {
        const uint32_t dut_bits = (uint32_t)sram[logits_base + i].to_uint();
        const uint32_t ref_bits = f32_to_bits((float)ref_logits[i]);
        if (dut_bits != ref_bits) {
            r.logit_exact = false;
            r.logit_first_mismatch_idx = i;
            r.logit_dut_bits = dut_bits;
            r.logit_ref_bits = ref_bits;
            break;
        }
    }
    if (r.logit_exact) {
        r.logit_dut_bits = (uint32_t)sram[logits_base + focused_idx].to_uint();
        r.logit_ref_bits = f32_to_bits((float)ref_logits[focused_idx]);
    }

    REF_XPRED_COMPARE_LOOP: for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_OUT_XPRED_WORDS; ++i) {
        const uint32_t dut_bits = got_words[i];
        const uint32_t ref_bits = ref_xpred_bit_to_word_bits(ref_xpred[i]);
        if (dut_bits != ref_bits) {
            r.xpred_exact = false;
            r.xpred_first_mismatch_idx = i;
            r.xpred_dut_bits = dut_bits;
            r.xpred_ref_bits = ref_bits;
            break;
        }
    }
    if (r.xpred_exact) {
        r.xpred_dut_bits = got_words[focused_idx];
        r.xpred_ref_bits = ref_xpred_bit_to_word_bits(ref_xpred[focused_idx]);
    }

    r.all_exact = r.st_exact && r.logit_exact && r.xpred_exact;
    if (!r.st_exact) {
        r.boundary_bucket = 1u;
    } else if (!r.logit_exact) {
        r.boundary_bucket = 2u;
    } else if (!r.xpred_exact) {
        r.boundary_bucket = 3u;
    } else {
        r.boundary_bucket = 4u;
    }

    const uint32_t focused_dut_st = (uint32_t)sram[final_scalar_base + focused_idx].to_uint();
    const uint32_t focused_ref_st = f32_to_bits((float)ref_st[focused_idx]);
    const uint32_t focused_dut_logit = (uint32_t)sram[logits_base + focused_idx].to_uint();
    const uint32_t focused_ref_logit = f32_to_bits((float)ref_logits[focused_idx]);
    const uint32_t focused_trace_logit = trace_logits_words[focused_idx];
    const uint32_t focused_dut_xpred = got_words[focused_idx];
    const uint32_t focused_ref_xpred = ref_xpred_bit_to_word_bits(ref_xpred[focused_idx]);
    const uint32_t focused_trace_xpred = trace_xpred_words[focused_idx];
    std::printf(
        "[backup_io8][ref_model_probe] sample=%u idx=%u dut_st=0x%08X ref_model_st=0x%08X dut_logit=0x%08X ref_model_logit=0x%08X trace_logit=0x%08X dut_xpred=0x%08X ref_model_xpred=0x%08X trace_xpred=0x%08X\n",
        (unsigned)sample_idx,
        (unsigned)focused_idx,
        (unsigned)focused_dut_st,
        (unsigned)focused_ref_st,
        (unsigned)focused_dut_logit,
        (unsigned)focused_ref_logit,
        (unsigned)focused_trace_logit,
        (unsigned)focused_dut_xpred,
        (unsigned)focused_ref_xpred,
        (unsigned)focused_trace_xpred);
    std::printf(
        "[backup_io8][ref_model_probe] sample=%u st_exact=%u logit_exact=%u xpred_exact=%u first_st_mismatch_idx=%u first_logit_mismatch_idx=%u first_xpred_mismatch_idx=%u boundary_class=%u\n",
        (unsigned)sample_idx,
        (unsigned)(r.st_exact ? 1u : 0u),
        (unsigned)(r.logit_exact ? 1u : 0u),
        (unsigned)(r.xpred_exact ? 1u : 0u),
        (unsigned)r.st_first_mismatch_idx,
        (unsigned)r.logit_first_mismatch_idx,
        (unsigned)r.xpred_first_mismatch_idx,
        (unsigned)r.boundary_bucket);
    if (!r.all_exact) {
        if (r.boundary_bucket == 1u) {
            std::printf(
                "[backup_io8][ref_model_probe] first_divergence=final_embedding_or_upstream idx=%u dut=0x%08X ref=0x%08X\n",
                (unsigned)r.st_first_mismatch_idx,
                (unsigned)r.st_dut_bits,
                (unsigned)r.st_ref_bits);
        } else if (r.boundary_bucket == 2u) {
            std::printf(
                "[backup_io8][ref_model_probe] first_divergence=out_fc_consume idx=%u dut=0x%08X ref=0x%08X\n",
                (unsigned)r.logit_first_mismatch_idx,
                (unsigned)r.logit_dut_bits,
                (unsigned)r.logit_ref_bits);
        } else {
            std::printf(
                "[backup_io8][ref_model_probe] first_divergence=xpred_decision idx=%u dut=0x%08X ref=0x%08X\n",
                (unsigned)r.xpred_first_mismatch_idx,
                (unsigned)r.xpred_dut_bits,
                (unsigned)r.xpred_ref_bits);
        }
    } else {
        std::printf(
            "[backup_io8][ref_model_probe] first_divergence=none sample=%u all_stages_exact=1\n",
            (unsigned)sample_idx);
    }

    return r;
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
    ret.focused_idx = kDebugFocusedIdx;
    ret.st31_exact = false;
    ret.logit31_exact = false;
    ret.xpred31_exact = false;
    ret.st_any_mismatch = false;
    ret.st_first_mismatch_idx = 0u;
    ret.dut_st31_bits = 0u;
    ret.ref_st31_bits = 0u;
    ret.dut_logit31_bits = 0u;
    ret.ref_logit31_bits = 0u;
    ret.trace_logit31_bits = 0u;
    ret.dut_xpred31_bits = 0u;
    ret.ref_xpred31_bits = 0u;
    ret.y31_bits = 0u;
    ret.boundary_bucket = 4u;

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

    const uint32_t idx31 = ret.focused_idx;
    const bool idx31_in_range =
        (idx31 < (uint32_t)EXP_LEN_OUT_XPRED_WORDS) &&
        (idx31 < (uint32_t)EXP_LEN_OUT_LOGITS_WORDS) &&
        (idx31 < (uint32_t)got_words.size()) &&
        (idx31 < (uint32_t)infer_words.size()) &&
        (idx31 < (uint32_t)expected_xpred_words.size()) &&
        (idx31 < (uint32_t)expected_logits_words.size()) &&
        (idx31 < (uint32_t)N_NODES);
    if (!idx31_in_range) {
        fail("debug focused idx out of range");
    }

    const aecct::u32_t* sram = aecct::top_sram();
    const uint32_t x_end_base = (uint32_t)aecct::top_peek_infer_final_x_base_word().to_uint();
    const uint32_t logits_base = (uint32_t)aecct::top_peek_infer_logits_base_word().to_uint();
    const uint32_t xpred_base = (uint32_t)aecct::top_peek_infer_xpred_base_word().to_uint();
    const uint32_t final_scalar_base = (uint32_t)sram_map::SCR_FINAL_SCALAR_BASE_W;
    const aecct::HeadParamBase hp = aecct::make_head_param_base(aecct::top_peek_w_base_word());
    const uint32_t ffn1_w_base = (uint32_t)hp.ffn1_w_base_word.to_uint();
    const uint32_t ffn1_b_base = (uint32_t)hp.ffn1_b_base_word.to_uint();
    const uint32_t out_fc_w_base = (uint32_t)hp.out_fc_w_base_word.to_uint();
    const uint32_t out_fc_b_base = (uint32_t)hp.out_fc_b_base_word.to_uint();
    const uint32_t token_count = (uint32_t)N_NODES;
    const uint32_t d_model = (uint32_t)D_MODEL;

    std::vector<uint32_t> dut_st_words(token_count, 0u);
    std::vector<uint32_t> ref_st_words(token_count, 0u);
    REF_ST_BUILD_LOOP: for (uint32_t t = 0u; t < token_count; ++t) {
        const uint32_t x_row_base = x_end_base + t * d_model;
        float acc = bits_to_f32((uint32_t)sram[ffn1_b_base + 0u].to_uint());
        REF_ST_DOT_LOOP: for (uint32_t i = 0u; i < d_model; ++i) {
            const float xv = bits_to_f32((uint32_t)sram[x_row_base + i].to_uint());
            const float wv = bits_to_f32((uint32_t)sram[ffn1_w_base + i].to_uint());
            acc += (xv * wv);
        }
        ref_st_words[t] = f32_to_bits(acc);
        dut_st_words[t] = (uint32_t)sram[final_scalar_base + t].to_uint();
        if (!ret.st_any_mismatch && dut_st_words[t] != ref_st_words[t]) {
            ret.st_any_mismatch = true;
            ret.st_first_mismatch_idx = t;
        }
    }

    ret.dut_st31_bits = dut_st_words[idx31];
    ret.ref_st31_bits = ref_st_words[idx31];
    ret.st31_exact = (ret.dut_st31_bits == ret.ref_st31_bits);

    float ref_logit31 = bits_to_f32((uint32_t)sram[out_fc_b_base + idx31].to_uint());
    REF_LOGIT31_ACC_LOOP: for (uint32_t t = 0u; t < token_count; ++t) {
        const float stv = bits_to_f32(ref_st_words[t]);
        const float wv = bits_to_f32((uint32_t)sram[out_fc_w_base + idx31 * token_count + t].to_uint());
        ref_logit31 += (stv * wv);
    }

    ret.ref_logit31_bits = f32_to_bits(ref_logit31);
    ret.dut_logit31_bits = (uint32_t)sram[logits_base + idx31].to_uint();
    ret.trace_logit31_bits = expected_logits_words[idx31];
    ret.logit31_exact = (ret.dut_logit31_bits == ret.ref_logit31_bits);

    ret.ref_xpred31_bits = expected_xpred_words[idx31];
    ret.dut_xpred31_bits = got_words[idx31];
    const uint32_t dut_xpred31_sram_bits = (uint32_t)sram[xpred_base + idx31].to_uint();
    ret.xpred31_exact = (ret.dut_xpred31_bits == ret.ref_xpred31_bits);
    ret.y31_bits = infer_words[idx31];

    const float y31 = bits_to_f32(ret.y31_bits);
    const float dut_logit31 = bits_to_f32(ret.dut_logit31_bits);
    const float trace_logit31 = bits_to_f32(ret.trace_logit31_bits);
    const float dut_st31 = bits_to_f32(ret.dut_st31_bits);
    const float ref_st31 = bits_to_f32(ret.ref_st31_bits);
    const float sign_y31 = (y31 > 0.0f) ? 1.0f : ((y31 < 0.0f) ? -1.0f : 0.0f);
    const uint32_t ref_rule_xpred31_bits = f32_to_bits(((dut_logit31 * sign_y31) < 0.0f) ? 1.0f : 0.0f);
    const uint32_t dut_rule_xpred31_bits = f32_to_bits(((dut_logit31 * y31) < 0.0f) ? 1.0f : 0.0f);

    if (ret.st_any_mismatch) {
        std::printf(
            "[backup_io8][debug_st] sample=%u st_exact=FAIL first_mismatch_t=%u dut=0x%08X ref=0x%08X\n",
            (unsigned)pick.sample_id,
            (unsigned)ret.st_first_mismatch_idx,
            (unsigned)dut_st_words[ret.st_first_mismatch_idx],
            (unsigned)ref_st_words[ret.st_first_mismatch_idx]);
    } else {
        std::printf(
            "[backup_io8][debug_st] sample=%u st_exact=PASS hash_dut=0x%08X hash_ref=0x%08X\n",
            (unsigned)pick.sample_id,
            (unsigned)fnv1a_u32_words(dut_st_words),
            (unsigned)fnv1a_u32_words(ref_st_words));
    }
    std::printf(
        "[backup_io8][debug_st31] sample=%u idx=%u dut=0x%08X(%.9g) ref=0x%08X(%.9g) exact=%u\n",
        (unsigned)pick.sample_id,
        (unsigned)idx31,
        (unsigned)ret.dut_st31_bits,
        (double)dut_st31,
        (unsigned)ret.ref_st31_bits,
        (double)ref_st31,
        (unsigned)(ret.st31_exact ? 1u : 0u));

    const uint32_t st_win_begin = (idx31 >= 2u) ? (idx31 - 2u) : 0u;
    const uint32_t st_win_end = ((idx31 + 2u) < token_count) ? (idx31 + 2u) : (token_count - 1u);
    DEBUG_ST_WINDOW_LOOP: for (uint32_t t = st_win_begin; t <= st_win_end; ++t) {
        std::printf(
            "[backup_io8][debug_st_window] sample=%u t=%u dut=0x%08X ref=0x%08X\n",
            (unsigned)pick.sample_id,
            (unsigned)t,
            (unsigned)dut_st_words[t],
            (unsigned)ref_st_words[t]);
    }

    std::printf(
        "[backup_io8][debug_logit31] sample=%u idx=%u dut=0x%08X(%.9g) ref_formula=0x%08X(%.9g) ref_trace=0x%08X(%.9g) exact_formula=%u\n",
        (unsigned)pick.sample_id,
        (unsigned)idx31,
        (unsigned)ret.dut_logit31_bits,
        (double)dut_logit31,
        (unsigned)ret.ref_logit31_bits,
        (double)ref_logit31,
        (unsigned)ret.trace_logit31_bits,
        (double)trace_logit31,
        (unsigned)(ret.logit31_exact ? 1u : 0u));

    std::printf(
        "[backup_io8][debug_xpred31] sample=%u idx=%u dut_stream=0x%08X dut_sram=0x%08X ref_trace=0x%08X exact=%u\n",
        (unsigned)pick.sample_id,
        (unsigned)idx31,
        (unsigned)ret.dut_xpred31_bits,
        (unsigned)dut_xpred31_sram_bits,
        (unsigned)ret.ref_xpred31_bits,
        (unsigned)(ret.xpred31_exact ? 1u : 0u));

    std::printf(
        "[backup_io8][debug_decision31] sample=%u idx=%u y_bits_raw=0x%08X y=%.9g sign_y=%.1f dut_rule_xpred=0x%08X ref_rule_xpred=0x%08X\n",
        (unsigned)pick.sample_id,
        (unsigned)idx31,
        (unsigned)ret.y31_bits,
        (double)y31,
        (double)sign_y31,
        (unsigned)dut_rule_xpred31_bits,
        (unsigned)ref_rule_xpred31_bits);

    if (!ret.st31_exact || ret.st_any_mismatch) {
        ret.boundary_bucket = 1u; // A
        std::printf(
            "[backup_io8][debug_focus_boundary] sample=%u idx=%u class=A_s_t_consume\n",
            (unsigned)pick.sample_id,
            (unsigned)idx31);
    } else if (!ret.logit31_exact) {
        ret.boundary_bucket = 2u; // B
        std::printf(
            "[backup_io8][debug_focus_boundary] sample=%u idx=%u class=B_out_fc_consume\n",
            (unsigned)pick.sample_id,
            (unsigned)idx31);
    } else if (!ret.xpred31_exact) {
        ret.boundary_bucket = 3u; // C
        std::printf(
            "[backup_io8][debug_focus_boundary] sample=%u idx=%u class=C_xpred_decision\n",
            (unsigned)pick.sample_id,
            (unsigned)idx31);
    } else {
        ret.boundary_bucket = 4u; // D
        std::printf(
            "[backup_io8][debug_focus_boundary] sample=%u idx=%u class=D_no_local_divergence\n",
            (unsigned)pick.sample_id,
            (unsigned)idx31);
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

    Io8Top io_debug;
    run_setup_cfg_loadw(io_debug, param_words, "debug");

    const uint32_t infer_input_base_dbg = (uint32_t)aecct::top_peek_infer_input_base_word().to_uint();
    const uint32_t infer_logits_base_dbg = (uint32_t)aecct::top_peek_infer_logits_base_word().to_uint();
    const uint32_t final_logits_base_dbg = (uint32_t)aecct::FINAL_LOGITS_BASE_WORD;
    const bool y_logits_alias = (infer_input_base_dbg == final_logits_base_dbg);
    std::printf(
        "[backup_io8][debug_alias] infer_input_base_word=0x%08X infer_logits_base_word=0x%08X FINAL_LOGITS_BASE_WORD=0x%08X alias=%u\n",
        (unsigned)infer_input_base_dbg,
        (unsigned)infer_logits_base_dbg,
        (unsigned)final_logits_base_dbg,
        (unsigned)(y_logits_alias ? 1u : 0u));

    const uint32_t payload_words_to_check = clip_words_to_check(param_words, kDebugPayloadReadbackWords);
    if (payload_words_to_check == 0u) {
        fail("payload debug words_to_check is zero");
    }

    std::vector<uint32_t> expected_payload_prefix(
        param_words.begin(),
        param_words.begin() + (size_t)payload_words_to_check);
    std::vector<uint32_t> readmem_payload_prefix;
    std::vector<uint32_t> direct_payload_prefix;
    read_mem_words(io_debug, (uint32_t)sram_map::PARAM_BASE_DEFAULT, payload_words_to_check, readmem_payload_prefix);
    read_sram_words_direct((uint32_t)sram_map::PARAM_BASE_DEFAULT, payload_words_to_check, direct_payload_prefix);

    uint32_t payload_readmem_bad_idx = 0u;
    uint32_t payload_readmem_bad_got = 0u;
    uint32_t payload_readmem_bad_exp = 0u;
    const bool payload_readmem_ok = compare_word_vectors_exact(
        expected_payload_prefix,
        readmem_payload_prefix,
        payload_readmem_bad_idx,
        payload_readmem_bad_got,
        payload_readmem_bad_exp);

    uint32_t payload_direct_bad_idx = 0u;
    uint32_t payload_direct_bad_got = 0u;
    uint32_t payload_direct_bad_exp = 0u;
    const bool payload_direct_ok = compare_word_vectors_exact(
        expected_payload_prefix,
        direct_payload_prefix,
        payload_direct_bad_idx,
        payload_direct_bad_got,
        payload_direct_bad_exp);

    print_expected_payload_prefix_with_source(expected_payload_prefix, kDebugPayloadWindowWords);
    print_payload_compare_window(
        "debug_payload_readmem",
        expected_payload_prefix,
        readmem_payload_prefix,
        kDebugPayloadWindowWords);
    print_payload_compare_window(
        "debug_payload_direct",
        expected_payload_prefix,
        direct_payload_prefix,
        kDebugPayloadWindowWords);

    const uint32_t top_w_base = (uint32_t)aecct::top_peek_w_base_word().to_uint();
    const uint32_t commit_valid = aecct::top_peek_accepted_commit_record_valid() ? 1u : 0u;
    const uint32_t commit_owner = (uint32_t)aecct::top_peek_accepted_commit_owner_opcode().to_uint();
    const uint32_t commit_base = (uint32_t)aecct::top_peek_accepted_commit_base_word().to_uint();
    const uint32_t commit_len_expected = (uint32_t)aecct::top_peek_accepted_commit_len_words_expected().to_uint();
    const uint32_t commit_len_valid = (uint32_t)aecct::top_peek_accepted_commit_len_words_valid().to_uint();
    std::printf(
        "[backup_io8][debug_payload_commit] top_w_base=0x%08X accepted_valid=%u owner_opcode=0x%02X base=0x%08X len_expected=%u len_valid=%u\n",
        (unsigned)top_w_base,
        (unsigned)commit_valid,
        (unsigned)commit_owner,
        (unsigned)commit_base,
        (unsigned)commit_len_expected,
        (unsigned)commit_len_valid);

    uint32_t idx0_param_id = 0u;
    uint32_t idx0_local_idx = 0u;
    uint32_t idx0_source_word = 0u;
    const bool idx0_mapped = map_param_word_index(0u, idx0_param_id, idx0_local_idx);
    const bool idx0_source_known = idx0_mapped && try_get_source_word_for_expected(idx0_param_id, idx0_local_idx, idx0_source_word);
    const bool idx0_source_match = idx0_source_known && (idx0_source_word == expected_payload_prefix[0]);
    std::printf(
        "[backup_io8][debug_payload_expected_idx0] mapped=%u param_id=%u key=%s expected=0x%08X source_known=%u source=0x%08X source_match=%u\n",
        (unsigned)(idx0_mapped ? 1u : 0u),
        (unsigned)idx0_param_id,
        idx0_mapped ? kParamKey[idx0_param_id] : "unmapped",
        (unsigned)expected_payload_prefix[0],
        (unsigned)(idx0_source_known ? 1u : 0u),
        (unsigned)idx0_source_word,
        (unsigned)(idx0_source_match ? 1u : 0u));

    if (payload_readmem_ok) {
        std::printf(
            "[backup_io8][debug_payload] readback_prefix_words=%u exact=PASS hash=0x%08X\n",
            (unsigned)payload_words_to_check,
            (unsigned)fnv1a_u32_words(readmem_payload_prefix));
    } else {
        std::printf(
            "[backup_io8][debug_payload] readback_prefix_words=%u exact=FAIL first_mismatch_idx=%u expected=0x%08X actual=0x%08X\n",
            (unsigned)payload_words_to_check,
            (unsigned)payload_readmem_bad_idx,
            (unsigned)payload_readmem_bad_exp,
            (unsigned)payload_readmem_bad_got);
    }
    if (payload_direct_ok) {
        std::printf(
            "[backup_io8][debug_payload_direct] direct_prefix_words=%u exact=PASS hash=0x%08X\n",
            (unsigned)payload_words_to_check,
            (unsigned)fnv1a_u32_words(direct_payload_prefix));
    } else {
        std::printf(
            "[backup_io8][debug_payload_direct] direct_prefix_words=%u exact=FAIL first_mismatch_idx=%u expected=0x%08X actual=0x%08X\n",
            (unsigned)payload_words_to_check,
            (unsigned)payload_direct_bad_idx,
            (unsigned)payload_direct_bad_exp,
            (unsigned)payload_direct_bad_got);
    }

    uint32_t readmem_hit_idx = 0u;
    if (!payload_readmem_ok && find_first_occurrence(param_words, payload_readmem_bad_got, readmem_hit_idx)) {
        uint32_t hit_param_id = 0u;
        uint32_t hit_local_idx = 0u;
        if (map_param_word_index(readmem_hit_idx, hit_param_id, hit_local_idx)) {
            std::printf(
                "[backup_io8][debug_payload_hint] readmem_actual_first_mismatch_word_hit expected_idx=%u param_id=%u key=%s local_word=%u\n",
                (unsigned)readmem_hit_idx,
                (unsigned)hit_param_id,
                kParamKey[hit_param_id],
                (unsigned)hit_local_idx);
        }
    }

    PayloadResponsibilityBoundary payload_boundary = PAYLOAD_BOUNDARY_UNKNOWN;
    if (!idx0_source_match) {
        payload_boundary = PAYLOAD_BOUNDARY_A_EXPECTED_STREAM;
    } else if (!payload_direct_ok) {
        payload_boundary = PAYLOAD_BOUNDARY_C_LOADW_WRITE;
    } else if (!payload_readmem_ok) {
        payload_boundary = PAYLOAD_BOUNDARY_D_READ_MEM;
    }
    if (payload_boundary == PAYLOAD_BOUNDARY_A_EXPECTED_STREAM) {
        std::printf("[backup_io8][debug_payload_boundary] class=A_weights_streamer_or_expected_stream\n");
    } else if (payload_boundary == PAYLOAD_BOUNDARY_C_LOADW_WRITE) {
        std::printf("[backup_io8][debug_payload_boundary] class=C_loadw_write_or_base_order (direct_sram mismatch)\n");
    } else if (payload_boundary == PAYLOAD_BOUNDARY_D_READ_MEM) {
        std::printf("[backup_io8][debug_payload_boundary] class=D_read_mem_window_or_addr (direct_sram match, read_mem mismatch)\n");
    } else {
        std::printf("[backup_io8][debug_payload_boundary] class=payload_path_aligned_no_mismatch\n");
    }

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

    bool mismatch_found = false;
    uint32_t mismatch_sample = 0u;
    DebugCompareResult mismatch_ret;
    DEBUG_SAMPLE_COMPARE_LOOP: for (uint32_t i = 0u; i < (uint32_t)xpred_one_samples.size(); ++i) {
        const XpredOneSample& pick = xpred_one_samples[i];
        DebugCompareResult r = run_one_xpred_one_debug_sample(io_debug, pick);
        if (!r.exact) {
            mismatch_found = true;
            mismatch_sample = pick.sample_id;
            mismatch_ret = r;
            break;
        }
    }
    if (mismatch_found) {
        std::printf(
            "[backup_io8][debug_focus_summary] sample=%u idx=%u st_exact=%u logit_exact=%u xpred_exact=%u boundary_class=%u\n",
            (unsigned)mismatch_sample,
            (unsigned)mismatch_ret.focused_idx,
            (unsigned)(mismatch_ret.st31_exact ? 1u : 0u),
            (unsigned)(mismatch_ret.logit31_exact ? 1u : 0u),
            (unsigned)(mismatch_ret.xpred31_exact ? 1u : 0u),
            (unsigned)mismatch_ret.boundary_bucket);
    }

    const LocalContractResult triage_s5 =
        run_one_local_contract_focus_sample(io_debug, kDebugPreferredSampleId, kDebugFocusedIdx);
    const LocalContractResult triage_s0 =
        run_one_local_contract_focus_sample(io_debug, kContractSample0, kContractIdx0);
    const RefModelStageCompareResult ref_probe_s5 =
        run_one_ref_model_stage_probe(io_debug, kDebugPreferredSampleId, kDebugFocusedIdx);
    RefModelStageCompareResult ref_probe = ref_probe_s5;
    if (ref_probe_s5.all_exact) {
        ref_probe = run_one_ref_model_stage_probe(io_debug, kContractSample0, kContractIdx0);
    }
    const bool local_ref_gate_ok = triage_s5.local_ref_ok && triage_s0.local_ref_ok;
    std::printf(
        "[backup_io8][contract_gate] local_ref_gate_ok=%u sample5_local_ref_ok=%u sample0_local_ref_ok=%u sample0_trace_mismatch=%u\n",
        (unsigned)(local_ref_gate_ok ? 1u : 0u),
        (unsigned)(triage_s5.local_ref_ok ? 1u : 0u),
        (unsigned)(triage_s0.local_ref_ok ? 1u : 0u),
        (unsigned)(triage_s0.trace_mismatch ? 1u : 0u));
    std::printf(
        "[backup_io8][ref_model_gate] sample=%u all_exact=%u boundary_class=%u\n",
        (unsigned)ref_probe.sample_idx,
        (unsigned)(ref_probe.all_exact ? 1u : 0u),
        (unsigned)ref_probe.boundary_bucket);

    std::vector<uint32_t> readmem_payload_prefix_post;
    std::vector<uint32_t> direct_payload_prefix_post;
    read_mem_words(io_debug, (uint32_t)sram_map::PARAM_BASE_DEFAULT, payload_words_to_check, readmem_payload_prefix_post);
    read_sram_words_direct((uint32_t)sram_map::PARAM_BASE_DEFAULT, payload_words_to_check, direct_payload_prefix_post);

    uint32_t payload_readmem_bad_idx_post = 0u;
    uint32_t payload_readmem_bad_got_post = 0u;
    uint32_t payload_readmem_bad_exp_post = 0u;
    const bool payload_readmem_ok_post = compare_word_vectors_exact(
        expected_payload_prefix,
        readmem_payload_prefix_post,
        payload_readmem_bad_idx_post,
        payload_readmem_bad_got_post,
        payload_readmem_bad_exp_post);
    uint32_t payload_direct_bad_idx_post = 0u;
    uint32_t payload_direct_bad_got_post = 0u;
    uint32_t payload_direct_bad_exp_post = 0u;
    const bool payload_direct_ok_post = compare_word_vectors_exact(
        expected_payload_prefix,
        direct_payload_prefix_post,
        payload_direct_bad_idx_post,
        payload_direct_bad_got_post,
        payload_direct_bad_exp_post);

    std::printf(
        "[backup_io8][debug_payload_post_infer] readmem_exact=%u direct_exact=%u first_readmem_mismatch_idx=%u first_direct_mismatch_idx=%u\n",
        (unsigned)(payload_readmem_ok_post ? 1u : 0u),
        (unsigned)(payload_direct_ok_post ? 1u : 0u),
        (unsigned)payload_readmem_bad_idx_post,
        (unsigned)payload_direct_bad_idx_post);
    if (!payload_readmem_ok_post) {
        std::printf(
            "[backup_io8][debug_payload_post_infer] readmem_first_mismatch expected=0x%08X actual=0x%08X\n",
            (unsigned)payload_readmem_bad_exp_post,
            (unsigned)payload_readmem_bad_got_post);
    }
    if (!payload_direct_ok_post) {
        std::printf(
            "[backup_io8][debug_payload_post_infer] direct_first_mismatch expected=0x%08X actual=0x%08X\n",
            (unsigned)payload_direct_bad_exp_post,
            (unsigned)payload_direct_bad_got_post);
    }

    DebugBoundary boundary = DEBUG_BOUNDARY_NONE;
    if (!payload_readmem_ok || !payload_direct_ok) {
        boundary = DEBUG_BOUNDARY_PAYLOAD_LOADW;
    } else if (!payload_readmem_ok_post || !payload_direct_ok_post) {
        boundary = DEBUG_BOUNDARY_DUT_ALGO;
    } else if (mismatch_found && !mismatch_ret.byte_mismatch_found) {
        boundary = DEBUG_BOUNDARY_OUTPUT_PACKING;
    } else if (mismatch_found) {
        boundary = DEBUG_BOUNDARY_DUT_ALGO;
    }

    if (boundary == DEBUG_BOUNDARY_PAYLOAD_LOADW) {
        std::printf(
            "[backup_io8][debug_boundary] earliest_boundary=payload_or_loadw payload_boundary_class=%u first_readmem_mismatch_idx=%u first_direct_mismatch_idx=%u\n",
            (unsigned)payload_boundary,
            (unsigned)payload_readmem_bad_idx,
            (unsigned)payload_direct_bad_idx);
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

    Io8Top io_trace;
    run_setup_cfg_loadw(io_trace, param_words, "trace");
    uint32_t trace_mismatch_count = 0u;
    uint32_t trace_first_bad_sample = 0u;
    uint32_t trace_first_bad_idx = 0u;
    uint32_t trace_first_bad_got = 0u;
    uint32_t trace_first_bad_exp = 0u;
    bool trace_first_bad_valid = false;
    TRACE_PATTERN_LOOP: for (uint32_t pattern_idx = 0u; pattern_idx < kTracePatternCount; ++pattern_idx) {
        const uint32_t sample_idx = kTraceSampleIds[pattern_idx];
        uint32_t bad_idx = 0u;
        uint32_t bad_got = 0u;
        uint32_t bad_exp = 0u;
        const bool exact = run_one_trace_sample_and_compare(io_trace, sample_idx, bad_idx, bad_got, bad_exp);
        if (!exact) {
            ++trace_mismatch_count;
            if (!trace_first_bad_valid) {
                trace_first_bad_valid = true;
                trace_first_bad_sample = sample_idx;
                trace_first_bad_idx = bad_idx;
                trace_first_bad_got = bad_got;
                trace_first_bad_exp = bad_exp;
            }
        }
    }
    if (trace_first_bad_valid) {
        std::printf(
            "DIAG: tb_backup_io8_loadw_infer_trace_compare mismatches=%u patterns=%u words_per_pattern=%u first_sample=%u first_idx=%u got=0x%08X expect=0x%08X\n",
            (unsigned)trace_mismatch_count,
            (unsigned)kTracePatternCount,
            (unsigned)EXP_LEN_OUT_XPRED_WORDS,
            (unsigned)trace_first_bad_sample,
            (unsigned)trace_first_bad_idx,
            (unsigned)trace_first_bad_got,
            (unsigned)trace_first_bad_exp);
    } else {
        std::printf(
            "DIAG: tb_backup_io8_loadw_infer_trace_compare mismatches=0 patterns=%u words_per_pattern=%u\n",
            (unsigned)kTracePatternCount,
            (unsigned)EXP_LEN_OUT_XPRED_WORDS);
    }

    const bool payload_hard_ok =
        payload_readmem_ok && payload_direct_ok && payload_readmem_ok_post && payload_direct_ok_post;
    if (!payload_hard_ok) {
        std::printf(
            "[backup_io8][FAIL] payload_hard_gate_fail pre_readmem=%u pre_direct=%u post_readmem=%u post_direct=%u\n",
            (unsigned)(payload_readmem_ok ? 1u : 0u),
            (unsigned)(payload_direct_ok ? 1u : 0u),
            (unsigned)(payload_readmem_ok_post ? 1u : 0u),
            (unsigned)(payload_direct_ok_post ? 1u : 0u));
        return 1;
    }
    if (!local_ref_gate_ok) {
        std::printf("[backup_io8][FAIL] local_ref_hard_gate_fail sample5_or_sample0_not_aligned\n");
        return 1;
    }

    std::printf("PASS: tb_backup_io8_loadw_infer_local_ref_compare\n");
    std::printf("PASS: tb_backup_io8_loadw_infer_ref_model_stage_probe\n");
    std::printf("PASS: tb_backup_io8_loadw_infer_xpred1_debug_bridge\n");
    std::printf("PASS: tb_backup_io8_loadw_infer_smoke\n");
    return 0;
}
