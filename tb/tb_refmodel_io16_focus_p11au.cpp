#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "AecctProtocol.h"
#include "Top.h"
#include "input_y_step0.h"
#include "RefModel.h"
#include "tb/weights_streamer.h"

namespace {

struct TopIo {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;
};

static const uint32_t kFocusedSampleCount = 8u;
static const uint32_t kFocusedSampleIds[kFocusedSampleCount] = {
    20u, 21u, 22u, 23u, 61u, 62u, 63u, 77u
};

static uint32_t f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = f;
    return cvt.u;
}

static void fail_now(const char* msg) {
    std::printf("[p11au][FAIL] %s\n", msg);
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
        std::printf("[p11au][FAIL] %s missing ctrl_rsp\n", tag);
        std::exit(1);
    }
    const uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
    const uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);
    if (kind != kind_exp || payload != payload_exp) {
        std::printf("[p11au][FAIL] %s rsp kind=%u payload=%u expect_kind=%u expect_payload=%u\n",
                    tag,
                    (unsigned)kind,
                    (unsigned)payload,
                    (unsigned)kind_exp,
                    (unsigned)payload_exp);
        std::exit(1);
    }
}

static void recv_rsp_expect_either(TopIo& io,
                                   uint8_t kind_a,
                                   uint8_t kind_b,
                                   uint8_t payload_exp,
                                   const char* tag) {
    aecct::u16_t w;
    if (!io.ctrl_rsp.nb_read(w)) {
        std::printf("[p11au][FAIL] %s missing ctrl_rsp\n", tag);
        std::exit(1);
    }
    const uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
    const uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);
    if (((kind != kind_a) && (kind != kind_b)) || payload != payload_exp) {
        std::printf("[p11au][FAIL] %s rsp kind=%u payload=%u expect_kind=%u|%u expect_payload=%u\n",
                    tag,
                    (unsigned)kind,
                    (unsigned)payload,
                    (unsigned)kind_a,
                    (unsigned)kind_b,
                    (unsigned)payload_exp);
        std::exit(1);
    }
}

static void recv_no_rsp(TopIo& io, const char* tag) {
    aecct::u16_t w;
    if (io.ctrl_rsp.nb_read(w)) {
        std::printf("[p11au][FAIL] %s unexpected ctrl_rsp kind=%u payload=%u\n",
                    tag,
                    (unsigned)aecct::unpack_ctrl_rsp_kind(w),
                    (unsigned)aecct::unpack_ctrl_rsp_payload(w));
        std::exit(1);
    }
}

static uint32_t recv_data_word(TopIo& io, const char* tag) {
    aecct::u32_t w;
    uint32_t guard = 0u;
    while (!io.data_out.nb_read(w)) {
        tick(io);
        ++guard;
        if (guard > 100000u) {
            std::printf("[p11au][FAIL] %s data_out timeout\n", tag);
            std::exit(1);
        }
    }
    return (uint32_t)w.to_uint();
}

static void recv_data_n(TopIo& io, uint32_t n, std::vector<uint32_t>& out_words, const char* tag) {
    out_words.assign(n, 0u);
    for (uint32_t i = 0u; i < n; ++i) {
        out_words[i] = recv_data_word(io, tag);
    }
}

static void drain_no_data_out(TopIo& io, const char* tag) {
    aecct::u32_t w;
    if (io.data_out.nb_read(w)) {
        std::printf("[p11au][FAIL] %s unexpected data_out word=0x%08X\n", tag, (unsigned)w.to_uint());
        std::exit(1);
    }
}

static void build_cfg_words(uint32_t cfg_words[EXP_LEN_CFG_WORDS]) {
    for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_CFG_WORDS; ++i) {
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

static bool build_param_words_from_repo_reference(std::vector<uint32_t>& param_words,
                                                  std::string& error) {
    error.clear();
    aecct::data_ch_t stream_words;

    for (uint32_t i = 0u; i < (uint32_t)BIAS_COUNT; ++i) {
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

    for (uint32_t i = 0u; i < (uint32_t)WEIGHT_COUNT; ++i) {
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
    while (stream_words.nb_read(w)) {
        param_words.push_back((uint32_t)w.to_uint());
    }
    if ((uint32_t)param_words.size() != (uint32_t)EXP_LEN_PARAM_WORDS) {
        error = "PARAM word count mismatch against EXP_LEN_PARAM_WORDS";
        return false;
    }
    return true;
}

static void do_soft_reset(TopIo& io) {
    send_cmd(io, (uint8_t)aecct::OP_SOFT_RESET);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET, "soft_reset");
}

static void do_cfg_commit_ok(TopIo& io) {
    uint32_t cfg_words[EXP_LEN_CFG_WORDS];
    build_cfg_words(cfg_words);
    send_cmd(io, (uint8_t)aecct::OP_CFG_BEGIN);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN, "cfg_begin");
    for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_CFG_WORDS; ++i) {
        send_data_and_tick(io, cfg_words[i]);
        recv_no_rsp(io, "cfg_ingest");
    }
    send_cmd(io, (uint8_t)aecct::OP_CFG_COMMIT);
    recv_rsp_expect_either(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_CFG_COMMIT, "cfg_commit");
}

static void do_set_w_base(TopIo& io, uint32_t w_base_word) {
    send_u32(io.data_in, w_base_word);
    send_cmd(io, (uint8_t)aecct::OP_SET_W_BASE);
    recv_rsp_expect_either(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_W_BASE, "set_w_base");
}

static void do_load_w_repo_reference(TopIo& io, const std::vector<uint32_t>& param_words) {
    send_cmd(io, (uint8_t)aecct::OP_LOAD_W);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W, "load_w_begin");
    for (uint32_t i = 0u; i < (uint32_t)param_words.size(); ++i) {
        send_data_and_tick(io, param_words[i]);
        if (i + 1u < (uint32_t)param_words.size()) {
            recv_no_rsp(io, "load_w_ingest");
        } else {
            recv_rsp_expect(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W, "load_w_done");
        }
    }
}

static void do_set_outmode(TopIo& io, uint32_t outmode) {
    send_u32(io.data_in, outmode);
    send_cmd(io, (uint8_t)aecct::OP_SET_OUTMODE);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE, "set_outmode");
}

static void build_trace_infer_words(uint32_t sample_idx, std::vector<uint32_t>& infer_words) {
    infer_words.assign((uint32_t)EXP_LEN_INFER_IN_WORDS, 0u);
    const uint32_t base = sample_idx * (uint32_t)EXP_LEN_INFER_IN_WORDS;
    for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_INFER_IN_WORDS; ++i) {
        infer_words[i] = f32_to_bits((float)trace_input_y_step0_tensor[base + i]);
    }
}

static void run_infer_collect_words(TopIo& io,
                                    const std::vector<uint32_t>& infer_words,
                                    std::vector<uint32_t>& out_words,
                                    uint32_t out_word_count) {
    send_cmd(io, (uint8_t)aecct::OP_INFER);
    recv_rsp_expect(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_INFER, "infer_begin");
    for (uint32_t i = 0u; i < (uint32_t)infer_words.size(); ++i) {
        send_data_and_tick(io, infer_words[i]);
        if (i + 1u < (uint32_t)infer_words.size()) {
            recv_no_rsp(io, "infer_ingest");
        } else {
            recv_rsp_expect(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_INFER, "infer_done");
        }
    }
    recv_data_n(io, out_word_count, out_words, "infer_data");
    drain_no_data_out(io, "infer_no_extra");
}

static void do_read_mem_words(TopIo& io,
                              uint32_t addr_word,
                              uint32_t len_words,
                              std::vector<uint32_t>& out_words) {
    send_u32(io.data_in, addr_word);
    send_u32(io.data_in, len_words);
    send_cmd(io, (uint8_t)aecct::OP_READ_MEM);
    recv_data_n(io, len_words, out_words, "read_mem_data");
    recv_rsp_expect(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM, "read_mem_done");
}

static void words_u32_to_words16_le(const std::vector<uint32_t>& words_u32,
                                    std::vector<uint16_t>& words16_out) {
    words16_out.clear();
    words16_out.reserve((std::size_t)words_u32.size() * 2u);
    for (uint32_t i = 0u; i < (uint32_t)words_u32.size(); ++i) {
        const uint32_t w = words_u32[i];
        words16_out.push_back((uint16_t)(w & 0xFFFFu));
        words16_out.push_back((uint16_t)((w >> 16) & 0xFFFFu));
    }
}

static bool compare_u32_exact(const std::vector<uint32_t>& a,
                              const std::vector<uint32_t>& b,
                              uint32_t& bad_idx,
                              uint32_t& bad_got,
                              uint32_t& bad_exp) {
    bad_idx = 0u;
    bad_got = 0u;
    bad_exp = 0u;
    if (a.size() != b.size()) {
        return false;
    }
    for (uint32_t i = 0u; i < (uint32_t)a.size(); ++i) {
        if (a[i] != b[i]) {
            bad_idx = i;
            bad_got = b[i];
            bad_exp = a[i];
            return false;
        }
    }
    return true;
}

static bool compare_u16_exact(const std::vector<uint16_t>& a,
                              const std::vector<uint16_t>& b,
                              uint32_t& bad_idx,
                              uint16_t& bad_got,
                              uint16_t& bad_exp) {
    bad_idx = 0u;
    bad_got = 0u;
    bad_exp = 0u;
    if (a.size() != b.size()) {
        return false;
    }
    for (uint32_t i = 0u; i < (uint32_t)a.size(); ++i) {
        if (a[i] != b[i]) {
            bad_idx = i;
            bad_got = b[i];
            bad_exp = a[i];
            return false;
        }
    }
    return true;
}

static uint32_t ref_xpred_bit_to_word_bits(const aecct_ref::bit1_t& bit) {
    return f32_to_bits(bit.to_uint() ? 1.0f : 0.0f);
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<uint32_t> param_words;
    std::string build_error;
    std::printf("[p11au][debug] build_param begin\n");
    if (!build_param_words_from_repo_reference(param_words, build_error)) {
        std::printf("[p11au][FAIL] build_param_words_from_repo_reference: %s\n", build_error.c_str());
        return 1;
    }

    std::printf("[p11au][debug] build_param done words=%u\n", (unsigned)param_words.size());

    TopIo io;
    std::printf("[p11au][debug] soft_reset begin\n");
    do_soft_reset(io);
    std::printf("[p11au][debug] soft_reset done\n");
    std::printf("[p11au][debug] cfg begin\n");
    do_cfg_commit_ok(io);
    std::printf("[p11au][debug] cfg done\n");
    std::printf("[p11au][debug] set_w_base begin\n");
    do_set_w_base(io, (uint32_t)sram_map::PARAM_BASE_DEFAULT);
    std::printf("[p11au][debug] set_w_base done\n");
    std::printf("[p11au][debug] load_w begin\n");
    do_load_w_repo_reference(io, param_words);
    std::printf("[p11au][debug] load_w done\n");

    aecct_ref::RefModel ref_model;
    uint32_t matched_samples = 0u;
    uint32_t final_scalar_direct_diag_exact_samples = 0u;
    uint32_t final_scalar_readmem_diag_exact_samples = 0u;
    uint32_t logits_diag_exact_samples = 0u;
    bool final_scalar_direct_first_mismatch_valid = false;
    uint32_t final_scalar_direct_first_mismatch_sample = 0u;
    uint32_t final_scalar_direct_first_mismatch_idx = 0u;
    uint16_t final_scalar_direct_first_mismatch_dut = 0u;
    uint16_t final_scalar_direct_first_mismatch_ref = 0u;
    bool final_scalar_readmem_first_mismatch_valid = false;
    uint32_t final_scalar_readmem_first_mismatch_sample = 0u;
    uint32_t final_scalar_readmem_first_mismatch_idx = 0u;
    uint16_t final_scalar_readmem_first_mismatch_dut = 0u;
    uint16_t final_scalar_readmem_first_mismatch_ref = 0u;
    bool logits_diag_first_mismatch_valid = false;
    uint32_t logits_diag_first_mismatch_sample = 0u;
    uint32_t logits_diag_first_mismatch_idx = 0u;
    uint16_t logits_diag_first_mismatch_dut = 0u;
    uint16_t logits_diag_first_mismatch_ref = 0u;

    for (uint32_t si = 0u; si < kFocusedSampleCount; ++si) {
        const uint32_t sample_id = kFocusedSampleIds[si];
        std::printf("[p11au][debug] sample=%u begin\n", (unsigned)sample_id);
        std::vector<uint32_t> infer_words;
        build_trace_infer_words(sample_id, infer_words);

        std::vector<double> ref_input_fp32((uint32_t)EXP_LEN_INFER_IN_WORDS, 0.0);
        for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_INFER_IN_WORDS; ++i) {
            union { uint32_t u; float f; } cvt;
            cvt.u = infer_words[i];
            ref_input_fp32[i] = (double)cvt.f;
        }
        std::vector<double> ref_logits((uint32_t)EXP_LEN_OUT_LOGITS_WORDS, 0.0);
        std::vector<aecct_ref::bit1_t> ref_xpred((uint32_t)EXP_LEN_OUT_XPRED_WORDS);
        std::vector<double> ref_final_s((uint32_t)N_NODES, 0.0);
        aecct_ref::RefModelIO ref_io;
        ref_io.input_y = 0;
        ref_io.input_y_fp32 = ref_input_fp32.data();
        ref_io.out_logits = ref_logits.data();
        ref_io.out_x_pred = ref_xpred.data();
        ref_io.out_finalhead_s_t = ref_final_s.data();
        ref_io.B = 1;
        ref_io.N = (int)EXP_LEN_OUT_XPRED_WORDS;
        aecct_ref::RefStep0Io16Image logits_image;
        if (!ref_model.build_step0_io16_image(ref_io, aecct_ref::RefStep0OutputMode::LOGITS, logits_image)) {
            std::printf("[p11au][FAIL] sample=%u build_step0_io16_image failed\n", (unsigned)sample_id);
            return 1;
        }

        std::vector<uint16_t> ref_final_scalar_words16;
        if (!aecct_ref::RefModel::read_mem_words16(logits_image,
                                                   logits_image.report.final_scalar_base_word16,
                                                   (uint32_t)N_NODES * 2u,
                                                   ref_final_scalar_words16)) {
            std::printf("[p11au][FAIL] sample=%u ref final-scalar readback failed\n", (unsigned)sample_id);
            return 1;
        }

        std::printf("[p11au][debug] sample=%u xpred infer begin\n", (unsigned)sample_id);
        do_set_outmode(io, 0u);
        std::vector<uint32_t> dut_xpred_words;
        run_infer_collect_words(io, infer_words, dut_xpred_words, (uint32_t)EXP_LEN_OUT_XPRED_WORDS);
        std::vector<uint32_t> ref_xpred_words((uint32_t)EXP_LEN_OUT_XPRED_WORDS, 0u);
        for (uint32_t i = 0u; i < (uint32_t)EXP_LEN_OUT_XPRED_WORDS; ++i) {
            ref_xpred_words[i] = ref_xpred_bit_to_word_bits(ref_xpred[i]);
        }
        uint32_t xpred_bad_idx = 0u;
        uint32_t xpred_bad_got = 0u;
        uint32_t xpred_bad_exp = 0u;
        const bool xpred_exact = compare_u32_exact(ref_xpred_words,
                                                   dut_xpred_words,
                                                   xpred_bad_idx,
                                                   xpred_bad_got,
                                                   xpred_bad_exp);
        if (!xpred_exact) {
            std::printf("[p11au][FAIL] sample=%u xpred_exact=0 idx=%u dut=0x%08X ref=0x%08X\n",
                        (unsigned)sample_id,
                        (unsigned)xpred_bad_idx,
                        (unsigned)xpred_bad_got,
                        (unsigned)xpred_bad_exp);
            return 1;
        }

        std::vector<uint32_t> dut_logits_words;
        do_read_mem_words(io,
                          (uint32_t)aecct::top_peek_infer_logits_base_word().to_uint(),
                          (uint32_t)EXP_LEN_OUT_LOGITS_WORDS,
                          dut_logits_words);
        std::vector<uint16_t> dut_logits_words16;
        words_u32_to_words16_le(dut_logits_words, dut_logits_words16);
        uint32_t logits_bad_idx = 0u;
        uint16_t logits_bad_got = 0u;
        uint16_t logits_bad_exp = 0u;
        const bool logits_exact = compare_u16_exact(logits_image.data_out_words16,
                                                    dut_logits_words16,
                                                    logits_bad_idx,
                                                    logits_bad_got,
                                                    logits_bad_exp);
        if (logits_exact) {
            logits_diag_exact_samples += 1u;
        } else if (!logits_diag_first_mismatch_valid) {
            logits_diag_first_mismatch_valid = true;
            logits_diag_first_mismatch_sample = sample_id;
            logits_diag_first_mismatch_idx = logits_bad_idx;
            logits_diag_first_mismatch_dut = logits_bad_got;
            logits_diag_first_mismatch_ref = logits_bad_exp;
        }

        std::printf("[p11au][debug] sample=%u direct/read_mem final_scalar begin\n", (unsigned)sample_id);
        std::vector<uint32_t> dut_final_scalar_direct_words((uint32_t)N_NODES, 0u);
        const aecct::u32_t* sram = aecct::top_sram();
        for (uint32_t i = 0u; i < (uint32_t)N_NODES; ++i) {
            dut_final_scalar_direct_words[i] = (uint32_t)sram[(uint32_t)sram_map::SCR_FINAL_SCALAR_BASE_W + i].to_uint();
        }
        std::vector<uint16_t> dut_final_scalar_direct_words16;
        words_u32_to_words16_le(dut_final_scalar_direct_words, dut_final_scalar_direct_words16);
        uint32_t final_direct_bad_idx = 0u;
        uint16_t final_direct_bad_got = 0u;
        uint16_t final_direct_bad_exp = 0u;
        const bool final_direct_exact = compare_u16_exact(ref_final_scalar_words16,
                                                          dut_final_scalar_direct_words16,
                                                          final_direct_bad_idx,
                                                          final_direct_bad_got,
                                                          final_direct_bad_exp);
        if (final_direct_exact) {
            final_scalar_direct_diag_exact_samples += 1u;
        } else if (!final_scalar_direct_first_mismatch_valid) {
            final_scalar_direct_first_mismatch_valid = true;
            final_scalar_direct_first_mismatch_sample = sample_id;
            final_scalar_direct_first_mismatch_idx = final_direct_bad_idx;
            final_scalar_direct_first_mismatch_dut = final_direct_bad_got;
            final_scalar_direct_first_mismatch_ref = final_direct_bad_exp;
        }
        std::vector<uint32_t> dut_final_scalar_words;
        do_read_mem_words(io,
                          (uint32_t)sram_map::SCR_FINAL_SCALAR_BASE_W,
                          (uint32_t)N_NODES,
                          dut_final_scalar_words);
        std::vector<uint16_t> dut_final_scalar_words16;
        words_u32_to_words16_le(dut_final_scalar_words, dut_final_scalar_words16);
        uint32_t final_bad_idx = 0u;
        uint16_t final_bad_got = 0u;
        uint16_t final_bad_exp = 0u;
        const bool final_exact = compare_u16_exact(ref_final_scalar_words16,
                                                   dut_final_scalar_words16,
                                                   final_bad_idx,
                                                   final_bad_got,
                                                   final_bad_exp);
        if (final_exact) {
            final_scalar_readmem_diag_exact_samples += 1u;
        } else if (!final_scalar_readmem_first_mismatch_valid) {
            final_scalar_readmem_first_mismatch_valid = true;
            final_scalar_readmem_first_mismatch_sample = sample_id;
            final_scalar_readmem_first_mismatch_idx = final_bad_idx;
            final_scalar_readmem_first_mismatch_dut = final_bad_got;
            final_scalar_readmem_first_mismatch_ref = final_bad_exp;
        }

        matched_samples += 1u;
        std::printf("[p11au][sample] sample=%u xpred_exact=1 logits_io16_diag_exact=%u final_scalar_direct_diag_exact=%u final_scalar_readmem_diag_exact=%u logits_words16=%u final_scalar_words16=%u\n",
                    (unsigned)sample_id,
                    (unsigned)(logits_exact ? 1u : 0u),
                    (unsigned)(final_direct_exact ? 1u : 0u),
                    (unsigned)(final_exact ? 1u : 0u),
                    (unsigned)logits_image.data_out_words16.size(),
                    (unsigned)ref_final_scalar_words16.size());
    }

    std::printf("[p11au][summary] samples=%u matched=%u final_scalar_direct_diag_exact_samples=%u final_scalar_readmem_diag_exact_samples=%u logits_diag_exact_samples=%u verdict=PASS\n",
                (unsigned)kFocusedSampleCount,
                (unsigned)matched_samples,
                (unsigned)final_scalar_direct_diag_exact_samples,
                (unsigned)final_scalar_readmem_diag_exact_samples,
                (unsigned)logits_diag_exact_samples);
    if (final_scalar_direct_first_mismatch_valid) {
        std::printf("[p11au][final_scalar_direct_diag] first_mismatch_sample=%u idx=%u dut=0x%04X ref=0x%04X\n",
                    (unsigned)final_scalar_direct_first_mismatch_sample,
                    (unsigned)final_scalar_direct_first_mismatch_idx,
                    (unsigned)final_scalar_direct_first_mismatch_dut,
                    (unsigned)final_scalar_direct_first_mismatch_ref);
    } else {
        std::printf("[p11au][final_scalar_direct_diag] first_mismatch_sample=none idx=none dut=none ref=none\n");
    }
    if (final_scalar_readmem_first_mismatch_valid) {
        std::printf("[p11au][final_scalar_readmem_diag] first_mismatch_sample=%u idx=%u dut=0x%04X ref=0x%04X\n",
                    (unsigned)final_scalar_readmem_first_mismatch_sample,
                    (unsigned)final_scalar_readmem_first_mismatch_idx,
                    (unsigned)final_scalar_readmem_first_mismatch_dut,
                    (unsigned)final_scalar_readmem_first_mismatch_ref);
    } else {
        std::printf("[p11au][final_scalar_readmem_diag] first_mismatch_sample=none idx=none dut=none ref=none\n");
    }
    if (logits_diag_first_mismatch_valid) {
        std::printf("[p11au][logits_diag] first_mismatch_sample=%u idx=%u dut=0x%04X ref=0x%04X\n",
                    (unsigned)logits_diag_first_mismatch_sample,
                    (unsigned)logits_diag_first_mismatch_idx,
                    (unsigned)logits_diag_first_mismatch_dut,
                    (unsigned)logits_diag_first_mismatch_ref);
    } else {
        std::printf("[p11au][logits_diag] first_mismatch_sample=none idx=none dut=none ref=none\n");
    }
    std::printf("PASS: tb_refmodel_io16_focus_p11au\n");
    return 0;
}
