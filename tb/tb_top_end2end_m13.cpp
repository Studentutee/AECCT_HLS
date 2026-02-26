#define AECCT_FINAL_TRACE_MODE 1

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "AecctProtocol.h"
#include "AecctTypes.h"
#include "gen/ModelDesc.h"
#include "gen/ModelShapes.h"
#include "Top.h"
#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"

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

static void expect_rsp(aecct::ctrl_ch_t& ctrl_rsp, uint8_t kind_exp, uint8_t payload_exp) {
    aecct::u16_t w;
    if (!ctrl_rsp.nb_read(w)) {
        std::printf("ERROR: expected ctrl response but channel empty\n");
        std::exit(1);
    }
    uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
    uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);
    if (kind != kind_exp || payload != payload_exp) {
        std::printf("ERROR: ctrl response mismatch. kind=%u payload=%u expect_kind=%u expect_payload=%u\n",
            (unsigned)kind, (unsigned)payload, (unsigned)kind_exp, (unsigned)payload_exp);
        std::exit(1);
    }
}

static void expect_no_rsp(aecct::ctrl_ch_t& ctrl_rsp) {
    aecct::u16_t w;
    if (ctrl_rsp.nb_read(w)) {
        std::printf("ERROR: unexpected ctrl response. kind=%u payload=%u\n",
            (unsigned)aecct::unpack_ctrl_rsp_kind(w),
            (unsigned)aecct::unpack_ctrl_rsp_payload(w));
        std::exit(1);
    }
}

static bool has_data(aecct::data_ch_t& data_out) {
    aecct::u32_t w;
    if (data_out.nb_read(w)) {
        std::printf("ERROR: unexpected data_out word=0x%08X\n", (unsigned)w.to_uint());
        return true;
    }
    return false;
}

static uint32_t read_data_word(aecct::data_ch_t& data_out) {
    aecct::u32_t w;
    if (!data_out.nb_read(w)) {
        std::printf("ERROR: expected data word but channel empty\n");
        std::exit(1);
    }
    return (uint32_t)w.to_uint();
}

static void tick(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out
) {
    aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
}

static void drive_cmd(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint8_t opcode
) {
    ctrl_cmd.write(aecct::pack_ctrl_cmd(opcode));
    tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
}

static void drive_set_w_base(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint32_t w_base_word
) {
    data_in.write((aecct::u32_t)w_base_word);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SET_W_BASE);
}

static void drive_set_outmode(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint32_t outmode
) {
    data_in.write((aecct::u32_t)outmode);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SET_OUTMODE);
}

static void run_cfg_commit(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out
) {
    uint32_t cfg_words[EXP_LEN_CFG_WORDS];
    for (unsigned i = 0; i < (unsigned)EXP_LEN_CFG_WORDS; ++i) {
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

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);

    for (unsigned i = 0; i < (unsigned)EXP_LEN_CFG_WORDS; ++i) {
        data_in.write((aecct::u32_t)cfg_words[i]);
        tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
        expect_no_rsp(ctrl_rsp);
    }

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_CFG_COMMIT);
}

static void run_load_w_pattern(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out
) {
    drive_set_w_base(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint32_t)sram_map::PARAM_BASE_DEFAULT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_W_BASE);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W);

    for (uint32_t i = 0; i < (uint32_t)EXP_LEN_PARAM_WORDS; ++i) {
        uint32_t word = 0xA5000000u | (i & 0x00FFFFFFu);
        data_in.write((aecct::u32_t)word);
        tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
        if (i + 1u < (uint32_t)EXP_LEN_PARAM_WORDS) {
            expect_no_rsp(ctrl_rsp);
        }
        else {
            expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W);
        }
    }
}

static void run_infer_sample0(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out
) {
    const uint32_t sample_idx = 0u;
    const uint32_t in_words = (uint32_t)EXP_LEN_INFER_IN_WORDS;

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_INFER);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_INFER);

    for (uint32_t i = 0; i < in_words; ++i) {
        float v = (float)trace_input_y_step0_tensor[sample_idx * in_words + i];
        data_in.write((aecct::u32_t)f32_to_bits(v));
        tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
        if (i + 1u < in_words) {
            expect_no_rsp(ctrl_rsp);
        }
        else {
            expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_INFER);
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

int main() {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;

    const uint32_t sample_idx = 0u;
    static uint32_t logits_got[EXP_LEN_OUT_LOGITS_WORDS];
    static uint32_t xpred_got[EXP_LEN_OUT_XPRED_WORDS];

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);

    run_cfg_commit(ctrl_cmd, ctrl_rsp, data_in, data_out);
    run_load_w_pattern(ctrl_cmd, ctrl_rsp, data_in, data_out);

    // Case A: OUTMODE_LOGITS
    drive_set_outmode(ctrl_cmd, ctrl_rsp, data_in, data_out, 1u);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE);

    run_infer_sample0(ctrl_cmd, ctrl_rsp, data_in, data_out);
    for (uint32_t i = 0; i < (uint32_t)EXP_LEN_OUT_LOGITS_WORDS; ++i) {
        logits_got[i] = read_data_word(data_out);
    }
    if (compare_output(
        "output_logits_step0",
        logits_got,
        trace_output_logits_step0_tensor,
        sample_idx,
        (uint32_t)EXP_LEN_OUT_LOGITS_WORDS,
        1.0e-5
    ) != 0) {
        return 1;
    }

    // Case B: OUTMODE_XPRED
    drive_set_outmode(ctrl_cmd, ctrl_rsp, data_in, data_out, 0u);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE);

    run_infer_sample0(ctrl_cmd, ctrl_rsp, data_in, data_out);
    for (uint32_t i = 0; i < (uint32_t)EXP_LEN_OUT_XPRED_WORDS; ++i) {
        xpred_got[i] = read_data_word(data_out);
    }
    if (compare_output(
        "output_x_pred_step0",
        xpred_got,
        trace_output_x_pred_step0_tensor,
        sample_idx,
        (uint32_t)EXP_LEN_OUT_XPRED_WORDS,
        1.0e-5
    ) != 0) {
        return 1;
    }

    // Case C: OUTMODE_NONE
    drive_set_outmode(ctrl_cmd, ctrl_rsp, data_in, data_out, 2u);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE);

    run_infer_sample0(ctrl_cmd, ctrl_rsp, data_in, data_out);
    if (has_data(data_out)) {
        return 1;
    }

    std::printf("PASS: tb_top_end2end_m13\n");
    return 0;
}

