// tb_top_m6.cpp
// M6 TB嚗?霅?SET_OUTMODE + INFER stub + ?????
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "gen/ModelDesc.h"
#include "gen/SramMap.h"
#include "Top.h"

static const unsigned CFG_WORDS_EXPECTED = (unsigned)EXP_LEN_CFG_WORDS;
static const uint32_t INFER_PATTERN_HI = 0x10000000u;

static void expect_rsp(aecct::ctrl_ch_t& ctrl_rsp, uint8_t expect_kind, uint8_t expect_payload) {
    aecct::u16_t w;
    if (!ctrl_rsp.nb_read(w)) {
        std::printf("ERROR: expected ctrl_rsp but channel empty.\n");
        std::exit(1);
    }
    uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
    uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);
    if (kind != expect_kind || payload != expect_payload) {
        std::printf("ERROR: rsp mismatch. kind=%u payload=%u (expect kind=%u payload=%u)\n",
            (unsigned)kind, (unsigned)payload,
            (unsigned)expect_kind, (unsigned)expect_payload);
        std::exit(1);
    }
}

static void expect_no_rsp(aecct::ctrl_ch_t& ctrl_rsp) {
    aecct::u16_t w;
    if (ctrl_rsp.nb_read(w)) {
        std::printf("ERROR: unexpected ctrl_rsp. kind=%u payload=%u\n",
            (unsigned)aecct::unpack_ctrl_rsp_kind(w),
            (unsigned)aecct::unpack_ctrl_rsp_payload(w));
        std::exit(1);
    }
}

static void expect_no_data(aecct::data_ch_t& data_out) {
    aecct::u32_t w;
    if (data_out.nb_read(w)) {
        std::printf("ERROR: unexpected data_out. word=0x%08X\n", (unsigned)w.to_uint());
        std::exit(1);
    }
}

static uint32_t read_data_word(aecct::data_ch_t& data_out) {
    aecct::u32_t w;
    if (!data_out.nb_read(w)) {
        std::printf("ERROR: expected data_out word but channel empty.\n");
        std::exit(1);
    }
    return (uint32_t)w.to_uint();
}

static void expect_state(aecct::TopState expect_s) {
    aecct::TopState s = aecct::top_peek_state();
    if ((unsigned)s != (unsigned)expect_s) {
        std::printf("ERROR: state mismatch. got=%u expect=%u\n",
            (unsigned)s, (unsigned)expect_s);
        std::exit(1);
    }
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

static void build_valid_cfg(uint32_t* cfg_words) {
    for (unsigned i = 0; i < CFG_WORDS_EXPECTED; ++i) {
        cfg_words[i] = 0u;
    }

    cfg_words[CFG_CODE_N] = 63u;
    cfg_words[CFG_CODE_K] = 51u;
    cfg_words[CFG_CODE_C] = 12u;
    cfg_words[CFG_N_NODES] = 75u;
    cfg_words[CFG_D_MODEL] = 64u;
    cfg_words[CFG_N_HEAD] = 8u;
    cfg_words[CFG_N_LAYERS] = 2u;
    cfg_words[CFG_D_FFN] = 128u;
    cfg_words[CFG_ENABLE_LPE] = 1u;
    cfg_words[CFG_ENABLE_LPE_TOKEN] = 1u;
    cfg_words[CFG_OUT_MODE] = 0u;
    cfg_words[CFG_RESERVED0] = 0u;
}

static void send_cfg_words(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    const uint32_t* cfg_words
) {
    for (unsigned i = 0; i < CFG_WORDS_EXPECTED; ++i) {
        data_in.write((aecct::u32_t)cfg_words[i]);
        tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
        expect_no_rsp(ctrl_rsp);
    }
}

static void run_cfg_commit(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out
) {
    uint32_t cfg[CFG_WORDS_EXPECTED];
    build_valid_cfg(cfg);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_state(aecct::ST_CFG_RX);

    send_cfg_words(ctrl_cmd, ctrl_rsp, data_in, data_out, cfg);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_state(aecct::ST_IDLE);
}

static void run_infer_with_input(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out
) {
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_INFER);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_INFER);
    expect_state(aecct::ST_INFER_RX);

    for (unsigned i = 0; i < aecct::INFER_IN_WORDS_EXPECTED; ++i) {
        data_in.write((aecct::u32_t)(INFER_PATTERN_HI | i));
        tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
        if (i + 1u < aecct::INFER_IN_WORDS_EXPECTED) {
            expect_no_rsp(ctrl_rsp);
        }
        else {
            expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_INFER);
        }
    }
    expect_state(aecct::ST_IDLE);
}

int main() {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;

    // Case 1嚗ET_OUTMODE=NONE嚗NFER 銝? data_out
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    run_cfg_commit(ctrl_cmd, ctrl_rsp, data_in, data_out);

    drive_set_outmode(ctrl_cmd, ctrl_rsp, data_in, data_out, 2u);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE);
    run_infer_with_input(ctrl_cmd, ctrl_rsp, data_in, data_out);
    expect_no_data(data_out);

    // Case 2嚗ET_OUTMODE=X_PRED嚗NFER ??OUT_WORDS_X_PRED
    drive_set_outmode(ctrl_cmd, ctrl_rsp, data_in, data_out, 0u);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE);
    run_infer_with_input(ctrl_cmd, ctrl_rsp, data_in, data_out);
    for (unsigned i = 0; i < aecct::OUT_WORDS_X_PRED; ++i) {
        uint32_t got = read_data_word(data_out);
        uint32_t inw = INFER_PATTERN_HI | (i % aecct::INFER_IN_WORDS_EXPECTED);
        uint32_t expect = inw ^ 0x5A5A5A5Au;
        if (got != expect) {
            std::printf("ERROR: X_PRED mismatch at i=%u. got=0x%08X expect=0x%08X\n",
                (unsigned)i, (unsigned)got, (unsigned)expect);
            return 1;
        }
    }
    expect_no_data(data_out);

    // Case 3嚗ET_OUTMODE=LOGITS嚗NFER ??OUT_WORDS_LOGITS
    drive_set_outmode(ctrl_cmd, ctrl_rsp, data_in, data_out, 1u);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_OUTMODE);
    run_infer_with_input(ctrl_cmd, ctrl_rsp, data_in, data_out);
    for (unsigned i = 0; i < aecct::OUT_WORDS_LOGITS; ++i) {
        uint32_t got = read_data_word(data_out);
        uint32_t expect = 0xC0000000u | i;
        if (got != expect) {
            std::printf("ERROR: LOGITS mismatch at i=%u. got=0x%08X expect=0x%08X\n",
                (unsigned)i, (unsigned)got, (unsigned)expect);
            return 1;
        }
    }
    expect_no_data(data_out);

    // Case 4嚗?瘜?outmode
    drive_set_outmode(ctrl_cmd, ctrl_rsp, data_in, data_out, 3u);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_ARG);

    // Case 5嚗????塚?ST_CFG_RX / ST_PARAM_RX嚗?    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_state(aecct::ST_CFG_RX);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_INFER);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SET_OUTMODE);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    drive_set_w_base(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint32_t)sram_map::W_REGION_BASE);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_W_BASE);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W);
    expect_state(aecct::ST_PARAM_RX);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_INFER);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SET_OUTMODE);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);

    std::printf("PASS: tb_top_m6 (INFER stub + SET_OUTMODE)\n");
    return 0;
}

