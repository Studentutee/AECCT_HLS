// tb_top_m3.cpp
// M3 TB：驗證 CFG_RX expected_len / 合法性 / 狀態機

#include <cstdio>
#include <cstdlib>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "ModelDescBringup.h"
#include "Top.h"

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

static void send_cfg_words(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    const uint32_t* cfg_words,
    unsigned n
) {
    for (unsigned i = 0; i < n; ++i) {
        data_in.write((aecct::u32_t)cfg_words[i]);
        tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
        expect_no_rsp(ctrl_rsp);
    }
}

static void build_valid_cfg(uint32_t* cfg_words) {
    for (unsigned i = 0; i < aecct::CFG_WORDS_EXPECTED; ++i) {
        cfg_words[i] = 0u;
    }

    cfg_words[aecct::CFG_IDX_MAGIC] = 0xABCD0001u;
    cfg_words[aecct::CFG_IDX_CODE_N] = 16u;
    cfg_words[aecct::CFG_IDX_CODE_C] = 8u;
    cfg_words[aecct::CFG_IDX_D_MODEL] = 64u;
    cfg_words[aecct::CFG_IDX_N_HEADS] = 8u;
    cfg_words[aecct::CFG_IDX_D_HEAD] = 8u;
    cfg_words[aecct::CFG_IDX_D_FFN] = 128u;
    cfg_words[aecct::CFG_IDX_D_LPE] = 32u;
    cfg_words[aecct::CFG_IDX_N_LAYERS] = 2u;
    cfg_words[aecct::CFG_IDX_OUT_LEN_X_PRED] = 8u;
    cfg_words[aecct::CFG_IDX_OUT_LEN_LOGITS] = 16u;
}

int main() {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;

    uint32_t cfg[aecct::CFG_WORDS_EXPECTED];

    // Case 1：正常 cfg 流程
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    build_valid_cfg(cfg);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_state(aecct::ST_CFG_RX);

    send_cfg_words(ctrl_cmd, ctrl_rsp, data_in, data_out, cfg, aecct::CFG_WORDS_EXPECTED);
    if (!aecct::top_peek_cfg_ready()) {
        std::printf("ERROR: cfg_ready should be true after full cfg ingestion.\n");
        return 1;
    }

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_state(aecct::ST_IDLE);

    if ((uint32_t)aecct::top_peek_cfg_code_n().to_uint() != cfg[aecct::CFG_IDX_CODE_N] ||
        (uint32_t)aecct::top_peek_cfg_d_model().to_uint() != cfg[aecct::CFG_IDX_D_MODEL] ||
        (uint32_t)aecct::top_peek_cfg_n_heads().to_uint() != cfg[aecct::CFG_IDX_N_HEADS] ||
        (uint32_t)aecct::top_peek_cfg_n_layers().to_uint() != cfg[aecct::CFG_IDX_N_LAYERS]) {
        std::printf("ERROR: cfg regs not applied as expected.\n");
        return 1;
    }

    // Case 2：少送 cfg（len mismatch）
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    build_valid_cfg(cfg);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_state(aecct::ST_CFG_RX);

    send_cfg_words(ctrl_cmd, ctrl_rsp, data_in, data_out, cfg, aecct::CFG_WORDS_EXPECTED - 1u);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_CFG_LEN_MISMATCH);
    expect_state(aecct::ST_CFG_RX);

    send_cfg_words(ctrl_cmd, ctrl_rsp, data_in, data_out, &cfg[aecct::CFG_WORDS_EXPECTED - 1u], 1u);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_state(aecct::ST_IDLE);

    // Case 3：非法 cfg（d_model % n_heads != 0）
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    build_valid_cfg(cfg);
    cfg[aecct::CFG_IDX_D_MODEL] = 63u;
    cfg[aecct::CFG_IDX_N_HEADS] = 8u;
    cfg[aecct::CFG_IDX_D_HEAD] = 8u;

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_state(aecct::ST_CFG_RX);

    send_cfg_words(ctrl_cmd, ctrl_rsp, data_in, data_out, cfg, aecct::CFG_WORDS_EXPECTED);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_CFG_ILLEGAL);
    expect_state(aecct::ST_IDLE);

    // Case 4：狀態限制（ST_CFG_RX 禁止 INFER/LOAD_W）
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_state(aecct::ST_CFG_RX);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_INFER);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
    expect_state(aecct::ST_CFG_RX);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
    expect_state(aecct::ST_CFG_RX);

    std::printf("PASS: tb_top_m3 (CFG_RX expected_len + legality)\n");
    return 0;
}
