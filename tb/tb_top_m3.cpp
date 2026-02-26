// tb_top_m3.cpp
// M3 TB嚗?霅?CFG_RX expected_len / ????/ ???

#include <cstdio>
#include <cstdlib>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "gen/ModelDesc.h"
#include "Top.h"

static const unsigned CFG_WORDS_EXPECTED = (unsigned)EXP_LEN_CFG_WORDS;

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

int main() {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;

    uint32_t cfg[CFG_WORDS_EXPECTED];

    // Case 1嚗迤撣?cfg 瘚?
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    build_valid_cfg(cfg);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_state(aecct::ST_CFG_RX);

    send_cfg_words(ctrl_cmd, ctrl_rsp, data_in, data_out, cfg, CFG_WORDS_EXPECTED);
    if (!aecct::top_peek_cfg_ready()) {
        std::printf("ERROR: cfg_ready should be true after full cfg ingestion.\n");
        return 1;
    }

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_state(aecct::ST_IDLE);

    if ((uint32_t)aecct::top_peek_cfg_code_n().to_uint() != cfg[CFG_CODE_N] ||
        (uint32_t)aecct::top_peek_cfg_d_model().to_uint() != cfg[CFG_D_MODEL] ||
        (uint32_t)aecct::top_peek_cfg_n_heads().to_uint() != cfg[CFG_N_HEAD] ||
        (uint32_t)aecct::top_peek_cfg_n_layers().to_uint() != cfg[CFG_N_LAYERS]) {
        std::printf("ERROR: cfg regs not applied as expected.\n");
        return 1;
    }

    // Case 2嚗???cfg嚗en mismatch嚗?
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    build_valid_cfg(cfg);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_state(aecct::ST_CFG_RX);

    send_cfg_words(ctrl_cmd, ctrl_rsp, data_in, data_out, cfg, CFG_WORDS_EXPECTED - 1u);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_CFG_LEN_MISMATCH);
    expect_state(aecct::ST_CFG_RX);

    send_cfg_words(ctrl_cmd, ctrl_rsp, data_in, data_out, &cfg[CFG_WORDS_EXPECTED - 1u], 1u);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_state(aecct::ST_IDLE);

    // Case 3嚗?瘜?cfg嚗_model % n_heads != 0嚗?
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    build_valid_cfg(cfg);
    cfg[CFG_D_MODEL] = 63u;
    cfg[CFG_N_HEAD] = 8u;

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_state(aecct::ST_CFG_RX);

    send_cfg_words(ctrl_cmd, ctrl_rsp, data_in, data_out, cfg, CFG_WORDS_EXPECTED);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_CFG_ILLEGAL);
    expect_state(aecct::ST_IDLE);

    // Case 4嚗????塚?ST_CFG_RX 蝳迫 INFER/LOAD_W嚗?
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

