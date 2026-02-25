// tb_top_m0.cpp
// M0 最小 TB：只驗證 ctrl_rsp（NOOP / SOFT_RESET / 未實作 opcode -> ERR_UNIMPL）

#include <cstdio>
#include <cstdlib>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "Top.h"

static void expect_rsp_done(aecct::ctrl_ch_t& ctrl_rsp, uint8_t expect_op) {
    aecct::u16_t w;
    if (!ctrl_rsp.nb_read(w)) {
        std::printf("ERROR: expected ctrl_rsp but channel empty.\n");
        std::exit(1);
    }
    uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
    uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);

    if (kind != (uint8_t)aecct::RSP_DONE || payload != expect_op) {
        std::printf("ERROR: rsp mismatch. kind=%u payload=%u (expect DONE=%u, op=%u)\n",
            (unsigned)kind, (unsigned)payload,
            (unsigned)aecct::RSP_DONE, (unsigned)expect_op);
        std::exit(1);
    }
}

static void expect_rsp_err(aecct::ctrl_ch_t& ctrl_rsp, uint8_t expect_err) {
    aecct::u16_t w;
    if (!ctrl_rsp.nb_read(w)) {
        std::printf("ERROR: expected ctrl_rsp but channel empty.\n");
        std::exit(1);
    }
    uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
    uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);

    if (kind != (uint8_t)aecct::RSP_ERR || payload != expect_err) {
        std::printf("ERROR: rsp mismatch. kind=%u payload=%u (expect ERR=%u, err=%u)\n",
            (unsigned)kind, (unsigned)payload,
            (unsigned)aecct::RSP_ERR, (unsigned)expect_err);
        std::exit(1);
    }
}

int main() {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;

    // 1) NOOP -> DONE(NOOP)
    ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_NOOP));
    aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
    expect_rsp_done(ctrl_rsp, (uint8_t)aecct::OP_NOOP);

    // 2) SOFT_RESET -> DONE(SOFT_RESET)
    ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_SOFT_RESET));
    aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
    expect_rsp_done(ctrl_rsp, (uint8_t)aecct::OP_SOFT_RESET);

    // 3) 未實作：CFG_BEGIN -> ERR(ERR_UNIMPL)
    ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_CFG_BEGIN));
    aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
    expect_rsp_err(ctrl_rsp, (uint8_t)aecct::ERR_UNIMPL);

    std::printf("PASS: tb_top_m0 (NOOP, SOFT_RESET, CFG_BEGIN->ERR_UNIMPL)\n");
    return 0;
}