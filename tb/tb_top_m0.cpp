// tb_top_m0.cpp
// M0 minimal bring-up TB:
// - Verify ctrl_rsp format with NOOP and SOFT_RESET.
// - Verify unimplemented opcode returns ERR_UNIMPL.

#include <cstdio>
#include <cstdlib>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "Top.h"

static void expect_rsp(aecct::ctrl_ch_t& ctrl_rsp, uint8_t expect_kind, uint8_t expect_payload) {
  aecct::u16_t w;
  if (!ctrl_rsp.nb_read(w)) {
    std::printf("ERROR: expected ctrl_rsp but channel empty.\n");
    std::exit(1);
  }
  const uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
  const uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);
  if (kind != expect_kind || payload != expect_payload) {
    std::printf("ERROR: rsp mismatch. kind=%u payload=%u (expect kind=%u payload=%u)\n",
                (unsigned)kind, (unsigned)payload, (unsigned)expect_kind, (unsigned)expect_payload);
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
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_NOOP);

  // 2) SOFT_RESET -> DONE(SOFT_RESET)
  ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_SOFT_RESET));
  aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);

  // 3) Unknown opcode -> ERR(ERR_UNIMPL)
  const uint8_t kUnknownOp = 0xEEu;
  ctrl_cmd.write(aecct::pack_ctrl_cmd(kUnknownOp));
  aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_UNIMPL);

  std::printf("PASS: tb_top_m0\n");
  return 0;
}
