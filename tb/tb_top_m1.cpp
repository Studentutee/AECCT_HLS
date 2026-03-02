// tb_top_m1.cpp
// M1 bring-up TB: verify Top-FSM "state x opcode" rules.

#include <cstdio>
#include <cstdlib>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "gen/SramMap.h"
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

static void expect_state(aecct::TopState expect_s) {
  const aecct::TopState s = aecct::top_peek_state();
  if ((unsigned)s != (unsigned)expect_s) {
    std::printf("ERROR: state mismatch. state=%u expect=%u\n", (unsigned)s, (unsigned)expect_s);
    std::exit(1);
  }
}

static void drive_cmd0(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint8_t opcode) {
  ctrl_cmd.write(aecct::pack_ctrl_cmd(opcode));
  aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
}

static void drive_cmd1_u32(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint8_t opcode,
    uint32_t arg0_u32) {
  // NOTE: All stimuli are queued BEFORE calling top().
  ctrl_cmd.write(aecct::pack_ctrl_cmd(opcode));
  data_in.write((aecct::u32_t)arg0_u32);
  aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
}

int main() {
  aecct::ctrl_ch_t ctrl_cmd;
  aecct::ctrl_ch_t ctrl_rsp;
  aecct::data_ch_t data_in;
  aecct::data_ch_t data_out;

  // Always start from a clean state.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
  expect_state(aecct::ST_IDLE);

  // Case 1: IDLE basic commands.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_NOOP);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_NOOP);
  expect_state(aecct::ST_IDLE);

  // Case 2: Enter CFG_RX, then test illegal opcodes and early commit.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
  expect_state(aecct::ST_CFG_RX);

  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_INFER);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
  expect_state(aecct::ST_CFG_RX);

  // No cfg words provided => commit must fail with CFG_LEN_MISMATCH.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_CFG_LEN_MISMATCH);
  expect_state(aecct::ST_CFG_RX);

  // Exit CFG_RX via SOFT_RESET.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
  expect_state(aecct::ST_IDLE);

  // Case 3: LOAD_W requires SET_W_BASE first.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
  expect_state(aecct::ST_IDLE);

  // Provide a valid aligned base inside W_REGION.
  drive_cmd1_u32(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SET_W_BASE, sram_map::PARAM_BASE_DEFAULT);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_SET_W_BASE);
  expect_state(aecct::ST_IDLE);

  // LOAD_W enters PARAM_RX and replies OK (transaction continues until PARAM_WORDS_EXPECTED words are ingested).
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W);
  expect_state(aecct::ST_PARAM_RX);

  // Illegal opcode while in PARAM_RX.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
  expect_state(aecct::ST_PARAM_RX);

  // Exit PARAM_RX via SOFT_RESET.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
  expect_state(aecct::ST_IDLE);

  // Case 4: INFER requires cfg_ready (not set in this bring-up TB).
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_INFER);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
  expect_state(aecct::ST_IDLE);

  std::printf("PASS: tb_top_m1\n");
  return 0;
}
