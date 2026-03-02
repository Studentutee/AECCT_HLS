// tb_top_m2.cpp
// M2 bring-up TB: verify single SRAM map + READ_MEM behavior.

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

static void expect_no_data(aecct::data_ch_t& data_out) {
  aecct::u32_t w;
  if (data_out.nb_read(w)) {
    std::printf("ERROR: expect no data_out, but got word=0x%08X\n", (unsigned)w.to_uint());
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

static void drive_read_mem(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint32_t addr_word,
    uint32_t len_words) {
  // NOTE: All stimuli are queued BEFORE calling top().
  ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_READ_MEM));
  data_in.write((aecct::u32_t)addr_word);
  data_in.write((aecct::u32_t)len_words);
  aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
}

static uint32_t make_pattern(uint32_t region_id, uint32_t local_idx) {
  return ((region_id & 0xFFu) << 24) | (local_idx & 0x00FFFFFFu);
}

static void expect_data_word(aecct::data_ch_t& data_out, uint32_t expect_word) {
  aecct::u32_t w;
  if (!data_out.nb_read(w)) {
    std::printf("ERROR: expected data_out word but channel empty.\n");
    std::exit(1);
  }
  const uint32_t got = (uint32_t)w.to_uint();
  if (got != expect_word) {
    std::printf("ERROR: data mismatch. got=0x%08X expect=0x%08X\n", (unsigned)got, (unsigned)expect_word);
    std::exit(1);
  }
}

int main() {
  aecct::ctrl_ch_t ctrl_cmd;
  aecct::ctrl_ch_t ctrl_rsp;
  aecct::data_ch_t data_in;
  aecct::data_ch_t data_out;

  // Reset first to initialize SRAM pattern.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
  expect_state(aecct::ST_IDLE);

  // Case 1: Read from X_PAGE0 base.
  drive_read_mem(ctrl_cmd, ctrl_rsp, data_in, data_out, sram_map::X_PAGE0_BASE_W, 8u);
  for (uint32_t i = 0; i < 8u; ++i) {
    expect_data_word(data_out, make_pattern(0u, i));
  }
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM);
  expect_state(aecct::ST_IDLE);

  // Case 2: Out-of-range READ_MEM must return ERR_MEM_RANGE and output no data.
  drive_read_mem(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint32_t)(sram_map::SRAM_WORDS_TOTAL - 4u), 8u);
  expect_no_data(data_out);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_MEM_RANGE);
  expect_state(aecct::ST_IDLE);

  // Case 3: In CFG_RX, READ_MEM must return ERR_BAD_STATE and must NOT consume READ_MEM args.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
  expect_state(aecct::ST_CFG_RX);

  drive_read_mem(ctrl_cmd, ctrl_rsp, data_in, data_out, sram_map::X_PAGE0_BASE_W, 2u);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
  expect_no_data(data_out);
  expect_state(aecct::ST_CFG_RX);

  // Drain the two args; they must remain in data_in because READ_MEM is illegal in CFG_RX.
  aecct::u32_t sink;
  if (!data_in.nb_read(sink) || !data_in.nb_read(sink)) {
    std::printf("ERROR: READ_MEM args unexpectedly consumed in BAD_STATE path.\n");
    std::exit(1);
  }
  if (data_in.nb_read(sink)) {
    std::printf("ERROR: unexpected extra data_in residue.\n");
    std::exit(1);
  }

  // CFG_COMMIT without cfg words must fail with CFG_LEN_MISMATCH and stay in CFG_RX.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_CFG_LEN_MISMATCH);
  expect_state(aecct::ST_CFG_RX);

  // Exit CFG_RX and finish.
  drive_cmd0(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
  expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
  expect_state(aecct::ST_IDLE);

  std::printf("PASS: tb_top_m2\n");
  return 0;
}
