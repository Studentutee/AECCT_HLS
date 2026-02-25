// tb_top_m2.cpp
// M2 TB：驗證單一 SRAM + READ_MEM（讀回/越界/狀態限制）

#include <cstdio>
#include <cstdlib>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "SramMapBringup.h"
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

static void expect_no_data(aecct::data_ch_t& data_out) {
    aecct::u32_t w;
    if (data_out.nb_read(w)) {
        std::printf("ERROR: expect no data_out, but got word=0x%08X\n", (unsigned)w.to_uint());
        std::exit(1);
    }
}

static void expect_state(aecct::TopState expect_s) {
    aecct::TopState s = aecct::top_peek_state();
    if ((unsigned)s != (unsigned)expect_s) {
        std::printf("ERROR: state mismatch. state=%u expect=%u\n",
            (unsigned)s, (unsigned)expect_s);
        std::exit(1);
    }
}

static void drive_cmd(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint8_t opcode
) {
    ctrl_cmd.write(aecct::pack_ctrl_cmd(opcode));
    aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
}

static void drive_read_mem(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint32_t addr_word,
    uint32_t len_words
) {
    data_in.write((aecct::u32_t)addr_word);
    data_in.write((aecct::u32_t)len_words);
    ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_READ_MEM));
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
    uint32_t got = (uint32_t)w.to_uint();
    if (got != expect_word) {
        std::printf("ERROR: data mismatch. got=0x%08X expect=0x%08X\n",
            (unsigned)got, (unsigned)expect_word);
        std::exit(1);
    }
}

int main() {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;

    // 先 reset，確保 SRAM pattern 初始化
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    // Case 1：基本讀回（IDLE）
    drive_read_mem(ctrl_cmd, ctrl_rsp, data_in, data_out, aecct::X0_BASE_WORD, 8);
    for (uint32_t i = 0; i < 8; ++i) {
        expect_data_word(data_out, make_pattern(0u, i));
    }
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM);
    expect_state(aecct::ST_IDLE);

    // Case 3：越界錯誤
    drive_read_mem(
        ctrl_cmd, ctrl_rsp, data_in, data_out,
        (uint32_t)(aecct::SRAM_TOTAL_WORDS - 4u), 8u
    );
    expect_no_data(data_out);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_MEM_RANGE);
    expect_state(aecct::ST_IDLE);

    // Case 4：狀態限制（BAD_STATE）
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_state(aecct::ST_CFG_RX);

    // 依要求在 BAD_STATE case 也塞 READ_MEM args
    drive_read_mem(ctrl_cmd, ctrl_rsp, data_in, data_out, aecct::X0_BASE_WORD, 2u);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
    expect_no_data(data_out);
    expect_state(aecct::ST_CFG_RX);

    // 驗證 BAD_STATE 路徑不會消耗 data_in 參數
    aecct::u32_t sink;
    if (!data_in.nb_read(sink) || !data_in.nb_read(sink)) {
        std::printf("ERROR: READ_MEM args unexpectedly consumed in BAD_STATE path.\n");
        std::exit(1);
    }
    if (data_in.nb_read(sink)) {
        std::printf("ERROR: unexpected extra data_in residue.\n");
        std::exit(1);
    }

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_CFG_COMMIT);
    expect_state(aecct::ST_IDLE);

    std::printf("PASS: tb_top_m2 (single SRAM + READ_MEM)\n");
    return 0;
}
