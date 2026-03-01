// tb_top_m4.cpp
// M4 TB嚗?霅?SET_W_BASE + LOAD_W(PARAM_RX) + SRAM 撖怠/?航炊蝣?
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"
#include "Top.h"

static const uint32_t PARAM_WORDS_EXPECTED = (uint32_t)EXP_LEN_PARAM_WORDS;
static const uint32_t PARAM_ALIGN_WORDS = (uint32_t)W_LANES;

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
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_READ_MEM);
}

static void send_load_w_payload(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint32_t pattern_hi
) {
    for (uint32_t i = 0; i < PARAM_WORDS_EXPECTED; ++i) {
        data_in.write((aecct::u32_t)(pattern_hi | i));
        tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
        if (i + 1u < PARAM_WORDS_EXPECTED) {
            expect_no_rsp(ctrl_rsp);
        }
        else {
            expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W);
        }
    }
}

int main() {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;

    const uint32_t good_base = (uint32_t)sram_map::W_REGION_BASE;
    const uint32_t pattern_hi = 0xA0000000u;

    // Case 1嚗ET_W_BASE 甇?虜 + LOAD_W 甇?虜 + READ_MEM 霈??撠?    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    drive_set_w_base(ctrl_cmd, ctrl_rsp, data_in, data_out, good_base);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_W_BASE);
    expect_state(aecct::ST_IDLE);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W);
    expect_state(aecct::ST_PARAM_RX);

    send_load_w_payload(ctrl_cmd, ctrl_rsp, data_in, data_out, pattern_hi);
    expect_state(aecct::ST_IDLE);

    const uint32_t readback_words = 16u;
    drive_read_mem(ctrl_cmd, ctrl_rsp, data_in, data_out, good_base, readback_words);
    for (uint32_t i = 0; i < readback_words; ++i) {
        expect_data_word(data_out, pattern_hi | i);
    }
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM);
    expect_state(aecct::ST_IDLE);

    // Case 2嚗ET_W_BASE 頞?
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    drive_set_w_base(
        ctrl_cmd, ctrl_rsp, data_in, data_out,
        (uint32_t)(sram_map::W_REGION_BASE + sram_map::W_REGION_WORDS)
    );
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_PARAM_BASE_RANGE);
    expect_state(aecct::ST_IDLE);

    // Case 3嚗ET_W_BASE 撠??航炊嚗4 ?∠ W_LANES 撠?嚗?    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    if (PARAM_ALIGN_WORDS <= 1u) {
        std::printf("ERROR: PARAM_ALIGN_WORDS must be >1 for M4 alignment case.\n");
        return 1;
    }
    drive_set_w_base(ctrl_cmd, ctrl_rsp, data_in, data_out, good_base + 1u);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_PARAM_BASE_ALIGN);
    expect_state(aecct::ST_IDLE);

    // Case 4嚗OAD_W ?芾身 base
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
    expect_state(aecct::ST_IDLE);

    // Case 5嚗OAD_W ????base near end嚗?    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    uint32_t near_end_base = (uint32_t)(sram_map::W_REGION_BASE + sram_map::W_REGION_WORDS - (PARAM_WORDS_EXPECTED / 2u));
    if ((near_end_base % PARAM_ALIGN_WORDS) != 0u) {
        near_end_base = near_end_base - (near_end_base % PARAM_ALIGN_WORDS);
    }
    drive_set_w_base(ctrl_cmd, ctrl_rsp, data_in, data_out, near_end_base);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_W_BASE);
    expect_state(aecct::ST_IDLE);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_MEM_RANGE);
    expect_state(aecct::ST_IDLE);

    std::printf("PASS: tb_top_m4 (SET_W_BASE + LOAD_W PARAM_RX)\n");
    return 0;
}

