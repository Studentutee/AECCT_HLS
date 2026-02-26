// tb_top_m5.cpp
// M5 TB嚗?霅?DEBUG_CFG + HALTED(meta0/meta1) + READ_MEM + RESUME

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"
#include "Top.h"

static const uint32_t PARAM_WORDS_EXPECTED = (uint32_t)EXP_LEN_PARAM_WORDS;
static const uint32_t PATTERN_HI = 0xA0000000u;

static uint32_t make_dbg_word(uint32_t action, uint32_t trigger_sel, uint32_t k_value) {
    return ((action & 0x3u) | ((trigger_sel & 0xFFu) << 8) | ((k_value & 0xFFFFu) << 16));
}

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

static uint32_t read_data_word(aecct::data_ch_t& data_out) {
    aecct::u32_t w;
    if (!data_out.nb_read(w)) {
        std::printf("ERROR: expected data_out word but channel empty.\n");
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

static void drive_debug_cfg(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint32_t dbg_word
) {
    data_in.write((aecct::u32_t)dbg_word);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_DEBUG_CFG);
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

int main() {
    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;

    const uint32_t good_base = (uint32_t)sram_map::W_REGION_BASE;
    const uint32_t halt_k = 4u;

    // Case 1嚗OAD_W ?? halt + meta + READ_MEM + RESUME + ?蝯?DONE
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);

    drive_set_w_base(ctrl_cmd, ctrl_rsp, data_in, data_out, good_base);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_W_BASE);
    expect_state(aecct::ST_IDLE);

    drive_debug_cfg(
        ctrl_cmd, ctrl_rsp, data_in, data_out,
        make_dbg_word(1u, 1u, halt_k) // ARM + ON_LOADW_COUNT + k=4
    );
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_DEBUG_CFG);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W);
    expect_state(aecct::ST_PARAM_RX);

    uint32_t sent_words = 0u;
    for (uint32_t i = 0; i < PARAM_WORDS_EXPECTED; ++i) {
        data_in.write((aecct::u32_t)(PATTERN_HI | i));
        tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
        if (i < halt_k) {
            expect_no_rsp(ctrl_rsp);
        }
        else if (i == halt_k) {
            expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_DBG_HALT);
            uint32_t meta0 = read_data_word(data_out);
            uint32_t meta1 = read_data_word(data_out);
            if (meta0 != good_base) {
                std::printf("ERROR: meta0 mismatch. got=0x%08X expect=0x%08X\n",
                    (unsigned)meta0, (unsigned)good_base);
                return 1;
            }
            if (meta1 != 16u) {
                std::printf("ERROR: meta1 mismatch. got=%u expect=16\n", (unsigned)meta1);
                return 1;
            }
            expect_state(aecct::ST_HALTED);
            sent_words = i + 1u;

            drive_read_mem(ctrl_cmd, ctrl_rsp, data_in, data_out, meta0, meta1);
            for (uint32_t j = 0; j < meta1; ++j) {
                uint32_t got = read_data_word(data_out);
                if (j < sent_words) {
                    uint32_t expect = (PATTERN_HI | j);
                    if (got != expect) {
                        std::printf("ERROR: readback mismatch at j=%u. got=0x%08X expect=0x%08X\n",
                            (unsigned)j, (unsigned)got, (unsigned)expect);
                        return 1;
                    }
                }
            }
            expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM);

            drive_debug_cfg(
                ctrl_cmd, ctrl_rsp, data_in, data_out,
                make_dbg_word(2u, 0u, 0u) // RESUME
            );
            expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_DEBUG_CFG);
            expect_state(aecct::ST_PARAM_RX);
            break;
        }
    }

    for (uint32_t i = sent_words; i < PARAM_WORDS_EXPECTED; ++i) {
        data_in.write((aecct::u32_t)(PATTERN_HI | i));
        tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
        if (i + 1u < PARAM_WORDS_EXPECTED) {
            expect_no_rsp(ctrl_rsp);
        }
        else {
            expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W);
        }
    }
    expect_state(aecct::ST_IDLE);

    // Case 2嚗ALTED ??????賭誘
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    drive_set_w_base(ctrl_cmd, ctrl_rsp, data_in, data_out, good_base);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SET_W_BASE);
    drive_debug_cfg(
        ctrl_cmd, ctrl_rsp, data_in, data_out,
        make_dbg_word(1u, 1u, 1u) // ARM + ON_LOADW_COUNT + k=1
    );
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_DEBUG_CFG);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W);

    data_in.write((aecct::u32_t)(PATTERN_HI | 0u));
    tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
    expect_no_rsp(ctrl_rsp);
    data_in.write((aecct::u32_t)(PATTERN_HI | 1u));
    tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_DBG_HALT);
    (void)read_data_word(data_out);
    (void)read_data_word(data_out);
    expect_state(aecct::ST_HALTED);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_CFG_BEGIN);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
    expect_state(aecct::ST_HALTED);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_INFER);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE);
    expect_state(aecct::ST_HALTED);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    // Case 3嚗ESUME ?券? HALTED嚗?瘙?ERR_BAD_ARG嚗?    drive_debug_cfg(
        ctrl_cmd, ctrl_rsp, data_in, data_out,
        make_dbg_word(2u, 0u, 0u) // RESUME in IDLE
    );
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_ARG);
    expect_state(aecct::ST_IDLE);

    std::printf("PASS: tb_top_m5 (DEBUG_CFG + HALTED + RESUME)\n");
    return 0;
}

