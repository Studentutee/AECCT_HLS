// tb_backup_wave3_io8_boundary_smoke.cpp
// Backup profile Wave3 smoke:
// - Top IO boundary uses ac_channel<8-bit> serialized payload
// - verify lane and 16-byte SRAM-word serializers

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "AecctProtocol.h"
#include "AecctTypes.h"
#include "gen/SramMap.h"
#include "Top.h"

namespace {

static void fail(const char* msg) {
    std::printf("[wave3][FAIL] %s\n", msg);
    std::exit(1);
}

static void push_u32_le(aecct::data8_ch_t& ch, uint32_t w) {
    ch.write((aecct::u8_t)(w & 0xFFu));
    ch.write((aecct::u8_t)((w >> 8) & 0xFFu));
    ch.write((aecct::u8_t)((w >> 16) & 0xFFu));
    ch.write((aecct::u8_t)((w >> 24) & 0xFFu));
}

static aecct::ctrl_word_t wait_rsp_or_fail(
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::data8_ch_t& data_in,
    aecct::data8_ch_t& data_out
) {
    aecct::ctrl_word_t rsp;
    for (uint32_t i = 0u; i < 200u; ++i) {
        aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
        if (ctrl_rsp.nb_read(rsp)) {
            return rsp;
        }
    }
    fail("timeout waiting ctrl_rsp");
    return (aecct::ctrl_word_t)0;
}

} // namespace

int main() {
    aecct::u8_t lane_b0 = (aecct::u8_t)0xCDu;
    aecct::u8_t lane_b1 = (aecct::u8_t)0xABu;
    const aecct::u16_t lane = aecct::top_io8_deserialize_fp16_lane(lane_b0, lane_b1);
    aecct::u8_t lane_out_b0 = 0;
    aecct::u8_t lane_out_b1 = 0;
    aecct::top_io8_serialize_fp16_lane(lane, lane_out_b0, lane_out_b1);
    if ((uint32_t)lane_out_b0.to_uint() != 0xCDu || (uint32_t)lane_out_b1.to_uint() != 0xABu) {
        fail("fp16 lane 2-byte serializer/deserializer mismatch");
    }

    aecct::u8_t bytes_in[aecct::BACKUP_WORD_BYTES];
    WAVE3_INIT_WORD_BYTES_LOOP: for (uint32_t i = 0u; i < aecct::BACKUP_WORD_BYTES; ++i) {
        bytes_in[i] = (aecct::u8_t)i;
    }
    aecct::u32_t words[aecct::BACKUP_WORD_U32S];
    aecct::top_io8_deserialize_sram_word(bytes_in, words);
    aecct::u8_t bytes_out[aecct::BACKUP_WORD_BYTES];
    aecct::top_io8_serialize_sram_word(words, bytes_out);
    WAVE3_CHECK_WORD_BYTES_LOOP: for (uint32_t i = 0u; i < aecct::BACKUP_WORD_BYTES; ++i) {
        if ((uint32_t)bytes_out[i].to_uint() != i) {
            std::printf(
                "[wave3][FAIL] 16-byte word roundtrip mismatch idx=%u got=0x%02X expect=0x%02X\n",
                (unsigned)i,
                (unsigned)bytes_out[i].to_uint(),
                (unsigned)i);
            return 1;
        }
    }

    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data8_ch_t data_in;
    aecct::data8_ch_t data_out;

    ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_SOFT_RESET));
    aecct::ctrl_word_t rsp = wait_rsp_or_fail(ctrl_rsp, ctrl_cmd, data_in, data_out);
    if (aecct::unpack_ctrl_rsp_kind(rsp) != (uint8_t)aecct::RSP_DONE) {
        fail("SOFT_RESET rsp kind mismatch");
    }

    ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_SET_W_BASE));
    push_u32_le(data_in, (uint32_t)sram_map::PARAM_BASE_DEFAULT);
    rsp = wait_rsp_or_fail(ctrl_rsp, ctrl_cmd, data_in, data_out);
    if (aecct::unpack_ctrl_rsp_kind(rsp) != (uint8_t)aecct::RSP_OK ||
        aecct::unpack_ctrl_rsp_payload(rsp) != (uint8_t)aecct::OP_SET_W_BASE) {
        fail("OP_SET_W_BASE io8 path rsp mismatch");
    }

    ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_READ_MEM));
    push_u32_le(data_in, 0u);
    push_u32_le(data_in, 1u);
    rsp = wait_rsp_or_fail(ctrl_rsp, ctrl_cmd, data_in, data_out);
    if (aecct::unpack_ctrl_rsp_kind(rsp) != (uint8_t)aecct::RSP_DONE ||
        aecct::unpack_ctrl_rsp_payload(rsp) != (uint8_t)aecct::OP_READ_MEM) {
        fail("OP_READ_MEM io8 path rsp mismatch");
    }

    uint32_t out_word = 0u;
    WAVE3_COLLECT_READMEM_BYTES_LOOP: for (uint32_t i = 0u; i < 4u; ++i) {
        aecct::u8_t b = 0;
        if (!data_out.nb_read(b)) {
            fail("READ_MEM io8 data_out underflow");
        }
        out_word |= ((uint32_t)b.to_uint() << (i * 8u));
    }
    if (out_word != 0u) {
        fail("READ_MEM word[0] expected zero after SOFT_RESET");
    }

    std::printf("PASS: tb_backup_wave3_io8_boundary_smoke\n");
    return 0;
}
