// tb_backup_wave1_memory_packing_smoke.cpp
// Backup profile Wave1 smoke:
// - SRAM word = 8 lanes x 16-bit
// - verify lane order and byte order pack/unpack

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "AecctTypes.h"
#include "ModelShapes.h"

namespace {

static void fail(const char* msg) {
    std::printf("[wave1][FAIL] %s\n", msg);
    std::exit(1);
}

static void require_true(bool cond, const char* msg) {
    if (!cond) {
        fail(msg);
    }
}

} // namespace

int main() {
    require_true(W_LANES == 8u, "W_LANES must be 8 for backup profile");
    require_true(BYTES_PER_WORD == 16u, "BYTES_PER_WORD must be 16 for backup profile");
    require_true(SRAM_LANE_BITS == 16u, "SRAM_LANE_BITS must be 16 for backup profile");

    aecct::u8_t bytes_in[aecct::BACKUP_WORD_BYTES];
    WAVE1_INIT_BYTES_LOOP: for (uint32_t i = 0u; i < aecct::BACKUP_WORD_BYTES; ++i) {
        bytes_in[i] = (aecct::u8_t)i;
    }

    aecct::backup_word_lanes_t lanes;
    aecct::backup_pack_word_lanes_from_bytes(bytes_in, lanes);

    WAVE1_CHECK_LANE_ORDER_LOOP: for (uint32_t lane = 0u; lane < aecct::BACKUP_WORD_LANES; ++lane) {
        const uint32_t b = lane * 2u;
        const uint16_t expect = (uint16_t)((uint16_t)b | ((uint16_t)(b + 1u) << 8));
        const uint16_t got = (uint16_t)lanes.lanes[lane].to_uint();
        if (got != expect) {
            std::printf(
                "[wave1][FAIL] lane order mismatch lane=%u got=0x%04X expect=0x%04X\n",
                (unsigned)lane,
                (unsigned)got,
                (unsigned)expect);
            return 1;
        }
    }

    aecct::u8_t bytes_roundtrip[aecct::BACKUP_WORD_BYTES];
    aecct::backup_unpack_word_lanes_to_bytes(lanes, bytes_roundtrip);
    WAVE1_CHECK_BYTE_ROUNDTRIP_LOOP: for (uint32_t i = 0u; i < aecct::BACKUP_WORD_BYTES; ++i) {
        if ((uint32_t)bytes_roundtrip[i].to_uint() != i) {
            std::printf(
                "[wave1][FAIL] byte roundtrip mismatch idx=%u got=0x%02X expect=0x%02X\n",
                (unsigned)i,
                (unsigned)bytes_roundtrip[i].to_uint(),
                (unsigned)i);
            return 1;
        }
    }

    aecct::u32_t words[aecct::BACKUP_WORD_U32S];
    aecct::backup_pack_word_u32x4_from_bytes(bytes_in, words);
    aecct::u8_t bytes_from_words[aecct::BACKUP_WORD_BYTES];
    aecct::backup_unpack_word_u32x4_to_bytes(words, bytes_from_words);
    WAVE1_CHECK_U32_HELPER_LOOP: for (uint32_t i = 0u; i < aecct::BACKUP_WORD_BYTES; ++i) {
        if ((uint32_t)bytes_from_words[i].to_uint() != i) {
            std::printf(
                "[wave1][FAIL] u32 pack/unpack mismatch idx=%u got=0x%02X expect=0x%02X\n",
                (unsigned)i,
                (unsigned)bytes_from_words[i].to_uint(),
                (unsigned)i);
            return 1;
        }
    }

    std::printf("PASS: tb_backup_wave1_memory_packing_smoke\n");
    return 0;
}
