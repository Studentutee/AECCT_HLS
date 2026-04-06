#pragma once
// Core AC datatypes and channel aliases.

#include <cstdint>

#include <ac_channel.h>
#include <ac_int.h>

namespace aecct {

typedef ac_int<8, false> u8_t;
typedef ac_int<16, false> u16_t;
typedef ac_int<16, true> s16_t;
typedef ac_int<32, false> u32_t;

typedef ac_channel<u16_t> ctrl_ch_t;
typedef ac_channel<u32_t> data_ch_t;
typedef ac_channel<u8_t> data8_ch_t;

// Backup profile IO/SRAM packing constants:
// - 1 lane = 16 bits
// - 1 word = 8 lanes = 16 bytes = 128 bits
static const uint32_t BACKUP_LANE_BITS = 16u;
static const uint32_t BACKUP_WORD_LANES = 8u;
static const uint32_t BACKUP_WORD_BYTES = 16u;
static const uint32_t BACKUP_WORD_U32S = 4u;

struct backup_word_lanes_t {
    u16_t lanes[BACKUP_WORD_LANES];
};

static inline u32_t u32_from_uint(uint32_t v) { return (u32_t)v; }
static inline uint32_t uint_from_u32(const u32_t& v) { return (uint32_t)v.to_uint(); }

static inline u16_t backup_pack_lane_from_bytes(u8_t lo, u8_t hi) {
    u16_t lane = 0;
    lane.set_slc(0, lo);
    lane.set_slc(8, hi);
    return lane;
}

static inline void backup_unpack_lane_to_bytes(const u16_t lane, u8_t& lo, u8_t& hi) {
    lo = lane.template slc<8>(0);
    hi = lane.template slc<8>(8);
}

static inline void backup_pack_word_lanes_from_bytes(
    const u8_t in_bytes[BACKUP_WORD_BYTES],
    backup_word_lanes_t& out_word
) {
    BACKUP_PACK_WORD_LANE_LOOP: for (uint32_t lane = 0u; lane < BACKUP_WORD_LANES; ++lane) {
        const uint32_t b = lane * 2u;
        out_word.lanes[lane] = backup_pack_lane_from_bytes(in_bytes[b + 0u], in_bytes[b + 1u]);
    }
}

static inline void backup_unpack_word_lanes_to_bytes(
    const backup_word_lanes_t& in_word,
    u8_t out_bytes[BACKUP_WORD_BYTES]
) {
    BACKUP_UNPACK_WORD_LANE_LOOP: for (uint32_t lane = 0u; lane < BACKUP_WORD_LANES; ++lane) {
        const uint32_t b = lane * 2u;
        backup_unpack_lane_to_bytes(in_word.lanes[lane], out_bytes[b + 0u], out_bytes[b + 1u]);
    }
}

static inline void backup_pack_word_u32x4_from_bytes(
    const u8_t in_bytes[BACKUP_WORD_BYTES],
    u32_t out_words[BACKUP_WORD_U32S]
) {
    BACKUP_PACK_U32_LOOP: for (uint32_t wi = 0u; wi < BACKUP_WORD_U32S; ++wi) {
        const uint32_t b = wi * 4u;
        u32_t w = 0;
        w.set_slc(0, in_bytes[b + 0u]);
        w.set_slc(8, in_bytes[b + 1u]);
        w.set_slc(16, in_bytes[b + 2u]);
        w.set_slc(24, in_bytes[b + 3u]);
        out_words[wi] = w;
    }
}

static inline void backup_unpack_word_u32x4_to_bytes(
    const u32_t in_words[BACKUP_WORD_U32S],
    u8_t out_bytes[BACKUP_WORD_BYTES]
) {
    BACKUP_UNPACK_U32_LOOP: for (uint32_t wi = 0u; wi < BACKUP_WORD_U32S; ++wi) {
        const uint32_t b = wi * 4u;
        const u32_t w = in_words[wi];
        out_bytes[b + 0u] = w.template slc<8>(0);
        out_bytes[b + 1u] = w.template slc<8>(8);
        out_bytes[b + 2u] = w.template slc<8>(16);
        out_bytes[b + 3u] = w.template slc<8>(24);
    }
}

} // namespace aecct
