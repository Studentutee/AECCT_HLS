#pragma once
// Top-managed window descriptors for the fp16 rewrite path.
//
// Mainline meaning of "window":
// - Top cuts a token/tile work slice from shared SRAM-owned state
// - Top streams only that slice to a block
// - the block does not own whole-SRAM arbitration or raw-pointer traversal
//
// Backup track may still allow direct-read inside a Top-assigned window,
// but that is not the default contract for the clean rewrite path.

#include <cstdint>

#include "AecctProtocol.h"
#include "AecctTypes.h"
#include "Fp16RewriteTypes.h"

namespace aecct {
namespace fp16_rewrite {

struct TokenWindow {
    u16_t begin;
    u16_t end;
};

struct TileWindow {
    u16_t begin;
    u16_t end;
    u16_t valid_words;
};

struct WindowContext {
    u16_t phase_id;
    u16_t subphase_id;
    u16_t layer_id;
    TokenWindow token;
    TileWindow tile;
    u32_t sram_base_word;
    u32_t sram_len_words;
};

// Packet for fp16 activation tiles moved between Top and a compute block.
// The payload width is fixed to one 128-bit SRAM beat = 8 fp16 storage words.
struct Fp16WindowPacket {
    WindowContext ctx;
    u16_t token_idx;
    u16_t tile_idx;
    u16_t flags;
    u16_t data[SRAM_WORDS_PER_BEAT];
};

typedef ac_channel<Fp16WindowPacket> fp16_window_ch_t;

static inline TokenWindow make_token_window(uint32_t begin, uint32_t end) {
    TokenWindow w;
    w.begin = (u16_t)begin;
    w.end = (u16_t)end;
    return w;
}

static inline TileWindow make_tile_window(uint32_t begin, uint32_t end, uint32_t valid_words) {
    TileWindow w;
    w.begin = (u16_t)begin;
    w.end = (u16_t)end;
    w.valid_words = (u16_t)valid_words;
    return w;
}

static inline WindowContext make_window_context(
    uint32_t phase_id,
    uint32_t subphase_id,
    uint32_t layer_id,
    uint32_t token_begin,
    uint32_t token_end,
    uint32_t tile_begin,
    uint32_t tile_end,
    uint32_t valid_words,
    uint32_t sram_base_word,
    uint32_t sram_len_words
) {
    WindowContext ctx;
    ctx.phase_id = (u16_t)phase_id;
    ctx.subphase_id = (u16_t)subphase_id;
    ctx.layer_id = (u16_t)layer_id;
    ctx.token = make_token_window(token_begin, token_end);
    ctx.tile = make_tile_window(tile_begin, tile_end, valid_words);
    ctx.sram_base_word = (u32_t)sram_base_word;
    ctx.sram_len_words = (u32_t)sram_len_words;
    return ctx;
}

static inline void clear_fp16_window_packet(Fp16WindowPacket& packet) {
    packet.ctx = make_window_context(
        (uint32_t)PHASE_PREPROC,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u
    );
    packet.token_idx = 0;
    packet.tile_idx = 0;
    packet.flags = 0;
    CLEAR_FP16_WINDOW_PACKET_LOOP: for (uint32_t i = 0u; i < SRAM_WORDS_PER_BEAT; ++i) {
        packet.data[i] = 0;
    }
}

} // namespace fp16_rewrite
} // namespace aecct
