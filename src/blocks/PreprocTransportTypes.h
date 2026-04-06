#pragma once
// Preproc channelized transport packet contracts (pilot local-only).
// Top remains the only shared-SRAM owner; this file only defines stream payloads.

#include <ac_channel.h>
#include <cstdint>

#include "AecctTypes.h"
#include "gen/ModelShapes.h"

namespace aecct {

static const uint32_t PREPROC_PILOT_CHECK_PARITY_TILE_WORDS = 4u;
static const uint32_t PREPROC_PILOT_MAX_TOKENS = (uint32_t)N_NODES;
static const uint32_t PREPROC_PILOT_VAR_TOKENS = (uint32_t)CODE_N;
static const uint32_t PREPROC_PILOT_CHECK_TOKENS = (uint32_t)CODE_C;
static const uint32_t PREPROC_PILOT_X_WORDS = (uint32_t)D_MODEL;
static const uint32_t PREPROC_PILOT_EMBED_WORDS = (uint32_t)D_SRC_EMBED;
static const uint32_t PREPROC_PILOT_LPE_WORDS = (uint32_t)D_LPE_TOKEN;
static const uint32_t PREPROC_PILOT_MAX_ADJ_PER_VAR = (uint32_t)CODE_C;
static const uint32_t PREPROC_PILOT_CHECK_TILE_COUNT =
    ((uint32_t)CODE_C + PREPROC_PILOT_CHECK_PARITY_TILE_WORDS - 1u) /
    PREPROC_PILOT_CHECK_PARITY_TILE_WORDS;

static inline uint32_t preproc_pilot_check_tile_count() {
    return PREPROC_PILOT_CHECK_TILE_COUNT;
}

enum PreprocPilotTokenKind : unsigned {
    PREPROC_PILOT_TOKEN_VAR = 0u,
    PREPROC_PILOT_TOKEN_CHECK = 1u
};

struct PreprocYInPacket {
    u16_t var_idx;
    u16_t reserved;
    u32_t y_bits;
};

struct PreprocHByVarAdjPacket {
    u16_t var_idx;
    u16_t adj_count;
    u16_t check_idx_list[PREPROC_PILOT_MAX_ADJ_PER_VAR];
};

struct PreprocEmbedParamPacket {
    u16_t token_kind;
    u16_t token_idx;
    u16_t embed_word_count;
    u16_t reserved;
    u32_t embed_words[PREPROC_PILOT_EMBED_WORDS];
};

struct PreprocLpeTokenPacket {
    u16_t token_kind;
    u16_t token_idx;
    u16_t lpe_word_count;
    u16_t reserved;
    u32_t lpe_words[PREPROC_PILOT_LPE_WORDS];
};

struct PreprocCheckAccReadPacket {
    u16_t tile_id;
    u16_t word_count;
    u32_t acc_words[PREPROC_PILOT_CHECK_PARITY_TILE_WORDS];
};

struct PreprocCheckAccWritePacket {
    u16_t tile_id;
    u16_t word_count;
    u32_t acc_words[PREPROC_PILOT_CHECK_PARITY_TILE_WORDS];
};

struct PreprocXOutPacket {
    u16_t token_kind;
    u16_t token_idx;
    u16_t word_count;
    u16_t reserved;
    u32_t x_words[PREPROC_PILOT_X_WORDS];
};

struct PreprocChannelPilotStats {
    u32_t var_tokens_consumed;
    u32_t check_tokens_emitted;
    u32_t check_tiles_loaded;
    u32_t check_tiles_written;
    bool metadata_error;
};

typedef ac_channel<PreprocYInPacket> preproc_y_in_ch_t;
typedef ac_channel<PreprocHByVarAdjPacket> preproc_h_by_var_adj_ch_t;
typedef ac_channel<PreprocEmbedParamPacket> preproc_embed_param_ch_t;
typedef ac_channel<PreprocLpeTokenPacket> preproc_lpe_token_ch_t;
typedef ac_channel<PreprocCheckAccReadPacket> preproc_check_acc_rd_ch_t;
typedef ac_channel<PreprocCheckAccWritePacket> preproc_check_acc_wr_ch_t;
typedef ac_channel<PreprocXOutPacket> preproc_x_out_ch_t;

static inline void preproc_channel_pilot_stats_clear(PreprocChannelPilotStats& s) {
    s.var_tokens_consumed = (u32_t)0u;
    s.check_tokens_emitted = (u32_t)0u;
    s.check_tiles_loaded = (u32_t)0u;
    s.check_tiles_written = (u32_t)0u;
    s.metadata_error = false;
}

static inline void preproc_y_in_packet_clear(PreprocYInPacket& p) {
    p.var_idx = (u16_t)0u;
    p.reserved = (u16_t)0u;
    p.y_bits = (u32_t)0u;
}

static inline void preproc_h_by_var_adj_packet_clear(PreprocHByVarAdjPacket& p) {
    p.var_idx = (u16_t)0u;
    p.adj_count = (u16_t)0u;
    PREPROC_H_ADJ_CLEAR_LOOP: for (uint32_t i = 0u; i < PREPROC_PILOT_MAX_ADJ_PER_VAR; ++i) {
        p.check_idx_list[i] = (u16_t)0u;
    }
}

static inline void preproc_embed_param_packet_clear(PreprocEmbedParamPacket& p) {
    p.token_kind = (u16_t)0u;
    p.token_idx = (u16_t)0u;
    p.embed_word_count = (u16_t)0u;
    p.reserved = (u16_t)0u;
    PREPROC_EMBED_CLEAR_LOOP: for (uint32_t i = 0u; i < PREPROC_PILOT_EMBED_WORDS; ++i) {
        p.embed_words[i] = (u32_t)0u;
    }
}

static inline void preproc_lpe_token_packet_clear(PreprocLpeTokenPacket& p) {
    p.token_kind = (u16_t)0u;
    p.token_idx = (u16_t)0u;
    p.lpe_word_count = (u16_t)0u;
    p.reserved = (u16_t)0u;
    PREPROC_LPE_CLEAR_LOOP: for (uint32_t i = 0u; i < PREPROC_PILOT_LPE_WORDS; ++i) {
        p.lpe_words[i] = (u32_t)0u;
    }
}

static inline void preproc_check_acc_read_packet_clear(PreprocCheckAccReadPacket& p) {
    p.tile_id = (u16_t)0u;
    p.word_count = (u16_t)0u;
    PREPROC_CHECK_RD_CLEAR_LOOP: for (uint32_t i = 0u; i < PREPROC_PILOT_CHECK_PARITY_TILE_WORDS; ++i) {
        p.acc_words[i] = (u32_t)0u;
    }
}

static inline void preproc_check_acc_write_packet_clear(PreprocCheckAccWritePacket& p) {
    p.tile_id = (u16_t)0u;
    p.word_count = (u16_t)0u;
    PREPROC_CHECK_WR_CLEAR_LOOP: for (uint32_t i = 0u; i < PREPROC_PILOT_CHECK_PARITY_TILE_WORDS; ++i) {
        p.acc_words[i] = (u32_t)0u;
    }
}

static inline void preproc_x_out_packet_clear(PreprocXOutPacket& p) {
    p.token_kind = (u16_t)0u;
    p.token_idx = (u16_t)0u;
    p.word_count = (u16_t)0u;
    p.reserved = (u16_t)0u;
    PREPROC_X_CLEAR_LOOP: for (uint32_t i = 0u; i < PREPROC_PILOT_X_WORDS; ++i) {
        p.x_words[i] = (u32_t)0u;
    }
}

} // namespace aecct
