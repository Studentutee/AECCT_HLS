#pragma once
// Preproc channelized pilot wrapper and Top-side stream adapters (local-only pilot).

#include <cstdint>

#include "PreprocTransportTypes.h"

namespace aecct {

static inline bool preproc_embed_packet_meta_ok(
    const PreprocEmbedParamPacket& p,
    uint32_t expect_kind,
    uint32_t expect_idx
) {
    if ((uint32_t)p.token_kind.to_uint() != expect_kind) { return false; }
    if ((uint32_t)p.token_idx.to_uint() != expect_idx) { return false; }
    const uint32_t words = (uint32_t)p.embed_word_count.to_uint();
    return words <= PREPROC_PILOT_EMBED_WORDS;
}

static inline bool preproc_lpe_packet_meta_ok(
    const PreprocLpeTokenPacket& p,
    uint32_t expect_kind,
    uint32_t expect_idx
) {
    if ((uint32_t)p.token_kind.to_uint() != expect_kind) { return false; }
    if ((uint32_t)p.token_idx.to_uint() != expect_idx) { return false; }
    const uint32_t words = (uint32_t)p.lpe_word_count.to_uint();
    return words <= PREPROC_PILOT_LPE_WORDS;
}

static inline bool preproc_adj_packet_meta_ok(
    const PreprocHByVarAdjPacket& p,
    uint32_t expect_var_idx
) {
    if ((uint32_t)p.var_idx.to_uint() != expect_var_idx) { return false; }
    const uint32_t count = (uint32_t)p.adj_count.to_uint();
    if (count > PREPROC_PILOT_MAX_ADJ_PER_VAR) { return false; }
    PREPROC_ADJ_META_LOOP: for (uint32_t i = 0u; i < count; ++i) {
        const uint32_t c = (uint32_t)p.check_idx_list[i].to_uint();
        if (c >= PREPROC_PILOT_CHECK_TOKENS) { return false; }
    }
    return true;
}

static inline void preproc_compose_x_packet(
    const PreprocEmbedParamPacket& embed_pkt,
    const PreprocLpeTokenPacket& lpe_pkt,
    PreprocXOutPacket& out_pkt
) {
    preproc_x_out_packet_clear(out_pkt);
    out_pkt.token_kind = embed_pkt.token_kind;
    out_pkt.token_idx = embed_pkt.token_idx;

    uint32_t cursor = 0u;
    const uint32_t embed_words = (uint32_t)embed_pkt.embed_word_count.to_uint();
    const uint32_t lpe_words = (uint32_t)lpe_pkt.lpe_word_count.to_uint();

    PREPROC_X_EMBED_COPY_LOOP: for (uint32_t i = 0u; i < embed_words; ++i) {
        if (cursor >= PREPROC_PILOT_X_WORDS) { break; }
        out_pkt.x_words[cursor] = embed_pkt.embed_words[i];
        ++cursor;
    }

    PREPROC_X_LPE_COPY_LOOP: for (uint32_t i = 0u; i < lpe_words; ++i) {
        if (cursor >= PREPROC_PILOT_X_WORDS) { break; }
        out_pkt.x_words[cursor] = lpe_pkt.lpe_words[i];
        ++cursor;
    }

    out_pkt.word_count = (u16_t)cursor;
}

static inline bool preproc_embed_spe_channel_top(
    preproc_y_in_ch_t& y_in_ch,
    preproc_h_by_var_adj_ch_t& h_by_var_adj_ch,
    preproc_embed_param_ch_t& embed_param_ch,
    preproc_lpe_token_ch_t& lpe_token_ch,
    preproc_check_acc_rd_ch_t& check_acc_rd_ch,
    preproc_check_acc_wr_ch_t& check_acc_wr_ch,
    preproc_x_out_ch_t& preproc_x_out_ch,
    PreprocChannelPilotStats* out_stats = 0
) {
    PreprocChannelPilotStats local_stats;
    preproc_channel_pilot_stats_clear(local_stats);

    u32_t check_tiles[PREPROC_PILOT_CHECK_TILE_COUNT][PREPROC_PILOT_CHECK_PARITY_TILE_WORDS];
    bool tile_loaded[PREPROC_PILOT_CHECK_TILE_COUNT];
    bool tile_dirty[PREPROC_PILOT_CHECK_TILE_COUNT];
    bool var_seen[PREPROC_PILOT_VAR_TOKENS];
    bool check_seen[PREPROC_PILOT_CHECK_TOKENS];

    PREPROC_CHECK_TILE_INIT_LOOP: for (uint32_t t = 0u; t < PREPROC_PILOT_CHECK_TILE_COUNT; ++t) {
        tile_loaded[t] = false;
        tile_dirty[t] = false;
        PREPROC_CHECK_TILE_WORD_INIT_LOOP: for (uint32_t w = 0u; w < PREPROC_PILOT_CHECK_PARITY_TILE_WORDS; ++w) {
            check_tiles[t][w] = (u32_t)0u;
        }
    }
    PREPROC_VAR_SEEN_INIT_LOOP: for (uint32_t i = 0u; i < PREPROC_PILOT_VAR_TOKENS; ++i) {
        var_seen[i] = false;
    }
    PREPROC_CHECK_SEEN_INIT_LOOP: for (uint32_t i = 0u; i < PREPROC_PILOT_CHECK_TOKENS; ++i) {
        check_seen[i] = false;
    }

    // var_iter is bounded iteration only; formal var identity comes from packet metadata.
    PREPROC_VAR_TOKEN_LOOP: for (uint32_t var_iter = 0u; var_iter < PREPROC_PILOT_VAR_TOKENS; ++var_iter) {
        const PreprocYInPacket y_pkt = y_in_ch.read();
        const PreprocHByVarAdjPacket adj_pkt = h_by_var_adj_ch.read();
        const PreprocEmbedParamPacket embed_pkt = embed_param_ch.read();
        const PreprocLpeTokenPacket lpe_pkt = lpe_token_ch.read();

        const uint32_t pkt_var_idx = (uint32_t)y_pkt.var_idx.to_uint();
        if (pkt_var_idx >= PREPROC_PILOT_VAR_TOKENS) {
            local_stats.metadata_error = true;
            if (out_stats != 0) { *out_stats = local_stats; }
            return false;
        }
        if (var_seen[pkt_var_idx]) {
            local_stats.metadata_error = true;
            if (out_stats != 0) { *out_stats = local_stats; }
            return false;
        }
        if (!preproc_adj_packet_meta_ok(adj_pkt, pkt_var_idx)) {
            local_stats.metadata_error = true;
            if (out_stats != 0) { *out_stats = local_stats; }
            return false;
        }
        if (!preproc_embed_packet_meta_ok(embed_pkt, (uint32_t)PREPROC_PILOT_TOKEN_VAR, pkt_var_idx)) {
            local_stats.metadata_error = true;
            if (out_stats != 0) { *out_stats = local_stats; }
            return false;
        }
        if (!preproc_lpe_packet_meta_ok(lpe_pkt, (uint32_t)PREPROC_PILOT_TOKEN_VAR, pkt_var_idx)) {
            local_stats.metadata_error = true;
            if (out_stats != 0) { *out_stats = local_stats; }
            return false;
        }
        var_seen[pkt_var_idx] = true;

        const uint32_t hard_bit = ((uint32_t)y_pkt.y_bits.to_uint() >> 31u) & 1u;
        const uint32_t adj_count = (uint32_t)adj_pkt.adj_count.to_uint();

        PREPROC_VAR_ADJ_LOOP: for (uint32_t i = 0u; i < adj_count; ++i) {
            const uint32_t check_idx = (uint32_t)adj_pkt.check_idx_list[i].to_uint();
            const uint32_t tile_id = check_idx / PREPROC_PILOT_CHECK_PARITY_TILE_WORDS;
            const uint32_t lane = check_idx % PREPROC_PILOT_CHECK_PARITY_TILE_WORDS;
            if (tile_id >= PREPROC_PILOT_CHECK_TILE_COUNT) {
                local_stats.metadata_error = true;
                if (out_stats != 0) { *out_stats = local_stats; }
                return false;
            }

            if (!tile_loaded[tile_id]) {
                const PreprocCheckAccReadPacket rd_pkt = check_acc_rd_ch.read();
                const uint32_t rd_tile_id = (uint32_t)rd_pkt.tile_id.to_uint();
                const uint32_t rd_word_count = (uint32_t)rd_pkt.word_count.to_uint();
                if (rd_tile_id != tile_id || rd_word_count > PREPROC_PILOT_CHECK_PARITY_TILE_WORDS) {
                    local_stats.metadata_error = true;
                    if (out_stats != 0) { *out_stats = local_stats; }
                    return false;
                }
                PREPROC_TILE_RD_COPY_LOOP: for (uint32_t w = 0u; w < rd_word_count; ++w) {
                    check_tiles[tile_id][w] = rd_pkt.acc_words[w];
                }
                tile_loaded[tile_id] = true;
                local_stats.check_tiles_loaded = local_stats.check_tiles_loaded + 1u;
            }

            if (hard_bit != 0u) {
                const uint32_t curr = (uint32_t)check_tiles[tile_id][lane].to_uint();
                check_tiles[tile_id][lane] = (u32_t)(curr ^ 1u);
                tile_dirty[tile_id] = true;
            }
        }

        PreprocXOutPacket out_pkt;
        preproc_compose_x_packet(embed_pkt, lpe_pkt, out_pkt);
        preproc_x_out_ch.write(out_pkt);
        local_stats.var_tokens_consumed = local_stats.var_tokens_consumed + 1u;
    }

    PREPROC_CHECK_TILE_WRITEBACK_LOOP: for (uint32_t tile_id = 0u; tile_id < PREPROC_PILOT_CHECK_TILE_COUNT; ++tile_id) {
        if (!tile_loaded[tile_id] || !tile_dirty[tile_id]) {
            continue;
        }
        PreprocCheckAccWritePacket wr_pkt;
        preproc_check_acc_write_packet_clear(wr_pkt);
        wr_pkt.tile_id = (u16_t)tile_id;
        wr_pkt.word_count = (u16_t)PREPROC_PILOT_CHECK_PARITY_TILE_WORDS;
        PREPROC_TILE_WR_COPY_LOOP: for (uint32_t w = 0u; w < PREPROC_PILOT_CHECK_PARITY_TILE_WORDS; ++w) {
            wr_pkt.acc_words[w] = check_tiles[tile_id][w];
        }
        check_acc_wr_ch.write(wr_pkt);
        local_stats.check_tiles_written = local_stats.check_tiles_written + 1u;
    }

    // check_iter is bounded iteration only; formal check identity comes from packet metadata.
    PREPROC_CHECK_TOKEN_LOOP: for (uint32_t check_iter = 0u; check_iter < PREPROC_PILOT_CHECK_TOKENS; ++check_iter) {
        const PreprocEmbedParamPacket embed_pkt = embed_param_ch.read();
        const PreprocLpeTokenPacket lpe_pkt = lpe_token_ch.read();
        const uint32_t pkt_check_idx = (uint32_t)embed_pkt.token_idx.to_uint();

        if (pkt_check_idx >= PREPROC_PILOT_CHECK_TOKENS) {
            local_stats.metadata_error = true;
            if (out_stats != 0) { *out_stats = local_stats; }
            return false;
        }
        if (check_seen[pkt_check_idx]) {
            local_stats.metadata_error = true;
            if (out_stats != 0) { *out_stats = local_stats; }
            return false;
        }
        if (!preproc_embed_packet_meta_ok(embed_pkt, (uint32_t)PREPROC_PILOT_TOKEN_CHECK, pkt_check_idx)) {
            local_stats.metadata_error = true;
            if (out_stats != 0) { *out_stats = local_stats; }
            return false;
        }
        if (!preproc_lpe_packet_meta_ok(lpe_pkt, (uint32_t)PREPROC_PILOT_TOKEN_CHECK, pkt_check_idx)) {
            local_stats.metadata_error = true;
            if (out_stats != 0) { *out_stats = local_stats; }
            return false;
        }
        check_seen[pkt_check_idx] = true;

        PreprocXOutPacket out_pkt;
        preproc_compose_x_packet(embed_pkt, lpe_pkt, out_pkt);
        preproc_x_out_ch.write(out_pkt);
        local_stats.check_tokens_emitted = local_stats.check_tokens_emitted + 1u;
    }

    if (out_stats != 0) {
        *out_stats = local_stats;
    }
    return true;
}

// Top-side helper: seed check accumulator read channel from Top-managed backing words.
static inline void preproc_top_seed_check_acc_rd_from_backing(
    const u32_t* backing_words,
    preproc_check_acc_rd_ch_t& check_acc_rd_ch
) {
    PREPROC_TOP_SEED_CHECK_RD_LOOP: for (uint32_t tile_id = 0u; tile_id < PREPROC_PILOT_CHECK_TILE_COUNT; ++tile_id) {
        PreprocCheckAccReadPacket rd_pkt;
        preproc_check_acc_read_packet_clear(rd_pkt);
        rd_pkt.tile_id = (u16_t)tile_id;
        rd_pkt.word_count = (u16_t)PREPROC_PILOT_CHECK_PARITY_TILE_WORDS;
        PREPROC_TOP_SEED_CHECK_RD_WORD_LOOP: for (uint32_t w = 0u; w < PREPROC_PILOT_CHECK_PARITY_TILE_WORDS; ++w) {
            rd_pkt.acc_words[w] = backing_words[tile_id * PREPROC_PILOT_CHECK_PARITY_TILE_WORDS + w];
        }
        check_acc_rd_ch.write(rd_pkt);
    }
}

// Top-side helper: commit write-back packets to Top-managed backing words.
static inline uint32_t preproc_top_drain_check_acc_wr_to_backing(
    preproc_check_acc_wr_ch_t& check_acc_wr_ch,
    u32_t* backing_words
) {
    uint32_t writes = 0u;
    PreprocCheckAccWritePacket wr_pkt;
    while (check_acc_wr_ch.nb_read(wr_pkt)) {
        const uint32_t tile_id = (uint32_t)wr_pkt.tile_id.to_uint();
        const uint32_t count = (uint32_t)wr_pkt.word_count.to_uint();
        if (tile_id >= PREPROC_PILOT_CHECK_TILE_COUNT) {
            continue;
        }
        const uint32_t clipped =
            (count > PREPROC_PILOT_CHECK_PARITY_TILE_WORDS) ?
            PREPROC_PILOT_CHECK_PARITY_TILE_WORDS : count;
        PREPROC_TOP_DRAIN_CHECK_WR_WORD_LOOP: for (uint32_t w = 0u; w < clipped; ++w) {
            backing_words[tile_id * PREPROC_PILOT_CHECK_PARITY_TILE_WORDS + w] = wr_pkt.acc_words[w];
        }
        ++writes;
    }
    return writes;
}

// Top-side helper: emit variable-major y stream.
static inline void preproc_top_emit_y_var_major(
    const u32_t* input_y_words,
    preproc_y_in_ch_t& y_in_ch
) {
    PREPROC_TOP_EMIT_Y_LOOP: for (uint32_t v = 0u; v < PREPROC_PILOT_VAR_TOKENS; ++v) {
        PreprocYInPacket y_pkt;
        preproc_y_in_packet_clear(y_pkt);
        y_pkt.var_idx = (u16_t)v;
        y_pkt.y_bits = input_y_words[v];
        y_in_ch.write(y_pkt);
    }
}

// Top-side helper: emit adjacency-by-variable packets from row-major H[check][var].
static inline void preproc_top_emit_h_by_var_adj_from_h_matrix(
    const ac_int<1, false>* h_bits_row_major,
    preproc_h_by_var_adj_ch_t& h_by_var_adj_ch
) {
    PREPROC_TOP_EMIT_ADJ_VAR_LOOP: for (uint32_t v = 0u; v < PREPROC_PILOT_VAR_TOKENS; ++v) {
        PreprocHByVarAdjPacket adj_pkt;
        preproc_h_by_var_adj_packet_clear(adj_pkt);
        adj_pkt.var_idx = (u16_t)v;
        uint32_t count = 0u;
        PREPROC_TOP_EMIT_ADJ_CHECK_LOOP: for (uint32_t c = 0u; c < PREPROC_PILOT_CHECK_TOKENS; ++c) {
            const uint32_t flat = c * PREPROC_PILOT_VAR_TOKENS + v;
            if ((uint32_t)h_bits_row_major[flat].to_uint() == 0u) {
                continue;
            }
            if (count < PREPROC_PILOT_MAX_ADJ_PER_VAR) {
                adj_pkt.check_idx_list[count] = (u16_t)c;
                ++count;
            }
        }
        adj_pkt.adj_count = (u16_t)count;
        h_by_var_adj_ch.write(adj_pkt);
    }
}

// Top-side helper: emit embed/lpe token payloads in required output order.
static inline void preproc_top_emit_embed_lpe_payloads(
    const u32_t* embed_words_token_major,
    const u32_t* lpe_words_token_major,
    preproc_embed_param_ch_t& embed_param_ch,
    preproc_lpe_token_ch_t& lpe_token_ch
) {
    PREPROC_TOP_EMIT_TOKEN_LOOP: for (uint32_t token = 0u; token < PREPROC_PILOT_MAX_TOKENS; ++token) {
        const uint32_t token_kind = (token < PREPROC_PILOT_VAR_TOKENS) ?
            (uint32_t)PREPROC_PILOT_TOKEN_VAR :
            (uint32_t)PREPROC_PILOT_TOKEN_CHECK;
        const uint32_t token_idx = (token < PREPROC_PILOT_VAR_TOKENS) ?
            token : (token - PREPROC_PILOT_VAR_TOKENS);

        PreprocEmbedParamPacket embed_pkt;
        preproc_embed_param_packet_clear(embed_pkt);
        embed_pkt.token_kind = (u16_t)token_kind;
        embed_pkt.token_idx = (u16_t)token_idx;
        embed_pkt.embed_word_count = (u16_t)PREPROC_PILOT_EMBED_WORDS;
        PREPROC_TOP_EMIT_EMBED_WORD_LOOP: for (uint32_t i = 0u; i < PREPROC_PILOT_EMBED_WORDS; ++i) {
            embed_pkt.embed_words[i] =
                embed_words_token_major[token * PREPROC_PILOT_EMBED_WORDS + i];
        }
        embed_param_ch.write(embed_pkt);

        PreprocLpeTokenPacket lpe_pkt;
        preproc_lpe_token_packet_clear(lpe_pkt);
        lpe_pkt.token_kind = (u16_t)token_kind;
        lpe_pkt.token_idx = (u16_t)token_idx;
        lpe_pkt.lpe_word_count = (u16_t)PREPROC_PILOT_LPE_WORDS;
        PREPROC_TOP_EMIT_LPE_WORD_LOOP: for (uint32_t i = 0u; i < PREPROC_PILOT_LPE_WORDS; ++i) {
            lpe_pkt.lpe_words[i] =
                lpe_words_token_major[token * PREPROC_PILOT_LPE_WORDS + i];
        }
        lpe_token_ch.write(lpe_pkt);
    }
}

// Top-side helper: drain full preproc_x token stream into Top-owned memory.
static inline uint32_t preproc_top_drain_x_out_to_memory(
    preproc_x_out_ch_t& preproc_x_out_ch,
    u32_t* x_out_words_token_major,
    uint32_t token_capacity
) {
    uint32_t count = 0u;
    PreprocXOutPacket out_pkt;
    while (preproc_x_out_ch.nb_read(out_pkt)) {
        const uint32_t token_idx = (uint32_t)out_pkt.token_idx.to_uint();
        const uint32_t token_kind = (uint32_t)out_pkt.token_kind.to_uint();
        uint32_t token_order = token_idx;
        if (token_kind == (uint32_t)PREPROC_PILOT_TOKEN_CHECK) {
            token_order = PREPROC_PILOT_VAR_TOKENS + token_idx;
        }
        if (token_order >= token_capacity) {
            continue;
        }
        const uint32_t words = (uint32_t)out_pkt.word_count.to_uint();
        const uint32_t clipped = (words > PREPROC_PILOT_X_WORDS) ? PREPROC_PILOT_X_WORDS : words;
        PREPROC_TOP_DRAIN_X_WORD_LOOP: for (uint32_t i = 0u; i < clipped; ++i) {
            x_out_words_token_major[token_order * PREPROC_PILOT_X_WORDS + i] = out_pkt.x_words[i];
        }
        ++count;
    }
    return count;
}

} // namespace aecct
