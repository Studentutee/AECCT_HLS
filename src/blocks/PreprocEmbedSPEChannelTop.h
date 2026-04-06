#pragma once
// Preproc pilot numeric wrapper + Top-side stream adapters (local-only).
// This file implements channel transport and step0 numeric composition only.
// It does not change Top SRAM ownership or external Top contracts.

#include <cstdint>
#include <cstdio>

#include "AecctUtil.h"
#include "PreprocTransportTypes.h"

namespace aecct {

struct PreprocChannelPilotDebugControl {
    bool enable;
    u32_t sample_idx;
};

static PreprocChannelPilotDebugControl g_preproc_channel_pilot_debug_control = { false, (u32_t)0u };

static inline void preproc_channel_pilot_set_debug_context(bool enable, uint32_t sample_idx) {
    g_preproc_channel_pilot_debug_control.enable = enable;
    g_preproc_channel_pilot_debug_control.sample_idx = (u32_t)sample_idx;
}

static inline bool preproc_channel_pilot_debug_sample0_enabled() {
    return g_preproc_channel_pilot_debug_control.enable &&
           ((uint32_t)g_preproc_channel_pilot_debug_control.sample_idx.to_uint() == 0u);
}

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

// FP32-domain helpers for pilot numeric composition.
static inline fp32_t preproc_fp32_abs(fp32_t v) {
    if (v < fp32_zero()) {
        return fp32_zero() - v;
    }
    return v;
}

static inline fp32_t preproc_fp32_minus_one() {
    return fp32_from_bits((u32_t)0xBF800000u);
}

// Compose one token by ref-model step0 semantics:
// x[0:embed_words) = node_feature * src_embed, x[embed_words:embed_words+lpe_words) = lpe_token.
static inline void preproc_compose_x_packet_with_feature(
    const PreprocEmbedParamPacket& embed_pkt,
    const PreprocLpeTokenPacket& lpe_pkt,
    const fp32_t& node_feature,
    PreprocXOutPacket& out_pkt
) {
    preproc_x_out_packet_clear(out_pkt);
    out_pkt.token_kind = embed_pkt.token_kind;
    out_pkt.token_idx = embed_pkt.token_idx;

    const uint32_t embed_words = (uint32_t)embed_pkt.embed_word_count.to_uint();
    const uint32_t lpe_words = (uint32_t)lpe_pkt.lpe_word_count.to_uint();

    PREPROC_X_COMPOSE_LOOP: for (uint32_t d = 0u; d < PREPROC_PILOT_X_WORDS; ++d) {
        if (d < embed_words) {
            const fp32_t embed_fp = fp32_from_bits(embed_pkt.embed_words[d]);
            out_pkt.x_words[d] = bits_from_fp32(node_feature * embed_fp);
        } else {
            const uint32_t lpe_idx = d - embed_words;
            if (lpe_idx < lpe_words) {
                out_pkt.x_words[d] = lpe_pkt.lpe_words[lpe_idx];
            } else {
                out_pkt.x_words[d] = (u32_t)0u;
            }
        }
    }

    uint32_t word_count = embed_words + lpe_words;
    if (word_count > PREPROC_PILOT_X_WORDS) {
        word_count = PREPROC_PILOT_X_WORDS;
    }
    out_pkt.word_count = (u16_t)word_count;
}

// Legacy pack helper kept for bounded pilot compatibility; numeric path uses
// preproc_compose_x_packet_with_feature().
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

    // This wrapper is a pilot-local numeric path for Preproc step0.
    // It keeps Top-managed backing ownership and packet metadata authority.
    // It is not a closure path and does not change external Top contracts.

    u32_t check_tiles[PREPROC_PILOT_CHECK_TILE_COUNT][PREPROC_PILOT_CHECK_PARITY_TILE_WORDS];
    bool tile_loaded[PREPROC_PILOT_CHECK_TILE_COUNT];
    bool tile_dirty[PREPROC_PILOT_CHECK_TILE_COUNT];
    bool var_seen[PREPROC_PILOT_VAR_TOKENS];
    bool check_seen[PREPROC_PILOT_CHECK_TOKENS];
    bool debug_first_out_source_dumped = false;
    bool debug_first_token_ingredient_dumped = false;
    bool debug_first_check_update_dumped = false;
    const bool debug_sample0 = preproc_channel_pilot_debug_sample0_enabled();

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

    // Variable-side consume stage:
    // consume y/adj/embed/lpe, update check parity accumulator, then compose variable token.
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

        const fp32_t var_feature = preproc_fp32_abs(fp32_from_bits(y_pkt.y_bits));
        const uint32_t var_feature_bits = (uint32_t)bits_from_fp32(var_feature).to_uint();
        const uint32_t hard_bit = ((uint32_t)y_pkt.y_bits.to_uint() >> 31u) & 1u;
        const uint32_t adj_count = (uint32_t)adj_pkt.adj_count.to_uint();

        // Check-accumulator update stage (Top-backed parity tiles, local-only buffering).
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
#ifndef __SYNTHESIS__
                if (debug_sample0 && !debug_first_check_update_dumped) {
                    const uint32_t next = (uint32_t)check_tiles[tile_id][lane].to_uint();
                    std::printf(
                        "PREPROC_DEBUG_FIRST_CHECK_UPDATE var_idx=%u check_idx=%u tile_id=%u lane=%u hard_bit=%u\n",
                        (unsigned)pkt_var_idx,
                        (unsigned)check_idx,
                        (unsigned)tile_id,
                        (unsigned)lane,
                        (unsigned)hard_bit);
                    std::printf(
                        "PREPROC_DEBUG_FIRST_CHECK_UPDATE_VALUE before=0x%08X after=0x%08X\n",
                        (unsigned)curr,
                        (unsigned)next);
                    debug_first_check_update_dumped = true;
                }
#endif
            }
        }

        // Compose variable token by ref-model step0 semantics.
        PreprocXOutPacket out_pkt;
        preproc_compose_x_packet_with_feature(embed_pkt, lpe_pkt, var_feature, out_pkt);
#ifndef __SYNTHESIS__
        if (debug_sample0 && pkt_var_idx == 0u && !debug_first_token_ingredient_dumped) {
            PREPROC_DEBUG_FIRST_TOKEN_INGREDIENT_LOOP: for (uint32_t d = 0u; d < 8u; ++d) {
                const uint32_t embed_words = (uint32_t)embed_pkt.embed_word_count.to_uint();
                const uint32_t lpe_words = (uint32_t)lpe_pkt.lpe_word_count.to_uint();
                const uint32_t out_bits = (uint32_t)out_pkt.x_words[d].to_uint();
                if (d < embed_words) {
                    const uint32_t embed_bits = (uint32_t)embed_pkt.embed_words[d].to_uint();
                    const uint32_t lpe_bits = (d < lpe_words) ? (uint32_t)lpe_pkt.lpe_words[d].to_uint() : 0u;
                    std::printf(
                        "PREPROC_DEBUG_FIRST_TOKEN_INGREDIENT d=%u embed_u32=%u(0x%08X) lpe_u32=%u(0x%08X) varf_u32=%u(0x%08X) checkf=NA out_u32=%u(0x%08X)\n",
                        (unsigned)d,
                        (unsigned)embed_bits,
                        (unsigned)embed_bits,
                        (unsigned)lpe_bits,
                        (unsigned)lpe_bits,
                        (unsigned)var_feature_bits,
                        (unsigned)var_feature_bits,
                        (unsigned)out_bits,
                        (unsigned)out_bits);
                } else {
                    const uint32_t lpe_src = d - embed_words;
                    const uint32_t lpe_bits = (lpe_src < lpe_words) ? (uint32_t)lpe_pkt.lpe_words[lpe_src].to_uint() : 0u;
                    std::printf(
                        "PREPROC_DEBUG_FIRST_TOKEN_INGREDIENT d=%u embed=NA lpe_u32=%u(0x%08X) varf_u32=%u(0x%08X) checkf=NA out_u32=%u(0x%08X)\n",
                        (unsigned)d,
                        (unsigned)lpe_bits,
                        (unsigned)lpe_bits,
                        (unsigned)var_feature_bits,
                        (unsigned)var_feature_bits,
                        (unsigned)out_bits,
                        (unsigned)out_bits);
                }
            }
            debug_first_token_ingredient_dumped = true;
        }

        if (debug_sample0 && !debug_first_out_source_dumped) {
            PREPROC_DEBUG_FIRST_OUT_SOURCE_LOOP: for (uint32_t d = 0u; d < 8u; ++d) {
                const uint32_t embed_words = (uint32_t)embed_pkt.embed_word_count.to_uint();
                const uint32_t lpe_words = (uint32_t)lpe_pkt.lpe_word_count.to_uint();
                const uint32_t out_bits = (uint32_t)out_pkt.x_words[d].to_uint();
                if (d < embed_words) {
                    const uint32_t embed_bits = (uint32_t)embed_pkt.embed_words[d].to_uint();
                    std::printf(
                        "PREPROC_DEBUG_FIRST_OUT_SOURCE d=%u source=var_feature src_idx=%u varf_bits=0x%08X embed_bits=0x%08X out_u32=%u out_bits=0x%08X\n",
                        (unsigned)d,
                        (unsigned)d,
                        (unsigned)var_feature_bits,
                        (unsigned)embed_bits,
                        (unsigned)out_bits,
                        (unsigned)out_bits);
                } else {
                    const uint32_t lpe_src = d - embed_words;
                    if (lpe_src < lpe_words) {
                        std::printf(
                            "PREPROC_DEBUG_FIRST_OUT_SOURCE d=%u source=lpe src_idx=%u out_u32=%u out_bits=0x%08X\n",
                            (unsigned)d,
                            (unsigned)lpe_src,
                            (unsigned)out_bits,
                            (unsigned)out_bits);
                    } else {
                        std::printf(
                            "PREPROC_DEBUG_FIRST_OUT_SOURCE d=%u source=other src_idx=NA out_u32=%u out_bits=0x%08X\n",
                            (unsigned)d,
                            (unsigned)out_bits,
                            (unsigned)out_bits);
                    }
                }
            }
            debug_first_out_source_dumped = true;
        }
#endif
        preproc_x_out_ch.write(out_pkt);
        local_stats.var_tokens_consumed = local_stats.var_tokens_consumed + 1u;
    }

#ifndef __SYNTHESIS__
    if (debug_sample0 && !debug_first_check_update_dumped) {
        std::printf(
            "PREPROC_DEBUG_FIRST_CHECK_UPDATE var_idx=NA check_idx=NA tile_id=NA lane=NA hard_bit=0 note=no_update_observed\n");
        std::printf("PREPROC_DEBUG_FIRST_CHECK_UPDATE_VALUE before=NA after=NA\n");
    }
#endif

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

    // Check-side finalize stage:
    // after all y consumption, derive check_feature from parity tiles, then compose check token.
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

        const uint32_t tile_id = pkt_check_idx / PREPROC_PILOT_CHECK_PARITY_TILE_WORDS;
        const uint32_t lane = pkt_check_idx % PREPROC_PILOT_CHECK_PARITY_TILE_WORDS;
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
            PREPROC_CHECK_FINALIZE_TILE_RD_COPY_LOOP: for (uint32_t w = 0u; w < rd_word_count; ++w) {
                check_tiles[tile_id][w] = rd_pkt.acc_words[w];
            }
            tile_loaded[tile_id] = true;
            local_stats.check_tiles_loaded = local_stats.check_tiles_loaded + 1u;
        }
        const uint32_t parity_bit = (uint32_t)check_tiles[tile_id][lane].to_uint() & 1u;
        const fp32_t check_feature = (parity_bit == 0u) ? fp32_one() : preproc_fp32_minus_one();

        PreprocXOutPacket out_pkt;
        preproc_compose_x_packet_with_feature(embed_pkt, lpe_pkt, check_feature, out_pkt);
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
