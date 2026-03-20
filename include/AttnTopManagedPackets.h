#pragma once
// Provisional internal packet contract for AC bring-up.
// Freeze only after G-AC evidence.

#include <cstdint>

#include "AecctTypes.h"
#include "blocks/TernaryLiveQkvLeafKernelShapeConfig.h"

namespace aecct {

enum AttnTopManagedPacketKind : unsigned {
    ATTN_PKT_X = 0u,
    ATTN_PKT_WK = 1u,
    ATTN_PKT_WV = 2u,
    ATTN_PKT_K = 3u,
    ATTN_PKT_V = 4u,
    ATTN_PKT_WQ = 5u,
    ATTN_PKT_Q = 6u,
    ATTN_PKT_SCORE = 7u,
    ATTN_PKT_OUT = 8u
};

static const unsigned ATTN_TOP_MANAGED_TILE_WORDS = (unsigned)kQkvCtSupportedL0WkCols;
static_assert(ATTN_TOP_MANAGED_TILE_WORDS == (unsigned)kQkvCtSupportedL0WvCols, "WK/WV tile words must match");

// Active chain working tile contract (P11AP): 4x32b = 128b.
// Legacy packet tile words remain unchanged for backward-compatible helpers/tests.
static const unsigned ATTN_TOP_MANAGED_WORK_TILE_WORDS = 4u;
static const unsigned ATTN_TOP_MANAGED_WORK_TILE_BITS = (unsigned)(ATTN_TOP_MANAGED_WORK_TILE_WORDS * 32u);

enum AttnTopManagedPhaseId : unsigned {
    ATTN_PHASE_A = 0u,
    ATTN_PHASE_B = 1u,
    ATTN_PHASE_C = 2u
};

enum AttnTopManagedHeadGroupId : unsigned {
    ATTN_HEAD_GROUP_0 = 0u, // heads 0..3 -> rule1 -> one_ring_mask
    ATTN_HEAD_GROUP_1 = 1u  // heads 4..7 -> rule2 -> second_ring_mask
};

enum AttnTopManagedSubphaseId : unsigned {
    ATTN_SUBPHASE_QSRC = 0u,
    ATTN_SUBPHASE_WQ = 1u,
    ATTN_SUBPHASE_KVSCAN = 2u,
    ATTN_SUBPHASE_MASK = 3u,
    ATTN_SUBPHASE_WO = 4u,
    ATTN_SUBPHASE_OUT = 5u
};

static inline u16_t attn_phaseb_head_group_id_from_head_idx(uint32_t head_idx) {
    return (head_idx < 4u) ? (u16_t)ATTN_HEAD_GROUP_0 : (u16_t)ATTN_HEAD_GROUP_1;
}

static inline u16_t attn_phaseb_rule_id_from_head_group(u16_t head_group_id) {
    return (head_group_id == (u16_t)ATTN_HEAD_GROUP_0) ? (u16_t)1u : (u16_t)2u;
}

static inline uint32_t attn_top_managed_tile_count(uint32_t words, uint32_t tile_words) {
    if (tile_words == 0u) {
        return 0u;
    }
    return (words + tile_words - 1u) / tile_words;
}

static inline uint32_t attn_top_managed_tile_valid_words(
    uint32_t words,
    uint32_t tile_words,
    uint32_t tile_idx
) {
    if (tile_words == 0u) {
        return 0u;
    }
    const uint32_t tile_begin = tile_idx * tile_words;
    if (tile_begin >= words) {
        return 0u;
    }
    const uint32_t remain = words - tile_begin;
    return (remain < tile_words) ? remain : tile_words;
}

struct AttnTopManagedPacket {
    u16_t kind;
    u16_t token_idx;
    u16_t d_tile_idx;
    u16_t flags;
    u32_t inv_sw_bits;
    u32_t data[ATTN_TOP_MANAGED_TILE_WORDS];
};

// Active-chain tile packet with explicit phase/range metadata.
struct AttnTopManagedWorkPacket {
    u16_t kind;
    u16_t phase_id;
    u16_t subphase_id;
    u16_t head_group_id;
    u16_t token_idx;
    u16_t token_begin;
    u16_t token_end;
    u16_t d_tile_idx;
    u16_t tile_begin;
    u16_t tile_end;
    u16_t tile_valid_words;
    u16_t flags;
    u32_t inv_sw_bits;
    u32_t data[ATTN_TOP_MANAGED_WORK_TILE_WORDS];
};

static inline void attn_packet_clear(AttnTopManagedPacket& p) {
    p.kind = 0;
    p.token_idx = 0;
    p.d_tile_idx = 0;
    p.flags = 0;
    p.inv_sw_bits = 0;
    for (unsigned i = 0; i < ATTN_TOP_MANAGED_TILE_WORDS; ++i) {
        p.data[i] = 0;
    }
}

static inline void attn_work_packet_clear(AttnTopManagedWorkPacket& p) {
    p.kind = 0;
    p.phase_id = 0;
    p.subphase_id = 0;
    p.head_group_id = 0;
    p.token_idx = 0;
    p.token_begin = 0;
    p.token_end = 0;
    p.d_tile_idx = 0;
    p.tile_begin = 0;
    p.tile_end = 0;
    p.tile_valid_words = 0;
    p.flags = 0;
    p.inv_sw_bits = 0;
    for (unsigned i = 0; i < ATTN_TOP_MANAGED_WORK_TILE_WORDS; ++i) {
        p.data[i] = 0;
    }
}

} // namespace aecct
