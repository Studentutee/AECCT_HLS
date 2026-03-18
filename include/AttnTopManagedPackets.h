#pragma once
// Provisional internal packet contract for AC bring-up.
// Freeze only after G-AC evidence.

#include "AecctTypes.h"
#include "blocks/TernaryLiveQkvLeafKernelShapeConfig.h"

namespace aecct {

enum AttnTopManagedPacketKind : unsigned {
    ATTN_PKT_X = 0u,
    ATTN_PKT_WK = 1u,
    ATTN_PKT_WV = 2u,
    ATTN_PKT_K = 3u,
    ATTN_PKT_V = 4u
};

static const unsigned ATTN_TOP_MANAGED_TILE_WORDS = (unsigned)kQkvCtSupportedL0WkCols;
static_assert(ATTN_TOP_MANAGED_TILE_WORDS == (unsigned)kQkvCtSupportedL0WvCols, "WK/WV tile words must match");

struct AttnTopManagedPacket {
    u16_t kind;
    u16_t token_idx;
    u16_t d_tile_idx;
    u16_t flags;
    u32_t inv_sw_bits;
    u32_t data[ATTN_TOP_MANAGED_TILE_WORDS];
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

} // namespace aecct
