#pragma once
// Weight-source abstraction for the fp16 clean rewrite.
//
// Why this header exists:
// - the first rewrite stage may still reuse data/weights/weights.h
// - compute blocks should not directly include weight dumps forever
// - later stages can swap the backend to Top-loaded W_REGION without
//   rewriting the compute math again
//
// This is a preproc-focused provider for round 1.
// Transformer/FinalHead accessors will be added when those blocks move over.

#include <cstdint>

#include "Fp16RewriteTypes.h"
#include "gen/ModelShapes.h"

namespace aecct {
namespace fp16_rewrite {

class Fp16PreprocWeightProvider {
public:
    virtual ~Fp16PreprocWeightProvider() {}

    virtual fp16_t src_embed(uint32_t node_idx, uint32_t dim_idx) const = 0;
    virtual fp16_t lpe_token(uint32_t node_idx, uint32_t dim_idx) const = 0;
};

// Header-backed adapter.
//
// This backend is only a migration bridge. It preserves the original
// weight dump as a convenient source of truth while the Top-owned W_REGION
// load path is still under construction.
class HeaderFp16PreprocWeightProvider : public Fp16PreprocWeightProvider {
public:
    HeaderFp16PreprocWeightProvider() {}

    fp16_t src_embed(uint32_t node_idx, uint32_t dim_idx) const override;
    fp16_t lpe_token(uint32_t node_idx, uint32_t dim_idx) const override;
};

} // namespace fp16_rewrite
} // namespace aecct

#include "weights.h"

namespace aecct {
namespace fp16_rewrite {

static inline uint32_t preproc_src_embed_offset(uint32_t node_idx, uint32_t dim_idx) {
    return node_idx * D_SRC_EMBED + dim_idx;
}

static inline uint32_t preproc_lpe_token_offset(uint32_t node_idx, uint32_t dim_idx) {
    return node_idx * D_LPE_TOKEN + dim_idx;
}

inline fp16_t HeaderFp16PreprocWeightProvider::src_embed(uint32_t node_idx, uint32_t dim_idx) const {
    const uint32_t offset = preproc_src_embed_offset(node_idx, dim_idx);
    return fp16_from_double(w_src_embed[offset]);
}

inline fp16_t HeaderFp16PreprocWeightProvider::lpe_token(uint32_t node_idx, uint32_t dim_idx) const {
    const uint32_t offset = preproc_lpe_token_offset(node_idx, dim_idx);
    return fp16_from_double(w_lpe_token[offset]);
}

} // namespace fp16_rewrite
} // namespace aecct
