#pragma once
// Minimal Top-facing rewrite contract.
//
// This header does not replace src/Top.h today.
// It records the agreed direction for the new clean chain:
// - external system interface stays at the existing Top boundary
// - Top remains the only shared-SRAM owner
// - internal compute blocks move toward Top-managed window streaming
// - early rewrite rounds may still use header-backed weights through a provider

#include <cstdint>

#include "AecctProtocol.h"
#include "AecctTypes.h"
#include "TopManagedWindowTypes.h"

namespace aecct {
namespace fp16_rewrite {

struct TopRewriteIoContract {
    // External wrapper path keeps the existing byte-stream integration shell.
    // Internal clean compute will still reason in io16 / 16-bit storage words.
    uint32_t ctrl_bits;
    uint32_t data_bits;
    uint32_t storage_word_bits;
    bool top_owns_sram;
};

static inline TopRewriteIoContract make_default_top_rewrite_io_contract() {
    TopRewriteIoContract c;
    c.ctrl_bits = 16u;
    c.data_bits = 8u;
    c.storage_word_bits = SRAM_STORAGE_WORD_BITS;
    c.top_owns_sram = true;
    return c;
}

} // namespace fp16_rewrite
} // namespace aecct
