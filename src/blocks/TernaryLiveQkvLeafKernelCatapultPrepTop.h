#pragma once
// P00-011R: dedicated Catapult-facing compile-prep top wrapper for single L0_WQ slice.

#include "AecctTypes.h"
#include "TernaryLiveQkvLeafKernel.h"

#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#endif

#ifndef CCS_BLOCK
#define CCS_BLOCK(name) name
#endif

namespace aecct {

#pragma hls_design top
class TernaryLiveL0WqRowTopCatapultPrep {
public:
    TernaryLiveL0WqRowTopCatapultPrep() {}

#pragma hls_design interface
    bool CCS_BLOCK(run)(
        const u32_t x_row[kTernaryLiveL0WqCols],
        const u32_t payload_words[kTernaryLiveL0WqPayloadWords],
        u32_t inv_sw_bits,
        u32_t out_row[kTernaryLiveL0WqRows],
        u32_t out_act_q_row[kTernaryLiveL0WqRows],
        u32_t& out_inv_sw_bits
    ) {
        return ternary_live_l0_wq_materialize_row_kernel_split(
            x_row,
            payload_words,
            inv_sw_bits,
            out_row,
            out_act_q_row,
            out_inv_sw_bits);
    }
};

} // namespace aecct
