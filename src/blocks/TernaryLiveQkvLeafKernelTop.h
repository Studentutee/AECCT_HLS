#pragma once
// Local Catapult-friendly split-interface tops for live ternary L0_WQ/L0_WK/L0_WV.
// These wrappers only expose fixed-shape interfaces and delegate math to
// ternary_live_l0_w{q,k,v}_materialize_row_kernel_split().
// Non-ownership boundary: these tops do not define SRAM policy or runtime-variable shape behavior.

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
class TernaryLiveL0WqRowTop {
public:
    TernaryLiveL0WqRowTop() {}

#pragma hls_design interface
    // Input -> output: x_row + payload_words + inv_sw_bits -> out_row/out_act_q_row.
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

#pragma hls_design top
class TernaryLiveL0WkRowTop {
public:
    TernaryLiveL0WkRowTop() {}

#pragma hls_design interface
    // Input -> output: x_row + payload_words + inv_sw_bits -> out_row/out_act_q_row.
    bool CCS_BLOCK(run)(
        const u32_t x_row[kTernaryLiveL0WkCols],
        const u32_t payload_words[kTernaryLiveL0WkPayloadWords],
        u32_t inv_sw_bits,
        u32_t out_row[kTernaryLiveL0WkRows],
        u32_t out_act_q_row[kTernaryLiveL0WkRows],
        u32_t& out_inv_sw_bits
    ) {
        return ternary_live_l0_wk_materialize_row_kernel_split(
            x_row,
            payload_words,
            inv_sw_bits,
            out_row,
            out_act_q_row,
            out_inv_sw_bits);
    }
};

#pragma hls_design top
class TernaryLiveL0WvRowTop {
public:
    TernaryLiveL0WvRowTop() {}

#pragma hls_design interface
    // Input -> output: x_row + payload_words + inv_sw_bits -> out_row/out_act_q_row.
    bool CCS_BLOCK(run)(
        const u32_t x_row[kTernaryLiveL0WvCols],
        const u32_t payload_words[kTernaryLiveL0WvPayloadWords],
        u32_t inv_sw_bits,
        u32_t out_row[kTernaryLiveL0WvRows],
        u32_t out_act_q_row[kTernaryLiveL0WvRows],
        u32_t& out_inv_sw_bits
    ) {
        return ternary_live_l0_wv_materialize_row_kernel_split(
            x_row,
            payload_words,
            inv_sw_bits,
            out_row,
            out_act_q_row,
            out_inv_sw_bits);
    }
};

} // namespace aecct
