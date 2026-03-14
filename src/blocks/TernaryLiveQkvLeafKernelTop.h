#pragma once
// Local Catapult-friendly top wrapper for P11J live ternary WQ row kernel.

#include "AecctTypes.h"
#include "TernaryLiveQkvLeafKernel.h"

#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#endif

namespace aecct {

#pragma hls_design top
class TernaryLiveL0WqRowTop {
public:
#pragma hls_design interface
    bool run(
        u32_t* sram,
        u32_t param_base_word,
        u32_t x_row_base_word,
        u32_t out_row_base_word,
        u32_t out_act_q_row_base_word,
        u32_t& out_inv_sw_bits
    ) {
        return ternary_live_l0_wq_materialize_row_kernel(
            sram,
            param_base_word,
            x_row_base_word,
            out_row_base_word,
            out_act_q_row_base_word,
            out_inv_sw_bits);
    }
};

} // namespace aecct
