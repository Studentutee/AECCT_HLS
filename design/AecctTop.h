#pragma once
// Top-level class wrapper for HLS flow.

#include "AecctTypes.h"
#include "Top.h"

#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#endif

namespace aecct {

#pragma hls_design top
class AecctTop {
public:
#pragma hls_design interface
    void run(
        ctrl_ch_t& ctrl_cmd,
        ctrl_ch_t& ctrl_rsp,
        data_ch_t& data_in,
        data_ch_t& data_out
    ) {
        top(ctrl_cmd, ctrl_rsp, data_in, data_out);
    }
};

} // namespace aecct
