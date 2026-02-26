#pragma once
// LayerScratchDesc.h
// M11 single-source scratch layout for one Transformer layer

#include "AecctTypes.h"
#include "AttnDescBringup.h"
#include "FfnDescBringup.h"

namespace aecct {

    struct LayerScratch {
        AttnScratch attn;
        FfnScratch ffn;
        u32_t attn_out_base_word;
    };

    static inline LayerScratch make_layer_scratch(u32_t attn_out_base_word) {
        LayerScratch sc;
        sc.attn = default_attn_scratch();
        sc.ffn = default_ffn_scratch();
        sc.attn_out_base_word = attn_out_base_word;
        return sc;
    }

} // namespace aecct