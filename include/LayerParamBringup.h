#pragma once
// LayerParamBringup.h
// M11 bring-up per-layer param base mapping (single source of truth)

#include "AecctTypes.h"
#include "WeightStreamOrder.h"

namespace aecct {

    struct LayerParamBase {
        u32_t layer_id;
        u32_t param_base_word;
        u32_t attn_param_base_word;
        u32_t ffn_param_base_word;
        u32_t norm_param_base_word;
    };

    static inline LayerParamBase make_layer_param_base(u32_t w_base_word, u32_t layer_id) {
        uint32_t lid = (uint32_t)layer_id.to_uint();

        uint32_t attn_idx = (lid == 0u) ? 24u : 44u;
        uint32_t ffn_idx = (lid == 0u) ? 36u : 56u;
        uint32_t norm_idx = (lid == 0u) ? 43u : 63u;

        LayerParamBase pb;
        pb.layer_id = layer_id;
        pb.param_base_word = w_base_word;
        pb.attn_param_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[attn_idx].offset_w);
        pb.ffn_param_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[ffn_idx].offset_w);
        pb.norm_param_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[norm_idx].offset_w);
        return pb;
    }

} // namespace aecct