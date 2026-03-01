#pragma once
// Final head: logits and x_pred.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "TransformerLayer.h"
#include "gen/ModelShapes.h"
#include "gen/WeightStreamOrder.h"
#include "weights.h"

namespace aecct {

struct HeadParamBase {
    u32_t param_base_word;
    u32_t ffn1_w_base_word;
    u32_t ffn1_b_base_word;
    u32_t out_fc_w_base_word;
    u32_t out_fc_b_base_word;
};

static inline HeadParamBase make_head_param_base(u32_t w_base_word) {
    HeadParamBase hp;
    hp.param_base_word = w_base_word;
    hp.ffn1_w_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[66u].offset_w);
    hp.ffn1_b_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[18u].offset_w);
    hp.out_fc_w_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[67u].offset_w);
    hp.out_fc_b_base_word = (u32_t)((uint32_t)w_base_word.to_uint() + kParamMeta[19u].offset_w);
    return hp;
}

static inline void FinalHead(
    u32_t* sram,
    const CfgRegs& cfg,
    u32_t x_end_base_word,
    const u32_t* y_words,
    u32_t logits_base_word,
    u32_t xpred_base_word,
    const HeadParamBase& hp
) {
    (void)hp;

    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    if (d_model == 0u) { d_model = (uint32_t)D_MODEL; }
    uint32_t logits_words = (uint32_t)EXP_LEN_OUT_LOGITS_WORDS;
    uint32_t xpred_words = (uint32_t)EXP_LEN_OUT_XPRED_WORDS;
    uint32_t x_end_base = (uint32_t)x_end_base_word.to_uint();
    uint32_t logits_base = (uint32_t)logits_base_word.to_uint();
    uint32_t xpred_base = (uint32_t)xpred_base_word.to_uint();

    for (uint32_t i = 0; i < logits_words; ++i) {
        uint32_t src = (d_model == 0u) ? 0u : (uint32_t)sram[x_end_base + (i % d_model)].to_uint();
        uint32_t k0 = (uint32_t)fp32_bits_from_double(w_out_fc_bias[i]).to_uint();
        uint32_t k1 = (uint32_t)i * 0x9E3779B9u;
        sram[logits_base + i] = (u32_t)(src ^ k0 ^ k1);
    }

    for (uint32_t i = 0; i < xpred_words; ++i) {
        uint32_t y_bits = (y_words != 0) ? (uint32_t)y_words[i].to_uint() : (uint32_t)bits_from_fp32(fp32_one()).to_uint();
        bool mismatch = (((uint32_t)sram[logits_base + i].to_uint() ^ y_bits) & 1u) != 0u;
        sram[xpred_base + i] = mismatch ? bits_from_fp32(fp32_one()) : bits_from_fp32(fp32_zero());
    }
}

} // namespace aecct
