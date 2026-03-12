#pragma once
// One-layer wrapper: attention, FFN, residual add, LayerNorm.

#include <cstdint>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "AecctRanges.h"
#include "AecctUtil.h"
#include "AttnLayer0.h"
#include "FFNLayer0.h"
#include "LayerNormBlock.h"
#include "LayerNormDesc.h"
#include "LayerParamBringup.h"
#include "LayerScratchDesc.h"
#include "gen/WeightStreamOrder.h"

namespace aecct {

struct TransformerLayerContract {
    bool start;
    bool done;
    bool err_valid;
    u16_t err_code;
    TokenRange token_range;
    TileRange tile_range;
    PhaseId phase_id;
    u32_t x_work_base_word;
    u32_t scratch_base_word;
    u32_t w_base_word;
};

static inline void clear_transformer_layer_contract(TransformerLayerContract& c) {
    c.start = false;
    c.done = false;
    c.err_valid = false;
    c.err_code = 0;
    c.token_range = make_token_range(0, 0);
    c.tile_range = make_tile_range(0, 0);
    c.phase_id = PHASE_LAYER0;
    c.x_work_base_word = 0;
    c.scratch_base_word = 0;
    c.w_base_word = 0;
}

struct CfgRegs {
    u32_t d_model;
    u32_t n_heads;
    u32_t d_ffn;
    u32_t n_layers;
};

static inline void load_layer_sublayer1_norm_params(
    u32_t* sram,
    uint32_t param_base_word,
    uint32_t layer_id,
    uint32_t gamma_base,
    uint32_t beta_base,
    uint32_t d_model
) {
    const uint32_t norm_w_id = (layer_id == 0u) ? 43u : 63u;
    const uint32_t norm_b_id = (layer_id == 0u) ? 7u : 15u;
    const uint32_t norm_w_base = param_base_word + kParamMeta[norm_w_id].offset_w;
    const uint32_t norm_b_base = param_base_word + kParamMeta[norm_b_id].offset_w;

    for (uint32_t c = 0; c < d_model; ++c) {
        sram[gamma_base + c] = sram[norm_w_base + c];
        sram[beta_base + c] = sram[norm_b_base + c];
    }
}

static inline void TransformerLayer(
    u32_t* sram,
    const CfgRegs& cfg,
    u32_t layer_id,
    u32_t x_in_base_word,
    u32_t x_out_base_word,
    const LayerScratch& sc,
    const LayerParamBase& pb
) {
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t n_heads = (uint32_t)cfg.n_heads.to_uint();
    uint32_t d_ffn = (uint32_t)cfg.d_ffn.to_uint();
    if (d_model == 0u) { d_model = (uint32_t)ATTN_D_MODEL; }
    if (n_heads == 0u) { n_heads = (uint32_t)ATTN_N_HEADS; }
    if (d_ffn == 0u) { d_ffn = (uint32_t)FFN_D_FFN; }

    AttnCfg attn_cfg;
    attn_cfg.token_count = (u32_t)ATTN_TOKEN_COUNT;
    attn_cfg.d_model = (u32_t)d_model;
    attn_cfg.n_heads = (u32_t)n_heads;
    attn_cfg.d_head = (u32_t)(d_model / n_heads);

    AttnLayer0<ATTN_STAGE_FULL>(
        sram,
        attn_cfg,
        x_in_base_word,
        sc.attn_out_base_word,
        sc.attn
    );

    FfnCfg ffn_cfg;
    ffn_cfg.token_count = (u32_t)FFN_TOKEN_COUNT;
    ffn_cfg.d_model = (u32_t)d_model;
    ffn_cfg.d_ffn = (u32_t)d_ffn;

    FFNLayer0<FFN_STAGE_FULL>(
        sram,
        ffn_cfg,
        sc.attn_out_base_word,
        sc.ffn,
        pb.param_base_word,
        layer_id
    );

    uint32_t residual_base = (uint32_t)sc.attn_out_base_word.to_uint();
    uint32_t w2_base = (uint32_t)sc.ffn.w2_out_base_word.to_uint();
    uint32_t add2_base = (uint32_t)sc.ffn.add2_base_word.to_uint();
    uint32_t words = (uint32_t)FFN_X_WORDS;
    for (uint32_t i = 0; i < words; ++i) {
        fp32_t x = fp32_from_bits(sram[residual_base + i]);
        fp32_t y = fp32_from_bits(sram[w2_base + i]);
        sram[add2_base + i] = bits_from_fp32(x + y);
    }

    uint32_t gamma_base = (uint32_t)sc.ffn.ln_gamma_base_word.to_uint();
    uint32_t beta_base = (uint32_t)sc.ffn.ln_beta_base_word.to_uint();
    load_layer_sublayer1_norm_params(sram, (uint32_t)pb.param_base_word.to_uint(), (uint32_t)layer_id.to_uint(), gamma_base, beta_base, d_model);

    LayerNormCfg ln_cfg;
    ln_cfg.token_count = (u32_t)FFN_TOKEN_COUNT;
    ln_cfg.d_model = (u32_t)d_model;
    ln_cfg.eps_bits = LN_EPS_BITS;

    LayerNormBlock(
        sram,
        ln_cfg,
        (u32_t)add2_base,
        x_out_base_word,
        (u32_t)gamma_base,
        (u32_t)beta_base
    );
}

} // namespace aecct
