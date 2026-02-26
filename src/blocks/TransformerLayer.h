#pragma once
// TransformerLayer.h
// M11 one-layer wrapper: Attn -> FFN -> Add2 + LN

#include <cstdint>

#include "AecctTypes.h"
#include "LayerNormDesc.h"
#include "LayerParamBringup.h"
#include "LayerScratchDesc.h"
#include "LayerNormBlock.h"
#include "AttnLayer0.h"
#include "FFNLayer0.h"
#include "weights.h"

#ifdef AECCT_FFN_TRACE_MODE
#include "layer0_norm_ffn_out_step0.h"
#include "layer1_norm_ffn_out_step0.h"
#endif

namespace aecct {

    struct CfgRegs {
        u32_t d_model;
        u32_t n_heads;
        u32_t d_ffn;
        u32_t n_layers;
    };

    static inline void load_layer_sublayer1_norm_params(
        u32_t* sram,
        uint32_t layer_id,
        uint32_t gamma_base,
        uint32_t beta_base,
        uint32_t d_model
    ) {
        for (uint32_t c = 0; c < d_model; ++c) {
            float g = 0.0f;
            float b = 0.0f;
            if (layer_id == 0u) {
                g = (float)w_decoder_layers_0_sublayer_1_norm_weight[c];
                b = (float)w_decoder_layers_0_sublayer_1_norm_bias[c];
            }
            else {
                g = (float)w_decoder_layers_1_sublayer_1_norm_weight[c];
                b = (float)w_decoder_layers_1_sublayer_1_norm_bias[c];
            }

            union {
                float f;
                uint32_t u;
            } gcvt, bcvt;
            gcvt.f = g;
            bcvt.f = b;
            sram[gamma_base + c] = (u32_t)gcvt.u;
            sram[beta_base + c] = (u32_t)bcvt.u;
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
        (void)pb;

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
            layer_id
        );

#ifdef AECCT_FFN_TRACE_MODE
        if ((uint32_t)layer_id.to_uint() == 0u) {
            uint32_t sample_idx = ffn_find_sample_by_ref(
                sram,
                (uint32_t)x_in_base_word.to_uint(),
                (uint32_t)FFN_X_WORDS,
                ffn_trace_x_word
            );
            uint32_t out_base = (uint32_t)x_out_base_word.to_uint();
            for (uint32_t i = 0; i < (uint32_t)FFN_X_WORDS; ++i) {
                uint32_t flat = sample_idx * (uint32_t)FFN_X_WORDS + i;
                sram[out_base + i] = ffn_f32_to_bits((float)trace_layer0_norm_ffn_out_step0_tensor[flat]);
            }
            return;
        }
        if ((uint32_t)layer_id.to_uint() == 1u) {
            uint32_t sample_idx = 0u;
            uint32_t out_base = (uint32_t)x_out_base_word.to_uint();
            for (uint32_t i = 0; i < (uint32_t)FFN_X_WORDS; ++i) {
                uint32_t flat = sample_idx * (uint32_t)FFN_X_WORDS + i;
                sram[out_base + i] = ffn_f32_to_bits((float)trace_layer1_norm_ffn_out_step0_tensor[flat]);
            }
            return;
        }
#endif

        uint32_t residual_base = (uint32_t)sc.attn_out_base_word.to_uint();
        uint32_t w2_base = (uint32_t)sc.ffn.w2_out_base_word.to_uint();
        uint32_t add2_base = (uint32_t)sc.ffn.add2_base_word.to_uint();
        uint32_t words = (uint32_t)FFN_X_WORDS;
        for (uint32_t i = 0; i < words; ++i) {
            union {
                uint32_t u;
                float f;
            } x_cvt, w2_cvt, y_cvt;
            x_cvt.u = (uint32_t)sram[residual_base + i].to_uint();
            w2_cvt.u = (uint32_t)sram[w2_base + i].to_uint();
            y_cvt.f = x_cvt.f + w2_cvt.f;
            sram[add2_base + i] = (u32_t)y_cvt.u;
        }

        uint32_t gamma_base = (uint32_t)sc.ffn.ln_gamma_base_word.to_uint();
        uint32_t beta_base = (uint32_t)sc.ffn.ln_beta_base_word.to_uint();
        load_layer_sublayer1_norm_params(sram, (uint32_t)layer_id.to_uint(), gamma_base, beta_base, d_model);

        LayerNormCfg ln_cfg;
        ln_cfg.token_count = (u32_t)FFN_TOKEN_COUNT;
        ln_cfg.d_model = (u32_t)d_model;
        ln_cfg.eps = LN_EPS;

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
