// P11AV: Top-side FFN handoff assembly smoke (local-only).
// Scope:
// - Validate Top-side caller helper can assemble W1+W2 seam descriptors.
// - Validate Top-side dispatch helper forwards descriptors to TransformerLayer.
// - Validate invalid descriptors keep legacy preload fallback behavior.

#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>

#include "Top.h"
#include "gen/SramMap.h"

#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#define AECCT_HAS_SCVERIFY 1
#else
#define AECCT_HAS_SCVERIFY 0
#endif

#if !AECCT_HAS_SCVERIFY
#ifndef CCS_MAIN
#define CCS_MAIN(...) int main(__VA_ARGS__)
#endif
#ifndef CCS_RETURN
#define CCS_RETURN(x) return (x)
#endif
#endif

namespace {

static uint32_t f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = f;
    return cvt.u;
}

static aecct::u32_t local_alternate_x_page(aecct::u32_t x_base_word) {
    const uint32_t x_base = (uint32_t)x_base_word.to_uint();
    if (x_base == (uint32_t)sram_map::X_PAGE0_BASE_W) {
        return (aecct::u32_t)sram_map::X_PAGE1_BASE_W;
    }
    return (aecct::u32_t)sram_map::X_PAGE0_BASE_W;
}

static void clear_sram(aecct::u32_t* sram) {
    for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = (aecct::u32_t)0u;
    }
}

static void init_transformer_case_memory(
    aecct::u32_t* sram,
    const aecct::CfgRegs& cfg,
    const aecct::LayerScratch& sc,
    const aecct::LayerParamBase& pb,
    uint32_t layer_id
) {
    clear_sram(sram);
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t d_ffn = (uint32_t)cfg.d_ffn.to_uint();
    if (d_model == 0u) { d_model = (uint32_t)aecct::ATTN_D_MODEL; }
    if (d_ffn == 0u) { d_ffn = (uint32_t)aecct::FFN_D_FFN; }
    const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
    const uint32_t one_bits = f32_to_bits(1.0f);
    const uint32_t zero_bits = f32_to_bits(0.0f);

    const uint32_t attn_out_base = (uint32_t)sc.attn_out_base_word.to_uint();
    const uint32_t attn_out_words = token_count * d_model;
    for (uint32_t i = 0u; i < attn_out_words; ++i) {
        sram[attn_out_base + i] = (aecct::u32_t)one_bits;
    }

    const bool use_layer1 = (layer_id == 1u);
    const uint32_t w1_bias_id = use_layer1 ? 12u : 4u;
    const uint32_t w1_weight_id = use_layer1 ? 56u : 36u;
    const uint32_t w2_bias_id = use_layer1 ? 13u : 5u;
    const uint32_t w2_weight_id = use_layer1 ? 59u : 39u;
    const uint32_t param_base = (uint32_t)pb.param_base_word.to_uint();
    const uint32_t w1_bias_base = param_base + kParamMeta[w1_bias_id].offset_w;
    const uint32_t w1_weight_base = param_base + kParamMeta[w1_weight_id].offset_w;
    const uint32_t w2_bias_base = param_base + kParamMeta[w2_bias_id].offset_w;
    const uint32_t w2_weight_base = param_base + kParamMeta[w2_weight_id].offset_w;

    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_WEIGHT_WORDS; ++i) {
        sram[w1_weight_base + i] = (aecct::u32_t)one_bits;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_BIAS_WORDS; ++i) {
        sram[w1_bias_base + i] = (aecct::u32_t)one_bits;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_WEIGHT_WORDS; ++i) {
        sram[w2_weight_base + i] = (aecct::u32_t)one_bits;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_BIAS_WORDS; ++i) {
        sram[w2_bias_base + i] = (aecct::u32_t)one_bits;
    }

    const uint32_t gamma_base = (uint32_t)sc.ffn.ln_gamma_base_word.to_uint();
    const uint32_t beta_base = (uint32_t)sc.ffn.ln_beta_base_word.to_uint();
    for (uint32_t c = 0u; c < d_model; ++c) {
        sram[gamma_base + c] = (aecct::u32_t)one_bits;
        sram[beta_base + c] = (aecct::u32_t)zero_bits;
    }
}

template<typename RunFn>
static bool run_single_path_case(const char* case_label, RunFn run_fn) {
    static aecct::u32_t sram_baseline[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_seam[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_fallback[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t seam_w1_x_words[aecct::FFN_X_WORDS];
    static aecct::u32_t seam_w1_weight_words[aecct::FFN_W1_WEIGHT_WORDS];
    static aecct::u32_t seam_w1_bias_words[aecct::FFN_W1_BIAS_WORDS];
    static aecct::u32_t seam_w2_input_words[aecct::FFN_W2_INPUT_WORDS];
    static aecct::u32_t seam_w2_weight_words[aecct::FFN_W2_WEIGHT_WORDS];
    static aecct::u32_t seam_w2_bias_words[aecct::FFN_W2_BIAS_WORDS];
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_X_WORDS; ++i) { seam_w1_x_words[i] = (aecct::u32_t)0u; }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_WEIGHT_WORDS; ++i) { seam_w1_weight_words[i] = (aecct::u32_t)0u; }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_BIAS_WORDS; ++i) { seam_w1_bias_words[i] = (aecct::u32_t)0u; }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_INPUT_WORDS; ++i) { seam_w2_input_words[i] = (aecct::u32_t)0u; }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_WEIGHT_WORDS; ++i) { seam_w2_weight_words[i] = (aecct::u32_t)0u; }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_BIAS_WORDS; ++i) { seam_w2_bias_words[i] = (aecct::u32_t)0u; }

    aecct::CfgRegs cfg;
    cfg.d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
    cfg.n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
    cfg.d_ffn = (aecct::u32_t)aecct::FFN_D_FFN;
    cfg.n_layers = (aecct::u32_t)1u;

    const aecct::u32_t layer_id = (aecct::u32_t)0u;
    const aecct::u32_t x_in_base = (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD_DEFAULT;
    const aecct::u32_t x_out_base = local_alternate_x_page(x_in_base);
    const aecct::LayerScratch sc = aecct::make_layer_scratch(x_in_base);
    const aecct::LayerParamBase pb =
        aecct::make_layer_param_base((aecct::u32_t)sram_map::W_REGION_BASE, layer_id);

    init_transformer_case_memory(sram_baseline, cfg, sc, pb, (uint32_t)layer_id.to_uint());
    init_transformer_case_memory(sram_seam, cfg, sc, pb, (uint32_t)layer_id.to_uint());
    init_transformer_case_memory(sram_fallback, cfg, sc, pb, (uint32_t)layer_id.to_uint());

    const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t d_ffn = (uint32_t)cfg.d_ffn.to_uint();
    if (d_model == 0u) { d_model = (uint32_t)aecct::FFN_D_MODEL; }
    if (d_ffn == 0u) { d_ffn = (uint32_t)aecct::FFN_D_FFN; }
    const uint32_t w1_x_words = token_count * d_model;
    const uint32_t w1_weight_words = d_ffn * d_model;
    const uint32_t w1_bias_words = d_ffn;
    const uint32_t w2_input_words = token_count * d_ffn;
    const uint32_t w2_weight_words = d_model * d_ffn;
    const uint32_t w2_bias_words = d_model;

    run_fn(
        sram_baseline,
        cfg,
        layer_id,
        x_in_base,
        x_out_base,
        sc,
        pb,
        aecct::top_make_transformer_layer_ffn_topfed_handoff_desc()
    );
    run_fn(
        sram_seam,
        cfg,
        layer_id,
        x_in_base,
        x_out_base,
        sc,
        pb,
        aecct::top_make_transformer_layer_ffn_topfed_handoff_desc(
            seam_w1_x_words, (aecct::u32_t)w1_x_words,
            seam_w1_weight_words, (aecct::u32_t)w1_weight_words,
            seam_w1_bias_words, (aecct::u32_t)w1_bias_words,
            seam_w2_input_words, (aecct::u32_t)w2_input_words,
            seam_w2_weight_words, (aecct::u32_t)w2_weight_words,
            seam_w2_bias_words, (aecct::u32_t)w2_bias_words)
    );
    run_fn(
        sram_fallback,
        cfg,
        layer_id,
        x_in_base,
        x_out_base,
        sc,
        pb,
        aecct::top_make_transformer_layer_ffn_topfed_handoff_desc(
            seam_w1_x_words, (aecct::u32_t)0u,
            seam_w1_weight_words, (aecct::u32_t)0u,
            seam_w1_bias_words, (aecct::u32_t)0u,
            seam_w2_input_words, (aecct::u32_t)0u,
            seam_w2_weight_words, (aecct::u32_t)0u,
            seam_w2_bias_words, (aecct::u32_t)0u)
    );

    const uint32_t w1_out_base = (uint32_t)sc.ffn.w1_out_base_word.to_uint();
    const uint32_t w2_out_base = (uint32_t)sc.ffn.w2_out_base_word.to_uint();
    uint32_t w1_change_count = 0u;
    uint32_t w2_change_count = 0u;
    const uint32_t compare_words = (w2_bias_words < 16u) ? w2_bias_words : 16u;
    TOP_W1W2_COMPARE_LOOP: for (uint32_t i = 0u; i < compare_words; ++i) {
        const uint32_t base_w1 = (uint32_t)sram_baseline[w1_out_base + i].to_uint();
        const uint32_t seam_w1 = (uint32_t)sram_seam[w1_out_base + i].to_uint();
        const uint32_t fallback_w1 = (uint32_t)sram_fallback[w1_out_base + i].to_uint();
        const uint32_t base_w2 = (uint32_t)sram_baseline[w2_out_base + i].to_uint();
        const uint32_t seam_w2 = (uint32_t)sram_seam[w2_out_base + i].to_uint();
        const uint32_t fallback_w2 = (uint32_t)sram_fallback[w2_out_base + i].to_uint();
        if (base_w1 != seam_w1) { ++w1_change_count; }
        if (base_w2 != seam_w2) { ++w2_change_count; }
        if (base_w1 != fallback_w1 || base_w2 != fallback_w2) {
            std::printf("[p11av][FAIL] %s fallback mismatch idx=%u\n", case_label, (unsigned)i);
            return false;
        }
    }
    if (w1_change_count == 0u || w2_change_count == 0u) {
        std::printf("[p11av][FAIL] %s seam change missing (w1=%u w2=%u)\n",
            case_label, (unsigned)w1_change_count, (unsigned)w2_change_count);
        return false;
    }
    return true;
}

static bool run_pointer_dispatch_case() {
    const bool ok = run_single_path_case(
        "pointer_dispatch",
        [](
            aecct::u32_t* sram,
            const aecct::CfgRegs& cfg,
            aecct::u32_t layer_id,
            aecct::u32_t x_in_base,
            aecct::u32_t x_out_base,
            const aecct::LayerScratch& sc,
            const aecct::LayerParamBase& pb,
            aecct::TransformerLayerFfnTopfedHandoffDesc seam_desc
        ) {
            aecct::top_dispatch_transformer_layer(
                sram,
                cfg,
                layer_id,
                x_in_base,
                x_out_base,
                sc,
                pb,
                true,
                true,
                true,
                true,
                true,
                seam_desc
            );
        });
    if (!ok) { return false; }
    std::printf("TOP_FFN_HANDOFF_ASSEMBLY_POINTER_DISPATCH PASS\n");
    return true;
}

static bool run_deep_bridge_dispatch_case() {
    const bool ok = run_single_path_case(
        "deep_bridge_dispatch",
        [](
            aecct::u32_t* sram,
            const aecct::CfgRegs& cfg,
            aecct::u32_t layer_id,
            aecct::u32_t x_in_base,
            aecct::u32_t x_out_base,
            const aecct::LayerScratch& sc,
            const aecct::LayerParamBase& pb,
            aecct::TransformerLayerFfnTopfedHandoffDesc seam_desc
        ) {
            static aecct::u32_t sram_window[sram_map::SRAM_WORDS_TOTAL];
            for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                sram_window[i] = sram[i];
            }
            aecct::top_dispatch_transformer_layer_top_managed_attn_bridge<sram_map::SRAM_WORDS_TOTAL>(
                sram_window,
                cfg,
                layer_id,
                x_in_base,
                x_out_base,
                sc,
                pb,
                true,
                true,
                true,
                true,
                true,
                seam_desc
            );
            for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                sram[i] = sram_window[i];
            }
        });
    if (!ok) { return false; }
    std::printf("TOP_FFN_HANDOFF_ASSEMBLY_DEEP_BRIDGE_DISPATCH PASS\n");
    return true;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;

    if (!run_pointer_dispatch_case()) {
        CCS_RETURN(1);
    }
    if (!run_deep_bridge_dispatch_case()) {
        CCS_RETURN(1);
    }
    std::printf("TOP_FFN_HANDOFF_ASSEMBLY_EXPECTED_COMPARE PASS\n");
    std::printf("PASS: tb_top_ffn_handoff_assembly_smoke_p11av\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__

