// P11AU: TransformerLayer higher-level ownership seam smoke (local-only).
// Scope:
// - Validate optional Top-fed FFN W1/W2 payload handoff on TransformerLayer boundary.
// - Validate both pointer path and deep bridge path consume the seam consistently.

#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>

#include "blocks/TransformerLayer.h"
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

static aecct::u32_t local_canonical_x_work_base(aecct::u32_t x_base_word) {
    const uint32_t x_base = (uint32_t)x_base_word.to_uint();
    if (x_base == (uint32_t)sram_map::X_WORK_BASE_W) {
        return (aecct::u32_t)sram_map::X_WORK_BASE_W;
    }
    return (aecct::u32_t)sram_map::X_WORK_BASE_W;
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
        // Baseline path should pick this non-zero bias unless seam override is consumed.
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
static bool run_w1_input_observability_case(
    const char* case_label,
    const char* banner_suffix,
    RunFn run_fn
) {
    static aecct::u32_t sram_w1_ready[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_w1_fallback[sram_map::SRAM_WORDS_TOTAL];

    static aecct::u32_t seam_w1_x_words[aecct::FFN_X_WORDS];
    static aecct::u32_t seam_w1_weight_words[aecct::FFN_W1_WEIGHT_WORDS];
    static aecct::u32_t seam_w1_bias_words[aecct::FFN_W1_BIAS_WORDS];
    static aecct::u32_t seam_w2_input_words[aecct::FFN_W2_INPUT_WORDS];
    static aecct::u32_t seam_w2_weight_words[aecct::FFN_W2_WEIGHT_WORDS];
    static aecct::u32_t seam_w2_bias_words[aecct::FFN_W2_BIAS_WORDS];

    const uint32_t one_bits = f32_to_bits(1.0f);
    const uint32_t half_bits = f32_to_bits(0.5f);
    const uint32_t two_bits = f32_to_bits(2.0f);

    FILL_W1_INPUT_PROBE_X_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_X_WORDS; ++i) {
        seam_w1_x_words[i] = (aecct::u32_t)one_bits;
    }
    FILL_W1_INPUT_PROBE_W1_WEIGHT_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_WEIGHT_WORDS; ++i) {
        seam_w1_weight_words[i] = (aecct::u32_t)one_bits;
    }
    FILL_W1_INPUT_PROBE_W1_BIAS_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_BIAS_WORDS; ++i) {
        seam_w1_bias_words[i] = (aecct::u32_t)half_bits;
    }
    FILL_W1_INPUT_PROBE_W2_INPUT_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_INPUT_WORDS; ++i) {
        seam_w2_input_words[i] = ((i & 0x1u) == 0u) ? (aecct::u32_t)two_bits : (aecct::u32_t)one_bits;
    }
    FILL_W1_INPUT_PROBE_W2_WEIGHT_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_WEIGHT_WORDS; ++i) {
        seam_w2_weight_words[i] = (aecct::u32_t)one_bits;
    }
    FILL_W1_INPUT_PROBE_W2_BIAS_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_BIAS_WORDS; ++i) {
        seam_w2_bias_words[i] = (aecct::u32_t)half_bits;
    }

    aecct::CfgRegs cfg;
    cfg.d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
    cfg.n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
    cfg.d_ffn = (aecct::u32_t)aecct::FFN_D_FFN;
    cfg.n_layers = (aecct::u32_t)1u;

    const aecct::u32_t layer_id = (aecct::u32_t)0u;
    const aecct::u32_t x_in_base = (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD_DEFAULT;
    const aecct::u32_t x_out_base = local_canonical_x_work_base(x_in_base);
    const aecct::LayerScratch sc = aecct::make_layer_scratch(x_in_base);
    const aecct::LayerParamBase pb =
        aecct::make_layer_param_base((aecct::u32_t)sram_map::W_REGION_BASE, layer_id);

    const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t d_ffn = (uint32_t)cfg.d_ffn.to_uint();
    if (d_model == 0u) { d_model = (uint32_t)aecct::FFN_D_MODEL; }
    if (d_ffn == 0u) { d_ffn = (uint32_t)aecct::FFN_D_FFN; }
    uint32_t w1_x_words = token_count * d_model;
    if (w1_x_words > (uint32_t)aecct::FFN_X_WORDS) {
        w1_x_words = (uint32_t)aecct::FFN_X_WORDS;
    }
    uint32_t w1_weight_words = d_ffn * d_model;
    if (w1_weight_words > (uint32_t)aecct::FFN_W1_WEIGHT_WORDS) {
        w1_weight_words = (uint32_t)aecct::FFN_W1_WEIGHT_WORDS;
    }
    uint32_t w1_bias_words = d_ffn;
    if (w1_bias_words > (uint32_t)aecct::FFN_W1_BIAS_WORDS) {
        w1_bias_words = (uint32_t)aecct::FFN_W1_BIAS_WORDS;
    }
    uint32_t w2_input_words = token_count * d_ffn;
    if (w2_input_words > (uint32_t)aecct::FFN_W2_INPUT_WORDS) {
        w2_input_words = (uint32_t)aecct::FFN_W2_INPUT_WORDS;
    }
    uint32_t w2_weight_words = d_model * d_ffn;
    if (w2_weight_words > (uint32_t)aecct::FFN_W2_WEIGHT_WORDS) {
        w2_weight_words = (uint32_t)aecct::FFN_W2_WEIGHT_WORDS;
    }
    uint32_t w2_bias_words = d_model;
    if (w2_bias_words > (uint32_t)aecct::FFN_W2_BIAS_WORDS) {
        w2_bias_words = (uint32_t)aecct::FFN_W2_BIAS_WORDS;
    }

    aecct::TransformerLayerW2SeamProbe ready_probe;
    aecct::TransformerLayerW2SeamProbe fallback_probe;
    const auto run_case =
        [&](aecct::u32_t* sram_mem, bool w1_input_ready, aecct::TransformerLayerW2SeamProbe& probe) {
        // Probe this call boundary to capture W1 input mainline/fallback branch decisions.
        aecct::clear_transformer_layer_w2_seam_probe(probe);
        init_transformer_case_memory(sram_mem, cfg, sc, pb, (uint32_t)layer_id.to_uint());
        run_fn(
            sram_mem,
            cfg,
            layer_id,
            x_in_base,
            x_out_base,
            sc,
            pb,
            aecct::make_transformer_layer_ffn_topfed_handoff_desc(
                seam_w1_x_words,
                w1_input_ready ? (aecct::u32_t)w1_x_words : (aecct::u32_t)0u,
                seam_w1_weight_words,
                (aecct::u32_t)w1_weight_words,
                seam_w1_bias_words,
                (aecct::u32_t)w1_bias_words,
                seam_w2_input_words,
                (aecct::u32_t)w2_input_words,
                seam_w2_weight_words,
                (aecct::u32_t)w2_weight_words,
                seam_w2_bias_words,
                (aecct::u32_t)w2_bias_words),
            &probe
        );
    };

    run_case(sram_w1_ready, true, ready_probe);
    if ((uint32_t)ready_probe.w1_input_mainline_taken_count.to_uint() != 1u ||
        (uint32_t)ready_probe.w1_input_fallback_preload_count.to_uint() != 0u) {
        std::printf(
            "[p11au][FAIL] %s W1 ready probe mismatch mainline=%u fallback=%u\n",
            case_label,
            (unsigned)ready_probe.w1_input_mainline_taken_count.to_uint(),
            (unsigned)ready_probe.w1_input_fallback_preload_count.to_uint());
        return false;
    }

    run_case(sram_w1_fallback, false, fallback_probe);
    if ((uint32_t)fallback_probe.w1_input_mainline_taken_count.to_uint() != 0u ||
        (uint32_t)fallback_probe.w1_input_fallback_preload_count.to_uint() != 1u) {
        std::printf(
            "[p11au][FAIL] %s W1 fallback probe mismatch mainline=%u fallback=%u\n",
            case_label,
            (unsigned)fallback_probe.w1_input_mainline_taken_count.to_uint(),
            (unsigned)fallback_probe.w1_input_fallback_preload_count.to_uint());
        return false;
    }

    std::printf("W1_INPUT_MAINLINE_TAKEN_%s PASS\n", banner_suffix);
    std::printf("W1_INPUT_PRELOAD_FALLBACK_NOT_TAKEN_%s PASS\n", banner_suffix);
    std::printf("W1_INPUT_PRELOAD_FALLBACK_EXPECTED_%s PASS\n", banner_suffix);
    return true;
}

template<typename RunFn>
static bool run_w2_weight_bias_observability_case(
    const char* case_label,
    const char* banner_suffix,
    RunFn run_fn
) {
    static aecct::u32_t sram_weight_ready[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_weight_fallback[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_bias_fallback[sram_map::SRAM_WORDS_TOTAL];

    static aecct::u32_t seam_w1_x_words[aecct::FFN_X_WORDS];
    static aecct::u32_t seam_w1_weight_words[aecct::FFN_W1_WEIGHT_WORDS];
    static aecct::u32_t seam_w1_bias_words[aecct::FFN_W1_BIAS_WORDS];
    static aecct::u32_t seam_w2_input_words[aecct::FFN_W2_INPUT_WORDS];
    static aecct::u32_t seam_w2_weight_words[aecct::FFN_W2_WEIGHT_WORDS];
    static aecct::u32_t seam_w2_bias_words[aecct::FFN_W2_BIAS_WORDS];

    const uint32_t one_bits = f32_to_bits(1.0f);
    const uint32_t half_bits = f32_to_bits(0.5f);
    const uint32_t two_bits = f32_to_bits(2.0f);

    FILL_W1_X_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_X_WORDS; ++i) {
        seam_w1_x_words[i] = (aecct::u32_t)one_bits;
    }
    FILL_W1_WEIGHT_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_WEIGHT_WORDS; ++i) {
        seam_w1_weight_words[i] = (aecct::u32_t)one_bits;
    }
    FILL_W1_BIAS_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_BIAS_WORDS; ++i) {
        seam_w1_bias_words[i] = (aecct::u32_t)half_bits;
    }
    FILL_W2_INPUT_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_INPUT_WORDS; ++i) {
        seam_w2_input_words[i] = ((i & 0x1u) == 0u) ? (aecct::u32_t)two_bits : (aecct::u32_t)one_bits;
    }
    FILL_W2_WEIGHT_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_WEIGHT_WORDS; ++i) {
        seam_w2_weight_words[i] = ((i & 0x3u) == 0u) ? (aecct::u32_t)half_bits : (aecct::u32_t)one_bits;
    }
    FILL_W2_BIAS_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_BIAS_WORDS; ++i) {
        seam_w2_bias_words[i] = (aecct::u32_t)half_bits;
    }

    aecct::CfgRegs cfg;
    cfg.d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
    cfg.n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
    cfg.d_ffn = (aecct::u32_t)aecct::FFN_D_FFN;
    cfg.n_layers = (aecct::u32_t)1u;

    const aecct::u32_t layer_id = (aecct::u32_t)0u;
    const aecct::u32_t x_in_base = (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD_DEFAULT;
    const aecct::u32_t x_out_base = local_canonical_x_work_base(x_in_base);
    const aecct::LayerScratch sc = aecct::make_layer_scratch(x_in_base);
    const aecct::LayerParamBase pb =
        aecct::make_layer_param_base((aecct::u32_t)sram_map::W_REGION_BASE, layer_id);

    const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t d_ffn = (uint32_t)cfg.d_ffn.to_uint();
    if (d_model == 0u) { d_model = (uint32_t)aecct::FFN_D_MODEL; }
    if (d_ffn == 0u) { d_ffn = (uint32_t)aecct::FFN_D_FFN; }
    uint32_t w2_input_words = token_count * d_ffn;
    if (w2_input_words > (uint32_t)aecct::FFN_W2_INPUT_WORDS) {
        w2_input_words = (uint32_t)aecct::FFN_W2_INPUT_WORDS;
    }
    uint32_t w2_weight_words = d_model * d_ffn;
    if (w2_weight_words > (uint32_t)aecct::FFN_W2_WEIGHT_WORDS) {
        w2_weight_words = (uint32_t)aecct::FFN_W2_WEIGHT_WORDS;
    }
    uint32_t w2_bias_words = d_model;
    if (w2_bias_words > (uint32_t)aecct::FFN_W2_BIAS_WORDS) {
        w2_bias_words = (uint32_t)aecct::FFN_W2_BIAS_WORDS;
    }
    aecct::TransformerLayerW2SeamProbe weight_ready_probe;
    aecct::TransformerLayerW2SeamProbe weight_fallback_probe;
    aecct::TransformerLayerW2SeamProbe bias_fallback_probe;

    const auto run_case =
        [&](aecct::u32_t* sram_mem,
            bool w2_weight_ready,
            bool w2_bias_ready,
            aecct::TransformerLayerW2SeamProbe& probe) {
        // Probe this call boundary to capture W2 mainline/fallback branch decisions.
        aecct::clear_transformer_layer_w2_seam_probe(probe);
        init_transformer_case_memory(sram_mem, cfg, sc, pb, (uint32_t)layer_id.to_uint());
        run_fn(
            sram_mem,
            cfg,
            layer_id,
            x_in_base,
            x_out_base,
            sc,
            pb,
            aecct::make_transformer_layer_ffn_topfed_handoff_desc(
                seam_w1_x_words,
                (aecct::u32_t)(token_count * d_model),
                seam_w1_weight_words,
                (aecct::u32_t)(d_ffn * d_model),
                seam_w1_bias_words,
                (aecct::u32_t)d_ffn,
                seam_w2_input_words,
                (aecct::u32_t)w2_input_words,
                seam_w2_weight_words,
                w2_weight_ready ? (aecct::u32_t)w2_weight_words : (aecct::u32_t)0u,
                seam_w2_bias_words,
                w2_bias_ready ? (aecct::u32_t)w2_bias_words : (aecct::u32_t)0u),
            &probe
        );
    };

    // Descriptor-ready split: both W2 weight/bias should take mainline, no preload fallback.
    run_case(sram_weight_ready, true, true, weight_ready_probe);
    if ((uint32_t)weight_ready_probe.w2_weight_mainline_taken_count.to_uint() != 1u ||
        (uint32_t)weight_ready_probe.w2_bias_mainline_taken_count.to_uint() != 1u ||
        (uint32_t)weight_ready_probe.w2_weight_fallback_preload_count.to_uint() != 0u ||
        (uint32_t)weight_ready_probe.w2_bias_fallback_preload_count.to_uint() != 0u) {
        std::printf(
            "[p11au][FAIL] %s ready probe mismatch wm=%u wf=%u bm=%u bf=%u\n",
            case_label,
            (unsigned)weight_ready_probe.w2_weight_mainline_taken_count.to_uint(),
            (unsigned)weight_ready_probe.w2_weight_fallback_preload_count.to_uint(),
            (unsigned)weight_ready_probe.w2_bias_mainline_taken_count.to_uint(),
            (unsigned)weight_ready_probe.w2_bias_fallback_preload_count.to_uint());
        return false;
    }

    // Non-ready split for W2 weight: weight should fallback, bias should stay on mainline.
    run_case(sram_weight_fallback, false, true, weight_fallback_probe);
    if ((uint32_t)weight_fallback_probe.w2_weight_mainline_taken_count.to_uint() != 0u ||
        (uint32_t)weight_fallback_probe.w2_weight_fallback_preload_count.to_uint() != 1u ||
        (uint32_t)weight_fallback_probe.w2_bias_mainline_taken_count.to_uint() != 1u ||
        (uint32_t)weight_fallback_probe.w2_bias_fallback_preload_count.to_uint() != 0u) {
        std::printf(
            "[p11au][FAIL] %s weight-fallback probe mismatch wm=%u wf=%u bm=%u bf=%u\n",
            case_label,
            (unsigned)weight_fallback_probe.w2_weight_mainline_taken_count.to_uint(),
            (unsigned)weight_fallback_probe.w2_weight_fallback_preload_count.to_uint(),
            (unsigned)weight_fallback_probe.w2_bias_mainline_taken_count.to_uint(),
            (unsigned)weight_fallback_probe.w2_bias_fallback_preload_count.to_uint());
        return false;
    }

    // Non-ready split for W2 bias: bias should fallback, weight should stay on mainline.
    run_case(sram_bias_fallback, true, false, bias_fallback_probe);
    if ((uint32_t)bias_fallback_probe.w2_weight_mainline_taken_count.to_uint() != 1u ||
        (uint32_t)bias_fallback_probe.w2_weight_fallback_preload_count.to_uint() != 0u ||
        (uint32_t)bias_fallback_probe.w2_bias_mainline_taken_count.to_uint() != 0u ||
        (uint32_t)bias_fallback_probe.w2_bias_fallback_preload_count.to_uint() != 1u) {
        std::printf(
            "[p11au][FAIL] %s bias-fallback probe mismatch wm=%u wf=%u bm=%u bf=%u\n",
            case_label,
            (unsigned)bias_fallback_probe.w2_weight_mainline_taken_count.to_uint(),
            (unsigned)bias_fallback_probe.w2_weight_fallback_preload_count.to_uint(),
            (unsigned)bias_fallback_probe.w2_bias_mainline_taken_count.to_uint(),
            (unsigned)bias_fallback_probe.w2_bias_fallback_preload_count.to_uint());
        return false;
    }

    std::printf("W2_WEIGHT_MAINLINE_TAKEN_%s PASS\n", banner_suffix);
    std::printf("W2_PRELOAD_FALLBACK_NOT_TAKEN_%s PASS\n", banner_suffix);
    std::printf("W2_PRELOAD_FALLBACK_EXPECTED_%s PASS\n", banner_suffix);
    std::printf("W2_BIAS_MAINLINE_TAKEN_%s PASS\n", banner_suffix);
    std::printf("W2_BIAS_PRELOAD_FALLBACK_NOT_TAKEN_%s PASS\n", banner_suffix);
    std::printf("W2_BIAS_PRELOAD_FALLBACK_EXPECTED_%s PASS\n", banner_suffix);
    return true;
}

template<typename RunFn>
static bool run_w1_weight_bias_observability_case(
    const char* case_label,
    const char* banner_suffix,
    RunFn run_fn
) {
    static aecct::u32_t sram_weight_bias_ready[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_weight_fallback[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_bias_fallback[sram_map::SRAM_WORDS_TOTAL];

    static aecct::u32_t seam_w1_x_words[aecct::FFN_X_WORDS];
    static aecct::u32_t seam_w1_weight_words[aecct::FFN_W1_WEIGHT_WORDS];
    static aecct::u32_t seam_w1_bias_words[aecct::FFN_W1_BIAS_WORDS];
    static aecct::u32_t seam_w2_input_words[aecct::FFN_W2_INPUT_WORDS];
    static aecct::u32_t seam_w2_weight_words[aecct::FFN_W2_WEIGHT_WORDS];
    static aecct::u32_t seam_w2_bias_words[aecct::FFN_W2_BIAS_WORDS];

    const uint32_t one_bits = f32_to_bits(1.0f);
    const uint32_t half_bits = f32_to_bits(0.5f);
    const uint32_t two_bits = f32_to_bits(2.0f);

    FILL_W1_WEIGHT_BIAS_PROBE_X_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_X_WORDS; ++i) {
        seam_w1_x_words[i] = (aecct::u32_t)one_bits;
    }
    FILL_W1_WEIGHT_BIAS_PROBE_W1_WEIGHT_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_WEIGHT_WORDS; ++i) {
        seam_w1_weight_words[i] = (aecct::u32_t)one_bits;
    }
    FILL_W1_WEIGHT_BIAS_PROBE_W1_BIAS_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_BIAS_WORDS; ++i) {
        seam_w1_bias_words[i] = (aecct::u32_t)half_bits;
    }
    FILL_W1_WEIGHT_BIAS_PROBE_W2_INPUT_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_INPUT_WORDS; ++i) {
        seam_w2_input_words[i] = ((i & 0x1u) == 0u) ? (aecct::u32_t)two_bits : (aecct::u32_t)one_bits;
    }
    FILL_W1_WEIGHT_BIAS_PROBE_W2_WEIGHT_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_WEIGHT_WORDS; ++i) {
        seam_w2_weight_words[i] = (aecct::u32_t)one_bits;
    }
    FILL_W1_WEIGHT_BIAS_PROBE_W2_BIAS_DESC_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_BIAS_WORDS; ++i) {
        seam_w2_bias_words[i] = (aecct::u32_t)half_bits;
    }

    aecct::CfgRegs cfg;
    cfg.d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
    cfg.n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
    cfg.d_ffn = (aecct::u32_t)aecct::FFN_D_FFN;
    cfg.n_layers = (aecct::u32_t)1u;

    const aecct::u32_t layer_id = (aecct::u32_t)0u;
    const aecct::u32_t x_in_base = (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD_DEFAULT;
    const aecct::u32_t x_out_base = local_canonical_x_work_base(x_in_base);
    const aecct::LayerScratch sc = aecct::make_layer_scratch(x_in_base);
    const aecct::LayerParamBase pb =
        aecct::make_layer_param_base((aecct::u32_t)sram_map::W_REGION_BASE, layer_id);

    const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    uint32_t d_ffn = (uint32_t)cfg.d_ffn.to_uint();
    if (d_model == 0u) { d_model = (uint32_t)aecct::FFN_D_MODEL; }
    if (d_ffn == 0u) { d_ffn = (uint32_t)aecct::FFN_D_FFN; }
    uint32_t w1_x_words = token_count * d_model;
    if (w1_x_words > (uint32_t)aecct::FFN_X_WORDS) {
        w1_x_words = (uint32_t)aecct::FFN_X_WORDS;
    }
    uint32_t w1_weight_words = d_ffn * d_model;
    if (w1_weight_words > (uint32_t)aecct::FFN_W1_WEIGHT_WORDS) {
        w1_weight_words = (uint32_t)aecct::FFN_W1_WEIGHT_WORDS;
    }
    uint32_t w1_bias_words = d_ffn;
    if (w1_bias_words > (uint32_t)aecct::FFN_W1_BIAS_WORDS) {
        w1_bias_words = (uint32_t)aecct::FFN_W1_BIAS_WORDS;
    }
    uint32_t w2_input_words = token_count * d_ffn;
    if (w2_input_words > (uint32_t)aecct::FFN_W2_INPUT_WORDS) {
        w2_input_words = (uint32_t)aecct::FFN_W2_INPUT_WORDS;
    }
    uint32_t w2_weight_words = d_model * d_ffn;
    if (w2_weight_words > (uint32_t)aecct::FFN_W2_WEIGHT_WORDS) {
        w2_weight_words = (uint32_t)aecct::FFN_W2_WEIGHT_WORDS;
    }
    uint32_t w2_bias_words = d_model;
    if (w2_bias_words > (uint32_t)aecct::FFN_W2_BIAS_WORDS) {
        w2_bias_words = (uint32_t)aecct::FFN_W2_BIAS_WORDS;
    }

    aecct::TransformerLayerW2SeamProbe ready_probe;
    aecct::TransformerLayerW2SeamProbe weight_fallback_probe;
    aecct::TransformerLayerW2SeamProbe bias_fallback_probe;

    const auto run_case =
        [&](aecct::u32_t* sram_mem,
            bool w1_weight_ready,
            bool w1_bias_ready,
            aecct::TransformerLayerW2SeamProbe& probe) {
        // Probe this call boundary to capture W1 weight/bias mainline and fallback branches.
        aecct::clear_transformer_layer_w2_seam_probe(probe);
        init_transformer_case_memory(sram_mem, cfg, sc, pb, (uint32_t)layer_id.to_uint());
        run_fn(
            sram_mem,
            cfg,
            layer_id,
            x_in_base,
            x_out_base,
            sc,
            pb,
            aecct::make_transformer_layer_ffn_topfed_handoff_desc(
                seam_w1_x_words,
                (aecct::u32_t)w1_x_words,
                seam_w1_weight_words,
                w1_weight_ready ? (aecct::u32_t)w1_weight_words : (aecct::u32_t)0u,
                seam_w1_bias_words,
                w1_bias_ready ? (aecct::u32_t)w1_bias_words : (aecct::u32_t)0u,
                seam_w2_input_words,
                (aecct::u32_t)w2_input_words,
                seam_w2_weight_words,
                (aecct::u32_t)w2_weight_words,
                seam_w2_bias_words,
                (aecct::u32_t)w2_bias_words),
            &probe
        );
    };

    run_case(sram_weight_bias_ready, true, true, ready_probe);
    if ((uint32_t)ready_probe.w1_weight_mainline_taken_count.to_uint() != 1u ||
        (uint32_t)ready_probe.w1_weight_fallback_preload_count.to_uint() != 0u ||
        (uint32_t)ready_probe.w1_bias_mainline_taken_count.to_uint() != 1u ||
        (uint32_t)ready_probe.w1_bias_fallback_preload_count.to_uint() != 0u) {
        std::printf(
            "[p11au][FAIL] %s W1 weight/bias ready probe mismatch wm=%u wf=%u bm=%u bf=%u\n",
            case_label,
            (unsigned)ready_probe.w1_weight_mainline_taken_count.to_uint(),
            (unsigned)ready_probe.w1_weight_fallback_preload_count.to_uint(),
            (unsigned)ready_probe.w1_bias_mainline_taken_count.to_uint(),
            (unsigned)ready_probe.w1_bias_fallback_preload_count.to_uint());
        return false;
    }

    run_case(sram_weight_fallback, false, true, weight_fallback_probe);
    if ((uint32_t)weight_fallback_probe.w1_weight_mainline_taken_count.to_uint() != 0u ||
        (uint32_t)weight_fallback_probe.w1_weight_fallback_preload_count.to_uint() != 1u ||
        (uint32_t)weight_fallback_probe.w1_bias_mainline_taken_count.to_uint() != 1u ||
        (uint32_t)weight_fallback_probe.w1_bias_fallback_preload_count.to_uint() != 0u) {
        std::printf(
            "[p11au][FAIL] %s W1 weight fallback probe mismatch wm=%u wf=%u bm=%u bf=%u\n",
            case_label,
            (unsigned)weight_fallback_probe.w1_weight_mainline_taken_count.to_uint(),
            (unsigned)weight_fallback_probe.w1_weight_fallback_preload_count.to_uint(),
            (unsigned)weight_fallback_probe.w1_bias_mainline_taken_count.to_uint(),
            (unsigned)weight_fallback_probe.w1_bias_fallback_preload_count.to_uint());
        return false;
    }

    run_case(sram_bias_fallback, true, false, bias_fallback_probe);
    if ((uint32_t)bias_fallback_probe.w1_weight_mainline_taken_count.to_uint() != 1u ||
        (uint32_t)bias_fallback_probe.w1_weight_fallback_preload_count.to_uint() != 0u ||
        (uint32_t)bias_fallback_probe.w1_bias_mainline_taken_count.to_uint() != 0u ||
        (uint32_t)bias_fallback_probe.w1_bias_fallback_preload_count.to_uint() != 1u) {
        std::printf(
            "[p11au][FAIL] %s W1 bias fallback probe mismatch wm=%u wf=%u bm=%u bf=%u\n",
            case_label,
            (unsigned)bias_fallback_probe.w1_weight_mainline_taken_count.to_uint(),
            (unsigned)bias_fallback_probe.w1_weight_fallback_preload_count.to_uint(),
            (unsigned)bias_fallback_probe.w1_bias_mainline_taken_count.to_uint(),
            (unsigned)bias_fallback_probe.w1_bias_fallback_preload_count.to_uint());
        return false;
    }

    std::printf("W1_WEIGHT_MAINLINE_TAKEN_%s PASS\n", banner_suffix);
    std::printf("W1_WEIGHT_PRELOAD_FALLBACK_NOT_TAKEN_%s PASS\n", banner_suffix);
    std::printf("W1_WEIGHT_PRELOAD_FALLBACK_EXPECTED_%s PASS\n", banner_suffix);
    std::printf("W1_BIAS_MAINLINE_TAKEN_%s PASS\n", banner_suffix);
    std::printf("W1_BIAS_PRELOAD_FALLBACK_NOT_TAKEN_%s PASS\n", banner_suffix);
    std::printf("W1_BIAS_PRELOAD_FALLBACK_EXPECTED_%s PASS\n", banner_suffix);
    return true;
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
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_X_WORDS; ++i) {
        seam_w1_x_words[i] = (aecct::u32_t)0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_WEIGHT_WORDS; ++i) {
        seam_w1_weight_words[i] = (aecct::u32_t)0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_BIAS_WORDS; ++i) {
        seam_w1_bias_words[i] = (aecct::u32_t)0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_INPUT_WORDS; ++i) {
        seam_w2_input_words[i] = (aecct::u32_t)0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_WEIGHT_WORDS; ++i) {
        seam_w2_weight_words[i] = (aecct::u32_t)0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_BIAS_WORDS; ++i) {
        seam_w2_bias_words[i] = (aecct::u32_t)0u;
    }

    aecct::CfgRegs cfg;
    cfg.d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
    cfg.n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
    cfg.d_ffn = (aecct::u32_t)aecct::FFN_D_FFN;
    cfg.n_layers = (aecct::u32_t)1u;

    const aecct::u32_t layer_id = (aecct::u32_t)0u;
    const aecct::u32_t x_in_base = (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD_DEFAULT;
    const aecct::u32_t x_out_base = local_canonical_x_work_base(x_in_base);
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
        aecct::make_transformer_layer_ffn_topfed_handoff_desc(),
        0
    );
    run_fn(
        sram_seam,
        cfg,
        layer_id,
        x_in_base,
        x_out_base,
        sc,
        pb,
        aecct::make_transformer_layer_ffn_topfed_handoff_desc(
            seam_w1_x_words,
            (aecct::u32_t)w1_x_words,
            seam_w1_weight_words,
            (aecct::u32_t)w1_weight_words,
            seam_w1_bias_words,
            (aecct::u32_t)w1_bias_words,
            seam_w2_input_words,
            (aecct::u32_t)w2_input_words,
            seam_w2_weight_words,
            (aecct::u32_t)w2_weight_words,
            seam_w2_bias_words,
            (aecct::u32_t)w2_bias_words),
        0
    );
    run_fn(
        sram_fallback,
        cfg,
        layer_id,
        x_in_base,
        x_out_base,
        sc,
        pb,
        aecct::make_transformer_layer_ffn_topfed_handoff_desc(
            seam_w1_x_words,
            (aecct::u32_t)0u,
            seam_w1_weight_words,
            (aecct::u32_t)0u,
            seam_w1_bias_words,
            (aecct::u32_t)0u,
            seam_w2_input_words,
            (aecct::u32_t)0u,
            seam_w2_weight_words,
            (aecct::u32_t)0u,
            seam_w2_bias_words,
            (aecct::u32_t)0u),
        0
    );

    const uint32_t w1_out_base = (uint32_t)sc.ffn.w1_out_base_word.to_uint();
    uint32_t seam_change_count = 0u;
    const uint32_t compare_words = (w1_bias_words < 16u) ? w1_bias_words : 16u;
    W1_SEAM_COMPARE_LOOP: for (uint32_t i = 0u; i < compare_words; ++i) {
        const uint32_t baseline_w = (uint32_t)sram_baseline[w1_out_base + i].to_uint();
        const uint32_t seam_w = (uint32_t)sram_seam[w1_out_base + i].to_uint();
        const uint32_t fallback_w = (uint32_t)sram_fallback[w1_out_base + i].to_uint();
        if (baseline_w != seam_w) {
            ++seam_change_count;
        }
        if (baseline_w != fallback_w) {
            std::printf(
                "[p11au][FAIL] %s fallback mismatch idx=%u baseline=0x%08X fallback=0x%08X\n",
                case_label,
                (unsigned)i,
                (unsigned)baseline_w,
                (unsigned)fallback_w);
            return false;
        }
    }

    if (seam_change_count == 0u) {
        std::printf(
            "[p11au][FAIL] %s seam did not change W1 output in first %u words\n",
            case_label,
            (unsigned)compare_words);
        return false;
    }

    const uint32_t seam_w1_word0 = (uint32_t)sram_seam[w1_out_base].to_uint();
    const uint32_t fallback_w1_word0 = (uint32_t)sram_fallback[w1_out_base].to_uint();
    if (seam_w1_word0 != 0u) {
        std::printf("[p11au][FAIL] %s seam word0 expected zero got=0x%08X\n", case_label, (unsigned)seam_w1_word0);
        return false;
    }
    if (fallback_w1_word0 == 0u) {
        std::printf("[p11au][FAIL] %s fallback word0 unexpectedly zero\n", case_label);
        return false;
    }

    const uint32_t w2_out_base = (uint32_t)sc.ffn.w2_out_base_word.to_uint();
    uint32_t w2_seam_change_count = 0u;
    const uint32_t w2_compare_words = (w2_bias_words < 16u) ? w2_bias_words : 16u;
    W2_SEAM_COMPARE_LOOP: for (uint32_t i = 0u; i < w2_compare_words; ++i) {
        const uint32_t baseline_w = (uint32_t)sram_baseline[w2_out_base + i].to_uint();
        const uint32_t seam_w = (uint32_t)sram_seam[w2_out_base + i].to_uint();
        const uint32_t fallback_w = (uint32_t)sram_fallback[w2_out_base + i].to_uint();
        if (baseline_w != seam_w) {
            ++w2_seam_change_count;
        }
        if (baseline_w != fallback_w) {
            std::printf(
                "[p11au][FAIL] %s W2 fallback mismatch idx=%u baseline=0x%08X fallback=0x%08X\n",
                case_label,
                (unsigned)i,
                (unsigned)baseline_w,
                (unsigned)fallback_w);
            return false;
        }
    }

    if (w2_seam_change_count == 0u) {
        std::printf(
            "[p11au][FAIL] %s seam did not change W2 output in first %u words\n",
            case_label,
            (unsigned)w2_compare_words);
        return false;
    }
    return true;
}

static bool run_pointer_path_case() {
    const auto pointer_run_fn =
        [](
            aecct::u32_t* sram,
            const aecct::CfgRegs& cfg,
            aecct::u32_t layer_id,
            aecct::u32_t x_in_base,
            aecct::u32_t x_out_base,
            const aecct::LayerScratch& sc,
            const aecct::LayerParamBase& pb,
            aecct::TransformerLayerFfnTopfedHandoffDesc seam_desc,
            aecct::TransformerLayerW2SeamProbe* w2_seam_probe
        ) {
            aecct::TransformerLayer(
                sram,
                cfg,
                layer_id,
                x_in_base,
                x_out_base,
                sc,
                pb,
                true,   // kv_prebuilt_from_top_managed
                true,   // q_prebuilt_from_top_managed
                true,   // score_prebuilt_from_top_managed
                true,   // out_prebuilt_from_top_managed
                true,   // sublayer1_norm_preloaded_by_top
                seam_desc,
                false,
                0,
                (aecct::u32_t)0u,
                true,
                w2_seam_probe
            );
        };
    const bool ok = run_single_path_case(
        "pointer_path",
        pointer_run_fn);
    if (!ok) {
        return false;
    }
    if (!run_w1_input_observability_case("pointer_path", "POINTER_PATH", pointer_run_fn)) {
        return false;
    }
    if (!run_w1_weight_bias_observability_case("pointer_path", "POINTER_PATH", pointer_run_fn)) {
        return false;
    }
    if (!run_w2_weight_bias_observability_case("pointer_path", "POINTER_PATH", pointer_run_fn)) {
        return false;
    }
    std::printf("TRANSFORMER_W1W2_PAYLOAD_SEAM_POINTER_PATH PASS\n");
    return true;
}

static bool run_deep_bridge_path_case() {
    const auto deep_bridge_run_fn =
        [](
            aecct::u32_t* sram,
            const aecct::CfgRegs& cfg,
            aecct::u32_t layer_id,
            aecct::u32_t x_in_base,
            aecct::u32_t x_out_base,
            const aecct::LayerScratch& sc,
            const aecct::LayerParamBase& pb,
            aecct::TransformerLayerFfnTopfedHandoffDesc seam_desc,
            aecct::TransformerLayerW2SeamProbe* w2_seam_probe
        ) {
            // Copy into static array-backed window to exercise the deep bridge entry.
            static aecct::u32_t sram_window[sram_map::SRAM_WORDS_TOTAL];
            for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                sram_window[i] = sram[i];
            }
            aecct::TransformerLayerTopManagedAttnBridge<sram_map::SRAM_WORDS_TOTAL>(
                sram_window,
                cfg,
                layer_id,
                x_in_base,
                x_out_base,
                sc,
                pb,
                true,   // kv_prebuilt_from_top_managed
                true,   // q_prebuilt_from_top_managed
                true,   // score_prebuilt_from_top_managed
                true,   // out_prebuilt_from_top_managed
                true,   // sublayer1_norm_preloaded_by_top
                seam_desc,
                false,
                0,
                (aecct::u32_t)0u,
                true,
                w2_seam_probe
            );
            for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                sram[i] = sram_window[i];
            }
        };
    const bool ok = run_single_path_case(
        "deep_bridge_path",
        deep_bridge_run_fn);
    if (!ok) {
        return false;
    }
    if (!run_w1_input_observability_case("deep_bridge_path", "DEEP_BRIDGE_PATH", deep_bridge_run_fn)) {
        return false;
    }
    if (!run_w1_weight_bias_observability_case("deep_bridge_path", "DEEP_BRIDGE_PATH", deep_bridge_run_fn)) {
        return false;
    }
    if (!run_w2_weight_bias_observability_case("deep_bridge_path", "DEEP_BRIDGE_PATH", deep_bridge_run_fn)) {
        return false;
    }
    std::printf("TRANSFORMER_W1W2_PAYLOAD_SEAM_DEEP_BRIDGE_PATH PASS\n");
    return true;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;

    if (!run_pointer_path_case()) {
        CCS_RETURN(1);
    }
    if (!run_deep_bridge_path_case()) {
        CCS_RETURN(1);
    }
    std::printf("TRANSFORMER_W1W2_PAYLOAD_SEAM_EXPECTED_COMPARE PASS\n");
    std::printf("PASS: tb_transformerlayer_ffn_higher_level_ownership_seam\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
