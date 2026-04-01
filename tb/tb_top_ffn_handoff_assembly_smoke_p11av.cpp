// P11AV: Top mainline lid0 FFN handoff hook-up smoke (local-only).
// Scope:
// - Validate representative mainline loop feeds non-empty lid0 FFN handoff.
// - Validate handoff reaches TransformerLayer through existing dispatch helper.
// - Validate lid!=0/disabled/invalid keep fallback behavior.

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

static void clear_sram(aecct::u32_t* sram) {
    for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = (aecct::u32_t)0u;
    }
}

static void seed_layer_param_words(
    aecct::u32_t* sram,
    uint32_t layer_id,
    uint32_t one_bits
) {
    const bool use_layer1 = (layer_id == 1u);
    const uint32_t w1_bias_id = use_layer1 ? 12u : 4u;
    const uint32_t w1_weight_id = use_layer1 ? 56u : 36u;
    const uint32_t w2_bias_id = use_layer1 ? 13u : 5u;
    const uint32_t w2_weight_id = use_layer1 ? 59u : 39u;
    const uint32_t sublayer1_norm_w_id = use_layer1 ? 63u : 43u;
    const uint32_t sublayer1_norm_b_id = use_layer1 ? 15u : 7u;
    const aecct::LayerParamBase pb =
        aecct::make_layer_param_base((aecct::u32_t)sram_map::W_REGION_BASE, (aecct::u32_t)layer_id);
    const uint32_t param_base = (uint32_t)pb.param_base_word.to_uint();
    const uint32_t w1_weight_base = param_base + kParamMeta[w1_weight_id].offset_w;
    const uint32_t w1_bias_base = param_base + kParamMeta[w1_bias_id].offset_w;
    const uint32_t w2_weight_base = param_base + kParamMeta[w2_weight_id].offset_w;
    const uint32_t w2_bias_base = param_base + kParamMeta[w2_bias_id].offset_w;
    const uint32_t sublayer1_norm_w_base = param_base + kParamMeta[sublayer1_norm_w_id].offset_w;
    const uint32_t sublayer1_norm_b_base = param_base + kParamMeta[sublayer1_norm_b_id].offset_w;

    PARAM_W1_WEIGHT_SEED_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_WEIGHT_WORDS; ++i) {
        sram[w1_weight_base + i] = (aecct::u32_t)one_bits;
    }
    PARAM_W1_BIAS_SEED_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W1_BIAS_WORDS; ++i) {
        sram[w1_bias_base + i] = (aecct::u32_t)one_bits;
    }
    PARAM_W2_WEIGHT_SEED_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_WEIGHT_WORDS; ++i) {
        sram[w2_weight_base + i] = (aecct::u32_t)one_bits;
    }
    PARAM_W2_BIAS_SEED_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::FFN_W2_BIAS_WORDS; ++i) {
        sram[w2_bias_base + i] = (aecct::u32_t)one_bits;
    }
    PARAM_LN_GAMMA_SEED_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::LN_D_MODEL; ++i) {
        sram[sublayer1_norm_w_base + i] = (aecct::u32_t)one_bits;
    }
    PARAM_LN_BETA_SEED_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::LN_D_MODEL; ++i) {
        sram[sublayer1_norm_b_base + i] = (aecct::u32_t)one_bits;
    }
}

static void init_mainline_case_memory(
    aecct::u32_t* sram,
    uint32_t n_layers
) {
    clear_sram(sram);
    const uint32_t one_bits = f32_to_bits(1.0f);

    // Seed the initial X_WORK page that run_transformer_layer_loop consumes.
    const uint32_t x_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
    X_WORK_SEED_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::LN_X_TOTAL_WORDS; ++i) {
        sram[x_base + i] = (aecct::u32_t)one_bits;
    }

    // Seed per-layer FFN/LN parameter slices used by representative path.
    if (n_layers == 0u) {
        n_layers = 1u;
    }
    if (n_layers > 2u) {
        n_layers = 2u;
    }
    PARAM_LAYER_SEED_LOOP: for (uint32_t lid = 0u; lid < n_layers; ++lid) {
        seed_layer_param_words(sram, lid, one_bits);
    }
}

template<typename RunFn>
static bool run_mainline_path_case(const char* case_label, RunFn run_fn) {
    static aecct::u32_t sram_baseline[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_seam[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_invalid[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_lid_probe[sram_map::SRAM_WORDS_TOTAL];

    aecct::TopRegs regs_baseline;
    aecct::TopRegs regs_seam;
    aecct::TopRegs regs_invalid;
    aecct::TopRegs regs_lid_probe;
    regs_baseline.clear();
    regs_seam.clear();
    regs_invalid.clear();
    regs_lid_probe.clear();

    init_mainline_case_memory(sram_baseline, 1u);
    init_mainline_case_memory(sram_seam, 1u);
    init_mainline_case_memory(sram_invalid, 1u);
    init_mainline_case_memory(sram_lid_probe, 2u);

    run_fn(sram_baseline, regs_baseline, 1u, false, true);
    run_fn(sram_seam, regs_seam, 1u, true, true);
    run_fn(sram_invalid, regs_invalid, 1u, true, false);

    // lid!=0 fallback probe in representative mainline path.
    run_fn(sram_lid_probe, regs_lid_probe, 2u, true, true);

    const uint32_t seam_non_empty_count = (uint32_t)regs_seam.p11av_ffn_handoff_non_empty_count.to_uint();
    const uint32_t seam_lid0_non_empty_count = (uint32_t)regs_seam.p11av_lid0_ffn_handoff_non_empty_count.to_uint();
    const uint32_t base_non_empty_count = (uint32_t)regs_baseline.p11av_ffn_handoff_non_empty_count.to_uint();
    const uint32_t invalid_non_empty_count = (uint32_t)regs_invalid.p11av_ffn_handoff_non_empty_count.to_uint();
    if (seam_non_empty_count != 1u || seam_lid0_non_empty_count != 1u) {
        std::printf(
            "[p11av][FAIL] %s seam non-empty count mismatch total=%u lid0=%u\n",
            case_label,
            (unsigned)seam_non_empty_count,
            (unsigned)seam_lid0_non_empty_count);
        return false;
    }
    if (base_non_empty_count != 0u || invalid_non_empty_count != 0u) {
        std::printf(
            "[p11av][FAIL] %s baseline/invalid non-empty count mismatch base=%u invalid=%u\n",
            case_label,
            (unsigned)base_non_empty_count,
            (unsigned)invalid_non_empty_count);
        return false;
    }

    const uint32_t lid_probe_non_empty_count = (uint32_t)regs_lid_probe.p11av_ffn_handoff_non_empty_count.to_uint();
    const uint32_t lid_probe_lid0_non_empty_count = (uint32_t)regs_lid_probe.p11av_lid0_ffn_handoff_non_empty_count.to_uint();
    if (lid_probe_non_empty_count != 1u || lid_probe_lid0_non_empty_count != 1u) {
        std::printf(
            "[p11av][FAIL] %s lid probe expected only lid0 non-empty total=%u lid0=%u\n",
            case_label,
            (unsigned)lid_probe_non_empty_count,
            (unsigned)lid_probe_lid0_non_empty_count);
        return false;
    }

    const aecct::u32_t x_in_base = (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD;
    const aecct::LayerScratch sc = aecct::make_layer_scratch(x_in_base);
    const uint32_t w1_out_base = (uint32_t)sc.ffn.w1_out_base_word.to_uint();
    const uint32_t w2_out_base = (uint32_t)sc.ffn.w2_out_base_word.to_uint();
    const uint32_t compare_words = 16u;
    uint32_t w1_change_count = 0u;
    uint32_t w2_change_count = 0u;
    MAINLINE_COMPARE_LOOP: for (uint32_t i = 0u; i < compare_words; ++i) {
        const uint32_t base_w1 = (uint32_t)sram_baseline[w1_out_base + i].to_uint();
        const uint32_t seam_w1 = (uint32_t)sram_seam[w1_out_base + i].to_uint();
        const uint32_t invalid_w1 = (uint32_t)sram_invalid[w1_out_base + i].to_uint();
        const uint32_t base_w2 = (uint32_t)sram_baseline[w2_out_base + i].to_uint();
        const uint32_t seam_w2 = (uint32_t)sram_seam[w2_out_base + i].to_uint();
        const uint32_t invalid_w2 = (uint32_t)sram_invalid[w2_out_base + i].to_uint();
        if (base_w1 != seam_w1) { ++w1_change_count; }
        if (base_w2 != seam_w2) { ++w2_change_count; }
        if (base_w1 != invalid_w1 || base_w2 != invalid_w2) {
            std::printf("[p11av][FAIL] %s invalid fallback mismatch idx=%u\n", case_label, (unsigned)i);
            return false;
        }
    }
    if (w2_change_count == 0u) {
        std::printf(
            "[p11av][FAIL] %s seam did not change W2 outputs w1=%u w2=%u\n",
            case_label,
            (unsigned)w1_change_count,
            (unsigned)w2_change_count);
        return false;
    }
    return true;
}

static void setup_mainline_regs(aecct::TopRegs& regs, uint32_t n_layers) {
    regs.clear();
    regs.w_base_set = true;
    regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
    regs.cfg_d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
    regs.cfg_n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
    regs.cfg_d_ffn = (aecct::u32_t)aecct::FFN_D_FFN;
    regs.cfg_n_layers = (aecct::u32_t)n_layers;
}

static bool run_pointer_mainline_case() {
    const bool ok = run_mainline_path_case(
        "pointer_mainline",
        [](
            aecct::u32_t* sram,
            aecct::TopRegs& regs,
            uint32_t n_layers,
            bool handoff_enable,
            bool descriptor_valid
        ) {
            setup_mainline_regs(regs, n_layers);
            aecct::run_transformer_layer_loop(
                regs,
                sram,
                handoff_enable,
                descriptor_valid
            );
        });
    if (!ok) { return false; }
    std::printf("TOP_MAINLINE_LID0_FFN_HANDOFF_POINTER_PATH PASS\n");
    return true;
}

static bool run_deep_bridge_mainline_case() {
    const bool ok = run_mainline_path_case(
        "deep_bridge_mainline",
        [](
            aecct::u32_t* sram,
            aecct::TopRegs& regs,
            uint32_t n_layers,
            bool handoff_enable,
            bool descriptor_valid
        ) {
            setup_mainline_regs(regs, n_layers);
            static aecct::u32_t sram_window[sram_map::SRAM_WORDS_TOTAL];
            for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                sram_window[i] = sram[i];
            }
            aecct::run_transformer_layer_loop_top_managed_attn_bridge<sram_map::SRAM_WORDS_TOTAL>(
                regs,
                sram_window,
                handoff_enable,
                descriptor_valid
            );
            for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                sram[i] = sram_window[i];
            }
        });
    if (!ok) { return false; }
    std::printf("TOP_MAINLINE_LID0_FFN_HANDOFF_DEEP_BRIDGE_PATH PASS\n");
    return true;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;

    if (!run_pointer_mainline_case()) {
        CCS_RETURN(1);
    }
    if (!run_deep_bridge_mainline_case()) {
        CCS_RETURN(1);
    }
    std::printf("TOP_MAINLINE_LID0_FFN_HANDOFF_EXPECTED_COMPARE PASS\n");
    std::printf("PASS: tb_top_ffn_handoff_assembly_smoke_p11av\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
