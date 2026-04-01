// P11AV: Top pipeline-level lid0 FFN handoff hook-up smoke (local-only).
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
    const bool use_layer1 = (layer_id != 0u);
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
    const uint32_t infer_in_base = (uint32_t)aecct::IN_BASE_WORD;
    INFER_IN_SEED_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::INFER_IN_WORDS_EXPECTED; ++i) {
        sram[infer_in_base + i] = (aecct::u32_t)one_bits;
    }

    // make_layer_param_base uses lid0 slice and lid!=0 slice.
    // Seed both slices when n_layers > 1 to keep long-loop behavior stable.
    uint32_t seed_layers = 1u;
    if (n_layers > 1u) {
        seed_layers = 2u;
    }
    PARAM_LAYER_SEED_LOOP: for (uint32_t lid = 0u; lid < seed_layers; ++lid) {
        seed_layer_param_words(sram, lid, one_bits);
    }
}

template<typename RunFn>
static bool run_mainline_path_case(
    const char* case_label,
    RunFn run_fn,
    uint32_t n_layers,
    bool require_w2_change
) {
    static aecct::u32_t sram_baseline[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_seam[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_invalid[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u32_t sram_disabled[sram_map::SRAM_WORDS_TOTAL];

    aecct::TopRegs regs_baseline;
    aecct::TopRegs regs_seam;
    aecct::TopRegs regs_invalid;
    aecct::TopRegs regs_disabled;
    regs_baseline.clear();
    regs_seam.clear();
    regs_invalid.clear();
    regs_disabled.clear();

    if (n_layers == 0u) {
        n_layers = 1u;
    }
    init_mainline_case_memory(sram_baseline, n_layers);
    init_mainline_case_memory(sram_seam, n_layers);
    init_mainline_case_memory(sram_invalid, n_layers);
    init_mainline_case_memory(sram_disabled, n_layers);

    run_fn(sram_baseline, regs_baseline, n_layers, false, true);
    run_fn(sram_seam, regs_seam, n_layers, true, true);
    run_fn(sram_invalid, regs_invalid, n_layers, true, false);
    run_fn(sram_disabled, regs_disabled, n_layers, false, false);

    const auto check_case_regs =
        [case_label, n_layers](const char* phase_label, const aecct::TopRegs& regs, bool gate_enable, bool descriptor_valid) -> bool {
        const bool gate_taken = regs.p11aw_pipeline_lid0_ffn_handoff_gate_taken;
        const bool fallback_seen = regs.p11aw_pipeline_lid0_ffn_handoff_fallback_seen;
        const uint32_t non_empty_count = (uint32_t)regs.p11aw_pipeline_ffn_handoff_non_empty_count.to_uint();
        const uint32_t lid0_non_empty_count = (uint32_t)regs.p11aw_pipeline_lid0_ffn_handoff_non_empty_count.to_uint();
        const uint32_t expected_non_empty_count =
            (gate_enable && descriptor_valid) ? 1u : 0u;
        const bool expected_fallback_seen = gate_enable && (expected_non_empty_count < n_layers);

        if (gate_taken != gate_enable) {
            std::printf(
                "[p11av][FAIL] %s %s gate-taken mismatch actual=%u expected=%u\n",
                case_label,
                phase_label,
                (unsigned)(gate_taken ? 1u : 0u),
                (unsigned)(gate_enable ? 1u : 0u));
            return false;
        }
        if (fallback_seen != expected_fallback_seen) {
            std::printf(
                "[p11av][FAIL] %s %s fallback marker mismatch actual=%u expected=%u n_layers=%u\n",
                case_label,
                phase_label,
                (unsigned)(fallback_seen ? 1u : 0u),
                (unsigned)(expected_fallback_seen ? 1u : 0u),
                (unsigned)n_layers);
            return false;
        }
        if (non_empty_count != expected_non_empty_count || lid0_non_empty_count != expected_non_empty_count) {
            std::printf(
                "[p11av][FAIL] %s %s non-empty mismatch total=%u lid0=%u expected=%u n_layers=%u\n",
                case_label,
                phase_label,
                (unsigned)non_empty_count,
                (unsigned)lid0_non_empty_count,
                (unsigned)expected_non_empty_count,
                (unsigned)n_layers);
            return false;
        }
        return true;
    };

    if (!check_case_regs("baseline_gate_off", regs_baseline, false, true)) { return false; }
    if (!check_case_regs("seam_gate_on_valid", regs_seam, true, true)) { return false; }
    if (!check_case_regs("seam_gate_on_invalid", regs_invalid, true, false)) { return false; }
    if (!check_case_regs("handoff_disabled", regs_disabled, false, false)) { return false; }

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
        const uint32_t disabled_w1 = (uint32_t)sram_disabled[w1_out_base + i].to_uint();
        const uint32_t base_w2 = (uint32_t)sram_baseline[w2_out_base + i].to_uint();
        const uint32_t seam_w2 = (uint32_t)sram_seam[w2_out_base + i].to_uint();
        const uint32_t invalid_w2 = (uint32_t)sram_invalid[w2_out_base + i].to_uint();
        const uint32_t disabled_w2 = (uint32_t)sram_disabled[w2_out_base + i].to_uint();
        if (base_w1 != seam_w1) { ++w1_change_count; }
        if (base_w2 != seam_w2) { ++w2_change_count; }
        if (base_w1 != invalid_w1 || base_w2 != invalid_w2 ||
            base_w1 != disabled_w1 || base_w2 != disabled_w2) {
            std::printf("[p11av][FAIL] %s invalid/disabled fallback mismatch idx=%u\n", case_label, (unsigned)i);
            return false;
        }
    }
    if (require_w2_change && w2_change_count == 0u) {
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
    regs.outmode = (aecct::u32_t)2u;
}

static bool run_pointer_mainline_case() {
    const bool ok_short = run_mainline_path_case(
        "pointer_mainline_n1",
        [](
            aecct::u32_t* sram,
            aecct::TopRegs& regs,
            uint32_t n_layers,
            bool handoff_enable,
            bool descriptor_valid
        ) {
            setup_mainline_regs(regs, n_layers);
            regs.p11aw_pipeline_lid0_ffn_handoff_gate_enable = handoff_enable;
            regs.p11aw_pipeline_lid0_ffn_handoff_descriptor_valid = descriptor_valid;
            aecct::data_ch_t data_out;
            (void)aecct::run_infer_pipeline(
                regs,
                sram,
                data_out
            );
        },
        1u,
        true);
    if (!ok_short) { return false; }
    std::printf("TOP_PIPELINE_LID0_FFN_HANDOFF_POINTER_PATH PASS\n");

    const bool ok_long = run_mainline_path_case(
        "pointer_mainline_n4_matrix",
        [](
            aecct::u32_t* sram,
            aecct::TopRegs& regs,
            uint32_t n_layers,
            bool handoff_enable,
            bool descriptor_valid
        ) {
            setup_mainline_regs(regs, n_layers);
            regs.p11aw_pipeline_lid0_ffn_handoff_gate_enable = handoff_enable;
            regs.p11aw_pipeline_lid0_ffn_handoff_descriptor_valid = descriptor_valid;
            aecct::data_ch_t data_out;
            (void)aecct::run_infer_pipeline(
                regs,
                sram,
                data_out
            );
        },
        4u,
        false);
    if (!ok_long) { return false; }
    std::printf("TOP_PIPELINE_LID0_FFN_HANDOFF_LONG_LOOP_POINTER_MATRIX PASS\n");
    return true;
}

static bool run_deep_bridge_mainline_case() {
    const bool ok_short = run_mainline_path_case(
        "deep_bridge_mainline_n1",
        [](
            aecct::u32_t* sram,
            aecct::TopRegs& regs,
            uint32_t n_layers,
            bool handoff_enable,
            bool descriptor_valid
        ) {
            setup_mainline_regs(regs, n_layers);
            regs.p11aw_pipeline_lid0_ffn_handoff_gate_enable = handoff_enable;
            regs.p11aw_pipeline_lid0_ffn_handoff_descriptor_valid = descriptor_valid;
            static aecct::u32_t sram_window[sram_map::SRAM_WORDS_TOTAL];
            for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                sram_window[i] = sram[i];
            }
            aecct::data_ch_t data_out;
            (void)aecct::run_infer_pipeline_top_managed_attn_bridge<sram_map::SRAM_WORDS_TOTAL>(
                regs,
                sram_window,
                data_out
            );
            for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                sram[i] = sram_window[i];
            }
        },
        1u,
        true);
    if (!ok_short) { return false; }
    std::printf("TOP_PIPELINE_LID0_FFN_HANDOFF_DEEP_BRIDGE_PATH PASS\n");

    const bool ok_long = run_mainline_path_case(
        "deep_bridge_mainline_n4_matrix",
        [](
            aecct::u32_t* sram,
            aecct::TopRegs& regs,
            uint32_t n_layers,
            bool handoff_enable,
            bool descriptor_valid
        ) {
            setup_mainline_regs(regs, n_layers);
            regs.p11aw_pipeline_lid0_ffn_handoff_gate_enable = handoff_enable;
            regs.p11aw_pipeline_lid0_ffn_handoff_descriptor_valid = descriptor_valid;
            static aecct::u32_t sram_window[sram_map::SRAM_WORDS_TOTAL];
            for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                sram_window[i] = sram[i];
            }
            aecct::data_ch_t data_out;
            (void)aecct::run_infer_pipeline_top_managed_attn_bridge<sram_map::SRAM_WORDS_TOTAL>(
                regs,
                sram_window,
                data_out
            );
            for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                sram[i] = sram_window[i];
            }
        },
        4u,
        false);
    if (!ok_long) { return false; }
    std::printf("TOP_PIPELINE_LID0_FFN_HANDOFF_LONG_LOOP_DEEP_BRIDGE_MATRIX PASS\n");
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
    std::printf("TOP_PIPELINE_LID0_FFN_HANDOFF_EXPECTED_COMPARE PASS\n");
    std::printf("PASS: tb_top_ffn_handoff_assembly_smoke_p11av\n");
    CCS_RETURN(0);
}

#endif // __SYNTHESIS__
