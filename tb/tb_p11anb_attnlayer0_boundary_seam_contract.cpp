// P11ANB: AttnLayer0 boundary seam contractization smoke (local-only).

#ifndef __SYNTHESIS__

#include <cstdint>
#include <cstdio>
#include <vector>

#include "blocks/AttnLayer0.h"
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

struct LocalAttnLayout {
    uint32_t x_base_word;
    uint32_t out_base_word;
    aecct::AttnScratch sc;
};

static inline uint32_t u32_bits(aecct::u32_t v) {
    return (uint32_t)v.to_uint();
}

static inline void fill_with_pattern(
    std::vector<aecct::u32_t>& sram,
    uint32_t base,
    uint32_t words,
    uint32_t pattern
) {
    for (uint32_t i = 0u; i < words; ++i) {
        sram[base + i] = (aecct::u32_t)(pattern + i);
    }
}

static inline bool check_equal_region(
    const std::vector<aecct::u32_t>& sram,
    uint32_t lhs_base,
    uint32_t rhs_base,
    uint32_t words,
    const char* fail_tag
) {
    for (uint32_t i = 0u; i < words; ++i) {
        const uint32_t lhs = u32_bits(sram[lhs_base + i]);
        const uint32_t rhs = u32_bits(sram[rhs_base + i]);
        if (lhs != rhs) {
            std::printf("[p11anb][FAIL] %s mismatch idx=%u lhs=0x%08X rhs=0x%08X\n",
                fail_tag,
                (unsigned)i,
                (unsigned)lhs,
                (unsigned)rhs);
            return false;
        }
    }
    return true;
}

static inline bool check_region_untouched(
    const std::vector<aecct::u32_t>& sram,
    uint32_t base,
    uint32_t words,
    uint32_t pattern,
    const char* fail_tag
) {
    for (uint32_t i = 0u; i < words; ++i) {
        const uint32_t got = u32_bits(sram[base + i]);
        const uint32_t exp = pattern + i;
        if (got != exp) {
            std::printf("[p11anb][FAIL] %s touched idx=%u got=0x%08X exp=0x%08X\n",
                fail_tag,
                (unsigned)i,
                (unsigned)got,
                (unsigned)exp);
            return false;
        }
    }
    return true;
}

static inline bool check_equal_region_to_words(
    const std::vector<aecct::u32_t>& sram,
    uint32_t lhs_base,
    const std::vector<aecct::u32_t>& rhs_words,
    uint32_t words,
    const char* fail_tag
) {
    for (uint32_t i = 0u; i < words; ++i) {
        const uint32_t lhs = u32_bits(sram[lhs_base + i]);
        const uint32_t rhs = u32_bits(rhs_words[i]);
        if (lhs != rhs) {
            std::printf("[p11anb][FAIL] %s mismatch idx=%u lhs=0x%08X rhs=0x%08X\n",
                fail_tag,
                (unsigned)i,
                (unsigned)lhs,
                (unsigned)rhs);
            return false;
        }
    }
    return true;
}

static inline uint32_t count_region_diff(
    const std::vector<aecct::u32_t>& lhs,
    const std::vector<aecct::u32_t>& rhs,
    uint32_t base,
    uint32_t words
) {
    uint32_t diff_count = 0u;
    for (uint32_t i = 0u; i < words; ++i) {
        const uint32_t lhs_v = u32_bits(lhs[base + i]);
        const uint32_t rhs_v = u32_bits(rhs[base + i]);
        if (lhs_v != rhs_v) {
            ++diff_count;
        }
    }
    return diff_count;
}

static inline aecct::u32_t local_alternate_x_page(aecct::u32_t x_base_word) {
    const uint32_t x_base = (uint32_t)x_base_word.to_uint();
    if (x_base == (uint32_t)sram_map::X_PAGE0_BASE_W) {
        return (aecct::u32_t)sram_map::X_PAGE1_BASE_W;
    }
    return (aecct::u32_t)sram_map::X_PAGE0_BASE_W;
}

class TbP11anbAttnLayer0BoundarySeamContract {
public:
    int run_all() {
        if (!run_qkv_descriptor_consume_case()) { return 1; }
        if (!run_qkv_descriptor_skip_case()) { return 1; }
        if (!run_out_descriptor_gate_case()) { return 1; }
        if (!run_out_topfed_payload_deeper_consume_case()) { return 1; }
        if (!run_transformerlayer_out_topfed_mapping_pointer_case()) { return 1; }
        if (!run_transformerlayer_out_topfed_mapping_deep_bridge_case()) { return 1; }
        if (!run_legacy_descriptor_equivalence_case()) { return 1; }
        std::printf("P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_MAPPING_EXPECTED_COMPARE PASS\n");
        std::printf("PASS: tb_p11anb_attnlayer0_boundary_seam_contract\n");
        return 0;
    }

private:
    static const uint32_t kSramWords = (uint32_t)::sram_map::SRAM_WORDS_TOTAL;

    static aecct::AttnCfg make_attn_cfg() {
        aecct::AttnCfg cfg;
        cfg.token_count = (aecct::u32_t)aecct::ATTN_TOKEN_COUNT;
        cfg.d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
        cfg.n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
        cfg.d_head = (aecct::u32_t)aecct::ATTN_D_HEAD;
        return cfg;
    }

    static LocalAttnLayout make_local_layout() {
        LocalAttnLayout layout;
        const uint32_t tensor_words = (uint32_t)aecct::ATTN_TENSOR_WORDS;
        const uint32_t guard_words = 32u;
        uint32_t cursor = 64u;

        layout.x_base_word = cursor;
        cursor += tensor_words + guard_words;
        layout.sc.q_base_word = (aecct::u32_t)cursor;
        cursor += tensor_words + guard_words;
        layout.sc.k_base_word = (aecct::u32_t)cursor;
        cursor += tensor_words + guard_words;
        layout.sc.v_base_word = (aecct::u32_t)cursor;
        cursor += tensor_words + guard_words;
        layout.sc.score_base_word = (aecct::u32_t)cursor;
        cursor += tensor_words + guard_words;
        layout.sc.softmax_base_word = (aecct::u32_t)cursor;
        cursor += tensor_words + guard_words;
        layout.sc.pre_concat_base_word = (aecct::u32_t)cursor;
        cursor += tensor_words + guard_words;
        layout.sc.post_concat_base_word = (aecct::u32_t)cursor;
        cursor += tensor_words + guard_words;
        layout.sc.q_act_q_base_word = (aecct::u32_t)cursor;
        cursor += tensor_words + guard_words;
        layout.sc.k_act_q_base_word = (aecct::u32_t)cursor;
        cursor += tensor_words + guard_words;
        layout.sc.v_act_q_base_word = (aecct::u32_t)cursor;
        cursor += tensor_words + guard_words;
        layout.sc.q_sx_base_word = (aecct::u32_t)cursor;
        cursor += guard_words;
        layout.out_base_word = cursor;
        return layout;
    }

    static bool layout_fits_local_sram(const LocalAttnLayout& layout) {
        const uint32_t tensor_words = (uint32_t)aecct::ATTN_TENSOR_WORDS;
        const uint32_t max_addr =
            layout.out_base_word +
            tensor_words;
        return max_addr < kSramWords;
    }

    bool run_qkv_descriptor_consume_case() {
        std::vector<aecct::u32_t> sram(kSramWords, (aecct::u32_t)0u);
        const aecct::AttnCfg cfg = make_attn_cfg();
        const LocalAttnLayout layout = make_local_layout();
        if (!layout_fits_local_sram(layout)) {
            std::printf("[p11anb][FAIL] local SRAM too small for QKV consume case\n");
            return false;
        }
        const uint32_t words = (uint32_t)aecct::ATTN_TENSOR_WORDS;
        const uint32_t x_base = layout.x_base_word;
        const uint32_t q_base = (uint32_t)layout.sc.q_base_word.to_uint();
        const uint32_t k_base = (uint32_t)layout.sc.k_base_word.to_uint();
        const uint32_t v_base = (uint32_t)layout.sc.v_base_word.to_uint();
        const uint32_t q_act_q_base = (uint32_t)layout.sc.q_act_q_base_word.to_uint();
        const uint32_t k_act_q_base = (uint32_t)layout.sc.k_act_q_base_word.to_uint();
        const uint32_t v_act_q_base = (uint32_t)layout.sc.v_act_q_base_word.to_uint();

        fill_with_pattern(sram, x_base, words, 0x3F000000u);
        fill_with_pattern(sram, q_base, words, 0xAA000000u);
        fill_with_pattern(sram, k_base, words, 0xBB000000u);
        fill_with_pattern(sram, v_base, words, 0xCC000000u);
        fill_with_pattern(sram, q_act_q_base, words, 0xDD000000u);
        fill_with_pattern(sram, k_act_q_base, words, 0xEE000000u);
        fill_with_pattern(sram, v_act_q_base, words, 0xFF000000u);

        const aecct::AttnLayer0PrebuiltHandoffDesc handoff =
            aecct::make_attn_layer0_prebuilt_handoff_desc(false, false, false, false);
        aecct::AttnLayer0<aecct::ATTN_STAGE_QKV>(
            sram.data(),
            cfg,
            (aecct::u32_t)x_base,
            (aecct::u32_t)layout.out_base_word,
            layout.sc,
            (aecct::u32_t)0u,
            handoff);

        if (!check_equal_region(sram, q_base, x_base, words, "Q-consume")) { return false; }
        if (!check_equal_region(sram, k_base, x_base, words, "K-consume")) { return false; }
        if (!check_equal_region(sram, v_base, x_base, words, "V-consume")) { return false; }
        if (!check_equal_region(sram, q_act_q_base, x_base, words, "QACT-consume")) { return false; }
        if (!check_equal_region(sram, k_act_q_base, x_base, words, "KACT-consume")) { return false; }
        if (!check_equal_region(sram, v_act_q_base, x_base, words, "VACT-consume")) { return false; }

        std::printf("P11ANB_ATTNLAYER0_QKV_DESCRIPTOR_CONSUME PASS\n");
        return true;
    }

    bool run_qkv_descriptor_skip_case() {
        std::vector<aecct::u32_t> sram(kSramWords, (aecct::u32_t)0u);
        const aecct::AttnCfg cfg = make_attn_cfg();
        const LocalAttnLayout layout = make_local_layout();
        if (!layout_fits_local_sram(layout)) {
            std::printf("[p11anb][FAIL] local SRAM too small for QKV skip case\n");
            return false;
        }
        const uint32_t words = (uint32_t)aecct::ATTN_TENSOR_WORDS;
        const uint32_t x_base = layout.x_base_word;
        const uint32_t q_base = (uint32_t)layout.sc.q_base_word.to_uint();
        const uint32_t k_base = (uint32_t)layout.sc.k_base_word.to_uint();
        const uint32_t v_base = (uint32_t)layout.sc.v_base_word.to_uint();
        const uint32_t q_pat = 0x11000000u;
        const uint32_t k_pat = 0x22000000u;
        const uint32_t v_pat = 0x33000000u;

        fill_with_pattern(sram, x_base, words, 0x3F100000u);
        fill_with_pattern(sram, q_base, words, q_pat);
        fill_with_pattern(sram, k_base, words, k_pat);
        fill_with_pattern(sram, v_base, words, v_pat);

        const aecct::AttnLayer0PrebuiltHandoffDesc handoff =
            aecct::make_attn_layer0_prebuilt_handoff_desc(true, true, false, false);
        aecct::AttnLayer0<aecct::ATTN_STAGE_QKV>(
            sram.data(),
            cfg,
            (aecct::u32_t)x_base,
            (aecct::u32_t)layout.out_base_word,
            layout.sc,
            (aecct::u32_t)0u,
            handoff);

        if (!check_region_untouched(sram, q_base, words, q_pat, "Q-skip")) { return false; }
        if (!check_region_untouched(sram, k_base, words, k_pat, "K-skip")) { return false; }
        if (!check_region_untouched(sram, v_base, words, v_pat, "V-skip")) { return false; }

        std::printf("P11ANB_ATTNLAYER0_QKV_DESCRIPTOR_SKIP PASS\n");
        return true;
    }

    bool run_out_descriptor_gate_case() {
        std::vector<aecct::u32_t> sram(kSramWords, (aecct::u32_t)0u);
        const aecct::AttnCfg cfg = make_attn_cfg();
        const LocalAttnLayout layout = make_local_layout();
        if (!layout_fits_local_sram(layout)) {
            std::printf("[p11anb][FAIL] local SRAM too small for OUT gate case\n");
            return false;
        }
        const uint32_t words = (uint32_t)aecct::ATTN_TENSOR_WORDS;
        const uint32_t post_base = (uint32_t)layout.sc.post_concat_base_word.to_uint();
        const uint32_t out_base = layout.out_base_word;

        fill_with_pattern(sram, post_base, words, 0x44000000u);
        fill_with_pattern(sram, out_base, words, 0x55000000u);

        const aecct::AttnLayer0PrebuiltHandoffDesc consume_handoff =
            aecct::make_attn_layer0_prebuilt_handoff_desc(false, false, false, false);
        aecct::AttnLayer0<aecct::ATTN_STAGE_OUT>(
            sram.data(),
            cfg,
            (aecct::u32_t)layout.x_base_word,
            (aecct::u32_t)out_base,
            layout.sc,
            (aecct::u32_t)0u,
            consume_handoff);

        if (!check_equal_region(sram, out_base, post_base, words, "OUT-consume")) { return false; }
        std::printf("P11ANB_ATTNLAYER0_OUT_DESCRIPTOR_CONSUME PASS\n");

        fill_with_pattern(sram, out_base, words, 0x66000000u);
        fill_with_pattern(sram, post_base, words, 0x77000000u);
        const aecct::AttnLayer0PrebuiltHandoffDesc skip_handoff =
            aecct::make_attn_layer0_prebuilt_handoff_desc(false, false, false, true);
        aecct::AttnLayer0<aecct::ATTN_STAGE_OUT>(
            sram.data(),
            cfg,
            (aecct::u32_t)layout.x_base_word,
            (aecct::u32_t)out_base,
            layout.sc,
            (aecct::u32_t)0u,
            skip_handoff);

        if (!check_region_untouched(sram, out_base, words, 0x66000000u, "OUT-anti-fallback")) {
            return false;
        }
        std::printf("P11ANB_ATTNLAYER0_OUT_DESCRIPTOR_ANTI_FALLBACK PASS\n");
        return true;
    }

    bool run_out_topfed_payload_deeper_consume_case() {
        std::vector<aecct::u32_t> sram(kSramWords, (aecct::u32_t)0u);
        const aecct::AttnCfg cfg = make_attn_cfg();
        const LocalAttnLayout layout = make_local_layout();
        if (!layout_fits_local_sram(layout)) {
            std::printf("[p11anb][FAIL] local SRAM too small for OUT topfed consume case\n");
            return false;
        }
        const uint32_t words = (uint32_t)aecct::ATTN_TENSOR_WORDS;
        const uint32_t post_base = (uint32_t)layout.sc.post_concat_base_word.to_uint();
        const uint32_t out_base = layout.out_base_word;

        std::vector<aecct::u32_t> topfed_words(words, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < words; ++i) {
            topfed_words[i] = (aecct::u32_t)(0x9A000000u + i);
        }

        fill_with_pattern(sram, post_base, words, 0x8A000000u);
        fill_with_pattern(sram, out_base, words, 0x8B000000u);
        const aecct::AttnLayer0PrebuiltHandoffDesc topfed_handoff =
            aecct::make_attn_layer0_prebuilt_handoff_desc(
                false,
                false,
                false,
                false,
                true,
                topfed_words.data(),
                (aecct::u32_t)words);
        aecct::AttnLayer0<aecct::ATTN_STAGE_OUT>(
            sram.data(),
            cfg,
            (aecct::u32_t)layout.x_base_word,
            (aecct::u32_t)out_base,
            layout.sc,
            (aecct::u32_t)0u,
            topfed_handoff);

        if (!check_equal_region_to_words(sram, out_base, topfed_words, words, "OUT-topfed-consume")) {
            return false;
        }
        std::printf("P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_CONSUME PASS\n");

        fill_with_pattern(sram, post_base, words, 0x8C000000u);
        fill_with_pattern(sram, out_base, words, 0x8D000000u);
        const aecct::AttnLayer0PrebuiltHandoffDesc invalid_handoff =
            aecct::make_attn_layer0_prebuilt_handoff_desc(
                false,
                false,
                false,
                false,
                true,
                topfed_words.data(),
                (aecct::u32_t)(words - 1u));
        aecct::AttnLayer0<aecct::ATTN_STAGE_OUT>(
            sram.data(),
            cfg,
            (aecct::u32_t)layout.x_base_word,
            (aecct::u32_t)out_base,
            layout.sc,
            (aecct::u32_t)0u,
            invalid_handoff);

        if (!check_equal_region(sram, out_base, post_base, words, "OUT-topfed-invalid-fallback")) {
            return false;
        }
        std::printf("P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_INVALID_FALLBACK PASS\n");

        fill_with_pattern(sram, post_base, words, 0x8E000000u);
        fill_with_pattern(sram, out_base, words, 0x8F000000u);
        const aecct::AttnLayer0PrebuiltHandoffDesc disabled_handoff =
            aecct::make_attn_layer0_prebuilt_handoff_desc(
                false,
                false,
                false,
                false,
                false,
                topfed_words.data(),
                (aecct::u32_t)words);
        aecct::AttnLayer0<aecct::ATTN_STAGE_OUT>(
            sram.data(),
            cfg,
            (aecct::u32_t)layout.x_base_word,
            (aecct::u32_t)out_base,
            layout.sc,
            (aecct::u32_t)0u,
            disabled_handoff);

        if (!check_equal_region(sram, out_base, post_base, words, "OUT-topfed-disabled-fallback")) {
            return false;
        }
        std::printf("P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_DISABLED_FALLBACK PASS\n");
        return true;
    }

    template<typename RunFn>
    bool run_transformerlayer_out_topfed_mapping_case(
        const char* consume_pass_banner,
        const char* invalid_fallback_pass_banner,
        const char* disabled_fallback_pass_banner,
        RunFn run_fn
    ) {
        const aecct::AttnCfg attn_cfg = make_attn_cfg();
        const uint32_t words = (uint32_t)aecct::ATTN_TENSOR_WORDS;
        if (words == 0u) {
            std::printf("[p11anb][FAIL] transformer mapping words must be non-zero\n");
            return false;
        }

        aecct::CfgRegs cfg;
        cfg.d_model = attn_cfg.d_model;
        cfg.n_heads = attn_cfg.n_heads;
        cfg.d_ffn = (aecct::u32_t)aecct::FFN_D_FFN;
        cfg.n_layers = (aecct::u32_t)1u;

        const aecct::u32_t layer_id = (aecct::u32_t)0u;
        const aecct::u32_t x_in_base = (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD_DEFAULT;
        const aecct::u32_t x_out_base = local_alternate_x_page(x_in_base);
        const aecct::LayerScratch sc = aecct::make_layer_scratch(x_in_base);
        const aecct::LayerParamBase pb =
            aecct::make_layer_param_base((aecct::u32_t)sram_map::W_REGION_BASE, layer_id);
        const uint32_t post_base = (uint32_t)sc.attn.post_concat_base_word.to_uint();
        const uint32_t out_base = (uint32_t)sc.attn_out_base_word.to_uint();

        std::vector<aecct::u32_t> sram_baseline(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_valid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_invalid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_disabled(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> topfed_words(words, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> fallback_words(words, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < words; ++i) {
            topfed_words[i] = (aecct::u32_t)(0xA5000000u + i);
            fallback_words[i] = (aecct::u32_t)(0xA1000000u + i);
        }

        const uint32_t fallback_pattern = 0xA1000000u;
        fill_with_pattern(sram_baseline, post_base, words, fallback_pattern);
        fill_with_pattern(sram_valid, post_base, words, fallback_pattern);
        fill_with_pattern(sram_invalid, post_base, words, fallback_pattern);
        fill_with_pattern(sram_disabled, post_base, words, fallback_pattern);

        fill_with_pattern(sram_baseline, out_base, words, 0xA2000000u);
        fill_with_pattern(sram_valid, out_base, words, 0xA3000000u);
        fill_with_pattern(sram_invalid, out_base, words, 0xA4000000u);
        fill_with_pattern(sram_disabled, out_base, words, 0xA6000000u);

        run_fn(
            sram_baseline,
            cfg,
            layer_id,
            x_in_base,
            x_out_base,
            sc,
            pb,
            false,
            (aecct::u32_t)0u,
            topfed_words.data());
        run_fn(
            sram_valid,
            cfg,
            layer_id,
            x_in_base,
            x_out_base,
            sc,
            pb,
            true,
            (aecct::u32_t)words,
            topfed_words.data());
        run_fn(
            sram_invalid,
            cfg,
            layer_id,
            x_in_base,
            x_out_base,
            sc,
            pb,
            true,
            (aecct::u32_t)(words - 1u),
            topfed_words.data());
        run_fn(
            sram_disabled,
            cfg,
            layer_id,
            x_in_base,
            x_out_base,
            sc,
            pb,
            false,
            (aecct::u32_t)words,
            topfed_words.data());

        if (!check_equal_region_to_words(sram_baseline, out_base, fallback_words, words, "TRANSFORMER-OUT-baseline-fallback")) {
            return false;
        }
        if (!check_equal_region_to_words(sram_valid, out_base, topfed_words, words, "TRANSFORMER-OUT-topfed-consume")) {
            return false;
        }
        if (!check_equal_region_to_words(sram_invalid, out_base, fallback_words, words, "TRANSFORMER-OUT-invalid-fallback")) {
            return false;
        }
        if (!check_equal_region_to_words(sram_disabled, out_base, fallback_words, words, "TRANSFORMER-OUT-disabled-fallback")) {
            return false;
        }

        const uint32_t diff_words = (words < 16u) ? words : 16u;
        const uint32_t diff_count = count_region_diff(sram_valid, sram_baseline, out_base, diff_words);
        if (diff_count == 0u) {
            std::printf("[p11anb][FAIL] transformer mapping expected compare unchanged first=%u words\n",
                (unsigned)diff_words);
            return false;
        }

        std::printf("%s PASS\n", consume_pass_banner);
        std::printf("%s PASS\n", invalid_fallback_pass_banner);
        std::printf("%s PASS\n", disabled_fallback_pass_banner);
        return true;
    }

    bool run_transformerlayer_out_topfed_mapping_pointer_case() {
        return run_transformerlayer_out_topfed_mapping_case(
            "P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_POINTER_MAPPING_CONSUME",
            "P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_POINTER_INVALID_FALLBACK",
            "P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_POINTER_DISABLED_FALLBACK",
            [](
                std::vector<aecct::u32_t>& sram,
                const aecct::CfgRegs& cfg,
                aecct::u32_t layer_id,
                aecct::u32_t x_in_base,
                aecct::u32_t x_out_base,
                const aecct::LayerScratch& sc,
                const aecct::LayerParamBase& pb,
                bool out_topfed_payload_enable,
                aecct::u32_t out_topfed_payload_words_valid,
                const aecct::u32_t* out_topfed_payload_words
            ) {
                aecct::TransformerLayer(
                    sram.data(),
                    cfg,
                    layer_id,
                    x_in_base,
                    x_out_base,
                    sc,
                    pb,
                    true,   // kv_prebuilt_from_top_managed
                    true,   // q_prebuilt_from_top_managed
                    true,   // score_prebuilt_from_top_managed
                    false,  // out_prebuilt_from_top_managed
                    true,   // sublayer1_norm_preloaded_by_top
                    aecct::make_transformer_layer_ffn_topfed_handoff_desc(),
                    out_topfed_payload_enable,
                    out_topfed_payload_words,
                    out_topfed_payload_words_valid);
            });
    }

    bool run_transformerlayer_out_topfed_mapping_deep_bridge_case() {
        return run_transformerlayer_out_topfed_mapping_case(
            "P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_DEEP_BRIDGE_MAPPING_CONSUME",
            "P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_DEEP_BRIDGE_INVALID_FALLBACK",
            "P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_DEEP_BRIDGE_DISABLED_FALLBACK",
            [](
                std::vector<aecct::u32_t>& sram,
                const aecct::CfgRegs& cfg,
                aecct::u32_t layer_id,
                aecct::u32_t x_in_base,
                aecct::u32_t x_out_base,
                const aecct::LayerScratch& sc,
                const aecct::LayerParamBase& pb,
                bool out_topfed_payload_enable,
                aecct::u32_t out_topfed_payload_words_valid,
                const aecct::u32_t* out_topfed_payload_words
            ) {
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
                    false,  // out_prebuilt_from_top_managed
                    true,   // sublayer1_norm_preloaded_by_top
                    aecct::make_transformer_layer_ffn_topfed_handoff_desc(),
                    out_topfed_payload_enable,
                    out_topfed_payload_words,
                    out_topfed_payload_words_valid);
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram[i] = sram_window[i];
                }
            });
    }

    bool run_legacy_descriptor_equivalence_case() {
        std::vector<aecct::u32_t> sram_legacy(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_descriptor(kSramWords, (aecct::u32_t)0u);
        const aecct::AttnCfg cfg = make_attn_cfg();
        const LocalAttnLayout layout = make_local_layout();
        if (!layout_fits_local_sram(layout)) {
            std::printf("[p11anb][FAIL] local SRAM too small for equivalence case\n");
            return false;
        }
        const uint32_t words = (uint32_t)aecct::ATTN_TENSOR_WORDS;
        const uint32_t post_base = (uint32_t)layout.sc.post_concat_base_word.to_uint();
        const uint32_t out_base = layout.out_base_word;
        fill_with_pattern(sram_legacy, post_base, words, 0x88000000u);
        fill_with_pattern(sram_descriptor, post_base, words, 0x88000000u);

        aecct::AttnLayer0<aecct::ATTN_STAGE_OUT>(
            sram_legacy.data(),
            cfg,
            (aecct::u32_t)layout.x_base_word,
            (aecct::u32_t)out_base,
            layout.sc,
            (aecct::u32_t)0u,
            false,
            false,
            false,
            false);
        const aecct::AttnLayer0PrebuiltHandoffDesc handoff =
            aecct::make_attn_layer0_prebuilt_handoff_desc(false, false, false, false);
        aecct::AttnLayer0<aecct::ATTN_STAGE_OUT>(
            sram_descriptor.data(),
            cfg,
            (aecct::u32_t)layout.x_base_word,
            (aecct::u32_t)out_base,
            layout.sc,
            (aecct::u32_t)0u,
            handoff);

        for (uint32_t i = 0u; i < words; ++i) {
            const uint32_t lhs = u32_bits(sram_legacy[out_base + i]);
            const uint32_t rhs = u32_bits(sram_descriptor[out_base + i]);
            if (lhs != rhs) {
                std::printf("[p11anb][FAIL] legacy-descriptor equivalence mismatch idx=%u lhs=0x%08X rhs=0x%08X\n",
                    (unsigned)i,
                    (unsigned)lhs,
                    (unsigned)rhs);
                return false;
            }
        }
        std::printf("P11ANB_ATTNLAYER0_LEGACY_DESCRIPTOR_EQUIVALENCE PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbP11anbAttnLayer0BoundarySeamContract tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__
