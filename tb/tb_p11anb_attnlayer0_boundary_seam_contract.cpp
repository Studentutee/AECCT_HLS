// P11ANB: AttnLayer0 boundary seam contractization smoke (local-only).

#ifndef __SYNTHESIS__

#include <cstdint>
#include <cstdio>
#include <vector>

#include "Top.h"
#include "tb_p11aeaf_common.h"
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

static inline bool check_equal_region_between_srams(
    const std::vector<aecct::u32_t>& lhs_sram,
    const std::vector<aecct::u32_t>& rhs_sram,
    uint32_t base,
    uint32_t words,
    const char* fail_tag
) {
    for (uint32_t i = 0u; i < words; ++i) {
        const uint32_t lhs = u32_bits(lhs_sram[base + i]);
        const uint32_t rhs = u32_bits(rhs_sram[base + i]);
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
        if (!run_transformerlayer_attn_shell_shrink_selector_case()) { return 1; }
        if (!run_transformerlayer_out_topfed_mapping_pointer_case()) { return 1; }
        if (!run_transformerlayer_out_topfed_mapping_deep_bridge_case()) { return 1; }
        if (!run_top_caller_out_topfed_mapping_pointer_case()) { return 1; }
        if (!run_top_caller_out_topfed_mapping_deep_bridge_case()) { return 1; }
        if (!run_loop_caller_out_topfed_mapping_pointer_case()) { return 1; }
        if (!run_loop_caller_out_topfed_mapping_deep_bridge_case()) { return 1; }
        if (!run_loop_caller_qkscore_mask_handoff_pointer_case()) { return 1; }
        if (!run_loop_caller_qkscore_mask_handoff_deep_bridge_case()) { return 1; }
        if (!run_loop_caller_qkscore_kvscan_handoff_pointer_case()) { return 1; }
        if (!run_loop_caller_qkscore_kvscan_handoff_deep_bridge_case()) { return 1; }
        if (!run_loop_caller_qkscore_qsrc_handoff_pointer_case()) { return 1; }
        if (!run_loop_caller_qkscore_qsrc_handoff_deep_bridge_case()) { return 1; }
        if (!run_loop_caller_qkscore_wq_handoff_pointer_case()) { return 1; }
        if (!run_loop_caller_qkscore_wq_handoff_deep_bridge_case()) { return 1; }
        if (!run_loop_caller_qkscore_multi_seam_mask_wq_priority_pointer_case()) { return 1; }
        if (!run_loop_caller_qkscore_multi_seam_mask_qsrc_priority_pointer_case()) { return 1; }
        if (!run_loop_caller_qkscore_multi_seam_wq_qsrc_priority_pointer_case()) { return 1; }
        if (!run_loop_caller_qkscore_multi_seam_qsrc_kvscan_priority_pointer_case()) { return 1; }
        if (!run_legacy_descriptor_equivalence_case()) { return 1; }
        std::printf("P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_MAPPING_EXPECTED_COMPARE PASS\n");
        std::printf("P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_CHAIN_EXPECTED_COMPARE PASS\n");
        std::printf("P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_HOOK_EXPECTED_COMPARE PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MASK_HOOK_EXPECTED_COMPARE PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_KVSCAN_HOOK_EXPECTED_COMPARE PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_QSRC_HOOK_EXPECTED_COMPARE PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_WQ_HOOK_EXPECTED_COMPARE PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_WQ_PRIORITY_EXPECTED_COMPARE PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_QSRC_PRIORITY_EXPECTED_COMPARE PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_WQ_QSRC_PRIORITY_EXPECTED_COMPARE PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_QSRC_KVSCAN_PRIORITY_EXPECTED_COMPARE PASS\n");
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

        // Prebuilt ownership seam: committed output must remain untouched even when payload descriptor is invalid.
        fill_with_pattern(sram, post_base, words, 0x8E000000u);
        fill_with_pattern(sram, out_base, words, 0x8F000000u);
        const aecct::AttnLayer0PrebuiltHandoffDesc prebuilt_invalid_handoff =
            aecct::make_attn_layer0_prebuilt_handoff_desc(
                false,
                false,
                false,
                true,
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
            prebuilt_invalid_handoff);

        if (!check_region_untouched(sram, out_base, words, 0x8F000000u, "OUT-topfed-invalid-prebuilt-skip")) {
            return false;
        }
        std::printf("P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_INVALID_PREBUILT_SKIP PASS\n");

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

    bool run_transformerlayer_attn_shell_shrink_selector_case() {
        const aecct::TransformerAttnCompatShellStage fully_prebuilt_payload_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                true,
                true,
                true,
                true,
                true);
        if (fully_prebuilt_payload_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
            std::printf(
                "[p11anb][FAIL] fully-prebuilt payload stage mismatch got=%u exp=%u\n",
                (unsigned)fully_prebuilt_payload_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_FULLY_PREBUILT_OUT_ONLY PASS\n");

        const aecct::TransformerAttnCompatShellStage fully_prebuilt_no_payload_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                true,
                true,
                true,
                true,
                false);
        if (fully_prebuilt_no_payload_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_DISABLED) {
            std::printf(
                "[p11anb][FAIL] fully-prebuilt no-payload stage mismatch got=%u exp=%u\n",
                (unsigned)fully_prebuilt_no_payload_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_DISABLED);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_FULLY_PREBUILT_NO_PAYLOAD_DISABLED PASS\n");

        const aecct::TransformerAttnCompatShellStage selected_partial_prebuilt_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                true,
                true,
                true,
                false,
                false);
        if (selected_partial_prebuilt_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
            std::printf(
                "[p11anb][FAIL] selected partial-prebuilt stage mismatch got=%u exp=%u\n",
                (unsigned)selected_partial_prebuilt_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_SELECTED_PARTIAL_QKV_SCORE_NO_PAYLOAD_OUT_ONLY PASS\n");

        const aecct::TransformerAttnCompatShellStage selected_partial_prebuilt_payload_enabled_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                true,
                true,
                true,
                false,
                true);
        if (selected_partial_prebuilt_payload_enabled_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
            std::printf(
                "[p11anb][FAIL] selected partial-prebuilt payload-enabled stage mismatch got=%u exp=%u\n",
                (unsigned)selected_partial_prebuilt_payload_enabled_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_SELECTED_PARTIAL_QKV_SCORE_PAYLOAD_ENABLED_OUT_ONLY PASS\n");

        const aecct::TransformerAttnCompatShellStage other_partial_prebuilt_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                true,
                false,
                false,
                false,
                true);
        if (other_partial_prebuilt_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_FULL) {
            std::printf(
                "[p11anb][FAIL] other partial-prebuilt stage mismatch got=%u exp=%u\n",
                (unsigned)other_partial_prebuilt_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_FULL);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_OTHER_PARTIAL_STILL_FULL PASS\n");

        const aecct::TransformerAttnCompatShellStage qkv_ready_score_not_prebuilt_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                true,
                true,
                false,
                false,
                false);
        if (qkv_ready_score_not_prebuilt_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY) {
            std::printf(
                "[p11anb][FAIL] qkv-ready score-not-prebuilt stage mismatch got=%u exp=%u\n",
                (unsigned)qkv_ready_score_not_prebuilt_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_QKV_READY_SCORE_NOT_PREBUILT_TO_SCORES_STAGE PASS\n");

        const aecct::TransformerAttnCompatShellStage q_ready_kv_not_prebuilt_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                false,
                true,
                false,
                false,
                false);
        if (q_ready_kv_not_prebuilt_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY) {
            std::printf(
                "[p11anb][FAIL] q-ready kv-not-prebuilt stage mismatch got=%u exp=%u\n",
                (unsigned)q_ready_kv_not_prebuilt_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_Q_READY_KV_NOT_PREBUILT_TO_QKV_SCORES_STAGE PASS\n");

        const aecct::TransformerAttnCompatShellStage kv_ready_q_not_prebuilt_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                true,
                false,
                false,
                false,
                false);
        if (kv_ready_q_not_prebuilt_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY) {
            std::printf(
                "[p11anb][FAIL] kv-ready q-not-prebuilt stage mismatch got=%u exp=%u\n",
                (unsigned)kv_ready_q_not_prebuilt_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_KV_READY_Q_NOT_PREBUILT_TO_QKV_SCORES_STAGE PASS\n");

        const aecct::TransformerAttnCompatShellStage qkv_not_prebuilt_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                false,
                false,
                false,
                false,
                false);
        if (qkv_not_prebuilt_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY) {
            std::printf(
                "[p11anb][FAIL] q-not-prebuilt kv-not-prebuilt stage mismatch got=%u exp=%u\n",
                (unsigned)qkv_not_prebuilt_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_QKV_NOT_PREBUILT_TO_QKV_SCORES_STAGE PASS\n");

        const aecct::TransformerAttnCompatShellStage q_ready_kv_not_prebuilt_score_ready_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                false,
                true,
                true,
                false,
                false);
        if (q_ready_kv_not_prebuilt_score_ready_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
            std::printf(
                "[p11anb][FAIL] q-ready kv-not-prebuilt score-ready stage mismatch got=%u exp=%u\n",
                (unsigned)q_ready_kv_not_prebuilt_score_ready_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_Q_READY_KV_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS\n");

        const aecct::TransformerAttnCompatShellStage q_ready_kv_not_prebuilt_score_ready_payload_enabled_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                false,
                true,
                true,
                false,
                true);
        if (q_ready_kv_not_prebuilt_score_ready_payload_enabled_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
            std::printf(
                "[p11anb][FAIL] q-ready kv-not-prebuilt score-ready payload-enabled stage mismatch got=%u exp=%u\n",
                (unsigned)q_ready_kv_not_prebuilt_score_ready_payload_enabled_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_Q_READY_KV_NOT_PREBUILT_SCORE_READY_PAYLOAD_ENABLED_TO_OUT_STAGE PASS\n");

        const aecct::TransformerAttnCompatShellStage kv_ready_q_not_prebuilt_score_ready_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                true,
                false,
                true,
                false,
                false);
        if (kv_ready_q_not_prebuilt_score_ready_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
            std::printf(
                "[p11anb][FAIL] kv-ready q-not-prebuilt score-ready stage mismatch got=%u exp=%u\n",
                (unsigned)kv_ready_q_not_prebuilt_score_ready_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_KV_READY_Q_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS\n");

        const aecct::TransformerAttnCompatShellStage kv_ready_q_not_prebuilt_score_ready_payload_enabled_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                true,
                false,
                true,
                false,
                true);
        if (kv_ready_q_not_prebuilt_score_ready_payload_enabled_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
            std::printf(
                "[p11anb][FAIL] kv-ready q-not-prebuilt score-ready payload-enabled stage mismatch got=%u exp=%u\n",
                (unsigned)kv_ready_q_not_prebuilt_score_ready_payload_enabled_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_KV_READY_Q_NOT_PREBUILT_SCORE_READY_PAYLOAD_ENABLED_TO_OUT_STAGE PASS\n");

        const aecct::TransformerAttnCompatShellStage qkv_not_prebuilt_score_ready_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                false,
                false,
                true,
                false,
                false);
        if (qkv_not_prebuilt_score_ready_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
            std::printf(
                "[p11anb][FAIL] q-not-prebuilt kv-not-prebuilt score-ready stage mismatch got=%u exp=%u\n",
                (unsigned)qkv_not_prebuilt_score_ready_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_QKV_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS\n");

        const aecct::TransformerAttnCompatShellStage qkv_not_prebuilt_score_ready_payload_enabled_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                false,
                false,
                true,
                false,
                true);
        if (qkv_not_prebuilt_score_ready_payload_enabled_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY) {
            std::printf(
                "[p11anb][FAIL] q-not-prebuilt kv-not-prebuilt score-ready payload-enabled stage mismatch got=%u exp=%u\n",
                (unsigned)qkv_not_prebuilt_score_ready_payload_enabled_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_QKV_NOT_PREBUILT_SCORE_READY_PAYLOAD_ENABLED_TO_OUT_STAGE PASS\n");

        const aecct::TransformerAttnCompatShellStage qkv_ready_score_not_prebuilt_payload_enabled_stage =
            aecct::transformer_layer_select_attn_compat_shell_stage(
                true,
                true,
                true,
                false,
                false,
                true);
        if (qkv_ready_score_not_prebuilt_payload_enabled_stage != aecct::TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY) {
            std::printf(
                "[p11anb][FAIL] qkv-ready score-not-prebuilt payload-enabled stage mismatch got=%u exp=%u\n",
                (unsigned)qkv_ready_score_not_prebuilt_payload_enabled_stage,
                (unsigned)aecct::TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY);
            return false;
        }
        std::printf("P11ANB_TRANSFORMER_ATTN_SHELL_QKV_READY_SCORE_NOT_PREBUILT_PAYLOAD_ENABLED_TO_SCORES_STAGE PASS\n");
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

    bool run_top_caller_out_topfed_mapping_pointer_case() {
        return run_transformerlayer_out_topfed_mapping_case(
            "P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_POINTER_CHAIN_CONSUME",
            "P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_POINTER_CHAIN_INVALID_FALLBACK",
            "P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_POINTER_CHAIN_DISABLED_FALLBACK",
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
                aecct::top_dispatch_transformer_layer(
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

    bool run_top_caller_out_topfed_mapping_deep_bridge_case() {
        return run_transformerlayer_out_topfed_mapping_case(
            "P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_CHAIN_CONSUME",
            "P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_CHAIN_INVALID_FALLBACK",
            "P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_CHAIN_DISABLED_FALLBACK",
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
                aecct::top_dispatch_transformer_layer_top_managed_attn_bridge<sram_map::SRAM_WORDS_TOTAL>(
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

    template<typename RunLoopFn>
    bool run_loop_caller_out_topfed_mapping_case(
        const char* consume_pass_banner,
        const char* invalid_fallback_pass_banner,
        const char* disabled_fallback_pass_banner,
        const char* lid_nonzero_fallback_pass_banner,
        RunLoopFn run_loop_fn
    ) {
        const uint32_t words = (uint32_t)aecct::ATTN_TENSOR_WORDS;
        if (words == 0u) {
            std::printf("[p11anb][FAIL] loop caller mapping words must be non-zero\n");
            return false;
        }

        auto init_regs = [&](aecct::TopRegs& regs, uint32_t n_layers) {
            regs.clear();
            regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
            regs.cfg_d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
            regs.cfg_n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
            regs.cfg_d_ffn = (aecct::u32_t)aecct::FFN_D_FFN;
            regs.cfg_n_layers = (aecct::u32_t)n_layers;
        };
        auto init_sram = [](std::vector<aecct::u32_t>& sram) {
            for (uint32_t i = 0u; i < kSramWords; ++i) {
                sram[i] = (aecct::u32_t)0u;
            }
            const uint32_t x_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
            const uint32_t x_words = (uint32_t)aecct::LN_X_TOTAL_WORDS;
            for (uint32_t i = 0u; i < x_words; ++i) {
                sram[x_base + i] = (aecct::u32_t)(0x3F000000u + i);
            }
        };

        std::vector<aecct::u32_t> sram_baseline(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_valid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_invalid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_disabled(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_lid_nonzero(kSramWords, (aecct::u32_t)0u);
        aecct::TopRegs regs_baseline;
        aecct::TopRegs regs_valid;
        aecct::TopRegs regs_invalid;
        aecct::TopRegs regs_disabled;
        aecct::TopRegs regs_lid_nonzero;

        init_sram(sram_baseline);
        init_sram(sram_valid);
        init_sram(sram_invalid);
        init_sram(sram_disabled);
        init_sram(sram_lid_nonzero);
        init_regs(regs_baseline, 1u);
        init_regs(regs_valid, 1u);
        init_regs(regs_invalid, 1u);
        init_regs(regs_disabled, 1u);
        init_regs(regs_lid_nonzero, 2u);

        run_loop_fn(regs_baseline, sram_baseline, false, true);
        run_loop_fn(regs_valid, sram_valid, true, true);
        run_loop_fn(regs_invalid, sram_invalid, true, false);
        run_loop_fn(regs_disabled, sram_disabled, false, true);
        run_loop_fn(regs_lid_nonzero, sram_lid_nonzero, true, true);

        const uint32_t final_base_baseline = u32_bits(regs_baseline.infer_final_x_base_word);
        const uint32_t final_base_valid = u32_bits(regs_valid.infer_final_x_base_word);
        const uint32_t final_base_invalid = u32_bits(regs_invalid.infer_final_x_base_word);
        const uint32_t final_base_disabled = u32_bits(regs_disabled.infer_final_x_base_word);
        if (final_base_valid != final_base_baseline ||
            final_base_invalid != final_base_baseline ||
            final_base_disabled != final_base_baseline) {
            std::printf("[p11anb][FAIL] top loop final base mismatch baseline=%u valid=%u invalid=%u disabled=%u\n",
                (unsigned)final_base_baseline,
                (unsigned)final_base_valid,
                (unsigned)final_base_invalid,
                (unsigned)final_base_disabled);
            return false;
        }

        const uint32_t compare_words = 16u;
        if (!check_equal_region_between_srams(
                sram_invalid,
                sram_baseline,
                final_base_baseline,
                compare_words,
                "TOP-LOOP-FINAL-invalid-fallback")) {
            return false;
        }
        if (!check_equal_region_between_srams(
                sram_disabled,
                sram_baseline,
                final_base_baseline,
                compare_words,
                "TOP-LOOP-FINAL-disabled-fallback")) {
            return false;
        }

        const uint32_t diff_count = count_region_diff(
            sram_valid,
            sram_baseline,
            final_base_baseline,
            compare_words);
        if (diff_count == 0u) {
            std::printf("[p11anb][FAIL] top loop caller expected compare unchanged at final X first=%u words\n",
                (unsigned)compare_words);
            return false;
        }

        if (u32_bits(regs_valid.p11ax_attn_out_payload_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11ax_attn_out_payload_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ax_lid0_attn_out_payload_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ax_attn_out_payload_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] top loop valid marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11ax_attn_out_payload_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11ax_attn_out_payload_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ax_lid0_attn_out_payload_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ax_attn_out_payload_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_invalid.p11ax_attn_out_payload_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11ax_attn_out_payload_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11ax_attn_out_payload_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] top loop invalid marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_invalid.p11ax_attn_out_payload_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11ax_attn_out_payload_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11ax_attn_out_payload_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_disabled.p11ax_attn_out_payload_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11ax_attn_out_payload_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11ax_attn_out_payload_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] top loop disabled marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_disabled.p11ax_attn_out_payload_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11ax_attn_out_payload_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11ax_attn_out_payload_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11ax_attn_out_payload_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11ax_attn_out_payload_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ax_lid0_attn_out_payload_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ax_attn_out_payload_fallback_seen_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ax_lid_nonzero_attn_out_payload_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] top loop lid!=0 marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11ax_attn_out_payload_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ax_attn_out_payload_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ax_lid0_attn_out_payload_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ax_attn_out_payload_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ax_lid_nonzero_attn_out_payload_fallback_seen_count));
            return false;
        }

        std::printf("%s PASS\n", consume_pass_banner);
        std::printf("%s PASS\n", invalid_fallback_pass_banner);
        std::printf("%s PASS\n", disabled_fallback_pass_banner);
        std::printf("%s PASS\n", lid_nonzero_fallback_pass_banner);
        return true;
    }

    bool run_loop_caller_out_topfed_mapping_pointer_case() {
        return run_loop_caller_out_topfed_mapping_case(
            "P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_CONSUME",
            "P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_INVALID_FALLBACK",
            "P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_DISABLED_FALLBACK",
            "P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_LID_NONZERO_FALLBACK",
            [](
                aecct::TopRegs& regs,
                std::vector<aecct::u32_t>& sram,
                bool payload_enable,
                bool descriptor_valid
            ) {
                aecct::run_transformer_layer_loop(
                    regs,
                    sram.data(),
                    false,  // lid0_local_only_ffn_handoff_enable
                    true,   // lid0_local_only_ffn_handoff_descriptor_valid
                    payload_enable,
                    descriptor_valid);
            });
    }

    bool run_loop_caller_out_topfed_mapping_deep_bridge_case() {
        return run_loop_caller_out_topfed_mapping_case(
            "P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_CONSUME",
            "P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_INVALID_FALLBACK",
            "P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_DISABLED_FALLBACK",
            "P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_LID_NONZERO_FALLBACK",
            [](
                aecct::TopRegs& regs,
                std::vector<aecct::u32_t>& sram,
                bool payload_enable,
                bool descriptor_valid
            ) {
                static aecct::u32_t sram_window[sram_map::SRAM_WORDS_TOTAL];
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram_window[i] = sram[i];
                }
                aecct::run_transformer_layer_loop_top_managed_attn_bridge<sram_map::SRAM_WORDS_TOTAL>(
                    regs,
                    sram_window,
                    false,  // lid0_local_only_ffn_handoff_enable
                    true,   // lid0_local_only_ffn_handoff_descriptor_valid
                    payload_enable,
                    descriptor_valid);
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram[i] = sram_window[i];
                }
            });
    }

    template<typename RunLoopFn>
    bool run_loop_caller_qkscore_mask_handoff_case(
        const char* consume_pass_banner,
        const char* invalid_fallback_pass_banner,
        const char* disabled_fallback_pass_banner,
        const char* lid_nonzero_fallback_pass_banner,
        RunLoopFn run_loop_fn
    ) {
        p11aeaf_tb::QkvPayloadSet payloads;
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads)) {
            std::printf("[p11anb][FAIL] qkscore mask payload preparation failed\n");
            return false;
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        const aecct::CfgRegs bootstrap_cfg = p11aeaf_tb::build_cfg();
        auto init_regs = [&](aecct::TopRegs& regs, uint32_t n_layers) {
            regs.clear();
            regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
            regs.cfg_d_model = bootstrap_cfg.d_model;
            regs.cfg_n_heads = bootstrap_cfg.n_heads;
            regs.cfg_d_ffn = bootstrap_cfg.d_ffn;
            regs.cfg_n_layers = (aecct::u32_t)n_layers;
        };
        auto init_sram = [&](std::vector<aecct::u32_t>& sram) {
            for (uint32_t i = 0u; i < kSramWords; ++i) {
                sram[i] = (aecct::u32_t)0u;
            }
            p11aeaf_tb::init_x_rows(sram);
            p11aeaf_tb::load_qkv_payload_set_to_sram(sram, payloads, param_base);
        };

        std::vector<aecct::u32_t> sram_baseline(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_valid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_invalid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_disabled(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_lid_nonzero(kSramWords, (aecct::u32_t)0u);
        aecct::TopRegs regs_baseline;
        aecct::TopRegs regs_valid;
        aecct::TopRegs regs_invalid;
        aecct::TopRegs regs_disabled;
        aecct::TopRegs regs_lid_nonzero;

        init_sram(sram_baseline);
        init_sram(sram_valid);
        init_sram(sram_invalid);
        init_sram(sram_disabled);
        init_sram(sram_lid_nonzero);
        init_regs(regs_baseline, 1u);
        init_regs(regs_valid, 1u);
        init_regs(regs_invalid, 1u);
        init_regs(regs_disabled, 1u);
        init_regs(regs_lid_nonzero, 2u);

        run_loop_fn(regs_baseline, sram_baseline, false, true);
        run_loop_fn(regs_valid, sram_valid, true, true);
        run_loop_fn(regs_invalid, sram_invalid, true, false);
        run_loop_fn(regs_disabled, sram_disabled, false, true);
        run_loop_fn(regs_lid_nonzero, sram_lid_nonzero, true, true);

        if (!regs_valid.p11ae_mainline_score_path_taken || regs_valid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore mask valid case did not stay on score mainline\n");
            return false;
        }
        if (regs_invalid.p11ae_mainline_score_path_taken || !regs_invalid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore mask invalid case did not take fallback\n");
            return false;
        }
        if (!regs_disabled.p11ae_mainline_score_path_taken || regs_disabled.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore mask disabled case unexpectedly left mainline\n");
            return false;
        }

        if (u32_bits(regs_valid.p11ay_qkscore_mask_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11ay_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ay_lid0_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ay_qkscore_mask_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] qkscore mask valid marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ay_lid0_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ay_qkscore_mask_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] qkscore mask invalid marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] qkscore mask disabled marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ay_lid0_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_fallback_seen_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ay_lid_nonzero_qkscore_mask_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] qkscore mask lid!=0 marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_lid0_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_lid_nonzero_qkscore_mask_handoff_fallback_seen_count));
            return false;
        }

        const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const uint32_t score_base = u32_bits(sc.attn.score_base_word);
        const uint32_t score_words = (uint32_t)aecct::ATTN_TOKEN_COUNT * (uint32_t)aecct::ATTN_N_HEADS;
        if (!check_equal_region_between_srams(
                sram_valid,
                sram_baseline,
                score_base,
                score_words,
                "TOP-LOOP-QKSCORE-MASK-valid-vs-baseline")) {
            return false;
        }

        std::printf("%s PASS\n", consume_pass_banner);
        std::printf("%s PASS\n", invalid_fallback_pass_banner);
        std::printf("%s PASS\n", disabled_fallback_pass_banner);
        std::printf("%s PASS\n", lid_nonzero_fallback_pass_banner);
        return true;
    }

    bool run_loop_caller_qkscore_mask_handoff_pointer_case() {
        return run_loop_caller_qkscore_mask_handoff_case(
            "P11ANB_LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_CONSUME",
            "P11ANB_LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_INVALID_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_DISABLED_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_LID_NONZERO_FALLBACK",
            [](
                aecct::TopRegs& regs,
                std::vector<aecct::u32_t>& sram,
                bool handoff_enable,
                bool descriptor_valid
            ) {
                aecct::run_transformer_layer_loop(
                    regs,
                    sram.data(),
                    false,  // lid0_local_only_ffn_handoff_enable
                    true,   // lid0_local_only_ffn_handoff_descriptor_valid
                    false,  // lid0_local_only_attn_out_payload_enable
                    true,   // lid0_local_only_attn_out_payload_descriptor_valid
                    handoff_enable,
                    descriptor_valid);
            });
    }

    bool run_loop_caller_qkscore_mask_handoff_deep_bridge_case() {
        return run_loop_caller_qkscore_mask_handoff_case(
            "P11ANB_LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_CONSUME",
            "P11ANB_LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_INVALID_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_DISABLED_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_LID_NONZERO_FALLBACK",
            [](
                aecct::TopRegs& regs,
                std::vector<aecct::u32_t>& sram,
                bool handoff_enable,
                bool descriptor_valid
            ) {
                static aecct::u32_t sram_window[sram_map::SRAM_WORDS_TOTAL];
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram_window[i] = sram[i];
                }
                aecct::run_transformer_layer_loop_top_managed_attn_bridge<sram_map::SRAM_WORDS_TOTAL>(
                    regs,
                    sram_window,
                    false,  // lid0_local_only_ffn_handoff_enable
                    true,   // lid0_local_only_ffn_handoff_descriptor_valid
                    false,  // lid0_local_only_attn_out_payload_enable
                    true,   // lid0_local_only_attn_out_payload_descriptor_valid
                    handoff_enable,
                    descriptor_valid);
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram[i] = sram_window[i];
                }
            });
    }

    template<typename RunLoopFn>
    bool run_loop_caller_qkscore_kvscan_handoff_case(
        const char* consume_pass_banner,
        const char* invalid_fallback_pass_banner,
        const char* disabled_fallback_pass_banner,
        const char* lid_nonzero_fallback_pass_banner,
        RunLoopFn run_loop_fn
    ) {
        p11aeaf_tb::QkvPayloadSet payloads;
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads)) {
            std::printf("[p11anb][FAIL] qkscore kvscan payload preparation failed\n");
            return false;
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        const aecct::CfgRegs bootstrap_cfg = p11aeaf_tb::build_cfg();
        auto init_regs = [&](aecct::TopRegs& regs, uint32_t n_layers) {
            regs.clear();
            regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
            regs.cfg_d_model = bootstrap_cfg.d_model;
            regs.cfg_n_heads = bootstrap_cfg.n_heads;
            regs.cfg_d_ffn = bootstrap_cfg.d_ffn;
            regs.cfg_n_layers = (aecct::u32_t)n_layers;
        };
        auto init_sram = [&](std::vector<aecct::u32_t>& sram) {
            for (uint32_t i = 0u; i < kSramWords; ++i) {
                sram[i] = (aecct::u32_t)0u;
            }
            p11aeaf_tb::init_x_rows(sram);
            p11aeaf_tb::load_qkv_payload_set_to_sram(sram, payloads, param_base);
        };

        std::vector<aecct::u32_t> sram_baseline(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_valid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_invalid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_disabled(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_lid_nonzero(kSramWords, (aecct::u32_t)0u);
        aecct::TopRegs regs_baseline;
        aecct::TopRegs regs_valid;
        aecct::TopRegs regs_invalid;
        aecct::TopRegs regs_disabled;
        aecct::TopRegs regs_lid_nonzero;

        init_sram(sram_baseline);
        init_sram(sram_valid);
        init_sram(sram_invalid);
        init_sram(sram_disabled);
        init_sram(sram_lid_nonzero);
        init_regs(regs_baseline, 1u);
        init_regs(regs_valid, 1u);
        init_regs(regs_invalid, 1u);
        init_regs(regs_disabled, 1u);
        init_regs(regs_lid_nonzero, 2u);

        run_loop_fn(regs_baseline, sram_baseline, false, true);
        run_loop_fn(regs_valid, sram_valid, true, true);
        run_loop_fn(regs_invalid, sram_invalid, true, false);
        run_loop_fn(regs_disabled, sram_disabled, false, true);
        run_loop_fn(regs_lid_nonzero, sram_lid_nonzero, true, true);

        if (!regs_valid.p11ae_mainline_score_path_taken || regs_valid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore kvscan valid case did not stay on score mainline\n");
            return false;
        }
        if (regs_invalid.p11ae_mainline_score_path_taken || !regs_invalid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore kvscan invalid case did not take fallback\n");
            return false;
        }
        if (!regs_disabled.p11ae_mainline_score_path_taken || regs_disabled.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore kvscan disabled case unexpectedly left mainline\n");
            return false;
        }

        if (u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11az_lid0_qkscore_kvscan_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] qkscore kvscan valid marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11az_lid0_qkscore_kvscan_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_invalid.p11az_qkscore_kvscan_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11az_qkscore_kvscan_handoff_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11az_qkscore_kvscan_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] qkscore kvscan invalid marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_invalid.p11az_qkscore_kvscan_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11az_qkscore_kvscan_handoff_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11az_qkscore_kvscan_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_disabled.p11az_qkscore_kvscan_handoff_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11az_qkscore_kvscan_handoff_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11az_qkscore_kvscan_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] qkscore kvscan disabled marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_disabled.p11az_qkscore_kvscan_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11az_qkscore_kvscan_handoff_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11az_qkscore_kvscan_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11az_lid0_qkscore_kvscan_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_fallback_seen_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11az_lid_nonzero_qkscore_kvscan_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] qkscore kvscan lid!=0 marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11az_lid0_qkscore_kvscan_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11az_lid_nonzero_qkscore_kvscan_handoff_fallback_seen_count));
            return false;
        }

        const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const uint32_t score_base = u32_bits(sc.attn.score_base_word);
        const uint32_t score_words = (uint32_t)aecct::ATTN_TOKEN_COUNT * (uint32_t)aecct::ATTN_N_HEADS;
        if (!check_equal_region_between_srams(
                sram_valid,
                sram_baseline,
                score_base,
                score_words,
                "TOP-LOOP-QKSCORE-KVSCAN-valid-vs-baseline")) {
            return false;
        }

        std::printf("%s PASS\n", consume_pass_banner);
        std::printf("%s PASS\n", invalid_fallback_pass_banner);
        std::printf("%s PASS\n", disabled_fallback_pass_banner);
        std::printf("%s PASS\n", lid_nonzero_fallback_pass_banner);
        return true;
    }

    bool run_loop_caller_qkscore_kvscan_handoff_pointer_case() {
        return run_loop_caller_qkscore_kvscan_handoff_case(
            "P11ANB_LOOP_CALLER_QKSCORE_KVSCAN_POINTER_HOOK_CONSUME",
            "P11ANB_LOOP_CALLER_QKSCORE_KVSCAN_POINTER_HOOK_INVALID_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_KVSCAN_POINTER_HOOK_DISABLED_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_KVSCAN_POINTER_HOOK_LID_NONZERO_FALLBACK",
            [](
                aecct::TopRegs& regs,
                std::vector<aecct::u32_t>& sram,
                bool handoff_enable,
                bool descriptor_valid
            ) {
                aecct::run_transformer_layer_loop(
                    regs,
                    sram.data(),
                    false,  // lid0_local_only_ffn_handoff_enable
                    true,   // lid0_local_only_ffn_handoff_descriptor_valid
                    false,  // lid0_local_only_attn_out_payload_enable
                    true,   // lid0_local_only_attn_out_payload_descriptor_valid
                    false,  // lid0_local_only_qkscore_mask_handoff_enable
                    true,   // lid0_local_only_qkscore_mask_handoff_descriptor_valid
                    handoff_enable,
                    descriptor_valid);
            });
    }

    bool run_loop_caller_qkscore_kvscan_handoff_deep_bridge_case() {
        return run_loop_caller_qkscore_kvscan_handoff_case(
            "P11ANB_LOOP_CALLER_QKSCORE_KVSCAN_DEEP_BRIDGE_HOOK_CONSUME",
            "P11ANB_LOOP_CALLER_QKSCORE_KVSCAN_DEEP_BRIDGE_HOOK_INVALID_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_KVSCAN_DEEP_BRIDGE_HOOK_DISABLED_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_KVSCAN_DEEP_BRIDGE_HOOK_LID_NONZERO_FALLBACK",
            [](
                aecct::TopRegs& regs,
                std::vector<aecct::u32_t>& sram,
                bool handoff_enable,
                bool descriptor_valid
            ) {
                static aecct::u32_t sram_window[sram_map::SRAM_WORDS_TOTAL];
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram_window[i] = sram[i];
                }
                aecct::run_transformer_layer_loop_top_managed_attn_bridge<sram_map::SRAM_WORDS_TOTAL>(
                    regs,
                    sram_window,
                    false,  // lid0_local_only_ffn_handoff_enable
                    true,   // lid0_local_only_ffn_handoff_descriptor_valid
                    false,  // lid0_local_only_attn_out_payload_enable
                    true,   // lid0_local_only_attn_out_payload_descriptor_valid
                    false,  // lid0_local_only_qkscore_mask_handoff_enable
                    true,   // lid0_local_only_qkscore_mask_handoff_descriptor_valid
                    handoff_enable,
                    descriptor_valid);
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram[i] = sram_window[i];
                }
            });
    }

    template<typename RunLoopFn>
    bool run_loop_caller_qkscore_qsrc_handoff_case(
        const char* consume_pass_banner,
        const char* invalid_fallback_pass_banner,
        const char* disabled_fallback_pass_banner,
        const char* lid_nonzero_fallback_pass_banner,
        RunLoopFn run_loop_fn
    ) {
        p11aeaf_tb::QkvPayloadSet payloads;
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads)) {
            std::printf("[p11anb][FAIL] qkscore qsrc payload preparation failed\n");
            return false;
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        const aecct::CfgRegs bootstrap_cfg = p11aeaf_tb::build_cfg();
        auto init_regs = [&](aecct::TopRegs& regs, uint32_t n_layers) {
            regs.clear();
            regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
            regs.cfg_d_model = bootstrap_cfg.d_model;
            regs.cfg_n_heads = bootstrap_cfg.n_heads;
            regs.cfg_d_ffn = bootstrap_cfg.d_ffn;
            regs.cfg_n_layers = (aecct::u32_t)n_layers;
        };
        auto init_sram = [&](std::vector<aecct::u32_t>& sram) {
            for (uint32_t i = 0u; i < kSramWords; ++i) {
                sram[i] = (aecct::u32_t)0u;
            }
            p11aeaf_tb::init_x_rows(sram);
            p11aeaf_tb::load_qkv_payload_set_to_sram(sram, payloads, param_base);
        };

        std::vector<aecct::u32_t> sram_baseline(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_valid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_invalid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_disabled(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_lid_nonzero(kSramWords, (aecct::u32_t)0u);
        aecct::TopRegs regs_baseline;
        aecct::TopRegs regs_valid;
        aecct::TopRegs regs_invalid;
        aecct::TopRegs regs_disabled;
        aecct::TopRegs regs_lid_nonzero;

        init_sram(sram_baseline);
        init_sram(sram_valid);
        init_sram(sram_invalid);
        init_sram(sram_disabled);
        init_sram(sram_lid_nonzero);
        init_regs(regs_baseline, 1u);
        init_regs(regs_valid, 1u);
        init_regs(regs_invalid, 1u);
        init_regs(regs_disabled, 1u);
        init_regs(regs_lid_nonzero, 2u);

        run_loop_fn(regs_baseline, sram_baseline, false, true);
        run_loop_fn(regs_valid, sram_valid, true, true);
        run_loop_fn(regs_invalid, sram_invalid, true, false);
        run_loop_fn(regs_disabled, sram_disabled, false, true);
        run_loop_fn(regs_lid_nonzero, sram_lid_nonzero, true, true);

        if (!regs_valid.p11ae_mainline_score_path_taken || regs_valid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore qsrc valid case did not stay on score mainline\n");
            return false;
        }
        if (regs_invalid.p11ae_mainline_score_path_taken || !regs_invalid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore qsrc invalid case did not take fallback\n");
            return false;
        }
        if (!regs_disabled.p11ae_mainline_score_path_taken || regs_disabled.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore qsrc disabled case unexpectedly left mainline\n");
            return false;
        }

        if (u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] qkscore qsrc valid marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] qkscore qsrc invalid marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] qkscore qsrc disabled marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ba_lid_nonzero_qkscore_qsrc_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] qkscore qsrc lid!=0 marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_lid_nonzero_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }

        const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const uint32_t score_base = u32_bits(sc.attn.score_base_word);
        const uint32_t score_words = (uint32_t)aecct::ATTN_TOKEN_COUNT * (uint32_t)aecct::ATTN_N_HEADS;
        if (!check_equal_region_between_srams(
                sram_valid,
                sram_baseline,
                score_base,
                score_words,
                "TOP-LOOP-QKSCORE-QSRC-valid-vs-baseline")) {
            return false;
        }

        std::printf("%s PASS\n", consume_pass_banner);
        std::printf("%s PASS\n", invalid_fallback_pass_banner);
        std::printf("%s PASS\n", disabled_fallback_pass_banner);
        std::printf("%s PASS\n", lid_nonzero_fallback_pass_banner);
        return true;
    }

    bool run_loop_caller_qkscore_qsrc_handoff_pointer_case() {
        return run_loop_caller_qkscore_qsrc_handoff_case(
            "P11ANB_LOOP_CALLER_QKSCORE_QSRC_POINTER_HOOK_CONSUME",
            "P11ANB_LOOP_CALLER_QKSCORE_QSRC_POINTER_HOOK_INVALID_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_QSRC_POINTER_HOOK_DISABLED_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_QSRC_POINTER_HOOK_LID_NONZERO_FALLBACK",
            [](
                aecct::TopRegs& regs,
                std::vector<aecct::u32_t>& sram,
                bool handoff_enable,
                bool descriptor_valid
            ) {
                aecct::run_transformer_layer_loop(
                    regs,
                    sram.data(),
                    false,  // lid0_local_only_ffn_handoff_enable
                    true,   // lid0_local_only_ffn_handoff_descriptor_valid
                    false,  // lid0_local_only_attn_out_payload_enable
                    true,   // lid0_local_only_attn_out_payload_descriptor_valid
                    false,  // lid0_local_only_qkscore_mask_handoff_enable
                    true,   // lid0_local_only_qkscore_mask_handoff_descriptor_valid
                    false,  // lid0_local_only_qkscore_kvscan_handoff_enable
                    true,   // lid0_local_only_qkscore_kvscan_handoff_descriptor_valid
                    handoff_enable,
                    descriptor_valid);
            });
    }

    bool run_loop_caller_qkscore_qsrc_handoff_deep_bridge_case() {
        return run_loop_caller_qkscore_qsrc_handoff_case(
            "P11ANB_LOOP_CALLER_QKSCORE_QSRC_DEEP_BRIDGE_HOOK_CONSUME",
            "P11ANB_LOOP_CALLER_QKSCORE_QSRC_DEEP_BRIDGE_HOOK_INVALID_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_QSRC_DEEP_BRIDGE_HOOK_DISABLED_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_QSRC_DEEP_BRIDGE_HOOK_LID_NONZERO_FALLBACK",
            [](
                aecct::TopRegs& regs,
                std::vector<aecct::u32_t>& sram,
                bool handoff_enable,
                bool descriptor_valid
            ) {
                static aecct::u32_t sram_window[sram_map::SRAM_WORDS_TOTAL];
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram_window[i] = sram[i];
                }
                aecct::run_transformer_layer_loop_top_managed_attn_bridge<sram_map::SRAM_WORDS_TOTAL>(
                    regs,
                    sram_window,
                    false,  // lid0_local_only_ffn_handoff_enable
                    true,   // lid0_local_only_ffn_handoff_descriptor_valid
                    false,  // lid0_local_only_attn_out_payload_enable
                    true,   // lid0_local_only_attn_out_payload_descriptor_valid
                    false,  // lid0_local_only_qkscore_mask_handoff_enable
                    true,   // lid0_local_only_qkscore_mask_handoff_descriptor_valid
                    false,  // lid0_local_only_qkscore_kvscan_handoff_enable
                    true,   // lid0_local_only_qkscore_kvscan_handoff_descriptor_valid
                    handoff_enable,
                    descriptor_valid);
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram[i] = sram_window[i];
                }
            });
    }

    template<typename RunLoopFn>
    bool run_loop_caller_qkscore_wq_handoff_case(
        const char* consume_pass_banner,
        const char* invalid_fallback_pass_banner,
        const char* disabled_fallback_pass_banner,
        const char* lid_nonzero_fallback_pass_banner,
        RunLoopFn run_loop_fn
    ) {
        p11aeaf_tb::QkvPayloadSet payloads;
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads)) {
            std::printf("[p11anb][FAIL] qkscore wq payload preparation failed\n");
            return false;
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        const aecct::CfgRegs bootstrap_cfg = p11aeaf_tb::build_cfg();
        auto init_regs = [&](aecct::TopRegs& regs, uint32_t n_layers) {
            regs.clear();
            regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
            regs.cfg_d_model = bootstrap_cfg.d_model;
            regs.cfg_n_heads = bootstrap_cfg.n_heads;
            regs.cfg_d_ffn = bootstrap_cfg.d_ffn;
            regs.cfg_n_layers = (aecct::u32_t)n_layers;
        };
        auto init_sram = [&](std::vector<aecct::u32_t>& sram) {
            for (uint32_t i = 0u; i < kSramWords; ++i) {
                sram[i] = (aecct::u32_t)0u;
            }
            p11aeaf_tb::init_x_rows(sram);
            p11aeaf_tb::load_qkv_payload_set_to_sram(sram, payloads, param_base);
        };

        std::vector<aecct::u32_t> sram_baseline(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_valid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_invalid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_disabled(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_lid_nonzero(kSramWords, (aecct::u32_t)0u);
        aecct::TopRegs regs_baseline;
        aecct::TopRegs regs_valid;
        aecct::TopRegs regs_invalid;
        aecct::TopRegs regs_disabled;
        aecct::TopRegs regs_lid_nonzero;

        init_sram(sram_baseline);
        init_sram(sram_valid);
        init_sram(sram_invalid);
        init_sram(sram_disabled);
        init_sram(sram_lid_nonzero);
        init_regs(regs_baseline, 1u);
        init_regs(regs_valid, 1u);
        init_regs(regs_invalid, 1u);
        init_regs(regs_disabled, 1u);
        init_regs(regs_lid_nonzero, 2u);

        run_loop_fn(regs_baseline, sram_baseline, false, true);
        run_loop_fn(regs_valid, sram_valid, true, true);
        run_loop_fn(regs_invalid, sram_invalid, true, false);
        run_loop_fn(regs_disabled, sram_disabled, false, true);
        run_loop_fn(regs_lid_nonzero, sram_lid_nonzero, true, true);

        if (!regs_valid.p11ae_mainline_score_path_taken || regs_valid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore wq valid case did not stay on score mainline\n");
            return false;
        }
        if (regs_invalid.p11ae_mainline_score_path_taken || !regs_invalid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore wq invalid case did not take fallback\n");
            return false;
        }
        if (!regs_disabled.p11ae_mainline_score_path_taken || regs_disabled.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore wq disabled case unexpectedly left mainline\n");
            return false;
        }

        if (u32_bits(regs_valid.p11bb_qkscore_wq_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11bb_qkscore_wq_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11bb_lid0_qkscore_wq_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11bb_qkscore_wq_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] qkscore wq valid marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11bb_lid0_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11bb_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] qkscore wq invalid marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] qkscore wq disabled marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11bb_lid0_qkscore_wq_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_fallback_seen_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11bb_lid_nonzero_qkscore_wq_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] qkscore wq lid!=0 marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_lid0_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_lid_nonzero_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }

        const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const uint32_t score_base = u32_bits(sc.attn.score_base_word);
        const uint32_t score_words = (uint32_t)aecct::ATTN_TOKEN_COUNT * (uint32_t)aecct::ATTN_N_HEADS;
        if (!check_equal_region_between_srams(
                sram_valid,
                sram_baseline,
                score_base,
                score_words,
                "TOP-LOOP-QKSCORE-WQ-valid-vs-baseline")) {
            return false;
        }

        std::printf("%s PASS\n", consume_pass_banner);
        std::printf("%s PASS\n", invalid_fallback_pass_banner);
        std::printf("%s PASS\n", disabled_fallback_pass_banner);
        std::printf("%s PASS\n", lid_nonzero_fallback_pass_banner);
        return true;
    }

    bool run_loop_caller_qkscore_wq_handoff_pointer_case() {
        return run_loop_caller_qkscore_wq_handoff_case(
            "P11ANB_LOOP_CALLER_QKSCORE_WQ_POINTER_HOOK_CONSUME",
            "P11ANB_LOOP_CALLER_QKSCORE_WQ_POINTER_HOOK_INVALID_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_WQ_POINTER_HOOK_DISABLED_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_WQ_POINTER_HOOK_LID_NONZERO_FALLBACK",
            [](
                aecct::TopRegs& regs,
                std::vector<aecct::u32_t>& sram,
                bool handoff_enable,
                bool descriptor_valid
            ) {
                aecct::run_transformer_layer_loop(
                    regs,
                    sram.data(),
                    false,  // lid0_local_only_ffn_handoff_enable
                    true,   // lid0_local_only_ffn_handoff_descriptor_valid
                    false,  // lid0_local_only_attn_out_payload_enable
                    true,   // lid0_local_only_attn_out_payload_descriptor_valid
                    false,  // lid0_local_only_qkscore_mask_handoff_enable
                    true,   // lid0_local_only_qkscore_mask_handoff_descriptor_valid
                    false,  // lid0_local_only_qkscore_kvscan_handoff_enable
                    true,   // lid0_local_only_qkscore_kvscan_handoff_descriptor_valid
                    false,  // lid0_local_only_qkscore_qsrc_handoff_enable
                    true,   // lid0_local_only_qkscore_qsrc_handoff_descriptor_valid
                    handoff_enable,
                    descriptor_valid);
            });
    }

    bool run_loop_caller_qkscore_wq_handoff_deep_bridge_case() {
        return run_loop_caller_qkscore_wq_handoff_case(
            "P11ANB_LOOP_CALLER_QKSCORE_WQ_DEEP_BRIDGE_HOOK_CONSUME",
            "P11ANB_LOOP_CALLER_QKSCORE_WQ_DEEP_BRIDGE_HOOK_INVALID_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_WQ_DEEP_BRIDGE_HOOK_DISABLED_FALLBACK",
            "P11ANB_LOOP_CALLER_QKSCORE_WQ_DEEP_BRIDGE_HOOK_LID_NONZERO_FALLBACK",
            [](
                aecct::TopRegs& regs,
                std::vector<aecct::u32_t>& sram,
                bool handoff_enable,
                bool descriptor_valid
            ) {
                static aecct::u32_t sram_window[sram_map::SRAM_WORDS_TOTAL];
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram_window[i] = sram[i];
                }
                aecct::run_transformer_layer_loop_top_managed_attn_bridge<sram_map::SRAM_WORDS_TOTAL>(
                    regs,
                    sram_window,
                    false,  // lid0_local_only_ffn_handoff_enable
                    true,   // lid0_local_only_ffn_handoff_descriptor_valid
                    false,  // lid0_local_only_attn_out_payload_enable
                    true,   // lid0_local_only_attn_out_payload_descriptor_valid
                    false,  // lid0_local_only_qkscore_mask_handoff_enable
                    true,   // lid0_local_only_qkscore_mask_handoff_descriptor_valid
                    false,  // lid0_local_only_qkscore_kvscan_handoff_enable
                    true,   // lid0_local_only_qkscore_kvscan_handoff_descriptor_valid
                    false,  // lid0_local_only_qkscore_qsrc_handoff_enable
                    true,   // lid0_local_only_qkscore_qsrc_handoff_descriptor_valid
                    handoff_enable,
                    descriptor_valid);
                for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
                    sram[i] = sram_window[i];
                }
            });
    }

    bool run_loop_caller_qkscore_multi_seam_mask_wq_priority_pointer_case() {
        p11aeaf_tb::QkvPayloadSet payloads;
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads)) {
            std::printf("[p11anb][FAIL] qkscore multi-seam MASK+WQ payload preparation failed\n");
            return false;
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        const aecct::CfgRegs bootstrap_cfg = p11aeaf_tb::build_cfg();
        auto run_case = [&](aecct::TopRegs& regs,
                            std::vector<aecct::u32_t>& sram,
                            uint32_t n_layers,
                            bool mask_enable,
                            bool mask_descriptor_valid,
                            bool wq_enable,
                            bool wq_descriptor_valid) {
            regs.clear();
            regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
            regs.cfg_d_model = bootstrap_cfg.d_model;
            regs.cfg_n_heads = bootstrap_cfg.n_heads;
            regs.cfg_d_ffn = bootstrap_cfg.d_ffn;
            regs.cfg_n_layers = (aecct::u32_t)n_layers;
            for (uint32_t i = 0u; i < kSramWords; ++i) {
                sram[i] = (aecct::u32_t)0u;
            }
            p11aeaf_tb::init_x_rows(sram);
            p11aeaf_tb::load_qkv_payload_set_to_sram(sram, payloads, param_base);
            aecct::run_transformer_layer_loop(
                regs,
                sram.data(),
                false,  // lid0_local_only_ffn_handoff_enable
                true,   // lid0_local_only_ffn_handoff_descriptor_valid
                false,  // lid0_local_only_attn_out_payload_enable
                true,   // lid0_local_only_attn_out_payload_descriptor_valid
                mask_enable,
                mask_descriptor_valid,
                false,  // lid0_local_only_qkscore_kvscan_handoff_enable
                true,   // lid0_local_only_qkscore_kvscan_handoff_descriptor_valid
                false,  // lid0_local_only_qkscore_qsrc_handoff_enable
                true,   // lid0_local_only_qkscore_qsrc_handoff_descriptor_valid
                wq_enable,
                wq_descriptor_valid);
        };

        std::vector<aecct::u32_t> sram_baseline(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_valid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_invalid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_disabled(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_lid_nonzero(kSramWords, (aecct::u32_t)0u);
        aecct::TopRegs regs_baseline;
        aecct::TopRegs regs_valid;
        aecct::TopRegs regs_invalid;
        aecct::TopRegs regs_disabled;
        aecct::TopRegs regs_lid_nonzero;

        run_case(regs_baseline, sram_baseline, 1u, false, true, false, true);
        run_case(regs_valid, sram_valid, 1u, true, true, true, true);
        run_case(regs_invalid, sram_invalid, 1u, true, false, true, true);
        run_case(regs_disabled, sram_disabled, 1u, false, true, false, true);
        run_case(regs_lid_nonzero, sram_lid_nonzero, 2u, true, true, true, true);

        if (!regs_valid.p11ae_mainline_score_path_taken || regs_valid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore multi-seam MASK+WQ valid case did not stay on score mainline\n");
            return false;
        }
        if (regs_invalid.p11ae_mainline_score_path_taken || !regs_invalid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore multi-seam MASK+WQ invalid case did not take fallback\n");
            return false;
        }
        if (!regs_disabled.p11ae_mainline_score_path_taken || regs_disabled.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore multi-seam MASK+WQ disabled case unexpectedly left mainline\n");
            return false;
        }

        if (u32_bits(regs_valid.p11ay_qkscore_mask_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11ay_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ay_lid0_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ay_qkscore_mask_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+WQ MASK-selected marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ay_lid0_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ay_qkscore_mask_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_valid.p11bb_qkscore_wq_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11bb_qkscore_wq_handoff_non_empty_count) != 0u ||
            u32_bits(regs_valid.p11bb_lid0_qkscore_wq_handoff_non_empty_count) != 0u ||
            u32_bits(regs_valid.p11bb_qkscore_wq_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+WQ non-selected WQ marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11bb_lid0_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11bb_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }

        if (u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_fallback_seen_count) != 1u ||
            u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+WQ invalid marker mismatch mask(g=%u ne=%u fb=%u) wq(g=%u ne=%u fb=%u)\n",
                (unsigned)u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_fallback_seen_count) != 0u ||
            u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+WQ disabled marker mismatch mask(g=%u ne=%u fb=%u) wq(g=%u ne=%u fb=%u)\n",
                (unsigned)u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ay_lid0_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_fallback_seen_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ay_lid_nonzero_qkscore_mask_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+WQ lid!=0 MASK marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_lid0_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_lid_nonzero_qkscore_mask_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_non_empty_count) != 0u ||
            u32_bits(regs_lid_nonzero.p11bb_lid0_qkscore_wq_handoff_non_empty_count) != 0u ||
            u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_fallback_seen_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11bb_lid_nonzero_qkscore_wq_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+WQ lid!=0 WQ marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_lid0_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_lid_nonzero_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }

        const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const uint32_t score_base = u32_bits(sc.attn.score_base_word);
        const uint32_t score_words = (uint32_t)aecct::ATTN_TOKEN_COUNT * (uint32_t)aecct::ATTN_N_HEADS;
        if (!check_equal_region_between_srams(
                sram_valid,
                sram_baseline,
                score_base,
                score_words,
                "TOP-LOOP-QKSCORE-MULTI-MASK-WQ-valid-vs-baseline")) {
            return false;
        }

        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_WQ_POINTER_PRIORITY_CONSUME PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_WQ_POINTER_PRIORITY_INVALID_FALLBACK PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_WQ_POINTER_PRIORITY_DISABLED_FALLBACK PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_WQ_POINTER_PRIORITY_LID_NONZERO_FALLBACK PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_WQ_POINTER_PRIORITY_NON_SELECTED_ANTI_SPURIOUS_TOUCH PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_WQ_POINTER_PRIORITY_MARKER_CONSISTENCY PASS\n");
        return true;
    }

    bool run_loop_caller_qkscore_multi_seam_mask_qsrc_priority_pointer_case() {
        p11aeaf_tb::QkvPayloadSet payloads;
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads)) {
            std::printf("[p11anb][FAIL] qkscore multi-seam MASK+QSRC payload preparation failed\n");
            return false;
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        const aecct::CfgRegs bootstrap_cfg = p11aeaf_tb::build_cfg();
        auto run_case = [&](aecct::TopRegs& regs,
                            std::vector<aecct::u32_t>& sram,
                            uint32_t n_layers,
                            bool mask_enable,
                            bool mask_descriptor_valid,
                            bool qsrc_enable,
                            bool qsrc_descriptor_valid) {
            regs.clear();
            regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
            regs.cfg_d_model = bootstrap_cfg.d_model;
            regs.cfg_n_heads = bootstrap_cfg.n_heads;
            regs.cfg_d_ffn = bootstrap_cfg.d_ffn;
            regs.cfg_n_layers = (aecct::u32_t)n_layers;
            for (uint32_t i = 0u; i < kSramWords; ++i) {
                sram[i] = (aecct::u32_t)0u;
            }
            p11aeaf_tb::init_x_rows(sram);
            p11aeaf_tb::load_qkv_payload_set_to_sram(sram, payloads, param_base);
            aecct::run_transformer_layer_loop(
                regs,
                sram.data(),
                false,  // lid0_local_only_ffn_handoff_enable
                true,   // lid0_local_only_ffn_handoff_descriptor_valid
                false,  // lid0_local_only_attn_out_payload_enable
                true,   // lid0_local_only_attn_out_payload_descriptor_valid
                mask_enable,
                mask_descriptor_valid,
                false,  // lid0_local_only_qkscore_kvscan_handoff_enable
                true,   // lid0_local_only_qkscore_kvscan_handoff_descriptor_valid
                qsrc_enable,
                qsrc_descriptor_valid,
                false,  // lid0_local_only_qkscore_wq_handoff_enable
                true);  // lid0_local_only_qkscore_wq_handoff_descriptor_valid
        };

        std::vector<aecct::u32_t> sram_baseline(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_valid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_invalid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_disabled(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_lid_nonzero(kSramWords, (aecct::u32_t)0u);
        aecct::TopRegs regs_baseline;
        aecct::TopRegs regs_valid;
        aecct::TopRegs regs_invalid;
        aecct::TopRegs regs_disabled;
        aecct::TopRegs regs_lid_nonzero;

        run_case(regs_baseline, sram_baseline, 1u, false, true, false, true);
        run_case(regs_valid, sram_valid, 1u, true, true, true, true);
        run_case(regs_invalid, sram_invalid, 1u, true, false, true, true);
        run_case(regs_disabled, sram_disabled, 1u, false, true, false, true);
        run_case(regs_lid_nonzero, sram_lid_nonzero, 2u, true, true, true, true);

        if (!regs_valid.p11ae_mainline_score_path_taken || regs_valid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore multi-seam MASK+QSRC valid case did not stay on score mainline\n");
            return false;
        }
        if (regs_invalid.p11ae_mainline_score_path_taken || !regs_invalid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore multi-seam MASK+QSRC invalid case did not take fallback\n");
            return false;
        }
        if (!regs_disabled.p11ae_mainline_score_path_taken || regs_disabled.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore multi-seam MASK+QSRC disabled case unexpectedly left mainline\n");
            return false;
        }

        if (u32_bits(regs_valid.p11ay_qkscore_mask_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11ay_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ay_lid0_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ay_qkscore_mask_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+QSRC MASK-selected marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ay_lid0_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ay_qkscore_mask_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_valid.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+QSRC non-selected QSRC marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }

        if (u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+QSRC invalid selected marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11ay_qkscore_mask_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+QSRC invalid non-selected marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_fallback_seen_count) != 0u ||
            u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+QSRC disabled marker mismatch mask(g=%u ne=%u fb=%u) qsrc(g=%u ne=%u fb=%u)\n",
                (unsigned)u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11ay_qkscore_mask_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ay_lid0_qkscore_mask_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_fallback_seen_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ay_lid_nonzero_qkscore_mask_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+QSRC lid!=0 MASK marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_lid0_qkscore_mask_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_qkscore_mask_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ay_lid_nonzero_qkscore_mask_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_lid_nonzero.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11ba_lid_nonzero_qkscore_qsrc_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam MASK+QSRC lid!=0 QSRC marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_lid_nonzero_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }

        const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const uint32_t score_base = u32_bits(sc.attn.score_base_word);
        const uint32_t score_words = (uint32_t)aecct::ATTN_TOKEN_COUNT * (uint32_t)aecct::ATTN_N_HEADS;
        if (!check_equal_region_between_srams(
                sram_valid,
                sram_baseline,
                score_base,
                score_words,
                "TOP-LOOP-QKSCORE-MULTI-MASK-QSRC-valid-vs-baseline")) {
            return false;
        }

        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_QSRC_POINTER_PRIORITY_CONSUME PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_QSRC_POINTER_PRIORITY_INVALID_FALLBACK PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_QSRC_POINTER_PRIORITY_DISABLED_FALLBACK PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_QSRC_POINTER_PRIORITY_LID_NONZERO_FALLBACK PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_QSRC_POINTER_PRIORITY_NON_SELECTED_ANTI_SPURIOUS_TOUCH PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_MASK_QSRC_POINTER_PRIORITY_MARKER_CONSISTENCY PASS\n");
        return true;
    }

    bool run_loop_caller_qkscore_multi_seam_wq_qsrc_priority_pointer_case() {
        p11aeaf_tb::QkvPayloadSet payloads;
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads)) {
            std::printf("[p11anb][FAIL] qkscore multi-seam WQ+QSRC payload preparation failed\n");
            return false;
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        const aecct::CfgRegs bootstrap_cfg = p11aeaf_tb::build_cfg();
        auto run_case = [&](aecct::TopRegs& regs,
                            std::vector<aecct::u32_t>& sram,
                            uint32_t n_layers,
                            bool qsrc_enable,
                            bool qsrc_descriptor_valid,
                            bool wq_enable,
                            bool wq_descriptor_valid) {
            regs.clear();
            regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
            regs.cfg_d_model = bootstrap_cfg.d_model;
            regs.cfg_n_heads = bootstrap_cfg.n_heads;
            regs.cfg_d_ffn = bootstrap_cfg.d_ffn;
            regs.cfg_n_layers = (aecct::u32_t)n_layers;
            for (uint32_t i = 0u; i < kSramWords; ++i) {
                sram[i] = (aecct::u32_t)0u;
            }
            p11aeaf_tb::init_x_rows(sram);
            p11aeaf_tb::load_qkv_payload_set_to_sram(sram, payloads, param_base);
            aecct::run_transformer_layer_loop(
                regs,
                sram.data(),
                false,  // lid0_local_only_ffn_handoff_enable
                true,   // lid0_local_only_ffn_handoff_descriptor_valid
                false,  // lid0_local_only_attn_out_payload_enable
                true,   // lid0_local_only_attn_out_payload_descriptor_valid
                false,  // lid0_local_only_qkscore_mask_handoff_enable
                true,   // lid0_local_only_qkscore_mask_handoff_descriptor_valid
                false,  // lid0_local_only_qkscore_kvscan_handoff_enable
                true,   // lid0_local_only_qkscore_kvscan_handoff_descriptor_valid
                qsrc_enable,
                qsrc_descriptor_valid,
                wq_enable,
                wq_descriptor_valid);
        };

        std::vector<aecct::u32_t> sram_baseline(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_valid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_invalid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_disabled(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_lid_nonzero(kSramWords, (aecct::u32_t)0u);
        aecct::TopRegs regs_baseline;
        aecct::TopRegs regs_valid;
        aecct::TopRegs regs_invalid;
        aecct::TopRegs regs_disabled;
        aecct::TopRegs regs_lid_nonzero;

        run_case(regs_baseline, sram_baseline, 1u, false, true, false, true);
        run_case(regs_valid, sram_valid, 1u, true, true, true, true);
        run_case(regs_invalid, sram_invalid, 1u, true, true, true, false);
        run_case(regs_disabled, sram_disabled, 1u, false, true, false, true);
        run_case(regs_lid_nonzero, sram_lid_nonzero, 2u, true, true, true, true);

        if (!regs_valid.p11ae_mainline_score_path_taken || regs_valid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore multi-seam WQ+QSRC valid case did not stay on score mainline\n");
            return false;
        }
        if (regs_invalid.p11ae_mainline_score_path_taken || !regs_invalid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore multi-seam WQ+QSRC invalid case did not take fallback\n");
            return false;
        }
        if (!regs_disabled.p11ae_mainline_score_path_taken || regs_disabled.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore multi-seam WQ+QSRC disabled case unexpectedly left mainline\n");
            return false;
        }

        if (u32_bits(regs_valid.p11bb_qkscore_wq_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11bb_qkscore_wq_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11bb_lid0_qkscore_wq_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11bb_qkscore_wq_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] multi-seam WQ+QSRC WQ-selected marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11bb_lid0_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11bb_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_valid.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam WQ+QSRC non-selected QSRC marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }

        if (u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam WQ+QSRC invalid selected marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11bb_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam WQ+QSRC invalid non-selected marker mismatch gate=%u non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_invalid.p11ba_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_fallback_seen_count) != 0u ||
            u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 0u ||
            u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] multi-seam WQ+QSRC disabled marker mismatch wq(g=%u ne=%u fb=%u) qsrc(g=%u ne=%u fb=%u)\n",
                (unsigned)u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11bb_qkscore_wq_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_disabled.p11ba_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11bb_lid0_qkscore_wq_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_fallback_seen_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11bb_lid_nonzero_qkscore_wq_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam WQ+QSRC lid!=0 WQ marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_lid0_qkscore_wq_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_qkscore_wq_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11bb_lid_nonzero_qkscore_wq_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_lid_nonzero.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count) != 0u ||
            u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11ba_lid_nonzero_qkscore_qsrc_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam WQ+QSRC lid!=0 QSRC marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_lid_nonzero_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }

        const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const uint32_t score_base = u32_bits(sc.attn.score_base_word);
        const uint32_t score_words = (uint32_t)aecct::ATTN_TOKEN_COUNT * (uint32_t)aecct::ATTN_N_HEADS;
        if (!check_equal_region_between_srams(
                sram_valid,
                sram_baseline,
                score_base,
                score_words,
                "TOP-LOOP-QKSCORE-MULTI-WQ-QSRC-valid-vs-baseline")) {
            return false;
        }

        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_WQ_QSRC_POINTER_PRIORITY_CONSUME PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_WQ_QSRC_POINTER_PRIORITY_INVALID_FALLBACK PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_WQ_QSRC_POINTER_PRIORITY_DISABLED_FALLBACK PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_WQ_QSRC_POINTER_PRIORITY_LID_NONZERO_FALLBACK PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_WQ_QSRC_POINTER_PRIORITY_NON_SELECTED_ANTI_SPURIOUS_TOUCH PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_WQ_QSRC_POINTER_PRIORITY_MARKER_CONSISTENCY PASS\n");
        return true;
    }

    bool run_loop_caller_qkscore_multi_seam_qsrc_kvscan_priority_pointer_case() {
        p11aeaf_tb::QkvPayloadSet payloads;
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads)) {
            std::printf("[p11anb][FAIL] qkscore multi-seam QSRC+KVSCAN payload preparation failed\n");
            return false;
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        const aecct::CfgRegs bootstrap_cfg = p11aeaf_tb::build_cfg();
        auto run_case = [&](aecct::TopRegs& regs,
                            std::vector<aecct::u32_t>& sram,
                            uint32_t n_layers,
                            bool kvscan_enable,
                            bool kvscan_descriptor_valid,
                            bool qsrc_enable,
                            bool qsrc_descriptor_valid) {
            regs.clear();
            regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
            regs.cfg_d_model = bootstrap_cfg.d_model;
            regs.cfg_n_heads = bootstrap_cfg.n_heads;
            regs.cfg_d_ffn = bootstrap_cfg.d_ffn;
            regs.cfg_n_layers = (aecct::u32_t)n_layers;
            for (uint32_t i = 0u; i < kSramWords; ++i) {
                sram[i] = (aecct::u32_t)0u;
            }
            p11aeaf_tb::init_x_rows(sram);
            p11aeaf_tb::load_qkv_payload_set_to_sram(sram, payloads, param_base);
            aecct::run_transformer_layer_loop(
                regs,
                sram.data(),
                false,  // lid0_local_only_ffn_handoff_enable
                true,   // lid0_local_only_ffn_handoff_descriptor_valid
                false,  // lid0_local_only_attn_out_payload_enable
                true,   // lid0_local_only_attn_out_payload_descriptor_valid
                false,  // lid0_local_only_qkscore_mask_handoff_enable
                true,   // lid0_local_only_qkscore_mask_handoff_descriptor_valid
                kvscan_enable,
                kvscan_descriptor_valid,
                qsrc_enable,
                qsrc_descriptor_valid,
                false,  // lid0_local_only_qkscore_wq_handoff_enable
                true);  // lid0_local_only_qkscore_wq_handoff_descriptor_valid
        };

        std::vector<aecct::u32_t> sram_baseline(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_valid(kSramWords, (aecct::u32_t)0u);
        std::vector<aecct::u32_t> sram_lid_nonzero(kSramWords, (aecct::u32_t)0u);
        aecct::TopRegs regs_baseline;
        aecct::TopRegs regs_valid;
        aecct::TopRegs regs_lid_nonzero;

        run_case(regs_baseline, sram_baseline, 1u, false, true, false, true);
        run_case(regs_valid, sram_valid, 1u, true, true, true, true);
        run_case(regs_lid_nonzero, sram_lid_nonzero, 2u, true, true, true, true);

        if (!regs_valid.p11ae_mainline_score_path_taken || regs_valid.p11ae_score_fallback_taken) {
            std::printf("[p11anb][FAIL] qkscore multi-seam QSRC+KVSCAN valid case did not stay on score mainline\n");
            return false;
        }
        if (u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count) != 1u ||
            u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 0u) {
            std::printf("[p11anb][FAIL] multi-seam QSRC+KVSCAN QSRC-selected marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11ba_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_gate_taken_count) != 1u ||
            u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_non_empty_count) != 0u ||
            u32_bits(regs_valid.p11az_lid0_qkscore_kvscan_handoff_non_empty_count) != 0u ||
            u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam QSRC+KVSCAN non-selected KVSCAN marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u\n",
                (unsigned)u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11az_lid0_qkscore_kvscan_handoff_non_empty_count),
                (unsigned)u32_bits(regs_valid.p11az_qkscore_kvscan_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_fallback_seen_count) != 1u ||
            u32_bits(regs_lid_nonzero.p11ba_lid_nonzero_qkscore_qsrc_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam QSRC+KVSCAN lid!=0 QSRC marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_lid0_qkscore_qsrc_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_qkscore_qsrc_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11ba_lid_nonzero_qkscore_qsrc_handoff_fallback_seen_count));
            return false;
        }
        if (u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_gate_taken_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_non_empty_count) != 0u ||
            u32_bits(regs_lid_nonzero.p11az_lid0_qkscore_kvscan_handoff_non_empty_count) != 0u ||
            u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_fallback_seen_count) != 2u ||
            u32_bits(regs_lid_nonzero.p11az_lid_nonzero_qkscore_kvscan_handoff_fallback_seen_count) != 1u) {
            std::printf("[p11anb][FAIL] multi-seam QSRC+KVSCAN lid!=0 KVSCAN marker mismatch gate=%u non_empty=%u lid0_non_empty=%u fallback=%u lid_nonzero_fallback=%u\n",
                (unsigned)u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_gate_taken_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11az_lid0_qkscore_kvscan_handoff_non_empty_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11az_qkscore_kvscan_handoff_fallback_seen_count),
                (unsigned)u32_bits(regs_lid_nonzero.p11az_lid_nonzero_qkscore_kvscan_handoff_fallback_seen_count));
            return false;
        }

        const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const uint32_t score_base = u32_bits(sc.attn.score_base_word);
        const uint32_t score_words = (uint32_t)aecct::ATTN_TOKEN_COUNT * (uint32_t)aecct::ATTN_N_HEADS;
        if (!check_equal_region_between_srams(
                sram_valid,
                sram_baseline,
                score_base,
                score_words,
                "TOP-LOOP-QKSCORE-MULTI-QSRC-KVSCAN-valid-vs-baseline")) {
            return false;
        }

        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_QSRC_KVSCAN_POINTER_PRIORITY_CONSUME PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_QSRC_KVSCAN_POINTER_PRIORITY_LID_NONZERO_FALLBACK PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_QSRC_KVSCAN_POINTER_PRIORITY_NON_SELECTED_ANTI_SPURIOUS_TOUCH PASS\n");
        std::printf("P11ANB_LOOP_CALLER_QKSCORE_MULTI_SEAM_QSRC_KVSCAN_POINTER_PRIORITY_MARKER_CONSISTENCY PASS\n");
        return true;
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
