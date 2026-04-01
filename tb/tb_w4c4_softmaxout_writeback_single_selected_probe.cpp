// W4-C4: SoftmaxOut WRITEBACK-path single selected probe (local-only).

#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <vector>

#include "tb_p11aeaf_common.h"
#include "tb_w4cfamily_softmaxout_harness_common.h"

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

class TbW4c4SoftmaxOutWritebackSingleSelectedProbe {
public:
    int run_all() {
        if (!init_and_bootstrap()) { return 1; }
        if (!run_positive_probe_case()) { return 1; }
        if (!run_negative_mismatch_case()) { return 1; }
        std::printf("PASS: tb_w4c4_softmaxout_writeback_single_selected_probe\n");
        return 0;
    }

private:
    static const uint32_t kFamilyCases = 1u;

    std::vector<aecct::u32_t> sram_bootstrap_;
    std::vector<aecct::u32_t> expected_out_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::LayerScratch sc_;
    aecct::CfgRegs cfg_;
    uint32_t selected_head_ = 0u;
    uint32_t selected_d_tile_ = 0u;
    uint32_t selected_valid_words_ = 0u;
    uint32_t selected_key_token_ = 1u;

    bool init_and_bootstrap() {
        if (!w4c_softmax_family::bootstrap_mainline_context(
                "w4c4",
                sram_bootstrap_,
                payloads_,
                sc_,
                cfg_)) {
            return false;
        }

        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        const uint32_t tile_words = (uint32_t)aecct::ATTN_TOP_MANAGED_WORK_TILE_WORDS;
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        if (token_count < 2u) {
            std::printf("[w4c4][FAIL] token_count < 2 cannot validate later-token writeback path\n");
            return false;
        }

        selected_head_ = 0u;
        selected_d_tile_ = 0u;
        selected_valid_words_ =
            aecct::attn_top_managed_tile_valid_words(d_head, tile_words, selected_d_tile_);
        if (selected_valid_words_ == 0u || selected_valid_words_ > tile_words) {
            std::printf("[w4c4][FAIL] invalid selected tile valid words\n");
            return false;
        }

        // Force selected later token onto RENORM branch first.
        // C4 then probes the final WRITEBACK visibility for this selected descriptor.
        w4c_softmax_family::force_head_score(
            sram_bootstrap_, sc_, selected_head_, 0u, -8.0f);
        w4c_softmax_family::force_head_score(
            sram_bootstrap_, sc_, selected_head_, selected_key_token_, 8.0f);
        return true;
    }

    void build_selected_family_payload(
        std::vector<aecct::u32_t>& family_base_words,
        std::vector<aecct::u32_t>& family_words_flat,
        std::vector<aecct::u32_t>& family_valid_words,
        std::vector<aecct::u32_t>& family_d_tile_idx,
        std::vector<aecct::u32_t>& family_head_idx,
        std::vector<aecct::u32_t>& family_key_token_begin,
        std::vector<aecct::u32_t>& family_key_token_count) const {
        const uint32_t v_base = (uint32_t)sc_.attn.v_base_word.to_uint();
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t tile_words = (uint32_t)aecct::ATTN_TOP_MANAGED_WORK_TILE_WORDS;

        family_base_words.assign(kFamilyCases, (aecct::u32_t)0u);
        family_valid_words.assign(kFamilyCases, (aecct::u32_t)0u);
        family_d_tile_idx.assign(kFamilyCases, (aecct::u32_t)0u);
        family_head_idx.assign(kFamilyCases, (aecct::u32_t)0u);
        family_key_token_begin.assign(kFamilyCases, (aecct::u32_t)0u);
        family_key_token_count.assign(kFamilyCases, (aecct::u32_t)0u);
        family_words_flat.assign(kFamilyCases * tile_words, (aecct::u32_t)0u);

        const uint32_t tile_offset = selected_d_tile_ * tile_words;
        const uint32_t base = v_base + selected_key_token_ * d_model +
                              selected_head_ * d_head + tile_offset;
        family_base_words[0u] = (aecct::u32_t)base;
        family_valid_words[0u] = (aecct::u32_t)selected_valid_words_;
        family_d_tile_idx[0u] = (aecct::u32_t)selected_d_tile_;
        family_head_idx[0u] = (aecct::u32_t)selected_head_;
        family_key_token_begin[0u] = (aecct::u32_t)selected_key_token_;
        family_key_token_count[0u] = (aecct::u32_t)1u;

        ATTN_W4C4_BUILD_FAMILY_PAYLOAD_LOOP: for (uint32_t i = 0u; i < selected_valid_words_; ++i) {
            family_words_flat[i] = sram_bootstrap_[base + i];
        }
    }

    bool run_positive_probe_case() {
        std::vector<aecct::u32_t> sram_probe = sram_bootstrap_;
        std::vector<aecct::u32_t> sram_legacy = sram_bootstrap_;
        const uint32_t token_idx = 0u;
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;

        p11aeaf_tb::compute_expected_output_row_online(
            sram_bootstrap_,
            sc_.attn,
            token_idx,
            token_count,
            n_heads,
            d_head,
            expected_out_);

        std::vector<aecct::u32_t> family_base_words;
        std::vector<aecct::u32_t> family_words_flat;
        std::vector<aecct::u32_t> family_valid_words;
        std::vector<aecct::u32_t> family_d_tile_idx;
        std::vector<aecct::u32_t> family_head_idx;
        std::vector<aecct::u32_t> family_key_token_begin;
        std::vector<aecct::u32_t> family_key_token_count;
        build_selected_family_payload(
            family_base_words,
            family_words_flat,
            family_valid_words,
            family_d_tile_idx,
            family_head_idx,
            family_key_token_begin,
            family_key_token_count);

        bool fallback_taken_probe = true;
        aecct::u32_t family_visible_count = 0;
        aecct::u32_t family_owner_ok = 0;
        aecct::u32_t family_consumed_count = 0;
        aecct::u32_t family_compare_ok = 0;
        aecct::u32_t family_case_mask = 0;
        aecct::u32_t family_desc_visible_count = 0;
        aecct::u32_t family_desc_case_mask = 0;
        aecct::u32_t family_renorm_selected_count = 0;
        aecct::u32_t family_renorm_case_mask = 0;
        aecct::u32_t family_writeback_selected_count = 0;
        aecct::u32_t family_writeback_case_mask = 0;
        aecct::u32_t family_writeback_touch_count = 0;
        const bool softmax_mainline_taken_probe =
            aecct::run_p11af_layer0_top_managed_softmax_out(
                sram_probe.data(),
                cfg_,
                sc_,
                (aecct::u32_t)token_idx,
                fallback_taken_probe,
                (aecct::u32_t)0u,
                0,
                (aecct::u32_t)0u,
                0,
                0,
                0,
                (aecct::u32_t)0u,
                0,
                (aecct::u32_t)0u,
                (aecct::u32_t)0u,
                0,
                0,
                0,
                0,
                (aecct::u32_t)kFamilyCases,
                family_base_words.data(),
                family_words_flat.data(),
                family_valid_words.data(),
                family_d_tile_idx.data(),
                &family_visible_count,
                &family_owner_ok,
                &family_consumed_count,
                &family_compare_ok,
                &family_case_mask,
                family_head_idx.data(),
                family_key_token_begin.data(),
                family_key_token_count.data(),
                &family_desc_visible_count,
                &family_desc_case_mask,
                &family_renorm_selected_count,
                &family_renorm_case_mask,
                &family_writeback_selected_count,
                &family_writeback_case_mask,
                &family_writeback_touch_count);

        if (!softmax_mainline_taken_probe || fallback_taken_probe) {
            std::printf("[w4c4][FAIL] positive writeback probe run did not stay on mainline\n");
            return false;
        }

        if ((uint32_t)family_visible_count.to_uint() != 1u ||
            (uint32_t)family_owner_ok.to_uint() != 1u ||
            (uint32_t)family_compare_ok.to_uint() != 1u ||
            (uint32_t)family_consumed_count.to_uint() != selected_valid_words_ ||
            (uint32_t)family_case_mask.to_uint() != 0x1u ||
            (uint32_t)family_desc_visible_count.to_uint() != 1u ||
            (uint32_t)family_desc_case_mask.to_uint() != 0x1u ||
            (uint32_t)family_renorm_selected_count.to_uint() != 1u ||
            (uint32_t)family_renorm_case_mask.to_uint() != 0x1u ||
            (uint32_t)family_writeback_selected_count.to_uint() != 1u ||
            (uint32_t)family_writeback_case_mask.to_uint() != 0x1u ||
            (uint32_t)family_writeback_touch_count.to_uint() != (selected_valid_words_ * 3u)) {
            std::printf("[w4c4][FAIL] writeback observability mismatch visible=%u owner=%u consumed=%u compare=%u mask=0x%X desc_visible=%u desc_mask=0x%X renorm_count=%u renorm_mask=0x%X wb_count=%u wb_mask=0x%X wb_touch=%u wb_expected_touch=%u\n",
                (unsigned)(uint32_t)family_visible_count.to_uint(),
                (unsigned)(uint32_t)family_owner_ok.to_uint(),
                (unsigned)(uint32_t)family_consumed_count.to_uint(),
                (unsigned)(uint32_t)family_compare_ok.to_uint(),
                (unsigned)(uint32_t)family_case_mask.to_uint(),
                (unsigned)(uint32_t)family_desc_visible_count.to_uint(),
                (unsigned)(uint32_t)family_desc_case_mask.to_uint(),
                (unsigned)(uint32_t)family_renorm_selected_count.to_uint(),
                (unsigned)(uint32_t)family_renorm_case_mask.to_uint(),
                (unsigned)(uint32_t)family_writeback_selected_count.to_uint(),
                (unsigned)(uint32_t)family_writeback_case_mask.to_uint(),
                (unsigned)(uint32_t)family_writeback_touch_count.to_uint(),
                (unsigned)(selected_valid_words_ * 3u));
            return false;
        }

        std::printf("W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_VISIBLE PASS\n");
        std::printf("W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_OWNERSHIP_CHECK PASS\n");
        std::printf("W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_SELECTOR_CASE_MASK PASS\n");
        std::printf("W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_WRITEBACK_PATH_VISIBILITY PASS\n");
        std::printf("W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_WRITEBACK_TOUCH_COUNT_EXACT PASS\n");
        std::printf("W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_ANTI_FALLBACK PASS\n");

        bool fallback_taken_legacy = true;
        const bool softmax_mainline_taken_legacy =
            aecct::run_p11af_layer0_top_managed_softmax_out(
                sram_legacy.data(),
                cfg_,
                sc_,
                (aecct::u32_t)token_idx,
                fallback_taken_legacy);
        if (!softmax_mainline_taken_legacy || fallback_taken_legacy) {
            std::printf("[w4c4][FAIL] legacy run did not stay on mainline\n");
            return false;
        }

        const uint32_t out_row_base = (uint32_t)sc_.attn_out_base_word.to_uint() + token_idx * d_model;
        const uint32_t pre_row_base = (uint32_t)sc_.attn.pre_concat_base_word.to_uint() + token_idx * d_model;
        const uint32_t post_row_base = (uint32_t)sc_.attn.post_concat_base_word.to_uint() + token_idx * d_model;
        ATTN_W4C4_EXPECTED_COMPARE_LOOP: for (uint32_t i = 0u; i < d_model; ++i) {
            const uint32_t exp = (uint32_t)expected_out_[i].to_uint();
            const uint32_t got_pre = (uint32_t)sram_probe[pre_row_base + i].to_uint();
            const uint32_t got_post = (uint32_t)sram_probe[post_row_base + i].to_uint();
            const uint32_t got_out = (uint32_t)sram_probe[out_row_base + i].to_uint();
            const uint32_t legacy_pre = (uint32_t)sram_legacy[pre_row_base + i].to_uint();
            const uint32_t legacy_post = (uint32_t)sram_legacy[post_row_base + i].to_uint();
            const uint32_t legacy_out = (uint32_t)sram_legacy[out_row_base + i].to_uint();
            if (got_pre != exp || got_post != exp || got_out != exp) {
                std::printf("[w4c4][FAIL] expected compare mismatch idx=%u pre=0x%08X post=0x%08X out=0x%08X exp=0x%08X\n",
                    (unsigned)i,
                    (unsigned)got_pre,
                    (unsigned)got_post,
                    (unsigned)got_out,
                    (unsigned)exp);
                return false;
            }
            if (got_pre != legacy_pre || got_post != legacy_post || got_out != legacy_out) {
                std::printf("[w4c4][FAIL] legacy compare mismatch idx=%u bridge(pre/post/out)=(0x%08X/0x%08X/0x%08X) legacy=(0x%08X/0x%08X/0x%08X)\n",
                    (unsigned)i,
                    (unsigned)got_pre,
                    (unsigned)got_post,
                    (unsigned)got_out,
                    (unsigned)legacy_pre,
                    (unsigned)legacy_post,
                    (unsigned)legacy_out);
                return false;
            }
        }
        std::printf("W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_EXPECTED_COMPARE PASS\n");
        std::printf("W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_LEGACY_COMPARE PASS\n");

        std::vector<uint8_t> allowed_write((uint32_t)sram_probe.size(), 0u);
        for (uint32_t i = 0u; i < d_model; ++i) {
            allowed_write[pre_row_base + i] = 1u;
            allowed_write[post_row_base + i] = 1u;
            allowed_write[out_row_base + i] = 1u;
        }
        ATTN_W4C4_SPURIOUS_CHECK_LOOP: for (uint32_t i = 0u; i < (uint32_t)sram_probe.size(); ++i) {
            const uint32_t before = (uint32_t)sram_bootstrap_[i].to_uint();
            const uint32_t after = (uint32_t)sram_probe[i].to_uint();
            if (before != after && allowed_write[i] == 0u) {
                std::printf("[w4c4][FAIL] spurious write addr=%u before=0x%08X after=0x%08X\n",
                    (unsigned)i,
                    (unsigned)before,
                    (unsigned)after);
                return false;
            }
        }
        std::printf("W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_NO_SPURIOUS_TOUCH PASS\n");
        return true;
    }

    bool run_negative_mismatch_case() {
        std::vector<aecct::u32_t> sram_negative = sram_bootstrap_;
        const uint32_t token_idx = 0u;

        std::vector<aecct::u32_t> family_base_words;
        std::vector<aecct::u32_t> family_words_flat;
        std::vector<aecct::u32_t> family_valid_words;
        std::vector<aecct::u32_t> family_d_tile_idx;
        std::vector<aecct::u32_t> family_head_idx;
        std::vector<aecct::u32_t> family_key_token_begin;
        std::vector<aecct::u32_t> family_key_token_count;
        build_selected_family_payload(
            family_base_words,
            family_words_flat,
            family_valid_words,
            family_d_tile_idx,
            family_head_idx,
            family_key_token_begin,
            family_key_token_count);

        // Mismatch at selected payload should reject before writeback.
        family_words_flat[0u] =
            (aecct::u32_t)((uint32_t)family_words_flat[0u].to_uint() ^ 0x00000001u);

        bool fallback_taken = true;
        aecct::u32_t family_visible_count = 0;
        aecct::u32_t family_owner_ok = 0;
        aecct::u32_t family_consumed_count = 0;
        aecct::u32_t family_compare_ok = 0;
        aecct::u32_t family_case_mask = 0;
        aecct::u32_t family_desc_visible_count = 0;
        aecct::u32_t family_desc_case_mask = 0;
        aecct::u32_t family_renorm_selected_count = 0;
        aecct::u32_t family_renorm_case_mask = 0;
        aecct::u32_t family_writeback_selected_count = 0;
        aecct::u32_t family_writeback_case_mask = 0;
        aecct::u32_t family_writeback_touch_count = 0;
        const bool softmax_mainline_taken =
            aecct::run_p11af_layer0_top_managed_softmax_out(
                sram_negative.data(),
                cfg_,
                sc_,
                (aecct::u32_t)token_idx,
                fallback_taken,
                (aecct::u32_t)0u,
                0,
                (aecct::u32_t)0u,
                0,
                0,
                0,
                (aecct::u32_t)0u,
                0,
                (aecct::u32_t)0u,
                (aecct::u32_t)0u,
                0,
                0,
                0,
                0,
                (aecct::u32_t)kFamilyCases,
                family_base_words.data(),
                family_words_flat.data(),
                family_valid_words.data(),
                family_d_tile_idx.data(),
                &family_visible_count,
                &family_owner_ok,
                &family_consumed_count,
                &family_compare_ok,
                &family_case_mask,
                family_head_idx.data(),
                family_key_token_begin.data(),
                family_key_token_count.data(),
                &family_desc_visible_count,
                &family_desc_case_mask,
                &family_renorm_selected_count,
                &family_renorm_case_mask,
                &family_writeback_selected_count,
                &family_writeback_case_mask,
                &family_writeback_touch_count);

        if (softmax_mainline_taken || !fallback_taken) {
            std::printf("[w4c4][FAIL] mismatch payload did not reject as expected\n");
            return false;
        }

        if ((uint32_t)family_visible_count.to_uint() != 1u ||
            (uint32_t)family_owner_ok.to_uint() != 1u ||
            (uint32_t)family_consumed_count.to_uint() != 0u ||
            (uint32_t)family_compare_ok.to_uint() != 0u ||
            (uint32_t)family_case_mask.to_uint() != 0x1u ||
            (uint32_t)family_desc_visible_count.to_uint() != 1u ||
            (uint32_t)family_desc_case_mask.to_uint() != 0x1u ||
            (uint32_t)family_renorm_selected_count.to_uint() != 0u ||
            (uint32_t)family_renorm_case_mask.to_uint() != 0u ||
            (uint32_t)family_writeback_selected_count.to_uint() != 0u ||
            (uint32_t)family_writeback_case_mask.to_uint() != 0u ||
            (uint32_t)family_writeback_touch_count.to_uint() != 0u) {
            std::printf("[w4c4][FAIL] mismatch flags mismatch visible=%u owner=%u consumed=%u compare=%u mask=0x%X desc_visible=%u desc_mask=0x%X renorm_count=%u renorm_mask=0x%X wb_count=%u wb_mask=0x%X wb_touch=%u\n",
                (unsigned)(uint32_t)family_visible_count.to_uint(),
                (unsigned)(uint32_t)family_owner_ok.to_uint(),
                (unsigned)(uint32_t)family_consumed_count.to_uint(),
                (unsigned)(uint32_t)family_compare_ok.to_uint(),
                (unsigned)(uint32_t)family_case_mask.to_uint(),
                (unsigned)(uint32_t)family_desc_visible_count.to_uint(),
                (unsigned)(uint32_t)family_desc_case_mask.to_uint(),
                (unsigned)(uint32_t)family_renorm_selected_count.to_uint(),
                (unsigned)(uint32_t)family_renorm_case_mask.to_uint(),
                (unsigned)(uint32_t)family_writeback_selected_count.to_uint(),
                (unsigned)(uint32_t)family_writeback_case_mask.to_uint(),
                (unsigned)(uint32_t)family_writeback_touch_count.to_uint());
            return false;
        }

        ATTN_W4C4_REJECT_STALE_CHECK_LOOP: for (uint32_t i = 0u; i < (uint32_t)sram_negative.size(); ++i) {
            if ((uint32_t)sram_negative[i].to_uint() != (uint32_t)sram_bootstrap_[i].to_uint()) {
                std::printf("[w4c4][FAIL] reject path wrote stale state addr=%u\n", (unsigned)i);
                return false;
            }
        }
        std::printf("W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_MISMATCH_REJECT PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbW4c4SoftmaxOutWritebackSingleSelectedProbe tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__

