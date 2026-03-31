// W4-B7: QkScore bounded mixed single+family bridge validation (local-only).

#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <vector>

#include "tb_p11aeaf_common.h"

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

class TbW4b7QkScoreMixedBridge {
public:
    int run_all() {
        if (!init_and_bootstrap()) { return 1; }
        if (!run_positive_mixed_case()) { return 1; }
        if (!run_negative_mixed_mismatch_case()) { return 1; }
        std::printf("PASS: tb_w4b7_qkscore_mixed_bridge\n");
        return 0;
    }

private:
    static const uint32_t kMaxFamilyCases = 4u;

    std::vector<aecct::u32_t> sram_bootstrap_;
    std::vector<aecct::u32_t> expected_scores_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::LayerScratch sc_;
    aecct::CfgRegs cfg_;
    uint32_t case_count_ = 0u;
    uint32_t case_head_[kMaxFamilyCases];
    uint32_t case_key_begin_[kMaxFamilyCases];
    uint32_t case_valid_words_[kMaxFamilyCases];
    uint32_t case_total_words_ = 0u;
    uint32_t single_head_ = 0u;
    uint32_t single_key_begin_ = 0u;
    uint32_t single_valid_words_ = 1u;

    bool init_and_bootstrap() {
        sram_bootstrap_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        p11aeaf_tb::init_x_rows(sram_bootstrap_);
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads_)) {
            std::printf("[w4b7][FAIL] payload preparation failed\n");
            return false;
        }

        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_bootstrap_, payloads_, param_base);
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        cfg_ = p11aeaf_tb::build_cfg();

        bool q_fallback_taken = true;
        bool kv_fallback_taken = true;
        if (!p11aeaf_tb::run_ac_ad_mainline(sram_bootstrap_, q_fallback_taken, kv_fallback_taken)) {
            std::printf("[w4b7][FAIL] AC/AD bootstrap failed\n");
            return false;
        }
        if (q_fallback_taken || kv_fallback_taken) {
            std::printf("[w4b7][FAIL] AC/AD bootstrap fallback detected\n");
            return false;
        }

        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t tile_words = (uint32_t)aecct::ATTN_TOP_MANAGED_WORK_TILE_WORDS;
        if (token_count < 2u || n_heads < 2u || tile_words == 0u) {
            std::printf("[w4b7][FAIL] invalid topology for mixed bridge\n");
            return false;
        }

        case_count_ = n_heads >= 5u ? 4u : (n_heads >= 4u ? 3u : 2u);
        case_head_[0] = 0u;
        case_key_begin_[0] = 0u;
        case_valid_words_[0] = (token_count >= 2u && tile_words >= 2u) ? 2u : 1u;

        case_head_[1] = 1u;
        case_key_begin_[1] = (token_count > 3u) ? 2u : 1u;
        const uint32_t remain1 = token_count - case_key_begin_[1];
        case_valid_words_[1] = (remain1 >= 2u && tile_words >= 2u) ? 2u : 1u;

        if (case_count_ >= 3u) {
            case_head_[2] = 2u;
            case_key_begin_[2] = (token_count > 6u) ? 5u : (token_count - 1u);
            const uint32_t remain2 = token_count - case_key_begin_[2];
            case_valid_words_[2] = (remain2 >= 2u && tile_words >= 2u) ? 2u : 1u;
        }
        if (case_count_ >= 4u) {
            case_head_[3] = 3u;
            case_key_begin_[3] = (token_count > 8u) ? 7u : (token_count - 1u);
            const uint32_t remain3 = token_count - case_key_begin_[3];
            case_valid_words_[3] = (remain3 >= 2u && tile_words >= 2u) ? 2u : 1u;
        }

        case_total_words_ = 0u;
        for (uint32_t c = 0u; c < case_count_; ++c) {
            if (case_valid_words_[c] == 0u || case_valid_words_[c] > tile_words) {
                std::printf("[w4b7][FAIL] invalid case_valid_words for case %u\n", (unsigned)c);
                return false;
            }
            if (case_key_begin_[c] >= token_count ||
                (case_key_begin_[c] + case_valid_words_[c]) > token_count) {
                std::printf("[w4b7][FAIL] invalid case key range for case %u\n", (unsigned)c);
                return false;
            }
            case_total_words_ += case_valid_words_[c];
        }
        if (n_heads <= case_count_) {
            std::printf("[w4b7][FAIL] mixed bridge requires one extra head beyond family cases\n");
            return false;
        }
        single_head_ = case_count_;
        single_key_begin_ = (token_count > 4u) ? 3u : (token_count - 1u);
        single_valid_words_ = 1u;
        return true;
    }

    void build_family_bridge_payload(
        std::vector<aecct::u32_t>& base_words,
        std::vector<aecct::u32_t>& words_flat,
        std::vector<aecct::u32_t>& valid_words,
        std::vector<aecct::u32_t>& key_begin,
        std::vector<aecct::u32_t>& head_idx) const {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t family_stride = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        base_words.assign(case_count_, (aecct::u32_t)0u);
        words_flat.assign(case_count_ * family_stride, (aecct::u32_t)0u);
        valid_words.assign(case_count_, (aecct::u32_t)0u);
        key_begin.assign(case_count_, (aecct::u32_t)0u);
        head_idx.assign(case_count_, (aecct::u32_t)0u);
        for (uint32_t c = 0u; c < case_count_; ++c) {
            const uint32_t h = case_head_[c];
            const uint32_t begin = case_key_begin_[c];
            const uint32_t valid = case_valid_words_[c];
            base_words[c] = (aecct::u32_t)(score_base + h * token_count + begin);
            valid_words[c] = (aecct::u32_t)valid;
            key_begin[c] = (aecct::u32_t)begin;
            head_idx[c] = (aecct::u32_t)h;
            for (uint32_t i = 0u; i < valid; ++i) {
                const uint32_t src = h * token_count + begin + i;
                words_flat[c * family_stride + i] = expected_scores_[src];
            }
        }
    }

    bool run_positive_mixed_case() {
        std::vector<aecct::u32_t> sram_bridge = sram_bootstrap_;
        std::vector<aecct::u32_t> sram_legacy = sram_bootstrap_;
        const uint32_t token_idx = 0u;
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;

        p11aeaf_tb::compute_expected_score_row(
            sram_bootstrap_,
            sc_.attn,
            token_idx,
            token_count,
            n_heads,
            d_head,
            expected_scores_);

        std::vector<aecct::u32_t> family_base_words;
        std::vector<aecct::u32_t> family_words_flat;
        std::vector<aecct::u32_t> family_valid_words;
        std::vector<aecct::u32_t> family_key_begin;
        std::vector<aecct::u32_t> family_head_idx;
        build_family_bridge_payload(
            family_base_words,
            family_words_flat,
            family_valid_words,
            family_key_begin,
            family_head_idx);

        bool fallback_taken_bridge = true;
        const uint32_t single_head = single_head_;
        const uint32_t single_key_begin = single_key_begin_;
        const uint32_t single_valid_words = single_valid_words_;
        const uint32_t single_base_word =
            (uint32_t)sc_.attn.score_base_word.to_uint() + single_head * token_count + single_key_begin;
        const aecct::u32_t single_bridge_word =
            expected_scores_[single_head * token_count + single_key_begin];
        aecct::u32_t single_visible = 0;
        aecct::u32_t single_owner_ok = 0;
        aecct::u32_t single_consumed = 0;
        aecct::u32_t single_compare_ok = 0;
        aecct::u32_t family_visible_count = 0;
        aecct::u32_t family_owner_ok = 0;
        aecct::u32_t family_consumed_count = 0;
        aecct::u32_t family_compare_ok = 0;
        aecct::u32_t family_case_mask = 0;
        const bool score_mainline_taken_bridge = aecct::run_p11ae_layer0_top_managed_qk_score(
            sram_bridge.data(),
            cfg_,
            sc_,
            (aecct::u32_t)token_idx,
            fallback_taken_bridge,
            (aecct::u32_t)0u,
            (aecct::u32_t)0u,
            0,
            0,
            (aecct::u32_t)0u,
            0,
            0,
            0,
            (aecct::u32_t)single_base_word,
            &single_bridge_word,
            (aecct::u32_t)single_valid_words,
            (aecct::u32_t)single_key_begin,
            &single_visible,
            &single_owner_ok,
            &single_consumed,
            &single_compare_ok,
            (aecct::u32_t)single_head,
            (aecct::u32_t)case_count_,
            family_base_words.data(),
            family_words_flat.data(),
            family_valid_words.data(),
            family_key_begin.data(),
            family_head_idx.data(),
            &family_visible_count,
            &family_owner_ok,
            &family_consumed_count,
            &family_compare_ok,
            &family_case_mask);

        if (!score_mainline_taken_bridge || fallback_taken_bridge) {
            std::printf("[w4b7][FAIL] mixed bridge run did not stay on mainline\n");
            return false;
        }
        const uint32_t expected_case_mask = (case_count_ == 32u) ? 0xFFFFFFFFu : ((1u << case_count_) - 1u);
        if ((uint32_t)family_visible_count.to_uint() != case_count_ ||
            (uint32_t)family_owner_ok.to_uint() != 1u ||
            (uint32_t)family_consumed_count.to_uint() != case_total_words_ ||
            (uint32_t)family_compare_ok.to_uint() != 1u ||
            (uint32_t)family_case_mask.to_uint() != expected_case_mask) {
            std::printf("[w4b7][FAIL] family observability mismatch visible=%u owner=%u consumed=%u compare=%u mask=0x%X expected_mask=0x%X\n",
                (unsigned)(uint32_t)family_visible_count.to_uint(),
                (unsigned)(uint32_t)family_owner_ok.to_uint(),
                (unsigned)(uint32_t)family_consumed_count.to_uint(),
                (unsigned)(uint32_t)family_compare_ok.to_uint(),
                (unsigned)(uint32_t)family_case_mask.to_uint(),
                (unsigned)expected_case_mask);
            return false;
        }
        if ((uint32_t)single_visible.to_uint() != 1u ||
            (uint32_t)single_owner_ok.to_uint() != 1u ||
            (uint32_t)single_consumed.to_uint() != 1u ||
            (uint32_t)single_compare_ok.to_uint() != 1u) {
            std::printf("[w4b7][FAIL] single observability mismatch visible=%u owner=%u consumed=%u compare=%u\n",
                (unsigned)(uint32_t)single_visible.to_uint(),
                (unsigned)(uint32_t)single_owner_ok.to_uint(),
                (unsigned)(uint32_t)single_consumed.to_uint(),
                (unsigned)(uint32_t)single_compare_ok.to_uint());
            return false;
        }
        std::printf("W4B7_QKSCORE_MIXED_BRIDGE_VISIBLE PASS\n");
        std::printf("W4B7_QKSCORE_MIXED_OWNERSHIP_CHECK PASS\n");
        std::printf("W4B7_QKSCORE_MIXED_MULTI_PATH_ANTI_FALLBACK PASS\n");

        bool fallback_taken_legacy = true;
        const bool score_mainline_taken_legacy = aecct::run_p11ae_layer0_top_managed_qk_score(
            sram_legacy.data(),
            cfg_,
            sc_,
            (aecct::u32_t)token_idx,
            fallback_taken_legacy);
        if (!score_mainline_taken_legacy || fallback_taken_legacy) {
            std::printf("[w4b7][FAIL] legacy baseline run did not stay on mainline\n");
            return false;
        }

        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        for (uint32_t h = 0u; h < n_heads; ++h) {
            const uint32_t head_base = score_base + h * token_count;
            for (uint32_t j = 0u; j < token_count; ++j) {
                const uint32_t idx = h * token_count + j;
                const uint32_t got_bridge = (uint32_t)sram_bridge[head_base + j].to_uint();
                const uint32_t got_legacy = (uint32_t)sram_legacy[head_base + j].to_uint();
                const uint32_t exp = (uint32_t)expected_scores_[idx].to_uint();
                if (got_bridge != exp) {
                    std::printf("[w4b7][FAIL] expected compare mismatch head=%u key=%u got=0x%08X exp=0x%08X\n",
                        (unsigned)h, (unsigned)j, (unsigned)got_bridge, (unsigned)exp);
                    return false;
                }
                if (got_bridge != got_legacy) {
                    std::printf("[w4b7][FAIL] legacy compare mismatch head=%u key=%u bridge=0x%08X legacy=0x%08X\n",
                        (unsigned)h, (unsigned)j, (unsigned)got_bridge, (unsigned)got_legacy);
                    return false;
                }
            }
        }
        std::printf("W4B7_QKSCORE_MIXED_EXPECTED_COMPARE PASS\n");
        std::printf("W4B7_QKSCORE_MIXED_LEGACY_COMPARE PASS\n");

        std::vector<uint8_t> allowed_write((uint32_t)sram_bridge.size(), 0u);
        for (uint32_t h = 0u; h < n_heads; ++h) {
            const uint32_t head_base = score_base + h * token_count;
            for (uint32_t j = 0u; j < token_count; ++j) {
                allowed_write[head_base + j] = 1u;
            }
        }
        for (uint32_t i = 0u; i < (uint32_t)sram_bridge.size(); ++i) {
            const uint32_t before = (uint32_t)sram_bootstrap_[i].to_uint();
            const uint32_t after = (uint32_t)sram_bridge[i].to_uint();
            if (before != after && allowed_write[i] == 0u) {
                std::printf("[w4b7][FAIL] spurious write addr=%u before=0x%08X after=0x%08X\n",
                    (unsigned)i, (unsigned)before, (unsigned)after);
                return false;
            }
        }
        std::printf("W4B7_QKSCORE_MIXED_NO_SPURIOUS_TOUCH PASS\n");
        return true;
    }

    bool run_negative_mixed_mismatch_case() {
        std::vector<aecct::u32_t> sram_negative = sram_bootstrap_;
        const uint32_t token_idx = 0u;
        const uint32_t family_stride = (uint32_t)aecct::ATTN_TOKEN_COUNT;

        std::vector<aecct::u32_t> family_base_words;
        std::vector<aecct::u32_t> family_words_flat;
        std::vector<aecct::u32_t> family_valid_words;
        std::vector<aecct::u32_t> family_key_begin;
        std::vector<aecct::u32_t> family_head_idx;
        build_family_bridge_payload(
            family_base_words,
            family_words_flat,
            family_valid_words,
            family_key_begin,
            family_head_idx);
        family_words_flat[0u * family_stride + 0u] =
            (aecct::u32_t)((uint32_t)family_words_flat[0u * family_stride + 0u].to_uint() ^ 0x00000001u);

        bool fallback_taken = true;
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t single_head = single_head_;
        const uint32_t single_key_begin = single_key_begin_;
        const uint32_t single_valid_words = single_valid_words_;
        const uint32_t single_base_word =
            (uint32_t)sc_.attn.score_base_word.to_uint() + single_head * token_count + single_key_begin;
        const aecct::u32_t single_bridge_word =
            expected_scores_[single_head * token_count + single_key_begin];
        aecct::u32_t single_visible = 0;
        aecct::u32_t single_owner_ok = 0;
        aecct::u32_t single_consumed = 0;
        aecct::u32_t single_compare_ok = 0;
        aecct::u32_t family_visible_count = 0;
        aecct::u32_t family_owner_ok = 0;
        aecct::u32_t family_consumed_count = 0;
        aecct::u32_t family_compare_ok = 0;
        aecct::u32_t family_case_mask = 0;
        const bool score_mainline_taken = aecct::run_p11ae_layer0_top_managed_qk_score(
            sram_negative.data(),
            cfg_,
            sc_,
            (aecct::u32_t)token_idx,
            fallback_taken,
            (aecct::u32_t)0u,
            (aecct::u32_t)0u,
            0,
            0,
            (aecct::u32_t)0u,
            0,
            0,
            0,
            (aecct::u32_t)single_base_word,
            &single_bridge_word,
            (aecct::u32_t)single_valid_words,
            (aecct::u32_t)single_key_begin,
            &single_visible,
            &single_owner_ok,
            &single_consumed,
            &single_compare_ok,
            (aecct::u32_t)single_head,
            (aecct::u32_t)case_count_,
            family_base_words.data(),
            family_words_flat.data(),
            family_valid_words.data(),
            family_key_begin.data(),
            family_head_idx.data(),
            &family_visible_count,
            &family_owner_ok,
            &family_consumed_count,
            &family_compare_ok,
            &family_case_mask);

        if (score_mainline_taken || !fallback_taken) {
            std::printf("[w4b7][FAIL] mismatch mixed bridge did not reject as expected\n");
            return false;
        }
        if ((uint32_t)family_visible_count.to_uint() != 1u ||
            (uint32_t)family_owner_ok.to_uint() != 1u ||
            (uint32_t)family_consumed_count.to_uint() != 0u ||
            (uint32_t)family_compare_ok.to_uint() != 0u ||
            (uint32_t)family_case_mask.to_uint() != 0x1u) {
            std::printf("[w4b7][FAIL] mismatch observability flags mismatch visible=%u owner=%u consumed=%u compare=%u mask=0x%X\n",
                (unsigned)(uint32_t)family_visible_count.to_uint(),
                (unsigned)(uint32_t)family_owner_ok.to_uint(),
                (unsigned)(uint32_t)family_consumed_count.to_uint(),
                (unsigned)(uint32_t)family_compare_ok.to_uint(),
                (unsigned)(uint32_t)family_case_mask.to_uint());
            return false;
        }
        if ((uint32_t)single_visible.to_uint() != 0u ||
            (uint32_t)single_owner_ok.to_uint() != 0u ||
            (uint32_t)single_consumed.to_uint() != 0u ||
            (uint32_t)single_compare_ok.to_uint() != 0u) {
            std::printf("[w4b7][FAIL] mismatch single flags expected zero visible=%u owner=%u consumed=%u compare=%u\n",
                (unsigned)(uint32_t)single_visible.to_uint(),
                (unsigned)(uint32_t)single_owner_ok.to_uint(),
                (unsigned)(uint32_t)single_consumed.to_uint(),
                (unsigned)(uint32_t)single_compare_ok.to_uint());
            return false;
        }
        for (uint32_t i = 0u; i < (uint32_t)sram_negative.size(); ++i) {
            if ((uint32_t)sram_negative[i].to_uint() !=
                (uint32_t)sram_bootstrap_[i].to_uint()) {
                std::printf("[w4b7][FAIL] reject path wrote stale state addr=%u\n", (unsigned)i);
                return false;
            }
        }
        std::printf("W4B7_QKSCORE_MIXED_MISMATCH_REJECT PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbW4b7QkScoreMixedBridge tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__



