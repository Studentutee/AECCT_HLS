// W4-C1: SoftmaxOut head/token family descriptor contract probe (local-only).

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

class TbW4c1SoftmaxOutHeadTokenContractProbe {
public:
    int run_all() {
        if (!init_and_bootstrap()) { return 1; }
        if (!run_positive_contract_case()) { return 1; }
        if (!run_negative_contract_mismatch_case()) { return 1; }
        std::printf("PASS: tb_w4c1_softmaxout_head_token_contract_probe\n");
        return 0;
    }

private:
    static const uint32_t kMaxFamilyCases = 8u;

    std::vector<aecct::u32_t> sram_bootstrap_;
    std::vector<aecct::u32_t> expected_out_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::LayerScratch sc_;
    aecct::CfgRegs cfg_;
    uint32_t d_tile_count_ = 0u;
    uint32_t single_d_tile_ = 0u;
    uint32_t single_valid_words_ = 0u;
    bool enable_single_bridge_ = false;
    uint32_t family_case_count_ = 0u;
    uint32_t family_d_tile_[kMaxFamilyCases];
    uint32_t family_head_idx_[kMaxFamilyCases];
    uint32_t family_valid_words_[kMaxFamilyCases];
    uint32_t family_total_words_ = 0u;

    bool init_and_bootstrap() {
        sram_bootstrap_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        p11aeaf_tb::init_x_rows(sram_bootstrap_);
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads_)) {
            std::printf("[w4c1][FAIL] payload preparation failed\n");
            return false;
        }

        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_bootstrap_, payloads_, param_base);
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        cfg_ = p11aeaf_tb::build_cfg();

        bool q_fallback_taken = true;
        bool kv_fallback_taken = true;
        if (!p11aeaf_tb::run_ac_ad_mainline(sram_bootstrap_, q_fallback_taken, kv_fallback_taken)) {
            std::printf("[w4c1][FAIL] AC/AD bootstrap failed\n");
            return false;
        }
        if (q_fallback_taken || kv_fallback_taken) {
            std::printf("[w4c1][FAIL] AC/AD bootstrap fallback detected\n");
            return false;
        }

        bool score_fallback_taken = true;
        const bool score_mainline_taken = aecct::run_p11ae_layer0_top_managed_qk_score(
            sram_bootstrap_.data(),
            cfg_,
            sc_,
            (aecct::u32_t)0u,
            score_fallback_taken);
        if (!score_mainline_taken || score_fallback_taken) {
            std::printf("[w4c1][FAIL] AE bootstrap failed\n");
            return false;
        }

        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t tile_words = (uint32_t)aecct::ATTN_TOP_MANAGED_WORK_TILE_WORDS;
        d_tile_count_ = aecct::attn_top_managed_tile_count(d_head, tile_words);

        enable_single_bridge_ = (d_tile_count_ > 1u);
        const uint32_t candidate_family_cases =
            enable_single_bridge_ ? (d_tile_count_ - 1u) : 1u;
        if (candidate_family_cases > kMaxFamilyCases) {
            std::printf("[w4c1][FAIL] family case count overflow count=%u max=%u\n",
                (unsigned)candidate_family_cases, (unsigned)kMaxFamilyCases);
            return false;
        }

        single_d_tile_ = enable_single_bridge_ ? (d_tile_count_ - 1u) : 0u;
        single_valid_words_ = aecct::attn_top_managed_tile_valid_words(d_head, tile_words, single_d_tile_);
        if (single_valid_words_ == 0u || single_valid_words_ > tile_words) {
            std::printf("[w4c1][FAIL] invalid single tile valid words\n");
            return false;
        }

        family_case_count_ = candidate_family_cases;
        family_total_words_ = 0u;
        ATTN_W4C1_INIT_FAMILY_LOOP: for (uint32_t c = 0u; c < family_case_count_; ++c) {
            family_d_tile_[c] = enable_single_bridge_ ? c : 0u;
            family_head_idx_[c] = (n_heads > 1u && c == 0u) ? 1u : 0u;
            family_valid_words_[c] =
                aecct::attn_top_managed_tile_valid_words(d_head, tile_words, family_d_tile_[c]);
            if (family_valid_words_[c] == 0u || family_valid_words_[c] > tile_words) {
                std::printf("[w4c1][FAIL] invalid family tile valid words case=%u\n", (unsigned)c);
                return false;
            }
            family_total_words_ += family_valid_words_[c];
        }
        return true;
    }

    void build_head_token_family_payload(
        std::vector<aecct::u32_t>& single_words,
        aecct::u32_t& single_base_word,
        std::vector<aecct::u32_t>& family_base_words,
        std::vector<aecct::u32_t>& family_words_flat,
        std::vector<aecct::u32_t>& family_valid_words,
        std::vector<aecct::u32_t>& family_d_tile_idx,
        std::vector<aecct::u32_t>& family_head_idx,
        std::vector<aecct::u32_t>& family_key_token_begin,
        std::vector<aecct::u32_t>& family_key_token_count) const {
        const uint32_t v_base = (uint32_t)sc_.attn.v_base_word.to_uint();
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        const uint32_t tile_words = (uint32_t)aecct::ATTN_TOP_MANAGED_WORK_TILE_WORDS;

        const uint32_t single_tile_offset = single_d_tile_ * tile_words;
        const uint32_t single_base = v_base + single_tile_offset;
        single_base_word = (aecct::u32_t)single_base;
        single_words.assign(single_valid_words_, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < single_valid_words_; ++i) {
            single_words[i] = sram_bootstrap_[single_base + i];
        }

        family_base_words.assign(family_case_count_, (aecct::u32_t)0u);
        family_valid_words.assign(family_case_count_, (aecct::u32_t)0u);
        family_d_tile_idx.assign(family_case_count_, (aecct::u32_t)0u);
        family_head_idx.assign(family_case_count_, (aecct::u32_t)0u);
        family_key_token_begin.assign(family_case_count_, (aecct::u32_t)0u);
        family_key_token_count.assign(family_case_count_, (aecct::u32_t)0u);
        family_words_flat.assign(family_case_count_ * tile_words, (aecct::u32_t)0u);

        ATTN_W4C1_BUILD_FAMILY_LOOP: for (uint32_t c = 0u; c < family_case_count_; ++c) {
            const uint32_t dt = family_d_tile_[c];
            const uint32_t h = family_head_idx_[c];
            const uint32_t valid = family_valid_words_[c];
            const uint32_t tile_offset = dt * tile_words;
            const uint32_t base = v_base + h * d_head + tile_offset;
            family_base_words[c] = (aecct::u32_t)base;
            family_valid_words[c] = (aecct::u32_t)valid;
            family_d_tile_idx[c] = (aecct::u32_t)dt;
            family_head_idx[c] = (aecct::u32_t)h;
            family_key_token_begin[c] = (aecct::u32_t)0u;
            family_key_token_count[c] = (aecct::u32_t)1u;
            ATTN_W4C1_BUILD_FAMILY_COL_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                family_words_flat[c * tile_words + i] = sram_bootstrap_[base + i];
            }
        }
    }

    bool run_positive_contract_case() {
        std::vector<aecct::u32_t> sram_bridge = sram_bootstrap_;
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

        std::vector<aecct::u32_t> single_words;
        aecct::u32_t single_base_word = (aecct::u32_t)0u;
        std::vector<aecct::u32_t> family_base_words;
        std::vector<aecct::u32_t> family_words_flat;
        std::vector<aecct::u32_t> family_valid_words;
        std::vector<aecct::u32_t> family_d_tile_idx;
        std::vector<aecct::u32_t> family_head_idx;
        std::vector<aecct::u32_t> family_key_token_begin;
        std::vector<aecct::u32_t> family_key_token_count;
        build_head_token_family_payload(
            single_words,
            single_base_word,
            family_base_words,
            family_words_flat,
            family_valid_words,
            family_d_tile_idx,
            family_head_idx,
            family_key_token_begin,
            family_key_token_count);

        bool fallback_taken_bridge = true;
        aecct::u32_t single_visible = 0;
        aecct::u32_t single_owner_ok = 0;
        aecct::u32_t single_consumed = 0;
        aecct::u32_t single_compare_ok = 0;
        aecct::u32_t family_visible_count = 0;
        aecct::u32_t family_owner_ok = 0;
        aecct::u32_t family_consumed_count = 0;
        aecct::u32_t family_compare_ok = 0;
        aecct::u32_t family_case_mask = 0;
        aecct::u32_t family_desc_visible_count = 0;
        aecct::u32_t family_desc_case_mask = 0;
        const aecct::u32_t single_base_arg =
            enable_single_bridge_ ? single_base_word : (aecct::u32_t)0u;
        const aecct::u32_t* single_words_arg =
            enable_single_bridge_ ? single_words.data() : 0;
        const aecct::u32_t single_valid_arg =
            enable_single_bridge_ ? (aecct::u32_t)single_valid_words_ : (aecct::u32_t)0u;
        const aecct::u32_t single_d_tile_arg =
            enable_single_bridge_ ? (aecct::u32_t)single_d_tile_ : (aecct::u32_t)0u;
        const bool softmax_mainline_taken_bridge =
            aecct::run_p11af_layer0_top_managed_softmax_out(
                sram_bridge.data(),
                cfg_,
                sc_,
                (aecct::u32_t)token_idx,
                fallback_taken_bridge,
                (aecct::u32_t)0u,
                0,
                (aecct::u32_t)0u,
                0,
                0,
                0,
                single_base_arg,
                single_words_arg,
                single_valid_arg,
                single_d_tile_arg,
                &single_visible,
                &single_owner_ok,
                &single_consumed,
                &single_compare_ok,
                (aecct::u32_t)family_case_count_,
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
                &family_desc_case_mask);

        if (!softmax_mainline_taken_bridge || fallback_taken_bridge) {
            std::printf("[w4c1][FAIL] positive head/token family bridge run did not stay on mainline\n");
            return false;
        }

        const uint32_t expected_case_mask =
            (family_case_count_ == 32u) ? 0xFFFFFFFFu : ((1u << family_case_count_) - 1u);
        if ((uint32_t)family_visible_count.to_uint() != family_case_count_ ||
            (uint32_t)family_owner_ok.to_uint() != 1u ||
            (uint32_t)family_compare_ok.to_uint() != 1u ||
            (uint32_t)family_consumed_count.to_uint() != family_total_words_ ||
            (uint32_t)family_case_mask.to_uint() != expected_case_mask) {
            std::printf("[w4c1][FAIL] family observability mismatch visible=%u owner=%u consumed=%u compare=%u mask=0x%X expected=0x%X\n",
                (unsigned)(uint32_t)family_visible_count.to_uint(),
                (unsigned)(uint32_t)family_owner_ok.to_uint(),
                (unsigned)(uint32_t)family_consumed_count.to_uint(),
                (unsigned)(uint32_t)family_compare_ok.to_uint(),
                (unsigned)(uint32_t)family_case_mask.to_uint(),
                (unsigned)expected_case_mask);
            return false;
        }
        if ((uint32_t)family_desc_visible_count.to_uint() != family_case_count_ ||
            (uint32_t)family_desc_case_mask.to_uint() != expected_case_mask) {
            std::printf("[w4c1][FAIL] descriptor visibility mismatch visible=%u mask=0x%X expected_mask=0x%X\n",
                (unsigned)(uint32_t)family_desc_visible_count.to_uint(),
                (unsigned)(uint32_t)family_desc_case_mask.to_uint(),
                (unsigned)expected_case_mask);
            return false;
        }
        if (enable_single_bridge_) {
            if ((uint32_t)single_visible.to_uint() != 1u ||
                (uint32_t)single_owner_ok.to_uint() != 1u ||
                (uint32_t)single_consumed.to_uint() != 1u ||
                (uint32_t)single_compare_ok.to_uint() != 1u) {
                std::printf("[w4c1][FAIL] single observability mismatch visible=%u owner=%u consumed=%u compare=%u\n",
                    (unsigned)(uint32_t)single_visible.to_uint(),
                    (unsigned)(uint32_t)single_owner_ok.to_uint(),
                    (unsigned)(uint32_t)single_consumed.to_uint(),
                    (unsigned)(uint32_t)single_compare_ok.to_uint());
                return false;
            }
        } else {
            if ((uint32_t)single_visible.to_uint() != 0u ||
                (uint32_t)single_owner_ok.to_uint() != 0u ||
                (uint32_t)single_consumed.to_uint() != 0u ||
                (uint32_t)single_compare_ok.to_uint() != 0u) {
                std::printf("[w4c1][FAIL] single observability must stay zero in family-only mode visible=%u owner=%u consumed=%u compare=%u\n",
                    (unsigned)(uint32_t)single_visible.to_uint(),
                    (unsigned)(uint32_t)single_owner_ok.to_uint(),
                    (unsigned)(uint32_t)single_consumed.to_uint(),
                    (unsigned)(uint32_t)single_compare_ok.to_uint());
                return false;
            }
        }
        std::printf("W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_BRIDGE_VISIBLE PASS\n");
        std::printf("W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_OWNERSHIP_CHECK PASS\n");
        std::printf("W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_DESC_VISIBLE PASS\n");
        std::printf("W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_CONSUME_COUNT_EXACT PASS\n");
        std::printf("W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_ANTI_FALLBACK PASS\n");

        bool fallback_taken_legacy = true;
        const bool softmax_mainline_taken_legacy =
            aecct::run_p11af_layer0_top_managed_softmax_out(
                sram_legacy.data(),
                cfg_,
                sc_,
                (aecct::u32_t)token_idx,
                fallback_taken_legacy);
        if (!softmax_mainline_taken_legacy || fallback_taken_legacy) {
            std::printf("[w4c1][FAIL] legacy run did not stay on mainline\n");
            return false;
        }

        const uint32_t out_row_base = (uint32_t)sc_.attn_out_base_word.to_uint() + token_idx * d_model;
        const uint32_t pre_row_base = (uint32_t)sc_.attn.pre_concat_base_word.to_uint() + token_idx * d_model;
        const uint32_t post_row_base = (uint32_t)sc_.attn.post_concat_base_word.to_uint() + token_idx * d_model;
        ATTN_W4C1_EXPECTED_COMPARE_LOOP: for (uint32_t i = 0u; i < d_model; ++i) {
            const uint32_t exp = (uint32_t)expected_out_[i].to_uint();
            const uint32_t got_pre = (uint32_t)sram_bridge[pre_row_base + i].to_uint();
            const uint32_t got_post = (uint32_t)sram_bridge[post_row_base + i].to_uint();
            const uint32_t got_out = (uint32_t)sram_bridge[out_row_base + i].to_uint();
            const uint32_t legacy_pre = (uint32_t)sram_legacy[pre_row_base + i].to_uint();
            const uint32_t legacy_post = (uint32_t)sram_legacy[post_row_base + i].to_uint();
            const uint32_t legacy_out = (uint32_t)sram_legacy[out_row_base + i].to_uint();
            if (got_pre != exp || got_post != exp || got_out != exp) {
                std::printf("[w4c1][FAIL] expected compare mismatch idx=%u pre=0x%08X post=0x%08X out=0x%08X exp=0x%08X\n",
                    (unsigned)i,
                    (unsigned)got_pre,
                    (unsigned)got_post,
                    (unsigned)got_out,
                    (unsigned)exp);
                return false;
            }
            if (got_pre != legacy_pre || got_post != legacy_post || got_out != legacy_out) {
                std::printf("[w4c1][FAIL] legacy compare mismatch idx=%u bridge(pre/post/out)=(0x%08X/0x%08X/0x%08X) legacy=(0x%08X/0x%08X/0x%08X)\n",
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
        std::printf("W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_EXPECTED_COMPARE PASS\n");
        std::printf("W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_LEGACY_COMPARE PASS\n");

        std::vector<uint8_t> allowed_write((uint32_t)sram_bridge.size(), 0u);
        for (uint32_t i = 0u; i < d_model; ++i) {
            allowed_write[pre_row_base + i] = 1u;
            allowed_write[post_row_base + i] = 1u;
            allowed_write[out_row_base + i] = 1u;
        }
        ATTN_W4C1_SPURIOUS_CHECK_LOOP: for (uint32_t i = 0u; i < (uint32_t)sram_bridge.size(); ++i) {
            const uint32_t before = (uint32_t)sram_bootstrap_[i].to_uint();
            const uint32_t after = (uint32_t)sram_bridge[i].to_uint();
            if (before != after && allowed_write[i] == 0u) {
                std::printf("[w4c1][FAIL] spurious write addr=%u before=0x%08X after=0x%08X\n",
                    (unsigned)i,
                    (unsigned)before,
                    (unsigned)after);
                return false;
            }
        }
        std::printf("W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_NO_SPURIOUS_TOUCH PASS\n");
        return true;
    }

    bool run_negative_contract_mismatch_case() {
        std::vector<aecct::u32_t> sram_negative = sram_bootstrap_;
        const uint32_t token_idx = 0u;

        std::vector<aecct::u32_t> single_words;
        aecct::u32_t single_base_word = (aecct::u32_t)0u;
        std::vector<aecct::u32_t> family_base_words;
        std::vector<aecct::u32_t> family_words_flat;
        std::vector<aecct::u32_t> family_valid_words;
        std::vector<aecct::u32_t> family_d_tile_idx;
        std::vector<aecct::u32_t> family_head_idx;
        std::vector<aecct::u32_t> family_key_token_begin;
        std::vector<aecct::u32_t> family_key_token_count;
        build_head_token_family_payload(
            single_words,
            single_base_word,
            family_base_words,
            family_words_flat,
            family_valid_words,
            family_d_tile_idx,
            family_head_idx,
            family_key_token_begin,
            family_key_token_count);

        // W4-C1 bounded descriptor guard: token_begin must stay 0 in current INIT_ACC-only scope.
        // Force an invalid descriptor to trigger early reject before any head writeback executes.
        family_key_token_begin[0] = (aecct::u32_t)1u;

        bool fallback_taken = true;
        aecct::u32_t single_visible = 0;
        aecct::u32_t single_owner_ok = 0;
        aecct::u32_t single_consumed = 0;
        aecct::u32_t single_compare_ok = 0;
        aecct::u32_t family_visible_count = 0;
        aecct::u32_t family_owner_ok = 0;
        aecct::u32_t family_consumed_count = 0;
        aecct::u32_t family_compare_ok = 0;
        aecct::u32_t family_case_mask = 0;
        aecct::u32_t family_desc_visible_count = 0;
        aecct::u32_t family_desc_case_mask = 0;
        const aecct::u32_t single_base_arg =
            enable_single_bridge_ ? single_base_word : (aecct::u32_t)0u;
        const aecct::u32_t* single_words_arg =
            enable_single_bridge_ ? single_words.data() : 0;
        const aecct::u32_t single_valid_arg =
            enable_single_bridge_ ? (aecct::u32_t)single_valid_words_ : (aecct::u32_t)0u;
        const aecct::u32_t single_d_tile_arg =
            enable_single_bridge_ ? (aecct::u32_t)single_d_tile_ : (aecct::u32_t)0u;
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
                single_base_arg,
                single_words_arg,
                single_valid_arg,
                single_d_tile_arg,
                &single_visible,
                &single_owner_ok,
                &single_consumed,
                &single_compare_ok,
                (aecct::u32_t)family_case_count_,
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
                &family_desc_case_mask);

        if (softmax_mainline_taken || !fallback_taken) {
            std::printf("[w4c1][FAIL] mismatch head/token descriptor did not reject as expected\n");
            return false;
        }

        if ((uint32_t)family_visible_count.to_uint() != 0u ||
            (uint32_t)family_consumed_count.to_uint() != 0u ||
            (uint32_t)family_case_mask.to_uint() != 0u ||
            (uint32_t)family_desc_visible_count.to_uint() != 0u ||
            (uint32_t)family_desc_case_mask.to_uint() != 0u) {
            std::printf("[w4c1][FAIL] precheck reject should keep selector observability at zero visible=%u consumed=%u mask=0x%X desc_visible=%u desc_mask=0x%X\n",
                (unsigned)(uint32_t)family_visible_count.to_uint(),
                (unsigned)(uint32_t)family_consumed_count.to_uint(),
                (unsigned)(uint32_t)family_case_mask.to_uint(),
                (unsigned)(uint32_t)family_desc_visible_count.to_uint(),
                (unsigned)(uint32_t)family_desc_case_mask.to_uint());
            return false;
        }

        ATTN_W4C1_REJECT_STALE_CHECK_LOOP: for (uint32_t i = 0u; i < (uint32_t)sram_negative.size(); ++i) {
            if ((uint32_t)sram_negative[i].to_uint() != (uint32_t)sram_bootstrap_[i].to_uint()) {
                std::printf("[w4c1][FAIL] reject path wrote stale state addr=%u\n", (unsigned)i);
                return false;
            }
        }
        std::printf("W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_MISMATCH_REJECT PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbW4c1SoftmaxOutHeadTokenContractProbe tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__
