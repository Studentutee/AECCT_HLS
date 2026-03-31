// W4-B3: QkScore selectable-head bounded score-tile bridge validation (local-only).

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

class TbW4b3QkScoreBridge {
public:
    int run_all() {
        if (!init_and_bootstrap()) { return 1; }
        if (!run_positive_selectable_head_case()) { return 1; }
        if (!run_negative_bridge_mismatch_case()) { return 1; }
        std::printf("PASS: tb_w4b3_qkscore_bridge\n");
        return 0;
    }

private:
    std::vector<aecct::u32_t> sram_;
    std::vector<aecct::u32_t> sram_before_positive_;
    std::vector<aecct::u32_t> expected_scores_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::LayerScratch sc_;
    aecct::CfgRegs cfg_;
    uint32_t bridge_key_begin_ = 0u;
    uint32_t bridge_words_valid_ = 0u;
    uint32_t bridge_head_idx_ = 1u;

    bool init_and_bootstrap() {
        sram_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        p11aeaf_tb::init_x_rows(sram_);
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads_)) {
            std::printf("[w4b3][FAIL] payload preparation failed\n");
            return false;
        }

        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_, payloads_, param_base);
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        cfg_ = p11aeaf_tb::build_cfg();

        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        if (n_heads < 2u) {
            std::printf("[w4b3][FAIL] selectable-head validation requires n_heads >= 2\n");
            return false;
        }
        bridge_head_idx_ = 1u;

        bool q_fallback_taken = true;
        bool kv_fallback_taken = true;
        if (!p11aeaf_tb::run_ac_ad_mainline(sram_, q_fallback_taken, kv_fallback_taken)) {
            std::printf("[w4b3][FAIL] AC/AD bootstrap failed\n");
            return false;
        }
        if (q_fallback_taken || kv_fallback_taken) {
            std::printf("[w4b3][FAIL] AC/AD bootstrap fallback detected\n");
            return false;
        }

        bridge_key_begin_ = 1u; // secondary-range (not first key slot)
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t tile_words = (uint32_t)aecct::ATTN_TOP_MANAGED_WORK_TILE_WORDS;
        const uint32_t remaining = (bridge_key_begin_ < token_count) ? (token_count - bridge_key_begin_) : 0u;
        bridge_words_valid_ = remaining < tile_words ? remaining : tile_words;
        if (bridge_words_valid_ == 0u) {
            std::printf("[w4b3][FAIL] invalid bridge_words_valid\n");
            return false;
        }
        return true;
    }

    bool run_positive_selectable_head_case() {
        const uint32_t token_idx = 0u;
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;

        p11aeaf_tb::compute_expected_score_row(
            sram_,
            sc_.attn,
            token_idx,
            token_count,
            n_heads,
            d_head,
            expected_scores_);

        sram_before_positive_ = sram_;

        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        const uint32_t bridge_base = score_base + bridge_head_idx_ * token_count + bridge_key_begin_;
        std::vector<aecct::u32_t> bridge_words(bridge_words_valid_, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < bridge_words_valid_; ++i) {
            const uint32_t idx = bridge_head_idx_ * token_count + bridge_key_begin_ + i;
            bridge_words[i] = expected_scores_[idx];
        }

        bool fallback_taken = true;
        aecct::u32_t bridge_visible = 0;
        aecct::u32_t bridge_owner_ok = 0;
        aecct::u32_t bridge_consumed = 0;
        aecct::u32_t bridge_compare_ok = 0;
        const bool score_mainline_taken = aecct::run_p11ae_layer0_top_managed_qk_score(
            sram_.data(),
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
            (aecct::u32_t)bridge_base,
            bridge_words.data(),
            (aecct::u32_t)bridge_words_valid_,
            (aecct::u32_t)bridge_key_begin_,
            &bridge_visible,
            &bridge_owner_ok,
            &bridge_consumed,
            &bridge_compare_ok,
            (aecct::u32_t)bridge_head_idx_);

        if (!score_mainline_taken || fallback_taken) {
            std::printf("[w4b3][FAIL] positive bridge run did not stay on mainline\n");
            return false;
        }
        if ((uint32_t)bridge_visible.to_uint() != 1u ||
            (uint32_t)bridge_owner_ok.to_uint() != 1u ||
            (uint32_t)bridge_consumed.to_uint() != 1u ||
            (uint32_t)bridge_compare_ok.to_uint() != 1u) {
            std::printf("[w4b3][FAIL] bridge observability flags mismatch\n");
            return false;
        }
        std::printf("W4B3_QKSCORE_BRIDGE_VISIBLE PASS\n");
        std::printf("W4B3_QKSCORE_OWNERSHIP_CHECK PASS\n");
        std::printf("W4B3_QKSCORE_NON_HEAD1_PATH PASS\n");

        for (uint32_t h = 0u; h < n_heads; ++h) {
            const uint32_t head_base = score_base + h * token_count;
            for (uint32_t j = 0u; j < token_count; ++j) {
                const uint32_t got = (uint32_t)sram_[head_base + j].to_uint();
                const uint32_t exp = (uint32_t)expected_scores_[h * token_count + j].to_uint();
                if (got != exp) {
                    std::printf("[w4b3][FAIL] expected compare mismatch head=%u key=%u got=0x%08X exp=0x%08X\n",
                        (unsigned)h, (unsigned)j, (unsigned)got, (unsigned)exp);
                    return false;
                }
            }
        }
        std::printf("W4B3_QKSCORE_EXPECTED_COMPARE PASS\n");

        std::vector<uint8_t> allowed_write((uint32_t)sram_.size(), 0u);
        for (uint32_t h = 0u; h < n_heads; ++h) {
            const uint32_t head_base = score_base + h * token_count;
            for (uint32_t j = 0u; j < token_count; ++j) {
                allowed_write[head_base + j] = 1u;
            }
        }
        for (uint32_t i = 0u; i < (uint32_t)sram_.size(); ++i) {
            const uint32_t before = (uint32_t)sram_before_positive_[i].to_uint();
            const uint32_t after = (uint32_t)sram_[i].to_uint();
            if (before != after && allowed_write[i] == 0u) {
                std::printf("[w4b3][FAIL] spurious write addr=%u before=0x%08X after=0x%08X\n",
                    (unsigned)i, (unsigned)before, (unsigned)after);
                return false;
            }
        }
        std::printf("W4B3_QKSCORE_NO_SPURIOUS_TOUCH PASS\n");
        return true;
    }

    bool run_negative_bridge_mismatch_case() {
        std::vector<aecct::u32_t> sram_negative = sram_before_positive_;
        const uint32_t token_idx = 0u;
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        const uint32_t bridge_base = score_base + bridge_head_idx_ * token_count + bridge_key_begin_;

        std::vector<aecct::u32_t> bridge_words(bridge_words_valid_, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < bridge_words_valid_; ++i) {
            const uint32_t idx = bridge_head_idx_ * token_count + bridge_key_begin_ + i;
            bridge_words[i] = expected_scores_[idx];
        }
        bridge_words[0] = (aecct::u32_t)((uint32_t)bridge_words[0].to_uint() ^ 0x00000001u);

        bool fallback_taken = true;
        aecct::u32_t bridge_visible = 0;
        aecct::u32_t bridge_owner_ok = 0;
        aecct::u32_t bridge_consumed = 0;
        aecct::u32_t bridge_compare_ok = 0;
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
            (aecct::u32_t)bridge_base,
            bridge_words.data(),
            (aecct::u32_t)bridge_words_valid_,
            (aecct::u32_t)bridge_key_begin_,
            &bridge_visible,
            &bridge_owner_ok,
            &bridge_consumed,
            &bridge_compare_ok,
            (aecct::u32_t)bridge_head_idx_);

        if (score_mainline_taken || !fallback_taken) {
            std::printf("[w4b3][FAIL] mismatch bridge did not reject as expected\n");
            return false;
        }
        if ((uint32_t)bridge_visible.to_uint() != 1u ||
            (uint32_t)bridge_owner_ok.to_uint() != 1u ||
            (uint32_t)bridge_consumed.to_uint() != 0u ||
            (uint32_t)bridge_compare_ok.to_uint() != 0u) {
            std::printf("[w4b3][FAIL] mismatch observability flags mismatch\n");
            return false;
        }
        const uint32_t bridge_head_base = score_base + bridge_head_idx_ * token_count;
        for (uint32_t i = 0u; i < bridge_words_valid_; ++i) {
            const uint32_t addr = bridge_head_base + bridge_key_begin_ + i;
            if ((uint32_t)sram_negative[addr].to_uint() != (uint32_t)sram_before_positive_[addr].to_uint()) {
                std::printf("[w4b3][FAIL] reject path polluted bridge window addr=%u\n", (unsigned)addr);
                return false;
            }
        }
        std::printf("W4B3_QKSCORE_BRIDGE_MISMATCH_REJECT PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbW4b3QkScoreBridge tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__
