// W4-M2: SoftmaxOut phase-entry caller-fed V-tile probe validation (local-only).

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

class TbW4m2SoftmaxOutPhaseEntryProbe {
public:
    int run_all() {
        if (!init_and_bootstrap()) { return 1; }
        if (!run_positive_probe_visibility_case()) { return 1; }
        if (!run_negative_probe_mismatch_case()) { return 1; }
        std::printf("PASS: tb_w4m2_softmaxout_phase_entry_probe\n");
        return 0;
    }

private:
    std::vector<aecct::u32_t> sram_;
    std::vector<aecct::u32_t> sram_before_positive_;
    std::vector<aecct::u32_t> expected_out_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::LayerScratch sc_;
    aecct::CfgRegs cfg_;

    bool init_and_bootstrap() {
        sram_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        p11aeaf_tb::init_x_rows(sram_);
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads_)) {
            std::printf("[w4m2][FAIL] payload preparation failed\n");
            return false;
        }

        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_, payloads_, param_base);
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        cfg_ = p11aeaf_tb::build_cfg();

        bool q_fallback_taken = true;
        bool kv_fallback_taken = true;
        if (!p11aeaf_tb::run_ac_ad_mainline(sram_, q_fallback_taken, kv_fallback_taken)) {
            std::printf("[w4m2][FAIL] AC/AD bootstrap failed\n");
            return false;
        }
        if (q_fallback_taken || kv_fallback_taken) {
            std::printf("[w4m2][FAIL] AC/AD bootstrap fallback detected\n");
            return false;
        }

        bool score_fallback_taken = true;
        const bool score_mainline_taken = aecct::run_p11ae_layer0_top_managed_qk_score(
            sram_.data(),
            cfg_,
            sc_,
            (aecct::u32_t)0u,
            score_fallback_taken);
        if (!score_mainline_taken || score_fallback_taken) {
            std::printf("[w4m2][FAIL] AE bootstrap failed\n");
            return false;
        }
        return true;
    }

    bool run_positive_probe_visibility_case() {
        const uint32_t token_idx = 0u;
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;

        p11aeaf_tb::compute_expected_output_row_online(
            sram_,
            sc_.attn,
            token_idx,
            token_count,
            n_heads,
            d_head,
            expected_out_);

        sram_before_positive_ = sram_;

        const uint32_t v_probe_base =
            (uint32_t)sc_.attn.v_base_word.to_uint();
        const uint32_t probe_words = d_head;

        std::vector<aecct::u32_t> v_probe_words(probe_words, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < probe_words; ++i) {
            v_probe_words[i] = sram_[v_probe_base + i];
        }

        bool fallback_taken = true;
        aecct::u32_t probe_visible = 0;
        aecct::u32_t probe_owner_ok = 0;
        aecct::u32_t probe_compare_ok = 0;
        const bool softmax_mainline_taken = aecct::run_p11af_layer0_top_managed_softmax_out(
            sram_.data(),
            cfg_,
            sc_,
            (aecct::u32_t)token_idx,
            fallback_taken,
            (aecct::u32_t)v_probe_base,
            v_probe_words.data(),
            (aecct::u32_t)probe_words,
            &probe_visible,
            &probe_owner_ok,
            &probe_compare_ok);

        if (!softmax_mainline_taken || fallback_taken) {
            std::printf("[w4m2][FAIL] positive probe run did not stay on mainline\n");
            return false;
        }
        if ((uint32_t)probe_visible.to_uint() != 1u) {
            std::printf("[w4m2][FAIL] probe visibility flag mismatch\n");
            return false;
        }
        if ((uint32_t)probe_owner_ok.to_uint() != 1u) {
            std::printf("[w4m2][FAIL] probe ownership flag mismatch\n");
            return false;
        }
        if ((uint32_t)probe_compare_ok.to_uint() != 1u) {
            std::printf("[w4m2][FAIL] probe compare flag mismatch\n");
            return false;
        }
        std::printf("W4M2_SOFTMAXOUT_CALLER_FED_VTILE_VISIBLE PASS\n");
        std::printf("W4M2_SOFTMAXOUT_OWNERSHIP_CHECK PASS\n");

        const uint32_t out_row_base = (uint32_t)sc_.attn_out_base_word.to_uint() + token_idx * d_model;
        const uint32_t pre_row_base = (uint32_t)sc_.attn.pre_concat_base_word.to_uint() + token_idx * d_model;
        const uint32_t post_row_base = (uint32_t)sc_.attn.post_concat_base_word.to_uint() + token_idx * d_model;
        for (uint32_t i = 0u; i < d_model; ++i) {
            const uint32_t exp = (uint32_t)expected_out_[i].to_uint();
            const uint32_t got_pre = (uint32_t)sram_[pre_row_base + i].to_uint();
            const uint32_t got_post = (uint32_t)sram_[post_row_base + i].to_uint();
            const uint32_t got_out = (uint32_t)sram_[out_row_base + i].to_uint();
            if (got_pre != exp || got_post != exp || got_out != exp) {
                std::printf("[w4m2][FAIL] expected compare mismatch idx=%u pre=0x%08X post=0x%08X out=0x%08X exp=0x%08X\n",
                    (unsigned)i, (unsigned)got_pre, (unsigned)got_post, (unsigned)got_out, (unsigned)exp);
                return false;
            }
        }
        std::printf("W4M2_SOFTMAXOUT_EXPECTED_COMPARE PASS\n");

        std::vector<uint8_t> allowed_write((uint32_t)sram_.size(), 0u);
        for (uint32_t i = 0u; i < d_model; ++i) {
            allowed_write[pre_row_base + i] = 1u;
            allowed_write[post_row_base + i] = 1u;
            allowed_write[out_row_base + i] = 1u;
        }
        for (uint32_t i = 0u; i < (uint32_t)sram_.size(); ++i) {
            const uint32_t before = (uint32_t)sram_before_positive_[i].to_uint();
            const uint32_t after = (uint32_t)sram_[i].to_uint();
            if (before != after && allowed_write[i] == 0u) {
                std::printf("[w4m2][FAIL] spurious write addr=%u before=0x%08X after=0x%08X\n",
                    (unsigned)i, (unsigned)before, (unsigned)after);
                return false;
            }
        }
        std::printf("W4M2_SOFTMAXOUT_NO_SPURIOUS_TOUCH PASS\n");
        return true;
    }

    bool run_negative_probe_mismatch_case() {
        std::vector<aecct::u32_t> sram_negative = sram_before_positive_;
        const uint32_t token_idx = 0u;
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        const uint32_t v_probe_base =
            (uint32_t)sc_.attn.v_base_word.to_uint();
        const uint32_t probe_words = d_head;

        std::vector<aecct::u32_t> v_probe_words(probe_words, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < probe_words; ++i) {
            v_probe_words[i] = sram_negative[v_probe_base + i];
        }
        v_probe_words[0] = (aecct::u32_t)((uint32_t)v_probe_words[0].to_uint() ^ 0x00000001u);

        bool fallback_taken = true;
        aecct::u32_t probe_visible = 0;
        aecct::u32_t probe_owner_ok = 0;
        aecct::u32_t probe_compare_ok = 0;
        const bool softmax_mainline_taken = aecct::run_p11af_layer0_top_managed_softmax_out(
            sram_negative.data(),
            cfg_,
            sc_,
            (aecct::u32_t)token_idx,
            fallback_taken,
            (aecct::u32_t)v_probe_base,
            v_probe_words.data(),
            (aecct::u32_t)probe_words,
            &probe_visible,
            &probe_owner_ok,
            &probe_compare_ok);

        if (softmax_mainline_taken || !fallback_taken) {
            std::printf("[w4m2][FAIL] mismatch probe did not reject as expected\n");
            return false;
        }
        if ((uint32_t)probe_visible.to_uint() != 1u ||
            (uint32_t)probe_owner_ok.to_uint() != 1u ||
            (uint32_t)probe_compare_ok.to_uint() != 0u) {
            std::printf("[w4m2][FAIL] mismatch observability flags mismatch\n");
            return false;
        }
        for (uint32_t i = 0u; i < (uint32_t)sram_negative.size(); ++i) {
            if ((uint32_t)sram_negative[i].to_uint() != (uint32_t)sram_before_positive_[i].to_uint()) {
                std::printf("[w4m2][FAIL] reject path wrote stale state addr=%u\n", (unsigned)i);
                return false;
            }
        }
        std::printf("W4M2_SOFTMAXOUT_PROBE_MISMATCH_REJECT PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbW4m2SoftmaxOutPhaseEntryProbe tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__

