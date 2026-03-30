// W4-M3 KV: Phase-A KV phase-entry caller-fed x-row probe validation (local-only).

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

class TbW4m3KvPhaseEntryProbe {
public:
    int run_all() {
        if (!init_and_prepare()) { return 1; }
        if (!run_positive_probe_visibility_case()) { return 1; }
        if (!run_negative_probe_reject_cases()) { return 1; }
        std::printf("PASS: tb_w4m3_kv_phase_entry_probe\n");
        return 0;
    }

private:
    std::vector<aecct::u32_t> sram_init_;
    std::vector<aecct::u32_t> sram_expected_;
    std::vector<aecct::u32_t> sram_before_positive_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::LayerScratch sc_;
    aecct::CfgRegs cfg_;
    aecct::LayerParamBase pb_;

    bool init_and_prepare() {
        sram_init_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        p11aeaf_tb::init_x_rows(sram_init_);
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads_)) {
            std::printf("[w4m3-kv][FAIL] payload preparation failed\n");
            return false;
        }

        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_init_, payloads_, param_base);
        cfg_ = p11aeaf_tb::build_cfg();
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        pb_ = aecct::make_layer_param_base((aecct::u32_t)param_base, (aecct::u32_t)0u);

        sram_expected_ = sram_init_;
        bool fallback_taken = true;
        const bool kv_mainline_taken = aecct::run_p11ac_layer0_top_managed_kv(
            sram_expected_.data(),
            cfg_,
            (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
            sc_,
            pb_,
            fallback_taken);
        if (!kv_mainline_taken || fallback_taken) {
            std::printf("[w4m3-kv][FAIL] baseline P11AC run failed\n");
            return false;
        }
        return true;
    }

    bool run_positive_probe_visibility_case() {
        std::vector<aecct::u32_t> sram_positive = sram_init_;
        sram_before_positive_ = sram_positive;

        uint32_t d_model = (uint32_t)cfg_.d_model.to_uint();
        if (d_model == 0u) {
            d_model = (uint32_t)aecct::ATTN_D_MODEL;
        }
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t tensor_words = token_count * d_model;

        const uint32_t x_probe_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
        const uint32_t probe_words = d_model;
        std::vector<aecct::u32_t> x_probe_words(probe_words, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < probe_words; ++i) {
            x_probe_words[i] = sram_positive[x_probe_base + i];
        }

        bool fallback_taken = true;
        aecct::u32_t probe_visible = 0;
        aecct::u32_t probe_owner_ok = 0;
        aecct::u32_t probe_compare_ok = 0;
        const bool kv_mainline_taken = aecct::run_p11ac_layer0_top_managed_kv(
            sram_positive.data(),
            cfg_,
            (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
            sc_,
            pb_,
            fallback_taken,
            (aecct::u32_t)x_probe_base,
            x_probe_words.data(),
            (aecct::u32_t)probe_words,
            &probe_visible,
            &probe_owner_ok,
            &probe_compare_ok);

        if (!kv_mainline_taken || fallback_taken) {
            std::printf("[w4m3-kv][FAIL] positive probe run did not stay on mainline\n");
            return false;
        }
        if ((uint32_t)probe_visible.to_uint() != 1u) {
            std::printf("[w4m3-kv][FAIL] probe visibility flag mismatch\n");
            return false;
        }
        if ((uint32_t)probe_owner_ok.to_uint() != 1u) {
            std::printf("[w4m3-kv][FAIL] probe ownership flag mismatch\n");
            return false;
        }
        if ((uint32_t)probe_compare_ok.to_uint() != 1u) {
            std::printf("[w4m3-kv][FAIL] probe compare flag mismatch\n");
            return false;
        }
        std::printf("W4M3_KV_CALLER_FED_XROW_VISIBLE PASS\n");
        std::printf("W4M3_KV_OWNERSHIP_CHECK PASS\n");

        const uint32_t k_base = (uint32_t)sc_.attn.k_base_word.to_uint();
        const uint32_t v_base = (uint32_t)sc_.attn.v_base_word.to_uint();
        const uint32_t k_act_q_base = (uint32_t)sc_.attn.k_act_q_base_word.to_uint();
        const uint32_t v_act_q_base = (uint32_t)sc_.attn.v_act_q_base_word.to_uint();
        for (uint32_t i = 0u; i < tensor_words; ++i) {
            const uint32_t got_k = (uint32_t)sram_positive[k_base + i].to_uint();
            const uint32_t exp_k = (uint32_t)sram_expected_[k_base + i].to_uint();
            const uint32_t got_v = (uint32_t)sram_positive[v_base + i].to_uint();
            const uint32_t exp_v = (uint32_t)sram_expected_[v_base + i].to_uint();
            const uint32_t got_k_act = (uint32_t)sram_positive[k_act_q_base + i].to_uint();
            const uint32_t exp_k_act = (uint32_t)sram_expected_[k_act_q_base + i].to_uint();
            const uint32_t got_v_act = (uint32_t)sram_positive[v_act_q_base + i].to_uint();
            const uint32_t exp_v_act = (uint32_t)sram_expected_[v_act_q_base + i].to_uint();
            if (got_k != exp_k || got_v != exp_v || got_k_act != exp_k_act || got_v_act != exp_v_act) {
                std::printf("[w4m3-kv][FAIL] expected compare mismatch idx=%u\n", (unsigned)i);
                return false;
            }
        }
        std::printf("W4M3_KV_EXPECTED_COMPARE PASS\n");

        std::vector<uint8_t> allowed_write((uint32_t)sram_positive.size(), 0u);
        for (uint32_t i = 0u; i < tensor_words; ++i) {
            allowed_write[k_base + i] = 1u;
            allowed_write[v_base + i] = 1u;
            allowed_write[k_act_q_base + i] = 1u;
            allowed_write[v_act_q_base + i] = 1u;
        }
        for (uint32_t i = 0u; i < (uint32_t)sram_positive.size(); ++i) {
            const uint32_t before = (uint32_t)sram_before_positive_[i].to_uint();
            const uint32_t after = (uint32_t)sram_positive[i].to_uint();
            if (before != after && allowed_write[i] == 0u) {
                std::printf("[w4m3-kv][FAIL] spurious write addr=%u before=0x%08X after=0x%08X\n",
                    (unsigned)i, (unsigned)before, (unsigned)after);
                return false;
            }
        }
        std::printf("W4M3_KV_NO_SPURIOUS_TOUCH PASS\n");
        return true;
    }

    bool run_negative_probe_reject_cases() {
        uint32_t d_model = (uint32_t)cfg_.d_model.to_uint();
        if (d_model == 0u) {
            d_model = (uint32_t)aecct::ATTN_D_MODEL;
        }
        const uint32_t x_probe_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
        const uint32_t probe_words = d_model;

        std::vector<aecct::u32_t> x_probe_words(probe_words, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < probe_words; ++i) {
            x_probe_words[i] = sram_init_[x_probe_base + i];
        }

        {
            std::vector<aecct::u32_t> sram_neg_owner = sram_init_;
            bool fallback_taken = true;
            aecct::u32_t probe_visible = 0;
            aecct::u32_t probe_owner_ok = 0;
            aecct::u32_t probe_compare_ok = 0;
            const bool kv_mainline_taken = aecct::run_p11ac_layer0_top_managed_kv(
                sram_neg_owner.data(),
                cfg_,
                (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
                sc_,
                pb_,
                fallback_taken,
                (aecct::u32_t)(x_probe_base + 1u),
                x_probe_words.data(),
                (aecct::u32_t)probe_words,
                &probe_visible,
                &probe_owner_ok,
                &probe_compare_ok);
            if (kv_mainline_taken || !fallback_taken) {
                std::printf("[w4m3-kv][FAIL] owner-mismatch probe did not reject as expected\n");
                return false;
            }
            if ((uint32_t)probe_visible.to_uint() != 1u || (uint32_t)probe_owner_ok.to_uint() != 0u) {
                std::printf("[w4m3-kv][FAIL] owner-mismatch observability flags mismatch\n");
                return false;
            }
            for (uint32_t i = 0u; i < (uint32_t)sram_neg_owner.size(); ++i) {
                if ((uint32_t)sram_neg_owner[i].to_uint() != (uint32_t)sram_init_[i].to_uint()) {
                    std::printf("[w4m3-kv][FAIL] owner-mismatch reject wrote stale state addr=%u\n", (unsigned)i);
                    return false;
                }
            }
        }

        {
            std::vector<aecct::u32_t> sram_neg_compare = sram_init_;
            std::vector<aecct::u32_t> probe_bad = x_probe_words;
            probe_bad[0] = (aecct::u32_t)((uint32_t)probe_bad[0].to_uint() ^ 0x00000001u);
            bool fallback_taken = true;
            aecct::u32_t probe_visible = 0;
            aecct::u32_t probe_owner_ok = 0;
            aecct::u32_t probe_compare_ok = 0;
            const bool kv_mainline_taken = aecct::run_p11ac_layer0_top_managed_kv(
                sram_neg_compare.data(),
                cfg_,
                (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
                sc_,
                pb_,
                fallback_taken,
                (aecct::u32_t)x_probe_base,
                probe_bad.data(),
                (aecct::u32_t)probe_words,
                &probe_visible,
                &probe_owner_ok,
                &probe_compare_ok);
            if (kv_mainline_taken || !fallback_taken) {
                std::printf("[w4m3-kv][FAIL] compare-mismatch probe did not reject as expected\n");
                return false;
            }
            if ((uint32_t)probe_visible.to_uint() != 1u ||
                (uint32_t)probe_owner_ok.to_uint() != 1u ||
                (uint32_t)probe_compare_ok.to_uint() != 0u) {
                std::printf("[w4m3-kv][FAIL] compare-mismatch observability flags mismatch\n");
                return false;
            }
            for (uint32_t i = 0u; i < (uint32_t)sram_neg_compare.size(); ++i) {
                if ((uint32_t)sram_neg_compare[i].to_uint() != (uint32_t)sram_init_[i].to_uint()) {
                    std::printf("[w4m3-kv][FAIL] compare-mismatch reject wrote stale state addr=%u\n", (unsigned)i);
                    return false;
                }
            }
        }

        std::printf("W4M3_KV_PROBE_MISMATCH_REJECT PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbW4m3KvPhaseEntryProbe tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__

