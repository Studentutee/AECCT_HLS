// P00-011AG: attention-chain real-mainline correction validator (local-only).
// Scope: AC(K/V) -> AD(Q) -> AE(score) -> AF(online softmax/output).

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

class TbP11agAttentionChainValidator {
public:
    TbP11agAttentionChainValidator()
        : any_fallback_taken_(false) {}

    int run_all() {
        if (!init_state()) {
            return 1;
        }
        if (!run_real_mainline_path()) {
            return 1;
        }
        if (!run_reference_staged_checks()) {
            return 1;
        }
        if (!validate_stage_kv_staging()) {
            return 1;
        }
        if (!validate_stage_q_path()) {
            return 1;
        }
        if (!validate_stage_final_output_writeback()) {
            return 1;
        }

        if (any_fallback_taken_) {
            std::printf("[p11ag][FAIL] one or more stage fallbacks were taken\n");
            return 1;
        }
        std::printf("fallback_taken = false\n");
        std::printf("FALLBACK_NOT_TAKEN PASS\n");
        std::printf("PASS: tb_attention_chain_correction_validator_p11ag\n");
        return 0;
    }

private:
    std::vector<aecct::u32_t> sram_mainline_;
    std::vector<aecct::u32_t> sram_ref_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::CfgRegs cfg_;
    aecct::LayerScratch sc_;
    bool any_fallback_taken_;

    static uint32_t f32_to_bits(float f) {
        union {
            float f;
            uint32_t u;
        } cvt;
        cvt.f = f;
        return cvt.u;
    }

    static void init_full_x_rows(std::vector<aecct::u32_t>& sram) {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t x_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
        for (uint32_t t = 0u; t < token_count; ++t) {
            const uint32_t row_base = x_base + t * d_model;
            for (uint32_t i = 0u; i < d_model; ++i) {
                const int32_t v = (int32_t)((t + 3u) * 17u + (i + 5u) * 11u) - 211;
                const float f = ((float)v) * 0.015625f;
                sram[row_base + i] = (aecct::u32_t)f32_to_bits(f);
            }
        }
    }

    bool init_state() {
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads_)) {
            std::printf("[p11ag][FAIL] payload preparation failed\n");
            return false;
        }

        sram_mainline_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        sram_ref_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);

        init_full_x_rows(sram_mainline_);
        init_full_x_rows(sram_ref_);

        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_mainline_, payloads_, param_base);
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_ref_, payloads_, param_base);

        cfg_ = p11aeaf_tb::build_cfg();
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);

        return true;
    }

    bool run_real_mainline_path() {
        const aecct::LayerParamBase pb =
            aecct::make_layer_param_base((aecct::u32_t)sram_map::W_REGION_BASE, (aecct::u32_t)0u);

        bool q_fallback = true;
        const bool q_taken = aecct::run_p11ad_layer0_top_managed_q(
            sram_mainline_.data(),
            cfg_,
            (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
            sc_,
            pb,
            q_fallback);
        any_fallback_taken_ = any_fallback_taken_ || q_fallback;
        if (!q_taken || q_fallback) {
            std::printf("[p11ag][FAIL] AD mainline path failed in chain run\n");
            return false;
        }

        bool kv_fallback = true;
        const bool kv_taken = aecct::run_p11ac_layer0_top_managed_kv(
            sram_mainline_.data(),
            cfg_,
            (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
            sc_,
            pb,
            kv_fallback);
        any_fallback_taken_ = any_fallback_taken_ || kv_fallback;
        if (!kv_taken || kv_fallback) {
            std::printf("[p11ag][FAIL] AC mainline path failed in chain run\n");
            return false;
        }

        for (uint32_t t = 0u; t < p11aeaf_tb::kTokenCount; ++t) {
            bool ae_fallback = true;
            const bool ae_taken = aecct::run_p11ae_layer0_top_managed_qk_score(
                sram_mainline_.data(),
                cfg_,
                sc_,
                (aecct::u32_t)t,
                ae_fallback);
            any_fallback_taken_ = any_fallback_taken_ || ae_fallback;
            if (!ae_taken || ae_fallback) {
                std::printf("[p11ag][FAIL] AE mainline path failed in chain run token=%u\n", (unsigned)t);
                return false;
            }

            bool af_fallback = true;
            const bool af_taken = aecct::run_p11af_layer0_top_managed_softmax_out(
                sram_mainline_.data(),
                cfg_,
                sc_,
                (aecct::u32_t)t,
                af_fallback);
            any_fallback_taken_ = any_fallback_taken_ || af_fallback;
            if (!af_taken || af_fallback) {
                std::printf("[p11ag][FAIL] AF mainline path failed in chain run token=%u\n", (unsigned)t);
                return false;
            }
        }

        std::printf("REAL_MAINLINE_PATH_TAKEN PASS\n");
        return true;
    }

    static bool compare_span(
        const char* stage_label,
        const char* span_label,
        const std::vector<aecct::u32_t>& got,
        const std::vector<aecct::u32_t>& ref,
        uint32_t base,
        uint32_t words
    ) {
        for (uint32_t i = 0u; i < words; ++i) {
            const uint32_t gv = (uint32_t)got[base + i].to_uint();
            const uint32_t rv = (uint32_t)ref[base + i].to_uint();
            if (gv != rv) {
                std::printf("[p11ag][FAIL] %s %s mismatch addr=%u offs=%u got=0x%08X ref=0x%08X\n",
                    stage_label, span_label,
                    (unsigned)(base + i), (unsigned)i,
                    (unsigned)gv, (unsigned)rv);
                return false;
            }
        }
        return true;
    }

    bool run_reference_staged_checks() {
        bool q_fallback = true;
        bool kv_fallback = true;
        const bool bootstrap_ok = p11aeaf_tb::run_ac_ad_mainline(sram_ref_, q_fallback, kv_fallback);
        any_fallback_taken_ = any_fallback_taken_ || q_fallback || kv_fallback;
        if (!bootstrap_ok || q_fallback || kv_fallback) {
            std::printf("[p11ag][FAIL] reference AC/AD bootstrap failed (q_fb=%d kv_fb=%d)\n",
                q_fallback ? 1 : 0, kv_fallback ? 1 : 0);
            return false;
        }

        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        const uint32_t pre_base = (uint32_t)sc_.attn.pre_concat_base_word.to_uint();
        const uint32_t post_base = (uint32_t)sc_.attn.post_concat_base_word.to_uint();
        const uint32_t out_base = (uint32_t)sc_.attn_out_base_word.to_uint();

        for (uint32_t t = 0u; t < p11aeaf_tb::kTokenCount; ++t) {
            std::vector<aecct::u32_t> expected_score;
            p11aeaf_tb::compute_expected_score_row(
                sram_ref_,
                sc_.attn,
                t,
                token_count,
                n_heads,
                d_head,
                expected_score);

            bool ae_fallback = true;
            const bool ae_taken = aecct::run_p11ae_layer0_top_managed_qk_score(
                sram_ref_.data(), cfg_, sc_, (aecct::u32_t)t, ae_fallback);
            any_fallback_taken_ = any_fallback_taken_ || ae_fallback;
            if (!ae_taken || ae_fallback) {
                std::printf("[p11ag][FAIL] reference AE execution failed token=%u\n", (unsigned)t);
                return false;
            }

            for (uint32_t h = 0u; h < n_heads; ++h) {
                for (uint32_t j = 0u; j < token_count; ++j) {
                    const uint32_t got = (uint32_t)sram_ref_[score_base + h * token_count + j].to_uint();
                    const uint32_t exp = (uint32_t)expected_score[h * token_count + j].to_uint();
                    if (got != exp) {
                        std::printf("[p11ag][FAIL] score mismatch token=%u head=%u key=%u got=0x%08X exp=0x%08X\n",
                            (unsigned)t, (unsigned)h, (unsigned)j, (unsigned)got, (unsigned)exp);
                        return false;
                    }
                }
            }

            std::vector<aecct::u32_t> expected_out;
            p11aeaf_tb::compute_expected_output_row_online(
                sram_ref_,
                sc_.attn,
                t,
                token_count,
                n_heads,
                d_head,
                expected_out);

            bool af_fallback = true;
            const bool af_taken = aecct::run_p11af_layer0_top_managed_softmax_out(
                sram_ref_.data(), cfg_, sc_, (aecct::u32_t)t, af_fallback);
            any_fallback_taken_ = any_fallback_taken_ || af_fallback;
            if (!af_taken || af_fallback) {
                std::printf("[p11ag][FAIL] reference AF execution failed token=%u\n", (unsigned)t);
                return false;
            }

            const uint32_t row_base = t * d_model;
            for (uint32_t i = 0u; i < d_model; ++i) {
                const uint32_t exp = (uint32_t)expected_out[i].to_uint();
                const uint32_t got_pre = (uint32_t)sram_ref_[pre_base + row_base + i].to_uint();
                const uint32_t got_post = (uint32_t)sram_ref_[post_base + row_base + i].to_uint();
                const uint32_t got_out = (uint32_t)sram_ref_[out_base + row_base + i].to_uint();
                if (got_pre != exp || got_post != exp || got_out != exp) {
                    std::printf(
                        "[p11ag][FAIL] online-softmax/output mismatch token=%u idx=%u pre=0x%08X post=0x%08X out=0x%08X exp=0x%08X\n",
                        (unsigned)t, (unsigned)i,
                        (unsigned)got_pre, (unsigned)got_post, (unsigned)got_out, (unsigned)exp);
                    return false;
                }
            }
        }

        std::printf("STAGE_SCORE_QK PASS\n");
        std::printf("STAGE_ONLINE_SOFTMAX_NORM PASS\n");
        return true;
    }

    bool validate_stage_kv_staging() {
        const uint32_t words = p11aeaf_tb::kTokenCount * (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t k_base = (uint32_t)sc_.attn.k_base_word.to_uint();
        const uint32_t v_base = (uint32_t)sc_.attn.v_base_word.to_uint();
        if (!compare_span("STAGE_KV_STAGING", "K", sram_mainline_, sram_ref_, k_base, words)) {
            return false;
        }
        if (!compare_span("STAGE_KV_STAGING", "V", sram_mainline_, sram_ref_, v_base, words)) {
            return false;
        }
        std::printf("STAGE_KV_STAGING PASS\n");
        return true;
    }

    bool validate_stage_q_path() {
        const uint32_t words = p11aeaf_tb::kTokenCount * (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t q_base = (uint32_t)sc_.attn.q_base_word.to_uint();
        const uint32_t q_act_q_base = (uint32_t)sc_.attn.q_act_q_base_word.to_uint();
        const uint32_t q_sx_base = (uint32_t)sc_.attn.q_sx_base_word.to_uint();
        if (!compare_span("STAGE_Q_PATH", "Q", sram_mainline_, sram_ref_, q_base, words)) {
            return false;
        }
        if (!compare_span("STAGE_Q_PATH", "Q_ACT_Q", sram_mainline_, sram_ref_, q_act_q_base, words)) {
            return false;
        }
        if (!compare_span("STAGE_Q_PATH", "Q_SX", sram_mainline_, sram_ref_, q_sx_base, 1u)) {
            return false;
        }
        std::printf("STAGE_Q_PATH PASS\n");
        return true;
    }

    bool validate_stage_final_output_writeback() {
        const uint32_t words = p11aeaf_tb::kTokenCount * (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t out_base = (uint32_t)sc_.attn_out_base_word.to_uint();
        if (!compare_span("STAGE_FINAL_OUTPUT_WRITEBACK", "ATTN_OUT",
                sram_mainline_, sram_ref_, out_base, words)) {
            return false;
        }
        std::printf("STAGE_FINAL_OUTPUT_WRITEBACK PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbP11agAttentionChainValidator tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__
