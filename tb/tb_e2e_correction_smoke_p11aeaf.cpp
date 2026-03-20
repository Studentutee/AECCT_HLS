// P00-011AEAF: local-only end-to-end correction smoke for AC+AD+AE+AF path.

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

class TbP11aeafE2e {
public:
    int run_all() {
        init();
        if (!run_ac_ad_mainline()) {
            std::printf("[p11aeaf][FAIL] AC/AD bootstrap failed\n");
            return 1;
        }
        if (!run_ae_af_mainline()) {
            return 1;
        }
        if (!validate_expected_out_rows()) {
            return 1;
        }
        std::printf("PASS: tb_e2e_correction_smoke_p11aeaf\n");
        return 0;
    }

private:
    std::vector<aecct::u32_t> sram_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::LayerScratch sc_;
    aecct::CfgRegs cfg_;

    void init() {
        sram_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        p11aeaf_tb::init_x_rows(sram_);
        p11aeaf_tb::prepare_qkv_payload_set(payloads_);
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_, payloads_, param_base);
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        cfg_ = p11aeaf_tb::build_cfg();
    }

    bool run_ac_ad_mainline() {
        bool q_fallback_taken = true;
        bool kv_fallback_taken = true;
        if (!p11aeaf_tb::run_ac_ad_mainline(sram_, q_fallback_taken, kv_fallback_taken)) {
            return false;
        }
        return (!q_fallback_taken && !kv_fallback_taken);
    }

    bool run_ae_af_mainline() {
        for (uint32_t t = 0u; t < p11aeaf_tb::kTokenCount; ++t) {
            bool score_fallback_taken = true;
            const bool score_mainline_taken = aecct::run_p11ae_layer0_top_managed_qk_score(
                sram_.data(),
                cfg_,
                sc_,
                (aecct::u32_t)t,
                score_fallback_taken);
            if (!score_mainline_taken || score_fallback_taken) {
                std::printf("[p11aeaf][FAIL] AE mainline/fallback check failed token=%u\n", (unsigned)t);
                return false;
            }

            bool out_fallback_taken = true;
            const bool out_mainline_taken = aecct::run_p11af_layer0_top_managed_softmax_out(
                sram_.data(),
                cfg_,
                sc_,
                (aecct::u32_t)t,
                out_fallback_taken);
            if (!out_mainline_taken || out_fallback_taken) {
                std::printf("[p11aeaf][FAIL] AF mainline/fallback check failed token=%u\n", (unsigned)t);
                return false;
            }
        }
        return true;
    }

    bool validate_expected_out_rows() {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t out_base = (uint32_t)sc_.attn_out_base_word.to_uint();
        for (uint32_t t = 0u; t < p11aeaf_tb::kTokenCount; ++t) {
            std::vector<aecct::u32_t> expected_out;
            p11aeaf_tb::compute_expected_output_row_online(
                sram_,
                sc_.attn,
                t,
                token_count,
                n_heads,
                d_head,
                expected_out);
            const uint32_t out_row_base = out_base + t * d_model;
            for (uint32_t i = 0u; i < d_model; ++i) {
                const uint32_t got = (uint32_t)sram_[out_row_base + i].to_uint();
                const uint32_t exp = (uint32_t)expected_out[i].to_uint();
                if (got != exp) {
                    std::printf("[p11aeaf][FAIL] e2e out mismatch token=%u idx=%u got=0x%08X exp=0x%08X\n",
                        (unsigned)t, (unsigned)i, (unsigned)got, (unsigned)exp);
                    return false;
                }
            }
        }
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbP11aeafE2e tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__

