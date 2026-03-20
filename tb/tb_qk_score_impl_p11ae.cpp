// P00-011AE: Top-managed QK/score mainline implementation proof (local-only).

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

class TbP11ae {
public:
    TbP11ae()
        : token_idx_(0u),
          mainline_score_path_taken_(false),
          fallback_taken_(true) {}

    int run_all() {
        init();
        if (!run_ac_ad_bootstrap()) {
            std::printf("[p11ae][FAIL] AC/AD bootstrap failed\n");
            return 1;
        }
        if (!run_design_mainline_probe()) {
            return 1;
        }
        if (!validate_expected_compare()) {
            return 1;
        }
        if (!validate_target_span_write()) {
            return 1;
        }
        if (!validate_no_spurious_write_and_source_preservation()) {
            return 1;
        }
        std::printf("PASS: tb_qk_score_impl_p11ae\n");
        return 0;
    }

private:
    uint32_t token_idx_;
    bool mainline_score_path_taken_;
    bool fallback_taken_;

    std::vector<aecct::u32_t> sram_;
    std::vector<aecct::u32_t> sram_before_ae_;
    std::vector<aecct::u32_t> expected_scores_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::LayerScratch sc_;
    aecct::CfgRegs cfg_;

    void init() {
        sram_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        p11aeaf_tb::init_x_rows(sram_);
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads_)) {
            std::printf("[p11ae][FAIL] payload preparation failed\n");
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_, payloads_, param_base);
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        cfg_ = p11aeaf_tb::build_cfg();
    }

    bool run_ac_ad_bootstrap() {
        bool q_fallback_taken = true;
        bool kv_fallback_taken = true;
        if (!p11aeaf_tb::run_ac_ad_mainline(sram_, q_fallback_taken, kv_fallback_taken)) {
            return false;
        }
        return (!q_fallback_taken && !kv_fallback_taken);
    }

    bool run_design_mainline_probe() {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        p11aeaf_tb::compute_expected_score_row(
            sram_,
            sc_.attn,
            token_idx_,
            token_count,
            n_heads,
            d_head,
            expected_scores_);

        sram_before_ae_ = sram_;
        fallback_taken_ = true;
        mainline_score_path_taken_ = aecct::run_p11ae_layer0_top_managed_qk_score(
            sram_.data(),
            cfg_,
            sc_,
            (aecct::u32_t)token_idx_,
            fallback_taken_);
        std::printf("fallback_taken = %s\n", fallback_taken_ ? "true" : "false");
        if (!mainline_score_path_taken_) {
            std::printf("[p11ae][FAIL] Top mainline score path was not taken\n");
            return false;
        }
        if (fallback_taken_) {
            std::printf("[p11ae][FAIL] fallback path was taken in Top mainline score probe\n");
            return false;
        }
        std::printf("QK_SCORE_MAINLINE PASS\n");
        return true;
    }

    bool validate_expected_compare() {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        for (uint32_t h = 0u; h < n_heads; ++h) {
            const uint32_t head_base = score_base + h * token_count;
            for (uint32_t j = 0u; j < token_count; ++j) {
                const uint32_t got = (uint32_t)sram_[head_base + j].to_uint();
                const uint32_t exp = (uint32_t)expected_scores_[h * token_count + j].to_uint();
                if (got != exp) {
                    std::printf("[p11ae][FAIL] score compare mismatch head=%u key=%u got=0x%08X exp=0x%08X\n",
                        (unsigned)h, (unsigned)j, (unsigned)got, (unsigned)exp);
                    return false;
                }
            }
        }
        std::printf("SCORE_EXPECTED_COMPARE PASS\n");
        return true;
    }

    bool validate_target_span_write() {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        uint32_t changed_scores = 0u;
        for (uint32_t h = 0u; h < n_heads; ++h) {
            const uint32_t head_base = score_base + h * token_count;
            for (uint32_t j = 0u; j < token_count; ++j) {
                if ((uint32_t)sram_[head_base + j].to_uint() !=
                    (uint32_t)sram_before_ae_[head_base + j].to_uint()) {
                    ++changed_scores;
                }
            }
        }
        if (changed_scores == 0u) {
            std::printf("[p11ae][FAIL] score target span was not written\n");
            return false;
        }
        std::printf("[p11ae][TARGET_SPAN_WRITE][PASS] changed_scores=%u\n", (unsigned)changed_scores);
        std::printf("SCORE_TARGET_SPAN_WRITE PASS\n");
        return true;
    }

    bool validate_no_spurious_write_and_source_preservation() {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        const uint32_t q_base = (uint32_t)sc_.attn.q_base_word.to_uint();
        const uint32_t k_base = (uint32_t)sc_.attn.k_base_word.to_uint();

        std::vector<uint8_t> allowed_write((uint32_t)sram_.size(), 0u);
        for (uint32_t h = 0u; h < n_heads; ++h) {
            const uint32_t head_base = score_base + h * token_count;
            for (uint32_t j = 0u; j < token_count; ++j) {
                allowed_write[head_base + j] = 1u;
            }
        }

        for (uint32_t t = 0u; t < p11aeaf_tb::kTokenCount; ++t) {
            const uint32_t q_row_base = q_base + t * d_model;
            const uint32_t k_row_base = k_base + t * d_model;
            for (uint32_t i = 0u; i < d_model; ++i) {
                if ((uint32_t)sram_[q_row_base + i].to_uint() !=
                    (uint32_t)sram_before_ae_[q_row_base + i].to_uint()) {
                    std::printf("[p11ae][FAIL] Q source preservation mismatch token=%u idx=%u\n",
                        (unsigned)t, (unsigned)i);
                    return false;
                }
                if ((uint32_t)sram_[k_row_base + i].to_uint() !=
                    (uint32_t)sram_before_ae_[k_row_base + i].to_uint()) {
                    std::printf("[p11ae][FAIL] K source preservation mismatch token=%u idx=%u\n",
                        (unsigned)t, (unsigned)i);
                    return false;
                }
            }
        }

        for (uint32_t i = 0u; i < (uint32_t)sram_.size(); ++i) {
            const uint32_t before = (uint32_t)sram_before_ae_[i].to_uint();
            const uint32_t after = (uint32_t)sram_[i].to_uint();
            if (before != after && allowed_write[i] == 0u) {
                std::printf("[p11ae][FAIL] no-spurious-write mismatch addr=%u before=0x%08X after=0x%08X\n",
                    (unsigned)i, (unsigned)before, (unsigned)after);
                return false;
            }
        }

        std::printf("NO_SPURIOUS_WRITE PASS\n");
        std::printf("SOURCE_PRESERVATION PASS\n");
        std::printf("MAINLINE_SCORE_PATH_TAKEN PASS\n");
        std::printf("FALLBACK_NOT_TAKEN PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbP11ae tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__

