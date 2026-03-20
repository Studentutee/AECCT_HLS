// P00-011AF: Top-managed single-pass online softmax/output mainline proof (local-only).

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

class TbP11af {
public:
    TbP11af()
        : token_idx_(0u),
          mainline_softmax_output_path_taken_(false),
          fallback_taken_(true) {}

    int run_all() {
        init();
        if (!run_ac_ad_ae_bootstrap()) {
            std::printf("[p11af][FAIL] AC/AD/AE bootstrap failed\n");
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
        std::printf("PASS: tb_softmax_out_impl_p11af\n");
        return 0;
    }

private:
    uint32_t token_idx_;
    bool mainline_softmax_output_path_taken_;
    bool fallback_taken_;

    std::vector<aecct::u32_t> sram_;
    std::vector<aecct::u32_t> sram_before_af_;
    std::vector<aecct::u32_t> expected_out_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::LayerScratch sc_;
    aecct::CfgRegs cfg_;

    void init() {
        sram_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        p11aeaf_tb::init_x_rows(sram_);
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads_)) {
            std::printf("[p11af][FAIL] payload preparation failed\n");
        }
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_, payloads_, param_base);
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        cfg_ = p11aeaf_tb::build_cfg();
    }

    bool run_ac_ad_ae_bootstrap() {
        bool q_fallback_taken = true;
        bool kv_fallback_taken = true;
        if (!p11aeaf_tb::run_ac_ad_mainline(sram_, q_fallback_taken, kv_fallback_taken)) {
            return false;
        }

        bool score_fallback_taken = true;
        const bool score_mainline_taken = aecct::run_p11ae_layer0_top_managed_qk_score(
            sram_.data(),
            cfg_,
            sc_,
            (aecct::u32_t)token_idx_,
            score_fallback_taken);
        if (!score_mainline_taken || score_fallback_taken) {
            return false;
        }
        return true;
    }

    bool run_design_mainline_probe() {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
        const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
        p11aeaf_tb::compute_expected_output_row_online(
            sram_,
            sc_.attn,
            token_idx_,
            token_count,
            n_heads,
            d_head,
            expected_out_);

        sram_before_af_ = sram_;
        fallback_taken_ = true;
        mainline_softmax_output_path_taken_ = aecct::run_p11af_layer0_top_managed_softmax_out(
            sram_.data(),
            cfg_,
            sc_,
            (aecct::u32_t)token_idx_,
            fallback_taken_);
        std::printf("fallback_taken = %s\n", fallback_taken_ ? "true" : "false");
        if (!mainline_softmax_output_path_taken_) {
            std::printf("[p11af][FAIL] Top mainline softmax/output path was not taken\n");
            return false;
        }
        if (fallback_taken_) {
            std::printf("[p11af][FAIL] fallback path was taken in Top mainline softmax/output probe\n");
            return false;
        }
        std::printf("SOFTMAX_MAINLINE PASS\n");
        return true;
    }

    bool validate_expected_compare() {
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t pre_row_base = (uint32_t)sc_.attn.pre_concat_base_word.to_uint() + token_idx_ * d_model;
        const uint32_t post_row_base = (uint32_t)sc_.attn.post_concat_base_word.to_uint() + token_idx_ * d_model;
        const uint32_t out_row_base = (uint32_t)sc_.attn_out_base_word.to_uint() + token_idx_ * d_model;
        for (uint32_t i = 0u; i < d_model; ++i) {
            const uint32_t exp = (uint32_t)expected_out_[i].to_uint();
            const uint32_t got_pre = (uint32_t)sram_[pre_row_base + i].to_uint();
            const uint32_t got_post = (uint32_t)sram_[post_row_base + i].to_uint();
            const uint32_t got_out = (uint32_t)sram_[out_row_base + i].to_uint();
            if (got_pre != exp || got_post != exp || got_out != exp) {
                std::printf("[p11af][FAIL] output compare mismatch idx=%u pre=0x%08X post=0x%08X out=0x%08X exp=0x%08X\n",
                    (unsigned)i, (unsigned)got_pre, (unsigned)got_post, (unsigned)got_out, (unsigned)exp);
                return false;
            }
        }
        std::printf("OUTPUT_EXPECTED_COMPARE PASS\n");
        return true;
    }

    bool validate_target_span_write() {
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t pre_row_base = (uint32_t)sc_.attn.pre_concat_base_word.to_uint() + token_idx_ * d_model;
        const uint32_t post_row_base = (uint32_t)sc_.attn.post_concat_base_word.to_uint() + token_idx_ * d_model;
        const uint32_t out_row_base = (uint32_t)sc_.attn_out_base_word.to_uint() + token_idx_ * d_model;
        uint32_t changed_pre = 0u;
        uint32_t changed_post = 0u;
        uint32_t changed_out = 0u;
        for (uint32_t i = 0u; i < d_model; ++i) {
            if ((uint32_t)sram_[pre_row_base + i].to_uint() != (uint32_t)sram_before_af_[pre_row_base + i].to_uint()) {
                ++changed_pre;
            }
            if ((uint32_t)sram_[post_row_base + i].to_uint() != (uint32_t)sram_before_af_[post_row_base + i].to_uint()) {
                ++changed_post;
            }
            if ((uint32_t)sram_[out_row_base + i].to_uint() != (uint32_t)sram_before_af_[out_row_base + i].to_uint()) {
                ++changed_out;
            }
        }
        if (changed_pre == 0u || changed_post == 0u || changed_out == 0u) {
            std::printf("[p11af][FAIL] output target spans were not fully written\n");
            return false;
        }
        std::printf("[p11af][TARGET_SPAN_WRITE][PASS] changed_pre=%u changed_post=%u changed_out=%u\n",
            (unsigned)changed_pre, (unsigned)changed_post, (unsigned)changed_out);
        std::printf("OUTPUT_TARGET_SPAN_WRITE PASS\n");
        return true;
    }

    bool validate_no_spurious_write_and_source_preservation() {
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t pre_row_base = (uint32_t)sc_.attn.pre_concat_base_word.to_uint() + token_idx_ * d_model;
        const uint32_t post_row_base = (uint32_t)sc_.attn.post_concat_base_word.to_uint() + token_idx_ * d_model;
        const uint32_t out_row_base = (uint32_t)sc_.attn_out_base_word.to_uint() + token_idx_ * d_model;
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        const uint32_t v_base = (uint32_t)sc_.attn.v_base_word.to_uint();

        std::vector<uint8_t> allowed_write((uint32_t)sram_.size(), 0u);
        for (uint32_t i = 0u; i < d_model; ++i) {
            allowed_write[pre_row_base + i] = 1u;
            allowed_write[post_row_base + i] = 1u;
            allowed_write[out_row_base + i] = 1u;
        }

        for (uint32_t h = 0u; h < (uint32_t)aecct::ATTN_N_HEADS; ++h) {
            const uint32_t head_base = score_base + h * token_count;
            for (uint32_t j = 0u; j < token_count; ++j) {
                if ((uint32_t)sram_[head_base + j].to_uint() !=
                    (uint32_t)sram_before_af_[head_base + j].to_uint()) {
                    std::printf("[p11af][FAIL] score source preservation mismatch head=%u key=%u\n",
                        (unsigned)h, (unsigned)j);
                    return false;
                }
            }
        }
        for (uint32_t t = 0u; t < p11aeaf_tb::kTokenCount; ++t) {
            const uint32_t v_row_base = v_base + t * d_model;
            for (uint32_t i = 0u; i < d_model; ++i) {
                if ((uint32_t)sram_[v_row_base + i].to_uint() !=
                    (uint32_t)sram_before_af_[v_row_base + i].to_uint()) {
                    std::printf("[p11af][FAIL] V source preservation mismatch token=%u idx=%u\n",
                        (unsigned)t, (unsigned)i);
                    return false;
                }
            }
        }

        for (uint32_t i = 0u; i < (uint32_t)sram_.size(); ++i) {
            const uint32_t before = (uint32_t)sram_before_af_[i].to_uint();
            const uint32_t after = (uint32_t)sram_[i].to_uint();
            if (before != after && allowed_write[i] == 0u) {
                std::printf("[p11af][FAIL] no-spurious-write mismatch addr=%u before=0x%08X after=0x%08X\n",
                    (unsigned)i, (unsigned)before, (unsigned)after);
                return false;
            }
        }

        std::printf("NO_SPURIOUS_WRITE PASS\n");
        std::printf("SOURCE_PRESERVATION PASS\n");
        std::printf("MAINLINE_SOFTMAX_OUTPUT_PATH_TAKEN PASS\n");
        std::printf("FALLBACK_NOT_TAKEN PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbP11af tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__

