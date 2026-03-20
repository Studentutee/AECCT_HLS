// P00-011AL: Catapult-facing Top wrapper smoke/compare harness (local-only).

#ifndef __SYNTHESIS__

#include <cstdint>
#include <cstdio>
#include <vector>

#include "tb_p11aeaf_common.h"
#include "blocks/TopManagedAttentionChainCatapultTop.h"
#include "gen/SramMap.h"

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

class TbTopManagedCatapultWrapperP11AL {
public:
    int run_all() {
        if (!prepare_inputs()) {
            return 1;
        }
        if (!run_catapult_facing_wrapper()) {
            return 1;
        }
        if (!run_reference_direct_path()) {
            return 1;
        }
        if (!compare_wrapper_vs_reference()) {
            return 1;
        }

        std::printf("CATAPULT_TOP_WRAPPER_INTERFACE_SHAPE PASS\n");
        std::printf("TOP_WRAPPER_MAINLINE_PATH_TAKEN PASS\n");
        std::printf("fallback_taken = false\n");
        std::printf("TOP_WRAPPER_FALLBACK_NOT_TAKEN PASS\n");
        std::printf("TOP_WRAPPER_ATTN_OUT_EXPECTED_COMPARE PASS\n");
        std::printf("TOP_WRAPPER_FINAL_X_EXPECTED_COMPARE PASS\n");
        std::printf("PASS: tb_top_managed_catapult_wrapper_p11al\n");
        return 0;
    }

private:
    aecct::u32_t x_in_words_[aecct::ATTN_TENSOR_WORDS];
    aecct::u32_t wq_payload_words_[aecct::kQkvCtExpectedL0WqPayloadWords];
    aecct::u32_t wk_payload_words_[aecct::kQkvCtExpectedL0WkPayloadWords];
    aecct::u32_t wv_payload_words_[aecct::kQkvCtExpectedL0WvPayloadWords];
    aecct::u32_t wq_inv_sw_bits_;
    aecct::u32_t wk_inv_sw_bits_;
    aecct::u32_t wv_inv_sw_bits_;

    aecct::u32_t attn_out_wrapper_[aecct::ATTN_TENSOR_WORDS];
    aecct::u32_t final_x_wrapper_[aecct::LN_X_TOTAL_WORDS];
    aecct::u32_t mainline_all_taken_wrapper_;
    aecct::u32_t fallback_taken_wrapper_;

    aecct::u32_t attn_out_ref_[aecct::ATTN_TENSOR_WORDS];
    aecct::u32_t final_x_ref_[aecct::LN_X_TOTAL_WORDS];

    static uint32_t f32_to_bits(float f) {
        union {
            float f;
            uint32_t u;
        } cvt;
        cvt.f = f;
        return cvt.u;
    }

    static int fail(const char* msg) {
        std::printf("[p11al][FAIL] %s\n", msg);
        return 1;
    }

    bool prepare_inputs() {
        for (uint32_t i = 0u; i < (uint32_t)aecct::ATTN_TENSOR_WORDS; ++i) {
            const int32_t v = (int32_t)((i + 7u) * 19u) - 311;
            const float f = ((float)v) * 0.0078125f;
            x_in_words_[i] = (aecct::u32_t)f32_to_bits(f);
        }
        for (uint32_t i = 0u; i < (uint32_t)aecct::ATTN_TENSOR_WORDS; ++i) {
            attn_out_wrapper_[i] = (aecct::u32_t)0u;
            attn_out_ref_[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < (uint32_t)aecct::LN_X_TOTAL_WORDS; ++i) {
            final_x_wrapper_[i] = (aecct::u32_t)0u;
            final_x_ref_[i] = (aecct::u32_t)0u;
        }

        p11aeaf_tb::QkvPayloadSet payloads;
        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads)) {
            return fail("payload preparation failed") == 0;
        }
        if (payloads.wq_payload.size() != (size_t)aecct::kQkvCtExpectedL0WqPayloadWords) {
            return fail("WQ payload size mismatch against fixed interface array") == 0;
        }
        if (payloads.wk_payload.size() != (size_t)aecct::kQkvCtExpectedL0WkPayloadWords) {
            return fail("WK payload size mismatch against fixed interface array") == 0;
        }
        if (payloads.wv_payload.size() != (size_t)aecct::kQkvCtExpectedL0WvPayloadWords) {
            return fail("WV payload size mismatch against fixed interface array") == 0;
        }

        for (uint32_t i = 0u; i < (uint32_t)aecct::kQkvCtExpectedL0WqPayloadWords; ++i) {
            wq_payload_words_[i] = payloads.wq_payload[i];
        }
        for (uint32_t i = 0u; i < (uint32_t)aecct::kQkvCtExpectedL0WkPayloadWords; ++i) {
            wk_payload_words_[i] = payloads.wk_payload[i];
        }
        for (uint32_t i = 0u; i < (uint32_t)aecct::kQkvCtExpectedL0WvPayloadWords; ++i) {
            wv_payload_words_[i] = payloads.wv_payload[i];
        }
        wq_inv_sw_bits_ = payloads.wq_inv_sw_bits;
        wk_inv_sw_bits_ = payloads.wk_inv_sw_bits;
        wv_inv_sw_bits_ = payloads.wv_inv_sw_bits;
        return true;
    }

    bool run_catapult_facing_wrapper() {
        aecct::TopManagedAttentionChainCatapultTop dut;
        mainline_all_taken_wrapper_ = (aecct::u32_t)0u;
        fallback_taken_wrapper_ = (aecct::u32_t)1u;
        const bool ok = dut.run(
            x_in_words_,
            wq_payload_words_, wq_inv_sw_bits_,
            wk_payload_words_, wk_inv_sw_bits_,
            wv_payload_words_, wv_inv_sw_bits_,
            attn_out_wrapper_,
            final_x_wrapper_,
            mainline_all_taken_wrapper_,
            fallback_taken_wrapper_);
        if (!ok) {
            return fail("catapult-facing wrapper returned false") == 0;
        }
        if ((uint32_t)mainline_all_taken_wrapper_.to_uint() != 1u) {
            return fail("wrapper mainline flag is not asserted") == 0;
        }
        if ((uint32_t)fallback_taken_wrapper_.to_uint() != 0u) {
            return fail("wrapper fallback flag is asserted") == 0;
        }
        return true;
    }

    bool run_reference_direct_path() {
        std::vector<aecct::u32_t> sram((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);

        const uint32_t x_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
        for (uint32_t i = 0u; i < (uint32_t)aecct::ATTN_TENSOR_WORDS; ++i) {
            sram[x_base + i] = x_in_words_[i];
        }

        p11aeaf_tb::QkvPayloadSet payloads;
        payloads.wq_payload.assign(
            wq_payload_words_,
            wq_payload_words_ + (uint32_t)aecct::kQkvCtExpectedL0WqPayloadWords);
        payloads.wk_payload.assign(
            wk_payload_words_,
            wk_payload_words_ + (uint32_t)aecct::kQkvCtExpectedL0WkPayloadWords);
        payloads.wv_payload.assign(
            wv_payload_words_,
            wv_payload_words_ + (uint32_t)aecct::kQkvCtExpectedL0WvPayloadWords);
        payloads.wq_inv_sw_bits = wq_inv_sw_bits_;
        payloads.wk_inv_sw_bits = wk_inv_sw_bits_;
        payloads.wv_inv_sw_bits = wv_inv_sw_bits_;
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram, payloads, (uint32_t)sram_map::W_REGION_BASE);

        aecct::TopRegs regs;
        regs.clear();
        regs.w_base_set = true;
        regs.w_base_word = (aecct::u32_t)sram_map::W_REGION_BASE;
        regs.cfg_d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
        regs.cfg_n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
        regs.cfg_d_ffn = (aecct::u32_t)D_FFN;
        regs.cfg_n_layers = (aecct::u32_t)1u;
        regs.cfg_ready = true;

        aecct::run_transformer_layer_loop(regs, sram.data());

        const bool ref_mainline_all_taken =
            regs.p11ac_mainline_path_taken &&
            regs.p11ad_mainline_q_path_taken &&
            regs.p11ae_mainline_score_path_taken &&
            regs.p11af_mainline_softmax_output_path_taken;
        if (!ref_mainline_all_taken) {
            return fail("reference mainline flags are not all asserted") == 0;
        }
        const bool ref_fallback_taken =
            regs.p11ac_fallback_taken ||
            regs.p11ad_q_fallback_taken ||
            regs.p11ae_score_fallback_taken ||
            regs.p11af_softmax_output_fallback_taken;
        if (ref_fallback_taken) {
            return fail("reference fallback flag is asserted") == 0;
        }

        const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const uint32_t attn_out_base = (uint32_t)sc.attn_out_base_word.to_uint();
        for (uint32_t i = 0u; i < (uint32_t)aecct::ATTN_TENSOR_WORDS; ++i) {
            attn_out_ref_[i] = sram[attn_out_base + i];
        }

        const uint32_t final_x_base = (uint32_t)regs.infer_final_x_base_word.to_uint();
        for (uint32_t i = 0u; i < (uint32_t)aecct::LN_X_TOTAL_WORDS; ++i) {
            final_x_ref_[i] = sram[final_x_base + i];
        }
        return true;
    }

    bool compare_wrapper_vs_reference() const {
        for (uint32_t i = 0u; i < (uint32_t)aecct::ATTN_TENSOR_WORDS; ++i) {
            const uint32_t got = (uint32_t)attn_out_wrapper_[i].to_uint();
            const uint32_t exp = (uint32_t)attn_out_ref_[i].to_uint();
            if (got != exp) {
                std::printf("[p11al][FAIL] attn_out mismatch idx=%u got=0x%08X exp=0x%08X\n",
                            (unsigned)i, (unsigned)got, (unsigned)exp);
                return false;
            }
        }

        for (uint32_t i = 0u; i < (uint32_t)aecct::LN_X_TOTAL_WORDS; ++i) {
            const uint32_t got = (uint32_t)final_x_wrapper_[i].to_uint();
            const uint32_t exp = (uint32_t)final_x_ref_[i].to_uint();
            if (got != exp) {
                std::printf("[p11al][FAIL] final_x mismatch idx=%u got=0x%08X exp=0x%08X\n",
                            (unsigned)i, (unsigned)got, (unsigned)exp);
                return false;
            }
        }
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbTopManagedCatapultWrapperP11AL tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__
