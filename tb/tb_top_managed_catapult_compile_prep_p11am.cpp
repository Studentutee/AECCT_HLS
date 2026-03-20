// P00-011AM: Catapult-facing compile-prep harness for TopManagedAttentionChainCatapultTop.

#include <cstdint>
#include <cstdio>

#include "AecctTypes.h"
#include "blocks/TopManagedAttentionChainCatapultTop.h"
#include "gen/WeightStreamOrder.h"
#include "weights.h"

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

class TbTopManagedCatapultCompilePrepP11AM {
public:
    int run_all() {
        aecct::u32_t x_in_words[aecct::ATTN_TENSOR_WORDS];
        aecct::u32_t wq_payload_words[aecct::kQkvCtExpectedL0WqPayloadWords];
        aecct::u32_t wk_payload_words[aecct::kQkvCtExpectedL0WkPayloadWords];
        aecct::u32_t wv_payload_words[aecct::kQkvCtExpectedL0WvPayloadWords];
        aecct::u32_t attn_out_words[aecct::ATTN_TENSOR_WORDS];
        aecct::u32_t final_x_words[aecct::LN_X_TOTAL_WORDS];
        aecct::u32_t wq_inv_sw_bits = (aecct::u32_t)0u;
        aecct::u32_t wk_inv_sw_bits = (aecct::u32_t)0u;
        aecct::u32_t wv_inv_sw_bits = (aecct::u32_t)0u;

        if (!fill_inputs(
                x_in_words,
                wq_payload_words,
                wk_payload_words,
                wv_payload_words,
                wq_inv_sw_bits,
                wk_inv_sw_bits,
                wv_inv_sw_bits)) {
            return fail("failed to build payload/inv_sw inputs");
        }
        clear_outputs(attn_out_words, final_x_words);

        aecct::u32_t out_mainline_all_taken = (aecct::u32_t)0u;
        aecct::u32_t out_fallback_taken = (aecct::u32_t)1u;

        aecct::TopManagedAttentionChainCatapultTop dut;
        const bool ok = dut.run(
            x_in_words,
            wq_payload_words, wq_inv_sw_bits,
            wk_payload_words, wk_inv_sw_bits,
            wv_payload_words, wv_inv_sw_bits,
            attn_out_words,
            final_x_words,
            out_mainline_all_taken,
            out_fallback_taken);

        if (!ok) {
            return fail("wrapper run() returned false");
        }
        if ((uint32_t)out_mainline_all_taken.to_uint() != 1u) {
            return fail("out_mainline_all_taken != 1");
        }
        if ((uint32_t)out_fallback_taken.to_uint() != 0u) {
            return fail("out_fallback_taken != 0");
        }
        if (is_all_zero(attn_out_words, (uint32_t)aecct::ATTN_TENSOR_WORDS) &&
            is_all_zero(final_x_words, (uint32_t)aecct::LN_X_TOTAL_WORDS)) {
            return fail("both output windows are all zero");
        }

        std::printf("P11AM_WRAPPER_RUNTIME_SMOKE PASS\n");
        std::printf("P11AM_MAINLINE_PATH_TAKEN PASS\n");
        std::printf("fallback_taken = false\n");
        std::printf("P11AM_FALLBACK_NOT_TAKEN PASS\n");
        std::printf("PASS: tb_top_managed_catapult_compile_prep_p11am\n");
        return 0;
    }

private:
    static uint32_t encode_ternary_code(double w) {
        if (w > 0.5) {
            return (uint32_t)TERNARY_CODE_POS;
        }
        if (w < -0.5) {
            return (uint32_t)TERNARY_CODE_NEG;
        }
        return (uint32_t)TERNARY_CODE_ZERO;
    }

    static bool matrix_weight_at(uint32_t matrix_id, uint32_t elem_idx, double& out_w) {
        if (matrix_id == (uint32_t)QLM_L0_WQ) {
            out_w = w_decoder_layers_0_self_attn_linears_0_weight[elem_idx];
            return true;
        }
        if (matrix_id == (uint32_t)QLM_L0_WK) {
            out_w = w_decoder_layers_0_self_attn_linears_1_weight[elem_idx];
            return true;
        }
        if (matrix_id == (uint32_t)QLM_L0_WV) {
            out_w = w_decoder_layers_0_self_attn_linears_2_weight[elem_idx];
            return true;
        }
        return false;
    }

    static bool matrix_inv_sw(uint32_t matrix_id, double& out_inv_sw) {
        if (matrix_id == (uint32_t)QLM_L0_WQ) {
            out_inv_sw = w_decoder_layers_0_self_attn_linears_0_s_w[0];
            return true;
        }
        if (matrix_id == (uint32_t)QLM_L0_WK) {
            out_inv_sw = w_decoder_layers_0_self_attn_linears_1_s_w[0];
            return true;
        }
        if (matrix_id == (uint32_t)QLM_L0_WV) {
            out_inv_sw = w_decoder_layers_0_self_attn_linears_2_s_w[0];
            return true;
        }
        return false;
    }

    static bool build_payload_for_matrix(
        uint32_t matrix_id,
        const QuantLinearMeta& meta,
        aecct::u32_t* payload_words,
        uint32_t payload_words_count
    ) {
        if (payload_words_count != meta.payload_words_2b) {
            return false;
        }
        for (uint32_t i = 0u; i < payload_words_count; ++i) {
            payload_words[i] = (aecct::u32_t)0u;
        }
        for (uint32_t out = 0u; out < meta.rows; ++out) {
            for (uint32_t in = 0u; in < meta.cols; ++in) {
                const uint32_t elem_idx = out * meta.cols + in;
                const uint32_t word_idx = (elem_idx >> 4);
                const uint32_t slot = (elem_idx & 15u);
                if (word_idx >= payload_words_count) {
                    return false;
                }
                double w = 0.0;
                if (!matrix_weight_at(matrix_id, elem_idx, w)) {
                    return false;
                }
                const uint32_t code = encode_ternary_code(w);
                if (code == (uint32_t)TERNARY_CODE_RSVD) {
                    return false;
                }
                payload_words[word_idx] = (aecct::u32_t)((uint32_t)payload_words[word_idx].to_uint() |
                    ((code & 0x3u) << (slot * 2u)));
            }
        }
        return true;
    }

    static uint32_t f32_to_bits(float f) {
        union {
            float f;
            uint32_t u;
        } cvt;
        cvt.f = f;
        return cvt.u;
    }

    static int fail(const char* msg) {
        std::printf("[p11am][FAIL] %s\n", msg);
        return 1;
    }

    static bool fill_inputs(
        aecct::u32_t x_in_words[aecct::ATTN_TENSOR_WORDS],
        aecct::u32_t wq_payload_words[aecct::kQkvCtExpectedL0WqPayloadWords],
        aecct::u32_t wk_payload_words[aecct::kQkvCtExpectedL0WkPayloadWords],
        aecct::u32_t wv_payload_words[aecct::kQkvCtExpectedL0WvPayloadWords],
        aecct::u32_t& wq_inv_sw_bits,
        aecct::u32_t& wk_inv_sw_bits,
        aecct::u32_t& wv_inv_sw_bits
    ) {
        for (uint32_t i = 0u; i < (uint32_t)aecct::ATTN_TENSOR_WORDS; ++i) {
            const int32_t v = (int32_t)((i + 5u) * 29u) - 701;
            x_in_words[i] = (aecct::u32_t)f32_to_bits(((float)v) * 0.00390625f);
        }
        const QuantLinearMeta wq_meta = aecct::ternary_linear_live_l0_wq_meta();
        const QuantLinearMeta wk_meta = aecct::ternary_linear_live_l0_wk_meta();
        const QuantLinearMeta wv_meta = aecct::ternary_linear_live_l0_wv_meta();
        if (!build_payload_for_matrix((uint32_t)QLM_L0_WQ, wq_meta, wq_payload_words, (uint32_t)aecct::kQkvCtExpectedL0WqPayloadWords)) { return false; }
        if (!build_payload_for_matrix((uint32_t)QLM_L0_WK, wk_meta, wk_payload_words, (uint32_t)aecct::kQkvCtExpectedL0WkPayloadWords)) { return false; }
        if (!build_payload_for_matrix((uint32_t)QLM_L0_WV, wv_meta, wv_payload_words, (uint32_t)aecct::kQkvCtExpectedL0WvPayloadWords)) { return false; }

        double wq_sw = 0.0;
        double wk_sw = 0.0;
        double wv_sw = 0.0;
        if (!matrix_inv_sw((uint32_t)QLM_L0_WQ, wq_sw)) { return false; }
        if (!matrix_inv_sw((uint32_t)QLM_L0_WK, wk_sw)) { return false; }
        if (!matrix_inv_sw((uint32_t)QLM_L0_WV, wv_sw)) { return false; }

        wq_inv_sw_bits = (aecct::u32_t)aecct::fp32_bits_from_double(1.0 / wq_sw);
        wk_inv_sw_bits = (aecct::u32_t)aecct::fp32_bits_from_double(1.0 / wk_sw);
        wv_inv_sw_bits = (aecct::u32_t)aecct::fp32_bits_from_double(1.0 / wv_sw);
        return true;
    }

    static void clear_outputs(
        aecct::u32_t attn_out_words[aecct::ATTN_TENSOR_WORDS],
        aecct::u32_t final_x_words[aecct::LN_X_TOTAL_WORDS]
    ) {
        for (uint32_t i = 0u; i < (uint32_t)aecct::ATTN_TENSOR_WORDS; ++i) {
            attn_out_words[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < (uint32_t)aecct::LN_X_TOTAL_WORDS; ++i) {
            final_x_words[i] = (aecct::u32_t)0u;
        }
    }

    static bool is_all_zero(const aecct::u32_t* words, uint32_t n) {
        for (uint32_t i = 0u; i < n; ++i) {
            if ((uint32_t)words[i].to_uint() != 0u) {
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
    TbTopManagedCatapultCompilePrepP11AM tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}
