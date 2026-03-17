// tb_ternary_live_leaf_top_compile_prep_family_p11s.cpp
// P00-011S compile-prep family probe TB for L0_WK/L0_WV Catapult-facing wrappers.

#include <cstdio>
#include <cstdint>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "QuantDesc.h"
#include "blocks/TernaryLiveQkvLeafKernel.h"
#include "blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h"
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

class TbTernaryLiveLeafTopCompilePrepFamilyP11S {
public:
    int run_all() {
        const int wk_rc = run_wk_subtest();
        if (wk_rc != 0) {
            return wk_rc;
        }

        const int wv_rc = run_wv_subtest();
        if (wv_rc != 0) {
            return wv_rc;
        }

        std::printf("PASS: tb_ternary_live_leaf_top_compile_prep_family_p11s\n");
        return 0;
    }

private:
    static int run_wk_subtest() {
        const QuantLinearMeta wk_meta = kQuantLinearMeta[(uint32_t)QLM_L0_WK];
        if (wk_meta.matrix_id != (uint32_t)QLM_L0_WK) {
            return fail("L0_WK metadata matrix_id mismatch");
        }
        if (wk_meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
            return fail("L0_WK metadata layout mismatch");
        }
        if (wk_meta.rows != aecct::kTernaryLiveL0WkRows || wk_meta.cols != aecct::kTernaryLiveL0WkCols) {
            return fail("L0_WK shape mismatch against split-interface guard");
        }
        if (wk_meta.payload_words_2b != aecct::kTernaryLiveL0WkPayloadWords) {
            return fail("L0_WK payload_words mismatch against split-interface guard");
        }

        aecct::u32_t x_row[aecct::kTernaryLiveL0WkCols];
        aecct::u32_t payload_words[aecct::kTernaryLiveL0WkPayloadWords];
        aecct::u32_t out_row[aecct::kTernaryLiveL0WkRows];
        aecct::u32_t out_act_q_row[aecct::kTernaryLiveL0WkRows];
        aecct::u32_t ref_out_row[aecct::kTernaryLiveL0WkRows];
        aecct::u32_t ref_out_act_q_row[aecct::kTernaryLiveL0WkRows];

        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WkCols; ++i) {
            x_row[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WkPayloadWords; ++i) {
            payload_words[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WkRows; ++i) {
            out_row[i] = (aecct::u32_t)0u;
            out_act_q_row[i] = (aecct::u32_t)0u;
            ref_out_row[i] = (aecct::u32_t)0u;
            ref_out_act_q_row[i] = (aecct::u32_t)0u;
        }

        fill_x_row_split(x_row, wk_meta.cols);
        if (!build_payload_for_matrix((uint32_t)QLM_L0_WK, payload_words, wk_meta)) {
            return fail("failed to pack L0_WK ternary payload");
        }

        const uint32_t inv_sw_bits =
            (uint32_t)aecct::fp32_bits_from_double(1.0 / w_decoder_layers_0_self_attn_linears_1_s_w[0]).to_uint();

        aecct::u32_t ref_inv_sw_bits = (aecct::u32_t)0u;
        const bool ref_ok = aecct::ternary_live_l0_wk_materialize_row_kernel_split(
            x_row,
            payload_words,
            (aecct::u32_t)inv_sw_bits,
            ref_out_row,
            ref_out_act_q_row,
            ref_inv_sw_bits);

        aecct::u32_t top_inv_sw_bits = (aecct::u32_t)0u;
        aecct::TernaryLiveL0WkRowTopCatapultPrep dut;
        const bool top_ok = dut.run(
            x_row,
            payload_words,
            (aecct::u32_t)inv_sw_bits,
            out_row,
            out_act_q_row,
            top_inv_sw_bits);

        if (top_ok != ref_ok) {
            return fail("L0_WK return value mismatch between compile-prep top and reference path");
        }
        if (!top_ok) {
            return fail("L0_WK compile-prep top returned false");
        }
        if (compare_exact(
                "L0_WK out_inv_sw_bits compile-prep top vs reference",
                (uint32_t)top_inv_sw_bits.to_uint(),
                (uint32_t)ref_inv_sw_bits.to_uint(),
                0u) != 0) {
            return 1;
        }

        for (uint32_t i = 0u; i < wk_meta.rows; ++i) {
            if (compare_exact(
                    "L0_WK out_row compile-prep top vs reference",
                    (uint32_t)out_row[i].to_uint(),
                    (uint32_t)ref_out_row[i].to_uint(),
                    i) != 0) {
                return 1;
            }
            if (compare_exact(
                    "L0_WK out_act_q_row compile-prep top vs reference",
                    (uint32_t)out_act_q_row[i].to_uint(),
                    (uint32_t)ref_out_act_q_row[i].to_uint(),
                    i) != 0) {
                return 1;
            }
        }

        return 0;
    }

    static int run_wv_subtest() {
        const QuantLinearMeta wv_meta = kQuantLinearMeta[(uint32_t)QLM_L0_WV];
        if (wv_meta.matrix_id != (uint32_t)QLM_L0_WV) {
            return fail("L0_WV metadata matrix_id mismatch");
        }
        if (wv_meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
            return fail("L0_WV metadata layout mismatch");
        }
        if (wv_meta.rows != aecct::kTernaryLiveL0WvRows || wv_meta.cols != aecct::kTernaryLiveL0WvCols) {
            return fail("L0_WV shape mismatch against split-interface guard");
        }
        if (wv_meta.payload_words_2b != aecct::kTernaryLiveL0WvPayloadWords) {
            return fail("L0_WV payload_words mismatch against split-interface guard");
        }

        aecct::u32_t x_row[aecct::kTernaryLiveL0WvCols];
        aecct::u32_t payload_words[aecct::kTernaryLiveL0WvPayloadWords];
        aecct::u32_t out_row[aecct::kTernaryLiveL0WvRows];
        aecct::u32_t out_act_q_row[aecct::kTernaryLiveL0WvRows];
        aecct::u32_t ref_out_row[aecct::kTernaryLiveL0WvRows];
        aecct::u32_t ref_out_act_q_row[aecct::kTernaryLiveL0WvRows];

        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WvCols; ++i) {
            x_row[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WvPayloadWords; ++i) {
            payload_words[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WvRows; ++i) {
            out_row[i] = (aecct::u32_t)0u;
            out_act_q_row[i] = (aecct::u32_t)0u;
            ref_out_row[i] = (aecct::u32_t)0u;
            ref_out_act_q_row[i] = (aecct::u32_t)0u;
        }

        fill_x_row_split(x_row, wv_meta.cols);
        if (!build_payload_for_matrix((uint32_t)QLM_L0_WV, payload_words, wv_meta)) {
            return fail("failed to pack L0_WV ternary payload");
        }

        const uint32_t inv_sw_bits =
            (uint32_t)aecct::fp32_bits_from_double(1.0 / w_decoder_layers_0_self_attn_linears_2_s_w[0]).to_uint();

        aecct::u32_t ref_inv_sw_bits = (aecct::u32_t)0u;
        const bool ref_ok = aecct::ternary_live_l0_wv_materialize_row_kernel_split(
            x_row,
            payload_words,
            (aecct::u32_t)inv_sw_bits,
            ref_out_row,
            ref_out_act_q_row,
            ref_inv_sw_bits);

        aecct::u32_t top_inv_sw_bits = (aecct::u32_t)0u;
        aecct::TernaryLiveL0WvRowTopCatapultPrep dut;
        const bool top_ok = dut.run(
            x_row,
            payload_words,
            (aecct::u32_t)inv_sw_bits,
            out_row,
            out_act_q_row,
            top_inv_sw_bits);

        if (top_ok != ref_ok) {
            return fail("L0_WV return value mismatch between compile-prep top and reference path");
        }
        if (!top_ok) {
            return fail("L0_WV compile-prep top returned false");
        }
        if (compare_exact(
                "L0_WV out_inv_sw_bits compile-prep top vs reference",
                (uint32_t)top_inv_sw_bits.to_uint(),
                (uint32_t)ref_inv_sw_bits.to_uint(),
                0u) != 0) {
            return 1;
        }

        for (uint32_t i = 0u; i < wv_meta.rows; ++i) {
            if (compare_exact(
                    "L0_WV out_row compile-prep top vs reference",
                    (uint32_t)out_row[i].to_uint(),
                    (uint32_t)ref_out_row[i].to_uint(),
                    i) != 0) {
                return 1;
            }
            if (compare_exact(
                    "L0_WV out_act_q_row compile-prep top vs reference",
                    (uint32_t)out_act_q_row[i].to_uint(),
                    (uint32_t)ref_out_act_q_row[i].to_uint(),
                    i) != 0) {
                return 1;
            }
        }

        return 0;
    }

    static int fail(const char* msg) {
        std::printf("[p11s][FAIL] %s\n", msg);
        return 1;
    }

    static uint32_t f32_to_bits(float f) {
        union {
            float f;
            uint32_t u;
        } cvt;
        cvt.f = f;
        return cvt.u;
    }

    static float bits_to_f32(uint32_t u) {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.u = u;
        return cvt.f;
    }

    static int compare_exact(const char* label, uint32_t got_bits, uint32_t expect_bits, uint32_t idx) {
        if (got_bits != expect_bits) {
            std::printf("[p11s][FAIL] %s mismatch idx=%u got=0x%08X expect=0x%08X got_f=%g expect_f=%g\n",
                        label,
                        (unsigned)idx,
                        (unsigned)got_bits,
                        (unsigned)expect_bits,
                        (double)bits_to_f32(got_bits),
                        (double)bits_to_f32(expect_bits));
            return 1;
        }
        return 0;
    }

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

    static bool build_payload_for_matrix(
        uint32_t matrix_id,
        aecct::u32_t payload_words[aecct::kTernaryLiveL0WqPayloadWords],
        const QuantLinearMeta& meta
    ) {
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WqPayloadWords; ++i) {
            payload_words[i] = (aecct::u32_t)0u;
        }
        for (uint32_t out = 0u; out < meta.rows; ++out) {
            for (uint32_t in = 0u; in < meta.cols; ++in) {
                const uint32_t elem_idx = out * meta.cols + in;
                const uint32_t word_idx = (elem_idx >> 4);
                const uint32_t slot = (elem_idx & 15u);
                if (word_idx >= meta.payload_words_2b || word_idx >= aecct::kTernaryLiveL0WqPayloadWords) {
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

                payload_words[word_idx] =
                    (aecct::u32_t)((uint32_t)payload_words[word_idx].to_uint() | ((code & 0x3u) << (slot * 2u)));
            }
        }
        return true;
    }

    static void fill_x_row_split(aecct::u32_t* x_row, uint32_t cols) {
        for (uint32_t i = 0u; i < cols; ++i) {
            const float x = (float)((int)(i % 9u) - 4) * 0.125f;
            x_row[i] = (aecct::u32_t)f32_to_bits(x);
        }
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbTernaryLiveLeafTopCompilePrepFamilyP11S tb;
    const int err = tb.run_all();
    CCS_RETURN(err);
}
