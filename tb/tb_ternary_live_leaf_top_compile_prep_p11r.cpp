// tb_ternary_live_leaf_top_compile_prep_p11r.cpp
// P00-011R compile-prep probe TB for single-slice L0_WQ Catapult-facing wrapper.

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

class TbTernaryLiveLeafTopCompilePrepP11R {
public:
    int run_all() {
        const QuantLinearMeta wq_meta = kQuantLinearMeta[(uint32_t)QLM_L0_WQ];
        if (wq_meta.matrix_id != (uint32_t)QLM_L0_WQ) {
            return fail("L0_WQ metadata matrix_id mismatch");
        }
        if (wq_meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
            return fail("L0_WQ metadata layout mismatch");
        }
        if (wq_meta.rows != aecct::kTernaryLiveL0WqRows || wq_meta.cols != aecct::kTernaryLiveL0WqCols) {
            return fail("L0_WQ shape mismatch against split-interface guard");
        }
        if (wq_meta.payload_words_2b != aecct::kTernaryLiveL0WqPayloadWords) {
            return fail("L0_WQ payload_words mismatch against split-interface guard");
        }

        aecct::u32_t x_row[aecct::kTernaryLiveL0WqCols];
        aecct::u32_t payload_words[aecct::kTernaryLiveL0WqPayloadWords];
        aecct::u32_t out_row[aecct::kTernaryLiveL0WqRows];
        aecct::u32_t out_act_q_row[aecct::kTernaryLiveL0WqRows];
        aecct::u32_t ref_out_row[aecct::kTernaryLiveL0WqRows];
        aecct::u32_t ref_out_act_q_row[aecct::kTernaryLiveL0WqRows];

        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WqCols; ++i) {
            x_row[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WqPayloadWords; ++i) {
            payload_words[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WqRows; ++i) {
            out_row[i] = (aecct::u32_t)0u;
            out_act_q_row[i] = (aecct::u32_t)0u;
            ref_out_row[i] = (aecct::u32_t)0u;
            ref_out_act_q_row[i] = (aecct::u32_t)0u;
        }

        fill_x_row_split(x_row, wq_meta.cols);
        if (!build_l0_wq_payload(payload_words, wq_meta)) {
            return fail("failed to pack L0_WQ ternary payload");
        }

        const uint32_t inv_sw_bits =
            (uint32_t)aecct::fp32_bits_from_double(1.0 / w_decoder_layers_0_self_attn_linears_0_s_w[0]).to_uint();

        aecct::u32_t ref_inv_sw_bits = (aecct::u32_t)0u;
        const bool ref_ok = aecct::ternary_live_l0_wq_materialize_row_kernel_split(
            x_row,
            payload_words,
            (aecct::u32_t)inv_sw_bits,
            ref_out_row,
            ref_out_act_q_row,
            ref_inv_sw_bits);

        aecct::u32_t top_inv_sw_bits = (aecct::u32_t)0u;
        aecct::TernaryLiveL0WqRowTopCatapultPrep dut;
        const bool top_ok = dut.run(
            x_row,
            payload_words,
            (aecct::u32_t)inv_sw_bits,
            out_row,
            out_act_q_row,
            top_inv_sw_bits);

        if (top_ok != ref_ok) {
            return fail("return value mismatch between compile-prep top and reference path");
        }
        if (!top_ok) {
            return fail("compile-prep top returned false");
        }
        if (compare_exact(
                "out_inv_sw_bits compile-prep top vs reference",
                (uint32_t)top_inv_sw_bits.to_uint(),
                (uint32_t)ref_inv_sw_bits.to_uint(),
                0u) != 0) {
            return 1;
        }
        for (uint32_t i = 0u; i < wq_meta.rows; ++i) {
            if (compare_exact(
                    "out_row compile-prep top vs reference",
                    (uint32_t)out_row[i].to_uint(),
                    (uint32_t)ref_out_row[i].to_uint(),
                    i) != 0) {
                return 1;
            }
            if (compare_exact(
                    "out_act_q_row compile-prep top vs reference",
                    (uint32_t)out_act_q_row[i].to_uint(),
                    (uint32_t)ref_out_act_q_row[i].to_uint(),
                    i) != 0) {
                return 1;
            }
        }

        std::printf("PASS: tb_ternary_live_leaf_top_compile_prep_p11r\n");
        return 0;
    }

private:
    static int fail(const char* msg) {
        std::printf("[p11r][FAIL] %s\n", msg);
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
            std::printf("[p11r][FAIL] %s mismatch idx=%u got=0x%08X expect=0x%08X got_f=%g expect_f=%g\n",
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

    static bool build_l0_wq_payload(
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
                if (word_idx >= aecct::kTernaryLiveL0WqPayloadWords) {
                    return false;
                }
                const double w = w_decoder_layers_0_self_attn_linears_0_weight[elem_idx];
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

    static void fill_x_row_split(aecct::u32_t x_row[aecct::kTernaryLiveL0WqCols], uint32_t cols) {
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
    TbTernaryLiveLeafTopCompilePrepP11R tb;
    int err = tb.run_all();
    CCS_RETURN(err);
}
