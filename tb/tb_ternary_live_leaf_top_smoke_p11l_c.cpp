// tb_ternary_live_leaf_top_smoke_p11l_c.cpp
// P00-011L-C: family-level common smoke driver for L0_WQ/L0_WK/L0_WV split-interface tops.

#include <cstdio>
#include <cstdint>
#include <vector>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "QuantDesc.h"
#include "blocks/TernaryLiveQkvLeafKernel.h"
#include "blocks/TernaryLiveQkvLeafKernelTop.h"
#include "gen/SramMap.h"
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

class TbTernaryLiveLeafTopP11LC {
public:
    int run_all() {
        const int wq_rc = run_case(
            (uint32_t)QLM_L0_WQ,
            "L0_WQ split-interface top run() exact-match equivalent to direct kernel output");
        if (wq_rc != 0) {
            return wq_rc;
        }
        const int wk_rc = run_case(
            (uint32_t)QLM_L0_WK,
            "L0_WK split-interface top run() exact-match equivalent to direct kernel output");
        if (wk_rc != 0) {
            return wk_rc;
        }
        const int wv_rc = run_case(
            (uint32_t)QLM_L0_WV,
            "L0_WV split-interface top run() exact-match equivalent to direct kernel output");
        if (wv_rc != 0) {
            return wv_rc;
        }

        std::printf("PASS: tb_ternary_live_leaf_top_smoke_p11l_c\n");
        return 0;
    }

private:
    static int run_case(uint32_t matrix_id, const char* pass_message) {
        const QuantLinearMeta meta = kQuantLinearMeta[matrix_id];
        if (meta.matrix_id != matrix_id) {
            return fail("metadata matrix_id mismatch");
        }
        if (meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
            return fail("metadata layout mismatch");
        }
        if (meta.rows != 32u || meta.cols != 32u) {
            return fail("matrix shape mismatch against split-interface guard");
        }
        if (meta.payload_words_2b != 64u) {
            return fail("payload_words mismatch against split-interface guard");
        }
        if (meta.num_weights != (meta.rows * meta.cols)) {
            return fail("metadata num_weights mismatch");
        }

        std::vector<aecct::u32_t> sram_ref((uint32_t)sram_map::SRAM_WORDS_TOTAL);
        for (uint32_t i = 0u; i < (uint32_t)sram_ref.size(); ++i) {
            sram_ref[i] = (aecct::u32_t)0u;
        }

        const uint32_t param_base_word = (uint32_t)sram_map::PARAM_BASE_DEFAULT;
        const uint32_t x_row_base = (uint32_t)sram_map::BASE_X_WORK_W;
        const uint32_t out_row_base = (uint32_t)sram_map::BASE_SCR_K_W;
        const uint32_t out_act_q_row_base = out_row_base + 64u;

        aecct::u32_t x_row[aecct::kTernaryLiveL0WqCols];
        aecct::u32_t payload_words[aecct::kTernaryLiveL0WqPayloadWords];
        aecct::u32_t out_row[aecct::kTernaryLiveL0WqRows];
        aecct::u32_t out_act_q_row[aecct::kTernaryLiveL0WqRows];
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WqCols; ++i) {
            x_row[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WqPayloadWords; ++i) {
            payload_words[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WqRows; ++i) {
            out_row[i] = (aecct::u32_t)0u;
            out_act_q_row[i] = (aecct::u32_t)0u;
        }

        fill_x_row_sram(sram_ref.data(), x_row_base, meta.cols);
        fill_x_row_split(x_row, meta.cols);

        if (!build_payload_for_matrix(matrix_id, payload_words, meta)) {
            return fail("failed to pack ternary payload");
        }
        const ParamMeta payload_meta = kParamMeta[meta.weight_param_id];
        const ParamMeta inv_meta = kParamMeta[meta.inv_sw_param_id];
        for (uint32_t i = 0u; i < meta.payload_words_2b; ++i) {
            sram_ref[param_base_word + payload_meta.offset_w + i] = payload_words[i];
        }

        double inv_sw_scale = 0.0;
        if (!matrix_inv_sw(matrix_id, inv_sw_scale)) {
            return fail("failed to map inv_s_w source");
        }
        const uint32_t expect_inv_sw_bits = (uint32_t)aecct::fp32_bits_from_double(1.0 / inv_sw_scale).to_uint();
        sram_ref[param_base_word + inv_meta.offset_w] = (aecct::u32_t)expect_inv_sw_bits;

        aecct::u32_t inv_ref = (aecct::u32_t)0u;
        if (!call_direct_kernel(
                matrix_id,
                sram_ref.data(),
                (aecct::u32_t)param_base_word,
                (aecct::u32_t)x_row_base,
                (aecct::u32_t)out_row_base,
                (aecct::u32_t)out_act_q_row_base,
                inv_ref)) {
            return fail("direct kernel call returned false");
        }

        aecct::u32_t inv_top = (aecct::u32_t)0u;
        if (!call_split_top(
                matrix_id,
                x_row,
                payload_words,
                (aecct::u32_t)expect_inv_sw_bits,
                out_row,
                out_act_q_row,
                inv_top)) {
            return fail("split-interface top run() returned false");
        }

        if (compare_exact(
                "split top inv vs direct",
                (uint32_t)inv_top.to_uint(),
                (uint32_t)inv_ref.to_uint(),
                0u) != 0) {
            return 1;
        }
        if (compare_exact(
                "split top inv vs expected",
                (uint32_t)inv_top.to_uint(),
                expect_inv_sw_bits,
                0u) != 0) {
            return 1;
        }
        for (uint32_t out_idx = 0u; out_idx < meta.rows; ++out_idx) {
            if (compare_exact(
                    "split top out vs direct",
                    (uint32_t)out_row[out_idx].to_uint(),
                    (uint32_t)sram_ref[out_row_base + out_idx].to_uint(),
                    out_idx) != 0) {
                return 1;
            }
            if (compare_exact(
                    "split top out_act_q vs direct",
                    (uint32_t)out_act_q_row[out_idx].to_uint(),
                    (uint32_t)sram_ref[out_act_q_row_base + out_idx].to_uint(),
                    out_idx) != 0) {
                return 1;
            }
        }
        std::printf("%s\n", pass_message);

        const uint32_t ref_out_idx = (meta.rows > 7u) ? 7u : 0u;
        const aecct::quant_acc_t inv_sw_q = inv_sw_from_bits(expect_inv_sw_bits);
        const uint32_t ref_expect_bits =
            compute_expect_q_elem_bits_manual(matrix_id, x_row, meta.cols, ref_out_idx, inv_sw_q);
        const uint32_t ref_got_bits = (uint32_t)out_row[ref_out_idx].to_uint();
        if (compare_exact(
                "split top independent reference one-index",
                ref_got_bits,
                ref_expect_bits,
                ref_out_idx) != 0) {
            return 1;
        }
        return 0;
    }

    static bool call_direct_kernel(
        uint32_t matrix_id,
        aecct::u32_t* sram,
        aecct::u32_t param_base_word,
        aecct::u32_t x_row_base_word,
        aecct::u32_t out_row_base_word,
        aecct::u32_t out_act_q_row_base_word,
        aecct::u32_t& out_inv_sw_bits
    ) {
        switch (matrix_id) {
            case (uint32_t)QLM_L0_WQ:
                return aecct::ternary_live_l0_wq_materialize_row_kernel(
                    sram, param_base_word, x_row_base_word, out_row_base_word, out_act_q_row_base_word, out_inv_sw_bits);
            case (uint32_t)QLM_L0_WK:
                return aecct::ternary_live_l0_wk_materialize_row_kernel(
                    sram, param_base_word, x_row_base_word, out_row_base_word, out_act_q_row_base_word, out_inv_sw_bits);
            case (uint32_t)QLM_L0_WV:
                return aecct::ternary_live_l0_wv_materialize_row_kernel(
                    sram, param_base_word, x_row_base_word, out_row_base_word, out_act_q_row_base_word, out_inv_sw_bits);
            default:
                return false;
        }
    }

    static bool call_split_top(
        uint32_t matrix_id,
        const aecct::u32_t* x_row,
        const aecct::u32_t* payload_words,
        aecct::u32_t inv_sw_bits,
        aecct::u32_t* out_row,
        aecct::u32_t* out_act_q_row,
        aecct::u32_t& out_inv_sw_bits
    ) {
        switch (matrix_id) {
            case (uint32_t)QLM_L0_WQ: {
                aecct::TernaryLiveL0WqRowTop dut;
                return dut.run(
                    x_row, payload_words, inv_sw_bits, out_row, out_act_q_row, out_inv_sw_bits);
            }
            case (uint32_t)QLM_L0_WK: {
                aecct::TernaryLiveL0WkRowTop dut;
                return dut.run(
                    x_row, payload_words, inv_sw_bits, out_row, out_act_q_row, out_inv_sw_bits);
            }
            case (uint32_t)QLM_L0_WV: {
                aecct::TernaryLiveL0WvRowTop dut;
                return dut.run(
                    x_row, payload_words, inv_sw_bits, out_row, out_act_q_row, out_inv_sw_bits);
            }
            default:
                return false;
        }
    }

    static int fail(const char* msg) {
        std::printf("[p11l_c][FAIL] %s\n", msg);
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
            std::printf("[p11l_c][FAIL] %s mismatch idx=%u got=0x%08X expect=0x%08X got_f=%g expect_f=%g\n",
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
        switch (matrix_id) {
            case (uint32_t)QLM_L0_WQ:
                out_w = w_decoder_layers_0_self_attn_linears_0_weight[elem_idx];
                return true;
            case (uint32_t)QLM_L0_WK:
                out_w = w_decoder_layers_0_self_attn_linears_1_weight[elem_idx];
                return true;
            case (uint32_t)QLM_L0_WV:
                out_w = w_decoder_layers_0_self_attn_linears_2_weight[elem_idx];
                return true;
            default:
                return false;
        }
    }

    static bool matrix_inv_sw(uint32_t matrix_id, double& out_inv_sw) {
        switch (matrix_id) {
            case (uint32_t)QLM_L0_WQ:
                out_inv_sw = w_decoder_layers_0_self_attn_linears_0_s_w[0];
                return true;
            case (uint32_t)QLM_L0_WK:
                out_inv_sw = w_decoder_layers_0_self_attn_linears_1_s_w[0];
                return true;
            case (uint32_t)QLM_L0_WV:
                out_inv_sw = w_decoder_layers_0_self_attn_linears_2_s_w[0];
                return true;
            default:
                return false;
        }
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

    static void fill_x_row_sram(aecct::u32_t* sram, uint32_t x_row_base, uint32_t cols) {
        for (uint32_t i = 0u; i < cols; ++i) {
            const float x = (float)((int)(i % 9u) - 4) * 0.125f;
            sram[x_row_base + i] = (aecct::u32_t)f32_to_bits(x);
        }
    }

    static void fill_x_row_split(aecct::u32_t x_row[aecct::kTernaryLiveL0WqCols], uint32_t cols) {
        for (uint32_t i = 0u; i < cols; ++i) {
            const float x = (float)((int)(i % 9u) - 4) * 0.125f;
            x_row[i] = (aecct::u32_t)f32_to_bits(x);
        }
    }

    static aecct::quant_acc_t inv_sw_from_bits(uint32_t bits) {
        aecct::fp32_t inv_sw_fp = aecct::fp32_from_bits((aecct::u32_t)bits);
        return inv_sw_fp.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
    }

    static uint32_t compute_expect_q_elem_bits_manual(
        uint32_t matrix_id,
        const aecct::u32_t x_row[aecct::kTernaryLiveL0WqCols],
        uint32_t cols,
        uint32_t out_idx,
        aecct::quant_acc_t inv_sw
    ) {
        aecct::quant_acc_t acc = 0;
        const uint32_t w_row_base = out_idx * cols;
        for (uint32_t in = 0u; in < cols; ++in) {
            const uint32_t x_bits = (uint32_t)x_row[in].to_uint();
            const aecct::quant_act_t x = aecct::quant_act_from_bits((aecct::u32_t)x_bits);
            double w = 0.0;
            if (!matrix_weight_at(matrix_id, w_row_base + in, w)) {
                return 0u;
            }
            const aecct::quant_w_t w_q = aecct::quant_w_t((int)w);
            acc += aecct::quant_acc_t(x) * aecct::quant_acc_t(w_q);
        }
        return (uint32_t)aecct::quant_bits_from_acc(acc / inv_sw).to_uint();
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbTernaryLiveLeafTopP11LC tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}
