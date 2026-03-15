// tb_ternary_live_leaf_top_smoke_p11l_b.cpp
// P00-011L-B: split-interface wrapper smoke for L0_WK and L0_WV local tops.

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

class TbTernaryLiveLeafTopP11LB {
public:
    int run_all() {
        const int wk_rc = run_wk_case();
        if (wk_rc != 0) {
            return wk_rc;
        }
        const int wv_rc = run_wv_case();
        if (wv_rc != 0) {
            return wv_rc;
        }
        std::printf("PASS: tb_ternary_live_leaf_top_smoke_p11l_b\n");
        return 0;
    }

private:
    static int run_wk_case() {
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
        if (wk_meta.num_weights != (wk_meta.rows * wk_meta.cols)) {
            return fail("L0_WK metadata num_weights mismatch");
        }

        std::vector<aecct::u32_t> sram_ref((uint32_t)sram_map::SRAM_WORDS_TOTAL);
        for (uint32_t i = 0u; i < (uint32_t)sram_ref.size(); ++i) {
            sram_ref[i] = (aecct::u32_t)0u;
        }

        const uint32_t param_base_word = (uint32_t)sram_map::PARAM_BASE_DEFAULT;
        const uint32_t x_row_base = (uint32_t)sram_map::BASE_X_WORK_W;
        const uint32_t out_row_base = (uint32_t)sram_map::BASE_SCR_K_W;
        const uint32_t out_act_q_row_base = out_row_base + 64u;

        aecct::u32_t x_row[aecct::kTernaryLiveL0WkCols];
        aecct::u32_t payload_words[aecct::kTernaryLiveL0WkPayloadWords];
        aecct::u32_t out_row[aecct::kTernaryLiveL0WkRows];
        aecct::u32_t out_act_q_row[aecct::kTernaryLiveL0WkRows];
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WkCols; ++i) {
            x_row[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WkPayloadWords; ++i) {
            payload_words[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WkRows; ++i) {
            out_row[i] = (aecct::u32_t)0u;
            out_act_q_row[i] = (aecct::u32_t)0u;
        }

        fill_x_row_sram(sram_ref.data(), x_row_base, wk_meta.cols);
        fill_x_row_split(x_row, wk_meta.cols);

        if (!build_payload_for_matrix((uint32_t)QLM_L0_WK, payload_words, wk_meta)) {
            return fail("failed to pack L0_WK ternary payload");
        }

        const ParamMeta wk_payload_meta = kParamMeta[wk_meta.weight_param_id];
        const ParamMeta wk_inv_meta = kParamMeta[wk_meta.inv_sw_param_id];
        for (uint32_t i = 0u; i < wk_meta.payload_words_2b; ++i) {
            sram_ref[param_base_word + wk_payload_meta.offset_w + i] = payload_words[i];
        }

        const uint32_t expect_inv_sw_bits =
            (uint32_t)aecct::fp32_bits_from_double(1.0 / w_decoder_layers_0_self_attn_linears_1_s_w[0]).to_uint();
        sram_ref[param_base_word + wk_inv_meta.offset_w] = (aecct::u32_t)expect_inv_sw_bits;

        aecct::u32_t inv_ref = (aecct::u32_t)0u;
        if (!aecct::ternary_live_l0_wk_materialize_row_kernel(
                sram_ref.data(),
                (aecct::u32_t)param_base_word,
                (aecct::u32_t)x_row_base,
                (aecct::u32_t)out_row_base,
                (aecct::u32_t)out_act_q_row_base,
                inv_ref)) {
            return fail("direct L0_WK kernel call returned false");
        }

        aecct::u32_t inv_top = (aecct::u32_t)0u;
        aecct::TernaryLiveL0WkRowTop dut;
        if (!dut.run(
                x_row,
                payload_words,
                (aecct::u32_t)expect_inv_sw_bits,
                out_row,
                out_act_q_row,
                inv_top)) {
            return fail("L0_WK split-interface top run() returned false");
        }

        if (compare_exact(
                "L0_WK split top inv vs direct",
                (uint32_t)inv_top.to_uint(),
                (uint32_t)inv_ref.to_uint(),
                0u) != 0) {
            return 1;
        }
        if (compare_exact(
                "L0_WK split top inv vs expected",
                (uint32_t)inv_top.to_uint(),
                expect_inv_sw_bits,
                0u) != 0) {
            return 1;
        }
        for (uint32_t out_idx = 0u; out_idx < wk_meta.rows; ++out_idx) {
            if (compare_exact(
                    "L0_WK split top out vs direct",
                    (uint32_t)out_row[out_idx].to_uint(),
                    (uint32_t)sram_ref[out_row_base + out_idx].to_uint(),
                    out_idx) != 0) {
                return 1;
            }
            if (compare_exact(
                    "L0_WK split top out_act_q vs direct",
                    (uint32_t)out_act_q_row[out_idx].to_uint(),
                    (uint32_t)sram_ref[out_act_q_row_base + out_idx].to_uint(),
                    out_idx) != 0) {
                return 1;
            }
        }
        std::printf("L0_WK split-interface top run() exact-match equivalent to direct kernel output\n");

        const uint32_t ref_out_idx = (wk_meta.rows > 7u) ? 7u : 0u;
        const aecct::quant_acc_t inv_sw = inv_sw_from_bits(expect_inv_sw_bits);
        const uint32_t ref_expect_bits =
            compute_expect_q_elem_bits_manual((uint32_t)QLM_L0_WK, x_row, wk_meta.cols, ref_out_idx, inv_sw);
        const uint32_t ref_got_bits = (uint32_t)out_row[ref_out_idx].to_uint();
        if (compare_exact(
                "L0_WK split top independent reference one-index",
                ref_got_bits,
                ref_expect_bits,
                ref_out_idx) != 0) {
            return 1;
        }
        return 0;
    }

    static int run_wv_case() {
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
        if (wv_meta.num_weights != (wv_meta.rows * wv_meta.cols)) {
            return fail("L0_WV metadata num_weights mismatch");
        }

        std::vector<aecct::u32_t> sram_ref((uint32_t)sram_map::SRAM_WORDS_TOTAL);
        for (uint32_t i = 0u; i < (uint32_t)sram_ref.size(); ++i) {
            sram_ref[i] = (aecct::u32_t)0u;
        }

        const uint32_t param_base_word = (uint32_t)sram_map::PARAM_BASE_DEFAULT;
        const uint32_t x_row_base = (uint32_t)sram_map::BASE_X_WORK_W;
        const uint32_t out_row_base = (uint32_t)sram_map::BASE_SCR_K_W;
        const uint32_t out_act_q_row_base = out_row_base + 64u;

        aecct::u32_t x_row[aecct::kTernaryLiveL0WvCols];
        aecct::u32_t payload_words[aecct::kTernaryLiveL0WvPayloadWords];
        aecct::u32_t out_row[aecct::kTernaryLiveL0WvRows];
        aecct::u32_t out_act_q_row[aecct::kTernaryLiveL0WvRows];
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WvCols; ++i) {
            x_row[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WvPayloadWords; ++i) {
            payload_words[i] = (aecct::u32_t)0u;
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WvRows; ++i) {
            out_row[i] = (aecct::u32_t)0u;
            out_act_q_row[i] = (aecct::u32_t)0u;
        }

        fill_x_row_sram(sram_ref.data(), x_row_base, wv_meta.cols);
        fill_x_row_split(x_row, wv_meta.cols);

        if (!build_payload_for_matrix((uint32_t)QLM_L0_WV, payload_words, wv_meta)) {
            return fail("failed to pack L0_WV ternary payload");
        }

        const ParamMeta wv_payload_meta = kParamMeta[wv_meta.weight_param_id];
        const ParamMeta wv_inv_meta = kParamMeta[wv_meta.inv_sw_param_id];
        for (uint32_t i = 0u; i < wv_meta.payload_words_2b; ++i) {
            sram_ref[param_base_word + wv_payload_meta.offset_w + i] = payload_words[i];
        }

        const uint32_t expect_inv_sw_bits =
            (uint32_t)aecct::fp32_bits_from_double(1.0 / w_decoder_layers_0_self_attn_linears_2_s_w[0]).to_uint();
        sram_ref[param_base_word + wv_inv_meta.offset_w] = (aecct::u32_t)expect_inv_sw_bits;

        aecct::u32_t inv_ref = (aecct::u32_t)0u;
        if (!aecct::ternary_live_l0_wv_materialize_row_kernel(
                sram_ref.data(),
                (aecct::u32_t)param_base_word,
                (aecct::u32_t)x_row_base,
                (aecct::u32_t)out_row_base,
                (aecct::u32_t)out_act_q_row_base,
                inv_ref)) {
            return fail("direct L0_WV kernel call returned false");
        }

        aecct::u32_t inv_top = (aecct::u32_t)0u;
        aecct::TernaryLiveL0WvRowTop dut;
        if (!dut.run(
                x_row,
                payload_words,
                (aecct::u32_t)expect_inv_sw_bits,
                out_row,
                out_act_q_row,
                inv_top)) {
            return fail("L0_WV split-interface top run() returned false");
        }

        if (compare_exact(
                "L0_WV split top inv vs direct",
                (uint32_t)inv_top.to_uint(),
                (uint32_t)inv_ref.to_uint(),
                0u) != 0) {
            return 1;
        }
        if (compare_exact(
                "L0_WV split top inv vs expected",
                (uint32_t)inv_top.to_uint(),
                expect_inv_sw_bits,
                0u) != 0) {
            return 1;
        }
        for (uint32_t out_idx = 0u; out_idx < wv_meta.rows; ++out_idx) {
            if (compare_exact(
                    "L0_WV split top out vs direct",
                    (uint32_t)out_row[out_idx].to_uint(),
                    (uint32_t)sram_ref[out_row_base + out_idx].to_uint(),
                    out_idx) != 0) {
                return 1;
            }
            if (compare_exact(
                    "L0_WV split top out_act_q vs direct",
                    (uint32_t)out_act_q_row[out_idx].to_uint(),
                    (uint32_t)sram_ref[out_act_q_row_base + out_idx].to_uint(),
                    out_idx) != 0) {
                return 1;
            }
        }
        std::printf("L0_WV split-interface top run() exact-match equivalent to direct kernel output\n");

        const uint32_t ref_out_idx = (wv_meta.rows > 7u) ? 7u : 0u;
        const aecct::quant_acc_t inv_sw = inv_sw_from_bits(expect_inv_sw_bits);
        const uint32_t ref_expect_bits =
            compute_expect_q_elem_bits_manual((uint32_t)QLM_L0_WV, x_row, wv_meta.cols, ref_out_idx, inv_sw);
        const uint32_t ref_got_bits = (uint32_t)out_row[ref_out_idx].to_uint();
        if (compare_exact(
                "L0_WV split top independent reference one-index",
                ref_got_bits,
                ref_expect_bits,
                ref_out_idx) != 0) {
            return 1;
        }
        return 0;
    }

    static int fail(const char* msg) {
        std::printf("[p11l_b][FAIL] %s\n", msg);
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
            std::printf("[p11l_b][FAIL] %s mismatch idx=%u got=0x%08X expect=0x%08X got_f=%g expect_f=%g\n",
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

    static void fill_x_row_sram(aecct::u32_t* sram, uint32_t x_row_base, uint32_t cols) {
        for (uint32_t i = 0u; i < cols; ++i) {
            const float x = (float)((int)(i % 9u) - 4) * 0.125f;
            sram[x_row_base + i] = (aecct::u32_t)f32_to_bits(x);
        }
    }

    static void fill_x_row_split(aecct::u32_t* x_row, uint32_t cols) {
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
        const aecct::u32_t* x_row,
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
    TbTernaryLiveLeafTopP11LB tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}
