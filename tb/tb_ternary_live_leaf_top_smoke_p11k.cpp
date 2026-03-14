// tb_ternary_live_leaf_top_smoke_p11k.cpp
// P00-011K: wrapper smoke for local HLS top shim over P11J leaf kernel.

#include <cstdio>
#include <cstdlib>
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

namespace {

static void fail(const char* msg) {
    std::printf("[p11k][FAIL] %s\n", msg);
    std::exit(1);
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

static void compare_exact_or_die(const char* label, uint32_t got_bits, uint32_t expect_bits, uint32_t idx) {
    if (got_bits != expect_bits) {
        std::printf("[p11k][FAIL] %s mismatch idx=%u got=0x%08X expect=0x%08X got_f=%g expect_f=%g\n",
                    label,
                    (unsigned)idx,
                    (unsigned)got_bits,
                    (unsigned)expect_bits,
                    (double)bits_to_f32(got_bits),
                    (double)bits_to_f32(expect_bits));
        std::exit(1);
    }
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

static void build_l0_wq_payload_or_die(const QuantLinearMeta& meta, std::vector<uint32_t>& payload_words) {
    payload_words.assign(meta.payload_words_2b, 0u);
    for (uint32_t out = 0u; out < meta.rows; ++out) {
        for (uint32_t in = 0u; in < meta.cols; ++in) {
            const uint32_t elem_idx = out * meta.cols + in;
            const uint32_t word_idx = (elem_idx >> 4);
            const uint32_t slot = (elem_idx & 15u);
            if (word_idx >= payload_words.size()) {
                fail("payload word index overflow while packing L0_WQ");
            }
            const double w = w_decoder_layers_0_self_attn_linears_0_weight[elem_idx];
            const uint32_t code = encode_ternary_code(w);
            if (code == (uint32_t)TERNARY_CODE_RSVD) {
                fail("reserved ternary code generated while packing L0_WQ");
            }
            payload_words[word_idx] |= (code & 0x3u) << (slot * 2u);
        }
    }
}

static void fill_x_row(aecct::u32_t* sram, uint32_t x_row_base, uint32_t cols) {
    for (uint32_t i = 0u; i < cols; ++i) {
        const float x = (float)((int)(i % 9u) - 4) * 0.125f;
        sram[x_row_base + i] = (aecct::u32_t)f32_to_bits(x);
    }
}

static aecct::quant_acc_t inv_sw_from_bits(uint32_t bits) {
    aecct::fp32_t inv_sw_fp = aecct::fp32_from_bits((aecct::u32_t)bits);
    return inv_sw_fp.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
}

static uint32_t compute_expect_q_elem_bits_manual(
    const aecct::u32_t* sram,
    uint32_t x_row_base,
    uint32_t cols,
    uint32_t out_idx,
    aecct::quant_acc_t inv_sw
) {
    aecct::quant_acc_t acc = 0;
    const uint32_t w_row_base = out_idx * cols;
    for (uint32_t in = 0u; in < cols; ++in) {
        const uint32_t x_bits = (uint32_t)sram[x_row_base + in].to_uint();
        const aecct::quant_act_t x = aecct::quant_act_from_bits((aecct::u32_t)x_bits);
        const aecct::quant_w_t w = aecct::quant_w_t((int)w_decoder_layers_0_self_attn_linears_0_weight[w_row_base + in]);
        acc += aecct::quant_acc_t(x) * aecct::quant_acc_t(w);
    }
    return (uint32_t)aecct::quant_bits_from_acc(acc / inv_sw).to_uint();
}

} // namespace

int main() {
    const QuantLinearMeta wq_meta = kQuantLinearMeta[(uint32_t)QLM_L0_WQ];
    if (wq_meta.matrix_id != (uint32_t)QLM_L0_WQ) {
        fail("L0_WQ metadata matrix_id mismatch");
    }
    if (wq_meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
        fail("L0_WQ metadata layout mismatch");
    }
    if (wq_meta.num_weights != (wq_meta.rows * wq_meta.cols)) {
        fail("L0_WQ metadata num_weights mismatch");
    }

    std::vector<aecct::u32_t> sram_ref((uint32_t)sram_map::SRAM_WORDS_TOTAL);
    std::vector<aecct::u32_t> sram_top((uint32_t)sram_map::SRAM_WORDS_TOTAL);
    for (uint32_t i = 0u; i < (uint32_t)sram_ref.size(); ++i) {
        sram_ref[i] = (aecct::u32_t)0u;
        sram_top[i] = (aecct::u32_t)0u;
    }

    const uint32_t param_base_word = (uint32_t)sram_map::PARAM_BASE_DEFAULT;
    const uint32_t x_row_base = (uint32_t)sram_map::BASE_X_WORK_W;
    const uint32_t out_row_base = (uint32_t)sram_map::BASE_SCR_K_W;
    const uint32_t out_act_q_row_base = out_row_base + 64u;
    fill_x_row(sram_ref.data(), x_row_base, wq_meta.cols);
    fill_x_row(sram_top.data(), x_row_base, wq_meta.cols);

    std::vector<uint32_t> wq_payload_words;
    build_l0_wq_payload_or_die(wq_meta, wq_payload_words);
    if (wq_payload_words.size() != wq_meta.payload_words_2b) {
        fail("packed payload size mismatch with L0_WQ metadata");
    }

    const ParamMeta wq_payload_meta = kParamMeta[wq_meta.weight_param_id];
    const ParamMeta wq_inv_meta = kParamMeta[wq_meta.inv_sw_param_id];
    for (uint32_t i = 0u; i < wq_meta.payload_words_2b; ++i) {
        sram_ref[param_base_word + wq_payload_meta.offset_w + i] = (aecct::u32_t)wq_payload_words[i];
        sram_top[param_base_word + wq_payload_meta.offset_w + i] = (aecct::u32_t)wq_payload_words[i];
    }
    const uint32_t expect_inv_sw_bits =
        (uint32_t)aecct::fp32_bits_from_double(1.0 / w_decoder_layers_0_self_attn_linears_0_s_w[0]).to_uint();
    sram_ref[param_base_word + wq_inv_meta.offset_w] = (aecct::u32_t)expect_inv_sw_bits;
    sram_top[param_base_word + wq_inv_meta.offset_w] = (aecct::u32_t)expect_inv_sw_bits;

    aecct::u32_t inv_ref = (aecct::u32_t)0u;
    if (!aecct::ternary_live_l0_wq_materialize_row_kernel(
            sram_ref.data(),
            (aecct::u32_t)param_base_word,
            (aecct::u32_t)x_row_base,
            (aecct::u32_t)out_row_base,
            (aecct::u32_t)out_act_q_row_base,
            inv_ref)) {
        fail("direct P11J kernel call returned false");
    }

    aecct::u32_t inv_top = (aecct::u32_t)0u;
    aecct::TernaryLiveL0WqRowTop dut;
    if (!dut.run(
            sram_top.data(),
            (aecct::u32_t)param_base_word,
            (aecct::u32_t)x_row_base,
            (aecct::u32_t)out_row_base,
            (aecct::u32_t)out_act_q_row_base,
            inv_top)) {
        fail("wrapper top run() returned false");
    }

    compare_exact_or_die("wrapper inv vs direct", (uint32_t)inv_top.to_uint(), (uint32_t)inv_ref.to_uint(), 0u);
    compare_exact_or_die("wrapper inv vs expected", (uint32_t)inv_top.to_uint(), expect_inv_sw_bits, 0u);
    for (uint32_t out = 0u; out < wq_meta.rows; ++out) {
        compare_exact_or_die(
            "wrapper out vs direct",
            (uint32_t)sram_top[out_row_base + out].to_uint(),
            (uint32_t)sram_ref[out_row_base + out].to_uint(),
            out);
        compare_exact_or_die(
            "wrapper out_act_q vs direct",
            (uint32_t)sram_top[out_act_q_row_base + out].to_uint(),
            (uint32_t)sram_ref[out_act_q_row_base + out].to_uint(),
            out);
    }
    std::printf("[p11k][PASS] wrapper top run() exact-match equivalent to direct P11J kernel output\n");

    const uint32_t ref_out_idx = (wq_meta.rows > 7u) ? 7u : 0u;
    const aecct::quant_acc_t inv_sw = inv_sw_from_bits(expect_inv_sw_bits);
    const uint32_t ref_expect_bits = compute_expect_q_elem_bits_manual(
        sram_top.data(),
        x_row_base,
        wq_meta.cols,
        ref_out_idx,
        inv_sw);
    const uint32_t ref_got_bits = (uint32_t)sram_top[out_row_base + ref_out_idx].to_uint();
    compare_exact_or_die("wrapper independent reference one-index", ref_got_bits, ref_expect_bits, ref_out_idx);
    std::printf("[p11k][PASS] wrapper top independent exact-bit reference matched for out_idx=%u\n",
                (unsigned)ref_out_idx);

    std::printf("PASS: tb_ternary_live_leaf_top_smoke_p11k\n");
    return 0;
}
