// tb_ternary_live_source_integration_smoke_p11m.cpp
// P00-011M: WQ source-side integration slice smoke with baseline/macro dual-build support.

#include <cstdio>
#include <cstdint>
#include <vector>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "QuantDesc.h"
#include "blocks/AttnLayer0.h"
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

static int fail(const char* msg) {
    std::printf("[p11m][FAIL] %s\n", msg);
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
        std::printf("[p11m][FAIL] %s mismatch idx=%u got=0x%08X expect=0x%08X got_f=%g expect_f=%g\n",
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

static aecct::AttnScratch make_tb_attn_scratch(uint32_t base_word) {
    aecct::AttnScratch sc;
    sc.q_base_word = (aecct::u32_t)(base_word + 0u);
    sc.k_base_word = (aecct::u32_t)(base_word + 32u);
    sc.v_base_word = (aecct::u32_t)(base_word + 64u);
    sc.score_base_word = 0;
    sc.softmax_base_word = 0;
    sc.pre_concat_base_word = 0;
    sc.post_concat_base_word = 0;
    sc.q_act_q_base_word = (aecct::u32_t)(base_word + 96u);
    sc.k_act_q_base_word = (aecct::u32_t)(base_word + 128u);
    sc.v_act_q_base_word = (aecct::u32_t)(base_word + 160u);
    sc.q_sx_base_word = (aecct::u32_t)(base_word + 192u);
    return sc;
}

static void clear_attn_outputs(aecct::u32_t* sram, const aecct::AttnScratch& sc) {
    const uint32_t base = (uint32_t)sc.q_base_word.to_uint();
    for (uint32_t i = 0u; i < 193u; ++i) {
        sram[base + i] = (aecct::u32_t)0u;
    }
}

static void fill_x_row(aecct::u32_t* sram, uint32_t x_base, uint32_t cols) {
    for (uint32_t i = 0u; i < cols; ++i) {
        const float x = (float)((int)(i % 11u) - 5) * 0.125f;
        sram[x_base + i] = (aecct::u32_t)f32_to_bits(x);
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

static uint64_t fnv1a_u32(uint64_t hash, uint32_t word) {
    static const uint64_t kFnvPrime = 1099511628211ull;
    hash ^= (uint64_t)(word & 0xFFu);
    hash *= kFnvPrime;
    hash ^= (uint64_t)((word >> 8) & 0xFFu);
    hash *= kFnvPrime;
    hash ^= (uint64_t)((word >> 16) & 0xFFu);
    hash *= kFnvPrime;
    hash ^= (uint64_t)((word >> 24) & 0xFFu);
    hash *= kFnvPrime;
    return hash;
}

static void compute_kv_sig(
    const aecct::u32_t* sram,
    const aecct::AttnScratch& sc,
    uint32_t tensor_words,
    uint64_t& out_k_sig,
    uint64_t& out_v_sig
) {
    out_k_sig = 1469598103934665603ull;
    out_v_sig = 1469598103934665603ull;
    const uint32_t k_base = (uint32_t)sc.k_base_word.to_uint();
    const uint32_t v_base = (uint32_t)sc.v_base_word.to_uint();
    const uint32_t k_act_q_base = (uint32_t)sc.k_act_q_base_word.to_uint();
    const uint32_t v_act_q_base = (uint32_t)sc.v_act_q_base_word.to_uint();
    for (uint32_t i = 0u; i < tensor_words; ++i) {
        out_k_sig = fnv1a_u32(out_k_sig, (uint32_t)sram[k_base + i].to_uint());
        out_k_sig = fnv1a_u32(out_k_sig, (uint32_t)sram[k_act_q_base + i].to_uint());
        out_v_sig = fnv1a_u32(out_v_sig, (uint32_t)sram[v_base + i].to_uint());
        out_v_sig = fnv1a_u32(out_v_sig, (uint32_t)sram[v_act_q_base + i].to_uint());
    }
}

static int verify_kv_fallback(
    const aecct::u32_t* sram,
    uint32_t x_base,
    const aecct::AttnScratch& sc,
    uint32_t tensor_words
) {
    const uint32_t k_base = (uint32_t)sc.k_base_word.to_uint();
    const uint32_t v_base = (uint32_t)sc.v_base_word.to_uint();
    const uint32_t k_act_q_base = (uint32_t)sc.k_act_q_base_word.to_uint();
    const uint32_t v_act_q_base = (uint32_t)sc.v_act_q_base_word.to_uint();
    for (uint32_t i = 0u; i < tensor_words; ++i) {
        const uint32_t x_bits = (uint32_t)sram[x_base + i].to_uint();
        if (compare_exact("K fallback vs x", (uint32_t)sram[k_base + i].to_uint(), x_bits, i) != 0) {
            return 1;
        }
        if (compare_exact("K_act_q fallback vs x", (uint32_t)sram[k_act_q_base + i].to_uint(), x_bits, i) != 0) {
            return 1;
        }
        if (compare_exact("V fallback vs x", (uint32_t)sram[v_base + i].to_uint(), x_bits, i) != 0) {
            return 1;
        }
        if (compare_exact("V_act_q fallback vs x", (uint32_t)sram[v_act_q_base + i].to_uint(), x_bits, i) != 0) {
            return 1;
        }
    }
    return 0;
}

static int run_all() {
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

    std::vector<aecct::u32_t> sram((uint32_t)sram_map::SRAM_WORDS_TOTAL);
    for (uint32_t i = 0u; i < (uint32_t)sram.size(); ++i) {
        sram[i] = (aecct::u32_t)0u;
    }

    const uint32_t param_base_word = (uint32_t)sram_map::PARAM_BASE_DEFAULT;
    const uint32_t x_base = (uint32_t)aecct::ATTN_X_IN_BASE_WORD_DEFAULT;
    const aecct::AttnScratch sc = make_tb_attn_scratch((uint32_t)sram_map::BASE_SCR_K_W);

    aecct::u32_t payload_words[aecct::kTernaryLiveL0WqPayloadWords];
    if (!build_l0_wq_payload(payload_words, wq_meta)) {
        return fail("failed to build L0_WQ payload");
    }
    const ParamMeta wq_payload_meta = kParamMeta[wq_meta.weight_param_id];
    const ParamMeta wq_inv_meta = kParamMeta[wq_meta.inv_sw_param_id];
    for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WqPayloadWords; ++i) {
        sram[param_base_word + wq_payload_meta.offset_w + i] = payload_words[i];
    }
    const uint32_t expect_inv_sw_bits =
        (uint32_t)aecct::fp32_bits_from_double(1.0 / w_decoder_layers_0_self_attn_linears_0_s_w[0]).to_uint();
    sram[param_base_word + wq_inv_meta.offset_w] = (aecct::u32_t)expect_inv_sw_bits;

    aecct::AttnCfg cfg;
    cfg.token_count = (aecct::u32_t)1u;
    cfg.d_model = (aecct::u32_t)wq_meta.cols;
    cfg.n_heads = (aecct::u32_t)1u;
    cfg.d_head = (aecct::u32_t)wq_meta.cols;
    const uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    const uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    const uint32_t tensor_words = token_count * d_model;

    fill_x_row(sram.data(), x_base, tensor_words);
    clear_attn_outputs(sram.data(), sc);

    aecct::AttnLayer0<aecct::ATTN_STAGE_QKV>(
        sram.data(),
        cfg,
        (aecct::u32_t)x_base,
        (aecct::u32_t)aecct::ATTN_OUT_BASE_WORD_DEFAULT,
        sc,
        (aecct::u32_t)param_base_word
    );

    if (verify_kv_fallback(sram.data(), x_base, sc, tensor_words) != 0) {
        return 1;
    }

    uint64_t k_sig = 0u;
    uint64_t v_sig = 0u;
    compute_kv_sig(sram.data(), sc, tensor_words, k_sig, v_sig);
    std::printf("[p11m][KV_SIG] K=0x%016llX V=0x%016llX\n",
                (unsigned long long)k_sig,
                (unsigned long long)v_sig);

#if defined(AECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE)
    aecct::u32_t x_row[aecct::kTernaryLiveL0WqCols];
    aecct::u32_t ref_out_row[aecct::kTernaryLiveL0WqRows];
    aecct::u32_t ref_out_act_q_row[aecct::kTernaryLiveL0WqRows];
    for (uint32_t in = 0u; in < aecct::kTernaryLiveL0WqCols; ++in) {
        x_row[in] = sram[x_base + in];
    }
    aecct::u32_t ref_inv_sw_bits = (aecct::u32_t)0u;
    aecct::TernaryLiveL0WqRowTop dut;
    if (!dut.run(
            x_row,
            payload_words,
            (aecct::u32_t)expect_inv_sw_bits,
            ref_out_row,
            ref_out_act_q_row,
            ref_inv_sw_bits)) {
        return fail("TernaryLiveL0WqRowTop::run failed in macro build");
    }

    const uint32_t q_base = (uint32_t)sc.q_base_word.to_uint();
    const uint32_t q_act_q_base = (uint32_t)sc.q_act_q_base_word.to_uint();
    for (uint32_t out = 0u; out < aecct::kTernaryLiveL0WqRows; ++out) {
        if (compare_exact(
                "Q integration vs top direct",
                (uint32_t)sram[q_base + out].to_uint(),
                (uint32_t)ref_out_row[out].to_uint(),
                out) != 0) {
            return 1;
        }
        if (compare_exact(
                "Q_act_q integration vs top direct",
                (uint32_t)sram[q_act_q_base + out].to_uint(),
                (uint32_t)ref_out_act_q_row[out].to_uint(),
                out) != 0) {
            return 1;
        }
    }
    if (compare_exact(
            "Q_sx integration vs top direct",
            (uint32_t)sram[(uint32_t)sc.q_sx_base_word.to_uint()].to_uint(),
            (uint32_t)ref_inv_sw_bits.to_uint(),
            0u) != 0) {
        return 1;
    }
    std::printf("[p11m][PASS] source-side WQ integration path exact-match equivalent to split-interface local top\n");
#endif

    std::printf("[p11m][PASS] K/V fallback retained under WQ-only integration slice\n");
    std::printf("PASS: tb_ternary_live_source_integration_smoke_p11m\n");
    return 0;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    const int rc = run_all();
    CCS_RETURN(rc);
}
