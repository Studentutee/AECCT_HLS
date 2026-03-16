// tb_ternary_live_family_source_integration_smoke_p11n.cpp
// P00-011N: WK/WV source-side integration slice smoke with family unified dual-binary signature validation.

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
    std::printf("[p11n][FAIL] %s\n", msg);
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
        std::printf("[p11n][FAIL] %s mismatch idx=%u got=0x%08X expect=0x%08X got_f=%g expect_f=%g\n",
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
    for (uint32_t i = 0; i < 193u; ++i) {
        sram[base + i] = (aecct::u32_t)0u;
    }
}

static void fill_x_row(aecct::u32_t* sram, uint32_t x_base, uint32_t cols) {
    for (uint32_t i = 0u; i < cols; ++i) {
        const float x = (float)((int)(i % 13u) - 6) * 0.125f;
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

template <uint32_t PayloadWords>
static bool build_payload_for_matrix(
    uint32_t matrix_id,
    aecct::u32_t (&payload_words)[PayloadWords],
    const QuantLinearMeta& meta
) {
    for (uint32_t i = 0u; i < PayloadWords; ++i) {
        payload_words[i] = (aecct::u32_t)0u;
    }
    for (uint32_t out = 0u; out < meta.rows; ++out) {
        for (uint32_t in = 0u; in < meta.cols; ++in) {
            const uint32_t elem_idx = out * meta.cols + in;
            const uint32_t word_idx = (elem_idx >> 4);
            const uint32_t slot = (elem_idx & 15u);
            if (word_idx >= PayloadWords || word_idx >= meta.payload_words_2b) {
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

template <uint32_t PayloadWords>
static bool write_payload_and_inv_to_sram(
    aecct::u32_t* sram,
    uint32_t param_base_word,
    uint32_t matrix_id,
    const QuantLinearMeta& meta,
    const aecct::u32_t (&payload_words)[PayloadWords],
    uint32_t& out_inv_sw_bits
) {
    const ParamMeta payload_meta = kParamMeta[meta.weight_param_id];
    const ParamMeta inv_meta = kParamMeta[meta.inv_sw_param_id];
    if (payload_meta.len_w < meta.payload_words_2b || inv_meta.len_w == 0u) {
        return false;
    }
    for (uint32_t i = 0u; i < meta.payload_words_2b; ++i) {
        sram[param_base_word + payload_meta.offset_w + i] = payload_words[i];
    }
    double inv_sw_scale = 0.0;
    if (!matrix_inv_sw(matrix_id, inv_sw_scale)) {
        return false;
    }
    out_inv_sw_bits = (uint32_t)aecct::fp32_bits_from_double(1.0 / inv_sw_scale).to_uint();
    sram[param_base_word + inv_meta.offset_w] = (aecct::u32_t)out_inv_sw_bits;
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

static void compute_family_signatures(
    const aecct::u32_t* sram,
    const aecct::AttnScratch& sc,
    uint32_t tensor_words,
    uint64_t& out_q_sig,
    uint64_t& out_k_sig,
    uint64_t& out_v_sig
) {
    out_q_sig = 1469598103934665603ull;
    out_k_sig = 1469598103934665603ull;
    out_v_sig = 1469598103934665603ull;
    const uint32_t q_base = (uint32_t)sc.q_base_word.to_uint();
    const uint32_t k_base = (uint32_t)sc.k_base_word.to_uint();
    const uint32_t v_base = (uint32_t)sc.v_base_word.to_uint();
    const uint32_t q_act_q_base = (uint32_t)sc.q_act_q_base_word.to_uint();
    const uint32_t k_act_q_base = (uint32_t)sc.k_act_q_base_word.to_uint();
    const uint32_t v_act_q_base = (uint32_t)sc.v_act_q_base_word.to_uint();
    for (uint32_t i = 0u; i < tensor_words; ++i) {
        out_q_sig = fnv1a_u32(out_q_sig, (uint32_t)sram[q_base + i].to_uint());
        out_q_sig = fnv1a_u32(out_q_sig, (uint32_t)sram[q_act_q_base + i].to_uint());
        out_k_sig = fnv1a_u32(out_k_sig, (uint32_t)sram[k_base + i].to_uint());
        out_k_sig = fnv1a_u32(out_k_sig, (uint32_t)sram[k_act_q_base + i].to_uint());
        out_v_sig = fnv1a_u32(out_v_sig, (uint32_t)sram[v_base + i].to_uint());
        out_v_sig = fnv1a_u32(out_v_sig, (uint32_t)sram[v_act_q_base + i].to_uint());
    }
    out_q_sig = fnv1a_u32(out_q_sig, (uint32_t)sram[(uint32_t)sc.q_sx_base_word.to_uint()].to_uint());
}

static int verify_wkv_fallback(
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
        if (compare_exact("WK fallback vs x", (uint32_t)sram[k_base + i].to_uint(), x_bits, i) != 0) {
            return 1;
        }
        if (compare_exact("WK_act_q fallback vs x", (uint32_t)sram[k_act_q_base + i].to_uint(), x_bits, i) != 0) {
            return 1;
        }
        if (compare_exact("WV fallback vs x", (uint32_t)sram[v_base + i].to_uint(), x_bits, i) != 0) {
            return 1;
        }
        if (compare_exact("WV_act_q fallback vs x", (uint32_t)sram[v_act_q_base + i].to_uint(), x_bits, i) != 0) {
            return 1;
        }
    }
    return 0;
}

static bool validate_meta(
    const QuantLinearMeta& meta,
    uint32_t matrix_id,
    uint32_t rows,
    uint32_t cols,
    uint32_t payload_words
) {
    if (meta.matrix_id != matrix_id) {
        return false;
    }
    if (meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
        return false;
    }
    if (meta.rows != rows || meta.cols != cols) {
        return false;
    }
    if (meta.num_weights != (rows * cols)) {
        return false;
    }
    if (meta.payload_words_2b != payload_words) {
        return false;
    }
    if (meta.last_word_valid_count == 0u || meta.last_word_valid_count > 16u) {
        return false;
    }
    return true;
}

static int run_all() {
    const QuantLinearMeta wq_meta = kQuantLinearMeta[(uint32_t)QLM_L0_WQ];
    const QuantLinearMeta wk_meta = kQuantLinearMeta[(uint32_t)QLM_L0_WK];
    const QuantLinearMeta wv_meta = kQuantLinearMeta[(uint32_t)QLM_L0_WV];
    if (!validate_meta(
            wq_meta,
            (uint32_t)QLM_L0_WQ,
            aecct::kTernaryLiveL0WqRows,
            aecct::kTernaryLiveL0WqCols,
            aecct::kTernaryLiveL0WqPayloadWords)) {
        return fail("L0_WQ metadata mismatch");
    }
    if (!validate_meta(
            wk_meta,
            (uint32_t)QLM_L0_WK,
            aecct::kTernaryLiveL0WkRows,
            aecct::kTernaryLiveL0WkCols,
            aecct::kTernaryLiveL0WkPayloadWords)) {
        return fail("L0_WK metadata mismatch");
    }
    if (!validate_meta(
            wv_meta,
            (uint32_t)QLM_L0_WV,
            aecct::kTernaryLiveL0WvRows,
            aecct::kTernaryLiveL0WvCols,
            aecct::kTernaryLiveL0WvPayloadWords)) {
        return fail("L0_WV metadata mismatch");
    }

    std::vector<aecct::u32_t> sram((uint32_t)sram_map::SRAM_WORDS_TOTAL);
    for (uint32_t i = 0u; i < (uint32_t)sram.size(); ++i) {
        sram[i] = (aecct::u32_t)0u;
    }

    const uint32_t param_base_word = (uint32_t)sram_map::PARAM_BASE_DEFAULT;
    const uint32_t x_base = (uint32_t)aecct::ATTN_X_IN_BASE_WORD_DEFAULT;
    const aecct::AttnScratch sc = make_tb_attn_scratch((uint32_t)sram_map::BASE_SCR_K_W);

    aecct::AttnCfg cfg;
    cfg.token_count = (aecct::u32_t)1u;
    cfg.d_model = (aecct::u32_t)wq_meta.cols;
    cfg.n_heads = (aecct::u32_t)1u;
    cfg.d_head = (aecct::u32_t)wq_meta.cols;
    const uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    const uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    const uint32_t tensor_words = token_count * d_model;

    aecct::u32_t wq_payload[aecct::kTernaryLiveL0WqPayloadWords];
    aecct::u32_t wk_payload[aecct::kTernaryLiveL0WkPayloadWords];
    aecct::u32_t wv_payload[aecct::kTernaryLiveL0WvPayloadWords];
    if (!build_payload_for_matrix((uint32_t)QLM_L0_WQ, wq_payload, wq_meta)) {
        return fail("failed to build L0_WQ payload");
    }
    if (!build_payload_for_matrix((uint32_t)QLM_L0_WK, wk_payload, wk_meta)) {
        return fail("failed to build L0_WK payload");
    }
    if (!build_payload_for_matrix((uint32_t)QLM_L0_WV, wv_payload, wv_meta)) {
        return fail("failed to build L0_WV payload");
    }

    uint32_t wq_inv_sw_bits = 0u;
    uint32_t wk_inv_sw_bits = 0u;
    uint32_t wv_inv_sw_bits = 0u;
    if (!write_payload_and_inv_to_sram(
            sram.data(),
            param_base_word,
            (uint32_t)QLM_L0_WQ,
            wq_meta,
            wq_payload,
            wq_inv_sw_bits)) {
        return fail("failed to write L0_WQ payload/inv_s_w");
    }
    if (!write_payload_and_inv_to_sram(
            sram.data(),
            param_base_word,
            (uint32_t)QLM_L0_WK,
            wk_meta,
            wk_payload,
            wk_inv_sw_bits)) {
        return fail("failed to write L0_WK payload/inv_s_w");
    }
    if (!write_payload_and_inv_to_sram(
            sram.data(),
            param_base_word,
            (uint32_t)QLM_L0_WV,
            wv_meta,
            wv_payload,
            wv_inv_sw_bits)) {
        return fail("failed to write L0_WV payload/inv_s_w");
    }

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

    uint64_t q_sig = 0u;
    uint64_t k_sig = 0u;
    uint64_t v_sig = 0u;
    compute_family_signatures(sram.data(), sc, tensor_words, q_sig, k_sig, v_sig);
    std::printf("[p11n][WK_SIG] K=0x%016llX\n", (unsigned long long)k_sig);
    std::printf("[p11n][WV_SIG] V=0x%016llX\n", (unsigned long long)v_sig);
    std::printf("[p11n][WQ_SIG] Q=0x%016llX\n", (unsigned long long)q_sig);

#if defined(AECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE)
    aecct::u32_t x_row_wk[aecct::kTernaryLiveL0WkCols];
    aecct::u32_t wk_ref_out[aecct::kTernaryLiveL0WkRows];
    aecct::u32_t wk_ref_out_act_q[aecct::kTernaryLiveL0WkRows];
    for (uint32_t in = 0u; in < aecct::kTernaryLiveL0WkCols; ++in) {
        x_row_wk[in] = sram[x_base + in];
    }
    aecct::u32_t wk_ref_inv_sw_bits = (aecct::u32_t)0u;
    aecct::TernaryLiveL0WkRowTop wk_top;
    if (!wk_top.run(
            x_row_wk,
            wk_payload,
            (aecct::u32_t)wk_inv_sw_bits,
            wk_ref_out,
            wk_ref_out_act_q,
            wk_ref_inv_sw_bits)) {
        return fail("TernaryLiveL0WkRowTop::run failed in macro build");
    }
    if (compare_exact(
            "WK inv_s_w split-top direct",
            (uint32_t)wk_ref_inv_sw_bits.to_uint(),
            wk_inv_sw_bits,
            0u) != 0) {
        return 1;
    }
    const uint32_t k_base = (uint32_t)sc.k_base_word.to_uint();
    const uint32_t k_act_q_base = (uint32_t)sc.k_act_q_base_word.to_uint();
    for (uint32_t out = 0u; out < aecct::kTernaryLiveL0WkRows; ++out) {
        if (compare_exact(
                "WK integration vs top direct",
                (uint32_t)sram[k_base + out].to_uint(),
                (uint32_t)wk_ref_out[out].to_uint(),
                out) != 0) {
            return 1;
        }
        if (compare_exact(
                "WK_act_q integration vs top direct",
                (uint32_t)sram[k_act_q_base + out].to_uint(),
                (uint32_t)wk_ref_out_act_q[out].to_uint(),
                out) != 0) {
            return 1;
        }
    }
    std::printf("[p11n][PASS] source-side WK integration path exact-match equivalent to split-interface local top\n");

    aecct::u32_t x_row_wv[aecct::kTernaryLiveL0WvCols];
    aecct::u32_t wv_ref_out[aecct::kTernaryLiveL0WvRows];
    aecct::u32_t wv_ref_out_act_q[aecct::kTernaryLiveL0WvRows];
    for (uint32_t in = 0u; in < aecct::kTernaryLiveL0WvCols; ++in) {
        x_row_wv[in] = sram[x_base + in];
    }
    aecct::u32_t wv_ref_inv_sw_bits = (aecct::u32_t)0u;
    aecct::TernaryLiveL0WvRowTop wv_top;
    if (!wv_top.run(
            x_row_wv,
            wv_payload,
            (aecct::u32_t)wv_inv_sw_bits,
            wv_ref_out,
            wv_ref_out_act_q,
            wv_ref_inv_sw_bits)) {
        return fail("TernaryLiveL0WvRowTop::run failed in macro build");
    }
    if (compare_exact(
            "WV inv_s_w split-top direct",
            (uint32_t)wv_ref_inv_sw_bits.to_uint(),
            wv_inv_sw_bits,
            0u) != 0) {
        return 1;
    }
    const uint32_t v_base = (uint32_t)sc.v_base_word.to_uint();
    const uint32_t v_act_q_base = (uint32_t)sc.v_act_q_base_word.to_uint();
    for (uint32_t out = 0u; out < aecct::kTernaryLiveL0WvRows; ++out) {
        if (compare_exact(
                "WV integration vs top direct",
                (uint32_t)sram[v_base + out].to_uint(),
                (uint32_t)wv_ref_out[out].to_uint(),
                out) != 0) {
            return 1;
        }
        if (compare_exact(
                "WV_act_q integration vs top direct",
                (uint32_t)sram[v_act_q_base + out].to_uint(),
                (uint32_t)wv_ref_out_act_q[out].to_uint(),
                out) != 0) {
            return 1;
        }
    }
    std::printf("[p11n][PASS] source-side WV integration path exact-match equivalent to split-interface local top\n");
#endif

    for (uint32_t i = 0u; i < (uint32_t)sram.size(); ++i) {
        sram[i] = (aecct::u32_t)0u;
    }
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
    if (verify_wkv_fallback(sram.data(), x_base, sc, tensor_words) != 0) {
        return 1;
    }
    std::printf("[p11n][PASS] WK/WV fallback retained under WK/WV-only integration slice\n");

    std::printf("PASS: tb_ternary_live_family_source_integration_smoke_p11n\n");
    return 0;
}

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    const int rc = run_all();
    CCS_RETURN(rc);
}
