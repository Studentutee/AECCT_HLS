#pragma once

#include <cstdint>
#include <vector>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "Top.h"
#include "blocks/AttnPhaseBTopManagedQkScore.h"
#include "blocks/AttnPhaseBTopManagedSoftmaxOut.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"
#include "weights.h"

namespace p11aeaf_tb {

static const uint32_t kTokenCount = 2u;
static const uint32_t kTileWords = (uint32_t)aecct::ATTN_TOP_MANAGED_TILE_WORDS;

static inline uint32_t f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = f;
    return cvt.u;
}

static inline uint32_t encode_ternary_code(double w) {
    if (w > 0.5) {
        return (uint32_t)TERNARY_CODE_POS;
    }
    if (w < -0.5) {
        return (uint32_t)TERNARY_CODE_NEG;
    }
    return (uint32_t)TERNARY_CODE_ZERO;
}

static inline bool matrix_weight_at(uint32_t matrix_id, uint32_t elem_idx, double& out_w) {
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

static inline bool matrix_inv_sw(uint32_t matrix_id, double& out_inv_sw) {
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

static inline bool build_payload_for_matrix(
    uint32_t matrix_id,
    const QuantLinearMeta& meta,
    std::vector<aecct::u32_t>& out_payload
) {
    out_payload.assign(meta.payload_words_2b, (aecct::u32_t)0u);
    for (uint32_t out = 0u; out < meta.rows; ++out) {
        for (uint32_t in = 0u; in < meta.cols; ++in) {
            const uint32_t elem_idx = out * meta.cols + in;
            const uint32_t word_idx = (elem_idx >> 4);
            const uint32_t slot = (elem_idx & 15u);
            if (word_idx >= meta.payload_words_2b) {
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
            out_payload[word_idx] = (aecct::u32_t)((uint32_t)out_payload[word_idx].to_uint() |
                ((code & 0x3u) << (slot * 2u)));
        }
    }
    return true;
}

struct QkvPayloadSet {
    std::vector<aecct::u32_t> wq_payload;
    std::vector<aecct::u32_t> wk_payload;
    std::vector<aecct::u32_t> wv_payload;
    aecct::u32_t wq_inv_sw_bits;
    aecct::u32_t wk_inv_sw_bits;
    aecct::u32_t wv_inv_sw_bits;
};

static inline bool prepare_qkv_payload_set(QkvPayloadSet& out) {
    const QuantLinearMeta wq_meta = aecct::ternary_linear_live_l0_wq_meta();
    const QuantLinearMeta wk_meta = aecct::ternary_linear_live_l0_wk_meta();
    const QuantLinearMeta wv_meta = aecct::ternary_linear_live_l0_wv_meta();
    if (!build_payload_for_matrix((uint32_t)QLM_L0_WQ, wq_meta, out.wq_payload)) {
        return false;
    }
    if (!build_payload_for_matrix((uint32_t)QLM_L0_WK, wk_meta, out.wk_payload)) {
        return false;
    }
    if (!build_payload_for_matrix((uint32_t)QLM_L0_WV, wv_meta, out.wv_payload)) {
        return false;
    }

    double wq_inv_sw = 0.0;
    double wk_inv_sw = 0.0;
    double wv_inv_sw = 0.0;
    if (!matrix_inv_sw((uint32_t)QLM_L0_WQ, wq_inv_sw)) { return false; }
    if (!matrix_inv_sw((uint32_t)QLM_L0_WK, wk_inv_sw)) { return false; }
    if (!matrix_inv_sw((uint32_t)QLM_L0_WV, wv_inv_sw)) { return false; }

    out.wq_inv_sw_bits = (aecct::u32_t)aecct::fp32_bits_from_double(1.0 / wq_inv_sw);
    out.wk_inv_sw_bits = (aecct::u32_t)aecct::fp32_bits_from_double(1.0 / wk_inv_sw);
    out.wv_inv_sw_bits = (aecct::u32_t)aecct::fp32_bits_from_double(1.0 / wv_inv_sw);
    return true;
}

static inline void load_qkv_payload_set_to_sram(
    std::vector<aecct::u32_t>& sram,
    const QkvPayloadSet& payload,
    uint32_t param_base
) {
    const QuantLinearMeta wq_meta = aecct::ternary_linear_live_l0_wq_meta();
    const QuantLinearMeta wk_meta = aecct::ternary_linear_live_l0_wk_meta();
    const QuantLinearMeta wv_meta = aecct::ternary_linear_live_l0_wv_meta();
    const ParamMeta wq_payload_meta = kParamMeta[wq_meta.weight_param_id];
    const ParamMeta wk_payload_meta = kParamMeta[wk_meta.weight_param_id];
    const ParamMeta wv_payload_meta = kParamMeta[wv_meta.weight_param_id];
    const ParamMeta wq_inv_meta = kParamMeta[wq_meta.inv_sw_param_id];
    const ParamMeta wk_inv_meta = kParamMeta[wk_meta.inv_sw_param_id];
    const ParamMeta wv_inv_meta = kParamMeta[wv_meta.inv_sw_param_id];

    for (uint32_t i = 0u; i < wq_meta.payload_words_2b; ++i) {
        sram[param_base + wq_payload_meta.offset_w + i] = payload.wq_payload[i];
    }
    for (uint32_t i = 0u; i < wk_meta.payload_words_2b; ++i) {
        sram[param_base + wk_payload_meta.offset_w + i] = payload.wk_payload[i];
    }
    for (uint32_t i = 0u; i < wv_meta.payload_words_2b; ++i) {
        sram[param_base + wv_payload_meta.offset_w + i] = payload.wv_payload[i];
    }
    sram[param_base + wq_inv_meta.offset_w] = payload.wq_inv_sw_bits;
    sram[param_base + wk_inv_meta.offset_w] = payload.wk_inv_sw_bits;
    sram[param_base + wv_inv_meta.offset_w] = payload.wv_inv_sw_bits;
}

static inline void init_x_rows(std::vector<aecct::u32_t>& sram) {
    const uint32_t x_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
    for (uint32_t t = 0u; t < kTokenCount; ++t) {
        const uint32_t row_base = x_base + t * kTileWords;
        for (uint32_t i = 0u; i < kTileWords; ++i) {
            const int32_t v = (int32_t)((t + 1u) * 37u + (i * 9u)) - 53;
            const float f = ((float)v) * 0.03125f;
            sram[row_base + i] = (aecct::u32_t)f32_to_bits(f);
        }
    }
}

static inline aecct::CfgRegs build_cfg() {
    aecct::CfgRegs cfg;
    cfg.d_model = (aecct::u32_t)kTileWords;
    cfg.n_heads = (aecct::u32_t)8u;
    cfg.d_ffn = (aecct::u32_t)kTileWords;
    cfg.n_layers = (aecct::u32_t)1u;
    return cfg;
}

static inline aecct::quant_acc_t inv_sqrt_d_head(uint32_t d_head) {
    return aecct::attn_phaseb_inv_sqrt_d_head(d_head);
}

static inline bool run_ac_ad_mainline(
    std::vector<aecct::u32_t>& sram,
    bool& q_fallback_taken,
    bool& kv_fallback_taken
) {
    const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
    const aecct::CfgRegs cfg = build_cfg();
    const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
    const aecct::LayerParamBase pb =
        aecct::make_layer_param_base((aecct::u32_t)param_base, (aecct::u32_t)0u);

    q_fallback_taken = true;
    const bool q_ok = aecct::run_p11ad_layer0_top_managed_q(
        sram.data(),
        cfg,
        (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
        sc,
        pb,
        q_fallback_taken
    );
    if (!q_ok || q_fallback_taken) {
        return false;
    }

    kv_fallback_taken = true;
    const bool kv_ok = aecct::run_p11ac_layer0_top_managed_kv(
        sram.data(),
        cfg,
        (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
        sc,
        pb,
        kv_fallback_taken
    );
    if (!kv_ok || kv_fallback_taken) {
        return false;
    }
    return true;
}

static inline void compute_expected_score_row(
    const std::vector<aecct::u32_t>& sram,
    const aecct::AttnScratch& sc,
    uint32_t token_idx,
    uint32_t token_count,
    uint32_t n_heads,
    uint32_t d_head,
    std::vector<aecct::u32_t>& expected
) {
    expected.assign(n_heads * token_count, (aecct::u32_t)0u);
    const uint32_t d_model = n_heads * d_head;
    const uint32_t q_base = (uint32_t)sc.q_base_word.to_uint();
    const uint32_t k_base = (uint32_t)sc.k_base_word.to_uint();
    const uint32_t q_row_base = q_base + token_idx * d_model;
    const aecct::quant_acc_t inv = inv_sqrt_d_head(d_head);

    for (uint32_t h = 0u; h < n_heads; ++h) {
        const uint32_t head_col_base = h * d_head;
        for (uint32_t j = 0u; j < token_count; ++j) {
            const uint32_t k_row_base = k_base + j * d_model + head_col_base;
            aecct::quant_acc_t dot = 0;
            for (uint32_t d = 0u; d < d_head; ++d) {
                const aecct::quant_act_t qv =
                    aecct::quant_act_from_bits(sram[q_row_base + head_col_base + d]);
                const aecct::quant_act_t kv =
                    aecct::quant_act_from_bits(sram[k_row_base + d]);
                dot += aecct::quant_acc_t(qv) * aecct::quant_acc_t(kv);
            }
            expected[h * token_count + j] = aecct::quant_bits_from_acc(dot * inv);
        }
    }
}

static inline void compute_expected_output_row_online(
    const std::vector<aecct::u32_t>& sram,
    const aecct::AttnScratch& sc,
    uint32_t token_idx,
    uint32_t token_count,
    uint32_t n_heads,
    uint32_t d_head,
    std::vector<aecct::u32_t>& expected_out
) {
    expected_out.assign(n_heads * d_head, (aecct::u32_t)0u);
    const uint32_t d_model = n_heads * d_head;
    const uint32_t score_base = (uint32_t)sc.score_base_word.to_uint();
    const uint32_t v_base = (uint32_t)sc.v_base_word.to_uint();

    for (uint32_t h = 0u; h < n_heads; ++h) {
        const uint32_t head_col_base = h * d_head;
        const uint32_t score_head_base = score_base + h * token_count;

        softmax_score_t running_max = softmax_score_t(0);
        softmax_sum_t running_l = softmax_sum_t(0);
        aecct::quant_acc_t running_acc[aecct::ATTN_D_MODEL];
        for (uint32_t d = 0u; d < d_head; ++d) {
            running_acc[d] = aecct::quant_acc_t(0);
        }

        bool have_state = false;
        for (uint32_t j = 0u; j < token_count; ++j) {
            const aecct::fp32_t score_fp = aecct::fp32_from_bits(sram[score_head_base + j]);
            const softmax_score_t score =
                score_fp.template convert_to_ac_fixed<18, 6, true, AC_RND, AC_SAT>(false);
            const uint32_t v_row_base = v_base + j * d_model + head_col_base;

            if (!have_state) {
                running_max = score;
                running_l = softmax_sum_t(1);
                for (uint32_t d = 0u; d < d_head; ++d) {
                    const aecct::quant_act_t vv = aecct::quant_act_from_bits(sram[v_row_base + d]);
                    running_acc[d] = aecct::quant_acc_t(vv);
                }
                have_state = true;
                continue;
            }

            if (score > running_max) {
                const softmax_x_t old_minus_new = softmax_x_t(running_max - score);
                const softmax_exp_t alpha = softmax_exp_lut(old_minus_new);
                running_l = softmax_sum_t(running_l * softmax_sum_t(alpha)) + softmax_sum_t(1);
                for (uint32_t d = 0u; d < d_head; ++d) {
                    const aecct::quant_act_t vv = aecct::quant_act_from_bits(sram[v_row_base + d]);
                    running_acc[d] =
                        aecct::quant_acc_t(running_acc[d] * aecct::quant_acc_t(alpha)) +
                        aecct::quant_acc_t(vv);
                }
                running_max = score;
            } else {
                const softmax_x_t score_minus_old = softmax_x_t(score - running_max);
                const softmax_exp_t beta = softmax_exp_lut(score_minus_old);
                running_l += softmax_sum_t(beta);
                for (uint32_t d = 0u; d < d_head; ++d) {
                    const aecct::quant_act_t vv = aecct::quant_act_from_bits(sram[v_row_base + d]);
                    running_acc[d] += aecct::quant_acc_t(beta) * aecct::quant_acc_t(vv);
                }
            }
        }

        const softmax_inv_t inv_l = softmax_rcp_lut(running_l);
        for (uint32_t d = 0u; d < d_head; ++d) {
            const aecct::quant_acc_t out_val = running_acc[d] * aecct::quant_acc_t(inv_l);
            expected_out[head_col_base + d] = aecct::quant_bits_from_acc(out_val);
        }
    }
}

} // namespace p11aeaf_tb
