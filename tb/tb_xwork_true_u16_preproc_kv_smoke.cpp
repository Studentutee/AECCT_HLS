#ifndef __SYNTHESIS__

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <ac_int.h>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "XWorkU16HybridView.h"
#include "PreprocDescBringup.h"
#include "AttnDescBringup.h"
#include "blocks/PreprocEmbedSPE.h"
#include "blocks/AttnPhaseATopManagedKv.h"
#include "blocks/TernaryLiveQkvLeafKernel.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"
#include "input_y_step0.h"
#include "weights.h"

namespace {

static uint32_t f32_to_bits(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_to_f32(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static void write_param_h_bitpack(aecct::u32_t* sram, uint32_t param_base_word) {
    const uint32_t h_base = param_base_word + (uint32_t)kParamMeta[20u].offset_w;
    const uint32_t h_words = (uint32_t)kParamMeta[20u].len_w;
    for (uint32_t i = 0u; i < h_words; ++i) {
        sram[h_base + i] = (aecct::u32_t)0u;
    }
    for (uint32_t c = 0u; c < (uint32_t)CODE_C; ++c) {
        for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
            const uint32_t flat = c * (uint32_t)CODE_N + v;
            if ((uint32_t)h_H[flat].to_uint() == 0u) {
                continue;
            }
            const uint32_t word_index = flat >> 5;
            const uint32_t bit_in_word = flat & 31u;
            const uint32_t prior = (uint32_t)sram[h_base + word_index].to_uint();
            sram[h_base + word_index] = (aecct::u32_t)(prior | (1u << bit_in_word));
        }
    }
}

static void write_param_src_embed(aecct::u32_t* sram, uint32_t param_base_word) {
    const uint32_t base = param_base_word + (uint32_t)kParamMeta[21u].offset_w;
    for (uint32_t token = 0u; token < (uint32_t)w_src_embed_shape[0]; ++token) {
        for (uint32_t d = 0u; d < (uint32_t)w_src_embed_shape[1]; ++d) {
            const float value = (float)w_src_embed[token * (uint32_t)w_src_embed_shape[1] + d];
            sram[base + token * (uint32_t)w_src_embed_shape[1] + d] = (aecct::u32_t)f32_to_bits(value);
        }
    }
}

static void write_param_lpe_token(aecct::u32_t* sram, uint32_t param_base_word) {
    const uint32_t base = param_base_word + (uint32_t)kParamMeta[68u].offset_w;
    for (uint32_t token = 0u; token < (uint32_t)w_lpe_token_shape[0]; ++token) {
        for (uint32_t d = 0u; d < (uint32_t)w_lpe_token_shape[1]; ++d) {
            const float value = (float)w_lpe_token[token * (uint32_t)w_lpe_token_shape[1] + d];
            sram[base + token * (uint32_t)w_lpe_token_shape[1] + d] = (aecct::u32_t)f32_to_bits(value);
        }
    }
}

static float ref_node_feature(uint32_t sample_idx, uint32_t token_idx) {
    const size_t sample_base = (size_t)sample_idx * (size_t)CODE_N;
    if (token_idx < (uint32_t)CODE_N) {
        const float y = (float)trace_input_y_step0_tensor[sample_base + token_idx];
        return std::fabs(y);
    }
    const uint32_t check_idx = token_idx - (uint32_t)CODE_N;
    uint32_t parity = 0u;
    for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
        const uint32_t flat = check_idx * (uint32_t)CODE_N + v;
        if ((uint32_t)h_H[flat].to_uint() == 0u) {
            continue;
        }
        const float y = (float)trace_input_y_step0_tensor[sample_base + v];
        parity ^= (y < 0.0f) ? 1u : 0u;
    }
    return (parity == 0u) ? 1.0f : -1.0f;
}

static uint32_t ref_x_bits(uint32_t sample_idx, uint32_t token_idx, uint32_t d) {
    if (d < (uint32_t)w_src_embed_shape[1]) {
        const float node = ref_node_feature(sample_idx, token_idx);
        const float embed = (float)w_src_embed[token_idx * (uint32_t)w_src_embed_shape[1] + d];
        return f32_to_bits(node * embed);
    }
    if (d < ((uint32_t)w_src_embed_shape[1] + (uint32_t)w_lpe_token_shape[1])) {
        const uint32_t lpe_d = d - (uint32_t)w_src_embed_shape[1];
        return f32_to_bits((float)w_lpe_token[token_idx * (uint32_t)w_lpe_token_shape[1] + lpe_d]);
    }
    return 0u;
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

static bool matrix_inv_sw(uint32_t matrix_id, double& out_inv_sw) {
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

static uint32_t encode_ternary_code(double w) {
    if (w > 0.5) return (uint32_t)TERNARY_CODE_POS;
    if (w < -0.5) return (uint32_t)TERNARY_CODE_NEG;
    return (uint32_t)TERNARY_CODE_ZERO;
}

static bool build_payload_for_matrix(
    uint32_t matrix_id,
    const QuantLinearMeta& meta,
    std::vector<aecct::u32_t>& out_payload) {
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
            out_payload[word_idx] = (aecct::u32_t)((uint32_t)out_payload[word_idx].to_uint() |
                                                   ((code & 0x3u) << (slot * 2u)));
        }
    }
    return true;
}

int run_one_sample(uint32_t sample_idx) {
    static aecct::u32_t main_sram[sram_map::SRAM_WORDS_TOTAL];
    static aecct::u16_t x_work_words[(uint32_t)storage_words_fp16(ELEMS_X)];
    for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        main_sram[i] = (aecct::u32_t)0u;
    }
    for (uint32_t i = 0u; i < (uint32_t)storage_words_fp16(ELEMS_X); ++i) {
        x_work_words[i] = (aecct::u16_t)0u;
    }

    const uint32_t param_base = (uint32_t)sram_map::PARAM_STREAM_DEFAULT_BASE_W;
    write_param_h_bitpack(main_sram, param_base);
    write_param_src_embed(main_sram, param_base);
    write_param_lpe_token(main_sram, param_base);

    const QuantLinearMeta wk_meta = aecct::ternary_linear_live_l0_wk_meta();
    const QuantLinearMeta wv_meta = aecct::ternary_linear_live_l0_wv_meta();
    std::vector<aecct::u32_t> wk_payload;
    std::vector<aecct::u32_t> wv_payload;
    if (!build_payload_for_matrix((uint32_t)QLM_L0_WK, wk_meta, wk_payload) ||
        !build_payload_for_matrix((uint32_t)QLM_L0_WV, wv_meta, wv_payload)) {
        std::printf("[xwork_true_u16][FAIL] payload build failed\n");
        return 1;
    }
    const uint32_t wk_payload_base = param_base + (uint32_t)kParamMeta[wk_meta.weight_param_id].offset_w;
    const uint32_t wv_payload_base = param_base + (uint32_t)kParamMeta[wv_meta.weight_param_id].offset_w;
    const uint32_t wk_inv_addr = param_base + (uint32_t)kParamMeta[wk_meta.inv_sw_param_id].offset_w;
    const uint32_t wv_inv_addr = param_base + (uint32_t)kParamMeta[wv_meta.inv_sw_param_id].offset_w;
    for (uint32_t i = 0u; i < wk_meta.payload_words_2b; ++i) main_sram[wk_payload_base + i] = wk_payload[i];
    for (uint32_t i = 0u; i < wv_meta.payload_words_2b; ++i) main_sram[wv_payload_base + i] = wv_payload[i];
    double wk_inv_sw = 0.0, wv_inv_sw = 0.0;
    matrix_inv_sw((uint32_t)QLM_L0_WK, wk_inv_sw);
    matrix_inv_sw((uint32_t)QLM_L0_WV, wv_inv_sw);
    main_sram[wk_inv_addr] = aecct::fp32_bits_from_double(wk_inv_sw);
    main_sram[wv_inv_addr] = aecct::fp32_bits_from_double(wv_inv_sw);

    const uint32_t in_base = (uint32_t)aecct::PREPROC_IN_BASE_WORD_DEFAULT;
    const size_t sample_base = (size_t)sample_idx * (size_t)CODE_N;
    for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
        main_sram[in_base + v] = (aecct::u32_t)f32_to_bits((float)trace_input_y_step0_tensor[sample_base + v]);
    }

    aecct::PreprocCfg cfg;
    cfg.infer_in_words = (aecct::u32_t)aecct::PREPROC_IN_WORDS_EXPECTED;
    cfg.x_out_words = (aecct::u32_t)aecct::PREPROC_X_OUT_WORDS_EXPECTED;

    aecct::PreprocBlockContract contract;
    aecct::clear_preproc_contract(contract);
    contract.start = true;
    contract.phase_id = aecct::PHASE_PREPROC;
    contract.x_work_base_word = (aecct::u32_t)0u;
    contract.w_base_word = (aecct::u32_t)param_base;
    contract.token_range = aecct::make_token_range((aecct::u32_t)0u, (aecct::u32_t)2u);
    contract.tile_range = aecct::make_tile_range((aecct::u32_t)0u,
        aecct::attn_top_managed_tile_count((aecct::u32_t)aecct::PREPROC_X_TOKEN_STRIDE_WORDS,
                                           (aecct::u32_t)aecct::ATTN_TOP_MANAGED_WORK_TILE_WORDS));

    aecct::XWorkU16HybridView<(uint32_t)sram_map::SRAM_WORDS_TOTAL, (uint32_t)storage_words_fp16(ELEMS_X)> view{main_sram, x_work_words};
    aecct::PreprocEmbedSPECoreWindow(view, cfg, (aecct::u32_t)in_base, (aecct::u32_t)0u, contract);

    uint32_t lane_mismatch = 0u;
    for (uint32_t t = 0u; t < 2u; ++t) {
        for (uint32_t d = 0u; d < (uint32_t)D_MODEL; ++d) {
            const uint32_t idx = t * (uint32_t)D_MODEL + d;
            const uint16_t got_lane = (uint16_t)x_work_words[idx].to_uint();
            const uint16_t ref_lane = (uint16_t)aecct::fp16_lane_from_fp32_bits((aecct::u32_t)ref_x_bits(sample_idx, t, d)).to_uint();
            if (got_lane != ref_lane) {
                if (lane_mismatch == 0u) {
                    std::printf("[xwork_true_u16][FAIL] preproc lane mismatch sample=%u token=%u d=%u got=0x%04X exp=0x%04X\n",
                                (unsigned)sample_idx, (unsigned)t, (unsigned)d, got_lane, ref_lane);
                }
                ++lane_mismatch;
            }
        }
    }
    if (lane_mismatch != 0u) {
        return 1;
    }

    aecct::AttnCfg attn_cfg;
    attn_cfg.token_count = (aecct::u32_t)2u;
    attn_cfg.d_model = (aecct::u32_t)D_MODEL;
    attn_cfg.n_heads = (aecct::u32_t)N_HEAD;
    attn_cfg.d_head = (aecct::u32_t)D_HEAD;
    aecct::AttnScratch sc = aecct::default_attn_scratch();
    bool fallback_taken = true;
    if (!aecct::attn_phasea_top_managed_kv_mainline(view, (aecct::u32_t)param_base, (aecct::u32_t)0u, attn_cfg, sc, fallback_taken)) {
        std::printf("[xwork_true_u16][FAIL] kv mainline returned false sample=%u fallback=%u\n",
                    (unsigned)sample_idx, (unsigned)(fallback_taken ? 1u : 0u));
        return 1;
    }
    if (fallback_taken) {
        std::printf("[xwork_true_u16][FAIL] kv mainline took fallback sample=%u\n", (unsigned)sample_idx);
        return 1;
    }

    uint32_t kv_mismatch = 0u;
    for (uint32_t t = 0u; t < 2u; ++t) {
        aecct::u32_t x_row[(uint32_t)D_MODEL];
        for (uint32_t d = 0u; d < (uint32_t)D_MODEL; ++d) {
            x_row[d] = aecct::fp32_bits_from_fp16_lane(x_work_words[t * (uint32_t)D_MODEL + d]);
        }
        aecct::u32_t k_out[aecct::kTernaryLiveL0WkRows];
        aecct::u32_t k_out_act_q[aecct::kTernaryLiveL0WkRows];
        aecct::u32_t k_out_inv_sw_bits = (aecct::u32_t)0u;
        aecct::u32_t v_out[aecct::kTernaryLiveL0WvRows];
        aecct::u32_t v_out_act_q[aecct::kTernaryLiveL0WvRows];
        aecct::u32_t v_out_inv_sw_bits = (aecct::u32_t)0u;
        if (!aecct::ternary_live_l0_wk_materialize_row_kernel_split(
                x_row, wk_payload.data(), main_sram[wk_inv_addr],
                k_out, k_out_act_q, k_out_inv_sw_bits) ||
            !aecct::ternary_live_l0_wv_materialize_row_kernel_split(
                x_row, wv_payload.data(), main_sram[wv_inv_addr],
                v_out, v_out_act_q, v_out_inv_sw_bits)) {
            std::printf("[xwork_true_u16][FAIL] reference kernel failed sample=%u token=%u\n",
                        (unsigned)sample_idx, (unsigned)t);
            return 1;
        }
        const uint32_t row_k_base = (uint32_t)sc.k_base_word.to_uint() + t * (uint32_t)D_MODEL;
        const uint32_t row_v_base = (uint32_t)sc.v_base_word.to_uint() + t * (uint32_t)D_MODEL;
        for (uint32_t d = 0u; d < (uint32_t)D_MODEL; ++d) {
            const uint32_t got_k = (uint32_t)main_sram[row_k_base + d].to_uint();
            const uint32_t got_v = (uint32_t)main_sram[row_v_base + d].to_uint();
            const uint32_t exp_k = (uint32_t)k_out[d].to_uint();
            const uint32_t exp_v = (uint32_t)v_out[d].to_uint();
            if (got_k != exp_k || got_v != exp_v) {
                if (kv_mismatch == 0u) {
                    std::printf("[xwork_true_u16][FAIL] kv mismatch sample=%u token=%u d=%u got_k=%g exp_k=%g got_v=%g exp_v=%g\n",
                                (unsigned)sample_idx, (unsigned)t, (unsigned)d,
                                bits_to_f32(got_k), bits_to_f32(exp_k), bits_to_f32(got_v), bits_to_f32(exp_v));
                }
                ++kv_mismatch;
            }
        }
    }

    if (kv_mismatch != 0u) {
        return 1;
    }

    std::printf("[xwork_true_u16][PASS] sample=%u lane_mismatch=0 kv_mismatch=0 x_work_words16=%u\n",
                (unsigned)sample_idx, (unsigned)storage_words_fp16(ELEMS_X));
    return 0;
}

} // namespace

int main() {
    if (trace_input_y_step0_tensor_ndim != 2) {
        std::printf("[xwork_true_u16][FAIL] input trace rank=%d expect=2\n", trace_input_y_step0_tensor_ndim);
        return 1;
    }
    if ((uint32_t)trace_input_y_step0_tensor_shape[1] != (uint32_t)CODE_N) {
        std::printf("[xwork_true_u16][FAIL] input trace cols=%d expect=%u\n",
                    trace_input_y_step0_tensor_shape[1], (unsigned)CODE_N);
        return 1;
    }
    for (uint32_t sample = 0u; sample < 3u; ++sample) {
        if (run_one_sample(sample) != 0) {
            return 1;
        }
    }
    std::printf("PASS: tb_xwork_true_u16_preproc_kv_smoke\n");
    return 0;
}

#endif
