#ifndef __SYNTHESIS__

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <ac_int.h>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "PreprocDescBringup.h"
#include "blocks/PreprocEmbedSPE.h"
#include "gen/SramMap.h"
#include "input_y_step0.h"
#include "weights.h"

namespace {

struct ShadowStats {
    uint32_t fp32_mismatch_count;
    uint32_t shadow_lane_mismatch_count;
    uint32_t first_token;
    uint32_t first_dim;
    uint32_t first_got_bits;
    uint32_t first_ref_bits;
    uint16_t first_got_lane;
    uint16_t first_ref_lane;
    float max_abs_diff_fp32;
    float max_abs_diff_shadow;
};

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

static uint16_t fp16_lane_from_fp32_bits(uint32_t fp32_bits) {
    const aecct::fp32_t x = aecct::fp32_from_bits((aecct::u32_t)fp32_bits);
    const aecct::fp16_t h(x);
    const ac_int<16, true> raw = h.data_ac_int();
    return (uint16_t)((ac_int<16, false>)raw).to_uint();
}

static uint32_t fp32_bits_from_fp16_lane(uint16_t lane) {
    aecct::fp16_t h;
    h.set_data((ac_int<16, true>)(ac_int<16, false>)lane);
    const aecct::fp32_t y(h);
    return (uint32_t)aecct::bits_from_fp32(y).to_uint();
}

static void clear_stats(ShadowStats& s) {
    s.fp32_mismatch_count = 0u;
    s.shadow_lane_mismatch_count = 0u;
    s.first_token = 0u;
    s.first_dim = 0u;
    s.first_got_bits = 0u;
    s.first_ref_bits = 0u;
    s.first_got_lane = 0u;
    s.first_ref_lane = 0u;
    s.max_abs_diff_fp32 = 0.0f;
    s.max_abs_diff_shadow = 0.0f;
}

static void write_param_h_bitpack(aecct::u32_t* sram, uint32_t param_base_word) {
    const uint32_t h_base = param_base_word + (uint32_t)kParamMeta[20u].offset_w;
    const uint32_t h_words = (uint32_t)kParamMeta[20u].len_w;
    H_ZERO_LOOP: for (uint32_t i = 0u; i < h_words; ++i) {
        sram[h_base + i] = (aecct::u32_t)0u;
    }
    H_PACK_ROW_LOOP: for (uint32_t c = 0u; c < (uint32_t)CODE_C; ++c) {
        H_PACK_COL_LOOP: for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
            const uint32_t flat = c * (uint32_t)CODE_N + v;
            if ((uint32_t)h_H[flat].to_uint() == 0u) {
                continue;
            }
            const uint32_t bit_index = flat;
            const uint32_t word_index = bit_index >> 5;
            const uint32_t bit_in_word = bit_index & 31u;
            const uint32_t prior = (uint32_t)sram[h_base + word_index].to_uint();
            sram[h_base + word_index] = (aecct::u32_t)(prior | (1u << bit_in_word));
        }
    }
}

static void write_param_src_embed(aecct::u32_t* sram, uint32_t param_base_word) {
    const uint32_t base = param_base_word + (uint32_t)kParamMeta[21u].offset_w;
    SRC_EMBED_TOKEN_LOOP: for (uint32_t token = 0u; token < (uint32_t)w_src_embed_shape[0]; ++token) {
        SRC_EMBED_DIM_LOOP: for (uint32_t d = 0u; d < (uint32_t)w_src_embed_shape[1]; ++d) {
            const float value = (float)w_src_embed[token * (uint32_t)w_src_embed_shape[1] + d];
            sram[base + token * (uint32_t)w_src_embed_shape[1] + d] = (aecct::u32_t)f32_to_bits(value);
        }
    }
}

static void write_param_lpe_token(aecct::u32_t* sram, uint32_t param_base_word) {
    const uint32_t base = param_base_word + (uint32_t)kParamMeta[68u].offset_w;
    LPE_TOKEN_TOKEN_LOOP: for (uint32_t token = 0u; token < (uint32_t)w_lpe_token_shape[0]; ++token) {
        LPE_TOKEN_DIM_LOOP: for (uint32_t d = 0u; d < (uint32_t)w_lpe_token_shape[1]; ++d) {
            const float value = (float)w_lpe_token[token * (uint32_t)w_lpe_token_shape[1] + d];
            sram[base + token * (uint32_t)w_lpe_token_shape[1] + d] = (aecct::u32_t)f32_to_bits(value);
        }
    }
}

static void write_param_subset(aecct::u32_t* sram, uint32_t param_base_word) {
    write_param_h_bitpack(sram, param_base_word);
    write_param_src_embed(sram, param_base_word);
    write_param_lpe_token(sram, param_base_word);
}

static void build_topfed_input_words(uint32_t sample_idx, aecct::u32_t* dst_words) {
    const size_t sample_base = (size_t)sample_idx * (size_t)CODE_N;
    INPUT_BUILD_LOOP: for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
        const float value = (float)trace_input_y_step0_tensor[sample_base + v];
        dst_words[v] = (aecct::u32_t)f32_to_bits(value);
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
    REF_PARITY_VAR_LOOP: for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
        const uint32_t flat = check_idx * (uint32_t)CODE_N + v;
        if ((uint32_t)h_H[flat].to_uint() == 0u) {
            continue;
        }
        const float y = (float)trace_input_y_step0_tensor[sample_base + v];
        parity ^= (y < 0.0f) ? 1u : 0u;
    }
    return (parity == 0u) ? 1.0f : -1.0f;
}

static uint32_t ref_x_word_bits(uint32_t sample_idx, uint32_t token_idx, uint32_t d) {
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

static int run_one_sample(uint32_t sample_idx, ShadowStats& stats) {
    clear_stats(stats);
    static aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
    SRAM_INIT_LOOP: for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
        sram[i] = (aecct::u32_t)0xDEADBEEFu;
    }

    const uint32_t param_base = (uint32_t)sram_map::PARAM_STREAM_DEFAULT_BASE_W;
    const uint32_t in_base = (uint32_t)aecct::PREPROC_IN_BASE_WORD_DEFAULT;
    const uint32_t x_base = (uint32_t)aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT;

    write_param_subset(sram, param_base);

    aecct::u32_t topfed_in[(uint32_t)aecct::PREPROC_IN_WORDS_EXPECTED];
    build_topfed_input_words(sample_idx, topfed_in);
    INPUT_STAGE_LOOP: for (uint32_t i = 0u; i < (uint32_t)aecct::PREPROC_IN_WORDS_EXPECTED; ++i) {
        sram[in_base + i] = topfed_in[i];
    }

    aecct::PreprocCfg cfg;
    cfg.infer_in_words = (aecct::u32_t)aecct::PREPROC_IN_WORDS_EXPECTED;
    cfg.x_out_words = (aecct::u32_t)aecct::PREPROC_X_OUT_WORDS_EXPECTED;

    aecct::PreprocBlockContract contract;
    aecct::clear_preproc_contract(contract);
    contract.start = true;
    contract.phase_id = aecct::PHASE_PREPROC;
    contract.x_work_base_word = (aecct::u32_t)x_base;
    contract.w_base_word = (aecct::u32_t)param_base;
    contract.token_range = aecct::make_token_range((aecct::u32_t)0u, (aecct::u32_t)N_NODES);
    contract.tile_range = aecct::make_tile_range((aecct::u32_t)0u,
        aecct::attn_top_managed_tile_count((aecct::u32_t)aecct::PREPROC_X_TOKEN_STRIDE_WORDS,
                                           (aecct::u32_t)aecct::ATTN_TOP_MANAGED_WORK_TILE_WORDS));

    aecct::u32_t* sram_view = sram;
    aecct::PreprocEmbedSPECoreWindow<aecct::u32_t*>(
        sram_view,
        cfg,
        (aecct::u32_t)in_base,
        (aecct::u32_t)x_base,
        contract,
        topfed_in);

    static uint16_t got_shadow[(uint32_t)ELEMS_X];
    static uint16_t ref_shadow[(uint32_t)ELEMS_X];

    REF_COMPARE_TOKEN_LOOP: for (uint32_t token = 0u; token < (uint32_t)N_NODES; ++token) {
        REF_COMPARE_DIM_LOOP: for (uint32_t d = 0u; d < (uint32_t)D_MODEL; ++d) {
            const uint32_t linear_idx = token * (uint32_t)D_MODEL + d;
            const uint32_t got_bits = (uint32_t)sram[x_base + linear_idx].to_uint();
            const uint32_t ref_bits = ref_x_word_bits(sample_idx, token, d);
            const float abs_diff_fp32 = std::fabs(bits_to_f32(got_bits) - bits_to_f32(ref_bits));
            if (abs_diff_fp32 > stats.max_abs_diff_fp32) {
                stats.max_abs_diff_fp32 = abs_diff_fp32;
            }
            if (got_bits != ref_bits) {
                if (stats.fp32_mismatch_count == 0u) {
                    stats.first_token = token;
                    stats.first_dim = d;
                    stats.first_got_bits = got_bits;
                    stats.first_ref_bits = ref_bits;
                }
                ++stats.fp32_mismatch_count;
            }

            const uint16_t got_lane = fp16_lane_from_fp32_bits(got_bits);
            const uint16_t ref_lane = fp16_lane_from_fp32_bits(ref_bits);
            got_shadow[linear_idx] = got_lane;
            ref_shadow[linear_idx] = ref_lane;
            const float abs_diff_shadow = std::fabs(bits_to_f32(fp32_bits_from_fp16_lane(got_lane)) -
                                                   bits_to_f32(fp32_bits_from_fp16_lane(ref_lane)));
            if (abs_diff_shadow > stats.max_abs_diff_shadow) {
                stats.max_abs_diff_shadow = abs_diff_shadow;
            }
            if (got_lane != ref_lane) {
                if (stats.shadow_lane_mismatch_count == 0u) {
                    stats.first_token = token;
                    stats.first_dim = d;
                    stats.first_got_lane = got_lane;
                    stats.first_ref_lane = ref_lane;
                }
                ++stats.shadow_lane_mismatch_count;
            }
        }
    }

    const uint32_t current_storage_word16 = (uint32_t)sram_map::X_PAGE0_WORDS_WORD16;
    const uint32_t shadow_storage_word16 = align_up_storage_words((uint32_t)ELEMS_X, (uint32_t)ALIGN_STORAGE_WORD16);
    std::printf(
        "[xwork_fp16_shadow] sample=%u fp32_mismatch=%u shadow_lane_mismatch=%u first_token=%u first_dim=%u first_got_lane=0x%04X first_ref_lane=0x%04X current_word16=%u shadow_word16=%u max_abs_diff_fp32=%.9g max_abs_diff_shadow=%.9g\n",
        (unsigned)sample_idx,
        (unsigned)stats.fp32_mismatch_count,
        (unsigned)stats.shadow_lane_mismatch_count,
        (unsigned)stats.first_token,
        (unsigned)stats.first_dim,
        (unsigned)stats.first_got_lane,
        (unsigned)stats.first_ref_lane,
        (unsigned)current_storage_word16,
        (unsigned)shadow_storage_word16,
        (double)stats.max_abs_diff_fp32,
        (double)stats.max_abs_diff_shadow);

    if (shadow_storage_word16 >= current_storage_word16) {
        std::printf("[xwork_fp16_shadow][FAIL] shadow storage did not shrink X_PAGE0\n");
        return 1;
    }
    if (stats.shadow_lane_mismatch_count != 0u) {
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (trace_input_y_step0_tensor_ndim != 2) {
        std::printf("[xwork_fp16_shadow][FAIL] input_y trace rank=%d expect=2\n", trace_input_y_step0_tensor_ndim);
        return 1;
    }
    if (trace_input_y_step0_tensor_shape[1] != (int)CODE_N) {
        std::printf("[xwork_fp16_shadow][FAIL] input_y cols=%d expect=%u\n",
            trace_input_y_step0_tensor_shape[1], (unsigned)CODE_N);
        return 1;
    }
    if ((uint32_t)D_MODEL != ((uint32_t)w_src_embed_shape[1] + (uint32_t)w_lpe_token_shape[1])) {
        std::printf("[xwork_fp16_shadow][FAIL] D_MODEL=%u src_embed+lpe=%u\n",
            (unsigned)D_MODEL,
            (unsigned)((uint32_t)w_src_embed_shape[1] + (uint32_t)w_lpe_token_shape[1]));
        return 1;
    }

    const uint32_t samples[3] = {0u, 1u, 2u};
    uint32_t total_shadow_mismatch = 0u;
    float global_max_abs_shadow = 0.0f;
    SAMPLE_LOOP: for (uint32_t i = 0u; i < 3u; ++i) {
        ShadowStats stats;
        if (run_one_sample(samples[i], stats) != 0) {
            return 1;
        }
        total_shadow_mismatch += stats.shadow_lane_mismatch_count;
        if (stats.max_abs_diff_shadow > global_max_abs_shadow) {
            global_max_abs_shadow = stats.max_abs_diff_shadow;
        }
    }

    std::printf("[xwork_fp16_shadow][PASS] samples=3 total_shadow_mismatch=%u current_word16=%u shadow_word16=%u max_abs_diff_shadow=%.9g\n",
        (unsigned)total_shadow_mismatch,
        (unsigned)sram_map::X_PAGE0_WORDS_WORD16,
        (unsigned)align_up_storage_words((uint32_t)ELEMS_X, (uint32_t)ALIGN_STORAGE_WORD16),
        (double)global_max_abs_shadow);
    std::printf("PASS: tb_xwork_fp16_shadow_storage_smoke\n");
    return 0;
}

#endif // __SYNTHESIS__
