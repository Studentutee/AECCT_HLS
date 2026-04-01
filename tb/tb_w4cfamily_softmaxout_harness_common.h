// W4-C-family shared bootstrap harness for SoftmaxOut local probes.

#pragma once

#include <cstdint>
#include <vector>

#include "tb_p11aeaf_common.h"

namespace w4c_softmax_family {

static inline aecct::u32_t bits_from_float(float v) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = v;
    return (aecct::u32_t)cvt.u;
}

static inline void force_head_score(
    std::vector<aecct::u32_t>& sram_words,
    const aecct::LayerScratch& sc,
    uint32_t head_idx,
    uint32_t key_token_idx,
    float score_val) {
    const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
    const uint32_t score_base = (uint32_t)sc.attn.score_base_word.to_uint();
    const uint32_t score_head_base = score_base + head_idx * token_count;
    sram_words[score_head_base + key_token_idx] = bits_from_float(score_val);
}

static inline bool bootstrap_mainline_context(
    const char* tag,
    std::vector<aecct::u32_t>& sram_bootstrap,
    p11aeaf_tb::QkvPayloadSet& payloads,
    aecct::LayerScratch& sc,
    aecct::CfgRegs& cfg) {
    sram_bootstrap.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
    p11aeaf_tb::init_x_rows(sram_bootstrap);
    if (!p11aeaf_tb::prepare_qkv_payload_set(payloads)) {
        std::printf("[%s][FAIL] payload preparation failed\n", tag);
        return false;
    }

    const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;
    p11aeaf_tb::load_qkv_payload_set_to_sram(sram_bootstrap, payloads, param_base);
    sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
    cfg = p11aeaf_tb::build_cfg();

    bool q_fallback_taken = true;
    bool kv_fallback_taken = true;
    if (!p11aeaf_tb::run_ac_ad_mainline(sram_bootstrap, q_fallback_taken, kv_fallback_taken)) {
        std::printf("[%s][FAIL] AC/AD bootstrap failed\n", tag);
        return false;
    }
    if (q_fallback_taken || kv_fallback_taken) {
        std::printf("[%s][FAIL] AC/AD bootstrap fallback detected\n", tag);
        return false;
    }

    bool score_fallback_taken = true;
    const bool score_mainline_taken = aecct::run_p11ae_layer0_top_managed_qk_score(
        sram_bootstrap.data(),
        cfg,
        sc,
        (aecct::u32_t)0u,
        score_fallback_taken);
    if (!score_mainline_taken || score_fallback_taken) {
        std::printf("[%s][FAIL] AE bootstrap failed\n", tag);
        return false;
    }
    return true;
}

} // namespace w4c_softmax_family
