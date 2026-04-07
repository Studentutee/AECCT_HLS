#pragma once
// LayerNorm block with two-pass implementation.
// Pass 1 accumulates token-wise mean inputs.
// Pass 2 accumulates variance around mean, then normalizes and applies gamma/beta.
// Ownership boundary: caller/Top selects token/tile ranges and owns shared-SRAM policy.

#include <cstdio>
#include <cstdint>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "AecctRanges.h"
#include "AecctUtil.h"
#include "AttnTopManagedPackets.h"
#include "LayerNormDesc.h"
#include "QuantDesc.h"

namespace aecct {

struct LayerNormBlockContract {
    bool start;
    bool done;
    bool err_valid;
    u16_t err_code;
    TokenRange token_range;
    TileRange tile_range;
    PhaseId phase_id;
    u32_t x_work_base_word;
    u32_t scratch_base_word;
    u32_t gamma_base_word;
    u32_t beta_base_word;
};

static inline void clear_layernorm_contract(LayerNormBlockContract& c) {
    c.start = false;
    c.done = false;
    c.err_valid = false;
    c.err_code = 0;
    c.token_range = make_token_range(0, 0);
    c.tile_range = make_tile_range(0, 0);
    c.phase_id = PHASE_END_LN;
    c.x_work_base_word = 0;
    c.scratch_base_word = 0;
    c.gamma_base_word = 0;
    c.beta_base_word = 0;
}

struct LayerNormCfg {
    u32_t token_count;
    u32_t d_model;
    u32_t eps_bits;
};

struct LayerNormTopManagedTileMeta {
    u16_t phase_id;
    u16_t subphase_id;
    u16_t token_begin;
    u16_t token_end;
    u16_t token_idx;
    u16_t tile_begin;
    u16_t tile_end;
    u16_t tile_idx;
    u16_t tile_valid_words;
};

static inline bool layernorm_fp32_is_finite(const fp32_t& x) {
    const uint32_t bits = (uint32_t)bits_from_fp32(x).to_uint();
    return (bits & 0x7F800000u) != 0x7F800000u;
}

static inline fp32_t layernorm_fp32_sanitize_input(const fp32_t& x) {
    if (layernorm_fp32_is_finite(x)) {
        return x;
    }
    return fp32_zero();
}

static inline fp32_t layernorm_refstyle_inv_sqrt_approx(const fp32_t& x_eps_safe) {
    const fp32_t half = fp32_from_bits((u32_t)0x3F000000u);       // 0.5
    const fp32_t three_halves = fp32_from_bits((u32_t)0x3FC00000u); // 1.5
    const fp32_t y0 = fp32_one().template div<AC_RND_CONV, false>(
        x_eps_safe.template sqrt<AC_RND_CONV, false>());
    fp32_t y1 = y0 * (three_halves - (half * x_eps_safe * y0 * y0));
    if (!layernorm_fp32_is_finite(y1) || y1 <= fp32_zero()) {
        y1 = y0;
    }
    if (!layernorm_fp32_is_finite(y1) || y1 <= fp32_zero()) {
        return fp32_one();
    }
    return y1;
}

// Local-only observability for affine consume seam selection.
// This does not affect production ownership semantics.
struct LayerNormAffineConsumeTrace {
    bool saw_topfed_gamma;
    bool saw_topfed_beta;
    bool used_topfed_gamma;
    bool used_topfed_beta;
    bool used_fallback_gamma;
    bool used_fallback_beta;
    u32_t topfed_gamma_words_consumed;
    u32_t topfed_beta_words_consumed;
    u32_t fallback_gamma_words_consumed;
    u32_t fallback_beta_words_consumed;
};

static inline void clear_layernorm_affine_consume_trace(LayerNormAffineConsumeTrace& t) {
    t.saw_topfed_gamma = false;
    t.saw_topfed_beta = false;
    t.used_topfed_gamma = false;
    t.used_topfed_beta = false;
    t.used_fallback_gamma = false;
    t.used_fallback_beta = false;
    t.topfed_gamma_words_consumed = (u32_t)0u;
    t.topfed_beta_words_consumed = (u32_t)0u;
    t.fallback_gamma_words_consumed = (u32_t)0u;
    t.fallback_beta_words_consumed = (u32_t)0u;
}

static inline void layernorm_affine_trace_inc(u32_t& counter) {
    counter = (u32_t)((uint32_t)counter.to_uint() + 1u);
}

static inline bool layernorm_top_managed_tile_meta_ok(
    const LayerNormTopManagedTileMeta& m,
    uint32_t expect_phase_id,
    uint32_t expect_token_idx,
    uint32_t expect_tile_idx
) {
    if ((uint32_t)m.phase_id.to_uint() != expect_phase_id) { return false; }
    if ((uint32_t)m.token_idx.to_uint() != expect_token_idx) { return false; }
    if ((uint32_t)m.tile_idx.to_uint() != expect_tile_idx) { return false; }

    const uint32_t t_begin = (uint32_t)m.token_begin.to_uint();
    const uint32_t t_end = (uint32_t)m.token_end.to_uint();
    const uint32_t dt_begin = (uint32_t)m.tile_begin.to_uint();
    const uint32_t dt_end = (uint32_t)m.tile_end.to_uint();
    const uint32_t valid = (uint32_t)m.tile_valid_words.to_uint();

    if (t_end <= t_begin) { return false; }
    if (dt_end <= dt_begin) { return false; }
    if (valid == 0u || valid > (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS) { return false; }
    return true;
}

template<typename SramView>
static inline void LayerNormBlockCoreWindow(
    SramView& sram,
    const LayerNormCfg& cfg,
    u32_t x_in_base_word,
    u32_t x_out_base_word,
    const LayerNormBlockContract& contract,
    const u32_t* topfed_gamma_words = 0,
    const u32_t* topfed_beta_words = 0,
    LayerNormAffineConsumeTrace* affine_trace = 0
) {
    uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    if (token_count == 0u) { token_count = (uint32_t)LN_TOKEN_COUNT; }
    if (d_model == 0u) { d_model = (uint32_t)LN_D_MODEL; }
    if (token_count == 0u || d_model == 0u) {
        return;
    }

    const fp32_t eps = fp32_from_bits(cfg.eps_bits);
    const uint32_t x_in_base = (uint32_t)x_in_base_word.to_uint();
    const uint32_t x_out_base = (uint32_t)x_out_base_word.to_uint();
    const uint32_t gamma_base = (uint32_t)contract.gamma_base_word.to_uint();
    const uint32_t beta_base = (uint32_t)contract.beta_base_word.to_uint();

    const uint32_t tile_words = (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS;
    const uint32_t d_model_tile_count = attn_top_managed_tile_count(d_model, tile_words);
    if (tile_words == 0u || d_model_tile_count == 0u) {
        return;
    }

    // Contract clipping keeps execution inside Top-selected token boundaries.
    uint32_t token_begin = (uint32_t)contract.token_range.begin.to_uint();
    uint32_t token_end = (uint32_t)contract.token_range.end.to_uint();
    if (token_begin > token_count) { token_begin = token_count; }
    if (token_end > token_count) { token_end = token_count; }
    if (token_end <= token_begin) {
        return;
    }

    // Contract clipping keeps execution inside Top-selected tile boundaries.
    uint32_t tile_begin = (uint32_t)contract.tile_range.begin.to_uint();
    uint32_t tile_end = (uint32_t)contract.tile_range.end.to_uint();
    if (tile_begin > d_model_tile_count) { tile_begin = d_model_tile_count; }
    if (tile_end > d_model_tile_count) { tile_end = d_model_tile_count; }
    if (tile_end <= tile_begin) {
        return;
    }

    const uint32_t phase_id_u32 = (uint32_t)contract.phase_id;
    const uint32_t subphase_id_u32 = (uint32_t)ATTN_SUBPHASE_OUT;
    const bool has_topfed_gamma = (topfed_gamma_words != 0);
    const bool has_topfed_beta = (topfed_beta_words != 0);
    if (affine_trace != 0) {
        clear_layernorm_affine_consume_trace(*affine_trace);
        affine_trace->saw_topfed_gamma = has_topfed_gamma;
        affine_trace->saw_topfed_beta = has_topfed_beta;
    }

    LAYERNORM_TOP_MANAGED_TOKEN_LOOP: for (uint32_t t = token_begin; t < token_end; ++t) {
        const uint32_t row_in_base = x_in_base + t * d_model;
        const uint32_t row_out_base = x_out_base + t * d_model;

        fp32_t sum_fp = fp32_zero();
        fp32_t sq_sum_fp = fp32_zero();
#ifndef __SYNTHESIS__
        // Legacy fixed-point accumulators are retained for P11AI root-cause diagnostics.
        quant_acc_t legacy_sum = 0;
        quant_acc_t legacy_sq_sum = 0;
        fp32_t x_min = fp32_zero();
        fp32_t x_max = fp32_zero();
        bool x_minmax_init = false;
        bool input_nonfinite = false;
#endif

        // Pass-1: token-wise mean/variance accumulation over Top-provided d-model tiles.
        LAYERNORM_TOP_MANAGED_PASS1_TILE_LOOP: for (uint32_t dt = tile_begin; dt < tile_end; ++dt) {
            const uint32_t tile_offset = dt * tile_words;
            const uint32_t valid = attn_top_managed_tile_valid_words(d_model, tile_words, dt);

            LayerNormTopManagedTileMeta meta;
            meta.phase_id = (u16_t)phase_id_u32;
            meta.subphase_id = (u16_t)subphase_id_u32;
            meta.token_begin = (u16_t)token_begin;
            meta.token_end = (u16_t)token_end;
            meta.token_idx = (u16_t)t;
            meta.tile_begin = (u16_t)tile_begin;
            meta.tile_end = (u16_t)tile_end;
            meta.tile_idx = (u16_t)dt;
            meta.tile_valid_words = (u16_t)valid;
            if (!layernorm_top_managed_tile_meta_ok(meta, phase_id_u32, t, dt)) {
                continue;
            }

            LAYERNORM_TOP_MANAGED_PASS1_TILE_LOAD_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                const uint32_t c = tile_offset + i;
                const u32_t x_bits = sram[row_in_base + c];
                const fp32_t x_raw = fp32_from_bits(x_bits);
                const fp32_t x = layernorm_fp32_sanitize_input(x_raw);
                sum_fp += x;
                sq_sum_fp += (x * x);
#ifndef __SYNTHESIS__
                const quant_act_t x_q = quant_act_from_bits(x_bits);
                legacy_sum += x_q;
                legacy_sq_sum += (quant_acc_t(x_q) * quant_acc_t(x_q));
                if (!x_minmax_init) {
                    x_min = x_raw;
                    x_max = x_raw;
                    x_minmax_init = true;
                } else {
                    if (x_raw < x_min) { x_min = x_raw; }
                    if (x_raw > x_max) { x_max = x_raw; }
                }
                const uint32_t raw = (uint32_t)x_bits.to_uint();
                if ((raw & 0x7F800000u) == 0x7F800000u) {
                    input_nonfinite = true;
                }
#endif
            }
        }

        fp32_t mean = fp32_zero();
        fp32_t inv_std = fp32_one();
        ac_int<32, true> d_model_i = (ac_int<32, true>)d_model;
        fp32_t inv_n_den(d_model_i);

        mean = sum_fp / inv_n_den;
        fp32_t var_acc = fp32_zero();
        LAYERNORM_TOP_MANAGED_VAR_TILE_LOOP: for (uint32_t dt = tile_begin; dt < tile_end; ++dt) {
            const uint32_t tile_offset = dt * tile_words;
            const uint32_t valid = attn_top_managed_tile_valid_words(d_model, tile_words, dt);
            LayerNormTopManagedTileMeta meta;
            meta.phase_id = (u16_t)phase_id_u32;
            meta.subphase_id = (u16_t)subphase_id_u32;
            meta.token_begin = (u16_t)token_begin;
            meta.token_end = (u16_t)token_end;
            meta.token_idx = (u16_t)t;
            meta.tile_begin = (u16_t)tile_begin;
            meta.tile_end = (u16_t)tile_end;
            meta.tile_idx = (u16_t)dt;
            meta.tile_valid_words = (u16_t)valid;
            if (!layernorm_top_managed_tile_meta_ok(meta, phase_id_u32, t, dt)) {
                continue;
            }
            LAYERNORM_TOP_MANAGED_VAR_ELEM_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                const uint32_t c = tile_offset + i;
                const fp32_t x = layernorm_fp32_sanitize_input(fp32_from_bits(sram[row_in_base + c]));
                const fp32_t d = x - mean;
                var_acc += (d * d);
            }
        }
        fp32_t var = var_acc / inv_n_den;
#ifndef __SYNTHESIS__
        static bool p11ai_ln_root_cause_logged = false;
        if (!p11ai_ln_root_cause_logged) {
            const fp32_t legacy_sum_fp(legacy_sum);
            const fp32_t legacy_sq_sum_fp(legacy_sq_sum);
            const fp32_t legacy_mean = legacy_sum_fp / inv_n_den;
            const fp32_t legacy_var = (legacy_sq_sum_fp / inv_n_den) - (legacy_mean * legacy_mean);
            const fp32_t ex2_var = (sq_sum_fp / inv_n_den) - (mean * mean);
            if (legacy_var <= fp32_zero() && var > fp32_zero()) {
                p11ai_ln_root_cause_logged = true;
                std::printf(
                    "[p11ai][LN_ROOT_CAUSE] token=%u input_nonfinite=%d legacy_var_bits=0x%08X fp32_var_bits=0x%08X ex2_var_bits=0x%08X legacy_sq_sum_bits=0x%08X fp32_sq_sum_bits=0x%08X legacy_sum_bits=0x%08X fp32_sum_bits=0x%08X x_min_bits=0x%08X x_max_bits=0x%08X\n",
                    (unsigned)t,
                    input_nonfinite ? 1 : 0,
                    (unsigned)bits_from_fp32(legacy_var).to_uint(),
                    (unsigned)bits_from_fp32(var).to_uint(),
                    (unsigned)bits_from_fp32(ex2_var).to_uint(),
                    (unsigned)bits_from_fp32(legacy_sq_sum_fp).to_uint(),
                    (unsigned)bits_from_fp32(sq_sum_fp).to_uint(),
                    (unsigned)bits_from_fp32(legacy_sum_fp).to_uint(),
                    (unsigned)bits_from_fp32(sum_fp).to_uint(),
                    (unsigned)bits_from_fp32(x_min).to_uint(),
                    (unsigned)bits_from_fp32(x_max).to_uint());
            }
        }
#endif

        fp32_t var_plus_eps = var + eps;
        if (!layernorm_fp32_is_finite(var_plus_eps) || var_plus_eps <= fp32_zero()) {
#ifndef __SYNTHESIS__
            static bool p11ai_ln_guard_logged = false;
            if (!p11ai_ln_guard_logged) {
                p11ai_ln_guard_logged = true;
                const u32_t var_bits = bits_from_fp32(var);
                const u32_t vpe_bits = bits_from_fp32(var_plus_eps);
                std::printf(
                    "[p11ai][LN_ASSERT_GUARD] token=%u var_bits=0x%08X var_plus_eps_bits=0x%08X eps_bits=0x%08X\n",
                    (unsigned)t,
                    (unsigned)var_bits.to_uint(),
                    (unsigned)vpe_bits.to_uint(),
                    (unsigned)cfg.eps_bits.to_uint());
            }
#endif
            var_plus_eps = eps;
        }
        if (!layernorm_fp32_is_finite(var_plus_eps) || var_plus_eps <= fp32_zero()) {
            var_plus_eps = fp32_one();
        }
        inv_std = layernorm_refstyle_inv_sqrt_approx(var_plus_eps);

        // Pass-2: normalize + affine writeback, still token-wise and tile-driven.
        LAYERNORM_TOP_MANAGED_PASS2_TILE_LOOP: for (uint32_t dt = tile_begin; dt < tile_end; ++dt) {
            const uint32_t tile_offset = dt * tile_words;
            const uint32_t valid = attn_top_managed_tile_valid_words(d_model, tile_words, dt);

            LayerNormTopManagedTileMeta meta;
            meta.phase_id = (u16_t)phase_id_u32;
            meta.subphase_id = (u16_t)subphase_id_u32;
            meta.token_begin = (u16_t)token_begin;
            meta.token_end = (u16_t)token_end;
            meta.token_idx = (u16_t)t;
            meta.tile_begin = (u16_t)tile_begin;
            meta.tile_end = (u16_t)tile_end;
            meta.tile_idx = (u16_t)dt;
            meta.tile_valid_words = (u16_t)valid;
            if (!layernorm_top_managed_tile_meta_ok(meta, phase_id_u32, t, dt)) {
                continue;
            }

            LAYERNORM_TOP_MANAGED_PASS2_TILE_STORE_LOOP: for (uint32_t i = 0u; i < valid; ++i) {
                const uint32_t c = tile_offset + i;
                const fp32_t x = layernorm_fp32_sanitize_input(fp32_from_bits(sram[row_in_base + c]));
                // Affine consume seam: prefer Top-fed gamma/beta words, otherwise read caller SRAM.
                const u32_t g_bits =
                    (topfed_gamma_words != 0) ? topfed_gamma_words[c] : sram[gamma_base + c];
                const u32_t b_bits =
                    (topfed_beta_words != 0) ? topfed_beta_words[c] : sram[beta_base + c];
                if (affine_trace != 0) {
                    if (topfed_gamma_words != 0) {
                        affine_trace->used_topfed_gamma = true;
                        layernorm_affine_trace_inc(affine_trace->topfed_gamma_words_consumed);
                    } else {
                        affine_trace->used_fallback_gamma = true;
                        layernorm_affine_trace_inc(affine_trace->fallback_gamma_words_consumed);
                    }
                    if (topfed_beta_words != 0) {
                        affine_trace->used_topfed_beta = true;
                        layernorm_affine_trace_inc(affine_trace->topfed_beta_words_consumed);
                    } else {
                        affine_trace->used_fallback_beta = true;
                        layernorm_affine_trace_inc(affine_trace->fallback_beta_words_consumed);
                    }
                }
                const fp32_t g = fp32_from_bits(g_bits);
                const fp32_t b = fp32_from_bits(b_bits);
                const fp32_t y = ((x - mean) * inv_std) * g + b;
                sram[row_out_base + c] = bits_from_fp32(y);
            }
        }
    }
}

template<typename SramView>
static inline void LayerNormBlockCoreWindowDirect(
    SramView& sram,
    const LayerNormCfg& cfg,
    u32_t x_in_base_word,
    u32_t x_out_base_word,
    u32_t gamma_base_word,
    u32_t beta_base_word
) {
    uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    fp32_t eps = fp32_from_bits(cfg.eps_bits);

    uint32_t x_in_base = (uint32_t)x_in_base_word.to_uint();
    uint32_t x_out_base = (uint32_t)x_out_base_word.to_uint();
    uint32_t gamma_base = (uint32_t)gamma_base_word.to_uint();
    uint32_t beta_base = (uint32_t)beta_base_word.to_uint();

    for (uint32_t t = 0; t < token_count; ++t) {
        uint32_t row_in_base = x_in_base + t * d_model;
        uint32_t row_out_base = x_out_base + t * d_model;

        // LayerNorm arithmetic is FP32-domain per spec; keep pass1 accumulators in FP32.
        fp32_t sum_fp = fp32_zero();
        fp32_t sq_sum_fp = fp32_zero();
#ifndef __SYNTHESIS__
        // Legacy fixed-point accumulator reconstruction for P11AI root-cause isolation only.
        quant_acc_t legacy_sum = 0;
        quant_acc_t legacy_sq_sum = 0;
        fp32_t x_min = fp32_zero();
        fp32_t x_max = fp32_zero();
        bool x_minmax_init = false;
        bool input_nonfinite = false;
#endif
        for (uint32_t c = 0; c < d_model; ++c) {
            const u32_t x_bits = sram[row_in_base + c];
            const fp32_t x_raw = fp32_from_bits(x_bits);
            const fp32_t x = layernorm_fp32_sanitize_input(x_raw);
            sum_fp += x;
            sq_sum_fp += (x * x);
#ifndef __SYNTHESIS__
            const quant_act_t x_q = quant_act_from_bits(x_bits);
            legacy_sum += x_q;
            legacy_sq_sum += (quant_acc_t(x_q) * quant_acc_t(x_q));
            if (!x_minmax_init) {
                x_min = x_raw;
                x_max = x_raw;
                x_minmax_init = true;
            } else {
                if (x_raw < x_min) { x_min = x_raw; }
                if (x_raw > x_max) { x_max = x_raw; }
            }
            const uint32_t raw = (uint32_t)x_bits.to_uint();
            if ((raw & 0x7F800000u) == 0x7F800000u) {
                input_nonfinite = true;
            }
#endif
        }

        fp32_t mean = fp32_zero();
        fp32_t inv_std = fp32_one();
        if (d_model != 0u) {
            ac_int<32, true> d_model_i = (ac_int<32, true>)d_model;
            fp32_t inv_n_den(d_model_i);

            mean = sum_fp / inv_n_den;
            fp32_t var_acc = fp32_zero();
            for (uint32_t c = 0; c < d_model; ++c) {
                const fp32_t x = layernorm_fp32_sanitize_input(fp32_from_bits(sram[row_in_base + c]));
                const fp32_t d = x - mean;
                var_acc += (d * d);
            }
            fp32_t var = var_acc / inv_n_den;
#ifndef __SYNTHESIS__
            static bool p11ai_ln_root_cause_logged = false;
            if (!p11ai_ln_root_cause_logged) {
                const fp32_t legacy_sum_fp(legacy_sum);
                const fp32_t legacy_sq_sum_fp(legacy_sq_sum);
                const fp32_t legacy_mean = legacy_sum_fp / inv_n_den;
                const fp32_t legacy_var = (legacy_sq_sum_fp / inv_n_den) - (legacy_mean * legacy_mean);
                const fp32_t ex2_var = (sq_sum_fp / inv_n_den) - (mean * mean);
                if (legacy_var <= fp32_zero() && var > fp32_zero()) {
                    p11ai_ln_root_cause_logged = true;
                    std::printf(
                        "[p11ai][LN_ROOT_CAUSE] token=%u input_nonfinite=%d legacy_var_bits=0x%08X fp32_var_bits=0x%08X ex2_var_bits=0x%08X legacy_sq_sum_bits=0x%08X fp32_sq_sum_bits=0x%08X legacy_sum_bits=0x%08X fp32_sum_bits=0x%08X x_min_bits=0x%08X x_max_bits=0x%08X\n",
                        (unsigned)t,
                        input_nonfinite ? 1 : 0,
                        (unsigned)bits_from_fp32(legacy_var).to_uint(),
                        (unsigned)bits_from_fp32(var).to_uint(),
                        (unsigned)bits_from_fp32(ex2_var).to_uint(),
                        (unsigned)bits_from_fp32(legacy_sq_sum_fp).to_uint(),
                        (unsigned)bits_from_fp32(sq_sum_fp).to_uint(),
                        (unsigned)bits_from_fp32(legacy_sum_fp).to_uint(),
                        (unsigned)bits_from_fp32(sum_fp).to_uint(),
                        (unsigned)bits_from_fp32(x_min).to_uint(),
                        (unsigned)bits_from_fp32(x_max).to_uint());
                }
            }
#endif
            fp32_t var_plus_eps = var + eps;
            if (!layernorm_fp32_is_finite(var_plus_eps) || var_plus_eps <= fp32_zero()) {
#ifndef __SYNTHESIS__
                static bool p11ai_ln_guard_logged = false;
                if (!p11ai_ln_guard_logged) {
                    p11ai_ln_guard_logged = true;
                    const u32_t var_bits = bits_from_fp32(var);
                    const u32_t vpe_bits = bits_from_fp32(var_plus_eps);
                    std::printf(
                        "[p11ai][LN_ASSERT_GUARD] token=%u var_bits=0x%08X var_plus_eps_bits=0x%08X eps_bits=0x%08X\n",
                        (unsigned)t,
                        (unsigned)var_bits.to_uint(),
                        (unsigned)vpe_bits.to_uint(),
                        (unsigned)cfg.eps_bits.to_uint());
                }
#endif
                var_plus_eps = eps;
            }
            if (!layernorm_fp32_is_finite(var_plus_eps) || var_plus_eps <= fp32_zero()) {
                var_plus_eps = fp32_one();
            }
            inv_std = layernorm_refstyle_inv_sqrt_approx(var_plus_eps);
        }

        for (uint32_t c = 0; c < d_model; ++c) {
            fp32_t x = layernorm_fp32_sanitize_input(fp32_from_bits(sram[row_in_base + c]));
            fp32_t g = fp32_from_bits(sram[gamma_base + c]);
            fp32_t b = fp32_from_bits(sram[beta_base + c]);
            fp32_t y = ((x - mean) * inv_std) * g + b;
            sram[row_out_base + c] = bits_from_fp32(y);
        }
    }
}

// Public LayerNorm entry.
// Default mainline already uses the Top-managed token/tile core; direct-window access
// remains only as the worker implementation detail under the caller-owned base words.
static inline void LayerNormBlock(
    u32_t* sram,
    const LayerNormCfg& cfg,
    u32_t x_in_base_word,
    u32_t x_out_base_word,
    u32_t gamma_base_word,
    u32_t beta_base_word
) {
    // Public wrapper builds the contract that carries Top-owned range policy into the core.
    LayerNormBlockContract contract;
    clear_layernorm_contract(contract);
    contract.start = true;
    contract.done = false;
    contract.phase_id = PHASE_END_LN;
    contract.x_work_base_word = x_in_base_word;
    contract.gamma_base_word = gamma_base_word;
    contract.beta_base_word = beta_base_word;
    uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    if (token_count == 0u) { token_count = (uint32_t)LN_TOKEN_COUNT; }
    if (d_model == 0u) { d_model = (uint32_t)LN_D_MODEL; }
    const uint32_t tile_count =
        attn_top_managed_tile_count(d_model, (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS);
    contract.token_range = make_token_range((u32_t)0u, (u32_t)token_count);
    contract.tile_range = make_tile_range((u32_t)0u, (u32_t)tile_count);

    // Mainline migration: default LN entry consumes Top-managed token/tile range metadata.
    LayerNormBlockCoreWindow<u32_t*>(
        sram,
        cfg,
        x_in_base_word,
        x_out_base_word,
        contract
    );
    contract.done = true;
}

template<uint32_t SRAM_WORDS>
static inline void LayerNormBlockTopManagedWindowBridge(
    u32_t (&sram_window)[SRAM_WORDS],
    const LayerNormCfg& cfg,
    u32_t x_in_base_word,
    u32_t x_out_base_word,
    const LayerNormBlockContract& contract
) {
    LayerNormBlockCoreWindow<u32_t (&)[SRAM_WORDS]>(
        sram_window,
        cfg,
        x_in_base_word,
        x_out_base_word,
        contract
    );
}

template<uint32_t SRAM_WORDS>
static inline void LayerNormBlockTopManagedWindowBridge(
    u32_t (&sram_window)[SRAM_WORDS],
    const LayerNormCfg& cfg,
    u32_t x_in_base_word,
    u32_t x_out_base_word,
    u32_t gamma_base_word,
    u32_t beta_base_word,
    PhaseId phase_id = PHASE_END_LN
) {
    LayerNormBlockContract contract;
    clear_layernorm_contract(contract);
    contract.start = true;
    contract.done = false;
    contract.phase_id = phase_id;
    contract.x_work_base_word = x_in_base_word;
    contract.gamma_base_word = gamma_base_word;
    contract.beta_base_word = beta_base_word;
    uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    if (token_count == 0u) { token_count = (uint32_t)LN_TOKEN_COUNT; }
    if (d_model == 0u) { d_model = (uint32_t)LN_D_MODEL; }
    const uint32_t tile_count =
        attn_top_managed_tile_count(d_model, (uint32_t)ATTN_TOP_MANAGED_WORK_TILE_WORDS);
    contract.token_range = make_token_range((u32_t)0u, (u32_t)token_count);
    contract.tile_range = make_tile_range((u32_t)0u, (u32_t)tile_count);

    LayerNormBlockTopManagedWindowBridge(
        sram_window,
        cfg,
        x_in_base_word,
        x_out_base_word,
        contract
    );
}

} // namespace aecct
