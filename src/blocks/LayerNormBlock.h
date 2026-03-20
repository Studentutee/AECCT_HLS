#pragma once
// LayerNorm block with two-pass implementation.

#include <cstdio>
#include <cstdint>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "AecctRanges.h"
#include "AecctUtil.h"
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

static inline void LayerNormBlock(
    u32_t* sram,
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

        quant_acc_t sum = 0;
        quant_acc_t sq_sum = 0;
        for (uint32_t c = 0; c < d_model; ++c) {
            quant_act_t x = quant_act_from_bits(sram[row_in_base + c]);
            sum += x;
            sq_sum += (quant_acc_t(x) * quant_acc_t(x));
        }

        fp32_t mean = fp32_zero();
        fp32_t inv_std = fp32_one();
        if (d_model != 0u) {
            ac_int<32, true> d_model_i = (ac_int<32, true>)d_model;
            fp32_t inv_n_den(d_model_i);
            fp32_t sum_fp(sum);
            fp32_t sq_sum_fp(sq_sum);

            mean = sum_fp / inv_n_den;
            fp32_t var = (sq_sum_fp / inv_n_den) - (mean * mean);
            fp32_t var_plus_eps = var + eps;
            if (var_plus_eps <= fp32_zero()) {
#ifndef __SYNTHESIS__
                static bool p11ah_ln_guard_logged = false;
                if (!p11ah_ln_guard_logged) {
                    p11ah_ln_guard_logged = true;
                    const u32_t var_bits = bits_from_fp32(var);
                    const u32_t vpe_bits = bits_from_fp32(var_plus_eps);
                    std::printf(
                        "[p11ah][LN_ASSERT_GUARD] token=%u var_bits=0x%08X var_plus_eps_bits=0x%08X eps_bits=0x%08X\n",
                        (unsigned)t,
                        (unsigned)var_bits.to_uint(),
                        (unsigned)vpe_bits.to_uint(),
                        (unsigned)cfg.eps_bits.to_uint());
                }
#endif
                var_plus_eps = eps;
            }
            fp32_t std_val = var_plus_eps.template sqrt<AC_RND_CONV, false>();
            inv_std = fp32_one().template div<AC_RND_CONV, false>(std_val);
        }

        for (uint32_t c = 0; c < d_model; ++c) {
            fp32_t x = fp32_from_bits(sram[row_in_base + c]);
            fp32_t g = fp32_from_bits(sram[gamma_base + c]);
            fp32_t b = fp32_from_bits(sram[beta_base + c]);
            fp32_t y = ((x - mean) * inv_std) * g + b;
            sram[row_out_base + c] = bits_from_fp32(y);
        }
    }
}

} // namespace aecct
