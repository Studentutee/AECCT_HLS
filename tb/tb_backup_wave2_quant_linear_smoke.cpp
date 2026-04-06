// tb_backup_wave2_quant_linear_smoke.cpp
// Backup profile Wave2 smoke:
// - INT8 activation
// - ternary decode
// - INT16 accumulate

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "AecctUtil.h"
#include "QuantDesc.h"
#include "blocks/TernaryLinearLive.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"

namespace {

static void fail(const char* msg) {
    std::printf("[wave2][FAIL] %s\n", msg);
    std::exit(1);
}

static uint32_t pattern_code(uint32_t out_idx, uint32_t in_idx) {
    return (out_idx * 5u + in_idx * 11u) % 3u;
}

static int32_t pattern_weight(uint32_t code) {
    if (code == 0u) { return 0; }
    if (code == 1u) { return 1; }
    return -1;
}

static int32_t pattern_activation(uint32_t in_idx) {
    const int32_t base = (int32_t)(in_idx % 23u) - 11;
    return base * 3;
}

static int32_t sat_i16(int32_t x) {
    if (x > 32767) { return 32767; }
    if (x < -32768) { return -32768; }
    return x;
}

} // namespace

int main() {
    const aecct::quant_acc_i16_t sat_pos =
        aecct::quant_acc_i16_saturating_add((aecct::quant_acc_i16_t)32760, (aecct::quant_acc_i16_t)100);
    const aecct::quant_acc_i16_t sat_neg =
        aecct::quant_acc_i16_saturating_add((aecct::quant_acc_i16_t)-32760, (aecct::quant_acc_i16_t)-100);
    if ((int32_t)sat_pos.to_int() != 32767 || (int32_t)sat_neg.to_int() != -32768) {
        fail("quant_acc_i16_saturating_add helper mismatch");
    }

    if (!aecct::quant_no_overflow_for_frozen_shape(D_MODEL) ||
        !aecct::quant_no_overflow_for_frozen_shape(D_FFN)) {
        fail("frozen-shape INT16 no-overflow assumption failed");
    }

    const QuantLinearMeta meta = kQuantLinearMeta[(uint32_t)QLM_L0_WQ];
    if (meta.rows != D_MODEL || meta.cols != D_MODEL) {
        fail("unexpected L0_WQ metadata shape");
    }

    std::vector<aecct::u32_t> sram((uint32_t)sram_map::SRAM_WORDS_TOTAL);
    WAVE2_SRAM_ZERO_LOOP: for (uint32_t i = 0u; i < (uint32_t)sram.size(); ++i) {
        sram[i] = (aecct::u32_t)0u;
    }

    const uint32_t param_base = (uint32_t)sram_map::PARAM_BASE_DEFAULT;
    const uint32_t x_base = (uint32_t)sram_map::BASE_X_WORK_W;

    WAVE2_FILL_X_LOOP: for (uint32_t in = 0u; in < meta.cols; ++in) {
        const int32_t act = pattern_activation(in);
        sram[x_base + in] = aecct::quant_word_from_act_i8((aecct::quant_act_i8_t)act);
    }

    const ParamMeta payload_meta = kParamMeta[meta.weight_param_id];
    WAVE2_PACK_PAYLOAD_WORD_LOOP: for (uint32_t wi = 0u; wi < meta.payload_words_2b; ++wi) {
        sram[param_base + payload_meta.offset_w + wi] = (aecct::u32_t)0u;
    }
    WAVE2_PACK_PAYLOAD_ELEM_LOOP: for (uint32_t out = 0u; out < meta.rows; ++out) {
        for (uint32_t in = 0u; in < meta.cols; ++in) {
            const uint32_t code = pattern_code(out, in);
            const uint32_t elem_idx = out * meta.cols + in;
            const uint32_t word_idx = (elem_idx >> 4);
            const uint32_t slot = (elem_idx & 15u);
            const uint32_t addr = param_base + payload_meta.offset_w + word_idx;
            const uint32_t old_word = (uint32_t)sram[addr].to_uint();
            const uint32_t new_word = old_word | ((code & 0x3u) << (slot * 2u));
            sram[addr] = (aecct::u32_t)new_word;
        }
    }

    const ParamMeta inv_meta = kParamMeta[meta.inv_sw_param_id];
    const uint32_t inv_sw_bits = (uint32_t)aecct::fp32_bits_from_double(1.0).to_uint();
    sram[param_base + inv_meta.offset_w] = (aecct::u32_t)inv_sw_bits;

    WAVE2_COMPARE_ROWS_LOOP: for (uint32_t out = 0u; out < 8u; ++out) {
        aecct::u32_t got_q_bits = (aecct::u32_t)0u;
        aecct::u32_t got_inv_sw_bits = (aecct::u32_t)0u;
        if (!aecct::ternary_linear_live_compute_q_elem(
                sram.data(),
                (aecct::u32_t)param_base,
                QLM_L0_WQ,
                (aecct::u32_t)x_base,
                out,
                got_q_bits,
                got_inv_sw_bits)) {
            fail("ternary_linear_live_compute_q_elem returned false");
        }
        if ((uint32_t)got_inv_sw_bits.to_uint() != inv_sw_bits) {
            fail("inv_sw bits mismatch");
        }

        int32_t ref_acc = 0;
        WAVE2_REF_DOT_LOOP: for (uint32_t in = 0u; in < meta.cols; ++in) {
            const int32_t w = pattern_weight(pattern_code(out, in));
            const int32_t x = pattern_activation(in);
            ref_acc = sat_i16(ref_acc + (w * x));
        }
        const int16_t got_i16 = (int16_t)((uint16_t)got_q_bits.to_uint());
        if ((int32_t)got_i16 != ref_acc) {
            std::printf(
                "[wave2][FAIL] ref compare mismatch out=%u got=%d expect=%d\n",
                (unsigned)out,
                (int)got_i16,
                (int)ref_acc);
            return 1;
        }
    }

    std::printf("PASS: tb_backup_wave2_quant_linear_smoke\n");
    return 0;
}
