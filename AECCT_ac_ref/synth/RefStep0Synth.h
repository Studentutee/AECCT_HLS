#pragma once

#include "ac_channel.h"
#include "ac_int.h"
#include "ac_std_float.h"

#include "../include/RefStep0RunReport.h"

namespace aecct_ref {

typedef ac_std_float<16, 5> fp16_ref_t;
typedef ac_int<1, false> bit1_t;
typedef ac_int<8, false> io8_word_t;
typedef ac_int<16, false> u16_word_t;
typedef ac_int<32, false> u32_word_t;

// Core synth entry:
// - compute domain uses fp16
// - output transport uses io16 words
// - x_pred mode emits two io16 words per legacy 32-bit packed word
void ref_step0_synth_core(
  ac_channel<fp16_ref_t> &in_y_ch,
  ac_channel<u16_word_t> &data_out_u16
);

// Optional outer wrapper for byte-stream integration.
// Input:  two bytes per fp16 sample (little-endian).
// Output: two bytes per io16 output word (little-endian).
void ref_step0_synth_io8(
  ac_channel<io8_word_t> &in_y_bytes,
  ac_channel<io8_word_t> &data_out_bytes
);

// Backward-compatible alias to the new fp16 core.
inline void ref_step0_synth(
  ac_channel<fp16_ref_t> &in_y_ch,
  ac_channel<u16_word_t> &data_out_u16
) {
  ref_step0_synth_core(in_y_ch, data_out_u16);
}

void ref_step0_set_outmode(ac_int<2, false> mode);
const RefStep0RunReport &ref_step0_get_last_report();

} // namespace aecct_ref
