#pragma once

#include "ac_channel.h"
#include "ac_int.h"
#include "ac_std_float.h"

#include "../include/RefStep0RunReport.h"

namespace aecct_ref {

typedef ac_ieee_float<binary32> fp32_ref_t;
typedef ac_int<1, false> bit1_t;
typedef ac_int<32, false> u32_word_t;

void ref_step0_synth(
  ac_channel<fp32_ref_t> &in_y_ch,
  ac_channel<u32_word_t> &data_out
);

void ref_step0_set_outmode(ac_int<2, false> mode);
const RefStep0RunReport &ref_step0_get_last_report();

} // namespace aecct_ref
