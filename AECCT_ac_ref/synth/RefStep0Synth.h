#pragma once

#include "ac_channel.h"
#include "ac_int.h"
#include "ac_std_float.h"

namespace aecct_ref {

typedef ac_ieee_float<binary32> fp32_ref_t;
typedef ac_int<1, false> bit1_t;

void ref_step0_synth(
  ac_channel<fp32_ref_t> &in_y_ch,
  ac_channel<fp32_ref_t> &out_logits_ch,
  ac_channel<bit1_t> &out_xpred_ch
);

} // namespace aecct_ref

