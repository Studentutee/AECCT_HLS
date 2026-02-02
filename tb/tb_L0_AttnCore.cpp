#include <cstdio>
#include <algorithm>
#include <cmath>

#include "ac_channel.h"
#include "ac_fixed.h"
#include "ac_int.h"
#include "compare_array_abs.h"

#ifndef ATOL
#define ATOL (1e-3)
#endif

#ifndef FAIL_FAST
#define FAIL_FAST 0
#endif

#ifndef PATTERN_INDEX
#define PATTERN_INDEX (-1)
#endif

// Trace headers
#include "layer0_attn_Q_step0.h"
#include "layer0_attn_K_step0.h"
#include "layer0_attn_V_step0.h"
#include "layer0_attn_post_concat_step0.h"

// DUT
#include "Layer0.h"

int main() {
  std::printf("==== tb_L0_AttnCore start ====" "\n");
  std::printf("ATOL = %.10g\n", (double)ATOL);

  constexpr int NUM_SAMPLES = trace_layer0_attn_Q_step0_tensor_shape[0];
  constexpr int H = trace_layer0_attn_Q_step0_tensor_shape[1]; // 75
  constexpr int W = trace_layer0_attn_Q_step0_tensor_shape[2]; // 32
  constexpr int LEN = H * W;

  static_assert(H == L0_AttnCore::N_NODES, "N_NODES mismatch");
  static_assert(W == L0_AttnCore::D_MODEL, "D_MODEL mismatch");

  static_assert(trace_layer0_attn_post_concat_step0_tensor_shape[0] == NUM_SAMPLES, "post_concat NUM_SAMPLES mismatch");
  static_assert(trace_layer0_attn_post_concat_step0_tensor_shape[1] == H, "post_concat H mismatch");
  static_assert(trace_layer0_attn_post_concat_step0_tensor_shape[2] == W, "post_concat W mismatch");

  L0_AttnCore dut;

  static fx_utils::fx_t out_buf[LEN];

  int s_begin = 0;
  int s_end = 100;// NUM_SAMPLES;
  if (PATTERN_INDEX >= 0) {
    if (PATTERN_INDEX >= NUM_SAMPLES) {
      std::printf("ERROR: PATTERN_INDEX=%d out of range (0..%d)\n", PATTERN_INDEX, NUM_SAMPLES-1);
      return 2;
    }
    s_begin = PATTERN_INDEX;
    s_end   = PATTERN_INDEX + 1;
  }

  int total_mis = 0;

  for (int s = s_begin; s < s_end; ++s) {
    ac_channel<fx_utils::fx_t> q_in, k_in, v_in, out_ch;

    const double* q_ptr = &trace_layer0_attn_Q_step0_tensor[s * LEN];
    const double* k_ptr = &trace_layer0_attn_K_step0_tensor[s * LEN];
    const double* v_ptr = &trace_layer0_attn_V_step0_tensor[s * LEN];

    for (int i = 0; i < LEN; ++i) {
      q_in.write(fx_utils::fx_t(ac_fixed<64,32,true,AC_RND_CONV,AC_SAT_SYM>(q_ptr[i])));
      k_in.write(fx_utils::fx_t(ac_fixed<64,32,true,AC_RND_CONV,AC_SAT_SYM>(k_ptr[i])));
      v_in.write(fx_utils::fx_t(ac_fixed<64,32,true,AC_RND_CONV,AC_SAT_SYM>(v_ptr[i])));
    }

    dut.run(q_in, k_in, v_in, out_ch);

    for (int i = 0; i < LEN; ++i) out_buf[i] = out_ch.read();

    const double* exp_ptr = &trace_layer0_attn_post_concat_step0_tensor[s * LEN];

    char name[64];
    std::snprintf(name, sizeof(name), "layer0_attn_post_concat(sample=%d)", s);
    int mis = compare_array_abs<double, fx_utils::fx_t>(exp_ptr, out_buf, LEN, (double)ATOL, name);
    total_mis += mis;

#if FAIL_FAST
    if (mis) break;
#endif
  }

  if (total_mis == 0) {
    std::printf("==== PASS ====" "\n");
    return 0;
  } else {
    std::printf("==== FAIL: total mismatches = %d ====" "\n", total_mis);
    return 1;
  }
}
