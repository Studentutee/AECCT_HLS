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
#include "embed_plus_SPE_step0.h"
#include "layer0_attn_Q_step0.h"
#include "layer0_attn_K_step0.h"
#include "layer0_attn_V_step0.h"

// DUT
#include "Layer0.h"

int main() {
  std::printf("==== tb_Layer0_QKVProj start ====" "\n");
  std::printf("ATOL = %.10g\n", (double)ATOL);

  constexpr int NUM_SAMPLES = trace_embed_plus_SPE_step0_tensor_shape[0];
  constexpr int IN_H = trace_embed_plus_SPE_step0_tensor_shape[1];  // 75
  constexpr int IN_W = trace_embed_plus_SPE_step0_tensor_shape[2];  // 32
  constexpr int IN_LEN = IN_H * IN_W;

  static_assert(IN_W == Layer0_QKVProj::D_MODEL, "D_MODEL mismatch");
  static_assert(IN_H == Layer0_QKVProj::N_NODES, "N_NODES mismatch");

  static_assert(trace_layer0_attn_Q_step0_tensor_shape[0] == NUM_SAMPLES, "Q NUM_SAMPLES mismatch");
  static_assert(trace_layer0_attn_K_step0_tensor_shape[0] == NUM_SAMPLES, "K NUM_SAMPLES mismatch");
  static_assert(trace_layer0_attn_V_step0_tensor_shape[0] == NUM_SAMPLES, "V NUM_SAMPLES mismatch");

  constexpr int OUT_H = trace_layer0_attn_Q_step0_tensor_shape[1]; // 75
  constexpr int OUT_W = trace_layer0_attn_Q_step0_tensor_shape[2]; // 32
  constexpr int OUT_LEN = OUT_H * OUT_W;

  Layer0_QKVProj dut;

  static ac_fixed<64,32,true> q_buf[OUT_LEN];
  static ac_fixed<64,32,true> k_buf[OUT_LEN];
  static ac_fixed<64,32,true> v_buf[OUT_LEN];

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
    ac_channel<ac_fixed<64,32,true>> x_in_ch;
    ac_channel<ac_fixed<64,32,true>> q_out_ch, k_out_ch, v_out_ch;

    const double* x_ptr = &trace_embed_plus_SPE_step0_tensor[s * IN_LEN];

    // push x (node-major, then d)
    for (int i = 0; i < IN_LEN; ++i) {
      x_in_ch.write(ac_fixed<64,32,true,AC_RND_CONV,AC_SAT_SYM>(x_ptr[i]));
    }

    dut.run(x_in_ch, q_out_ch, k_out_ch, v_out_ch);

    for (int i = 0; i < OUT_LEN; ++i) q_buf[i] = q_out_ch.read();
    for (int i = 0; i < OUT_LEN; ++i) k_buf[i] = k_out_ch.read();
    for (int i = 0; i < OUT_LEN; ++i) v_buf[i] = v_out_ch.read();

    const double* q_exp = &trace_layer0_attn_Q_step0_tensor[s * OUT_LEN];
    const double* k_exp = &trace_layer0_attn_K_step0_tensor[s * OUT_LEN];
    const double* v_exp = &trace_layer0_attn_V_step0_tensor[s * OUT_LEN];

    char name[64];

    std::snprintf(name, sizeof(name), "layer0_attn_Q(sample=%d)", s);
    int mis_q = compare_array_abs<double, ac_fixed<64,32,true>>(q_exp, q_buf, OUT_LEN, (double)ATOL, name);

    std::snprintf(name, sizeof(name), "layer0_attn_K(sample=%d)", s);
    int mis_k = compare_array_abs<double, ac_fixed<64,32,true>>(k_exp, k_buf, OUT_LEN, (double)ATOL, name);

    std::snprintf(name, sizeof(name), "layer0_attn_V(sample=%d)", s);
    int mis_v = compare_array_abs<double, ac_fixed<64,32,true>>(v_exp, v_buf, OUT_LEN, (double)ATOL, name);

    total_mis += (mis_q + mis_k + mis_v);

#if FAIL_FAST
    if (mis_q || mis_k || mis_v) break;
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
