#include <cstdio>
#include <algorithm>
#include <cmath>

#include "ac_channel.h"
#include "ac_fixed.h"
#include "ac_int.h"

#include "compare_array_abs.h"

// ===============================
// Trace headers
// ===============================
// Input to Layer0 = embed_plus_SPE_step0 (N_NODES x D_MODEL)
#include "embed_plus_SPE_step0.h"
#include "norm_mid_out_step0.h"

// Expected output of Layer0
// ------------------------------------------------------------
// Please include the header you generated from the python trace.
// Typical candidates in your notebook:
//   - PATH_norm_mid_out  (decoder.norm2 out)   => final output of Layer0
//   - PATH_l0_norm_ffn_out (sublayer.1 norm)   => before decoder.norm2
//
// Example (rename to your actual header file):
//   #include "norm_mid_out_step0.h"
// And the tensor symbol should look like:
//   trace_norm_mid_out_step0_tensor
//   trace_norm_mid_out_step0_tensor_shape
//
// Uncomment these two lines after you generate the expected header:
// #define L0_EXPECT_READY 1
// #include "norm_mid_out_step0.h"

// ===============================
// DUT
// ===============================
#include "Layer0.h"

// ===============================
// TB knobs
// ===============================
#ifndef ATOL
#define ATOL (1e-3)
#endif

#ifndef FAIL_FAST
#define FAIL_FAST 0
#endif

#ifndef PATTERN_INDEX
#define PATTERN_INDEX (-1)
#endif

// ============================================================
// Helper: cast channel element types
// ============================================================
template <typename Tin, typename Tout>
static void cast_stream(ac_channel<Tin> &in, ac_channel<Tout> &out, int n) {
  for (int i = 0; i < n; ++i) {
    Tin v = in.read();
    out.write(Tout(v));
  }
}

// ============================================================
// A minimal Layer0 wrapper (block-by-block) using Layer0.h
//
// If you already have a top wrapper (e.g. Layer0::run), you can
// replace this whole class with your own DUT call.
// ============================================================
class Layer0_BlockTB {
public:
  static constexpr int N_NODES = PreprocEmbedSPE::N_NODES; // 75
  static constexpr int D_MODEL = PreprocEmbedSPE::D_MODEL; // 32
  static constexpr int D_HID   = 128;

  // NOTE: This implementation assumes the following weight arrays exist in weights.h
  // (names based on the same naming style used elsewhere in Layer0.h).
  // If your weights.h uses different symbols, just rename them here.
  void run(ac_channel<fx_utils::fx_t> &x_in, ac_channel<fx_utils::fx_t> &y_out) {
    // ---------- QKV projection wants ac_fixed<64,32,true> streams ----------
    ac_channel<ac_fixed<64,32,true>> x_in_qkv;
    ac_channel<ac_fixed<64,32,true>> q_fix, k_fix, v_fix;

    // residual path for first Add
    ac_channel<fx_utils::fx_t> residual0;
    ac_channel<fx_utils::fx_t> x_in_forked;

    StreamFork<N_NODES * D_MODEL, fx_utils::fx_t> fork0;
    fork0.run(x_in, x_in_forked, residual0);

    // cast forked stream into qkv input type
    for (int i = 0; i < N_NODES * D_MODEL; ++i) {
      fx_utils::fx_t v = x_in_forked.read();
      x_in_qkv.write(ac_fixed<64,32,true>(v));
    }

    Layer0_QKVProj qkv;
    qkv.run(x_in_qkv, q_fix, k_fix, v_fix);

    // cast Q/K/V into fx_t
    ac_channel<fx_utils::fx_t> q_fx, k_fx, v_fx;
    cast_stream<ac_fixed<64,32,true>, fx_utils::fx_t>(q_fix, q_fx, N_NODES * D_MODEL);
    cast_stream<ac_fixed<64,32,true>, fx_utils::fx_t>(k_fix, k_fx, N_NODES * D_MODEL);
    cast_stream<ac_fixed<64,32,true>, fx_utils::fx_t>(v_fix, v_fx, N_NODES * D_MODEL);

    // ---------- Attention core + out projection ----------
    ac_channel<fx_utils::fx_t> attn_concat;
    ac_channel<fx_utils::fx_t> attn_out;

    L0_AttnCore attn;
    attn.run(q_fx, k_fx, v_fx, attn_concat);

    L0_OutProj outproj;
    outproj.run(attn_concat, attn_out);

    // ---------- Add & Norm (sublayer 0) ----------
    ac_channel<fx_utils::fx_t> add1;
    StreamAdd<N_NODES * D_MODEL> add_0;
    add_0.run(attn_out, residual0, add1);

    ac_channel<fx_utils::fx_t> norm0;
    StreamLayerNorm<N_NODES, D_MODEL> ln0;
    ln0.run(
      add1,
      w_decoder_layers_0_sublayer_0_norm_weight,
      w_decoder_layers_0_sublayer_0_norm_bias,
      norm0);

    // ---------- FFN path needs residual of norm0 ----------
    ac_channel<fx_utils::fx_t> norm0_ffn_in;
    ac_channel<fx_utils::fx_t> residual1;
    StreamFork<N_NODES * D_MODEL, fx_utils::fx_t> fork1;
    fork1.run(norm0, norm0_ffn_in, residual1);

    // ---------- FFN: w1 -> ReLU -> w2 ----------
    ac_channel<fx_utils::fx_t> ffn1;
    ac_channel<fx_utils::fx_t> relu;
    ac_channel<fx_utils::fx_t> ffn2;

    L0_FFN1 ffn_1;
    ffn_1.run(norm0_ffn_in, ffn1);

    StreamReLU<N_NODES * D_HID> relu0;
    relu0.run(ffn1, relu);

    L0_FFN2 ffn_2;
    ffn_2.run(relu, ffn2);

    // ---------- Add & Norm (sublayer 1) ----------
    ac_channel<fx_utils::fx_t> add2;
    StreamAdd<N_NODES * D_MODEL> add_1;
    add_1.run(ffn2, residual1, add2);

    ac_channel<fx_utils::fx_t> norm1;
    StreamLayerNorm<N_NODES, D_MODEL> ln1;
    ln1.run(
      add2,
      w_decoder_layers_0_sublayer_1_norm_weight,
      w_decoder_layers_0_sublayer_1_norm_bias,
      norm1);

    // ---------- Mid-norm (decoder.norm2) ----------
    // If you don't want this (i.e. you want norm1 as Layer0 output),
    // just change the wiring to: y_out = norm1.
    StreamLayerNorm<N_NODES, D_MODEL> midnorm;
    midnorm.run(
      norm1,
      w_decoder_norm2_weight,
      w_decoder_norm2_bias,
      y_out);
  }
};

int main() {
  std::printf("==== Layer0 TB start ====\n");
  std::printf("ATOL = %.10g\n", (double)ATOL);

  // Input tensor shape
  constexpr int NUM_SAMPLES = trace_embed_plus_SPE_step0_tensor_shape[0];
  constexpr int N_NODES     = trace_embed_plus_SPE_step0_tensor_shape[1];
  constexpr int D_MODEL     = trace_embed_plus_SPE_step0_tensor_shape[2];
  constexpr int IN_LEN      = N_NODES * D_MODEL;

  // --------- Expected output tensor ---------
  // TODO: set these to your expected tensor symbol.
  // Example:
  //   constexpr int OUT_H = trace_norm_mid_out_step0_tensor_shape[1];
  //   constexpr int OUT_W = trace_norm_mid_out_step0_tensor_shape[2];
  //   constexpr int OUT_LEN = OUT_H * OUT_W;
  //   const double* exp_ptr = &trace_norm_mid_out_step0_tensor[s * OUT_LEN];

  std::printf("NUM_SAMPLES=%d, N_NODES=%d, D_MODEL=%d (IN_LEN=%d)\n",
              NUM_SAMPLES, N_NODES, D_MODEL, IN_LEN);

  int s_begin = 0;
  int s_end = 100;// NUM_SAMPLES;
  if (PATTERN_INDEX >= 0) {
    if (PATTERN_INDEX >= NUM_SAMPLES) {
      std::printf("ERROR: PATTERN_INDEX=%d out of range (0..%d)\n", PATTERN_INDEX, NUM_SAMPLES - 1);
      return 2;
    }
    s_begin = PATTERN_INDEX;
    s_end   = PATTERN_INDEX + 1;
  }

  // DUT (wrapper)
  Layer0_BlockTB dut;

  int total_mis = 0;
  
  for (int s = s_begin; s < s_end; ++s) {
    ac_channel<fx_utils::fx_t> x_in_ch;
    ac_channel<fx_utils::fx_t> y_out_ch;

    // push input
    const double* in_ptr = &trace_embed_plus_SPE_step0_tensor[s * IN_LEN];
    for (int i = 0; i < IN_LEN; ++i) {
      x_in_ch.write(fx_utils::fx_t(ac_fixed<64,32,true,AC_RND_CONV,AC_SAT_SYM>(in_ptr[i])));
    }

    // run DUT
    dut.run(x_in_ch, y_out_ch);

    // collect outputs
    static fx_utils::fx_t out_buf[IN_LEN];
    for (int i = 0; i < IN_LEN; ++i) {
      out_buf[i] = y_out_ch.read();
    }

//#ifdef L0_EXPECT_READY
    // Expected tensor (update the symbol name to match your generated header)
    constexpr int OUT_LEN = IN_LEN;
    const double* exp_ptr = &trace_encoder_norm_mid_out_step0_tensor[s * OUT_LEN];

    char name[64];
    std::snprintf(name, sizeof(name), "layer0_norm_mid_out(sample=%d)", s);

    int mis = compare_array_abs<double, fx_utils::fx_t>(
      exp_ptr,
      out_buf,
      OUT_LEN,
      (double)ATOL,
      name);

    total_mis += mis;
    if (FAIL_FAST && mis) break;
//#else
//    // No expected header included yet: print a tiny peek for sanity.
//    if (s == s_begin) {
//      std::printf("[INFO] L0_EXPECT_READY not defined; skipping compare. First 8 outputs:\n");
//      for (int i = 0; i < 8; ++i) {
//        std::printf("  out[%d] = %.10g\n", i, (double)out_buf[i].to_double());
//      }
//    }
//#endif
  }

  if (total_mis == 0) {
    std::printf("==== PASS (or compare skipped) ====\n");
    return 0;
  }

  std::printf("==== FAIL: total mismatches = %d ====\n", total_mis);
  return 1;
}
