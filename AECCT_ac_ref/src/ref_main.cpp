#include <cstdio>
#include <cstdlib>

#include "ac_channel.h"

#include "../include/RefMetrics.h"
#include "../synth/RefStep0Synth.h"

#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"

namespace {

static const int VAR_N = 63;

static void print_first_k(const char *name, const double *x, int k) {
  std::printf("%s[0:%d]:\n", name, k);
  for (int i = 0; i < k; ++i) {
    std::printf("  [%d] %.9f\n", i, x[i]);
  }
}

static void run_synth_step0_pattern(
  const double *input_y,
  double *out_logits,
  aecct_ref::bit1_t *out_xpred
) {
  ac_channel<aecct_ref::fp32_ref_t> in_y_ch;
  ac_channel<aecct_ref::fp32_ref_t> out_logits_ch;
  ac_channel<aecct_ref::bit1_t> out_xpred_ch;

  for (int i = 0; i < VAR_N; ++i) {
    in_y_ch.write(aecct_ref::fp32_ref_t(static_cast<float>(input_y[i])));
  }

  aecct_ref::ref_step0_synth(in_y_ch, out_logits_ch, out_xpred_ch);

  for (int i = 0; i < VAR_N; ++i) {
    out_logits[i] = static_cast<double>(out_logits_ch.read().to_float());
    out_xpred[i] = out_xpred_ch.read();
  }
}

} // anonymous namespace

int main(int argc, char **argv) {
  int b_sel = -1;
  if (argc >= 2) {
    b_sel = std::atoi(argv[1]);
  }

  const int B = trace_input_y_step0_tensor_shape[0];
  const int N = trace_input_y_step0_tensor_shape[1];
  const int numel = trace_input_y_step0_tensor_numel;

  if (N != VAR_N) {
    std::printf("Unexpected N=%d, expected %d\n", N, VAR_N);
    return 1;
  }

  if (numel != trace_output_logits_step0_tensor_numel ||
      numel != trace_output_x_pred_step0_tensor_numel) {
    std::printf("Shape mismatch between input and golden outputs.\n");
    return 1;
  }

  if (b_sel >= 0 && (b_sel < 0 || b_sel >= B)) {
    std::printf("Usage: ref_sim [pattern_index]\n");
    std::printf("pattern_index must be in [0, %d)\n", B);
    return 1;
  }

  const int run_B = (b_sel >= 0) ? 1 : B;
  const int run_numel = run_B * N;

  double *out_logits = new double[run_numel];
  aecct_ref::bit1_t *out_xpred = new aecct_ref::bit1_t[run_numel];

  if (b_sel >= 0) {
    std::printf("=== Running single pattern b=%d (B=1, N=%d) ===\n", b_sel, N);
  }

  for (int rb = 0; rb < run_B; ++rb) {
    const int src_b = (b_sel >= 0) ? b_sel : rb;
    const int src_base = src_b * N;
    const int dst_base = rb * N;
    run_synth_step0_pattern(
      &trace_input_y_step0_tensor[src_base],
      &out_logits[dst_base],
      &out_xpred[dst_base]
    );
  }

  const double *golden_logits =
    (b_sel >= 0) ? &trace_output_logits_step0_tensor[b_sel * N]
                 : trace_output_logits_step0_tensor;

  aecct_ref::Metrics m_logits =
    aecct_ref::compute_metrics(golden_logits, out_logits, static_cast<std::size_t>(run_numel));

  std::printf("=== Step0 logits metrics vs golden ===\n");
  std::printf("MSE     : %.6e\n", m_logits.mse);
  std::printf("RMSE    : %.6e\n", m_logits.rmse);
  std::printf("MAE     : %.6e\n", m_logits.mae);
  std::printf("MaxAbs  : %.6e\n", m_logits.max_abs);

  const double *golden_xpred =
    (b_sel >= 0) ? &trace_output_x_pred_step0_tensor[b_sel * N]
                 : trace_output_x_pred_step0_tensor;

  std::size_t match = 0;
  for (int i = 0; i < run_numel; ++i) {
    const int g = (golden_xpred[i] != 0.0) ? 1 : 0;
    const int p = static_cast<int>(out_xpred[i].to_int());
    if (g == p) {
      match++;
    }
  }
  const double acc = (run_numel > 0) ? (100.0 * static_cast<double>(match) / static_cast<double>(run_numel)) : 0.0;
  std::printf("x_pred match: %.2f%% (%zu / %d)\n", acc, match, run_numel);

  print_first_k("golden_logits", golden_logits, 8);
  print_first_k("ref_logits   ", out_logits, 8);

  delete[] out_logits;
  delete[] out_xpred;

  return 0;
}

