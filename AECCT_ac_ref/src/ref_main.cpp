#include <cstdio>
#include <cstdlib>

#include "../include/RefModel.h"
#include "../include/RefMetrics.h"

// Traces / goldens
#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"

// Weights (used by RefModel.cpp)
#include "weights.h"

namespace {

static void print_first_k(const char* name, const double* x, int k) {
  std::printf("%s[0:%d]:\n", name, k);
  for (int i = 0; i < k; ++i) {
    std::printf("  [%d] %.9f\n", i, x[i]);
  }
}

} // anonymous namespace

int main(int argc, char** argv) {
  // Optional arg: pattern index (batch index). If omitted, run all patterns.
  int b_sel = -1;
  if (argc >= 2) {
    b_sel = std::atoi(argv[1]);
  }

  // Basic shape checks
  const int B = trace_input_y_step0_tensor_shape[0];
  const int N = trace_input_y_step0_tensor_shape[1];
  const int numel = trace_input_y_step0_tensor_numel;

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
  const int run_numel = (b_sel >= 0) ? N : numel;

  // Allocate (heap is fine for PC ref harness)
  aecct_ref::act_t* in_q = new aecct_ref::act_t[run_numel];
  double* out_logits = new double[run_numel];
  aecct_ref::bit1_t* out_xpred = new aecct_ref::bit1_t[run_numel];

  // Quantize input into act_t
  if (b_sel >= 0) {
    const int base = b_sel * N;
    for (int i = 0; i < N; ++i) {
      in_q[i] = aecct_ref::act_from_double(trace_input_y_step0_tensor[base + i]);
    }
    std::printf("=== Running single pattern b=%d (B=1, N=%d) ===\n", b_sel, N);
  } else {
    for (int i = 0; i < numel; ++i) {
      in_q[i] = aecct_ref::act_from_double(trace_input_y_step0_tensor[i]);
    }
  }

  // Run ref model
  aecct_ref::RefModel model;
  if (b_sel >= 0) {
    static char dump_dir[256];
    std::snprintf(dump_dir, sizeof(dump_dir), "logs/ref_cpp/pattern_%d", b_sel);

    aecct_ref::RefDumpConfig cfg;
    cfg.enabled = true;
    cfg.dump_dir = dump_dir;
    cfg.pattern_index = b_sel;
    model.set_dump_config(cfg);
    std::printf("checkpoint dump dir: %s\n", dump_dir);
  } else {
    model.clear_dump_config();
  }

  aecct_ref::RefModelIO io;
  io.input_y = in_q;
  io.input_y_fp32 = (b_sel >= 0) ? &trace_input_y_step0_tensor[b_sel * N]
                                 : trace_input_y_step0_tensor;
  io.out_logits = out_logits;
  io.out_x_pred = out_xpred;
  io.B = run_B;
  io.N = N;

  model.infer_step0(io);

  // Compare logits (double) against golden
  const double* golden_logits =
    (b_sel >= 0) ? &trace_output_logits_step0_tensor[b_sel * N]
                 : trace_output_logits_step0_tensor;

  aecct_ref::Metrics m_logits =
    aecct_ref::compute_metrics(golden_logits, out_logits, (std::size_t)run_numel);

  std::printf("=== Step0 logits metrics vs golden ===\n");
  std::printf("MSE     : %.6e\n", m_logits.mse);
  std::printf("RMSE    : %.6e\n", m_logits.rmse);
  std::printf("MAE     : %.6e\n", m_logits.mae);
  std::printf("MaxAbs  : %.6e\n", m_logits.max_abs);

  // Compare x_pred exact match rate (treat golden as double 0/1)
  const double* golden_xpred =
    (b_sel >= 0) ? &trace_output_x_pred_step0_tensor[b_sel * N]
                 : trace_output_x_pred_step0_tensor;

  std::size_t match = 0;
  for (int i = 0; i < run_numel; ++i) {
    const int g = (golden_xpred[i] != 0.0) ? 1 : 0;
    const int p = (int)out_xpred[i].to_int();
    if (g == p) match++;
  }
  const double acc = (run_numel > 0) ? (100.0 * (double)match / (double)run_numel) : 0.0;
  std::printf("x_pred match: %.2f%% (%zu / %d)\n", acc, match, run_numel);

  // Print a few samples for quick eyeballing
  print_first_k("golden_logits", golden_logits, 8);
  print_first_k("ref_logits   ", out_logits, 8);

  delete[] in_q;
  delete[] out_logits;
  delete[] out_xpred;

  return 0;
}