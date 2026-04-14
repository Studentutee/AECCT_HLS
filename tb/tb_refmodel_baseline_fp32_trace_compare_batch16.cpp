#include "AECCT_ac_ref/include/RefModel.h"
#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"
#include <cstdio>
#include <vector>
#include <cmath>

int main() {
  constexpr int B = 16;
  constexpr int N = 63;
  aecct_ref::RefModel model;
  std::vector<double> logits(B * N, 0.0);
  std::vector<aecct_ref::bit1_t> xpred(B * N);
  aecct_ref::RefModelIO io{};
  io.input_y_fp32 = trace_input_y_step0_tensor;
  io.B = B;
  io.N = N;
  io.out_logits = logits.data();
  io.out_x_pred = xpred.data();
  model.infer_step0(io);
  int total_xpred_mismatch = 0;
  int first_b = -1, first_i = -1;
  double max_logit_abs = 0.0;
  int max_logit_b = -1, max_logit_i = -1;
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < N; ++i) {
      const int got = xpred[b * N + i].to_int();
      const int exp = (trace_output_x_pred_step0_tensor[b * N + i] != 0.0) ? 1 : 0;
      if (got != exp) {
        ++total_xpred_mismatch;
        if (first_b < 0) { first_b = b; first_i = i; }
      }
      const double diff = std::fabs(logits[b * N + i] - trace_output_logits_step0_tensor[b * N + i]);
      if (diff > max_logit_abs) {
        max_logit_abs = diff;
        max_logit_b = b;
        max_logit_i = i;
      }
    }
  }
  std::printf("[refmodel_baseline_fp32_trace_compare_batch] B=%d total_xpred_mismatch=%d first=(%d,%d) max_logit_abs=%.9f max_logit=(%d,%d)\n",
              B, total_xpred_mismatch, first_b, first_i, max_logit_abs, max_logit_b, max_logit_i);
  return 0;
}
