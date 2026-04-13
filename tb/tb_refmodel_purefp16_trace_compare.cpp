#include "AECCT_ac_ref/include/RefModel.h"
#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"
#include <cstdio>
#include <vector>
#include <cmath>

int main() {
  constexpr int B = 1;
  constexpr int N = 63;
  aecct_ref::RefModel model;
  aecct_ref::RefModelIO io{};
  io.input_y_fp32 = trace_input_y_step0_tensor;
  io.B = B;
  io.N = N;
  std::vector<double> logits(B * N, 0.0);
  std::vector<aecct_ref::bit1_t> xpred(B * N);
  io.out_logits = logits.data();
  io.out_x_pred = xpred.data();
  model.infer_step0(io);

  int xpred_mismatch = 0;
  int first_xpred = -1;
  double max_logit_abs = 0.0;
  int max_logit_idx = -1;
  for (int i = 0; i < N; ++i) {
    const int got = xpred[i].to_int();
    const int exp = (trace_output_x_pred_step0_tensor[i] != 0.0) ? 1 : 0;
    if (got != exp) {
      ++xpred_mismatch;
      if (first_xpred < 0) first_xpred = i;
    }
    const double diff = std::fabs(logits[i] - trace_output_logits_step0_tensor[i]);
    if (diff > max_logit_abs) {
      max_logit_abs = diff;
      max_logit_idx = i;
    }
  }
  std::printf("[refmodel_purefp16_trace_compare] xpred_mismatch=%d first_xpred=%d max_logit_abs=%.9f max_logit_idx=%d\n",
              xpred_mismatch, first_xpred, max_logit_abs, max_logit_idx);
  return 0;
}
