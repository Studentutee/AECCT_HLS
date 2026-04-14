#include "AECCT_ac_ref/include/RefModel.h"
#include "input_y_step0.h"
#include <cstdio>
#include <vector>

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
  std::printf("[refmodel_baseline_fp32_smoke] logits0=%.6f logits1=%.6f xpred0=%d xpred1=%d\n",
              logits[0], logits[1], (int)xpred[0].to_int(), (int)xpred[1].to_int());
  return 0;
}
