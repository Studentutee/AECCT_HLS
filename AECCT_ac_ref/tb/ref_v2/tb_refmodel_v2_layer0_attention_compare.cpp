#include <cstdio>
#include <vector>

#include "AECCT_ac_ref/include/ref_v2/RefModel_v2.h"

namespace {
static constexpr int VAR_N = 63;
} // namespace

int main() {
  std::vector<double> input(static_cast<std::size_t>(VAR_N), 0.0);
  for (int i = 0; i < VAR_N; ++i) {
    const int s = (i % 13) - 6;
    const double bias = ((i & 1) == 0) ? 0.03125 : -0.046875;
    input[static_cast<std::size_t>(i)] = (static_cast<double>(s) * 0.125) + bias;
  }

  aecct_ref::RefModelIO io{};
  io.input_y_fp32 = input.data();
  io.B = 1;
  io.N = VAR_N;

  aecct_ref::ref_v2::RefModel_v2 model_v2;
  model_v2.set_run_config(aecct_ref::make_fp32_baseline_run_config());
  if (!model_v2.run_step0_layer0_attention_compare(io, 0)) {
    std::printf("FAIL: RefModel_v2 layer0 compare execution failed\n");
    return 1;
  }

  const aecct_ref::ref_v2::RefV2CompareStats stats = model_v2.last_compare_stats();
  std::printf(
    "[tb_ref_v2] attention_input mismatch=%d max_abs_diff=%.9e\n",
    stats.attention_input.mismatch_count,
    stats.attention_input.max_abs_diff);
  std::printf(
    "[tb_ref_v2] SCR_K mismatch=%d max_abs_diff=%.9e\n",
    stats.scr_k.mismatch_count,
    stats.scr_k.max_abs_diff);
  std::printf(
    "[tb_ref_v2] SCR_V mismatch=%d max_abs_diff=%.9e\n",
    stats.scr_v.mismatch_count,
    stats.scr_v.max_abs_diff);
  std::printf(
    "[tb_ref_v2] X_WORK writeback mismatch=%d max_abs_diff=%.9e\n",
    stats.x_work_writeback.mismatch_count,
    stats.x_work_writeback.max_abs_diff);

  if (!stats.all_match) {
    std::printf("FAIL: RefModel_v2 compare mismatch detected\n");
    return 2;
  }

  std::printf("PASS: tb_refmodel_v2_layer0_attention_compare\n");
  return 0;
}
