#include <cmath>
#include <cstdio>
#include <vector>

#include "AECCT_ac_ref/include/RefModel.h"
#include "AECCT_ac_ref/include/RefModelOptimized.h"

namespace {

constexpr int TOKENS_T = 75;
constexpr int VAR_N = 63;
constexpr int D_MODEL = 32;
constexpr double kCompareTol = 1.0e-4;

} // namespace

int main() {
  using namespace aecct_ref;

  std::vector<double> input(static_cast<std::size_t>(VAR_N), 0.0);
  for (int i = 0; i < VAR_N; ++i) {
    const int s = (i % 11) - 5;
    const double bias = ((i & 1) == 0) ? 0.03125 : -0.046875;
    input[static_cast<std::size_t>(i)] = (static_cast<double>(s) * 0.125) + bias;
  }

  std::vector<double> legacy_logits(static_cast<std::size_t>(VAR_N), 0.0);
  std::vector<bit1_t> legacy_xpred(static_cast<std::size_t>(VAR_N), bit1_t(0));
  std::vector<double> legacy_layer0_pre_ln(
    static_cast<std::size_t>(TOKENS_T * D_MODEL),
    0.0);

  RefModelIO io{};
  io.input_y_fp32 = input.data();
  io.out_logits = legacy_logits.data();
  io.out_x_pred = legacy_xpred.data();
  io.debug.out_layer0_pre_ln_input = legacy_layer0_pre_ln.data();
  io.B = 1;
  io.N = VAR_N;

  RefModel legacy_model;
  legacy_model.set_run_config(make_fp32_baseline_run_config());
  legacy_model.infer_step0(io);

  RefModelOptimized opt_model;
  opt_model.set_run_config(make_fp32_baseline_run_config());
  if (!opt_model.stage_step0_phase_a(io, 0)) {
    std::printf("FAIL: stage_step0_phase_a returned false\n");
    return 1;
  }

  std::vector<double> x_work_before(
    static_cast<std::size_t>(TOKENS_T * D_MODEL),
    0.0);
  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      x_work_before[static_cast<std::size_t>(t * D_MODEL + d)] =
        static_cast<double>(opt_model.x_work(t, d).to_float());
    }
  }

  if (!opt_model.run_step0_layer0_attention_writeback()) {
    std::printf("FAIL: run_step0_layer0_attention_writeback returned false\n");
    return 2;
  }
  if (!opt_model.layer0_attn_writeback_valid()) {
    std::printf("FAIL: layer0_attn_writeback_valid flag is false\n");
    return 3;
  }

  std::size_t changed_count = 0;
  std::size_t mismatch_count = 0;
  double max_abs_diff = 0.0;
  int first_mismatch_t = -1;
  int first_mismatch_d = -1;
  double first_mismatch_opt = 0.0;
  double first_mismatch_ref = 0.0;

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      const std::size_t idx = static_cast<std::size_t>(t * D_MODEL + d);
      const double before = x_work_before[idx];
      const double opt_after = static_cast<double>(opt_model.x_work(t, d).to_float());
      const double legacy_ref = legacy_layer0_pre_ln[idx];

      if (std::fabs(opt_after - before) > 0.0) {
        ++changed_count;
      }

      const double abs_diff = std::fabs(opt_after - legacy_ref);
      if (abs_diff > max_abs_diff) {
        max_abs_diff = abs_diff;
      }
      if (abs_diff > kCompareTol) {
        ++mismatch_count;
        if (first_mismatch_t < 0) {
          first_mismatch_t = t;
          first_mismatch_d = d;
          first_mismatch_opt = opt_after;
          first_mismatch_ref = legacy_ref;
        }
      }
    }
  }

  std::printf(
    "[opt_step4] writeback_valid=1 changed=%u/%u max_abs_diff_vs_legacy_pre_ln=%.9g "
    "mismatch_gt_tol=%u tol=%.1e\n",
    static_cast<unsigned>(changed_count),
    static_cast<unsigned>(TOKENS_T * D_MODEL),
    max_abs_diff,
    static_cast<unsigned>(mismatch_count),
    kCompareTol);

  if (first_mismatch_t >= 0) {
    std::printf(
      "[opt_step4] first_mismatch token=%d dim=%d opt=%.9g ref=%.9g abs=%.9g\n",
      first_mismatch_t,
      first_mismatch_d,
      first_mismatch_opt,
      first_mismatch_ref,
      std::fabs(first_mismatch_opt - first_mismatch_ref));
  }

  if (changed_count == 0) {
    std::printf("FAIL: X_WORK was not changed by layer0 attention writeback\n");
    return 4;
  }
  if (mismatch_count != 0) {
    std::printf("FAIL: optimized writeback mismatch against legacy layer0 pre_ln boundary\n");
    return 5;
  }

  std::printf("PASS: tb_refmodel_optimized_step4_layer0_attn_smoke\n");
  return 0;
}
