#include <cmath>
#include <cstdio>
#include <vector>

#include "AECCT_ac_ref/include/RefModel.h"
#include "AECCT_ac_ref/include/RefModelOptimized.h"

namespace {

constexpr int TOKENS_T = 75;
constexpr int VAR_N = 63;
constexpr int D_MODEL = 32;
constexpr double kStageTol = 1.0e-4;
constexpr double kHybridLogitsTol = 1.0e-9;

} // namespace

int main() {
  using namespace aecct_ref;

  std::vector<double> input(static_cast<std::size_t>(VAR_N), 0.0);
  for (int i = 0; i < VAR_N; ++i) {
    const int s = (i % 11) - 5;
    const double bias = ((i & 1) == 0) ? 0.03125 : -0.046875;
    input[static_cast<std::size_t>(i)] = (static_cast<double>(s) * 0.125) + bias;
  }

  RefRunConfig cfg = make_fp32_baseline_run_config();
  cfg.legacy.ln_mode = RefLayerNormMode::LN_EXACT_REFERENCE;

  std::vector<double> legacy_logits(static_cast<std::size_t>(VAR_N), 0.0);
  std::vector<bit1_t> legacy_xpred(static_cast<std::size_t>(VAR_N), bit1_t(0));
  std::vector<double> legacy_layer0_ffn_residual(
    static_cast<std::size_t>(TOKENS_T * D_MODEL),
    0.0);

  RefModelIO io_legacy{};
  io_legacy.input_y_fp32 = input.data();
  io_legacy.out_logits = legacy_logits.data();
  io_legacy.out_x_pred = legacy_xpred.data();
  io_legacy.debug.out_layer0_sublayer1_ln_in = legacy_layer0_ffn_residual.data();
  io_legacy.B = 1;
  io_legacy.N = VAR_N;

  RefModel legacy_model;
  legacy_model.set_run_config(cfg);
  legacy_model.infer_step0(io_legacy);

  RefModelOptimized opt_model;
  opt_model.set_run_config(cfg);
  if (!opt_model.stage_step0_phase_a(io_legacy, 0)) {
    std::printf("FAIL: stage_step0_phase_a returned false\n");
    return 1;
  }
  if (!opt_model.run_step0_layer0_attention_writeback()) {
    std::printf("FAIL: run_step0_layer0_attention_writeback returned false\n");
    return 2;
  }
  if (!opt_model.run_step0_layer0_ln_writeback()) {
    std::printf("FAIL: run_step0_layer0_ln_writeback returned false\n");
    return 3;
  }

  std::vector<double> x_work_after_ln(
    static_cast<std::size_t>(TOKENS_T * D_MODEL),
    0.0);
  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      x_work_after_ln[static_cast<std::size_t>(t * D_MODEL + d)] =
        static_cast<double>(opt_model.x_work(t, d).to_float());
    }
  }

  if (!opt_model.run_step0_layer0_ffn_writeback()) {
    std::printf("FAIL: run_step0_layer0_ffn_writeback returned false\n");
    return 4;
  }
  if (!opt_model.layer0_ffn_writeback_valid()) {
    std::printf("FAIL: layer0_ffn_writeback_valid flag is false\n");
    return 5;
  }

  std::size_t ffn_changed_count = 0;
  std::size_t mismatch_count = 0;
  double max_abs_diff = 0.0;
  int first_mismatch_t = -1;
  int first_mismatch_d = -1;
  double first_mismatch_opt = 0.0;
  double first_mismatch_ref = 0.0;

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      const std::size_t idx = static_cast<std::size_t>(t * D_MODEL + d);
      const double opt_ffn = static_cast<double>(opt_model.x_work(t, d).to_float());
      const double legacy_ffn = legacy_layer0_ffn_residual[idx];
      if (std::fabs(opt_ffn - x_work_after_ln[idx]) > 0.0) {
        ++ffn_changed_count;
      }

      const double abs_diff = std::fabs(opt_ffn - legacy_ffn);
      if (abs_diff > max_abs_diff) {
        max_abs_diff = abs_diff;
      }
      if (abs_diff > kStageTol) {
        ++mismatch_count;
        if (first_mismatch_t < 0) {
          first_mismatch_t = t;
          first_mismatch_d = d;
          first_mismatch_opt = opt_ffn;
          first_mismatch_ref = legacy_ffn;
        }
      }
    }
  }

  std::printf(
    "[opt_step5b][stage] ffn_writeback_valid=1 ffn_changed=%u/%u "
    "max_abs_diff_vs_legacy_l0_ffn_residual=%.9g mismatch_gt_tol=%u tol=%.1e\n",
    static_cast<unsigned>(ffn_changed_count),
    static_cast<unsigned>(TOKENS_T * D_MODEL),
    max_abs_diff,
    static_cast<unsigned>(mismatch_count),
    kStageTol);
  if (first_mismatch_t >= 0) {
    std::printf(
      "[opt_step5b][stage] first_mismatch token=%d dim=%d opt=%.9g ref=%.9g abs=%.9g\n",
      first_mismatch_t,
      first_mismatch_d,
      first_mismatch_opt,
      first_mismatch_ref,
      std::fabs(first_mismatch_opt - first_mismatch_ref));
  }
  if (ffn_changed_count == 0) {
    std::printf("FAIL: layer0 FFN did not change X_WORK on any element\n");
    return 6;
  }
  if (mismatch_count != 0) {
    std::printf("FAIL: optimized layer0 FFN mismatch against legacy layer0 sublayer1 ln-in boundary\n");
    return 7;
  }

  std::vector<double> hybrid_logits(static_cast<std::size_t>(VAR_N), 0.0);
  std::vector<bit1_t> hybrid_xpred(static_cast<std::size_t>(VAR_N), bit1_t(0));
  RefModelIO io_hybrid{};
  io_hybrid.input_y_fp32 = input.data();
  io_hybrid.out_logits = hybrid_logits.data();
  io_hybrid.out_x_pred = hybrid_xpred.data();
  io_hybrid.B = 1;
  io_hybrid.N = VAR_N;

  opt_model.infer_step0(io_hybrid);
  if (!opt_model.layer0_ffn_writeback_valid()) {
    std::printf("FAIL: infer_step0 did not leave layer0_ffn_writeback_valid asserted\n");
    return 8;
  }

  std::size_t xpred_mismatch_count = 0;
  int first_xpred_mismatch = -1;
  double logits_maxabs = 0.0;
  double logits_abs_acc = 0.0;
  double logits_sq_acc = 0.0;
  for (int i = 0; i < VAR_N; ++i) {
    if (legacy_xpred[static_cast<std::size_t>(i)].to_uint() !=
        hybrid_xpred[static_cast<std::size_t>(i)].to_uint()) {
      ++xpred_mismatch_count;
      if (first_xpred_mismatch < 0) {
        first_xpred_mismatch = i;
      }
    }
    const double diff =
      hybrid_logits[static_cast<std::size_t>(i)] - legacy_logits[static_cast<std::size_t>(i)];
    const double ad = std::fabs(diff);
    if (ad > logits_maxabs) {
      logits_maxabs = ad;
    }
    logits_abs_acc += ad;
    logits_sq_acc += diff * diff;
  }
  const double logits_mae = logits_abs_acc / static_cast<double>(VAR_N);
  const double logits_mse = logits_sq_acc / static_cast<double>(VAR_N);

  std::printf(
    "[opt_step5b][hybrid] x_pred_mismatch=%u/%u first_mismatch=%d "
    "logits_maxabs=%.9g logits_mae=%.9g logits_mse=%.9g\n",
    static_cast<unsigned>(xpred_mismatch_count),
    static_cast<unsigned>(VAR_N),
    first_xpred_mismatch,
    logits_maxabs,
    logits_mae,
    logits_mse);

  if (xpred_mismatch_count != 0) {
    std::printf("FAIL: hybrid final x_pred mismatch against legacy path\n");
    return 9;
  }
  if (logits_maxabs > kHybridLogitsTol) {
    std::printf("FAIL: hybrid final logits max abs diff too large (tol=%.1e)\n", kHybridLogitsTol);
    return 10;
  }

  std::printf("PASS: tb_refmodel_optimized_step5b_layer0_ffn_smoke\n");
  return 0;
}
