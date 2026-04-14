#include <cmath>
#include <cstdio>
#include <vector>

#include "AECCT_ac_ref/include/RefModel.h"
#include "AECCT_ac_ref/include/RefModelOptimized.h"
#include "data/weights/weights.h"

namespace {

constexpr int TOKENS_T = 75;
constexpr int VAR_N = 63;
constexpr int D_MODEL = 32;
constexpr double kStageTol = 1.0e-4;
constexpr double kHybridLogitsTol = 1.0e-9;

void apply_layernorm_token_exact(
  const double* x_token,
  const double* w,
  const double* b,
  double* y_token) {
  constexpr double kEps = 1.0e-5;

  double sum = 0.0;
  for (int d = 0; d < D_MODEL; ++d) {
    sum += x_token[d];
  }
  const double mean = sum / static_cast<double>(D_MODEL);

  double var_acc = 0.0;
  for (int d = 0; d < D_MODEL; ++d) {
    const double dv = x_token[d] - mean;
    var_acc += (dv * dv);
  }
  const double var = var_acc / static_cast<double>(D_MODEL);
  const double inv_std = 1.0 / std::sqrt(var + kEps);

  for (int d = 0; d < D_MODEL; ++d) {
    const double xn = (x_token[d] - mean) * inv_std;
    y_token[d] = (xn * w[d]) + b[d];
  }
}

void build_layer1_attn_input_boundary_from_layer0_residual(
  const std::vector<double>& layer0_ffn_residual,
  std::vector<double>& layer1_attn_input_boundary) {
  layer1_attn_input_boundary.assign(static_cast<std::size_t>(TOKENS_T * D_MODEL), 0.0);
  for (int t = 0; t < TOKENS_T; ++t) {
    apply_layernorm_token_exact(
      &layer0_ffn_residual[static_cast<std::size_t>(t * D_MODEL)],
      w_decoder_norm2_weight,
      w_decoder_norm2_bias,
      &layer1_attn_input_boundary[static_cast<std::size_t>(t * D_MODEL)]);
  }
}

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
  std::vector<double> legacy_layer1_attn_input_direct(
    static_cast<std::size_t>(TOKENS_T * D_MODEL),
    0.0);
  std::vector<double> legacy_layer1_attn_input_boundary;

  RefModelIO io_legacy{};
  io_legacy.input_y_fp32 = input.data();
  io_legacy.out_logits = legacy_logits.data();
  io_legacy.out_x_pred = legacy_xpred.data();
  io_legacy.debug.out_layer0_sublayer1_ln_in = legacy_layer0_ffn_residual.data();
  io_legacy.debug.out_layer1_attn_input = legacy_layer1_attn_input_direct.data();
  io_legacy.B = 1;
  io_legacy.N = VAR_N;

  RefModel legacy_model;
  legacy_model.set_run_config(cfg);
  legacy_model.infer_step0(io_legacy);
  build_layer1_attn_input_boundary_from_layer0_residual(
    legacy_layer0_ffn_residual,
    legacy_layer1_attn_input_boundary);

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
  if (!opt_model.run_step0_layer0_ffn_writeback()) {
    std::printf("FAIL: run_step0_layer0_ffn_writeback returned false\n");
    return 4;
  }
  if (!opt_model.run_step0_mid_norm_writeback()) {
    std::printf("FAIL: run_step0_mid_norm_writeback returned false\n");
    return 5;
  }
  if (!opt_model.mid_norm_writeback_valid()) {
    std::printf("FAIL: mid_norm_writeback_valid flag is false before handoff\n");
    return 6;
  }

  std::vector<double> x_work_before_handoff(
    static_cast<std::size_t>(TOKENS_T * D_MODEL),
    0.0);
  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      x_work_before_handoff[static_cast<std::size_t>(t * D_MODEL + d)] =
        static_cast<double>(opt_model.x_work(t, d).to_float());
    }
  }

  if (!opt_model.run_step0_layer1_attn_input_handoff()) {
    std::printf("FAIL: run_step0_layer1_attn_input_handoff returned false\n");
    return 7;
  }
  if (!opt_model.layer1_attn_input_handoff_valid()) {
    std::printf("FAIL: layer1_attn_input_handoff_valid flag is false\n");
    return 8;
  }

  std::size_t handoff_overwrite_count = 0;
  std::size_t mismatch_count = 0;
  double max_abs_diff = 0.0;
  std::size_t direct_tap_mismatch_count = 0;
  double direct_tap_max_abs_diff = 0.0;
  int first_mismatch_t = -1;
  int first_mismatch_d = -1;
  double first_mismatch_opt = 0.0;
  double first_mismatch_ref = 0.0;
  int first_direct_tap_mismatch_t = -1;
  int first_direct_tap_mismatch_d = -1;

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      const std::size_t idx = static_cast<std::size_t>(t * D_MODEL + d);
      const double opt_handoff = static_cast<double>(opt_model.x_work(t, d).to_float());
      const double ref_handoff = legacy_layer1_attn_input_boundary[idx];
      const double ref_handoff_direct = legacy_layer1_attn_input_direct[idx];
      if (std::fabs(opt_handoff - x_work_before_handoff[idx]) > 0.0) {
        ++handoff_overwrite_count;
      }
      const double abs_diff = std::fabs(opt_handoff - ref_handoff);
      if (abs_diff > max_abs_diff) {
        max_abs_diff = abs_diff;
      }
      if (abs_diff > kStageTol) {
        ++mismatch_count;
        if (first_mismatch_t < 0) {
          first_mismatch_t = t;
          first_mismatch_d = d;
          first_mismatch_opt = opt_handoff;
          first_mismatch_ref = ref_handoff;
        }
      }
      const double abs_diff_direct = std::fabs(opt_handoff - ref_handoff_direct);
      if (abs_diff_direct > direct_tap_max_abs_diff) {
        direct_tap_max_abs_diff = abs_diff_direct;
      }
      if (abs_diff_direct > kStageTol) {
        ++direct_tap_mismatch_count;
        if (first_direct_tap_mismatch_t < 0) {
          first_direct_tap_mismatch_t = t;
          first_direct_tap_mismatch_d = d;
        }
      }
    }
  }

  std::printf(
    "[opt_step7][stage] layer1_attn_input_handoff_valid=1 handoff_overwrite=%u/%u "
    "max_abs_diff_vs_legacy_layer1_attn_input_boundary=%.9g mismatch_gt_tol=%u tol=%.1e\n",
    static_cast<unsigned>(handoff_overwrite_count),
    static_cast<unsigned>(TOKENS_T * D_MODEL),
    max_abs_diff,
    static_cast<unsigned>(mismatch_count),
    kStageTol);
  if (first_mismatch_t >= 0) {
    std::printf(
      "[opt_step7][stage] first_mismatch token=%d dim=%d opt=%.9g ref=%.9g abs=%.9g\n",
      first_mismatch_t,
      first_mismatch_d,
      first_mismatch_opt,
      first_mismatch_ref,
      std::fabs(first_mismatch_opt - first_mismatch_ref));
  }
  std::printf(
    "[opt_step7][stage_info] direct_legacy_out_layer1_attn_input_max_abs_diff=%.9g "
    "direct_mismatch_gt_tol=%u first_direct_mismatch=(%d,%d)\n",
    direct_tap_max_abs_diff,
    static_cast<unsigned>(direct_tap_mismatch_count),
    first_direct_tap_mismatch_t,
    first_direct_tap_mismatch_d);

  if (handoff_overwrite_count != 0) {
    std::printf("FAIL: layer1 attention input handoff overwrote X_WORK\n");
    return 9;
  }
  if (mismatch_count != 0) {
    std::printf("FAIL: optimized handoff boundary mismatch against legacy-derived layer1 attention input boundary\n");
    return 10;
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
  if (!opt_model.layer1_attn_input_handoff_valid()) {
    std::printf("FAIL: infer_step0 did not leave layer1_attn_input_handoff_valid asserted\n");
    return 11;
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
    "[opt_step7][hybrid] x_pred_mismatch=%u/%u first_mismatch=%d "
    "logits_maxabs=%.9g logits_mae=%.9g logits_mse=%.9g\n",
    static_cast<unsigned>(xpred_mismatch_count),
    static_cast<unsigned>(VAR_N),
    first_xpred_mismatch,
    logits_maxabs,
    logits_mae,
    logits_mse);

  if (xpred_mismatch_count != 0) {
    std::printf("FAIL: hybrid final x_pred mismatch against legacy path\n");
    return 12;
  }
  if (logits_maxabs > kHybridLogitsTol) {
    std::printf("FAIL: hybrid final logits max abs diff too large (tol=%.1e)\n", kHybridLogitsTol);
    return 13;
  }

  std::printf("PASS: tb_refmodel_optimized_step7_layer1_attn_input_handoff_smoke\n");
  return 0;
}
