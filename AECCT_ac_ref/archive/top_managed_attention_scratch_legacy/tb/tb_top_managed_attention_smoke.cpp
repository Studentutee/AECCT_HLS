#include <cmath>
#include <cstdio>

#include "AECCT_ac_ref/include/RefStep0ShapeBridge.h"
#include "AECCT_ac_ref/include/top_managed_attention/TopManagedAttentionPipeline.h"

namespace {

static const double kDiffTol = 1.0e-12;

} // namespace

int main() {
  aecct_ref::top_managed_attention::TopManagedAttentionPipeline pipeline;

  double input_y[ModelShapes::N_VARS];
  TB_BUILD_INPUT_LOOP: for (int i = 0; i < ModelShapes::N_VARS; ++i) {
    const int centered = (i % 9) - 4;
    const double bias = ((i & 1) == 0) ? 0.0625 : -0.046875;
    input_y[i] = (static_cast<double>(centered) * 0.125) + bias;
  }

  pipeline.load_input_y_into_top_x_work(input_y, ModelShapes::N_VARS);

  aecct_ref::ref_fp32_t x_before[ModelShapes::T_TOKENS][ModelShapes::D_MODEL];
  TB_SNAPSHOT_TOKEN_LOOP: for (int token = 0; token < ModelShapes::T_TOKENS; ++token) {
    TB_SNAPSHOT_DIM_LOOP: for (int dim = 0; dim < ModelShapes::D_MODEL; ++dim) {
      x_before[token][dim] = pipeline.x_work(token, dim);
    }
  }

  if (!pipeline.run_layer0_attention_skeleton()) {
    std::printf("FAIL: run_layer0_attention_skeleton returned false\n");
    return 1;
  }

  int changed_count = 0;
  TB_COMPARE_TOKEN_LOOP: for (int token = 0; token < ModelShapes::T_TOKENS; ++token) {
    TB_COMPARE_DIM_LOOP: for (int dim = 0; dim < ModelShapes::D_MODEL; ++dim) {
      const double before = static_cast<double>(x_before[token][dim].to_float());
      const double after = static_cast<double>(pipeline.x_work(token, dim).to_float());
      if (std::fabs(after - before) > kDiffTol) {
        ++changed_count;
      }
    }
  }

  aecct_ref::bit1_t x_pred_out[ModelShapes::N_VARS];
  pipeline.export_x_pred_from_top(x_pred_out, ModelShapes::N_VARS);

  int x_pred_ones = 0;
  TB_COUNT_XPRED_LOOP: for (int i = 0; i < ModelShapes::N_VARS; ++i) {
    x_pred_ones += x_pred_out[i].to_int();
  }

  std::printf(
    "[tmattn_smoke] changed=%d/%d top_emit=%d kv_runs=%d qsoftres_runs=%d writeback=%d contract_ok=%d row_major_ok=%d x_pred_ones=%d\n",
    changed_count,
    ModelShapes::T_TOKENS * ModelShapes::D_MODEL,
    pipeline.top_payload_emit_count(),
    pipeline.kv_block_run_count(),
    pipeline.qsoftres_block_run_count(),
    pipeline.top_writeback_count(),
    pipeline.last_contract_ok() ? 1 : 0,
    pipeline.last_writeback_row_major_ok() ? 1 : 0,
    x_pred_ones);

  if (changed_count <= 0) {
    std::printf("FAIL: X_WORK not updated after layer0 attention skeleton\n");
    return 2;
  }
  if (pipeline.top_payload_emit_count() != 2) {
    std::printf("FAIL: Top payload emit count mismatch (expect 2)\n");
    return 3;
  }
  if (pipeline.kv_block_run_count() != 1 || pipeline.qsoftres_block_run_count() != 1) {
    std::printf("FAIL: block run count mismatch\n");
    return 4;
  }
  if (!pipeline.last_contract_ok() || !pipeline.last_writeback_row_major_ok()) {
    std::printf("FAIL: contract or row-major writeback check failed\n");
    return 5;
  }
  if (pipeline.top_writeback_count() != (ModelShapes::T_TOKENS * ModelShapes::D_MODEL)) {
    std::printf("FAIL: writeback count mismatch\n");
    return 6;
  }

  std::printf("PASS: tb_top_managed_attention_smoke\n");
  return 0;
}
