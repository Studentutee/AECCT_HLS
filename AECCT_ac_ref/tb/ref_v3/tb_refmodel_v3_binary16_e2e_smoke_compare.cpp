#include <array>
#include <cstdio>

#if defined(__SYNTHESIS__) || defined(REFV3_SYNTH_ONLY)
#error "tb_refmodel_v3_binary16_e2e_smoke_compare is host-only."
#endif

#include "AECCT_ac_ref/include/RefModel.h"
#include "AECCT_ac_ref/include/ref_v3/RefModel_v3.h"
#include "input_y_step0.h"

namespace {

static constexpr int kVarN = 63;
static constexpr std::array<int, 4> kPatternIndices = {906, 849, 217, 77};

} // namespace

int main() {
  if (trace_input_y_step0_tensor_ndim != 2 || trace_input_y_step0_tensor_shape[1] != kVarN) {
    std::printf("FAIL: unexpected input_y_step0 tensor shape\n");
    return 1;
  }

  const int trace_batch = trace_input_y_step0_tensor_shape[0];
  for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];
    if (pattern_idx < 0 || pattern_idx >= trace_batch) {
      std::printf("FAIL: pattern index out of range: %d (trace_batch=%d)\n", pattern_idx, trace_batch);
      return 1;
    }
  }

  int pass_count = 0;
  int fail_count = 0;

  for (int i = 0; i < static_cast<int>(kPatternIndices.size()); ++i) {
    const int pattern_idx = kPatternIndices[static_cast<std::size_t>(i)];

    aecct_ref::ref_v3::RefModel_v3 model_v3;
    aecct_ref::RefRunConfig cfg = aecct_ref::make_fp32_baseline_run_config();
    cfg.legacy.ln_mode = aecct_ref::RefLayerNormMode::LN_BASELINE;
    model_v3.set_run_config(cfg);

    aecct_ref::RefModelIO io{};
    io.input_y_fp32 = &trace_input_y_step0_tensor[pattern_idx * kVarN];
    io.B = 1;
    io.N = kVarN;

    const bool run_ok = model_v3.run_step0_layer0_attention_compare(io, 0);
    if (!run_ok) {
      ++fail_count;
      std::printf(
        "[ref_v3_binary16_smoke] pattern_idx=%d final_logits={mismatch_count=-1,max_abs_diff=nan,first_mismatch={idx=-1,v2=nan,ref=nan}} "
        "final_x_pred={mismatch_count=-1,first_mismatch={idx=-1,v2=-1,ref=-1}} result=FAIL run_ok=0\n",
        pattern_idx);
      continue;
    }

    const aecct_ref::ref_v3::RefV3CompareStats stats = model_v3.last_compare_stats();
    const aecct_ref::ref_v3::RefV3ComparePoint& logits = stats.final_logits;
    const aecct_ref::ref_v3::RefV3ComparePoint& xpred = stats.final_x_pred;

    const bool pass = (logits.mismatch_count == 0) && (xpred.mismatch_count == 0);
    if (pass) {
      ++pass_count;
    } else {
      ++fail_count;
    }

    std::printf(
      "[ref_v3_binary16_smoke] pattern_idx=%d final_logits={mismatch_count=%d,max_abs_diff=%.9e,first_mismatch={idx=%d,v2=%.9e,ref=%.9e}} "
      "final_x_pred={mismatch_count=%d,first_mismatch={idx=%d,v2=%d,ref=%d}} result=%s run_ok=1\n",
      pattern_idx,
      logits.mismatch_count,
      logits.max_abs_diff,
      logits.first_mismatch_token,
      logits.first_v2_value,
      logits.first_ref_value,
      xpred.mismatch_count,
      xpred.first_mismatch_token,
      static_cast<int>(xpred.first_v2_value),
      static_cast<int>(xpred.first_ref_value),
      pass ? "PASS" : "FAIL");
  }

  std::printf(
    "[ref_v3_binary16_smoke_summary] total=%d pass=%d fail=%d\n",
    static_cast<int>(kPatternIndices.size()),
    pass_count,
    fail_count);

  if (fail_count != 0) {
    std::printf("FAIL: tb_refmodel_v3_binary16_e2e_smoke_compare\n");
    return 2;
  }

  std::printf("PASS: tb_refmodel_v3_binary16_e2e_smoke_compare\n");
  return 0;
}
