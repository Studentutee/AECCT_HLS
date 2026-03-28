#pragma once

#include <cmath>
#include <cstddef>

#include "RefMetrics.h"
#include "RefTypes.h"

namespace aecct_ref {

struct RefNonFiniteCounters {
  std::size_t nan_count;
  std::size_t inf_count;
};

struct RefExperimentCompareMetrics {
  Metrics logits_diff;
  std::size_t x_pred_mismatch_count;
  double x_pred_mismatch_ratio;
  RefNonFiniteCounters baseline_logits_nonfinite;
  RefNonFiniteCounters experiment_logits_nonfinite;
};

static inline RefNonFiniteCounters count_nonfinite_values(const double* x, std::size_t n) {
  RefNonFiniteCounters counters{};
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isnan(x[i])) {
      counters.nan_count++;
    } else if (std::isinf(x[i])) {
      counters.inf_count++;
    }
  }
  return counters;
}

static inline RefExperimentCompareMetrics compute_experiment_compare_metrics(
  const double* baseline_logits,
  const double* experiment_logits,
  const bit1_t* baseline_x_pred,
  const bit1_t* experiment_x_pred,
  std::size_t logits_count,
  std::size_t x_pred_count
) {
  RefExperimentCompareMetrics m{};
  m.logits_diff = compute_metrics(baseline_logits, experiment_logits, logits_count);
  m.baseline_logits_nonfinite = count_nonfinite_values(baseline_logits, logits_count);
  m.experiment_logits_nonfinite = count_nonfinite_values(experiment_logits, logits_count);

  std::size_t mismatch = 0;
  for (std::size_t i = 0; i < x_pred_count; ++i) {
    if (baseline_x_pred[i].to_int() != experiment_x_pred[i].to_int()) {
      mismatch++;
    }
  }

  m.x_pred_mismatch_count = mismatch;
  m.x_pred_mismatch_ratio = (x_pred_count > 0U)
    ? (static_cast<double>(mismatch) / static_cast<double>(x_pred_count))
    : 0.0;
  return m;
}

} // namespace aecct_ref
