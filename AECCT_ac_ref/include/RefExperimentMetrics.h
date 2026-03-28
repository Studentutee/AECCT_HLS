#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

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
  std::size_t sign_flip_count;
  double baseline_min_abs_margin;
  double experiment_min_abs_margin;
  struct TopKEntry {
    std::size_t index;
    double value;
    double abs_value;
  };
  std::vector<TopKEntry> baseline_topk_smallest_abs;
  std::vector<TopKEntry> experiment_topk_smallest_abs;
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

static inline bool is_finite_double(double x) {
  return !std::isnan(x) && !std::isinf(x);
}

static inline std::size_t count_sign_flips_between_logits(
  const double* baseline_logits,
  const double* experiment_logits,
  std::size_t n
) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double b = baseline_logits[i];
    const double e = experiment_logits[i];
    if (!is_finite_double(b) || !is_finite_double(e)) {
      continue;
    }
    if ((b > 0.0 && e < 0.0) || (b < 0.0 && e > 0.0)) {
      count++;
    }
  }
  return count;
}

static inline double compute_min_abs_margin(const double* logits, std::size_t n) {
  double min_margin = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    if (!is_finite_double(logits[i])) {
      continue;
    }
    const double abs_v = std::fabs(logits[i]);
    if (abs_v < min_margin) {
      min_margin = abs_v;
    }
  }
  return min_margin;
}

static inline std::vector<RefExperimentCompareMetrics::TopKEntry> compute_topk_smallest_abs_logits(
  const double* logits,
  std::size_t n,
  std::size_t k
) {
  std::vector<RefExperimentCompareMetrics::TopKEntry> entries;
  entries.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (!is_finite_double(logits[i])) {
      continue;
    }
    RefExperimentCompareMetrics::TopKEntry e{};
    e.index = i;
    e.value = logits[i];
    e.abs_value = std::fabs(logits[i]);
    entries.push_back(e);
  }
  std::sort(entries.begin(), entries.end(),
    [](const RefExperimentCompareMetrics::TopKEntry& a,
       const RefExperimentCompareMetrics::TopKEntry& b) {
      if (a.abs_value != b.abs_value) {
        return a.abs_value < b.abs_value;
      }
      return a.index < b.index;
    });
  if (entries.size() > k) {
    entries.resize(k);
  }
  return entries;
}

static inline RefExperimentCompareMetrics compute_experiment_compare_metrics(
  const double* baseline_logits,
  const double* experiment_logits,
  const bit1_t* baseline_x_pred,
  const bit1_t* experiment_x_pred,
  std::size_t logits_count,
  std::size_t x_pred_count,
  std::size_t topk
) {
  RefExperimentCompareMetrics m{};
  m.logits_diff = compute_metrics(baseline_logits, experiment_logits, logits_count);
  m.baseline_logits_nonfinite = count_nonfinite_values(baseline_logits, logits_count);
  m.experiment_logits_nonfinite = count_nonfinite_values(experiment_logits, logits_count);
  m.sign_flip_count = count_sign_flips_between_logits(baseline_logits, experiment_logits, logits_count);
  m.baseline_min_abs_margin = compute_min_abs_margin(baseline_logits, logits_count);
  m.experiment_min_abs_margin = compute_min_abs_margin(experiment_logits, logits_count);
  m.baseline_topk_smallest_abs = compute_topk_smallest_abs_logits(baseline_logits, logits_count, topk);
  m.experiment_topk_smallest_abs = compute_topk_smallest_abs_logits(experiment_logits, logits_count, topk);

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
