#pragma once
#include <cmath>
#include <cstddef>

namespace aecct_ref {

struct Metrics {
  double mse;
  double mae;
  double max_abs;
  double rmse;
};

static inline Metrics compute_metrics(const double* ref, const double* dut, std::size_t n) {
  Metrics m{};
  double se_sum = 0.0;
  double ae_sum = 0.0;
  double max_abs = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const double e = dut[i] - ref[i];
    const double ae = std::fabs(e);
    se_sum += e * e;
    ae_sum += ae;
    if (ae > max_abs) max_abs = ae;
  }

  m.mse = (n > 0) ? (se_sum / (double)n) : 0.0;
  m.mae = (n > 0) ? (ae_sum / (double)n) : 0.0;
  m.rmse = std::sqrt(m.mse);
  m.max_abs = max_abs;
  return m;
}

} // namespace aecct_ref
