#pragma once

namespace aecct_ref {

enum class RefLayerNormMode : unsigned char {
  LN_BASELINE = 0,
  LN_SUM_SUMSQ_APPROX = 1
};

static inline const char* to_string(RefLayerNormMode mode) {
  switch (mode) {
    case RefLayerNormMode::LN_BASELINE:
      return "LN_BASELINE";
    case RefLayerNormMode::LN_SUM_SUMSQ_APPROX:
      return "LN_SUM_SUMSQ_APPROX";
    default:
      return "UNKNOWN_LN_MODE";
  }
}

} // namespace aecct_ref
