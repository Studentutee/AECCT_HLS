#pragma once

namespace aecct_ref {

enum class RefPrecisionMode : unsigned char {
  BASELINE_FP32 = 0,
  GENERIC_E4M3_FINALHEAD = 1
};

static inline const char* to_string(RefPrecisionMode mode) {
  switch (mode) {
    case RefPrecisionMode::BASELINE_FP32:
      return "BASELINE_FP32";
    case RefPrecisionMode::GENERIC_E4M3_FINALHEAD:
      return "GENERIC_E4M3_FINALHEAD";
    default:
      return "UNKNOWN_PRECISION_MODE";
  }
}

} // namespace aecct_ref
