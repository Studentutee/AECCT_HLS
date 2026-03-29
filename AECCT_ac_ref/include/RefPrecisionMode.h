#pragma once

namespace aecct_ref {

enum class RefPrecisionMode : unsigned char {
  BASELINE_FP32 = 0,
  GENERIC_E4M3_FINALHEAD = 1,
  FULL_E4M3_NONLINEAR_STRESS = 2,
  GENERIC_E4M3_FRAG_BISECT = 3,
  GENERIC_E4M3_EXCEPT_G5 = 4,
  GENERIC_E4M3_G5_G4 = 5,
  GENERIC_E4M3_G5_G1 = 6,
  GENERIC_E4M3_G5_G3 = 7,
  GENERIC_E4M3_G5_G2 = 8
};

static inline const char* to_string(RefPrecisionMode mode) {
  switch (mode) {
    case RefPrecisionMode::BASELINE_FP32:
      return "BASELINE_FP32";
    case RefPrecisionMode::GENERIC_E4M3_FINALHEAD:
      return "GENERIC_E4M3_FINALHEAD";
    case RefPrecisionMode::FULL_E4M3_NONLINEAR_STRESS:
      return "FULL_E4M3_NONLINEAR_STRESS";
    case RefPrecisionMode::GENERIC_E4M3_FRAG_BISECT:
      return "GENERIC_E4M3_FRAG_BISECT";
    case RefPrecisionMode::GENERIC_E4M3_EXCEPT_G5:
      return "GENERIC_E4M3_EXCEPT_G5";
    case RefPrecisionMode::GENERIC_E4M3_G5_G4:
      return "GENERIC_E4M3_G5_G4";
    case RefPrecisionMode::GENERIC_E4M3_G5_G1:
      return "GENERIC_E4M3_G5_G1";
    case RefPrecisionMode::GENERIC_E4M3_G5_G3:
      return "GENERIC_E4M3_G5_G3";
    case RefPrecisionMode::GENERIC_E4M3_G5_G2:
      return "GENERIC_E4M3_G5_G2";
    default:
      return "UNKNOWN_PRECISION_MODE";
  }
}

} // namespace aecct_ref
