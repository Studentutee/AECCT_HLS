#pragma once

namespace aecct_ref {

enum class RefSoftmaxExpMode : unsigned char {
  BASELINE_NEAREST_LUT = 0,
  V2_LERP_LUT = 1,
  V3_BASE2_RESERVED = 2
};

static inline const char* to_string(RefSoftmaxExpMode mode) {
  switch (mode) {
    case RefSoftmaxExpMode::BASELINE_NEAREST_LUT:
      return "BASELINE_NEAREST_LUT";
    case RefSoftmaxExpMode::V2_LERP_LUT:
      return "V2_LERP_LUT";
    case RefSoftmaxExpMode::V3_BASE2_RESERVED:
      return "V3_BASE2_RESERVED";
    default:
      return "UNKNOWN_SOFTMAX_EXP_MODE";
  }
}

} // namespace aecct_ref
