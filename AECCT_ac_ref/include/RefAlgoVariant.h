#pragma once

namespace aecct_ref {

enum class RefAlgoVariant : unsigned char {
  BASELINE_SPEC_FLOW = 0,
  RESERVED_SOFTMAX_ALT = 1,
  RESERVED_FINALHEAD_ALT = 2
};

static inline const char* to_string(RefAlgoVariant variant) {
  switch (variant) {
    case RefAlgoVariant::BASELINE_SPEC_FLOW:
      return "BASELINE_SPEC_FLOW";
    case RefAlgoVariant::RESERVED_SOFTMAX_ALT:
      return "RESERVED_SOFTMAX_ALT";
    case RefAlgoVariant::RESERVED_FINALHEAD_ALT:
      return "RESERVED_FINALHEAD_ALT";
    default:
      return "UNKNOWN_ALGO_VARIANT";
  }
}

} // namespace aecct_ref
