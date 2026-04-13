#pragma once

#include "RefTypes.h"

namespace aecct_ref {

template <typename FloatT>
static inline FloatT roundtrip_through_generic_e4m3(const FloatT& x) {
  const ref_generic_e4m3_t e4m3(x);
  return FloatT(e4m3.to_double());
}

} // namespace aecct_ref
