#pragma once

#include "ac_fixed.h"
#include "ac_int.h"

// ============================================================
// Fixed-point utilities (synth-friendly)
//
// NOTE:
// - 全部使用 ac_datatype (ac_fixed / ac_int)，不使用 double/float。
// - sqrt 以 Newton-Raphson 做近似（固定迭代次數，容易合成）。
// - LayerNorm：mean/var 用較寬的累加型別。
// ============================================================

namespace fx_utils {

using fx_t  = ac_fixed<32, 16, true,  AC_RND_CONV, AC_SAT_SYM>;
using ufx_t = ac_fixed<32, 16, false, AC_RND_CONV, AC_SAT_SYM>;

// Wider for accumulation
using acc_t  = ac_fixed<48, 24, true,  AC_RND_CONV, AC_SAT_SYM>;
using uacc_t = ac_fixed<48, 24, false, AC_RND_CONV, AC_SAT_SYM>;

static inline fx_t relu(const fx_t &x) {
  return (x > fx_t(0)) ? x : fx_t(0);
}

// ------------------------------------------------------------
// sqrt(x) for x >= 0 (Newton-Raphson)
// y_{k+1} = 0.5 * (y_k + x / y_k)
// ------------------------------------------------------------
template<typename T>
static inline T sqrt_nr(const T &x) {
  if (x <= T(0)) return T(0);
  T y = x;
  if (x < T(1)) y = T(1);
  for (int it = 0; it < 6; ++it) {
    y = (y + x / y) / T(2);
  }
  return y;
}

// ------------------------------------------------------------
// LayerNorm over last dimension D:
// y[d] = (x[d]-mean)/sqrt(var+eps) * gamma[d] + beta[d]
// ------------------------------------------------------------
template<int D>
static inline void layer_norm(const fx_t x[D],
                              const fx_t gamma[D],
                              const fx_t beta[D],
                              fx_t y[D]) {
  // mean
  acc_t sum = 0;
  for (int i = 0; i < D; ++i) sum += acc_t(x[i]);
  fx_t mean = fx_t(sum / acc_t(D));

  // variance
  uacc_t vsum = 0;
  for (int i = 0; i < D; ++i) {
    fx_t t = x[i] - mean;
    uacc_t tt = uacc_t(t * t);
    vsum += tt;
  }
  ufx_t var = ufx_t(vsum / uacc_t(D));

  const ufx_t eps = ufx_t(0.00001);
  ufx_t std = ufx_t(sqrt_nr<ufx_t>(var + eps));
  ufx_t inv_std = (std == ufx_t(0)) ? ufx_t(0) : ufx_t(ufx_t(1) / std);

  for (int i = 0; i < D; ++i) {
    fx_t norm = fx_t((x[i] - mean) * fx_t(inv_std));
    y[i] = fx_t(norm * gamma[i] + beta[i]);
  }
}

} // namespace fx_utils
