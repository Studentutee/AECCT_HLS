#pragma once
// Active Preproc mainline baseline.
// Input: trace y values for one sample.
// Intermediate: var/check/node features in pure fp16-domain bring-up logic.
// Output: X_WORK-style fp16 tensor for downstream attention handoff compare.
// Boundary: no SRAM, no fp32 carrier, no legacy dual-path on active path.

#include <cstdint>

#include <ac_int.h>
#include <ac_std_float.h>

#include "ModelShapes.h"
#include "weights.h"

namespace aecct {
namespace clean {

typedef ac_std_float<16, 5> fp16_t;
typedef ac_int<16, false> u16_t;

struct PreprocFp16Debug {
  fp16_t var_feature[CODE_N];
  fp16_t check_feature[CODE_C];
  fp16_t node_feature[N_NODES];
  fp16_t preproc_x[N_NODES][D_MODEL];
  fp16_t x_work[N_NODES][D_MODEL];
};

static inline u16_t bits_from_fp16(const fp16_t &x) {
  ac_int<16, true> raw = x.data_ac_int();
  return static_cast<u16_t>(raw);
}

static inline fp16_t fp16_from_double(double x) {
  return fp16_t(static_cast<float>(x));
}

static inline fp16_t fp16_abs(fp16_t x) {
  return (x < fp16_t(0.0f)) ? fp16_t(static_cast<float>(-x.to_float())) : x;
}

static inline fp16_t fp16_roundtrip(fp16_t x) {
  return fp16_t(x.to_ac_float());
}

static inline void run_preproc_fp16_clean(
    const double input_y[CODE_N],
    PreprocFp16Debug &dbg) {
  PREPROC_VAR_LOOP: for (uint32_t v = 0; v < CODE_N; ++v) {
    const fp16_t y = fp16_from_double(input_y[v]);
    dbg.var_feature[v] = fp16_abs(y);
    dbg.node_feature[v] = dbg.var_feature[v];
  }

  PREPROC_CHECK_LOOP: for (uint32_t c = 0; c < CODE_C; ++c) {
    ac_int<1, false> parity = 0;
    PREPROC_CHECK_VAR_LOOP: for (uint32_t v = 0; v < CODE_N; ++v) {
      const uint32_t flat = c * CODE_N + v;
      if ((uint32_t)h_H[flat].to_uint() == 0u) {
        continue;
      }
      if (input_y[v] < 0.0) {
        parity = static_cast<ac_int<1, false> >((uint32_t)parity.to_uint() ^ 1u);
      }
    }
    dbg.check_feature[c] = ((uint32_t)parity.to_uint() == 0u) ? fp16_t(1.0f) : fp16_t(-1.0f);
    dbg.node_feature[CODE_N + c] = dbg.check_feature[c];
  }

  PREPROC_TOKEN_LOOP: for (uint32_t t = 0; t < N_NODES; ++t) {
    PREPROC_SRC_LOOP: for (uint32_t k = 0; k < D_SRC_EMBED; ++k) {
      const fp16_t embed = fp16_from_double(w_src_embed[t * D_SRC_EMBED + k]);
      const fp16_t mul = fp16_roundtrip(fp16_t((dbg.node_feature[t] * embed).to_ac_float()));
      dbg.preproc_x[t][k] = fp16_roundtrip(mul);
      dbg.x_work[t][k] = fp16_roundtrip(dbg.preproc_x[t][k]);
    }
    PREPROC_LPE_LOOP: for (uint32_t k = 0; k < D_LPE_TOKEN; ++k) {
      const fp16_t lpe = fp16_roundtrip(fp16_from_double(w_lpe_token[t * D_LPE_TOKEN + k]));
      dbg.preproc_x[t][D_SRC_EMBED + k] = fp16_roundtrip(lpe);
      dbg.x_work[t][D_SRC_EMBED + k] = fp16_roundtrip(dbg.preproc_x[t][D_SRC_EMBED + k]);
    }
  }
}

static inline void pack_x_work_bits(const PreprocFp16Debug &dbg, u16_t out_bits[N_NODES][D_MODEL]) {
  PACK_TOKEN_LOOP: for (uint32_t t = 0; t < N_NODES; ++t) {
    PACK_D_LOOP: for (uint32_t d = 0; d < D_MODEL; ++d) {
      out_bits[t][d] = bits_from_fp16(dbg.x_work[t][d]);
    }
  }
}

} // namespace clean
} // namespace aecct
