#include "../include/RefModel.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../include/RefE4M3Helpers.h"
#include "../include/RefFullQuantStats.h"
#include "../include/InvSqrtApprox.h"
#include "../include/SoftmaxApprox.h"
#include "weights.h"

namespace aecct_ref {
namespace {

typedef ref_fp32_t fp32_ref_t;

static constexpr int TOKENS_T = 75;
static constexpr int VAR_N = 63;
static constexpr int CHECK_N = 12;
static constexpr int D_MODEL = 32;
static constexpr int HEADS = 8;
static constexpr int D_HEAD = 4;
static constexpr int FF_DIM = 128;
static constexpr float LN_EPS_F32 = 1.0e-5f;
static const fp32_ref_t kActQMin = fp32_ref_t(-127.0f);
static const fp32_ref_t kActQMax = fp32_ref_t(127.0f);

struct DumpContext {
  bool enabled;
  std::string root;
};

static inline fp32_ref_t fp32_abs(fp32_ref_t x) {
  return (x < fp32_ref_t(0.0f)) ? (fp32_ref_t(0.0f) - x) : x;
}

static inline fp32_ref_t sign_fp32(fp32_ref_t x) {
  if (x > fp32_ref_t(0.0f)) return fp32_ref_t(1.0f);
  if (x < fp32_ref_t(0.0f)) return fp32_ref_t(-1.0f);
  return fp32_ref_t(0.0f);
}

static inline fp32_ref_t fp32_round(fp32_ref_t x) {
  return x.round();
}

static inline bool use_generic_e4m3_finalhead(const RefRunConfig& cfg) {
  return cfg.precision_mode == RefPrecisionMode::GENERIC_E4M3_FINALHEAD;
}

static inline bool use_full_e4m3_nonlinear_stress(const RefRunConfig& cfg) {
  return cfg.precision_mode == RefPrecisionMode::FULL_E4M3_NONLINEAR_STRESS;
}

static inline bool use_frag_group_bisect(const RefRunConfig& cfg) {
  return cfg.precision_mode == RefPrecisionMode::GENERIC_E4M3_FRAG_BISECT;
}

static inline bool use_generic_e4m3_except_g5(const RefRunConfig& cfg) {
  return cfg.precision_mode == RefPrecisionMode::GENERIC_E4M3_EXCEPT_G5;
}

static inline bool frag_group_includes(RefFragGroup selected, RefFragGroup g) {
  switch (selected) {
    case RefFragGroup::G1_LAYERNORM: return g == RefFragGroup::G1_LAYERNORM;
    case RefFragGroup::G2_RESIDUAL: return g == RefFragGroup::G2_RESIDUAL;
    case RefFragGroup::G3_ATTN_CONTEXT: return g == RefFragGroup::G3_ATTN_CONTEXT;
    case RefFragGroup::G4_SOFTMAX_NEIGHBORHOOD: return g == RefFragGroup::G4_SOFTMAX_NEIGHBORHOOD;
    case RefFragGroup::G5_PREPROC_EMBED: return g == RefFragGroup::G5_PREPROC_EMBED;
    case RefFragGroup::C1_G1_G2:
      return g == RefFragGroup::G1_LAYERNORM || g == RefFragGroup::G2_RESIDUAL;
    case RefFragGroup::C2_G1_G3:
      return g == RefFragGroup::G1_LAYERNORM || g == RefFragGroup::G3_ATTN_CONTEXT;
    case RefFragGroup::C3_G2_G3:
      return g == RefFragGroup::G2_RESIDUAL || g == RefFragGroup::G3_ATTN_CONTEXT;
    case RefFragGroup::C4_G1_G4:
      return g == RefFragGroup::G1_LAYERNORM || g == RefFragGroup::G4_SOFTMAX_NEIGHBORHOOD;
    default:
      return false;
  }
}

static inline bool should_apply_e4m3_group_roundtrip(
  const RefRunConfig& cfg,
  RefFragGroup g
) {
  if (use_full_e4m3_nonlinear_stress(cfg)) {
    return true;
  }
  if (use_generic_e4m3_except_g5(cfg)) {
    return g == RefFragGroup::G1_LAYERNORM ||
           g == RefFragGroup::G2_RESIDUAL ||
           g == RefFragGroup::G3_ATTN_CONTEXT ||
           g == RefFragGroup::G4_SOFTMAX_NEIGHBORHOOD;
  }
  if (!use_frag_group_bisect(cfg)) {
    return false;
  }
  return frag_group_includes(cfg.frag_group, g);
}

static inline bool use_island_s0(const RefRunConfig& cfg) {
  return use_full_e4m3_nonlinear_stress(cfg) ||
         use_generic_e4m3_except_g5(cfg) ||
         use_frag_group_bisect(cfg) ||
         (use_generic_e4m3_finalhead(cfg) && stage_uses_island_s0(cfg.finalhead_stage));
}

static inline bool use_island_s1(const RefRunConfig& cfg) {
  return use_full_e4m3_nonlinear_stress(cfg) ||
         (use_generic_e4m3_finalhead(cfg) && stage_uses_island_s1(cfg.finalhead_stage));
}

static inline bool use_island_s3(const RefRunConfig& cfg) {
  return use_full_e4m3_nonlinear_stress(cfg) ||
         (use_generic_e4m3_finalhead(cfg) && stage_uses_island_s3(cfg.finalhead_stage));
}

static inline void bump_first_nonfinite_block(RefFullQuantStats* stats, const char* block_name) {
  if (stats != nullptr && stats->e4m3.first_nonfinite_block.empty()) {
    stats->e4m3.first_nonfinite_block = block_name;
  }
}

static inline fp32_ref_t stress_roundtrip_e4m3(
  fp32_ref_t x,
  const RefRunConfig& cfg,
  RefFragGroup group,
  RefFullQuantStats* stats,
  const char* block_name
) {
  if (!should_apply_e4m3_group_roundtrip(cfg, group)) {
    return x;
  }

  if (stats != nullptr) {
    stats->e4m3.roundtrip_count++;
    switch (group) {
      case RefFragGroup::G1_LAYERNORM: stats->e4m3.roundtrip_g1_count++; break;
      case RefFragGroup::G2_RESIDUAL: stats->e4m3.roundtrip_g2_count++; break;
      case RefFragGroup::G3_ATTN_CONTEXT: stats->e4m3.roundtrip_g3_count++; break;
      case RefFragGroup::G4_SOFTMAX_NEIGHBORHOOD: stats->e4m3.roundtrip_g4_count++; break;
      case RefFragGroup::G5_PREPROC_EMBED: stats->e4m3.roundtrip_g5_count++; break;
      default: break;
    }
    const float xin = x.to_float();
    if (std::isnan(xin)) {
      stats->e4m3.nan_in_count++;
      bump_first_nonfinite_block(stats, block_name);
    } else if (std::isinf(xin)) {
      stats->e4m3.inf_in_count++;
      bump_first_nonfinite_block(stats, block_name);
    }
  }

  const fp32_ref_t y = roundtrip_through_generic_e4m3(x);
  if (stats != nullptr) {
    const float yout = y.to_float();
    if (std::isnan(yout)) {
      stats->e4m3.nan_out_count++;
      bump_first_nonfinite_block(stats, block_name);
    } else if (std::isinf(yout)) {
      stats->e4m3.inf_out_count++;
      bump_first_nonfinite_block(stats, block_name);
    }
  }
  return y;
}

static inline fp32_ref_t quantize_int8_symmetric(fp32_ref_t x, fp32_ref_t s_x) {
  fp32_ref_t q = fp32_round(x * s_x);
  if (q > kActQMax) q = kActQMax;
  if (q < kActQMin) q = kActQMin;
  return q;
}

static inline int32_t clamp_int32(int32_t x, int32_t lo, int32_t hi) {
  return (x < lo) ? lo : ((x > hi) ? hi : x);
}

static inline int16_t quantize_int8_to_i16(
  fp32_ref_t x,
  float s_x,
  RefFullQuantStats* stats
) {
  const float scaled = x.to_float() * s_x;
  int32_t q = static_cast<int32_t>(std::lround(scaled));
  if (q > 127) {
    q = 127;
    if (stats != nullptr) stats->int_linear.int8_clamp_count++;
  } else if (q < -127) {
    q = -127;
    if (stats != nullptr) stats->int_linear.int8_clamp_count++;
  }
  return static_cast<int16_t>(q);
}

static inline int16_t quantize_weight_to_i16(double w, float s_w) {
  int32_t q = static_cast<int32_t>(std::lround(static_cast<float>(w) * s_w));
  q = clamp_int32(q, -127, 127);
  return static_cast<int16_t>(q);
}

static bool write_npy_f32(const std::string& path,
                          const float* data,
                          std::size_t count,
                          const std::vector<int>& shape) {
  std::filesystem::path out_path(path);
  std::filesystem::create_directories(out_path.parent_path());

  std::ostringstream shape_ss;
  shape_ss << "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0U) shape_ss << ", ";
    shape_ss << shape[i];
  }
  if (shape.size() == 1U) {
    shape_ss << ",";
  }
  shape_ss << ")";

  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " +
                       shape_ss.str() + ", }";

  const std::size_t preamble = 10U;
  const std::size_t align = 16U;
  std::size_t pad = (align - ((preamble + header.size() + 1U) % align)) % align;
  header.append(pad, ' ');
  header.push_back('\n');

  if (header.size() > 65535U) {
    return false;
  }

  std::ofstream ofs(path.c_str(), std::ios::binary);
  if (!ofs.good()) {
    return false;
  }

  const unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  ofs.write(reinterpret_cast<const char*>(magic), 6);
  const unsigned char ver[2] = {1, 0};
  ofs.write(reinterpret_cast<const char*>(ver), 2);

  const uint16_t hlen = static_cast<uint16_t>(header.size());
  unsigned char hlen_le[2];
  hlen_le[0] = static_cast<unsigned char>(hlen & 0xFFU);
  hlen_le[1] = static_cast<unsigned char>((hlen >> 8) & 0xFFU);
  ofs.write(reinterpret_cast<const char*>(hlen_le), 2);

  ofs.write(header.data(), static_cast<std::streamsize>(header.size()));
  ofs.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(count * sizeof(float)));

  return ofs.good();
}

static void dump_tensor(const DumpContext& dump,
                        const char* name,
                        const float* data,
                        std::size_t count,
                        const std::vector<int>& shape) {
  if (!dump.enabled) return;
  std::filesystem::path p(dump.root);
  p /= std::string(name) + ".npy";
  if (!write_npy_f32(p.string(), data, count, shape)) {
    std::printf("[warn] Failed to write dump: %s\n", p.string().c_str());
  }
}

template <int D0, int D1, typename T>
static void dump_2d(const DumpContext& dump,
                    const char* name,
                    const T x[D0][D1]) {
  std::vector<float> buf;
  buf.resize(static_cast<std::size_t>(D0 * D1));
  std::size_t idx = 0U;
  for (int i = 0; i < D0; ++i) {
    for (int j = 0; j < D1; ++j) {
      buf[idx++] = static_cast<float>(x[i][j].to_float());
    }
  }
  std::vector<int> shape;
  shape.push_back(D0);
  shape.push_back(D1);
  dump_tensor(dump, name, buf.data(), buf.size(), shape);
}

template <int D0, int D1, int D2, typename T>
static void dump_3d(const DumpContext& dump,
                    const char* name,
                    const T x[D0][D1][D2]) {
  std::vector<float> buf;
  buf.resize(static_cast<std::size_t>(D0 * D1 * D2));
  std::size_t idx = 0U;
  for (int i = 0; i < D0; ++i) {
    for (int j = 0; j < D1; ++j) {
      for (int k = 0; k < D2; ++k) {
        buf[idx++] = static_cast<float>(x[i][j][k].to_float());
      }
    }
  }
  std::vector<int> shape;
  shape.push_back(D0);
  shape.push_back(D1);
  shape.push_back(D2);
  dump_tensor(dump, name, buf.data(), buf.size(), shape);
}

// LN_APPROX_BEGIN
static inline void layernorm_32(const fp32_ref_t x[D_MODEL],
                                const double w[D_MODEL],
                                const double b[D_MODEL],
                                fp32_ref_t y[D_MODEL]) {
  float sum = 0.0f;
  for (int i = 0; i < D_MODEL; ++i) {
    sum += x[i].to_float();
  }
  const float mean = sum * 0.03125f; // 1/32

  float var_acc = 0.0f;
  for (int i = 0; i < D_MODEL; ++i) {
    const float d = x[i].to_float() - mean;
    var_acc += d * d;
  }
  const float var = var_acc * 0.03125f; // 1/32

  const float inv_std =
    ref_inv_sqrt_approx(fp32_ref_t(var + LN_EPS_F32)).to_float();
  for (int i = 0; i < D_MODEL; ++i) {
    const float xn = (x[i].to_float() - mean) * inv_std;
    const float yi = xn * static_cast<float>(w[i]) + static_cast<float>(b[i]);
    y[i] = fp32_ref_t(yi);
  }
}
// LN_APPROX_END

static void apply_layernorm_tokens(const fp32_ref_t x_in[TOKENS_T][D_MODEL],
                                   const double w[D_MODEL],
                                   const double b[D_MODEL],
                                   fp32_ref_t x_out[TOKENS_T][D_MODEL]) {
  for (int t = 0; t < TOKENS_T; ++t) {
    layernorm_32(x_in[t], w, b, x_out[t]);
  }
}

static void quant_linear_75x32_to32(const fp32_ref_t x[TOKENS_T][D_MODEL],
                                    const double w[D_MODEL * D_MODEL],
                                    const double b[D_MODEL],
                                    float s_x,
                                    float s_w,
                                    fp32_ref_t y[TOKENS_T][D_MODEL],
                                    bool strict_int16_acc,
                                    RefFullQuantStats* stats,
                                    const char* block_name) {
  fp32_ref_t inv = fp32_ref_t(1.0f) / (fp32_ref_t(s_x) * fp32_ref_t(s_w));
  for (int t = 0; t < TOKENS_T; ++t) {
    if (strict_int16_acc) {
      int16_t qx_i16[D_MODEL];
      for (int i = 0; i < D_MODEL; ++i) {
        qx_i16[i] = quantize_int8_to_i16(x[t][i], s_x, stats);
      }
      for (int o = 0; o < D_MODEL; ++o) {
        int32_t acc_i32 = 0;
        const int base = o * D_MODEL;
        for (int i = 0; i < D_MODEL; ++i) {
          const int16_t qw_i16 = quantize_weight_to_i16(w[base + i], s_w);
          const int32_t prod = static_cast<int32_t>(qx_i16[i]) * static_cast<int32_t>(qw_i16);
          int32_t sum = acc_i32 + prod;
          if (sum > 32767) {
            sum = 32767;
            if (stats != nullptr) {
              stats->int_linear.int16_overflow_count++;
              if (stats->int_linear.first_int16_overflow_block.empty()) {
                stats->int_linear.first_int16_overflow_block = block_name;
              }
            }
          } else if (sum < -32768) {
            sum = -32768;
            if (stats != nullptr) {
              stats->int_linear.int16_overflow_count++;
              if (stats->int_linear.first_int16_overflow_block.empty()) {
                stats->int_linear.first_int16_overflow_block = block_name;
              }
            }
          }
          acc_i32 = sum;
        }
        const int16_t acc_i16 = static_cast<int16_t>(acc_i32);
        const fp32_ref_t deq = fp32_ref_t(static_cast<float>(acc_i16)) * inv;
        y[t][o] = fp32_ref_t(static_cast<float>(b[o])) + deq;
        if (stats != nullptr) {
          stats->int_linear.dequant_restore_count++;
        }
      }
    } else {
      fp32_ref_t qx[D_MODEL];
      for (int i = 0; i < D_MODEL; ++i) {
        qx[i] = quantize_int8_symmetric(x[t][i], fp32_ref_t(s_x));
      }

      for (int o = 0; o < D_MODEL; ++o) {
        fp32_ref_t acc = fp32_ref_t(static_cast<float>(b[o]));
        const int base = o * D_MODEL;
        for (int i = 0; i < D_MODEL; ++i) {
          acc += qx[i] * (fp32_ref_t(static_cast<float>(w[base + i])) * inv);
        }
        y[t][o] = acc;
      }
    }
  }
}

static void quant_linear_75x32_to128(const fp32_ref_t x[TOKENS_T][D_MODEL],
                                     const double w[FF_DIM * D_MODEL],
                                     const double b[FF_DIM],
                                     float s_x,
                                     float s_w,
                                     fp32_ref_t y[TOKENS_T][FF_DIM],
                                     bool strict_int16_acc,
                                     RefFullQuantStats* stats,
                                     const char* block_name) {
  fp32_ref_t inv = fp32_ref_t(1.0f) / (fp32_ref_t(s_x) * fp32_ref_t(s_w));
  for (int t = 0; t < TOKENS_T; ++t) {
    if (strict_int16_acc) {
      int16_t qx_i16[D_MODEL];
      for (int i = 0; i < D_MODEL; ++i) {
        qx_i16[i] = quantize_int8_to_i16(x[t][i], s_x, stats);
      }
      for (int o = 0; o < FF_DIM; ++o) {
        int32_t acc_i32 = 0;
        const int base = o * D_MODEL;
        for (int i = 0; i < D_MODEL; ++i) {
          const int16_t qw_i16 = quantize_weight_to_i16(w[base + i], s_w);
          const int32_t prod = static_cast<int32_t>(qx_i16[i]) * static_cast<int32_t>(qw_i16);
          int32_t sum = acc_i32 + prod;
          if (sum > 32767) {
            sum = 32767;
            if (stats != nullptr) {
              stats->int_linear.int16_overflow_count++;
              if (stats->int_linear.first_int16_overflow_block.empty()) {
                stats->int_linear.first_int16_overflow_block = block_name;
              }
            }
          } else if (sum < -32768) {
            sum = -32768;
            if (stats != nullptr) {
              stats->int_linear.int16_overflow_count++;
              if (stats->int_linear.first_int16_overflow_block.empty()) {
                stats->int_linear.first_int16_overflow_block = block_name;
              }
            }
          }
          acc_i32 = sum;
        }
        const int16_t acc_i16 = static_cast<int16_t>(acc_i32);
        const fp32_ref_t deq = fp32_ref_t(static_cast<float>(acc_i16)) * inv;
        y[t][o] = fp32_ref_t(static_cast<float>(b[o])) + deq;
        if (stats != nullptr) {
          stats->int_linear.dequant_restore_count++;
        }
      }
    } else {
      fp32_ref_t qx[D_MODEL];
      for (int i = 0; i < D_MODEL; ++i) {
        qx[i] = quantize_int8_symmetric(x[t][i], fp32_ref_t(s_x));
      }

      for (int o = 0; o < FF_DIM; ++o) {
        fp32_ref_t acc = fp32_ref_t(static_cast<float>(b[o]));
        const int base = o * D_MODEL;
        for (int i = 0; i < D_MODEL; ++i) {
          acc += qx[i] * (fp32_ref_t(static_cast<float>(w[base + i])) * inv);
        }
        y[t][o] = acc;
      }
    }
  }
}

static void quant_linear_75x128_to32(const fp32_ref_t x[TOKENS_T][FF_DIM],
                                     const double w[D_MODEL * FF_DIM],
                                     const double b[D_MODEL],
                                     float s_x,
                                     float s_w,
                                     fp32_ref_t y[TOKENS_T][D_MODEL],
                                     bool strict_int16_acc,
                                     RefFullQuantStats* stats,
                                     const char* block_name) {
  fp32_ref_t inv = fp32_ref_t(1.0f) / (fp32_ref_t(s_x) * fp32_ref_t(s_w));
  for (int t = 0; t < TOKENS_T; ++t) {
    if (strict_int16_acc) {
      int16_t qx_i16[FF_DIM];
      for (int i = 0; i < FF_DIM; ++i) {
        qx_i16[i] = quantize_int8_to_i16(x[t][i], s_x, stats);
      }
      for (int o = 0; o < D_MODEL; ++o) {
        int32_t acc_i32 = 0;
        const int base = o * FF_DIM;
        for (int i = 0; i < FF_DIM; ++i) {
          const int16_t qw_i16 = quantize_weight_to_i16(w[base + i], s_w);
          const int32_t prod = static_cast<int32_t>(qx_i16[i]) * static_cast<int32_t>(qw_i16);
          int32_t sum = acc_i32 + prod;
          if (sum > 32767) {
            sum = 32767;
            if (stats != nullptr) {
              stats->int_linear.int16_overflow_count++;
              if (stats->int_linear.first_int16_overflow_block.empty()) {
                stats->int_linear.first_int16_overflow_block = block_name;
              }
            }
          } else if (sum < -32768) {
            sum = -32768;
            if (stats != nullptr) {
              stats->int_linear.int16_overflow_count++;
              if (stats->int_linear.first_int16_overflow_block.empty()) {
                stats->int_linear.first_int16_overflow_block = block_name;
              }
            }
          }
          acc_i32 = sum;
        }
        const int16_t acc_i16 = static_cast<int16_t>(acc_i32);
        const fp32_ref_t deq = fp32_ref_t(static_cast<float>(acc_i16)) * inv;
        y[t][o] = fp32_ref_t(static_cast<float>(b[o])) + deq;
        if (stats != nullptr) {
          stats->int_linear.dequant_restore_count++;
        }
      }
    } else {
      fp32_ref_t qx[FF_DIM];
      for (int i = 0; i < FF_DIM; ++i) {
        qx[i] = quantize_int8_symmetric(x[t][i], fp32_ref_t(s_x));
      }

      for (int o = 0; o < D_MODEL; ++o) {
        fp32_ref_t acc = fp32_ref_t(static_cast<float>(b[o]));
        const int base = o * FF_DIM;
        for (int i = 0; i < FF_DIM; ++i) {
          acc += qx[i] * (fp32_ref_t(static_cast<float>(w[base + i])) * inv);
        }
        y[t][o] = acc;
      }
    }
  }
}

static void build_masks(bool one_ring[TOKENS_T][TOKENS_T],
                        bool second_ring[TOKENS_T][TOKENS_T]) {
  bool src[TOKENS_T][TOKENS_T];
  for (int i = 0; i < TOKENS_T; ++i) {
    for (int j = 0; j < TOKENS_T; ++j) {
      src[i][j] = (w_src_mask[i * TOKENS_T + j].to_int() != 0);
    }
  }

  for (int i = 0; i < TOKENS_T; ++i) {
    for (int j = 0; j < TOKENS_T; ++j) {
      const bool is_v_i = (i < VAR_N);
      const bool is_v_j = (j < VAR_N);

      if (is_v_i && is_v_j) {
        one_ring[i][j] = true;
        second_ring[i][j] = src[i][j];
      } else if (is_v_i && !is_v_j) {
        one_ring[i][j] = src[i][j];
        second_ring[i][j] = true;
      } else if (!is_v_i && is_v_j) {
        one_ring[i][j] = src[i][j];
        second_ring[i][j] = true;
      } else {
        one_ring[i][j] = true;
        second_ring[i][j] = src[i][j];
      }
    }
  }
}

// SOFTMAX_APPROX_BEGIN
static inline void online_softmax_update(
  bool &is_init,
  fp32_ref_t score,
  const fp32_ref_t v_head[D_HEAD],
  fp32_ref_t &max_score,
  fp32_ref_t &sumexp,
  fp32_ref_t acc_vec[D_HEAD]
) {
  if (!is_init) {
    max_score = score;
    sumexp = fp32_ref_t(1.0f);
    for (int dh = 0; dh < D_HEAD; ++dh) {
      acc_vec[dh] = v_head[dh];
    }
    is_init = true;
    return;
  }

  if (score > max_score) {
    const fp32_ref_t rescale = ref_softmax_exp_lut(max_score - score);
    sumexp = (sumexp * rescale) + fp32_ref_t(1.0f);
    for (int dh = 0; dh < D_HEAD; ++dh) {
      acc_vec[dh] = (acc_vec[dh] * rescale) + v_head[dh];
    }
    max_score = score;
    return;
  }

  const fp32_ref_t w = ref_softmax_exp_lut(score - max_score);
  sumexp += w;
  for (int dh = 0; dh < D_HEAD; ++dh) {
    acc_vec[dh] += w * v_head[dh];
  }
}

static void attention_block(const fp32_ref_t q[TOKENS_T][D_MODEL],
                            const fp32_ref_t k[TOKENS_T][D_MODEL],
                            const fp32_ref_t v[TOKENS_T][D_MODEL],
                            const bool one_ring[TOKENS_T][TOKENS_T],
                            const bool second_ring[TOKENS_T][TOKENS_T],
                            const RefRunConfig& run_cfg,
                            RefFullQuantStats* stats,
                            fp32_ref_t scores[HEADS][TOKENS_T][TOKENS_T],
                            fp32_ref_t probs[HEADS][TOKENS_T][TOKENS_T],
                            fp32_ref_t ctx[HEADS][TOKENS_T][D_HEAD],
                            fp32_ref_t post_concat[TOKENS_T][D_MODEL]) {
  const fp32_ref_t inv_sqrt_dh = fp32_ref_t(0.5f); // 1/sqrt(4)
  const fp32_ref_t neg_inf = fp32_ref_t(-std::numeric_limits<float>::infinity());

  for (int h = 0; h < HEADS; ++h) {
    for (int i = 0; i < TOKENS_T; ++i) {
      const bool (*mask)[TOKENS_T] = (h < 4) ? one_ring : second_ring;
      const int base = h * D_HEAD;
      bool has_valid = false;
      bool online_init = false;
      fp32_ref_t online_max = neg_inf;
      fp32_ref_t online_sumexp = fp32_ref_t(0.0f);
      fp32_ref_t acc_vec[D_HEAD];
      for (int dh = 0; dh < D_HEAD; ++dh) {
        acc_vec[dh] = fp32_ref_t(0.0f);
      }

      for (int j = 0; j < TOKENS_T; ++j) {
        if (mask[i][j]) {
          scores[h][i][j] = neg_inf;
          probs[h][i][j] = fp32_ref_t(0.0f);
          continue;
        }

        has_valid = true;
        fp32_ref_t dot = fp32_ref_t(0.0f);
        for (int dh = 0; dh < D_HEAD; ++dh) {
          dot += q[i][base + dh] * k[j][base + dh];
        }
        const fp32_ref_t score = stress_roundtrip_e4m3(
          dot * inv_sqrt_dh,
          run_cfg,
          RefFragGroup::G4_SOFTMAX_NEIGHBORHOOD,
          stats,
          "attention_score");
        scores[h][i][j] = score;
        online_softmax_update(
          online_init,
          score,
          &v[j][base],
          online_max,
          online_sumexp,
          acc_vec
        );
      }

      if (!has_valid) {
        for (int dh = 0; dh < D_HEAD; ++dh) {
          ctx[h][i][dh] = fp32_ref_t(0.0f);
        }
        continue;
      }

      const fp32_ref_t inv_sumexp = ref_softmax_rcp_lut(online_sumexp);

      // Trace-only probability materialization from final online state.
      for (int j = 0; j < TOKENS_T; ++j) {
        if (mask[i][j]) {
          probs[h][i][j] = fp32_ref_t(0.0f);
          continue;
        }
        const fp32_ref_t w = stress_roundtrip_e4m3(
          ref_softmax_exp_lut(scores[h][i][j] - online_max),
          run_cfg,
          RefFragGroup::G4_SOFTMAX_NEIGHBORHOOD,
          stats,
          "softmax_weight"
        );
        probs[h][i][j] = stress_roundtrip_e4m3(
          w * inv_sumexp,
          run_cfg,
          RefFragGroup::G4_SOFTMAX_NEIGHBORHOOD,
          stats,
          "softmax_prob");
      }

      for (int dh = 0; dh < D_HEAD; ++dh) {
        ctx[h][i][dh] = stress_roundtrip_e4m3(
          acc_vec[dh] * inv_sumexp,
          run_cfg,
          RefFragGroup::G3_ATTN_CONTEXT,
          stats,
          "attention_ctx");
      }
    }
  }

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int h = 0; h < HEADS; ++h) {
      const int base = h * D_HEAD;
      for (int dh = 0; dh < D_HEAD; ++dh) {
        post_concat[t][base + dh] = stress_roundtrip_e4m3(
          ctx[h][t][dh],
          run_cfg,
          RefFragGroup::G3_ATTN_CONTEXT,
          stats,
          "attention_post_concat"
        );
      }
    }
  }
}
// SOFTMAX_APPROX_END

static void run_layer(const int layer_idx,
                      const RefRunConfig& run_cfg,
                      RefFullQuantStats* stats,
                      const fp32_ref_t x_in[TOKENS_T][D_MODEL],
                      const bool one_ring[TOKENS_T][TOKENS_T],
                      const bool second_ring[TOKENS_T][TOKENS_T],
                      fp32_ref_t q_out[TOKENS_T][D_MODEL],
                      fp32_ref_t k_out[TOKENS_T][D_MODEL],
                      fp32_ref_t v_out[TOKENS_T][D_MODEL],
                      fp32_ref_t attn_scores[HEADS][TOKENS_T][TOKENS_T],
                      fp32_ref_t attn_probs[HEADS][TOKENS_T][TOKENS_T],
                      fp32_ref_t ctx[HEADS][TOKENS_T][D_HEAD],
                      fp32_ref_t attn_out[TOKENS_T][D_MODEL],
                      fp32_ref_t ln_in[TOKENS_T][D_MODEL],
                      fp32_ref_t ln_out[TOKENS_T][D_MODEL],
                      fp32_ref_t ffn1_out[TOKENS_T][FF_DIM],
                      fp32_ref_t act_out[TOKENS_T][FF_DIM],
                      fp32_ref_t ffn2_out[TOKENS_T][D_MODEL],
                      fp32_ref_t ffn_ln_out[TOKENS_T][D_MODEL]) {
  const bool strict_int16 = use_full_e4m3_nonlinear_stress(run_cfg);
  const double* w_q = nullptr;
  const double* b_q = nullptr;
  const double* sw_q = nullptr;
  const double* w_k = nullptr;
  const double* b_k = nullptr;
  const double* sw_k = nullptr;
  const double* w_v = nullptr;
  const double* b_v = nullptr;
  const double* sw_v = nullptr;
  const double* w_o = nullptr;
  const double* b_o = nullptr;
  const double* sw_o = nullptr;
  const double* w_ff1 = nullptr;
  const double* b_ff1 = nullptr;
  const double* sw_ff1 = nullptr;
  const double* w_ff2 = nullptr;
  const double* b_ff2 = nullptr;
  const double* sw_ff2 = nullptr;
  const double* ln0_w = nullptr;
  const double* ln0_b = nullptr;
  const double* ln1_w = nullptr;
  const double* ln1_b = nullptr;

  float s_x_in = 0.0f;
  float s_x_o = 0.0f;
  float s_x_ff1 = 0.0f;
  float s_x_ff2 = 0.0f;

  if (layer_idx == 0) {
    w_q = w_decoder_layers_0_self_attn_linears_0_weight;
    b_q = w_decoder_layers_0_self_attn_linears_0_bias;
    sw_q = w_decoder_layers_0_self_attn_linears_0_s_w;
    w_k = w_decoder_layers_0_self_attn_linears_1_weight;
    b_k = w_decoder_layers_0_self_attn_linears_1_bias;
    sw_k = w_decoder_layers_0_self_attn_linears_1_s_w;
    w_v = w_decoder_layers_0_self_attn_linears_2_weight;
    b_v = w_decoder_layers_0_self_attn_linears_2_bias;
    sw_v = w_decoder_layers_0_self_attn_linears_2_s_w;
    w_o = w_decoder_layers_0_self_attn_linears_3_weight;
    b_o = w_decoder_layers_0_self_attn_linears_3_bias;
    sw_o = w_decoder_layers_0_self_attn_linears_3_s_w;
    w_ff1 = w_decoder_layers_0_feed_forward_w_1_weight;
    b_ff1 = w_decoder_layers_0_feed_forward_w_1_bias;
    sw_ff1 = w_decoder_layers_0_feed_forward_w_1_s_w;
    w_ff2 = w_decoder_layers_0_feed_forward_w_2_weight;
    b_ff2 = w_decoder_layers_0_feed_forward_w_2_bias;
    sw_ff2 = w_decoder_layers_0_feed_forward_w_2_s_w;
    ln0_w = w_decoder_layers_0_sublayer_0_norm_weight;
    ln0_b = w_decoder_layers_0_sublayer_0_norm_bias;
    ln1_w = w_decoder_layers_0_sublayer_1_norm_weight;
    ln1_b = w_decoder_layers_0_sublayer_1_norm_bias;

    s_x_in = static_cast<float>(l0_in_s_x);
    s_x_o = static_cast<float>(l0_o_s_x);
    s_x_ff1 = static_cast<float>(l0_ff1_s_x);
    s_x_ff2 = static_cast<float>(l0_ff2_s_x);
  } else {
    w_q = w_decoder_layers_1_self_attn_linears_0_weight;
    b_q = w_decoder_layers_1_self_attn_linears_0_bias;
    sw_q = w_decoder_layers_1_self_attn_linears_0_s_w;
    w_k = w_decoder_layers_1_self_attn_linears_1_weight;
    b_k = w_decoder_layers_1_self_attn_linears_1_bias;
    sw_k = w_decoder_layers_1_self_attn_linears_1_s_w;
    w_v = w_decoder_layers_1_self_attn_linears_2_weight;
    b_v = w_decoder_layers_1_self_attn_linears_2_bias;
    sw_v = w_decoder_layers_1_self_attn_linears_2_s_w;
    w_o = w_decoder_layers_1_self_attn_linears_3_weight;
    b_o = w_decoder_layers_1_self_attn_linears_3_bias;
    sw_o = w_decoder_layers_1_self_attn_linears_3_s_w;
    w_ff1 = w_decoder_layers_1_feed_forward_w_1_weight;
    b_ff1 = w_decoder_layers_1_feed_forward_w_1_bias;
    sw_ff1 = w_decoder_layers_1_feed_forward_w_1_s_w;
    w_ff2 = w_decoder_layers_1_feed_forward_w_2_weight;
    b_ff2 = w_decoder_layers_1_feed_forward_w_2_bias;
    sw_ff2 = w_decoder_layers_1_feed_forward_w_2_s_w;
    ln0_w = w_decoder_layers_1_sublayer_0_norm_weight;
    ln0_b = w_decoder_layers_1_sublayer_0_norm_bias;
    ln1_w = w_decoder_layers_1_sublayer_1_norm_weight;
    ln1_b = w_decoder_layers_1_sublayer_1_norm_bias;

    s_x_in = static_cast<float>(l1_in_s_x);
    s_x_o = static_cast<float>(l1_o_s_x);
    s_x_ff1 = static_cast<float>(l1_ff1_s_x);
    s_x_ff2 = static_cast<float>(l1_ff2_s_x);
  }

  quant_linear_75x32_to32(
    x_in, w_q, b_q, s_x_in, static_cast<float>(sw_q[0]), q_out, strict_int16, stats, "Wq");
  quant_linear_75x32_to32(
    x_in, w_k, b_k, s_x_in, static_cast<float>(sw_k[0]), k_out, strict_int16, stats, "Wk");
  quant_linear_75x32_to32(
    x_in, w_v, b_v, s_x_in, static_cast<float>(sw_v[0]), v_out, strict_int16, stats, "Wv");

  if (use_full_e4m3_nonlinear_stress(run_cfg)) {
    for (int t = 0; t < TOKENS_T; ++t) {
      for (int d = 0; d < D_MODEL; ++d) {
        q_out[t][d] = stress_roundtrip_e4m3(
          q_out[t][d], run_cfg, RefFragGroup::NONE, stats, "q_out");
        k_out[t][d] = stress_roundtrip_e4m3(
          k_out[t][d], run_cfg, RefFragGroup::NONE, stats, "k_out");
        v_out[t][d] = stress_roundtrip_e4m3(
          v_out[t][d], run_cfg, RefFragGroup::NONE, stats, "v_out");
      }
    }
  }

  fp32_ref_t post_concat[TOKENS_T][D_MODEL];
  attention_block(q_out,
                  k_out,
                  v_out,
                  one_ring,
                  second_ring,
                  run_cfg,
                  stats,
                  attn_scores,
                  attn_probs,
                  ctx,
                  post_concat);

  quant_linear_75x32_to32(post_concat,
                          w_o,
                          b_o,
                          s_x_o,
                          static_cast<float>(sw_o[0]),
                          attn_out,
                          strict_int16,
                          stats,
                          "Wo");

  if (use_full_e4m3_nonlinear_stress(run_cfg) ||
      should_apply_e4m3_group_roundtrip(run_cfg, RefFragGroup::G3_ATTN_CONTEXT)) {
    for (int t = 0; t < TOKENS_T; ++t) {
      for (int d = 0; d < D_MODEL; ++d) {
        attn_out[t][d] = stress_roundtrip_e4m3(
          attn_out[t][d], run_cfg, RefFragGroup::G3_ATTN_CONTEXT, stats, "attn_out");
      }
    }
  }

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      ln_in[t][d] = stress_roundtrip_e4m3(
        attn_out[t][d] + x_in[t][d],
        run_cfg,
        RefFragGroup::G2_RESIDUAL,
        stats,
        "residual_attn");
    }
  }
  apply_layernorm_tokens(ln_in, ln0_w, ln0_b, ln_out);
  if (use_full_e4m3_nonlinear_stress(run_cfg) ||
      should_apply_e4m3_group_roundtrip(run_cfg, RefFragGroup::G1_LAYERNORM)) {
    for (int t = 0; t < TOKENS_T; ++t) {
      for (int d = 0; d < D_MODEL; ++d) {
        ln_out[t][d] = stress_roundtrip_e4m3(
          ln_out[t][d], run_cfg, RefFragGroup::G1_LAYERNORM, stats, "ln_out");
      }
    }
  }

  quant_linear_75x32_to128(ln_out,
                           w_ff1,
                           b_ff1,
                           s_x_ff1,
                           static_cast<float>(sw_ff1[0]),
                           ffn1_out,
                           strict_int16,
                           stats,
                           "Wff1");

  if (use_full_e4m3_nonlinear_stress(run_cfg)) {
    for (int t = 0; t < TOKENS_T; ++t) {
      for (int i = 0; i < FF_DIM; ++i) {
        ffn1_out[t][i] = stress_roundtrip_e4m3(
          ffn1_out[t][i], run_cfg, RefFragGroup::NONE, stats, "ffn1_out");
      }
    }
  }

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int i = 0; i < FF_DIM; ++i) {
      fp32_ref_t vff = ffn1_out[t][i];
      act_out[t][i] = stress_roundtrip_e4m3(
        (vff > fp32_ref_t(0.0f)) ? vff : fp32_ref_t(0.0f),
        run_cfg,
        RefFragGroup::NONE,
        stats,
        "ffn_relu_out"
      );
    }
  }

  quant_linear_75x128_to32(act_out,
                           w_ff2,
                           b_ff2,
                           s_x_ff2,
                           static_cast<float>(sw_ff2[0]),
                           ffn2_out,
                           strict_int16,
                           stats,
                           "Wff2");

  if (use_full_e4m3_nonlinear_stress(run_cfg)) {
    for (int t = 0; t < TOKENS_T; ++t) {
      for (int d = 0; d < D_MODEL; ++d) {
        ffn2_out[t][d] = stress_roundtrip_e4m3(
          ffn2_out[t][d], run_cfg, RefFragGroup::NONE, stats, "ffn2_out");
      }
    }
  }

  fp32_ref_t ffn_ln_in[TOKENS_T][D_MODEL];
  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      ffn_ln_in[t][d] = stress_roundtrip_e4m3(
        ffn2_out[t][d] + ln_out[t][d],
        run_cfg,
        RefFragGroup::G2_RESIDUAL,
        stats,
        "residual_ffn");
    }
  }
  apply_layernorm_tokens(ffn_ln_in, ln1_w, ln1_b, ffn_ln_out);
  if (use_full_e4m3_nonlinear_stress(run_cfg) ||
      should_apply_e4m3_group_roundtrip(run_cfg, RefFragGroup::G1_LAYERNORM)) {
    for (int t = 0; t < TOKENS_T; ++t) {
      for (int d = 0; d < D_MODEL; ++d) {
        ffn_ln_out[t][d] = stress_roundtrip_e4m3(
          ffn_ln_out[t][d], run_cfg, RefFragGroup::G1_LAYERNORM, stats, "ffn_ln_out");
      }
    }
  }
}

} // namespace

RefModel::RefModel() {
  run_cfg_.precision_mode = RefPrecisionMode::BASELINE_FP32;
  run_cfg_.algo_variant = RefAlgoVariant::BASELINE_SPEC_FLOW;
  run_cfg_.finalhead_stage = RefFinalHeadExploreStage::S0;
  dump_cfg_.enabled = false;
  dump_cfg_.dump_dir = nullptr;
  dump_cfg_.pattern_index = -1;
}

void RefModel::set_run_config(const RefRunConfig& cfg) {
  run_cfg_ = cfg;
}

RefRunConfig RefModel::get_run_config() const {
  return run_cfg_;
}

void RefModel::set_dump_config(const RefDumpConfig& cfg) {
  dump_cfg_ = cfg;
}

void RefModel::clear_dump_config() {
  dump_cfg_.enabled = false;
  dump_cfg_.dump_dir = nullptr;
  dump_cfg_.pattern_index = -1;
}

void RefModel::infer_step0(const RefModelIO& io) const {
  const int B = io.B;
  const int N = io.N;
  const int N_out = (N < VAR_N) ? N : VAR_N;

  assert(io.input_y_fp32 != nullptr && "input_y_fp32 must be provided for alignment mode");
  if (run_cfg_.algo_variant != RefAlgoVariant::BASELINE_SPEC_FLOW) {
    std::printf("[warn] Unsupported algo variant %s. Fallback to BASELINE_SPEC_FLOW.\n",
      to_string(run_cfg_.algo_variant));
  }

  bool one_ring[TOKENS_T][TOKENS_T];
  bool second_ring[TOKENS_T][TOKENS_T];
  build_masks(one_ring, second_ring);
  RefFullQuantStats local_stats{};

  for (int b = 0; b < B; ++b) {
    DumpContext dump;
    dump.enabled = false;
    if (dump_cfg_.enabled && io.B == 1 && b == 0 && dump_cfg_.dump_dir != nullptr) {
      dump.enabled = true;
      dump.root = dump_cfg_.dump_dir;
    }

    fp32_ref_t y_var[VAR_N];
    int y_hard[VAR_N];
    for (int i = 0; i < VAR_N; ++i) {
      fp32_ref_t y = fp32_ref_t(static_cast<float>(io.input_y_fp32[b * N + i]));
      y_var[i] = y;
      y_hard[i] = (y < fp32_ref_t(0.0f)) ? 1 : 0;
    }

    fp32_ref_t node_feature[TOKENS_T];
    for (int i = 0; i < VAR_N; ++i) {
      node_feature[i] = fp32_abs(y_var[i]);
    }
    for (int c = 0; c < CHECK_N; ++c) {
      int parity = 0;
      for (int v = 0; v < VAR_N; ++v) {
        const int h = h_H[c * VAR_N + v].to_int();
        if (h != 0) {
          parity ^= y_hard[v];
        }
      }
      node_feature[VAR_N + c] = (parity == 0) ? fp32_ref_t(1.0f) : fp32_ref_t(-1.0f);
    }

    static fp32_ref_t preproc_x[TOKENS_T][D_MODEL];
    for (int t = 0; t < TOKENS_T; ++t) {
      for (int k = 0; k < 24; ++k) {
        preproc_x[t][k] = stress_roundtrip_e4m3(
          node_feature[t] * fp32_ref_t(static_cast<float>(w_src_embed[t * 24 + k])),
          run_cfg_,
          RefFragGroup::G5_PREPROC_EMBED,
          &local_stats,
          "preproc_src_embed"
        );
      }
      for (int k = 0; k < 8; ++k) {
        preproc_x[t][24 + k] = stress_roundtrip_e4m3(
          fp32_ref_t(static_cast<float>(w_lpe_token[t * 8 + k])),
          run_cfg_,
          RefFragGroup::G5_PREPROC_EMBED,
          &local_stats,
          "preproc_lpe"
        );
      }
    }

    dump_2d<TOKENS_T, D_MODEL>(dump, "preproc_x", preproc_x);

    static fp32_ref_t layer0_q[TOKENS_T][D_MODEL];
    static fp32_ref_t layer0_k[TOKENS_T][D_MODEL];
    static fp32_ref_t layer0_v[TOKENS_T][D_MODEL];
    static fp32_ref_t layer0_scores[HEADS][TOKENS_T][TOKENS_T];
    static fp32_ref_t layer0_probs[HEADS][TOKENS_T][TOKENS_T];
    static fp32_ref_t layer0_ctx[HEADS][TOKENS_T][D_HEAD];
    static fp32_ref_t layer0_attn_out[TOKENS_T][D_MODEL];
    static fp32_ref_t layer0_ln_in[TOKENS_T][D_MODEL];
    static fp32_ref_t layer0_ln_out[TOKENS_T][D_MODEL];
    static fp32_ref_t layer0_ffn1[TOKENS_T][FF_DIM];
    static fp32_ref_t layer0_act[TOKENS_T][FF_DIM];
    static fp32_ref_t layer0_ffn2[TOKENS_T][D_MODEL];
    static fp32_ref_t layer0_ffn_ln_out[TOKENS_T][D_MODEL];

    run_layer(0,
              run_cfg_,
              &local_stats,
              preproc_x,
              one_ring,
              second_ring,
              layer0_q,
              layer0_k,
              layer0_v,
              layer0_scores,
              layer0_probs,
              layer0_ctx,
              layer0_attn_out,
              layer0_ln_in,
              layer0_ln_out,
              layer0_ffn1,
              layer0_act,
              layer0_ffn2,
              layer0_ffn_ln_out);

    dump_2d<TOKENS_T, D_MODEL>(dump, "layer0_q", layer0_q);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer0_k", layer0_k);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer0_v", layer0_v);
    dump_3d<HEADS, TOKENS_T, TOKENS_T>(dump, "layer0_attn_scores", layer0_scores);
    dump_3d<HEADS, TOKENS_T, TOKENS_T>(dump, "layer0_attn_probs", layer0_probs);
    dump_3d<HEADS, TOKENS_T, D_HEAD>(dump, "layer0_ctx", layer0_ctx);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer0_attn_out", layer0_attn_out);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer0_ln_in", layer0_ln_in);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer0_ln_out", layer0_ln_out);
    dump_2d<TOKENS_T, FF_DIM>(dump, "layer0_ffn1_out", layer0_ffn1);
    dump_2d<TOKENS_T, FF_DIM>(dump, "layer0_act_out", layer0_act);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer0_ffn2_out", layer0_ffn2);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer0_ffn_ln_out", layer0_ffn_ln_out);

    static fp32_ref_t mid_norm[TOKENS_T][D_MODEL];
    apply_layernorm_tokens(layer0_ffn_ln_out,
                           w_decoder_norm2_weight,
                           w_decoder_norm2_bias,
                           mid_norm);
    if (use_full_e4m3_nonlinear_stress(run_cfg_) ||
        should_apply_e4m3_group_roundtrip(run_cfg_, RefFragGroup::G1_LAYERNORM)) {
      for (int t = 0; t < TOKENS_T; ++t) {
        for (int d = 0; d < D_MODEL; ++d) {
          mid_norm[t][d] = stress_roundtrip_e4m3(
            mid_norm[t][d],
            run_cfg_,
            RefFragGroup::G1_LAYERNORM,
            &local_stats,
            "mid_norm");
        }
      }
    }

    static fp32_ref_t layer1_q[TOKENS_T][D_MODEL];
    static fp32_ref_t layer1_k[TOKENS_T][D_MODEL];
    static fp32_ref_t layer1_v[TOKENS_T][D_MODEL];
    static fp32_ref_t layer1_scores[HEADS][TOKENS_T][TOKENS_T];
    static fp32_ref_t layer1_probs[HEADS][TOKENS_T][TOKENS_T];
    static fp32_ref_t layer1_ctx[HEADS][TOKENS_T][D_HEAD];
    static fp32_ref_t layer1_attn_out[TOKENS_T][D_MODEL];
    static fp32_ref_t layer1_ln_in[TOKENS_T][D_MODEL];
    static fp32_ref_t layer1_ln_out[TOKENS_T][D_MODEL];
    static fp32_ref_t layer1_ffn1[TOKENS_T][FF_DIM];
    static fp32_ref_t layer1_act[TOKENS_T][FF_DIM];
    static fp32_ref_t layer1_ffn2[TOKENS_T][D_MODEL];
    static fp32_ref_t layer1_ffn_ln_out[TOKENS_T][D_MODEL];

    run_layer(1,
              run_cfg_,
              &local_stats,
              mid_norm,
              one_ring,
              second_ring,
              layer1_q,
              layer1_k,
              layer1_v,
              layer1_scores,
              layer1_probs,
              layer1_ctx,
              layer1_attn_out,
              layer1_ln_in,
              layer1_ln_out,
              layer1_ffn1,
              layer1_act,
              layer1_ffn2,
              layer1_ffn_ln_out);

    dump_2d<TOKENS_T, D_MODEL>(dump, "layer1_q", layer1_q);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer1_k", layer1_k);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer1_v", layer1_v);
    dump_3d<HEADS, TOKENS_T, TOKENS_T>(dump, "layer1_attn_scores", layer1_scores);
    dump_3d<HEADS, TOKENS_T, TOKENS_T>(dump, "layer1_attn_probs", layer1_probs);
    dump_3d<HEADS, TOKENS_T, D_HEAD>(dump, "layer1_ctx", layer1_ctx);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer1_attn_out", layer1_attn_out);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer1_ln_in", layer1_ln_in);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer1_ln_out", layer1_ln_out);
    dump_2d<TOKENS_T, FF_DIM>(dump, "layer1_ffn1_out", layer1_ffn1);
    dump_2d<TOKENS_T, FF_DIM>(dump, "layer1_act_out", layer1_act);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer1_ffn2_out", layer1_ffn2);
    dump_2d<TOKENS_T, D_MODEL>(dump, "layer1_ffn_ln_out", layer1_ffn_ln_out);

    // Logical name: endLN_out, kept in end_norm for trace compatibility.
    static fp32_ref_t end_norm[TOKENS_T][D_MODEL];
    apply_layernorm_tokens(layer1_ffn_ln_out,
                           w_decoder_norm_weight,
                           w_decoder_norm_bias,
                           end_norm);
    if (use_full_e4m3_nonlinear_stress(run_cfg_) ||
        should_apply_e4m3_group_roundtrip(run_cfg_, RefFragGroup::G1_LAYERNORM)) {
      for (int t = 0; t < TOKENS_T; ++t) {
        for (int d = 0; d < D_MODEL; ++d) {
          end_norm[t][d] = stress_roundtrip_e4m3(
            end_norm[t][d],
            run_cfg_,
            RefFragGroup::G1_LAYERNORM,
            &local_stats,
            "end_norm");
        }
      }
    }

    // Logical name: s_t (token-wise FinalEmbedding scalar), trace tensor name kept stable.
    static fp32_ref_t final_node_logits[TOKENS_T][1];
    static fp32_ref_t out_fc_in[1][TOKENS_T];
    for (int t = 0; t < TOKENS_T; ++t) {
      fp32_ref_t acc = fp32_ref_t(static_cast<float>(w_oned_final_embed_0_bias[0]));
      for (int i = 0; i < D_MODEL; ++i) {
        acc += end_norm[t][i] * fp32_ref_t(static_cast<float>(w_oned_final_embed_0_weight[i]));
      }
      fp32_ref_t s_t_embed_out = acc;
      if (use_island_s3(run_cfg_)) {
        if (use_full_e4m3_nonlinear_stress(run_cfg_)) {
          s_t_embed_out = stress_roundtrip_e4m3(
            s_t_embed_out,
            run_cfg_,
            RefFragGroup::NONE,
            &local_stats,
            "final_embedding_s3");
        } else {
          s_t_embed_out = roundtrip_through_generic_e4m3(s_t_embed_out);
        }
      }
      final_node_logits[t][0] = s_t_embed_out;

      fp32_ref_t s_t_out_fc = s_t_embed_out;
      if (use_island_s0(run_cfg_)) {
        if (use_full_e4m3_nonlinear_stress(run_cfg_)) {
          s_t_out_fc = stress_roundtrip_e4m3(
            s_t_out_fc,
            run_cfg_,
            RefFragGroup::NONE,
            &local_stats,
            "final_readout_s0");
        } else {
          s_t_out_fc = roundtrip_through_generic_e4m3(s_t_out_fc);
        }
      }
      out_fc_in[0][t] = s_t_out_fc;
      if (io.out_finalhead_s_t != nullptr) {
        io.out_finalhead_s_t[b * TOKENS_T + t] = static_cast<double>(acc.to_float());
      }
    }

    static fp32_ref_t final_logits[1][VAR_N];
    static fp32_ref_t final_x_pred[VAR_N];
    for (int n = 0; n < VAR_N; ++n) {
      fp32_ref_t acc = fp32_ref_t(static_cast<float>(w_out_fc_bias[n]));
      for (int t = 0; t < TOKENS_T; ++t) {
        fp32_ref_t mul_in = out_fc_in[0][t];
        if (use_island_s1(run_cfg_)) {
          if (use_full_e4m3_nonlinear_stress(run_cfg_)) {
            mul_in = stress_roundtrip_e4m3(
              mul_in,
              run_cfg_,
              RefFragGroup::NONE,
              &local_stats,
              "out_fc_pre_mac_s1");
          } else {
            mul_in = roundtrip_through_generic_e4m3(mul_in);
          }
        }
        acc += fp32_ref_t(static_cast<float>(w_out_fc_weight[n * TOKENS_T + t])) * mul_in;
      }
      if (use_full_e4m3_nonlinear_stress(run_cfg_)) {
        acc = stress_roundtrip_e4m3(
          acc,
          run_cfg_,
          RefFragGroup::NONE,
          &local_stats,
          "final_logits");
      }
      final_logits[0][n] = acc;

      fp32_ref_t decision = acc * sign_fp32(y_var[n]);
      final_x_pred[n] = (decision < fp32_ref_t(0.0f)) ? fp32_ref_t(1.0f) : fp32_ref_t(0.0f);

      if (n < N_out) {
        io.out_logits[b * N + n] = static_cast<double>(acc.to_float());
        io.out_x_pred[b * N + n] = bit1_t((decision < fp32_ref_t(0.0f)) ? 1 : 0);
      }
    }

    for (int n = N_out; n < N; ++n) {
      io.out_logits[b * N + n] = 0.0;
      io.out_x_pred[b * N + n] = bit1_t(0);
    }

    dump_2d<TOKENS_T, 1>(dump, "final_node_logits", final_node_logits);
    dump_2d<1, TOKENS_T>(dump, "final_out_fc_in", out_fc_in);
    dump_2d<1, VAR_N>(dump, "final_logits", final_logits);

    std::vector<float> xp_buf;
    xp_buf.resize(VAR_N);
    for (int i = 0; i < VAR_N; ++i) {
      xp_buf[i] = final_x_pred[i].to_float();
    }
    std::vector<int> xp_shape;
    xp_shape.push_back(VAR_N);
    dump_tensor(dump, "final_x_pred", xp_buf.data(), xp_buf.size(), xp_shape);
  }

  add_ref_full_quant_stats(local_stats);
}

} // namespace aecct_ref
