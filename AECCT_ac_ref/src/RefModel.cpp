#include "../include/RefModel.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ac_std_float.h"

#include "../include/InvSqrtApprox.h"
#include "../include/SoftmaxApprox.h"
#include "weights.h"

namespace aecct_ref {
namespace {

typedef ac_ieee_float<binary32> fp32_ref_t;

static constexpr int TOKENS_T = 75;
static constexpr int VAR_N = 63;
static constexpr int CHECK_N = 12;
static constexpr int D_MODEL = 32;
static constexpr int HEADS = 8;
static constexpr int D_HEAD = 4;
static constexpr int FF_DIM = 128;
static constexpr float LN_EPS_F32 = 1.0e-5f;

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
                                    fp32_ref_t y[TOKENS_T][D_MODEL]) {
  fp32_ref_t inv = fp32_ref_t(1.0f) / (fp32_ref_t(s_x) * fp32_ref_t(s_w));
  for (int t = 0; t < TOKENS_T; ++t) {
    fp32_ref_t qx[D_MODEL];
    for (int i = 0; i < D_MODEL; ++i) {
      qx[i] = fp32_round(x[t][i] * fp32_ref_t(s_x));
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

static void quant_linear_75x32_to128(const fp32_ref_t x[TOKENS_T][D_MODEL],
                                     const double w[FF_DIM * D_MODEL],
                                     const double b[FF_DIM],
                                     float s_x,
                                     float s_w,
                                     fp32_ref_t y[TOKENS_T][FF_DIM]) {
  fp32_ref_t inv = fp32_ref_t(1.0f) / (fp32_ref_t(s_x) * fp32_ref_t(s_w));
  for (int t = 0; t < TOKENS_T; ++t) {
    fp32_ref_t qx[D_MODEL];
    for (int i = 0; i < D_MODEL; ++i) {
      qx[i] = fp32_round(x[t][i] * fp32_ref_t(s_x));
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

static void quant_linear_75x128_to32(const fp32_ref_t x[TOKENS_T][FF_DIM],
                                     const double w[D_MODEL * FF_DIM],
                                     const double b[D_MODEL],
                                     float s_x,
                                     float s_w,
                                     fp32_ref_t y[TOKENS_T][D_MODEL]) {
  fp32_ref_t inv = fp32_ref_t(1.0f) / (fp32_ref_t(s_x) * fp32_ref_t(s_w));
  for (int t = 0; t < TOKENS_T; ++t) {
    fp32_ref_t qx[FF_DIM];
    for (int i = 0; i < FF_DIM; ++i) {
      qx[i] = fp32_round(x[t][i] * fp32_ref_t(s_x));
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
static void attention_block(const fp32_ref_t q[TOKENS_T][D_MODEL],
                            const fp32_ref_t k[TOKENS_T][D_MODEL],
                            const fp32_ref_t v[TOKENS_T][D_MODEL],
                            const bool one_ring[TOKENS_T][TOKENS_T],
                            const bool second_ring[TOKENS_T][TOKENS_T],
                            fp32_ref_t scores[HEADS][TOKENS_T][TOKENS_T],
                            fp32_ref_t probs[HEADS][TOKENS_T][TOKENS_T],
                            fp32_ref_t ctx[HEADS][TOKENS_T][D_HEAD],
                            fp32_ref_t post_concat[TOKENS_T][D_MODEL]) {
  const fp32_ref_t inv_sqrt_dh = fp32_ref_t(0.5f); // 1/sqrt(4)
  const fp32_ref_t neg_inf = fp32_ref_t(-std::numeric_limits<float>::infinity());

  for (int h = 0; h < HEADS; ++h) {
    for (int i = 0; i < TOKENS_T; ++i) {
      const bool (*mask)[TOKENS_T] = (h < 4) ? one_ring : second_ring;

      bool has_valid = false;
      fp32_ref_t max_s = neg_inf;
      for (int j = 0; j < TOKENS_T; ++j) {
        if (mask[i][j]) {
          scores[h][i][j] = neg_inf;
          probs[h][i][j] = fp32_ref_t(0.0f);
          continue;
        }

        has_valid = true;
        fp32_ref_t dot = fp32_ref_t(0.0f);
        const int base = h * D_HEAD;
        for (int dh = 0; dh < D_HEAD; ++dh) {
          dot += q[i][base + dh] * k[j][base + dh];
        }
        fp32_ref_t s = dot * inv_sqrt_dh;
        scores[h][i][j] = s;
        if (s > max_s) {
          max_s = s;
        }
      }

      if (!has_valid) {
        for (int dh = 0; dh < D_HEAD; ++dh) {
          ctx[h][i][dh] = fp32_ref_t(0.0f);
        }
        continue;
      }

      fp32_ref_t sumexp = fp32_ref_t(0.0f);
      fp32_ref_t acc_vec[D_HEAD];
      for (int dh = 0; dh < D_HEAD; ++dh) {
        acc_vec[dh] = fp32_ref_t(0.0f);
      }

      for (int j = 0; j < TOKENS_T; ++j) {
        if (mask[i][j]) {
          continue;
        }
        fp32_ref_t x = scores[h][i][j] - max_s;
        fp32_ref_t w = ref_softmax_exp_lut(x);
        probs[h][i][j] = w;
        sumexp += w;

        const int base = h * D_HEAD;
        for (int dh = 0; dh < D_HEAD; ++dh) {
          acc_vec[dh] += w * v[j][base + dh];
        }
      }

      fp32_ref_t inv_sumexp = ref_softmax_rcp_lut(sumexp);
      for (int j = 0; j < TOKENS_T; ++j) {
        if (mask[i][j]) {
          probs[h][i][j] = fp32_ref_t(0.0f);
          continue;
        }
        probs[h][i][j] = probs[h][i][j] * inv_sumexp;
      }

      for (int dh = 0; dh < D_HEAD; ++dh) {
        ctx[h][i][dh] = acc_vec[dh] * inv_sumexp;
      }
    }
  }

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int h = 0; h < HEADS; ++h) {
      const int base = h * D_HEAD;
      for (int dh = 0; dh < D_HEAD; ++dh) {
        post_concat[t][base + dh] = ctx[h][t][dh];
      }
    }
  }
}
// SOFTMAX_APPROX_END

static void run_layer(const int layer_idx,
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

  quant_linear_75x32_to32(x_in, w_q, b_q, s_x_in, static_cast<float>(sw_q[0]), q_out);
  quant_linear_75x32_to32(x_in, w_k, b_k, s_x_in, static_cast<float>(sw_k[0]), k_out);
  quant_linear_75x32_to32(x_in, w_v, b_v, s_x_in, static_cast<float>(sw_v[0]), v_out);

  fp32_ref_t post_concat[TOKENS_T][D_MODEL];
  attention_block(q_out,
                  k_out,
                  v_out,
                  one_ring,
                  second_ring,
                  attn_scores,
                  attn_probs,
                  ctx,
                  post_concat);

  quant_linear_75x32_to32(post_concat,
                          w_o,
                          b_o,
                          s_x_o,
                          static_cast<float>(sw_o[0]),
                          attn_out);

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      ln_in[t][d] = attn_out[t][d] + x_in[t][d];
    }
  }
  apply_layernorm_tokens(ln_in, ln0_w, ln0_b, ln_out);

  quant_linear_75x32_to128(ln_out,
                           w_ff1,
                           b_ff1,
                           s_x_ff1,
                           static_cast<float>(sw_ff1[0]),
                           ffn1_out);

  for (int t = 0; t < TOKENS_T; ++t) {
    for (int i = 0; i < FF_DIM; ++i) {
      fp32_ref_t vff = ffn1_out[t][i];
      act_out[t][i] = (vff > fp32_ref_t(0.0f)) ? vff : fp32_ref_t(0.0f);
    }
  }

  quant_linear_75x128_to32(act_out,
                           w_ff2,
                           b_ff2,
                           s_x_ff2,
                           static_cast<float>(sw_ff2[0]),
                           ffn2_out);

  fp32_ref_t ffn_ln_in[TOKENS_T][D_MODEL];
  for (int t = 0; t < TOKENS_T; ++t) {
    for (int d = 0; d < D_MODEL; ++d) {
      ffn_ln_in[t][d] = ffn2_out[t][d] + ln_out[t][d];
    }
  }
  apply_layernorm_tokens(ffn_ln_in, ln1_w, ln1_b, ffn_ln_out);
}

} // namespace

RefModel::RefModel() {
  dump_cfg_.enabled = false;
  dump_cfg_.dump_dir = nullptr;
  dump_cfg_.pattern_index = -1;
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

  bool one_ring[TOKENS_T][TOKENS_T];
  bool second_ring[TOKENS_T][TOKENS_T];
  build_masks(one_ring, second_ring);

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
        preproc_x[t][k] = node_feature[t] * fp32_ref_t(static_cast<float>(w_src_embed[t * 24 + k]));
      }
      for (int k = 0; k < 8; ++k) {
        preproc_x[t][24 + k] = fp32_ref_t(static_cast<float>(w_lpe_token[t * 8 + k]));
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

    static fp32_ref_t end_norm[TOKENS_T][D_MODEL];
    apply_layernorm_tokens(layer1_ffn_ln_out,
                           w_decoder_norm_weight,
                           w_decoder_norm_bias,
                           end_norm);

    static fp32_ref_t final_node_logits[TOKENS_T][1];
    static fp32_ref_t out_fc_in[1][TOKENS_T];
    for (int t = 0; t < TOKENS_T; ++t) {
      fp32_ref_t acc = fp32_ref_t(static_cast<float>(w_oned_final_embed_0_bias[0]));
      for (int i = 0; i < D_MODEL; ++i) {
        acc += end_norm[t][i] * fp32_ref_t(static_cast<float>(w_oned_final_embed_0_weight[i]));
      }
      final_node_logits[t][0] = acc;
      out_fc_in[0][t] = acc;
    }

    static fp32_ref_t final_logits[1][VAR_N];
    static fp32_ref_t final_x_pred[VAR_N];
    for (int n = 0; n < VAR_N; ++n) {
      fp32_ref_t acc = fp32_ref_t(static_cast<float>(w_out_fc_bias[n]));
      for (int t = 0; t < TOKENS_T; ++t) {
        acc += fp32_ref_t(static_cast<float>(w_out_fc_weight[n * TOKENS_T + t])) * out_fc_in[0][t];
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
}

} // namespace aecct_ref





