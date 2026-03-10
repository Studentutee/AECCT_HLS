#include "RefStep0Synth.h"

#include <cstdint>

#include "../include/InvSqrtApprox.h"
#include "../include/SoftmaxApprox.h"
#include "../include/RefStep0ShapeBridge.h"
#include "weights.h"

namespace aecct_ref {
namespace {

static const int kTokens = ModelShapes::T_TOKENS;
static const int kVars = ModelShapes::N_VARS;
static const int kChecks = ModelShapes::T_TOKENS - ModelShapes::N_VARS;
static const int kDModel = ModelShapes::D_MODEL;
static const int kHeads = ModelShapes::N_HEADS;
static const int kDHead = ModelShapes::D_HEAD;
static const int kFfnDim = ModelShapes::D_FFN;

static const int kOutTile = 8;
static const int kClassTile = 16;
static const int kFifoDepth = 2;
static const int kXRegionWords = ModelShapes::X_WORK_BASE + ModelShapes::X_WORK_WORDS;

static const fp32_ref_t kLnEps = fp32_ref_t(1.0e-5f);
static const fp32_ref_t kInvDModel = fp32_ref_t(1.0f / static_cast<float>(kDModel));
static const fp32_ref_t kInvSqrtDHead = (kDHead == 1) ? fp32_ref_t(1.0f) : (kDHead == 2) ? fp32_ref_t(0.70710678f) : (kDHead == 4) ? fp32_ref_t(0.5f) : (kDHead == 8) ? fp32_ref_t(0.35355339f) : (kDHead == 16) ? fp32_ref_t(0.25f) : fp32_ref_t(1.0f);
static const fp32_ref_t kNegLarge = fp32_ref_t(-1.0e30f);
static const fp32_ref_t kActQMin = fp32_ref_t(-127.0f);
static const fp32_ref_t kActQMax = fp32_ref_t(127.0f);

static const fp32_ref_t kInvScaleL0Q = fp32_ref_t(0.0071290909f);
static const fp32_ref_t kInvScaleL0K = fp32_ref_t(0.0077177854f);
static const fp32_ref_t kInvScaleL0V = fp32_ref_t(0.0030414062f);
static const fp32_ref_t kInvScaleL0O = fp32_ref_t(0.00089474069f);
static const fp32_ref_t kInvScaleL0Ff1 = fp32_ref_t(0.0026840530f);
static const fp32_ref_t kInvScaleL0Ff2 = fp32_ref_t(0.0036128608f);

static const fp32_ref_t kInvScaleL1Q = fp32_ref_t(0.0043993234f);
static const fp32_ref_t kInvScaleL1K = fp32_ref_t(0.0071688984f);
static const fp32_ref_t kInvScaleL1V = fp32_ref_t(0.0049903886f);
static const fp32_ref_t kInvScaleL1O = fp32_ref_t(0.0043067653f);
static const fp32_ref_t kInvScaleL1Ff1 = fp32_ref_t(0.0024541419f);
static const fp32_ref_t kInvScaleL1Ff2 = fp32_ref_t(0.0055658957f);

static ac_int<2, false> g_outmode_reg = ac_int<2, false>(0);
static RefStep0RunReport g_last_report;

struct LayerConfig {
  const double *w_q;
  const double *b_q;
  const double *w_k;
  const double *b_k;
  const double *w_v;
  const double *b_v;
  const double *w_o;
  const double *b_o;
  const double *w_ff1;
  const double *b_ff1;
  const double *w_ff2;
  const double *b_ff2;
  const double *ln0_w;
  const double *ln0_b;
  const double *ln1_w;
  const double *ln1_b;
  fp32_ref_t s_x_in;
  fp32_ref_t s_x_o;
  fp32_ref_t s_x_ff1;
  fp32_ref_t s_x_ff2;
  fp32_ref_t inv_q;
  fp32_ref_t inv_k;
  fp32_ref_t inv_v;
  fp32_ref_t inv_o;
  fp32_ref_t inv_ff1;
  fp32_ref_t inv_ff2;
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

static inline fp32_ref_t quantize_int8_symmetric(fp32_ref_t x, fp32_ref_t s_x) {
  fp32_ref_t q = fp32_round(x * s_x);
  if (q > kActQMax) q = kActQMax;
  if (q < kActQMin) q = kActQMin;
  return q;
}

static inline fp32_ref_t fp32_relu(fp32_ref_t x) {
  return (x > fp32_ref_t(0.0f)) ? x : fp32_ref_t(0.0f);
}

static inline u32_word_t bits_from_fp32(const fp32_ref_t &x) {
  ac_int<32, true> raw = x.data_ac_int();
  return static_cast<u32_word_t>(raw);
}

static inline uint32_t output_words_for_mode(uint32_t mode) {
  if (mode == 1u) {
    return static_cast<uint32_t>(ModelShapes::OUT_DIM);
  }
  if (mode == 0u) {
    return static_cast<uint32_t>(ModelShapes::XPRED_WORDS);
  }
  return 0u;
}

static inline void clear_report(RefStep0RunReport &r) {
  r.final_scalar_base_word = static_cast<uint32_t>(ModelShapes::SCR_FINAL_SCALAR_BASE);
  r.final_scalar_words = static_cast<uint32_t>(ModelShapes::SCR_FINAL_SCALAR_WORDS);
  r.scratch_base_word = static_cast<uint32_t>(ModelShapes::SCRATCH_BASE);
  r.scratch_words = static_cast<uint32_t>(ModelShapes::SCRATCH_WORDS);

  r.final_scalar_in_scratch = false;
  r.final_scalar_range_ok = false;
  r.final_scalar_capacity_ok = false;
  r.final_scalar_addr_overlap_scr_k = false;
  r.final_scalar_addr_overlap_scr_v = false;
  r.final_scalar_live_conflict_scr_k = false;
  r.final_scalar_live_conflict_scr_v = false;
  r.final_scalar_overlap_conflict = false;

  r.final_layer_no_writeback_enforced = true;
  r.final_layer_writeback_words = 0u;
  r.final_head_used_page_next = false;

  r.pass_b_executed = false;
  r.output_words = 0u;

  r.has_error = false;
  r.error_code = REF_STEP0_ERR_NONE;
  r.error_msg = REF_STEP0_MSG_NONE;
}

static inline void set_report_error(RefStep0RunReport &r, uint32_t err_bit, uint32_t msg_code) {
  r.has_error = true;
  r.error_code |= err_bit;
  if (r.error_msg == REF_STEP0_MSG_NONE) {
    r.error_msg = msg_code;
  }
}

static inline bool ranges_overlap(uint32_t a_base, uint32_t a_words, uint32_t b_base, uint32_t b_words) {
  const uint32_t a_end = a_base + a_words;
  const uint32_t b_end = b_base + b_words;
  return (a_base < b_end) && (b_base < a_end);
}

static inline void check_final_scalar_region(
  RefStep0RunReport &r,
  bool scr_k_live_at_final_head,
  bool scr_v_live_at_final_head
) {
  const uint32_t final_base = r.final_scalar_base_word;
  const uint32_t final_words = r.final_scalar_words;
  const uint32_t scratch_base = r.scratch_base_word;
  const uint32_t scratch_words = r.scratch_words;

  r.final_scalar_range_ok =
    (final_base >= scratch_base) &&
    ((final_base + final_words) <= (scratch_base + scratch_words));
  r.final_scalar_capacity_ok = (final_words >= static_cast<uint32_t>(ModelShapes::T_TOKENS));
  r.final_scalar_in_scratch = r.final_scalar_range_ok;

  const uint32_t scr_k_base = static_cast<uint32_t>(ModelShapes::SCR_K_BASE);
  const uint32_t scr_k_words = static_cast<uint32_t>(ModelShapes::SCR_K_WORDS);
  const uint32_t scr_v_base = static_cast<uint32_t>(ModelShapes::SCR_V_BASE);
  const uint32_t scr_v_words = static_cast<uint32_t>(ModelShapes::SCR_V_WORDS);

  r.final_scalar_addr_overlap_scr_k = ranges_overlap(final_base, final_words, scr_k_base, scr_k_words);
  r.final_scalar_addr_overlap_scr_v = ranges_overlap(final_base, final_words, scr_v_base, scr_v_words);

  r.final_scalar_live_conflict_scr_k = r.final_scalar_addr_overlap_scr_k && scr_k_live_at_final_head;
  r.final_scalar_live_conflict_scr_v = r.final_scalar_addr_overlap_scr_v && scr_v_live_at_final_head;
  r.final_scalar_overlap_conflict = r.final_scalar_live_conflict_scr_k || r.final_scalar_live_conflict_scr_v;

  if (!r.final_scalar_range_ok) {
    set_report_error(r, REF_STEP0_ERR_FINAL_SCALAR_RANGE, REF_STEP0_MSG_FINAL_SCALAR_RANGE);
  }
  if (!r.final_scalar_capacity_ok) {
    set_report_error(r, REF_STEP0_ERR_FINAL_SCALAR_CAPACITY, REF_STEP0_MSG_FINAL_SCALAR_CAPACITY);
  }
  if (r.final_scalar_overlap_conflict) {
    set_report_error(r, REF_STEP0_ERR_FINAL_SCALAR_LIVE_CONFLICT, REF_STEP0_MSG_FINAL_SCALAR_LIVE_CONFLICT);
  }
}

static inline bool validate_var_to_class_map(RefStep0RunReport &r) {
  for (int var_idx = 0; var_idx < ModelShapes::N_VARS; ++var_idx) {
    const int class_idx = ModelShapes::map_var_to_class(var_idx);
    if (class_idx < 0 || class_idx >= ModelShapes::OUT_DIM) {
      set_report_error(r, REF_STEP0_ERR_MAP_OOB, REF_STEP0_MSG_MAP_OOB);
      return false;
    }
  }
  return true;
}

static inline void load_x_token(
  const fp32_ref_t x_region[kXRegionWords],
  int page_base,
  int token_idx,
  fp32_ref_t x_row[kDModel]
) {
  const int row_base = page_base + token_idx * kDModel;
  for (int d = 0; d < kDModel; ++d) {
    x_row[d] = x_region[row_base + d];
  }
}

static inline void store_x_token(
  fp32_ref_t x_region[kXRegionWords],
  int page_base,
  int token_idx,
  const fp32_ref_t x_row[kDModel]
) {
  const int row_base = page_base + token_idx * kDModel;
  for (int d = 0; d < kDModel; ++d) {
    x_region[row_base + d] = x_row[d];
  }
}

template <int OUT_DIM, int IN_DIM>
static inline void prefetch_quant_weight_tile(
  const double w[OUT_DIM * IN_DIM],
  fp32_ref_t inv_scale,
  int out_base,
  fp32_ref_t wbuf[kOutTile][IN_DIM]
) {
  for (int tile_o = 0; tile_o < kOutTile; ++tile_o) {
    const int out_idx = out_base + tile_o;
    for (int in_idx = 0; in_idx < IN_DIM; ++in_idx) {
      if (out_idx < OUT_DIM) {
        const float wf = static_cast<float>(w[out_idx * IN_DIM + in_idx]);
        wbuf[tile_o][in_idx] = fp32_ref_t(wf) * inv_scale;
      } else {
        wbuf[tile_o][in_idx] = fp32_ref_t(0.0f);
      }
    }
  }
}

template <int OUT_DIM, int IN_DIM>
static inline void prefetch_weight_tile(
  const double w[OUT_DIM * IN_DIM],
  int out_base,
  fp32_ref_t wbuf[kOutTile][IN_DIM]
) {
  for (int tile_o = 0; tile_o < kOutTile; ++tile_o) {
    const int out_idx = out_base + tile_o;
    for (int in_idx = 0; in_idx < IN_DIM; ++in_idx) {
      if (out_idx < OUT_DIM) {
        wbuf[tile_o][in_idx] = fp32_ref_t(static_cast<float>(w[out_idx * IN_DIM + in_idx]));
      } else {
        wbuf[tile_o][in_idx] = fp32_ref_t(0.0f);
      }
    }
  }
}

template <int OUT_DIM, int IN_DIM>
static inline void quant_linear_vec_tiled(
  const fp32_ref_t x[IN_DIM],
  const double w[OUT_DIM * IN_DIM],
  const double b[OUT_DIM],
  fp32_ref_t s_x,
  fp32_ref_t inv_scale,
  fp32_ref_t y[OUT_DIM]
) {
  fp32_ref_t qx[IN_DIM];
  for (int i = 0; i < IN_DIM; ++i) {
    qx[i] = quantize_int8_symmetric(x[i], s_x);
  }

  fp32_ref_t wbuf_ping[kOutTile][IN_DIM];
  fp32_ref_t wbuf_pong[kOutTile][IN_DIM];

  int out_base = 0;
  bool use_ping = true;
  prefetch_quant_weight_tile<OUT_DIM, IN_DIM>(w, inv_scale, out_base, wbuf_ping);

  while (out_base < OUT_DIM) {
    const int next_out_base = out_base + kOutTile;
    if (next_out_base < OUT_DIM) {
      if (use_ping) {
        prefetch_quant_weight_tile<OUT_DIM, IN_DIM>(w, inv_scale, next_out_base, wbuf_pong);
      } else {
        prefetch_quant_weight_tile<OUT_DIM, IN_DIM>(w, inv_scale, next_out_base, wbuf_ping);
      }
    }

    fp32_ref_t (*cur)[IN_DIM] = use_ping ? wbuf_ping : wbuf_pong;
    for (int tile_o = 0; tile_o < kOutTile; ++tile_o) {
      const int out_idx = out_base + tile_o;
      if (out_idx >= OUT_DIM) {
        continue;
      }
      fp32_ref_t acc = fp32_ref_t(static_cast<float>(b[out_idx]));
      for (int in_idx = 0; in_idx < IN_DIM; ++in_idx) {
        acc += qx[in_idx] * cur[tile_o][in_idx];
      }
      y[out_idx] = acc;
    }

    out_base = next_out_base;
    use_ping = !use_ping;
  }
}

template <int OUT_DIM, int IN_DIM>
static inline void dense_vec_tiled(
  const fp32_ref_t x[IN_DIM],
  const double w[OUT_DIM * IN_DIM],
  const double b[OUT_DIM],
  fp32_ref_t y[OUT_DIM]
) {
  fp32_ref_t wbuf_ping[kOutTile][IN_DIM];
  fp32_ref_t wbuf_pong[kOutTile][IN_DIM];

  int out_base = 0;
  bool use_ping = true;
  prefetch_weight_tile<OUT_DIM, IN_DIM>(w, out_base, wbuf_ping);

  while (out_base < OUT_DIM) {
    const int next_out_base = out_base + kOutTile;
    if (next_out_base < OUT_DIM) {
      if (use_ping) {
        prefetch_weight_tile<OUT_DIM, IN_DIM>(w, next_out_base, wbuf_pong);
      } else {
        prefetch_weight_tile<OUT_DIM, IN_DIM>(w, next_out_base, wbuf_ping);
      }
    }

    fp32_ref_t (*cur)[IN_DIM] = use_ping ? wbuf_ping : wbuf_pong;
    for (int tile_o = 0; tile_o < kOutTile; ++tile_o) {
      const int out_idx = out_base + tile_o;
      if (out_idx >= OUT_DIM) {
        continue;
      }
      fp32_ref_t acc = fp32_ref_t(static_cast<float>(b[out_idx]));
      for (int in_idx = 0; in_idx < IN_DIM; ++in_idx) {
        acc += x[in_idx] * cur[tile_o][in_idx];
      }
      y[out_idx] = acc;
    }

    out_base = next_out_base;
    use_ping = !use_ping;
  }
}

static inline fp32_ref_t dot_head(
  const fp32_ref_t q_vec[kDModel],
  const fp32_ref_t k_vec[kDModel],
  int head_idx
) {
  const int base = head_idx * kDHead;
  fp32_ref_t dot = fp32_ref_t(0.0f);
  for (int dh = 0; dh < kDHead; ++dh) {
    dot += q_vec[base + dh] * k_vec[base + dh];
  }
  return dot;
}

// Online single-pass softmax update for one head state.
static inline void online_softmax_update(
  bool &is_init,
  fp32_ref_t score,
  const fp32_ref_t v_head[kDHead],
  fp32_ref_t &max_score,
  fp32_ref_t &sumexp,
  fp32_ref_t acc_vec[kDHead]
) {
  if (!is_init) {
    max_score = score;
    sumexp = fp32_ref_t(1.0f);
    for (int dh = 0; dh < kDHead; ++dh) {
      acc_vec[dh] = v_head[dh];
    }
    is_init = true;
    return;
  }

  if (score > max_score) {
    const fp32_ref_t rescale = ref_softmax_exp_lut(max_score - score);
    sumexp = (sumexp * rescale) + fp32_ref_t(1.0f);
    for (int dh = 0; dh < kDHead; ++dh) {
      acc_vec[dh] = (acc_vec[dh] * rescale) + v_head[dh];
    }
    max_score = score;
    return;
  }

  const fp32_ref_t w = ref_softmax_exp_lut(score - max_score);
  sumexp += w;
  for (int dh = 0; dh < kDHead; ++dh) {
    acc_vec[dh] += w * v_head[dh];
  }
}

static inline void layernorm_token(
  const fp32_ref_t x[kDModel],
  const double gamma[kDModel],
  const double beta[kDModel],
  fp32_ref_t y[kDModel]
) {
  fp32_ref_t sum = fp32_ref_t(0.0f);
  for (int i = 0; i < kDModel; ++i) {
    sum += x[i];
  }
  const fp32_ref_t mean = sum * kInvDModel;

  fp32_ref_t var_acc = fp32_ref_t(0.0f);
  for (int i = 0; i < kDModel; ++i) {
    const fp32_ref_t d = x[i] - mean;
    var_acc += d * d;
  }
  const fp32_ref_t var = var_acc * kInvDModel;
  const fp32_ref_t inv_std = ref_inv_sqrt_approx(var + kLnEps);

  for (int i = 0; i < kDModel; ++i) {
    const fp32_ref_t xn = (x[i] - mean) * inv_std;
    const fp32_ref_t g = fp32_ref_t(static_cast<float>(gamma[i]));
    const fp32_ref_t b = fp32_ref_t(static_cast<float>(beta[i]));
    y[i] = xn * g + b;
  }
}

static inline void build_masks(
  bool one_ring[kTokens][kTokens],
  bool second_ring[kTokens][kTokens]
) {
  bool src[kTokens][kTokens];
  for (int i = 0; i < kTokens; ++i) {
    for (int j = 0; j < kTokens; ++j) {
      src[i][j] = (w_src_mask[i * kTokens + j].to_int() != 0);
    }
  }

  for (int i = 0; i < kTokens; ++i) {
    for (int j = 0; j < kTokens; ++j) {
      const bool i_is_var = (i < kVars);
      const bool j_is_var = (j < kVars);

      if (i_is_var && j_is_var) {
        one_ring[i][j] = true;
        second_ring[i][j] = src[i][j];
      } else if (i_is_var && (!j_is_var)) {
        one_ring[i][j] = src[i][j];
        second_ring[i][j] = true;
      } else if ((!i_is_var) && j_is_var) {
        one_ring[i][j] = src[i][j];
        second_ring[i][j] = true;
      } else {
        one_ring[i][j] = true;
        second_ring[i][j] = src[i][j];
      }
    }
  }
}

static inline bool get_layer_config(int layer_idx, LayerConfig &cfg) {
  if (layer_idx == 0) {
    cfg.w_q = w_decoder_layers_0_self_attn_linears_0_weight;
    cfg.b_q = w_decoder_layers_0_self_attn_linears_0_bias;
    cfg.w_k = w_decoder_layers_0_self_attn_linears_1_weight;
    cfg.b_k = w_decoder_layers_0_self_attn_linears_1_bias;
    cfg.w_v = w_decoder_layers_0_self_attn_linears_2_weight;
    cfg.b_v = w_decoder_layers_0_self_attn_linears_2_bias;
    cfg.w_o = w_decoder_layers_0_self_attn_linears_3_weight;
    cfg.b_o = w_decoder_layers_0_self_attn_linears_3_bias;
    cfg.w_ff1 = w_decoder_layers_0_feed_forward_w_1_weight;
    cfg.b_ff1 = w_decoder_layers_0_feed_forward_w_1_bias;
    cfg.w_ff2 = w_decoder_layers_0_feed_forward_w_2_weight;
    cfg.b_ff2 = w_decoder_layers_0_feed_forward_w_2_bias;
    cfg.ln0_w = w_decoder_layers_0_sublayer_0_norm_weight;
    cfg.ln0_b = w_decoder_layers_0_sublayer_0_norm_bias;
    cfg.ln1_w = w_decoder_layers_0_sublayer_1_norm_weight;
    cfg.ln1_b = w_decoder_layers_0_sublayer_1_norm_bias;
    cfg.s_x_in = fp32_ref_t(static_cast<float>(l0_in_s_x));
    cfg.s_x_o = fp32_ref_t(static_cast<float>(l0_o_s_x));
    cfg.s_x_ff1 = fp32_ref_t(static_cast<float>(l0_ff1_s_x));
    cfg.s_x_ff2 = fp32_ref_t(static_cast<float>(l0_ff2_s_x));
    cfg.inv_q = kInvScaleL0Q;
    cfg.inv_k = kInvScaleL0K;
    cfg.inv_v = kInvScaleL0V;
    cfg.inv_o = kInvScaleL0O;
    cfg.inv_ff1 = kInvScaleL0Ff1;
    cfg.inv_ff2 = kInvScaleL0Ff2;
    return true;
  }
  if (layer_idx == 1) {
    cfg.w_q = w_decoder_layers_1_self_attn_linears_0_weight;
    cfg.b_q = w_decoder_layers_1_self_attn_linears_0_bias;
    cfg.w_k = w_decoder_layers_1_self_attn_linears_1_weight;
    cfg.b_k = w_decoder_layers_1_self_attn_linears_1_bias;
    cfg.w_v = w_decoder_layers_1_self_attn_linears_2_weight;
    cfg.b_v = w_decoder_layers_1_self_attn_linears_2_bias;
    cfg.w_o = w_decoder_layers_1_self_attn_linears_3_weight;
    cfg.b_o = w_decoder_layers_1_self_attn_linears_3_bias;
    cfg.w_ff1 = w_decoder_layers_1_feed_forward_w_1_weight;
    cfg.b_ff1 = w_decoder_layers_1_feed_forward_w_1_bias;
    cfg.w_ff2 = w_decoder_layers_1_feed_forward_w_2_weight;
    cfg.b_ff2 = w_decoder_layers_1_feed_forward_w_2_bias;
    cfg.ln0_w = w_decoder_layers_1_sublayer_0_norm_weight;
    cfg.ln0_b = w_decoder_layers_1_sublayer_0_norm_bias;
    cfg.ln1_w = w_decoder_layers_1_sublayer_1_norm_weight;
    cfg.ln1_b = w_decoder_layers_1_sublayer_1_norm_bias;
    cfg.s_x_in = fp32_ref_t(static_cast<float>(l1_in_s_x));
    cfg.s_x_o = fp32_ref_t(static_cast<float>(l1_o_s_x));
    cfg.s_x_ff1 = fp32_ref_t(static_cast<float>(l1_ff1_s_x));
    cfg.s_x_ff2 = fp32_ref_t(static_cast<float>(l1_ff2_s_x));
    cfg.inv_q = kInvScaleL1Q;
    cfg.inv_k = kInvScaleL1K;
    cfg.inv_v = kInvScaleL1V;
    cfg.inv_o = kInvScaleL1O;
    cfg.inv_ff1 = kInvScaleL1Ff1;
    cfg.inv_ff2 = kInvScaleL1Ff2;
    return true;
  }
  return false;
}

static void run_layer_writeback(
  const LayerConfig &cfg,
  fp32_ref_t x_region[kXRegionWords],
  int x_work_base,
  const bool one_ring[kTokens][kTokens],
  const bool second_ring[kTokens][kTokens],
  fp32_ref_t scr_k[kTokens][kDModel],
  fp32_ref_t scr_v[kTokens][kDModel],
  bool &scr_k_live,
  bool &scr_v_live
) {
  for (int n = 0; n < kTokens; ++n) {
    fp32_ref_t x_row[kDModel];
    load_x_token(x_region, x_work_base, n, x_row);

    quant_linear_vec_tiled<kDModel, kDModel>(
      x_row,
      cfg.w_k,
      cfg.b_k,
      cfg.s_x_in,
      cfg.inv_k,
      scr_k[n]
    );
    quant_linear_vec_tiled<kDModel, kDModel>(
      x_row,
      cfg.w_v,
      cfg.b_v,
      cfg.s_x_in,
      cfg.inv_v,
      scr_v[n]
    );
  }
  scr_k_live = true;
  scr_v_live = true;

  fp32_ref_t attn_fifo[kFifoDepth][kDModel];
  fp32_ref_t ffn1_fifo[kFifoDepth][kFfnDim];

  for (int q_idx = 0; q_idx < kTokens; ++q_idx) {
    const int slot = q_idx & (kFifoDepth - 1);

    fp32_ref_t x_q[kDModel];
    fp32_ref_t q_vec[kDModel];
    fp32_ref_t post_concat[kDModel];
    fp32_ref_t ln0_in[kDModel];
    fp32_ref_t ln0_out[kDModel];
    fp32_ref_t ffn2_out[kDModel];
    fp32_ref_t ln1_in[kDModel];
    fp32_ref_t ln1_out[kDModel];

    load_x_token(x_region, x_work_base, q_idx, x_q);

    quant_linear_vec_tiled<kDModel, kDModel>(
      x_q,
      cfg.w_q,
      cfg.b_q,
      cfg.s_x_in,
      cfg.inv_q,
      q_vec
    );

    for (int h = 0; h < kHeads; ++h) {
      const bool (*mask)[kTokens] = (h < (kHeads / 2)) ? one_ring : second_ring;
      const int base = h * kDHead;
      bool has_valid = false;
      bool online_init = false;
      fp32_ref_t online_max = kNegLarge;
      fp32_ref_t online_sumexp = fp32_ref_t(0.0f);

      fp32_ref_t acc_vec[kDHead];
      for (int dh = 0; dh < kDHead; ++dh) {
        acc_vec[dh] = fp32_ref_t(0.0f);
      }

      for (int k_idx = 0; k_idx < kTokens; ++k_idx) {
        if (mask[q_idx][k_idx]) {
          continue;
        }
        has_valid = true;
        const fp32_ref_t score = dot_head(q_vec, scr_k[k_idx], h) * kInvSqrtDHead;
        online_softmax_update(
          online_init,
          score,
          &scr_v[k_idx][base],
          online_max,
          online_sumexp,
          acc_vec
        );
      }

      if (!has_valid) {
        for (int dh = 0; dh < kDHead; ++dh) {
          post_concat[base + dh] = fp32_ref_t(0.0f);
        }
        continue;
      }

      const fp32_ref_t inv_sumexp = ref_softmax_rcp_lut(online_sumexp);
      for (int dh = 0; dh < kDHead; ++dh) {
        post_concat[base + dh] = acc_vec[dh] * inv_sumexp;
      }
    }

    quant_linear_vec_tiled<kDModel, kDModel>(
      post_concat,
      cfg.w_o,
      cfg.b_o,
      cfg.s_x_o,
      cfg.inv_o,
      attn_fifo[slot]
    );

    for (int d = 0; d < kDModel; ++d) {
      ln0_in[d] = attn_fifo[slot][d] + x_q[d];
    }
    layernorm_token(ln0_in, cfg.ln0_w, cfg.ln0_b, ln0_out);

    quant_linear_vec_tiled<kFfnDim, kDModel>(
      ln0_out,
      cfg.w_ff1,
      cfg.b_ff1,
      cfg.s_x_ff1,
      cfg.inv_ff1,
      ffn1_fifo[slot]
    );

    for (int i = 0; i < kFfnDim; ++i) {
      ffn1_fifo[slot][i] = fp32_relu(ffn1_fifo[slot][i]);
    }

    quant_linear_vec_tiled<kDModel, kFfnDim>(
      ffn1_fifo[slot],
      cfg.w_ff2,
      cfg.b_ff2,
      cfg.s_x_ff2,
      cfg.inv_ff2,
      ffn2_out
    );

    for (int d = 0; d < kDModel; ++d) {
      ln1_in[d] = ffn2_out[d] + ln0_out[d];
    }
    layernorm_token(ln1_in, cfg.ln1_w, cfg.ln1_b, ln1_out);
    store_x_token(x_region, x_work_base, q_idx, ln1_out);
  }

  scr_k_live = false;
  scr_v_live = false;
}

static void run_final_layer_pass_a(
  const LayerConfig &cfg,
  fp32_ref_t x_region[kXRegionWords],
  int x_work_base,
  const bool one_ring[kTokens][kTokens],
  const bool second_ring[kTokens][kTokens],
  fp32_ref_t scr_k[kTokens][kDModel],
  fp32_ref_t scr_v[kTokens][kDModel],
  fp32_ref_t final_scalar_buf[ModelShapes::SCR_FINAL_SCALAR_WORDS],
  bool &scr_k_live,
  bool &scr_v_live
) {
  for (int n = 0; n < kTokens; ++n) {
    fp32_ref_t x_row[kDModel];
    load_x_token(x_region, x_work_base, n, x_row);

    quant_linear_vec_tiled<kDModel, kDModel>(
      x_row,
      cfg.w_k,
      cfg.b_k,
      cfg.s_x_in,
      cfg.inv_k,
      scr_k[n]
    );
    quant_linear_vec_tiled<kDModel, kDModel>(
      x_row,
      cfg.w_v,
      cfg.b_v,
      cfg.s_x_in,
      cfg.inv_v,
      scr_v[n]
    );
  }
  scr_k_live = true;
  scr_v_live = true;

  fp32_ref_t attn_fifo[kFifoDepth][kDModel];
  fp32_ref_t ffn1_fifo[kFifoDepth][kFfnDim];

  for (int q_idx = 0; q_idx < kTokens; ++q_idx) {
    const int slot = q_idx & (kFifoDepth - 1);

    fp32_ref_t x_q[kDModel];
    fp32_ref_t q_vec[kDModel];
    fp32_ref_t post_concat[kDModel];
    fp32_ref_t ln0_in[kDModel];
    fp32_ref_t ln0_out[kDModel];
    fp32_ref_t ffn2_out[kDModel];
    fp32_ref_t ln1_in[kDModel];
    fp32_ref_t ln1_out[kDModel];
    fp32_ref_t token_norm[kDModel];
    fp32_ref_t token_logit[1];

    load_x_token(x_region, x_work_base, q_idx, x_q);

    quant_linear_vec_tiled<kDModel, kDModel>(
      x_q,
      cfg.w_q,
      cfg.b_q,
      cfg.s_x_in,
      cfg.inv_q,
      q_vec
    );

    for (int h = 0; h < kHeads; ++h) {
      const bool (*mask)[kTokens] = (h < (kHeads / 2)) ? one_ring : second_ring;
      const int base = h * kDHead;
      bool has_valid = false;
      bool online_init = false;
      fp32_ref_t online_max = kNegLarge;
      fp32_ref_t online_sumexp = fp32_ref_t(0.0f);

      fp32_ref_t acc_vec[kDHead];
      for (int dh = 0; dh < kDHead; ++dh) {
        acc_vec[dh] = fp32_ref_t(0.0f);
      }

      for (int k_idx = 0; k_idx < kTokens; ++k_idx) {
        if (mask[q_idx][k_idx]) {
          continue;
        }
        has_valid = true;
        const fp32_ref_t score = dot_head(q_vec, scr_k[k_idx], h) * kInvSqrtDHead;
        online_softmax_update(
          online_init,
          score,
          &scr_v[k_idx][base],
          online_max,
          online_sumexp,
          acc_vec
        );
      }

      if (!has_valid) {
        for (int dh = 0; dh < kDHead; ++dh) {
          post_concat[base + dh] = fp32_ref_t(0.0f);
        }
        continue;
      }

      const fp32_ref_t inv_sumexp = ref_softmax_rcp_lut(online_sumexp);
      for (int dh = 0; dh < kDHead; ++dh) {
        post_concat[base + dh] = acc_vec[dh] * inv_sumexp;
      }
    }

    quant_linear_vec_tiled<kDModel, kDModel>(
      post_concat,
      cfg.w_o,
      cfg.b_o,
      cfg.s_x_o,
      cfg.inv_o,
      attn_fifo[slot]
    );

    for (int d = 0; d < kDModel; ++d) {
      ln0_in[d] = attn_fifo[slot][d] + x_q[d];
    }
    layernorm_token(ln0_in, cfg.ln0_w, cfg.ln0_b, ln0_out);

    quant_linear_vec_tiled<kFfnDim, kDModel>(
      ln0_out,
      cfg.w_ff1,
      cfg.b_ff1,
      cfg.s_x_ff1,
      cfg.inv_ff1,
      ffn1_fifo[slot]
    );

    for (int i = 0; i < kFfnDim; ++i) {
      ffn1_fifo[slot][i] = fp32_relu(ffn1_fifo[slot][i]);
    }

    quant_linear_vec_tiled<kDModel, kFfnDim>(
      ffn1_fifo[slot],
      cfg.w_ff2,
      cfg.b_ff2,
      cfg.s_x_ff2,
      cfg.inv_ff2,
      ffn2_out
    );

    for (int d = 0; d < kDModel; ++d) {
      ln1_in[d] = ffn2_out[d] + ln0_out[d];
    }
    layernorm_token(ln1_in, cfg.ln1_w, cfg.ln1_b, ln1_out);

    // Logical name: endLN_out for FinalHead input.
    layernorm_token(ln1_out, w_decoder_norm_weight, w_decoder_norm_bias, token_norm);
    store_x_token(x_region, x_work_base, q_idx, token_norm);
    dense_vec_tiled<1, kDModel>(
      token_norm,
      w_oned_final_embed_0_weight,
      w_oned_final_embed_0_bias,
      token_logit
    );

    if (q_idx < ModelShapes::SCR_FINAL_SCALAR_WORDS) {
      // Logical value s_t is staged in FINAL_SCALAR_BUF semantics.
      final_scalar_buf[q_idx] = token_logit[0];
    }
  }

  scr_k_live = false;
  scr_v_live = false;
}

static inline void xpred_set_bit(
  u32_word_t xpred_word_buf[ModelShapes::XPRED_WORDS],
  int var_idx,
  bool bit
) {
  if (!bit) {
    return;
  }
  const int word_idx = (var_idx >> 5);
  const int bit_idx = (var_idx & 31);
  xpred_word_buf[word_idx] = xpred_word_buf[word_idx] | (u32_word_t(1) << bit_idx);
}

static void run_pass_b_and_emit(
  const fp32_ref_t final_scalar_buf[ModelShapes::SCR_FINAL_SCALAR_WORDS],
  const fp32_ref_t y_var[kVars],
  uint32_t outmode,
  ac_channel<u32_word_t> &data_out,
  RefStep0RunReport &report
) {
  report.pass_b_executed = true;

  u32_word_t xpred_word_buf[ModelShapes::XPRED_WORDS];
  for (int w = 0; w < ModelShapes::XPRED_WORDS; ++w) {
    xpred_word_buf[w] = u32_word_t(0);
  }

  for (int class_base = 0; class_base < ModelShapes::OUT_DIM; class_base += kClassTile) {
    fp32_ref_t acc_tile[kClassTile];

    for (int tc = 0; tc < kClassTile; ++tc) {
      const int class_idx = class_base + tc;
      if (class_idx < ModelShapes::OUT_DIM) {
        acc_tile[tc] = fp32_ref_t(static_cast<float>(w_out_fc_bias[class_idx]));
      } else {
        acc_tile[tc] = fp32_ref_t(0.0f);
      }
    }

    for (int t = 0; t < ModelShapes::T_TOKENS; ++t) {
      const fp32_ref_t st = final_scalar_buf[t];
      for (int tc = 0; tc < kClassTile; ++tc) {
        const int class_idx = class_base + tc;
        if (class_idx >= ModelShapes::OUT_DIM) {
          continue;
        }
        const fp32_ref_t w_ct = fp32_ref_t(static_cast<float>(w_out_fc_weight[class_idx * ModelShapes::T_TOKENS + t]));
        acc_tile[tc] += w_ct * st;
      }
    }

    for (int tc = 0; tc < kClassTile; ++tc) {
      const int class_idx = class_base + tc;
      if (class_idx >= ModelShapes::OUT_DIM) {
        continue;
      }

      const fp32_ref_t logit = acc_tile[tc];
      if (outmode == 1u) {
        data_out.write(bits_from_fp32(logit));
        report.output_words += 1u;
        continue;
      }

      if (outmode == 0u) {
        for (int var_idx = 0; var_idx < ModelShapes::N_VARS; ++var_idx) {
          if (ModelShapes::map_var_to_class(var_idx) != class_idx) {
            continue;
          }
          const fp32_ref_t decision = logit * sign_fp32(y_var[var_idx]);
          const bool bit = (decision < fp32_ref_t(0.0f));
          xpred_set_bit(xpred_word_buf, var_idx, bit);
        }
      }
    }
  }

  if (outmode == 0u) {
    for (int w = 0; w < ModelShapes::XPRED_WORDS; ++w) {
      data_out.write(xpred_word_buf[w]);
      report.output_words += 1u;
    }
  }
}

} // namespace

void ref_step0_set_outmode(ac_int<2, false> mode) {
  const uint32_t m = static_cast<uint32_t>(mode.to_uint());
  if (m <= 2u) {
    g_outmode_reg = mode;
  } else {
    g_outmode_reg = ac_int<2, false>(2);
  }
}

const RefStep0RunReport &ref_step0_get_last_report() {
  return g_last_report;
}

void ref_step0_synth(
  ac_channel<fp32_ref_t> &in_y_ch,
  ac_channel<u32_word_t> &data_out
) {
  RefStep0RunReport report;
  clear_report(report);

  const uint32_t outmode = static_cast<uint32_t>(g_outmode_reg.to_uint());

  fp32_ref_t y_var[kVars];
  int y_hard[kVars];
  for (int i = 0; i < kVars; ++i) {
    const fp32_ref_t y = in_y_ch.read();
    y_var[i] = y;
    y_hard[i] = (y < fp32_ref_t(0.0f)) ? 1 : 0;
  }

  bool one_ring[kTokens][kTokens];
  bool second_ring[kTokens][kTokens];
  build_masks(one_ring, second_ring);

  bool scr_k_live = false;
  bool scr_v_live = false;
  check_final_scalar_region(report, scr_k_live, scr_v_live);

  if (outmode == 0u) {
    (void)validate_var_to_class_map(report);
  }

  const int src_embed_dim = w_src_embed_shape[1];
  const int lpe_token_dim = w_lpe_token_shape[1];
  if ((src_embed_dim + lpe_token_dim) > kDModel) {
    set_report_error(report, REF_STEP0_ERR_UNSUPPORTED_LAYER, REF_STEP0_MSG_UNSUPPORTED_LAYER);
  }

  if (report.has_error) {
    report.pass_b_executed = false;
    report.output_words = 0u;
    g_last_report = report;
    return;
  }

  static fp32_ref_t x_region[kXRegionWords];
  static fp32_ref_t scr_k[kTokens][kDModel];
  static fp32_ref_t scr_v[kTokens][kDModel];
  static fp32_ref_t final_scalar_buf[ModelShapes::SCR_FINAL_SCALAR_WORDS];

  for (int i = 0; i < ModelShapes::SCR_FINAL_SCALAR_WORDS; ++i) {
    final_scalar_buf[i] = fp32_ref_t(0.0f);
  }

  const int x_work_base = ModelShapes::X_WORK_BASE;

  fp32_ref_t node_feature[kTokens];
  for (int i = 0; i < kVars; ++i) {
    node_feature[i] = fp32_abs(y_var[i]);
  }
  for (int c = 0; c < kChecks; ++c) {
    int parity = 0;
    for (int v = 0; v < kVars; ++v) {
      if (h_H[c * kVars + v].to_int() != 0) {
        parity ^= y_hard[v];
      }
    }
    node_feature[kVars + c] = (parity == 0) ? fp32_ref_t(1.0f) : fp32_ref_t(-1.0f);
  }

  for (int t = 0; t < kTokens; ++t) {
    fp32_ref_t x_row[kDModel];
    for (int d = 0; d < kDModel; ++d) {
      x_row[d] = fp32_ref_t(0.0f);
    }
    for (int k = 0; k < src_embed_dim; ++k) {
      x_row[k] = node_feature[t] * fp32_ref_t(static_cast<float>(w_src_embed[t * src_embed_dim + k]));
    }
    for (int k = 0; k < lpe_token_dim; ++k) {
      x_row[src_embed_dim + k] = fp32_ref_t(static_cast<float>(w_lpe_token[t * lpe_token_dim + k]));
    }
    store_x_token(x_region, x_work_base, t, x_row);
  }

  if (ModelShapes::N_LAYERS == 1) {
    LayerConfig final_cfg;
    if (!get_layer_config(0, final_cfg)) {
      set_report_error(report, REF_STEP0_ERR_UNSUPPORTED_LAYER, REF_STEP0_MSG_UNSUPPORTED_LAYER);
    } else {
      run_final_layer_pass_a(
        final_cfg,
        x_region,
        x_work_base,
        one_ring,
        second_ring,
        scr_k,
        scr_v,
        final_scalar_buf,
        scr_k_live,
        scr_v_live
      );
    }
  } else if (ModelShapes::N_LAYERS == 2) {
    LayerConfig layer0_cfg;
    LayerConfig layer1_cfg;

    if (!get_layer_config(0, layer0_cfg) || !get_layer_config(1, layer1_cfg)) {
      set_report_error(report, REF_STEP0_ERR_UNSUPPORTED_LAYER, REF_STEP0_MSG_UNSUPPORTED_LAYER);
    } else {
      run_layer_writeback(
        layer0_cfg,
        x_region,
        x_work_base,
        one_ring,
        second_ring,
        scr_k,
        scr_v,
        scr_k_live,
        scr_v_live
      );

      for (int t = 0; t < kTokens; ++t) {
        fp32_ref_t in_row[kDModel];
        fp32_ref_t out_row[kDModel];
        load_x_token(x_region, x_work_base, t, in_row);
        layernorm_token(in_row, w_decoder_norm2_weight, w_decoder_norm2_bias, out_row);
        store_x_token(x_region, x_work_base, t, out_row);
      }

      run_final_layer_pass_a(
        layer1_cfg,
        x_region,
        x_work_base,
        one_ring,
        second_ring,
        scr_k,
        scr_v,
        final_scalar_buf,
        scr_k_live,
        scr_v_live
      );
    }
  } else {
    set_report_error(report, REF_STEP0_ERR_UNSUPPORTED_LAYER, REF_STEP0_MSG_UNSUPPORTED_LAYER);
  }

  report.final_layer_no_writeback_enforced = (report.final_layer_writeback_words == 0u);
  report.final_head_used_page_next = false;

  if (report.has_error) {
    report.pass_b_executed = false;
    report.output_words = 0u;
    g_last_report = report;
    return;
  }

  if (outmode == 2u) {
    report.pass_b_executed = false;
    report.output_words = 0u;
    g_last_report = report;
    return;
  }

  run_pass_b_and_emit(final_scalar_buf, y_var, outmode, data_out, report);

  const uint32_t expected_words = output_words_for_mode(outmode);
  if (report.output_words != expected_words) {
    set_report_error(report, REF_STEP0_ERR_UNSUPPORTED_LAYER, REF_STEP0_MSG_UNSUPPORTED_LAYER);
  }

  if (report.has_error) {
    report.output_words = 0u;
  }

  g_last_report = report;
}

} // namespace aecct_ref

