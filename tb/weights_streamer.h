// weights_streamer.h
#pragma once

// ============================================================
// TB helper: stream unified PARAM payload (v11.4)
// ------------------------------------------------------------
// - ? testbench ?函?????甈??極?瘀?銝閬◤ HLS ????// - 靘?WeightStreamOrder.h ??摨???weights.h ?抒??頧? u32 raw words嚗?data_in??// - v11.4嚗? SET_W_BASE嚗身摰?param_base_word嚗???LOAD_W嚗?PARAM嚗ias + weights + bitpack嚗?//
// English:
// - Testbench-only payload streamer for v11.4 unified PARAM.
// - Converts weights.h tensors to u32 raw stream following WeightStreamOrder.h.
// - This patch keeps the legacy helpers intact and adds v12.1 bridge helpers for
//   io16 transport and 16-bit SRAM storage words.
//
// Generated on: 2026-02-24
// ============================================================

#ifndef __SYNTHESIS__

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "ac_channel.h"
#include "ac_int.h"

#include "AecctUtil.h"
#include "gen/ModelShapes.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"
#include "weights.h"

// ------------------------------------------------------------
// Opcode constants (must match your v11.4 Top decode)
// ------------------------------------------------------------
static const uint8_t OPC_CFG_BEGIN   = 0x01;
static const uint8_t OPC_CFG_COMMIT  = 0x02;
static const uint8_t OPC_SET_W_BASE  = 0x09; // v11.4 new
static const uint8_t OPC_LOAD_W      = 0x04; // v11.4: unified PARAM stream
static const uint8_t OPC_SET_OUTMODE = 0x05;
static const uint8_t OPC_INFER       = 0x06;
static const uint8_t OPC_READ_MEM    = 0x07;
static const uint8_t OPC_DEBUG_CFG   = 0x08;
static const uint8_t OPC_SOFT_RESET  = 0x7F;

// ctrl_rsp format (v11.*): [3:0] kind, [15:8] payload/err_code
static const uint8_t RSP_KIND_OK   = 0;
static const uint8_t RSP_KIND_DONE = 1;
static const uint8_t RSP_KIND_ERR  = 2;

// ------------------------------------------------------------
// Bitcast helpers
// ------------------------------------------------------------
static inline uint32_t tb_fp32_bits_from_double(const double x) {
  return (uint32_t)aecct::fp32_bits_from_double(x).to_uint();
}

static inline uint32_t tb_u16_to_u32(const uint16_t x) {
  return (uint32_t)x;
}

template <class T>
static inline uint16_t tb_to_u16(const T x) {
  return (uint16_t)x;
}

template <class TDATA>
static inline void tb_write_u32(ac_channel<TDATA> &ch, const uint32_t w) {
  const TDATA t = (TDATA)w;
  ch.write(t);
}

template <class TDATA>
static inline void tb_write_u16(ac_channel<TDATA> &ch, const uint16_t w) {
  const TDATA t = (TDATA)w;
  ch.write(t);
}

template <class TDATA>
static inline void tb_write_logical_u32_as_io16(ac_channel<TDATA> &ch, const uint32_t w) {
  tb_write_u16(ch, (uint16_t)(w & 0xFFFFu));
  tb_write_u16(ch, (uint16_t)((w >> 16) & 0xFFFFu));
}

template <class TCTRL>
static inline void tb_write_cmd(ac_channel<TCTRL> &ch, const uint8_t opcode) {
  const TCTRL t = (TCTRL)opcode; // upper bits reserved=0
  ch.write(t);
}

template <class TRSP>
static inline uint8_t tb_rsp_kind(const TRSP rsp) {
  const uint16_t r = tb_to_u16(rsp);
  return (uint8_t)(r & 0x0Fu);
}

template <class TRSP>
static inline uint8_t tb_rsp_payload8(const TRSP rsp) {
  const uint16_t r = tb_to_u16(rsp);
  return (uint8_t)((r >> 8) & 0xFFu);
}

// ------------------------------------------------------------
// Tensor lookup (by enum id -> symbol in weights.h)
// ------------------------------------------------------------
// FP64 arrays in weights.h (we will cast to FP32 then stream bits)
// QUANT_SX_8: 8 scalar quantization scales (fp64 in weights.h; streamed as fp32 raw bits)
static const double tb_quant_sx_8[8] = {
  l0_in_s_x, l0_o_s_x, l0_ff1_s_x, l0_ff2_s_x,
  l1_in_s_x, l1_o_s_x, l1_ff1_s_x, l1_ff2_s_x
};
static inline const double* tb_lookup_weight_fp64(const WeightId id, uint32_t &out_numel) {
  switch (id) {
    case BCH_H_BITPACK: out_numel = 0u; return (const double*)0;
    case SRC_EMBED: out_numel = (uint32_t)w_src_embed_numel; return w_src_embed;
    // SRC_MASK is bitpack payload and must be consumed through tb_lookup_weight_bits.
    case SRC_MASK: out_numel = 0u; return (const double*)0;
    case QUANT_SX_8: out_numel = 8u; return tb_quant_sx_8;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_WEIGHT: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_0_weight_numel; return w_decoder_layers_0_self_attn_linears_0_weight;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_DELTA: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_0_delta_numel; return w_decoder_layers_0_self_attn_linears_0_delta;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_S_W: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_0_s_w_numel; return w_decoder_layers_0_self_attn_linears_0_s_w;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_WEIGHT: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_1_weight_numel; return w_decoder_layers_0_self_attn_linears_1_weight;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_DELTA: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_1_delta_numel; return w_decoder_layers_0_self_attn_linears_1_delta;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_S_W: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_1_s_w_numel; return w_decoder_layers_0_self_attn_linears_1_s_w;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_WEIGHT: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_2_weight_numel; return w_decoder_layers_0_self_attn_linears_2_weight;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_DELTA: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_2_delta_numel; return w_decoder_layers_0_self_attn_linears_2_delta;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_S_W: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_2_s_w_numel; return w_decoder_layers_0_self_attn_linears_2_s_w;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_WEIGHT: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_3_weight_numel; return w_decoder_layers_0_self_attn_linears_3_weight;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_DELTA: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_3_delta_numel; return w_decoder_layers_0_self_attn_linears_3_delta;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_S_W: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_3_s_w_numel; return w_decoder_layers_0_self_attn_linears_3_s_w;
    case DECODER_LAYERS_0_FEED_FORWARD_W_1_WEIGHT: out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_1_weight_numel; return w_decoder_layers_0_feed_forward_w_1_weight;
    case DECODER_LAYERS_0_FEED_FORWARD_W_1_DELTA: out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_1_delta_numel; return w_decoder_layers_0_feed_forward_w_1_delta;
    case DECODER_LAYERS_0_FEED_FORWARD_W_1_S_W: out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_1_s_w_numel; return w_decoder_layers_0_feed_forward_w_1_s_w;
    case DECODER_LAYERS_0_FEED_FORWARD_W_2_WEIGHT: out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_2_weight_numel; return w_decoder_layers_0_feed_forward_w_2_weight;
    case DECODER_LAYERS_0_FEED_FORWARD_W_2_DELTA: out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_2_delta_numel; return w_decoder_layers_0_feed_forward_w_2_delta;
    case DECODER_LAYERS_0_FEED_FORWARD_W_2_S_W: out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_2_s_w_numel; return w_decoder_layers_0_feed_forward_w_2_s_w;
    case DECODER_LAYERS_0_SUBLAYER_0_NORM_WEIGHT: out_numel = (uint32_t)w_decoder_layers_0_sublayer_0_norm_weight_numel; return w_decoder_layers_0_sublayer_0_norm_weight;
    case DECODER_LAYERS_0_SUBLAYER_1_NORM_WEIGHT: out_numel = (uint32_t)w_decoder_layers_0_sublayer_1_norm_weight_numel; return w_decoder_layers_0_sublayer_1_norm_weight;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_WEIGHT: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_0_weight_numel; return w_decoder_layers_1_self_attn_linears_0_weight;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_DELTA: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_0_delta_numel; return w_decoder_layers_1_self_attn_linears_0_delta;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_S_W: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_0_s_w_numel; return w_decoder_layers_1_self_attn_linears_0_s_w;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_WEIGHT: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_1_weight_numel; return w_decoder_layers_1_self_attn_linears_1_weight;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_DELTA: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_1_delta_numel; return w_decoder_layers_1_self_attn_linears_1_delta;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_S_W: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_1_s_w_numel; return w_decoder_layers_1_self_attn_linears_1_s_w;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_WEIGHT: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_2_weight_numel; return w_decoder_layers_1_self_attn_linears_2_weight;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_DELTA: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_2_delta_numel; return w_decoder_layers_1_self_attn_linears_2_delta;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_S_W: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_2_s_w_numel; return w_decoder_layers_1_self_attn_linears_2_s_w;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_WEIGHT: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_3_weight_numel; return w_decoder_layers_1_self_attn_linears_3_weight;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_DELTA: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_3_delta_numel; return w_decoder_layers_1_self_attn_linears_3_delta;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_S_W: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_3_s_w_numel; return w_decoder_layers_1_self_attn_linears_3_s_w;
    case DECODER_LAYERS_1_FEED_FORWARD_W_1_WEIGHT: out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_1_weight_numel; return w_decoder_layers_1_feed_forward_w_1_weight;
    case DECODER_LAYERS_1_FEED_FORWARD_W_1_DELTA: out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_1_delta_numel; return w_decoder_layers_1_feed_forward_w_1_delta;
    case DECODER_LAYERS_1_FEED_FORWARD_W_1_S_W: out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_1_s_w_numel; return w_decoder_layers_1_feed_forward_w_1_s_w;
    case DECODER_LAYERS_1_FEED_FORWARD_W_2_WEIGHT: out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_2_weight_numel; return w_decoder_layers_1_feed_forward_w_2_weight;
    case DECODER_LAYERS_1_FEED_FORWARD_W_2_DELTA: out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_2_delta_numel; return w_decoder_layers_1_feed_forward_w_2_delta;
    case DECODER_LAYERS_1_FEED_FORWARD_W_2_S_W: out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_2_s_w_numel; return w_decoder_layers_1_feed_forward_w_2_s_w;
    case DECODER_LAYERS_1_SUBLAYER_0_NORM_WEIGHT: out_numel = (uint32_t)w_decoder_layers_1_sublayer_0_norm_weight_numel; return w_decoder_layers_1_sublayer_0_norm_weight;
    case DECODER_LAYERS_1_SUBLAYER_1_NORM_WEIGHT: out_numel = (uint32_t)w_decoder_layers_1_sublayer_1_norm_weight_numel; return w_decoder_layers_1_sublayer_1_norm_weight;
    case DECODER_NORM_WEIGHT: out_numel = (uint32_t)w_decoder_norm_weight_numel; return w_decoder_norm_weight;
    case DECODER_NORM2_WEIGHT: out_numel = (uint32_t)w_decoder_norm2_weight_numel; return w_decoder_norm2_weight;
    case ONED_FINAL_EMBED_0_WEIGHT: out_numel = (uint32_t)w_oned_final_embed_0_weight_numel; return w_oned_final_embed_0_weight;
    case OUT_FC_WEIGHT: out_numel = (uint32_t)w_out_fc_weight_numel; return w_out_fc_weight;
    case LPE_TOKEN: out_numel = (uint32_t)w_lpe_token_numel; return w_lpe_token;
    default: out_numel = 0u; return (const double*)0;
  }
}

static inline const double* tb_lookup_bias_fp64(const BiasId id, uint32_t &out_numel) {
  switch (id) {
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_BIAS: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_0_bias_numel; return w_decoder_layers_0_self_attn_linears_0_bias;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_BIAS: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_1_bias_numel; return w_decoder_layers_0_self_attn_linears_1_bias;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_BIAS: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_2_bias_numel; return w_decoder_layers_0_self_attn_linears_2_bias;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_BIAS: out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_3_bias_numel; return w_decoder_layers_0_self_attn_linears_3_bias;
    case DECODER_LAYERS_0_FEED_FORWARD_W_1_BIAS: out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_1_bias_numel; return w_decoder_layers_0_feed_forward_w_1_bias;
    case DECODER_LAYERS_0_FEED_FORWARD_W_2_BIAS: out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_2_bias_numel; return w_decoder_layers_0_feed_forward_w_2_bias;
    case DECODER_LAYERS_0_SUBLAYER_0_NORM_BIAS: out_numel = (uint32_t)w_decoder_layers_0_sublayer_0_norm_bias_numel; return w_decoder_layers_0_sublayer_0_norm_bias;
    case DECODER_LAYERS_0_SUBLAYER_1_NORM_BIAS: out_numel = (uint32_t)w_decoder_layers_0_sublayer_1_norm_bias_numel; return w_decoder_layers_0_sublayer_1_norm_bias;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_BIAS: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_0_bias_numel; return w_decoder_layers_1_self_attn_linears_0_bias;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_BIAS: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_1_bias_numel; return w_decoder_layers_1_self_attn_linears_1_bias;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_BIAS: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_2_bias_numel; return w_decoder_layers_1_self_attn_linears_2_bias;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_BIAS: out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_3_bias_numel; return w_decoder_layers_1_self_attn_linears_3_bias;
    case DECODER_LAYERS_1_FEED_FORWARD_W_1_BIAS: out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_1_bias_numel; return w_decoder_layers_1_feed_forward_w_1_bias;
    case DECODER_LAYERS_1_FEED_FORWARD_W_2_BIAS: out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_2_bias_numel; return w_decoder_layers_1_feed_forward_w_2_bias;
    case DECODER_LAYERS_1_SUBLAYER_0_NORM_BIAS: out_numel = (uint32_t)w_decoder_layers_1_sublayer_0_norm_bias_numel; return w_decoder_layers_1_sublayer_0_norm_bias;
    case DECODER_LAYERS_1_SUBLAYER_1_NORM_BIAS: out_numel = (uint32_t)w_decoder_layers_1_sublayer_1_norm_bias_numel; return w_decoder_layers_1_sublayer_1_norm_bias;
    case DECODER_NORM_BIAS: out_numel = (uint32_t)w_decoder_norm_bias_numel; return w_decoder_norm_bias;
    case DECODER_NORM2_BIAS: out_numel = (uint32_t)w_decoder_norm2_bias_numel; return w_decoder_norm2_bias;
    case ONED_FINAL_EMBED_0_BIAS: out_numel = (uint32_t)w_oned_final_embed_0_bias_numel; return w_oned_final_embed_0_bias;
    case OUT_FC_BIAS: out_numel = (uint32_t)w_out_fc_bias_numel; return w_out_fc_bias;
    default: out_numel = 0u; return (const double*)0;
  }
}

// BIT arrays in weights.h (pack into u32 words, LSB-first)
static inline const ac_int<1,false>* tb_lookup_weight_bits(const WeightId id, uint32_t &out_num_bits) {
  switch (id) {
    case BCH_H_BITPACK: out_num_bits = (uint32_t)h_H_numel; return h_H;
    case SRC_MASK: out_num_bits = (uint32_t)w_src_mask_numel; return w_src_mask;
    default: out_num_bits = 0u; return (const ac_int<1,false>*)0;
  }
}

// ------------------------------------------------------------
// Streaming primitives
// ------------------------------------------------------------
template <class TDATA>
static inline void tb_emit_padding_zeros(ac_channel<TDATA> &data_in, const uint32_t n_words) {
  for (uint32_t i = 0; i < n_words; ++i) {
    tb_write_u32(data_in, 0u);
  }
}

template <class TDATA>
static inline void tb_emit_padding_zeros_io16(ac_channel<TDATA> &data_in, const uint32_t n_words16) {
  for (uint32_t i = 0; i < n_words16; ++i) {
    tb_write_u16(data_in, 0u);
  }
}

template <class TDATA>
static inline void tb_emit_fp32_words_from_fp64(ac_channel<TDATA> &data_in,
                                               const double *src,
                                               const uint32_t src_numel,
                                               const uint32_t stream_len_w) {
  // stream_len_w may be >= src_numel due to alignment/padding rules
  uint32_t i = 0;
  for (; i < src_numel; ++i) {
    tb_write_u32(data_in, tb_fp32_bits_from_double(src[i]));
  }
  if (stream_len_w > src_numel) {
    tb_emit_padding_zeros(data_in, stream_len_w - src_numel);
  }
}

template <class TDATA>
static inline void tb_emit_fp32_words_as_io16(ac_channel<TDATA> &data_in,
                                              const double *src,
                                              const uint32_t src_numel,
                                              const uint32_t stream_len_words16) {
  uint32_t out_words16 = 0u;
  for (uint32_t i = 0u; i < src_numel; ++i) {
    tb_write_logical_u32_as_io16(data_in, tb_fp32_bits_from_double(src[i]));
    out_words16 += 2u;
  }
  if (stream_len_words16 > out_words16) {
    tb_emit_padding_zeros_io16(data_in, stream_len_words16 - out_words16);
  }
}

template <class TDATA>
static inline bool tb_emit_inv_sw_words_from_fp64(ac_channel<TDATA> &data_in,
                                                 const double *src,
                                                 const uint32_t src_numel,
                                                 const uint32_t stream_len_w,
                                                 const WeightId wid) {
  uint32_t i = 0u;
  for (; i < src_numel; ++i) {
    const double s_w = src[i];
    if (s_w == 0.0) {
      std::fprintf(stderr,
                   "[tb][weights_streamer] ERROR: inv_s_w conversion failed (s_w==0), WeightId=%u, idx=%u\n",
                   (unsigned)wid, (unsigned)i);
      return false;
    }
    tb_write_u32(data_in, tb_fp32_bits_from_double(1.0 / s_w));
  }
  if (stream_len_w > src_numel) {
    tb_emit_padding_zeros(data_in, stream_len_w - src_numel);
  }
  return true;
}

static inline bool tb_ternary_code_from_fp64(const double v, uint32_t& out_code) {
  if (v == 1.0) {
    out_code = (uint32_t)TERNARY_CODE_POS;
    return true;
  }
  if (v == 0.0) {
    out_code = (uint32_t)TERNARY_CODE_ZERO;
    return true;
  }
  if (v == -1.0) {
    out_code = (uint32_t)TERNARY_CODE_NEG;
    return true;
  }
  return false;
}

static inline bool tb_pack_ternary_words_from_fp64(const double *src,
                                                  const uint32_t src_numel,
                                                  const uint32_t expected_num_weights,
                                                  uint32_t *out_payload,
                                                  const uint32_t out_capacity_words,
                                                  uint32_t& out_payload_words,
                                                  uint32_t& out_last_word_valid_count) {
  out_payload_words = 0u;
  out_last_word_valid_count = 0u;
  if (!src || !out_payload) {
    return false;
  }
  if (expected_num_weights == 0u || src_numel != expected_num_weights) {
    return false;
  }

  out_payload_words = ternary_payload_words_2b(expected_num_weights);
  out_last_word_valid_count = ternary_last_word_valid_count(expected_num_weights);
  if (out_capacity_words < out_payload_words) {
    return false;
  }

  for (uint32_t w = 0u; w < out_payload_words; ++w) {
    out_payload[w] = 0u;
  }

  for (uint32_t idx = 0u; idx < expected_num_weights; ++idx) {
    uint32_t code = 0u;
    if (!tb_ternary_code_from_fp64(src[idx], code)) {
      return false;
    }
    const uint32_t word_idx = (idx >> 4);          // /16
    const uint32_t shift = ((idx & 15u) << 1);     // *2
    out_payload[word_idx] |= ((code & 0x3u) << shift);
  }

  if (out_last_word_valid_count < 16u) {
    const uint32_t valid_bits = (out_last_word_valid_count << 1);
    const uint32_t mask = (1u << valid_bits) - 1u;
    out_payload[out_payload_words - 1u] &= mask;
  }

  return true;
}

static inline bool tb_pack_ternary_storage_words_from_fp64(const double *src,
                                                           const uint32_t src_numel,
                                                           const uint32_t expected_num_weights,
                                                           uint16_t *out_payload,
                                                           const uint32_t out_capacity_words16,
                                                           uint32_t& out_payload_words16,
                                                           uint32_t& out_last_word_valid_count16) {
  out_payload_words16 = 0u;
  out_last_word_valid_count16 = 0u;
  if (!src || !out_payload) {
    return false;
  }
  if (expected_num_weights == 0u || src_numel != expected_num_weights) {
    return false;
  }

  out_payload_words16 = ternary_payload_storage_words_2b(expected_num_weights);
  out_last_word_valid_count16 = ternary_last_storage_word_valid_count(expected_num_weights);
  if (out_capacity_words16 < out_payload_words16) {
    return false;
  }

  for (uint32_t w = 0u; w < out_payload_words16; ++w) {
    out_payload[w] = 0u;
  }

  for (uint32_t idx = 0u; idx < expected_num_weights; ++idx) {
    uint32_t code = 0u;
    if (!tb_ternary_code_from_fp64(src[idx], code)) {
      return false;
    }
    const uint32_t word_idx = (idx >> 3);      // /8
    const uint32_t shift = ((idx & 7u) << 1);  // *2
    out_payload[word_idx] = (uint16_t)(out_payload[word_idx] | ((uint16_t)(code & 0x3u) << shift));
  }

  if (out_last_word_valid_count16 < 8u) {
    const uint32_t valid_bits = (out_last_word_valid_count16 << 1);
    const uint16_t mask = (uint16_t)((1u << valid_bits) - 1u);
    out_payload[out_payload_words16 - 1u] = (uint16_t)(out_payload[out_payload_words16 - 1u] & mask);
  }

  return true;
}

static inline bool tb_decode_ternary_code_at(const uint32_t* payload,
                                           const uint32_t payload_words,
                                           const uint32_t weight_idx,
                                           uint32_t& out_code) {
  if (!payload) {
    return false;
  }
  const uint32_t word_idx = (weight_idx >> 4);      // /16
  if (word_idx >= payload_words) {
    return false;
  }
  const uint32_t shift = ((weight_idx & 15u) << 1); // *2
  out_code = (payload[word_idx] >> shift) & 0x3u;
  return true;
}

template <class TDATA>
static inline void tb_emit_bitpack_words(ac_channel<TDATA> &data_in,
                                        const ac_int<1,false> *bits,
                                        const uint32_t num_bits,
                                        const uint32_t stream_len_w) {
  // Packing rule: bit0 -> word0 bit0 (LSB), then increasing bit index.
  const uint32_t need_words = (num_bits + 31u) >> 5; // ceil(num_bits/32)
  uint32_t out_words = 0;
  uint32_t bit_idx = 0;

  for (uint32_t word_i = 0; word_i < need_words; ++word_i) {
    uint32_t word = 0u;
    for (uint32_t b = 0; b < 32u; ++b) {
      if (bit_idx < num_bits) {
        const uint32_t bit = (uint32_t)(bits[bit_idx].to_int());
        word |= (bit & 1u) << b;
      }
      ++bit_idx;
    }
    tb_write_u32(data_in, word);
    ++out_words;
  }
  // If stream_len_w > need_words, pad remaining words with zeros (must be zeros).
  if (stream_len_w > out_words) {
    tb_emit_padding_zeros(data_in, stream_len_w - out_words);
  }
}

template <class TDATA>
static inline void tb_emit_bitpack_words_as_io16(ac_channel<TDATA> &data_in,
                                                 const ac_int<1,false> *bits,
                                                 const uint32_t num_bits,
                                                 const uint32_t stream_len_words16) {
  const uint32_t need_words16 = storage_words_bits(num_bits);
  uint32_t out_words16 = 0u;
  uint32_t bit_idx = 0u;

  for (uint32_t word_i = 0u; word_i < need_words16; ++word_i) {
    uint16_t word = 0u;
    for (uint32_t b = 0u; b < 16u; ++b) {
      if (bit_idx < num_bits) {
        const uint32_t bit = (uint32_t)(bits[bit_idx].to_int());
        word = (uint16_t)(word | ((bit & 1u) << b));
      }
      ++bit_idx;
    }
    tb_write_u16(data_in, word);
    ++out_words16;
  }
  if (stream_len_words16 > out_words16) {
    tb_emit_padding_zeros_io16(data_in, stream_len_words16 - out_words16);
  }
}

// ------------------------------------------------------------
// High-level: v11.4 command helpers (with optional rsp checking)
// ------------------------------------------------------------
// NOTE:
// - This header assumes your TB calls read() on ctrl_rsp in-order.
// - If you don't want response checking, pass check_rsp=false.
template <class TCTRL, class TRSP, class TDATA>
static inline bool tb_send_set_w_base(ac_channel<TCTRL> &ctrl_cmd,
                                     ac_channel<TRSP> &ctrl_rsp,
                                     ac_channel<TDATA> &data_in,
                                     const uint32_t param_base_word,
                                     const bool check_rsp = true) {
  tb_write_cmd(ctrl_cmd, OPC_SET_W_BASE);
  tb_write_u32(data_in, param_base_word);

  if (!check_rsp) return true;

  const TRSP r0 = ctrl_rsp.read();
  if (tb_rsp_kind(r0) != RSP_KIND_OK || tb_rsp_payload8(r0) != OPC_SET_W_BASE) {
    return false;
  }
  return true;
}

template <class TCTRL, class TRSP, class TDATA>
static inline bool tb_send_load_w_unified_param(ac_channel<TCTRL> &ctrl_cmd,
                                               ac_channel<TRSP> &ctrl_rsp,
                                               ac_channel<TDATA> &data_in,
                                               const bool check_rsp = true) {
  // Issue LOAD_W (v11.4: unified PARAM stream)
  tb_write_cmd(ctrl_cmd, OPC_LOAD_W);

  if (check_rsp) {
    const TRSP r0 = ctrl_rsp.read();
    if (tb_rsp_kind(r0) != RSP_KIND_OK || tb_rsp_payload8(r0) != OPC_LOAD_W) {
      return false;
    }
  }

  // --------------------------------------------------------
  // Stream PARAM payload:
  //   [0] bias stream   (BIAS_COUNT entries, FP32)
  //   [1] weight stream (WEIGHT_COUNT entries, FP32 or BITPACK)
  // Total words = EXP_LEN_BIAS_WORDS + EXP_LEN_W_WORDS
  // --------------------------------------------------------

  // Bias part
  for (uint32_t i = 0; i < (uint32_t)BIAS_COUNT; ++i) {
    const BiasId bid = (BiasId)i;
    const TensorMeta meta = kBiasMeta[i];
    // dtype should be FP32 for bias
    uint32_t numel = 0u;
    const double *ptr = tb_lookup_bias_fp64(bid, numel);
    // Defensive: if ptr is null, emit zeros of the stream length
    if (!ptr || numel == 0u) {
      tb_emit_padding_zeros(data_in, meta.len_w);
    } else {
      tb_emit_fp32_words_from_fp64(data_in, ptr, numel, meta.len_w);
    }
  }

  // Weight part
  for (uint32_t i = 0; i < (uint32_t)WEIGHT_COUNT; ++i) {
    const WeightId wid = (WeightId)i;
    const TensorMeta meta = kWeightMeta[i];

    if (meta.dtype == 0u) {
      uint32_t numel = 0u;
      const double *ptr = tb_lookup_weight_fp64(wid, numel);
      if (!ptr || numel == 0u) {
        tb_emit_padding_zeros(data_in, meta.len_w);
      } else if (is_quant_linear_inv_sw_weight_slot(wid)) {
        if (!tb_emit_inv_sw_words_from_fp64(data_in, ptr, numel, meta.len_w, wid)) {
          return false;
        }
      } else {
        tb_emit_fp32_words_from_fp64(data_in, ptr, numel, meta.len_w);
      }
    } else {
      uint32_t num_bits = 0u;
      const ac_int<1,false> *bits = tb_lookup_weight_bits(wid, num_bits);
      if (!bits || num_bits == 0u) {
        tb_emit_padding_zeros(data_in, meta.len_w);
      } else {
        tb_emit_bitpack_words(data_in, bits, num_bits, meta.len_w);
      }
    }
  }

  // Optional DONE check
  if (check_rsp) {
    const TRSP r1 = ctrl_rsp.read();
    if (tb_rsp_kind(r1) != RSP_KIND_DONE || tb_rsp_payload8(r1) != OPC_LOAD_W) {
      return false;
    }
  }
  return true;
}

// Convenience: recommended default base if you keep v11.2 map contiguous
// (bias region immediately followed by weight region).
static inline uint32_t tb_default_param_base_word() {
  return (uint32_t)sram_map::PARAM_BASE_DEFAULT;
}

#endif // __SYNTHESIS__

