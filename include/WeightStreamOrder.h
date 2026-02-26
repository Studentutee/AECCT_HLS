// WeightStreamOrder.h
#pragma once
#include <cstdint>
#include "ModelShapes.h"
#include "SramMap.h"

// ============================================================
// WeightStreamOrder.h
// ------------------------------------------------------------
// Defines the *exact* streaming order.
//
// v11.4 main path:
//   - LOAD_W loads unified PARAM stream (bias + weights + bitpack)
//   - PARAM is relocated by runtime base set via SET_W_BASE
//
// v11.2 legacy path:
//   - LOAD_BIAS / LOAD_W split is kept as [legacy] metadata
//   - Use only if backward compatibility is required.
//
// TB and Top MUST follow the chosen order to match SRAM layout.
//
// Dtypes:
//   dtype = 0 -> FP32 (1 element = 1 u32 word)
//   dtype = 1 -> BITPACK (32 bits per u32 word, row-major packing)
// ============================================================

// [legacy] Meta used by separated LOAD_BIAS / LOAD_W flows (kept for compatibility)
struct TensorMeta {
  uint32_t offset_w;   // offset in words from the chosen base (v11.4: param_base_word; legacy: BASE_BIAS_W or BASE_W_W)
  uint32_t len_w;      // length in words
  uint32_t ndims;      // number of valid dims
  uint32_t d0;         // dim0 (列(Row))
  uint32_t d1;         // dim1 (行(Colume))
  uint32_t d2;         // dim2
  uint32_t d3;         // dim3
  uint32_t dtype;      // 0=FP32, 1=BITPACK
};

// ------------------------------------------------------------
// v11.4 main-path PARAM meta (unified table for LOAD_W / PARAM_RX)
// ------------------------------------------------------------
// NOTE:
// - 'id' is a stable sequential index (0..PARAM_COUNT-1) matching kParamMeta/kParamKey order.
// - 'valid_bits' is meaningful ONLY when dtype==BITPACK (1..32). For FP32, it MUST be 0.
// English:
// - Unified PARAM meta table required by v11.4 (id, dtype, offset_w, len_w, valid_bits, shape...).
enum ParamDType : uint32_t {
  PARAM_DTYPE_FP32   = 0u,
  PARAM_DTYPE_BITPACK = 1u
};

struct ParamMeta {
  uint32_t id;         // param id (stable index)
  uint32_t dtype;      // ParamDType
  uint32_t offset_w;   // offset in words from param_base_word
  uint32_t len_w;      // length in words
  uint32_t valid_bits; // BITPACK only: valid bits in last word (1..32); FP32: must be 0
  uint32_t ndims;      // number of valid dims
  uint32_t d0;         // dim0 (列(Row))
  uint32_t d1;         // dim1 (行(Colume))
  uint32_t d2;         // dim2
  uint32_t d3;         // dim3
};



// ----------------------------
// [legacy] WEIGHT stream (separated LOAD_W)
// ----------------------------
enum WeightId : uint32_t {
  BCH_H_BITPACK = 0,
  SRC_EMBED = 1,
  SRC_MASK = 2,
  QUANT_SX_8 = 3,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_WEIGHT = 4,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_DELTA = 5,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_S_W = 6,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_WEIGHT = 7,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_DELTA = 8,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_S_W = 9,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_WEIGHT = 10,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_DELTA = 11,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_S_W = 12,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_WEIGHT = 13,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_DELTA = 14,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_S_W = 15,
  DECODER_LAYERS_0_FEED_FORWARD_W_1_WEIGHT = 16,
  DECODER_LAYERS_0_FEED_FORWARD_W_1_DELTA = 17,
  DECODER_LAYERS_0_FEED_FORWARD_W_1_S_W = 18,
  DECODER_LAYERS_0_FEED_FORWARD_W_2_WEIGHT = 19,
  DECODER_LAYERS_0_FEED_FORWARD_W_2_DELTA = 20,
  DECODER_LAYERS_0_FEED_FORWARD_W_2_S_W = 21,
  DECODER_LAYERS_0_SUBLAYER_0_NORM_WEIGHT = 22,
  DECODER_LAYERS_0_SUBLAYER_1_NORM_WEIGHT = 23,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_WEIGHT = 24,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_DELTA = 25,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_S_W = 26,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_WEIGHT = 27,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_DELTA = 28,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_S_W = 29,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_WEIGHT = 30,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_DELTA = 31,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_S_W = 32,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_WEIGHT = 33,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_DELTA = 34,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_S_W = 35,
  DECODER_LAYERS_1_FEED_FORWARD_W_1_WEIGHT = 36,
  DECODER_LAYERS_1_FEED_FORWARD_W_1_DELTA = 37,
  DECODER_LAYERS_1_FEED_FORWARD_W_1_S_W = 38,
  DECODER_LAYERS_1_FEED_FORWARD_W_2_WEIGHT = 39,
  DECODER_LAYERS_1_FEED_FORWARD_W_2_DELTA = 40,
  DECODER_LAYERS_1_FEED_FORWARD_W_2_S_W = 41,
  DECODER_LAYERS_1_SUBLAYER_0_NORM_WEIGHT = 42,
  DECODER_LAYERS_1_SUBLAYER_1_NORM_WEIGHT = 43,
  DECODER_NORM_WEIGHT = 44,
  DECODER_NORM2_WEIGHT = 45,
  ONED_FINAL_EMBED_0_WEIGHT = 46,
  OUT_FC_WEIGHT = 47,
  LPE_TOKEN = 48,
  WEIGHT_COUNT = 49
};


// v11.1+: wide-port packing rule
static_assert((EXP_LEN_W_WORDS % W_LANES) == 0, "EXP_LEN_W_WORDS must be multiple of W_LANES");

static const TensorMeta kWeightMeta[WEIGHT_COUNT] = {
  /* BCH_H_BITPACK */ { 0u, 24u, 2u, 12u, 63u, 0u, 0u, 1u },
  /* SRC_EMBED */ { 24u, 1800u, 2u, 75u, 24u, 0u, 0u, 0u },
  /* SRC_MASK */ { 1824u, 176u, 2u, 75u, 75u, 0u, 0u, 1u },
  /* QUANT_SX_8 */ { 2000u, 8u, 1u, 8u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_WEIGHT */ { 2008u, 1024u, 2u, 32u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_DELTA */ { 3032u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_S_W */ { 3040u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_WEIGHT */ { 3048u, 1024u, 2u, 32u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_DELTA */ { 4072u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_S_W */ { 4080u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_WEIGHT */ { 4088u, 1024u, 2u, 32u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_DELTA */ { 5112u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_S_W */ { 5120u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_WEIGHT */ { 5128u, 1024u, 2u, 32u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_DELTA */ { 6152u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_S_W */ { 6160u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_1_WEIGHT */ { 6168u, 4096u, 2u, 128u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_1_DELTA */ { 10264u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_1_S_W */ { 10272u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_2_WEIGHT */ { 10280u, 4096u, 2u, 32u, 128u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_2_DELTA */ { 14376u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_2_S_W */ { 14384u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SUBLAYER_0_NORM_WEIGHT */ { 14392u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SUBLAYER_1_NORM_WEIGHT */ { 14424u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_WEIGHT */ { 14456u, 1024u, 2u, 32u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_DELTA */ { 15480u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_S_W */ { 15488u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_WEIGHT */ { 15496u, 1024u, 2u, 32u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_DELTA */ { 16520u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_S_W */ { 16528u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_WEIGHT */ { 16536u, 1024u, 2u, 32u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_DELTA */ { 17560u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_S_W */ { 17568u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_WEIGHT */ { 17576u, 1024u, 2u, 32u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_DELTA */ { 18600u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_S_W */ { 18608u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_1_WEIGHT */ { 18616u, 4096u, 2u, 128u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_1_DELTA */ { 22712u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_1_S_W */ { 22720u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_2_WEIGHT */ { 22728u, 4096u, 2u, 32u, 128u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_2_DELTA */ { 26824u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_2_S_W */ { 26832u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SUBLAYER_0_NORM_WEIGHT */ { 26840u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SUBLAYER_1_NORM_WEIGHT */ { 26872u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_NORM_WEIGHT */ { 26904u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_NORM2_WEIGHT */ { 26936u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* ONED_FINAL_EMBED_0_WEIGHT */ { 26968u, 32u, 2u, 1u, 32u, 0u, 0u, 0u },
  /* OUT_FC_WEIGHT */ { 27000u, 4728u, 2u, 63u, 75u, 0u, 0u, 0u },
  /* LPE_TOKEN */ { 31728u, 600u, 2u, 75u, 8u, 0u, 0u, 0u },
};

// [legacy] Word address in SRAM for a WEIGHT tensor (separated flow)
static inline uint32_t weight_addr_word(WeightId id) {
  return sram_map::BASE_W_W + kWeightMeta[(uint32_t)id].offset_w;
}

// ----------------------------
// [legacy] BIAS stream (separated LOAD_BIAS)
// ----------------------------
enum BiasId : uint32_t {
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_BIAS = 0,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_BIAS = 1,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_BIAS = 2,
  DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_BIAS = 3,
  DECODER_LAYERS_0_FEED_FORWARD_W_1_BIAS = 4,
  DECODER_LAYERS_0_FEED_FORWARD_W_2_BIAS = 5,
  DECODER_LAYERS_0_SUBLAYER_0_NORM_BIAS = 6,
  DECODER_LAYERS_0_SUBLAYER_1_NORM_BIAS = 7,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_BIAS = 8,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_BIAS = 9,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_BIAS = 10,
  DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_BIAS = 11,
  DECODER_LAYERS_1_FEED_FORWARD_W_1_BIAS = 12,
  DECODER_LAYERS_1_FEED_FORWARD_W_2_BIAS = 13,
  DECODER_LAYERS_1_SUBLAYER_0_NORM_BIAS = 14,
  DECODER_LAYERS_1_SUBLAYER_1_NORM_BIAS = 15,
  DECODER_NORM_BIAS = 16,
  DECODER_NORM2_BIAS = 17,
  ONED_FINAL_EMBED_0_BIAS = 18,
  OUT_FC_BIAS = 19,
  BIAS_COUNT = 20
};

static_assert((EXP_LEN_BIAS_WORDS % W_LANES) == 0, "EXP_LEN_BIAS_WORDS must be multiple of W_LANES");

static const TensorMeta kBiasMeta[BIAS_COUNT] = {
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_BIAS */ { 0u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_BIAS */ { 32u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_BIAS */ { 64u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_BIAS */ { 96u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_1_BIAS */ { 128u, 128u, 1u, 128u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_2_BIAS */ { 256u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SUBLAYER_0_NORM_BIAS */ { 288u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SUBLAYER_1_NORM_BIAS */ { 320u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_BIAS */ { 352u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_BIAS */ { 384u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_BIAS */ { 416u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_BIAS */ { 448u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_1_BIAS */ { 480u, 128u, 1u, 128u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_2_BIAS */ { 608u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SUBLAYER_0_NORM_BIAS */ { 640u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SUBLAYER_1_NORM_BIAS */ { 672u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_NORM_BIAS */ { 704u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* DECODER_NORM2_BIAS */ { 736u, 32u, 1u, 32u, 0u, 0u, 0u, 0u },
  /* ONED_FINAL_EMBED_0_BIAS */ { 768u, 8u, 1u, 1u, 0u, 0u, 0u, 0u },
  /* OUT_FC_BIAS */ { 776u, 64u, 1u, 63u, 0u, 0u, 0u, 0u },
};

// ----------------------------
// PARAM unified stream (v11.4 main path)
// ----------------------------
// PARAM = bias + weights + bitpack, streamed in this exact order:
//   1) All BIAS entries (same order as legacy LOAD_BIAS)
//   2) All WEIGHT entries (same order as legacy LOAD_W)
//
// Offsets in kParamMeta are relative to param_base_word (SET_W_BASE).
static const uint32_t PARAM_COUNT = (BIAS_COUNT + WEIGHT_COUNT);
static const uint32_t EXP_LEN_PARAM_WORDS = (EXP_LEN_BIAS_WORDS + EXP_LEN_W_WORDS);
static_assert((EXP_LEN_PARAM_WORDS % W_LANES) == 0, "EXP_LEN_PARAM_WORDS must be multiple of W_LANES");

static const ParamMeta kParamMeta[PARAM_COUNT] = {
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_BIAS */ { 0u, 0u, 0u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_BIAS */ { 1u, 0u, 32u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_BIAS */ { 2u, 0u, 64u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_BIAS */ { 3u, 0u, 96u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_1_BIAS */ { 4u, 0u, 128u, 128u, 0u, 1u, 128u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_2_BIAS */ { 5u, 0u, 256u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SUBLAYER_0_NORM_BIAS */ { 6u, 0u, 288u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SUBLAYER_1_NORM_BIAS */ { 7u, 0u, 320u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_BIAS */ { 8u, 0u, 352u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_BIAS */ { 9u, 0u, 384u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_BIAS */ { 10u, 0u, 416u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_BIAS */ { 11u, 0u, 448u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_1_BIAS */ { 12u, 0u, 480u, 128u, 0u, 1u, 128u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_2_BIAS */ { 13u, 0u, 608u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SUBLAYER_0_NORM_BIAS */ { 14u, 0u, 640u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SUBLAYER_1_NORM_BIAS */ { 15u, 0u, 672u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_NORM_BIAS */ { 16u, 0u, 704u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_NORM2_BIAS */ { 17u, 0u, 736u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* ONED_FINAL_EMBED_0_BIAS */ { 18u, 0u, 768u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* OUT_FC_BIAS */ { 19u, 0u, 776u, 64u, 0u, 1u, 63u, 0u, 0u, 0u },
  /* BCH_H_BITPACK */ { 20u, 1u, 840u, 24u, 20u, 2u, 12u, 63u, 0u, 0u },
  /* SRC_EMBED */ { 21u, 0u, 864u, 1800u, 0u, 2u, 75u, 24u, 0u, 0u },
  /* SRC_MASK */ { 22u, 1u, 2664u, 176u, 25u, 2u, 75u, 75u, 0u, 0u },
  /* QUANT_SX_8 */ { 23u, 0u, 2840u, 8u, 0u, 1u, 8u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_WEIGHT */ { 24u, 0u, 2848u, 1024u, 0u, 2u, 32u, 32u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_DELTA */ { 25u, 0u, 3872u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_S_W */ { 26u, 0u, 3880u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_WEIGHT */ { 27u, 0u, 3888u, 1024u, 0u, 2u, 32u, 32u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_DELTA */ { 28u, 0u, 4912u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_S_W */ { 29u, 0u, 4920u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_WEIGHT */ { 30u, 0u, 4928u, 1024u, 0u, 2u, 32u, 32u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_DELTA */ { 31u, 0u, 5952u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_S_W */ { 32u, 0u, 5960u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_WEIGHT */ { 33u, 0u, 5968u, 1024u, 0u, 2u, 32u, 32u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_DELTA */ { 34u, 0u, 6992u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_S_W */ { 35u, 0u, 7000u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_1_WEIGHT */ { 36u, 0u, 7008u, 4096u, 0u, 2u, 128u, 32u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_1_DELTA */ { 37u, 0u, 11104u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_1_S_W */ { 38u, 0u, 11112u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_2_WEIGHT */ { 39u, 0u, 11120u, 4096u, 0u, 2u, 32u, 128u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_2_DELTA */ { 40u, 0u, 15216u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_FEED_FORWARD_W_2_S_W */ { 41u, 0u, 15224u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SUBLAYER_0_NORM_WEIGHT */ { 42u, 0u, 15232u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_0_SUBLAYER_1_NORM_WEIGHT */ { 43u, 0u, 15264u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_WEIGHT */ { 44u, 0u, 15296u, 1024u, 0u, 2u, 32u, 32u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_DELTA */ { 45u, 0u, 16320u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_S_W */ { 46u, 0u, 16328u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_WEIGHT */ { 47u, 0u, 16336u, 1024u, 0u, 2u, 32u, 32u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_DELTA */ { 48u, 0u, 17360u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_S_W */ { 49u, 0u, 17368u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_WEIGHT */ { 50u, 0u, 17376u, 1024u, 0u, 2u, 32u, 32u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_DELTA */ { 51u, 0u, 18400u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_S_W */ { 52u, 0u, 18408u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_WEIGHT */ { 53u, 0u, 18416u, 1024u, 0u, 2u, 32u, 32u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_DELTA */ { 54u, 0u, 19440u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_S_W */ { 55u, 0u, 19448u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_1_WEIGHT */ { 56u, 0u, 19456u, 4096u, 0u, 2u, 128u, 32u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_1_DELTA */ { 57u, 0u, 23552u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_1_S_W */ { 58u, 0u, 23560u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_2_WEIGHT */ { 59u, 0u, 23568u, 4096u, 0u, 2u, 32u, 128u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_2_DELTA */ { 60u, 0u, 27664u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_FEED_FORWARD_W_2_S_W */ { 61u, 0u, 27672u, 8u, 0u, 1u, 1u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SUBLAYER_0_NORM_WEIGHT */ { 62u, 0u, 27680u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_LAYERS_1_SUBLAYER_1_NORM_WEIGHT */ { 63u, 0u, 27712u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_NORM_WEIGHT */ { 64u, 0u, 27744u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* DECODER_NORM2_WEIGHT */ { 65u, 0u, 27776u, 32u, 0u, 1u, 32u, 0u, 0u, 0u },
  /* ONED_FINAL_EMBED_0_WEIGHT */ { 66u, 0u, 27808u, 32u, 0u, 2u, 1u, 32u, 0u, 0u },
  /* OUT_FC_WEIGHT */ { 67u, 0u, 27840u, 4728u, 0u, 2u, 63u, 75u, 0u, 0u },
  /* LPE_TOKEN */ { 68u, 0u, 32568u, 600u, 0u, 2u, 75u, 8u, 0u, 0u },
};


// [legacy] Word address in SRAM for a BIAS tensor (separated flow)
static inline uint32_t bias_addr_word(BiasId id) {
  return sram_map::BASE_BIAS_W + kBiasMeta[(uint32_t)id].offset_w;
}

// ----------------------------
// TB-only helpers (string keys)
// ----------------------------
#ifndef __SYNTHESIS__
static const char* const kParamKey[PARAM_COUNT] = {
  "decoder.layers.0.self_attn.linears.0.bias",
  "decoder.layers.0.self_attn.linears.1.bias",
  "decoder.layers.0.self_attn.linears.2.bias",
  "decoder.layers.0.self_attn.linears.3.bias",
  "decoder.layers.0.feed_forward.w_1.bias",
  "decoder.layers.0.feed_forward.w_2.bias",
  "decoder.layers.0.sublayer.0.norm.bias",
  "decoder.layers.0.sublayer.1.norm.bias",
  "decoder.layers.1.self_attn.linears.0.bias",
  "decoder.layers.1.self_attn.linears.1.bias",
  "decoder.layers.1.self_attn.linears.2.bias",
  "decoder.layers.1.self_attn.linears.3.bias",
  "decoder.layers.1.feed_forward.w_1.bias",
  "decoder.layers.1.feed_forward.w_2.bias",
  "decoder.layers.1.sublayer.0.norm.bias",
  "decoder.layers.1.sublayer.1.norm.bias",
  "decoder.norm.bias",
  "decoder.norm2.bias",
  "oned_final_embed.0.bias",
  "out_fc.bias",
  "BCH_N63_K51_H",
  "src_embed",
  "src_mask",
  "quant_sx_8",
  "decoder.layers.0.self_attn.linears.0.weight",
  "decoder.layers.0.self_attn.linears.0.delta",
  "decoder.layers.0.self_attn.linears.0.s_w",
  "decoder.layers.0.self_attn.linears.1.weight",
  "decoder.layers.0.self_attn.linears.1.delta",
  "decoder.layers.0.self_attn.linears.1.s_w",
  "decoder.layers.0.self_attn.linears.2.weight",
  "decoder.layers.0.self_attn.linears.2.delta",
  "decoder.layers.0.self_attn.linears.2.s_w",
  "decoder.layers.0.self_attn.linears.3.weight",
  "decoder.layers.0.self_attn.linears.3.delta",
  "decoder.layers.0.self_attn.linears.3.s_w",
  "decoder.layers.0.feed_forward.w_1.weight",
  "decoder.layers.0.feed_forward.w_1.delta",
  "decoder.layers.0.feed_forward.w_1.s_w",
  "decoder.layers.0.feed_forward.w_2.weight",
  "decoder.layers.0.feed_forward.w_2.delta",
  "decoder.layers.0.feed_forward.w_2.s_w",
  "decoder.layers.0.sublayer.0.norm.weight",
  "decoder.layers.0.sublayer.1.norm.weight",
  "decoder.layers.1.self_attn.linears.0.weight",
  "decoder.layers.1.self_attn.linears.0.delta",
  "decoder.layers.1.self_attn.linears.0.s_w",
  "decoder.layers.1.self_attn.linears.1.weight",
  "decoder.layers.1.self_attn.linears.1.delta",
  "decoder.layers.1.self_attn.linears.1.s_w",
  "decoder.layers.1.self_attn.linears.2.weight",
  "decoder.layers.1.self_attn.linears.2.delta",
  "decoder.layers.1.self_attn.linears.2.s_w",
  "decoder.layers.1.self_attn.linears.3.weight",
  "decoder.layers.1.self_attn.linears.3.delta",
  "decoder.layers.1.self_attn.linears.3.s_w",
  "decoder.layers.1.feed_forward.w_1.weight",
  "decoder.layers.1.feed_forward.w_1.delta",
  "decoder.layers.1.feed_forward.w_1.s_w",
  "decoder.layers.1.feed_forward.w_2.weight",
  "decoder.layers.1.feed_forward.w_2.delta",
  "decoder.layers.1.feed_forward.w_2.s_w",
  "decoder.layers.1.sublayer.0.norm.weight",
  "decoder.layers.1.sublayer.1.norm.weight",
  "decoder.norm.weight",
  "decoder.norm2.weight",
  "oned_final_embed.0.weight",
  "out_fc.weight",
  "lpe_token",
};

static const char* const kWeightKey[WEIGHT_COUNT] = {
  "BCH_N63_K51_H",
  "src_embed",
  "src_mask",
  "quant_sx_8",
  "decoder.layers.0.self_attn.linears.0.weight",
  "decoder.layers.0.self_attn.linears.0.delta",
  "decoder.layers.0.self_attn.linears.0.s_w",
  "decoder.layers.0.self_attn.linears.1.weight",
  "decoder.layers.0.self_attn.linears.1.delta",
  "decoder.layers.0.self_attn.linears.1.s_w",
  "decoder.layers.0.self_attn.linears.2.weight",
  "decoder.layers.0.self_attn.linears.2.delta",
  "decoder.layers.0.self_attn.linears.2.s_w",
  "decoder.layers.0.self_attn.linears.3.weight",
  "decoder.layers.0.self_attn.linears.3.delta",
  "decoder.layers.0.self_attn.linears.3.s_w",
  "decoder.layers.0.feed_forward.w_1.weight",
  "decoder.layers.0.feed_forward.w_1.delta",
  "decoder.layers.0.feed_forward.w_1.s_w",
  "decoder.layers.0.feed_forward.w_2.weight",
  "decoder.layers.0.feed_forward.w_2.delta",
  "decoder.layers.0.feed_forward.w_2.s_w",
  "decoder.layers.0.sublayer.0.norm.weight",
  "decoder.layers.0.sublayer.1.norm.weight",
  "decoder.layers.1.self_attn.linears.0.weight",
  "decoder.layers.1.self_attn.linears.0.delta",
  "decoder.layers.1.self_attn.linears.0.s_w",
  "decoder.layers.1.self_attn.linears.1.weight",
  "decoder.layers.1.self_attn.linears.1.delta",
  "decoder.layers.1.self_attn.linears.1.s_w",
  "decoder.layers.1.self_attn.linears.2.weight",
  "decoder.layers.1.self_attn.linears.2.delta",
  "decoder.layers.1.self_attn.linears.2.s_w",
  "decoder.layers.1.self_attn.linears.3.weight",
  "decoder.layers.1.self_attn.linears.3.delta",
  "decoder.layers.1.self_attn.linears.3.s_w",
  "decoder.layers.1.feed_forward.w_1.weight",
  "decoder.layers.1.feed_forward.w_1.delta",
  "decoder.layers.1.feed_forward.w_1.s_w",
  "decoder.layers.1.feed_forward.w_2.weight",
  "decoder.layers.1.feed_forward.w_2.delta",
  "decoder.layers.1.feed_forward.w_2.s_w",
  "decoder.layers.1.sublayer.0.norm.weight",
  "decoder.layers.1.sublayer.1.norm.weight",
  "decoder.norm.weight",
  "decoder.norm2.weight",
  "oned_final_embed.0.weight",
  "out_fc.weight",
  "lpe_token",
};
static const char* const kBiasKey[BIAS_COUNT] = {
  "decoder.layers.0.self_attn.linears.0.bias",
  "decoder.layers.0.self_attn.linears.1.bias",
  "decoder.layers.0.self_attn.linears.2.bias",
  "decoder.layers.0.self_attn.linears.3.bias",
  "decoder.layers.0.feed_forward.w_1.bias",
  "decoder.layers.0.feed_forward.w_2.bias",
  "decoder.layers.0.sublayer.0.norm.bias",
  "decoder.layers.0.sublayer.1.norm.bias",
  "decoder.layers.1.self_attn.linears.0.bias",
  "decoder.layers.1.self_attn.linears.1.bias",
  "decoder.layers.1.self_attn.linears.2.bias",
  "decoder.layers.1.self_attn.linears.3.bias",
  "decoder.layers.1.feed_forward.w_1.bias",
  "decoder.layers.1.feed_forward.w_2.bias",
  "decoder.layers.1.sublayer.0.norm.bias",
  "decoder.layers.1.sublayer.1.norm.bias",
  "decoder.norm.bias",
  "decoder.norm2.bias",
  "oned_final_embed.0.bias",
  "out_fc.bias",
};
#endif

// ----------------------------
// BITPACK packing rule (important for TB)
// ----------------------------
// For BITPACK tensors (dtype=1), pack bits in row-major order:
//
//   for (r = 0..d0-1)        // 列(Row)
//     for (c = 0..d1-1)      // 行(Colume)
//        bit_index = r*d1 + c
//        word_index = bit_index >> 5
//        bit_in_word = bit_index & 31
//        set/get that bit in u32[word_index]
//
// BCH_H_BITPACK is [CODE_C, CODE_N] with this packing.
// src_mask is [N_NODES, N_NODES] with this packing.
