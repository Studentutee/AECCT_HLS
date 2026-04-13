#pragma once

#ifndef __SYNTHESIS__

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

#include "AecctProtocol.h"
#include "AecctTypes.h"
#include "AecctUtil.h"
#include "PreprocDescBringup.h"
#include "gen/ModelDesc.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"
#include "input_y_step0.h"
#include "weights.h"
#include "weights_streamer.h"

namespace fp16_branch_tb {

static inline uint16_t fp16_lane_from_double_tb(const double x) {
  return (uint16_t)aecct::fp16_lane_from_fp32_bits(aecct::fp32_bits_from_double(x)).to_uint();
}

static inline void pack_bits_to_word16(const ac_int<1, false>* bits,
                                       const uint32_t num_bits,
                                       std::vector<uint16_t>& out_words16) {
  const uint32_t need_words16 = storage_words_bits(num_bits);
  out_words16.assign(need_words16, 0u);
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
    out_words16[word_i] = word;
  }
}

static inline bool build_bias_words16(const BiasId bid,
                                      std::vector<uint16_t>& out_words16,
                                      uint32_t& out_numel) {
  uint32_t numel = 0u;
  const double* ptr = tb_lookup_bias_fp64(bid, numel);
  if (!ptr) {
    return false;
  }
  out_numel = numel;
  out_words16.assign(numel, 0u);
  for (uint32_t i = 0u; i < numel; ++i) {
    out_words16[i] = fp16_lane_from_double_tb(ptr[i]);
  }
  return true;
}

static inline bool build_weight_words16(const WeightId wid,
                                        std::vector<uint16_t>& out_words16,
                                        uint32_t& out_numel) {
  out_words16.clear();
  out_numel = 0u;
  switch (fp16_branch_weight_storage_kind(wid)) {
    case FP16_BRANCH_STORAGE_FP16: {
      uint32_t numel = 0u;
      const double* ptr = tb_lookup_weight_fp64(wid, numel);
      if (!ptr) {
        return false;
      }
      out_numel = numel;
      out_words16.assign(numel, 0u);
      if (is_quant_linear_inv_sw_weight_slot(wid)) {
        for (uint32_t i = 0u; i < numel; ++i) {
          const double s_w = ptr[i];
          if (s_w == 0.0) {
            return false;
          }
          out_words16[i] = fp16_lane_from_double_tb(1.0 / s_w);
        }
      } else {
        for (uint32_t i = 0u; i < numel; ++i) {
          out_words16[i] = fp16_lane_from_double_tb(ptr[i]);
        }
      }
      return true;
    }
    case FP16_BRANCH_STORAGE_TERNARY_PAYLOAD: {
      uint32_t numel = 0u;
      const double* ptr = tb_lookup_weight_fp64(wid, numel);
      if (!ptr) {
        return false;
      }
      out_numel = numel;
      uint32_t payload_words16 = 0u;
      uint32_t last_valid = 0u;
      out_words16.assign(fp16_branch_weight_storage_words16(wid), 0u);
      if (!tb_pack_ternary_storage_words_from_fp64(ptr,
                                                   numel,
                                                   numel,
                                                   out_words16.data(),
                                                   (uint32_t)out_words16.size(),
                                                   payload_words16,
                                                   last_valid)) {
        return false;
      }
      return (payload_words16 == (uint32_t)out_words16.size());
    }
    case FP16_BRANCH_STORAGE_BITPACK: {
      uint32_t num_bits = 0u;
      const ac_int<1, false>* bits = tb_lookup_weight_bits(wid, num_bits);
      if (!bits) {
        return false;
      }
      out_numel = num_bits;
      pack_bits_to_word16(bits, num_bits, out_words16);
      return ((uint32_t)out_words16.size() == fp16_branch_weight_storage_words16(wid));
    }
    default:
      return false;
  }
}

static inline bool build_param_image_word16(std::vector<uint16_t>& param_image) {
  param_image.assign(sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_WORDS_WORD16, 0u);

  for (uint32_t i = 0u; i < (uint32_t)BIAS_COUNT; ++i) {
    const BiasId bid = (BiasId)i;
    const Fp16BranchStorageDesc desc = fp16_branch_bias_storage_desc(bid);
    std::vector<uint16_t> words;
    uint32_t logical = 0u;
    if (!build_bias_words16(bid, words, logical)) {
      std::fprintf(stderr, "[fp16_common][FAIL] bias build failed bid=%u (%s)\n",
                   (unsigned)i,
                   desc.key);
      return false;
    }
    if ((uint32_t)words.size() != desc.words16 || (desc.offset_words16 + desc.words16) > param_image.size()) {
      std::fprintf(stderr,
                   "[fp16_common][FAIL] bias placement invalid bid=%u off16=%u words16=%u expect=%u total=%u\n",
                   (unsigned)i,
                   (unsigned)desc.offset_words16,
                   (unsigned)words.size(),
                   (unsigned)desc.words16,
                   (unsigned)param_image.size());
      return false;
    }
    for (uint32_t j = 0u; j < desc.words16; ++j) {
      param_image[desc.offset_words16 + j] = words[j];
    }
  }

  for (uint32_t i = 0u; i < (uint32_t)WEIGHT_COUNT; ++i) {
    const WeightId wid = (WeightId)i;
    const Fp16BranchStorageDesc desc = fp16_branch_weight_storage_desc(wid);
    std::vector<uint16_t> words;
    uint32_t logical = 0u;
    if (!build_weight_words16(wid, words, logical)) {
      std::fprintf(stderr, "[fp16_common][FAIL] weight build failed wid=%u (%s)\n",
                   (unsigned)i,
                   desc.key);
      return false;
    }
    if ((uint32_t)words.size() != desc.words16 || (desc.offset_words16 + desc.words16) > param_image.size()) {
      std::fprintf(stderr,
                   "[fp16_common][FAIL] weight placement invalid wid=%u off16=%u words16=%u expect=%u total=%u\n",
                   (unsigned)i,
                   (unsigned)desc.offset_words16,
                   (unsigned)words.size(),
                   (unsigned)desc.words16,
                   (unsigned)param_image.size());
      return false;
    }
    for (uint32_t j = 0u; j < desc.words16; ++j) {
      param_image[desc.offset_words16 + j] = words[j];
    }
  }

  return true;
}

static inline void pack_word16_to_u32_stream(const std::vector<uint16_t>& in_word16,
                                              std::vector<uint32_t>& out_u32) {
  const uint32_t words32 = (uint32_t)((in_word16.size() + 1u) >> 1);
  out_u32.assign(words32, 0u);
  for (uint32_t i = 0u; i < words32; ++i) {
    const uint32_t lo = (uint32_t)in_word16[i * 2u];
    const uint32_t hi = ((i * 2u + 1u) < in_word16.size()) ? (uint32_t)in_word16[i * 2u + 1u] : 0u;
    out_u32[i] = lo | (hi << 16);
  }
}

static inline void unpack_u32_to_word16_stream(const std::vector<uint32_t>& in_u32,
                                                const uint32_t expect_words16,
                                                std::vector<uint16_t>& out_word16) {
  out_word16.assign(expect_words16, 0u);
  uint32_t out_idx = 0u;
  for (uint32_t i = 0u; i < (uint32_t)in_u32.size() && out_idx < expect_words16; ++i) {
    const uint32_t w = in_u32[i];
    out_word16[out_idx++] = (uint16_t)(w & 0xFFFFu);
    if (out_idx < expect_words16) {
      out_word16[out_idx++] = (uint16_t)((w >> 16) & 0xFFFFu);
    }
  }
}

static inline bool compare_word16_vectors_exact(const std::vector<uint16_t>& exp,
                                                const std::vector<uint16_t>& got,
                                                const char* tag,
                                                const uint32_t base_word16) {
  if (exp.size() != got.size()) {
    std::fprintf(stderr,
                 "[fp16_common][FAIL] %s size mismatch exp=%u got=%u\n",
                 tag,
                 (unsigned)exp.size(),
                 (unsigned)got.size());
    return false;
  }
  for (uint32_t i = 0u; i < (uint32_t)exp.size(); ++i) {
    if (exp[i] != got[i]) {
      std::fprintf(stderr,
                   "[fp16_common][FAIL] %s first mismatch addr_word16=%u idx=%u exp=0x%04X got=0x%04X\n",
                   tag,
                   (unsigned)(base_word16 + i),
                   (unsigned)i,
                   (unsigned)exp[i],
                   (unsigned)got[i]);
      return false;
    }
  }
  return true;
}


static inline uint32_t fp16_roundtrip_bits_from_double_tb(const double x) {
  const uint16_t lane = fp16_lane_from_double_tb(x);
  return (uint32_t)aecct::fp32_bits_from_fp16_lane((aecct::u16_t)lane).to_uint();
}

static inline aecct::fp16_t fp16_from_double_tb(const double x) {
  return aecct::fp16_from_bits((aecct::u16_t)fp16_lane_from_double_tb(x));
}

static inline uint32_t f32_to_bits_tb(const float x) {
  uint32_t bits = 0u;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}

static inline aecct::fp16_t ref_node_feature_fp16(const uint32_t sample_idx, const uint32_t token_idx) {
  const size_t sample_base = (size_t)sample_idx * (size_t)CODE_N;
  if (token_idx < (uint32_t)CODE_N) {
    const aecct::fp16_t y = fp16_from_double_tb(trace_input_y_step0_tensor[sample_base + token_idx]);
    return (y < aecct::fp16_zero()) ? (aecct::fp16_zero() - y) : y;
  }
  const uint32_t check_idx = token_idx - (uint32_t)CODE_N;
  uint32_t parity = 0u;
  for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
    const uint32_t flat = check_idx * (uint32_t)CODE_N + v;
    if ((uint32_t)h_H[flat].to_uint() == 0u) {
      continue;
    }
    const aecct::fp16_t y = fp16_from_double_tb(trace_input_y_step0_tensor[sample_base + v]);
    parity ^= (y < aecct::fp16_zero()) ? 1u : 0u;
  }
  return (parity == 0u) ? aecct::fp16_one() : (aecct::fp16_zero() - aecct::fp16_one());
}

static inline uint32_t ref_preproc_x_fp32_bits(const uint32_t sample_idx,
                                               const uint32_t token_idx,
                                               const uint32_t d) {
  if (d < (uint32_t)w_src_embed_shape[1]) {
    const aecct::fp16_t node = ref_node_feature_fp16(sample_idx, token_idx);
    const aecct::fp16_t embed = fp16_from_double_tb(
        w_src_embed[token_idx * (uint32_t)w_src_embed_shape[1] + d]);
    const aecct::fp16_t x = aecct::fp16_t(node * embed);
    return (uint32_t)aecct::fp32_bits_from_fp16_lane(aecct::bits_from_fp16(x)).to_uint();
  }
  if (d < ((uint32_t)w_src_embed_shape[1] + (uint32_t)w_lpe_token_shape[1])) {
    const uint32_t lpe_d = d - (uint32_t)w_src_embed_shape[1];
    const aecct::fp16_t x = fp16_from_double_tb(
        w_lpe_token[token_idx * (uint32_t)w_lpe_token_shape[1] + lpe_d]);
    return (uint32_t)aecct::fp32_bits_from_fp16_lane(aecct::bits_from_fp16(x)).to_uint();
  }
  return 0u;
}

static inline bool build_infer_input_words_u32(const uint32_t sample_idx,
                                               std::vector<uint32_t>& out_u32) {
  if (trace_input_y_step0_tensor_ndim != 2) {
    std::fprintf(stderr, "[fp16_common][FAIL] input trace rank=%d expect=2\n", trace_input_y_step0_tensor_ndim);
    return false;
  }
  if ((uint32_t)trace_input_y_step0_tensor_shape[1] != (uint32_t)CODE_N) {
    std::fprintf(stderr,
                 "[fp16_common][FAIL] input trace cols=%d expect=%u\n",
                 trace_input_y_step0_tensor_shape[1],
                 (unsigned)CODE_N);
    return false;
  }
  if (sample_idx >= (uint32_t)trace_input_y_step0_tensor_shape[0]) {
    std::fprintf(stderr,
                 "[fp16_common][FAIL] sample_idx out of range sample=%u max=%u\n",
                 (unsigned)sample_idx,
                 (unsigned)((uint32_t)trace_input_y_step0_tensor_shape[0] - 1u));
    return false;
  }
  out_u32.assign((uint32_t)aecct::PREPROC_IN_WORDS_EXPECTED, 0u);
  const size_t sample_base = (size_t)sample_idx * (size_t)CODE_N;
  for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
    out_u32[v] = f32_to_bits_tb((float)trace_input_y_step0_tensor[sample_base + v]);
  }
  return true;
}

static inline bool build_preproc_x_ref_word16(const uint32_t sample_idx,
                                              std::vector<uint16_t>& out_words16) {
  out_words16.assign(sram_map::FP16_BASELINE_X_WORK_WORDS_WORD16, 0u);
  const uint32_t d_model = (uint32_t)D_MODEL;
  const uint32_t token_count = (uint32_t)N_NODES;
  if (d_model != ((uint32_t)w_src_embed_shape[1] + (uint32_t)w_lpe_token_shape[1])) {
    std::fprintf(stderr,
                 "[fp16_common][FAIL] d_model mismatch d_model=%u embed+lpe=%u\n",
                 (unsigned)d_model,
                 (unsigned)((uint32_t)w_src_embed_shape[1] + (uint32_t)w_lpe_token_shape[1]));
    return false;
  }
  for (uint32_t token = 0u; token < token_count; ++token) {
    for (uint32_t d = 0u; d < d_model; ++d) {
      const uint32_t elem_idx = token * d_model + d;
      out_words16[elem_idx] = (uint16_t)aecct::fp16_lane_from_fp32_bits((aecct::u32_t)ref_preproc_x_fp32_bits(sample_idx, token, d)).to_uint();
    }
  }
  return true;
}

static inline void seed_u32_words_into_sram(aecct::u32_t* sram,
                                            const uint32_t base_word,
                                            const std::vector<uint32_t>& words_u32) {
  for (uint32_t i = 0u; i < (uint32_t)words_u32.size(); ++i) {
    sram[base_word + i] = (aecct::u32_t)words_u32[i];
  }
}

static inline bool seed_param_image_word16_into_sram(aecct::u32_t* sram,
                                                     std::vector<uint16_t>& param_word16_out,
                                                     std::vector<uint32_t>& param_u32_out) {
  if (!build_param_image_word16(param_word16_out)) {
    return false;
  }
  pack_word16_to_u32_stream(param_word16_out, param_u32_out);
  const uint32_t param_base_w =
      storage_words_to_legacy_words_ceil(sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_BASE_WORD16);
  seed_u32_words_into_sram(sram, param_base_w, param_u32_out);
  return true;
}

static inline void extract_x_work_word16_from_u32_sram(const aecct::u32_t* sram,
                                                       std::vector<uint16_t>& out_words16) {
  const uint32_t x_base_w = (uint32_t)aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT;
  const uint32_t packed_words32 =
      storage_words_to_legacy_words_ceil(sram_map::FP16_BASELINE_X_WORK_WORDS_WORD16);
  std::vector<uint32_t> tmp_u32(packed_words32, 0u);
  for (uint32_t i = 0u; i < packed_words32; ++i) {
    tmp_u32[i] = (uint32_t)sram[x_base_w + i].to_uint();
  }
  unpack_u32_to_word16_stream(tmp_u32, sram_map::FP16_BASELINE_X_WORK_WORDS_WORD16, out_words16);
}

}  // namespace fp16_branch_tb

#endif
