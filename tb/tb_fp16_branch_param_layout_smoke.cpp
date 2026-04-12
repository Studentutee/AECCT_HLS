#include <cstdint>
#include <cstdio>
#include <vector>

#include "gen/SramMap.h"
#include "weights_streamer.h"

#ifndef __SYNTHESIS__

static uint16_t fp16_lane_from_double_tb(const double x) {
  return (uint16_t)aecct::fp16_lane_from_fp32_bits(aecct::fp32_bits_from_double(x)).to_uint();
}

static void pack_bits_to_word16(const ac_int<1,false>* bits,
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

static bool build_fp16_branch_bias_words(const BiasId bid,
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

static bool build_fp16_branch_weight_words(const WeightId wid,
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
      const ac_int<1,false>* bits = tb_lookup_weight_bits(wid, num_bits);
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

int main() {
  std::vector<uint16_t> param_image(fp16_branch_total_param_words16(), 0u);

  for (uint32_t i = 0u; i < (uint32_t)BIAS_COUNT; ++i) {
    const BiasId bid = (BiasId)i;
    std::vector<uint16_t> words;
    uint32_t logical = 0u;
    if (!build_fp16_branch_bias_words(bid, words, logical)) {
      std::fprintf(stderr, "[fp16_param_layout][FAIL] bias build failed bid=%u\n", (unsigned)i);
      return 1;
    }
    const uint32_t off = fp16_branch_bias_offset_words16(bid);
    const uint32_t expect = fp16_branch_bias_storage_words16(bid);
    if ((uint32_t)words.size() != expect || (off + expect) > param_image.size()) {
      std::fprintf(stderr, "[fp16_param_layout][FAIL] bias range invalid bid=%u off=%u words=%u expect=%u total=%u\n",
                   (unsigned)i, (unsigned)off, (unsigned)words.size(), (unsigned)expect, (unsigned)param_image.size());
      return 1;
    }
    for (uint32_t j = 0u; j < expect; ++j) {
      param_image[off + j] = words[j];
    }
  }

  for (uint32_t i = 0u; i < (uint32_t)WEIGHT_COUNT; ++i) {
    const WeightId wid = (WeightId)i;
    std::vector<uint16_t> words;
    uint32_t logical = 0u;
    if (!build_fp16_branch_weight_words(wid, words, logical)) {
      std::fprintf(stderr, "[fp16_param_layout][FAIL] weight build failed wid=%u (%s)\n", (unsigned)i, kWeightKey[i]);
      return 1;
    }
    const uint32_t off = fp16_branch_weight_offset_words16(wid);
    const uint32_t expect = fp16_branch_weight_storage_words16(wid);
    if ((uint32_t)words.size() != expect || (off + expect) > param_image.size()) {
      std::fprintf(stderr, "[fp16_param_layout][FAIL] weight range invalid wid=%u (%s) off=%u words=%u expect=%u total=%u\n",
                   (unsigned)i, kWeightKey[i], (unsigned)off, (unsigned)words.size(), (unsigned)expect, (unsigned)param_image.size());
      return 1;
    }
    for (uint32_t j = 0u; j < expect; ++j) {
      param_image[off + j] = words[j];
    }
  }

  if ((uint32_t)param_image.size() != sram_map::PARAM_STREAM_DEFAULT_WORDS_WORD16) {
    std::fprintf(stderr, "[fp16_param_layout][FAIL] param image words16 mismatch got=%u expect=%u\n",
                 (unsigned)param_image.size(),
                 (unsigned)sram_map::PARAM_STREAM_DEFAULT_WORDS_WORD16);
    return 1;
  }

  if (storage_words_to_legacy_words_ceil((uint32_t)param_image.size()) != sram_map::PARAM_STREAM_DEFAULT_WORDS) {
    std::fprintf(stderr, "[fp16_param_layout][FAIL] legacy bridge words mismatch got=%u expect=%u\n",
                 (unsigned)storage_words_to_legacy_words_ceil((uint32_t)param_image.size()),
                 (unsigned)sram_map::PARAM_STREAM_DEFAULT_WORDS);
    return 1;
  }

  std::printf("[fp16_param_layout][SUMMARY] param_words16=%u legacy_words=%u w_region_payload_words16=%u\n",
              (unsigned)param_image.size(),
              (unsigned)sram_map::PARAM_STREAM_DEFAULT_WORDS,
              (unsigned)sram_map::W_REGION_PAYLOAD_WORDS_WORD16);
  std::printf("[fp16_param_layout][SUMMARY] bias_words16=%u weight_words16=%u\n",
              (unsigned)sram_map::SIZE_BIAS_PAYLOAD_WORD16,
              (unsigned)sram_map::SIZE_WEIGHT_PAYLOAD_WORD16);
  std::printf("[fp16_param_layout][CHECK] out_fc.offset16=%u words16=%u\n",
              (unsigned)fp16_branch_weight_offset_words16(OUT_FC_WEIGHT),
              (unsigned)fp16_branch_weight_storage_words16(OUT_FC_WEIGHT));
  std::printf("PASS: tb_fp16_branch_param_layout_smoke\n");
  return 0;
}

#else
int main() { return 0; }
#endif
