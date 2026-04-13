#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"
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

static bool compare_word16_vectors_exact(const std::vector<uint16_t>& exp,
                                         const std::vector<uint16_t>& got,
                                         const char* tag,
                                         const uint32_t base_word16) {
  if (exp.size() != got.size()) {
    std::fprintf(stderr,
                 "[fp16_readback][FAIL] %s size mismatch exp=%u got=%u\n",
                 tag,
                 (unsigned)exp.size(),
                 (unsigned)got.size());
    return false;
  }
  for (uint32_t i = 0u; i < (uint32_t)exp.size(); ++i) {
    if (exp[i] != got[i]) {
      std::fprintf(stderr,
                   "[fp16_readback][FAIL] %s first mismatch addr_word16=%u idx=%u exp=0x%04X got=0x%04X\n",
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

static bool slice_read_exact(const std::vector<uint16_t>& sram_word16,
                             const uint32_t base_word16,
                             const uint32_t words16,
                             std::vector<uint16_t>& out) {
  if ((base_word16 + words16) > (uint32_t)sram_word16.size()) {
    return false;
  }
  out.assign(sram_word16.begin() + base_word16,
             sram_word16.begin() + base_word16 + words16);
  return true;
}

int main() {
  if (!sram_map::fp16_baseline_default_param_stream_fits_w_region()) {
    std::fprintf(stderr, "[fp16_readback][FAIL] baseline PARAM stream does not fit inside W_REGION\n");
    return 1;
  }

  std::vector<uint16_t> param_image(sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_WORDS_WORD16, 0u);

  for (uint32_t i = 0u; i < (uint32_t)BIAS_COUNT; ++i) {
    const BiasId bid = (BiasId)i;
    const Fp16BranchStorageDesc desc = fp16_branch_bias_storage_desc(bid);
    std::vector<uint16_t> words;
    uint32_t logical = 0u;
    if (!build_fp16_branch_bias_words(bid, words, logical)) {
      std::fprintf(stderr, "[fp16_readback][FAIL] bias build failed bid=%u (%s)\n",
                   (unsigned)i,
                   desc.key);
      return 1;
    }
    if ((uint32_t)words.size() != desc.words16 || (desc.offset_words16 + desc.words16) > param_image.size()) {
      std::fprintf(stderr,
                   "[fp16_readback][FAIL] bias placement invalid bid=%u off16=%u words16=%u expect=%u total=%u\n",
                   (unsigned)i,
                   (unsigned)desc.offset_words16,
                   (unsigned)words.size(),
                   (unsigned)desc.words16,
                   (unsigned)param_image.size());
      return 1;
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
    if (!build_fp16_branch_weight_words(wid, words, logical)) {
      std::fprintf(stderr, "[fp16_readback][FAIL] weight build failed wid=%u (%s)\n",
                   (unsigned)i,
                   desc.key);
      return 1;
    }
    if ((uint32_t)words.size() != desc.words16 || (desc.offset_words16 + desc.words16) > param_image.size()) {
      std::fprintf(stderr,
                   "[fp16_readback][FAIL] weight placement invalid wid=%u off16=%u words16=%u expect=%u total=%u\n",
                   (unsigned)i,
                   (unsigned)desc.offset_words16,
                   (unsigned)words.size(),
                   (unsigned)desc.words16,
                   (unsigned)param_image.size());
      return 1;
    }
    for (uint32_t j = 0u; j < desc.words16; ++j) {
      param_image[desc.offset_words16 + j] = words[j];
    }
  }

  std::vector<uint16_t> sram_word16(sram_map::FP16_BASELINE_STORAGE_WORDS_MIN_REQUIRED, 0u);
  const uint32_t param_base_word16 = sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_BASE_WORD16;
  for (uint32_t i = 0u; i < (uint32_t)param_image.size(); ++i) {
    sram_word16[param_base_word16 + i] = param_image[i];
  }

  std::vector<uint16_t> got_param;
  if (!slice_read_exact(sram_word16, param_base_word16, (uint32_t)param_image.size(), got_param)) {
    std::fprintf(stderr, "[fp16_readback][FAIL] param slice read failed\n");
    return 1;
  }
  if (!compare_word16_vectors_exact(param_image, got_param, "PARAM_STREAM_DEFAULT_word16_exact", param_base_word16)) {
    return 1;
  }

  std::vector<uint16_t> expected_w_region(sram_map::FP16_BASELINE_W_REGION_WORDS_WORD16, 0u);
  for (uint32_t i = 0u; i < (uint32_t)param_image.size(); ++i) {
    expected_w_region[i] = param_image[i];
  }
  std::vector<uint16_t> got_w_region;
  if (!slice_read_exact(sram_word16,
                        sram_map::FP16_BASELINE_W_REGION_BASE_WORD16,
                        sram_map::FP16_BASELINE_W_REGION_WORDS_WORD16,
                        got_w_region)) {
    std::fprintf(stderr, "[fp16_readback][FAIL] W_REGION slice read failed\n");
    return 1;
  }
  if (!compare_word16_vectors_exact(expected_w_region,
                                    got_w_region,
                                    "W_REGION_word16_exact_with_padding",
                                    sram_map::FP16_BASELINE_W_REGION_BASE_WORD16)) {
    return 1;
  }

  for (uint32_t i = 0u; i < (uint32_t)BIAS_COUNT; ++i) {
    Fp16BranchStorageDesc desc;
    if (!fp16_branch_param_storage_desc(i, desc)) {
      std::fprintf(stderr, "[fp16_readback][FAIL] bias desc lookup failed pid=%u\n", (unsigned)i);
      return 1;
    }
    std::vector<uint16_t> exp;
    std::vector<uint16_t> got;
    uint32_t logical = 0u;
    if (!build_fp16_branch_bias_words((BiasId)i, exp, logical)) {
      std::fprintf(stderr, "[fp16_readback][FAIL] bias rebuild failed pid=%u\n", (unsigned)i);
      return 1;
    }
    if (!slice_read_exact(sram_word16, param_base_word16 + desc.offset_words16, desc.words16, got)) {
      std::fprintf(stderr, "[fp16_readback][FAIL] bias readback slice failed pid=%u\n", (unsigned)i);
      return 1;
    }
    if (!compare_word16_vectors_exact(exp, got, desc.key, param_base_word16 + desc.offset_words16)) {
      return 1;
    }
  }

  for (uint32_t i = 0u; i < (uint32_t)WEIGHT_COUNT; ++i) {
    const uint32_t pid = (uint32_t)BIAS_COUNT + i;
    Fp16BranchStorageDesc desc;
    if (!fp16_branch_param_storage_desc(pid, desc)) {
      std::fprintf(stderr, "[fp16_readback][FAIL] weight desc lookup failed pid=%u\n", (unsigned)pid);
      return 1;
    }
    std::vector<uint16_t> exp;
    std::vector<uint16_t> got;
    uint32_t logical = 0u;
    if (!build_fp16_branch_weight_words((WeightId)i, exp, logical)) {
      std::fprintf(stderr, "[fp16_readback][FAIL] weight rebuild failed pid=%u\n", (unsigned)pid);
      return 1;
    }
    if (!slice_read_exact(sram_word16, param_base_word16 + desc.offset_words16, desc.words16, got)) {
      std::fprintf(stderr, "[fp16_readback][FAIL] weight readback slice failed pid=%u\n", (unsigned)pid);
      return 1;
    }
    if (!compare_word16_vectors_exact(exp, got, desc.key, param_base_word16 + desc.offset_words16)) {
      return 1;
    }
  }

  std::printf("[fp16_readback][SUMMARY] bridged_min_word16=%u baseline_min_word16=%u bridged_total_word16=%u baseline_total_word16=%u\n",
              (unsigned)sram_map::SRAM_STORAGE_WORDS_MIN_REQUIRED,
              (unsigned)sram_map::FP16_BASELINE_STORAGE_WORDS_MIN_REQUIRED,
              (unsigned)sram_map::SRAM_STORAGE_WORDS_TOTAL,
              (unsigned)sram_map::FP16_BASELINE_STORAGE_WORDS_TOTAL);
  std::printf("[fp16_readback][SUMMARY] param_words16=%u w_region_words16=%u bias_words16=%u weight_words16=%u\n",
              (unsigned)sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_WORDS_WORD16,
              (unsigned)sram_map::FP16_BASELINE_W_REGION_WORDS_WORD16,
              (unsigned)sram_map::FP16_BASELINE_BIAS_PAYLOAD_WORDS_WORD16,
              (unsigned)sram_map::FP16_BASELINE_WEIGHT_PAYLOAD_WORDS_WORD16);
  std::printf("PASS: tb_fp16_branch_readback_word16_smoke\n");
  return 0;
}

#else
int main() { return 0; }
#endif
