#include <cstdint>
#include <cstdio>
#include <vector>

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
            std::fprintf(stderr,
                         "[fp16_branch][FAIL] inv_s_w zero wid=%u idx=%u\n",
                         (unsigned)wid,
                         (unsigned)i);
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
      if (payload_words16 != (uint32_t)out_words16.size()) {
        std::fprintf(stderr,
                     "[fp16_branch][FAIL] ternary payload size mismatch wid=%u got=%u expect=%u\n",
                     (unsigned)wid,
                     (unsigned)payload_words16,
                     (unsigned)out_words16.size());
        return false;
      }
      (void)last_valid;
      return true;
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
  uint32_t total_words16 = 0u;
  uint32_t total_fp16_words16 = 0u;
  uint32_t total_ternary_words16 = 0u;
  uint32_t total_bitpack_words16 = 0u;

  for (uint32_t i = 0u; i < (uint32_t)BIAS_COUNT; ++i) {
    std::vector<uint16_t> words;
    uint32_t numel = 0u;
    const BiasId bid = (BiasId)i;
    if (!build_fp16_branch_bias_words(bid, words, numel)) {
      std::fprintf(stderr, "[fp16_branch][FAIL] bias build failed bid=%u\n", (unsigned)i);
      return 1;
    }
    const uint32_t expect_words16 = fp16_branch_bias_storage_words16(bid);
    if ((uint32_t)words.size() != expect_words16 || expect_words16 != numel) {
      std::fprintf(stderr,
                   "[fp16_branch][FAIL] bias size mismatch bid=%u words16=%u expect=%u numel=%u\n",
                   (unsigned)i,
                   (unsigned)words.size(),
                   (unsigned)expect_words16,
                   (unsigned)numel);
      return 1;
    }
    total_words16 += expect_words16;
    total_fp16_words16 += expect_words16;
  }

  for (uint32_t i = 0u; i < (uint32_t)WEIGHT_COUNT; ++i) {
    std::vector<uint16_t> words;
    uint32_t logical_elems = 0u;
    const WeightId wid = (WeightId)i;
    if (!build_fp16_branch_weight_words(wid, words, logical_elems)) {
      std::fprintf(stderr, "[fp16_branch][FAIL] weight build failed wid=%u (%s)\n", (unsigned)i, kWeightKey[i]);
      return 1;
    }
    const uint32_t expect_words16 = fp16_branch_weight_storage_words16(wid);
    if ((uint32_t)words.size() != expect_words16) {
      std::fprintf(stderr,
                   "[fp16_branch][FAIL] weight size mismatch wid=%u (%s) words16=%u expect=%u\n",
                   (unsigned)i,
                   kWeightKey[i],
                   (unsigned)words.size(),
                   (unsigned)expect_words16);
      return 1;
    }

    const Fp16BranchStorageKind kind = fp16_branch_weight_storage_kind(wid);
    switch (kind) {
      case FP16_BRANCH_STORAGE_FP16:
        total_fp16_words16 += expect_words16;
        if (is_quant_linear_weight_slot(wid)) {
          std::fprintf(stderr, "[fp16_branch][FAIL] quant-linear payload misclassified as fp16 wid=%u\n", (unsigned)i);
          return 1;
        }
        break;
      case FP16_BRANCH_STORAGE_TERNARY_PAYLOAD:
        total_ternary_words16 += expect_words16;
        if (!is_quant_linear_weight_slot(wid)) {
          std::fprintf(stderr, "[fp16_branch][FAIL] non-quant-linear weight misclassified as ternary wid=%u\n", (unsigned)i);
          return 1;
        }
        break;
      case FP16_BRANCH_STORAGE_BITPACK:
        total_bitpack_words16 += expect_words16;
        if (!(wid == BCH_H_BITPACK || wid == SRC_MASK)) {
          std::fprintf(stderr, "[fp16_branch][FAIL] unexpected bitpack wid=%u\n", (unsigned)i);
          return 1;
        }
        break;
      default:
        return 1;
    }

    total_words16 += expect_words16;

    if (i < 8u || is_quant_linear_inv_sw_weight_slot(wid) || wid == OUT_FC_WEIGHT || wid == ONED_FINAL_EMBED_0_WEIGHT) {
      std::printf("[fp16_branch][WEIGHT] wid=%u key=%s kind=%s logical=%u words16=%u offset16=%u\n",
                  (unsigned)i,
                  kWeightKey[i],
                  fp16_branch_storage_kind_name(kind),
                  (unsigned)logical_elems,
                  (unsigned)expect_words16,
                  (unsigned)fp16_branch_weight_offset_words16(wid));
    }
  }

  std::printf("[fp16_branch][SUMMARY] bias_words16=%u weight_words16=%u total_words16=%u\n",
              (unsigned)fp16_branch_total_bias_words16(),
              (unsigned)fp16_branch_total_weight_words16(),
              (unsigned)fp16_branch_total_param_words16());
  std::printf("[fp16_branch][SUMMARY] fp16_words16=%u ternary_words16=%u bitpack_words16=%u\n",
              (unsigned)total_fp16_words16,
              (unsigned)total_ternary_words16,
              (unsigned)total_bitpack_words16);
  std::printf("PASS: tb_fp16_branch_weight_contract_smoke\n");
  return 0;
}

#else
int main() { return 0; }
#endif
