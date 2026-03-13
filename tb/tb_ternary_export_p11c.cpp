// tb_ternary_export_p11c.cpp
#ifndef __SYNTHESIS__

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "weights.h"
#include "gen/WeightStreamOrder.h"

struct MatrixExportRecord {
  const char* matrix_name;
  uint32_t matrix_id;
  uint32_t weight_param_id;
  uint32_t inv_sw_param_id;
  uint32_t rows;
  uint32_t cols;
  uint32_t num_weights;
  uint32_t payload_words_2b;
  uint32_t last_word_valid_count;
  uint32_t count_neg;
  uint32_t count_zero;
  uint32_t count_pos;
  std::vector<std::string> inv_sw_fp32_hex;
  std::vector<std::string> payload_hex_words;
};

static inline uint32_t local_fp32_bits_from_double(const double x) {
  const float f = (float)x;
  uint32_t bits = 0u;
  std::memcpy(&bits, &f, sizeof(uint32_t));
  return bits;
}

static inline const double* local_lookup_weight_fp64(const WeightId id, uint32_t& out_numel) {
  switch (id) {
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_0_weight_numel;
      return w_decoder_layers_0_self_attn_linears_0_weight;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_0_s_w_numel;
      return w_decoder_layers_0_self_attn_linears_0_s_w;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_1_weight_numel;
      return w_decoder_layers_0_self_attn_linears_1_weight;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_1_s_w_numel;
      return w_decoder_layers_0_self_attn_linears_1_s_w;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_2_weight_numel;
      return w_decoder_layers_0_self_attn_linears_2_weight;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_2_s_w_numel;
      return w_decoder_layers_0_self_attn_linears_2_s_w;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_3_weight_numel;
      return w_decoder_layers_0_self_attn_linears_3_weight;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_3_s_w_numel;
      return w_decoder_layers_0_self_attn_linears_3_s_w;
    case DECODER_LAYERS_0_FEED_FORWARD_W_1_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_1_weight_numel;
      return w_decoder_layers_0_feed_forward_w_1_weight;
    case DECODER_LAYERS_0_FEED_FORWARD_W_1_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_1_s_w_numel;
      return w_decoder_layers_0_feed_forward_w_1_s_w;
    case DECODER_LAYERS_0_FEED_FORWARD_W_2_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_2_weight_numel;
      return w_decoder_layers_0_feed_forward_w_2_weight;
    case DECODER_LAYERS_0_FEED_FORWARD_W_2_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_2_s_w_numel;
      return w_decoder_layers_0_feed_forward_w_2_s_w;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_0_weight_numel;
      return w_decoder_layers_1_self_attn_linears_0_weight;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_0_s_w_numel;
      return w_decoder_layers_1_self_attn_linears_0_s_w;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_1_weight_numel;
      return w_decoder_layers_1_self_attn_linears_1_weight;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_1_s_w_numel;
      return w_decoder_layers_1_self_attn_linears_1_s_w;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_2_weight_numel;
      return w_decoder_layers_1_self_attn_linears_2_weight;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_2_s_w_numel;
      return w_decoder_layers_1_self_attn_linears_2_s_w;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_3_weight_numel;
      return w_decoder_layers_1_self_attn_linears_3_weight;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_3_s_w_numel;
      return w_decoder_layers_1_self_attn_linears_3_s_w;
    case DECODER_LAYERS_1_FEED_FORWARD_W_1_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_1_weight_numel;
      return w_decoder_layers_1_feed_forward_w_1_weight;
    case DECODER_LAYERS_1_FEED_FORWARD_W_1_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_1_s_w_numel;
      return w_decoder_layers_1_feed_forward_w_1_s_w;
    case DECODER_LAYERS_1_FEED_FORWARD_W_2_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_2_weight_numel;
      return w_decoder_layers_1_feed_forward_w_2_weight;
    case DECODER_LAYERS_1_FEED_FORWARD_W_2_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_2_s_w_numel;
      return w_decoder_layers_1_feed_forward_w_2_s_w;
    default:
      out_numel = 0u;
      return (const double*)0;
  }
}

static inline bool local_ternary_code_from_fp64(const double v, uint32_t& out_code) {
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

static inline bool local_pack_ternary_words_from_fp64(const double* src,
                                                      const uint32_t src_numel,
                                                      const uint32_t expected_num_weights,
                                                      uint32_t* out_payload,
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
    if (!local_ternary_code_from_fp64(src[idx], code)) {
      return false;
    }
    const uint32_t word_idx = (idx >> 4);
    const uint32_t shift = ((idx & 15u) << 1);
    out_payload[word_idx] |= ((code & 0x3u) << shift);
  }

  if (out_last_word_valid_count < 16u) {
    const uint32_t valid_bits = (out_last_word_valid_count << 1);
    const uint32_t mask = (1u << valid_bits) - 1u;
    out_payload[out_payload_words - 1u] &= mask;
  }

  return true;
}

static inline bool local_decode_ternary_code_at(const uint32_t* payload,
                                                const uint32_t payload_words,
                                                const uint32_t weight_idx,
                                                uint32_t& out_code) {
  if (!payload) {
    return false;
  }
  const uint32_t word_idx = (weight_idx >> 4);
  if (word_idx >= payload_words) {
    return false;
  }
  const uint32_t shift = ((weight_idx & 15u) << 1);
  out_code = (payload[word_idx] >> shift) & 0x3u;
  return true;
}

static std::string to_hex_word(const uint32_t word) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%08X", (unsigned)word);
  return std::string(buf);
}

static bool write_hex_array(FILE* fp, const std::vector<std::string>& values) {
  if (!fp) {
    return false;
  }
  std::fputc('[', fp);
  for (uint32_t i = 0u; i < (uint32_t)values.size(); ++i) {
    if (i != 0u) {
      std::fputs(", ", fp);
    }
    std::fprintf(fp, "\"%s\"", values[i].c_str());
  }
  std::fputc(']', fp);
  return true;
}

static int fail_matrix(const char* matrix_name,
                       const uint32_t matrix_id,
                       const uint32_t weight_param_id,
                       const char* reason,
                       const int64_t source_index,
                       const int64_t payload_word_index) {
  std::fprintf(stderr,
               "[p11c][FAIL] matrix=%s matrix_id=%u weight_param_id=%u source_index=%lld payload_word_index=%lld reason=%s\n",
               matrix_name,
               (unsigned)matrix_id,
               (unsigned)weight_param_id,
               (long long)source_index,
               (long long)payload_word_index,
               reason);
  return 1;
}

int main() {
  const char* const out_path = "gen/ternary_p11c_export.json";
  std::vector<MatrixExportRecord> records;
  records.reserve((uint32_t)QUANT_LINEAR_MATRIX_COUNT);

  for (uint32_t i = 0u; i < (uint32_t)QUANT_LINEAR_MATRIX_COUNT; ++i) {
    const QuantLinearMeta meta = kQuantLinearMeta[i];
    const char* const matrix_name = quant_linear_matrix_id_name(meta.matrix_id);

    WeightId weight_wid = WEIGHT_COUNT;
    if (!quant_linear_weight_param_id_to_weight_id(meta.weight_param_id, weight_wid)) {
      return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                         "weight_param_id to WeightId mapping failed", -1, -1);
    }

    WeightId inv_sw_wid = WEIGHT_COUNT;
    if (!quant_linear_matrix_id_to_inv_sw_weight_id(meta.matrix_id, inv_sw_wid)) {
      return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                         "matrix_id to inv_s_w WeightId mapping failed", -1, -1);
    }

    uint32_t weight_numel = 0u;
    const double* const weight_src = local_lookup_weight_fp64(weight_wid, weight_numel);
    if (!weight_src) {
      return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                         "weight source lookup failed", -1, -1);
    }
    if (weight_numel != meta.num_weights) {
      return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                         "weight numel mismatch", -1, -1);
    }

    uint32_t sw_numel = 0u;
    const double* const sw_src = local_lookup_weight_fp64(inv_sw_wid, sw_numel);
    if (!sw_src) {
      return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                         "s_w source lookup failed", -1, -1);
    }
    if (sw_numel < 1u) {
      return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                         "s_w source numel < 1", -1, -1);
    }

    std::vector<uint32_t> payload(meta.payload_words_2b, 0u);
    uint32_t payload_words_2b = 0u;
    uint32_t last_word_valid_count = 0u;
    if (!local_pack_ternary_words_from_fp64(weight_src,
                                            weight_numel,
                                            meta.num_weights,
                                            payload.data(),
                                            (uint32_t)payload.size(),
                                            payload_words_2b,
                                            last_word_valid_count)) {
      return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                         "pack failed", -1, -1);
    }

    if (payload_words_2b != meta.payload_words_2b) {
      return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                         "payload_words_2b mismatch", -1, -1);
    }
    if (last_word_valid_count != meta.last_word_valid_count) {
      return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                         "last_word_valid_count mismatch", -1, -1);
    }

    MatrixExportRecord rec;
    rec.matrix_name = matrix_name;
    rec.matrix_id = meta.matrix_id;
    rec.weight_param_id = meta.weight_param_id;
    rec.inv_sw_param_id = meta.inv_sw_param_id;
    rec.rows = meta.rows;
    rec.cols = meta.cols;
    rec.num_weights = meta.num_weights;
    rec.payload_words_2b = payload_words_2b;
    rec.last_word_valid_count = last_word_valid_count;
    rec.count_neg = 0u;
    rec.count_zero = 0u;
    rec.count_pos = 0u;

    for (uint32_t idx = 0u; idx < meta.num_weights; ++idx) {
      uint32_t src_code = 0u;
      if (!local_ternary_code_from_fp64(weight_src[idx], src_code)) {
        return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                           "illegal ternary source value", (int64_t)idx, -1);
      }
      if (src_code == (uint32_t)TERNARY_CODE_NEG) {
        ++rec.count_neg;
      } else if (src_code == (uint32_t)TERNARY_CODE_ZERO) {
        ++rec.count_zero;
      } else if (src_code == (uint32_t)TERNARY_CODE_POS) {
        ++rec.count_pos;
      }

      uint32_t dec_code = 0u;
      if (!local_decode_ternary_code_at(payload.data(), payload_words_2b, idx, dec_code)) {
        return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                           "decode failed", (int64_t)idx, -1);
      }
      if (dec_code != src_code) {
        return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                           "decode mismatch", (int64_t)idx, (int64_t)(idx >> 4));
      }
    }

    if (last_word_valid_count < 16u) {
      const uint32_t last_word = payload[payload_words_2b - 1u];
      for (uint32_t slot = last_word_valid_count; slot < 16u; ++slot) {
        const uint32_t code = (last_word >> (slot * 2u)) & 0x3u;
        if (code != 0u) {
          return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                             "tail padding mismatch", -1, (int64_t)(payload_words_2b - 1u));
        }
      }
    }

    rec.payload_hex_words.reserve(payload_words_2b);
    for (uint32_t w = 0u; w < payload_words_2b; ++w) {
      rec.payload_hex_words.push_back(to_hex_word(payload[w]));
    }

    rec.inv_sw_fp32_hex.reserve(sw_numel);
    for (uint32_t s = 0u; s < sw_numel; ++s) {
      const double s_w = sw_src[s];
      if (s_w == 0.0) {
        return fail_matrix(matrix_name, meta.matrix_id, meta.weight_param_id,
                           "s_w==0 during inv_s_w export", (int64_t)s, -1);
      }
      const uint32_t inv_bits = local_fp32_bits_from_double(1.0 / s_w);
      rec.inv_sw_fp32_hex.push_back(to_hex_word(inv_bits));
    }

    records.push_back(rec);
    std::printf("[p11c][PASS] matrix=%s weight_param_id=%u inv_sw_param_id=%u rows=%u cols=%u num_weights=%u payload_words_2b=%u last_word_valid_count=%u counts(neg,zero,pos)=(%u,%u,%u)\n",
                matrix_name,
                (unsigned)meta.weight_param_id,
                (unsigned)meta.inv_sw_param_id,
                (unsigned)meta.rows,
                (unsigned)meta.cols,
                (unsigned)meta.num_weights,
                (unsigned)payload_words_2b,
                (unsigned)last_word_valid_count,
                (unsigned)rec.count_neg,
                (unsigned)rec.count_zero,
                (unsigned)rec.count_pos);
  }

  FILE* fp = std::fopen(out_path, "wb");
  if (!fp) {
    std::fprintf(stderr, "[p11c][FAIL] cannot open output artifact: %s\n", out_path);
    return 1;
  }

  std::fprintf(fp, "{\n");
  std::fprintf(fp, "  \"version\": \"v12.1-p11c\",\n");
  std::fprintf(fp, "  \"format\": \"ternary-packed-2b-lsb-first-offline-export\",\n");
  std::fprintf(fp, "  \"matrix_count\": %u,\n", (unsigned)records.size());
  std::fprintf(fp, "  \"matrices\": [\n");

  for (uint32_t i = 0u; i < (uint32_t)records.size(); ++i) {
    const MatrixExportRecord& rec = records[i];
    std::fprintf(fp, "    {\n");
    std::fprintf(fp, "      \"matrix_id\": \"%s\",\n", rec.matrix_name);
    std::fprintf(fp, "      \"weight_param_id\": %u,\n", (unsigned)rec.weight_param_id);
    std::fprintf(fp, "      \"inv_sw_param_id\": %u,\n", (unsigned)rec.inv_sw_param_id);
    std::fprintf(fp, "      \"rows\": %u,\n", (unsigned)rec.rows);
    std::fprintf(fp, "      \"cols\": %u,\n", (unsigned)rec.cols);
    std::fprintf(fp, "      \"num_weights\": %u,\n", (unsigned)rec.num_weights);
    std::fprintf(fp, "      \"payload_words_2b\": %u,\n", (unsigned)rec.payload_words_2b);
    std::fprintf(fp, "      \"last_word_valid_count\": %u,\n", (unsigned)rec.last_word_valid_count);
    std::fprintf(fp, "      \"inv_sw_fp32_hex\": ");
    if (!write_hex_array(fp, rec.inv_sw_fp32_hex)) {
      std::fclose(fp);
      std::fprintf(stderr, "[p11c][FAIL] failed to write inv_sw_fp32_hex array\n");
      return 1;
    }
    std::fprintf(fp, ",\n");
    std::fprintf(fp, "      \"payload_hex_words\": ");
    if (!write_hex_array(fp, rec.payload_hex_words)) {
      std::fclose(fp);
      std::fprintf(stderr, "[p11c][FAIL] failed to write payload_hex_words array\n");
      return 1;
    }
    std::fprintf(fp, ",\n");
    std::fprintf(fp, "      \"count_neg\": %u,\n", (unsigned)rec.count_neg);
    std::fprintf(fp, "      \"count_zero\": %u,\n", (unsigned)rec.count_zero);
    std::fprintf(fp, "      \"count_pos\": %u,\n", (unsigned)rec.count_pos);
    std::fprintf(fp, "      \"status\": \"PASS\"\n");
    std::fprintf(fp, "    }%s\n", (i + 1u == (uint32_t)records.size()) ? "" : ",");
  }

  std::fprintf(fp, "  ]\n");
  std::fprintf(fp, "}\n");
  std::fclose(fp);

  std::printf("[p11c][PASS] exported artifact: %s\n", out_path);
  return 0;
}

#else
int main() { return 0; }
#endif
