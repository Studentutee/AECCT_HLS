#include <cstdint>
#include <cstdio>
#include <vector>

#include "weights_streamer.h"

static bool report_fail(const char* reason,
                        const QuantLinearMeta& meta,
                        const uint32_t source_idx,
                        const uint32_t payload_word_idx) {
  std::printf("FAIL: matrix_id=%u weight_param_id=%u source_idx=%u payload_word_idx=%u reason=%s\n",
              (unsigned)meta.matrix_id,
              (unsigned)meta.weight_param_id,
              (unsigned)source_idx,
              (unsigned)payload_word_idx,
              reason);
  return false;
}

int main() {
  bool all_pass = true;

  for (uint32_t i = 0u; i < (uint32_t)QUANT_LINEAR_MATRIX_COUNT; ++i) {
    const QuantLinearMeta& meta = kQuantLinearMeta[i];

    WeightId wid_from_param = BCH_H_BITPACK;
    if (!quant_linear_weight_param_id_to_weight_id(meta.weight_param_id, wid_from_param)) {
      all_pass = report_fail("weight_param_id_to_weight_id failed", meta, 0u, 0u) && all_pass;
      continue;
    }

    WeightId wid_from_matrix = BCH_H_BITPACK;
    if (!quant_linear_matrix_id_to_weight_id(meta.matrix_id, wid_from_matrix)) {
      all_pass = report_fail("matrix_id_to_weight_id failed", meta, 0u, 0u) && all_pass;
      continue;
    }

    if (wid_from_param != wid_from_matrix) {
      all_pass = report_fail("mapping mismatch between param_id and matrix_id", meta, 0u, 0u) && all_pass;
      continue;
    }

    if (!is_quant_linear_weight_slot(wid_from_param)) {
      all_pass = report_fail("weight slot is not quant-linear", meta, 0u, 0u) && all_pass;
      continue;
    }

    const QuantLinearMeta* lookup_meta = lookup_quant_linear_meta_by_weight_param_id(meta.weight_param_id);
    if (!lookup_meta || lookup_meta->matrix_id != meta.matrix_id) {
      all_pass = report_fail("lookup_quant_linear_meta_by_weight_param_id mismatch", meta, 0u, 0u) && all_pass;
      continue;
    }

    uint32_t src_numel = 0u;
    const double* src = tb_lookup_weight_fp64(wid_from_param, src_numel);
    if (!src) {
      all_pass = report_fail("tb_lookup_weight_fp64 returned null", meta, 0u, 0u) && all_pass;
      continue;
    }

    const uint32_t expected_num_weights = meta.rows * meta.cols;
    if (meta.num_weights != expected_num_weights) {
      all_pass = report_fail("meta num_weights mismatch rows*cols", meta, 0u, 0u) && all_pass;
      continue;
    }
    if (src_numel != expected_num_weights) {
      all_pass = report_fail("source numel mismatch", meta, src_numel, 0u) && all_pass;
      continue;
    }

    std::vector<uint32_t> payload(meta.payload_words_2b, 0u);
    uint32_t out_payload_words = 0u;
    uint32_t out_last_word_valid_count = 0u;
    if (!tb_pack_ternary_words_from_fp64(src,
                                         src_numel,
                                         expected_num_weights,
                                         payload.data(),
                                         (uint32_t)payload.size(),
                                         out_payload_words,
                                         out_last_word_valid_count)) {
      all_pass = report_fail("pack failed (illegal value/size/capacity)", meta, 0u, 0u) && all_pass;
      continue;
    }

    if (out_payload_words != meta.payload_words_2b) {
      all_pass = report_fail("payload_words mismatch", meta, out_payload_words, 0u) && all_pass;
      continue;
    }
    if (out_last_word_valid_count != meta.last_word_valid_count) {
      all_pass = report_fail("last_word_valid_count mismatch", meta, out_last_word_valid_count, 0u) && all_pass;
      continue;
    }

    bool decode_ok = true;
    for (uint32_t idx = 0u; idx < expected_num_weights; ++idx) {
      uint32_t expected_code = 0u;
      if (!tb_ternary_code_from_fp64(src[idx], expected_code)) {
        decode_ok = report_fail("illegal ternary source value", meta, idx, (idx >> 4));
        break;
      }

      uint32_t got_code = 0u;
      if (!tb_decode_ternary_code_at(payload.data(), out_payload_words, idx, got_code)) {
        decode_ok = report_fail("decode helper failed", meta, idx, (idx >> 4));
        break;
      }

      if (got_code != expected_code) {
        decode_ok = report_fail("decode mismatch", meta, idx, (idx >> 4));
        break;
      }
    }

    if (!decode_ok) {
      all_pass = false;
      continue;
    }

    if (out_last_word_valid_count < 16u) {
      const uint32_t last_word_idx = out_payload_words - 1u;
      const uint32_t last_word = payload[last_word_idx];
      for (uint32_t slot = out_last_word_valid_count; slot < 16u; ++slot) {
        const uint32_t code = (last_word >> (slot * 2u)) & 0x3u;
        if (code != (uint32_t)TERNARY_CODE_ZERO) {
          all_pass = report_fail("tail padding mismatch (non-zero)",
                                 meta,
                                 expected_num_weights,
                                 last_word_idx) && all_pass;
          decode_ok = false;
          break;
        }
      }
      if (!decode_ok) {
        continue;
      }
    }

    std::printf("PASS: matrix_id=%u weight_param_id=%u rows=%u cols=%u num_weights=%u payload_words_2b=%u last_word_valid_count=%u\n",
                (unsigned)meta.matrix_id,
                (unsigned)meta.weight_param_id,
                (unsigned)meta.rows,
                (unsigned)meta.cols,
                (unsigned)meta.num_weights,
                (unsigned)meta.payload_words_2b,
                (unsigned)meta.last_word_valid_count);
  }

  if (!all_pass) {
    std::printf("FAIL: tb_ternary_pack_p11b\n");
    return 1;
  }

  std::printf("PASS: tb_ternary_pack_p11b\n");
  return 0;
}
