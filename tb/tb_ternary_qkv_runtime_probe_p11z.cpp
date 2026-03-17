// P00-011Z: local-only QKV runtime consume probe.
// Validation-only TB. No algorithm change and no new metadata authority source.
#ifndef __SYNTHESIS__

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "gen/WeightStreamOrder.h"

namespace {

struct MatrixTarget {
  const char* matrix_name;
  uint32_t matrix_id;
};

static const MatrixTarget kQkvTargets[] = {
    {"L0_WQ", (uint32_t)QLM_L0_WQ},
    {"L0_WK", (uint32_t)QLM_L0_WK},
    {"L0_WV", (uint32_t)QLM_L0_WV},
};

static bool read_text_file(const char* path, std::string& out_text) {
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    return false;
  }
  out_text.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return true;
}

static inline void skip_ws(const std::string& text, size_t& pos) {
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
}

static bool parse_json_string(const std::string& text, size_t& pos, std::string& out_value) {
  out_value.clear();
  if (pos >= text.size() || text[pos] != '"') {
    return false;
  }
  ++pos;
  while (pos < text.size()) {
    const char c = text[pos++];
    if (c == '"') {
      return true;
    }
    if (c == '\\') {
      if (pos >= text.size()) {
        return false;
      }
      const char esc = text[pos++];
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          out_value.push_back(esc);
          break;
        case 'b':
          out_value.push_back('\b');
          break;
        case 'f':
          out_value.push_back('\f');
          break;
        case 'n':
          out_value.push_back('\n');
          break;
        case 'r':
          out_value.push_back('\r');
          break;
        case 't':
          out_value.push_back('\t');
          break;
        default:
          return false;
      }
      continue;
    }
    out_value.push_back(c);
  }
  return false;
}

static bool find_matching_delim(const std::string& text,
                                const size_t open_pos,
                                const char open_ch,
                                const char close_ch,
                                size_t& out_close_pos) {
  if (open_pos >= text.size() || text[open_pos] != open_ch) {
    return false;
  }

  uint32_t depth = 0u;
  bool in_string = false;
  bool in_escape = false;

  for (size_t i = open_pos; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (in_escape) {
        in_escape = false;
      } else if (c == '\\') {
        in_escape = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == open_ch) {
      ++depth;
      continue;
    }
    if (c == close_ch) {
      if (depth == 0u) {
        return false;
      }
      --depth;
      if (depth == 0u) {
        out_close_pos = i;
        return true;
      }
    }
  }
  return false;
}

static bool extract_string_field(const std::string& obj,
                                 const char* field_name,
                                 std::string& out_value,
                                 std::string& out_err) {
  const std::string key = std::string("\"") + field_name + "\"";
  const size_t key_pos = obj.find(key);
  if (key_pos == std::string::npos) {
    out_err = std::string("missing field: ") + field_name;
    return false;
  }

  size_t colon_pos = obj.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) {
    out_err = std::string("missing colon for field: ") + field_name;
    return false;
  }
  ++colon_pos;
  skip_ws(obj, colon_pos);
  if (!parse_json_string(obj, colon_pos, out_value)) {
    out_err = std::string("invalid JSON string field: ") + field_name;
    return false;
  }
  return true;
}

static bool extract_u32_field(const std::string& obj,
                              const char* field_name,
                              uint32_t& out_value,
                              std::string& out_err) {
  const std::string key = std::string("\"") + field_name + "\"";
  const size_t key_pos = obj.find(key);
  if (key_pos == std::string::npos) {
    out_err = std::string("missing field: ") + field_name;
    return false;
  }

  size_t colon_pos = obj.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) {
    out_err = std::string("missing colon for field: ") + field_name;
    return false;
  }
  ++colon_pos;
  skip_ws(obj, colon_pos);

  if (colon_pos >= obj.size() || !std::isdigit(static_cast<unsigned char>(obj[colon_pos]))) {
    out_err = std::string("field is not unsigned integer: ") + field_name;
    return false;
  }

  uint64_t value = 0u;
  while (colon_pos < obj.size() && std::isdigit(static_cast<unsigned char>(obj[colon_pos]))) {
    value = (value * 10u) + static_cast<uint64_t>(obj[colon_pos] - '0');
    if (value > 0xFFFFFFFFull) {
      out_err = std::string("field overflows uint32: ") + field_name;
      return false;
    }
    ++colon_pos;
  }

  out_value = static_cast<uint32_t>(value);
  return true;
}

static bool try_extract_array_count(const std::string& obj,
                                    const char* field_name,
                                    bool& out_found,
                                    uint32_t& out_count,
                                    std::string& out_err) {
  out_found = false;
  out_count = 0u;
  const std::string key = std::string("\"") + field_name + "\"";
  const size_t key_pos = obj.find(key);
  if (key_pos == std::string::npos) {
    return true;
  }

  out_found = true;
  size_t colon_pos = obj.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) {
    out_err = std::string("missing colon for array field: ") + field_name;
    return false;
  }
  ++colon_pos;
  skip_ws(obj, colon_pos);
  if (colon_pos >= obj.size() || obj[colon_pos] != '[') {
    out_err = std::string("array field is not '[': ") + field_name;
    return false;
  }

  size_t array_close = 0u;
  if (!find_matching_delim(obj, colon_pos, '[', ']', array_close)) {
    out_err = std::string("cannot find closing ']' for array field: ") + field_name;
    return false;
  }

  size_t pos = colon_pos + 1u;
  uint32_t count = 0u;
  while (pos < array_close) {
    skip_ws(obj, pos);
    if (pos >= array_close) {
      break;
    }
    if (obj[pos] == ',') {
      ++pos;
      continue;
    }
    if (obj[pos] == '"') {
      std::string ignored;
      if (!parse_json_string(obj, pos, ignored)) {
        out_err = std::string("invalid JSON string element in array field: ") + field_name;
        return false;
      }
      ++count;
      continue;
    }
    if (obj[pos] == '[' || obj[pos] == '{') {
      out_err = std::string("nested non-string array value is unsupported for field: ") + field_name;
      return false;
    }
    while (pos < array_close && obj[pos] != ',' && obj[pos] != ']') {
      ++pos;
    }
    ++count;
  }

  out_count = count;
  return true;
}

static bool extract_matrices_objects(const std::string& json_text,
                                     std::vector<std::string>& out_objects,
                                     std::string& out_err) {
  out_objects.clear();
  const std::string matrices_key = "\"matrices\"";
  const size_t key_pos = json_text.find(matrices_key);
  if (key_pos == std::string::npos) {
    out_err = "missing key: matrices";
    return false;
  }

  size_t colon_pos = json_text.find(':', key_pos + matrices_key.size());
  if (colon_pos == std::string::npos) {
    out_err = "missing ':' for matrices";
    return false;
  }
  ++colon_pos;
  skip_ws(json_text, colon_pos);
  if (colon_pos >= json_text.size() || json_text[colon_pos] != '[') {
    out_err = "matrices value is not an array";
    return false;
  }

  size_t array_close = 0u;
  if (!find_matching_delim(json_text, colon_pos, '[', ']', array_close)) {
    out_err = "cannot find closing ']' for matrices array";
    return false;
  }

  size_t pos = colon_pos + 1u;
  while (pos < array_close) {
    skip_ws(json_text, pos);
    if (pos >= array_close) {
      break;
    }
    if (json_text[pos] == ',') {
      ++pos;
      continue;
    }
    if (json_text[pos] != '{') {
      out_err = "matrices array contains non-object element";
      return false;
    }

    size_t obj_close = 0u;
    if (!find_matching_delim(json_text, pos, '{', '}', obj_close)) {
      out_err = "cannot find closing '}' for matrix object";
      return false;
    }
    out_objects.push_back(json_text.substr(pos, (obj_close - pos) + 1u));
    pos = obj_close + 1u;
  }

  if (out_objects.empty()) {
    out_err = "matrices array is empty";
    return false;
  }
  return true;
}

static int fail_with_reason(const char* matrix_name, const std::string& reason) {
  if (matrix_name && matrix_name[0] != '\0') {
    std::fprintf(stderr, "[p11z][FAIL] matrix=%s reason=%s\n", matrix_name, reason.c_str());
  } else {
    std::fprintf(stderr, "[p11z][FAIL] reason=%s\n", reason.c_str());
  }
  return 1;
}

}  // namespace

int main() {
  const char* const json_path = "gen/ternary_p11c_export.json";

  std::string json_text;
  if (!read_text_file(json_path, json_text)) {
    return fail_with_reason("", std::string("cannot read JSON artifact: ") + json_path);
  }

  std::vector<std::string> matrix_objects;
  std::string parse_err;
  if (!extract_matrices_objects(json_text, matrix_objects, parse_err)) {
    return fail_with_reason("", std::string("cannot parse matrices array: ") + parse_err);
  }

  for (uint32_t t = 0u; t < (uint32_t)(sizeof(kQkvTargets) / sizeof(kQkvTargets[0])); ++t) {
    const MatrixTarget& target = kQkvTargets[t];
    int32_t matched_index = -1;
    uint32_t match_count = 0u;

    for (uint32_t i = 0u; i < (uint32_t)matrix_objects.size(); ++i) {
      std::string matrix_name;
      std::string field_err;
      if (!extract_string_field(matrix_objects[i], "matrix_id", matrix_name, field_err)) {
        return fail_with_reason("", std::string("malformed matrix record: ") + field_err);
      }
      if (matrix_name == target.matrix_name) {
        ++match_count;
        matched_index = (int32_t)i;
      }
    }

    if (match_count != 1u || matched_index < 0) {
      std::string reason = "expected exactly one JSON record for matrix_id=";
      reason += target.matrix_name;
      reason += ", found ";
      reason += std::to_string(match_count);
      return fail_with_reason(target.matrix_name, reason);
    }

    const std::string& obj = matrix_objects[(uint32_t)matched_index];
    const QuantLinearMeta& meta = kQuantLinearMeta[target.matrix_id];
    const char* const meta_name = quant_linear_matrix_id_name(meta.matrix_id);

    std::string matrix_id_str;
    std::string field_err;
    if (!extract_string_field(obj, "matrix_id", matrix_id_str, field_err)) {
      return fail_with_reason(target.matrix_name, field_err);
    }
    if (matrix_id_str != target.matrix_name) {
      return fail_with_reason(target.matrix_name, "matrix_id string mismatch in JSON record");
    }
    if (std::strcmp(meta_name, target.matrix_name) != 0) {
      return fail_with_reason(target.matrix_name, "WeightStreamOrder matrix name mismatch");
    }

    uint32_t rows = 0u;
    uint32_t cols = 0u;
    uint32_t num_weights = 0u;
    uint32_t payload_words_2b = 0u;
    uint32_t last_word_valid_count = 0u;
    uint32_t weight_param_id = 0u;
    uint32_t inv_sw_param_id = 0u;

    if (!extract_u32_field(obj, "rows", rows, field_err) ||
        !extract_u32_field(obj, "cols", cols, field_err) ||
        !extract_u32_field(obj, "num_weights", num_weights, field_err) ||
        !extract_u32_field(obj, "payload_words_2b", payload_words_2b, field_err) ||
        !extract_u32_field(obj, "last_word_valid_count", last_word_valid_count, field_err) ||
        !extract_u32_field(obj, "weight_param_id", weight_param_id, field_err) ||
        !extract_u32_field(obj, "inv_sw_param_id", inv_sw_param_id, field_err)) {
      return fail_with_reason(target.matrix_name, field_err);
    }

    if (meta.matrix_id != target.matrix_id) {
      return fail_with_reason(target.matrix_name, "kQuantLinearMeta matrix_id mismatch");
    }
    if (rows != meta.rows || cols != meta.cols || num_weights != meta.num_weights ||
        payload_words_2b != meta.payload_words_2b ||
        last_word_valid_count != meta.last_word_valid_count ||
        weight_param_id != meta.weight_param_id ||
        inv_sw_param_id != meta.inv_sw_param_id) {
      return fail_with_reason(target.matrix_name, "JSON metadata mismatch against authoritative kQuantLinearMeta");
    }

    const uint32_t expected_payload_words = ternary_payload_words_2b(num_weights);
    if (payload_words_2b != expected_payload_words) {
      return fail_with_reason(target.matrix_name, "payload_words_2b mismatch against ternary_payload_words_2b(num_weights)");
    }
    const uint32_t expected_last_word_valid_count = ternary_last_word_valid_count(num_weights);
    if (last_word_valid_count != expected_last_word_valid_count) {
      return fail_with_reason(target.matrix_name, "last_word_valid_count mismatch against ternary_last_word_valid_count(num_weights)");
    }

    bool array_found = false;
    uint32_t payload_array_count = 0u;
    if (!try_extract_array_count(obj, "payload_hex_words", array_found, payload_array_count, field_err)) {
      return fail_with_reason(target.matrix_name, field_err);
    }
    if (!array_found) {
      if (!try_extract_array_count(obj, "payload_words_hex", array_found, payload_array_count, field_err)) {
        return fail_with_reason(target.matrix_name, field_err);
      }
    }
    if (!array_found) {
      return fail_with_reason(target.matrix_name,
                              "schema limitation/mismatch: missing payload-word array key (payload_hex_words or payload_words_hex)");
    }
    if (payload_array_count != payload_words_2b) {
      return fail_with_reason(target.matrix_name, "payload-word array length mismatch against payload_words_2b");
    }

    WeightId weight_wid = WEIGHT_COUNT;
    if (!quant_linear_weight_param_id_to_weight_id(weight_param_id, weight_wid)) {
      return fail_with_reason(target.matrix_name, "quant_linear_weight_param_id_to_weight_id failed");
    }
    WeightId weight_wid_ref = WEIGHT_COUNT;
    if (!quant_linear_matrix_id_to_weight_id(meta.matrix_id, weight_wid_ref)) {
      return fail_with_reason(target.matrix_name, "quant_linear_matrix_id_to_weight_id failed");
    }
    if (weight_wid != weight_wid_ref) {
      return fail_with_reason(target.matrix_name, "weight mapping mismatch between param_id path and matrix_id path");
    }

    uint32_t weight_param_roundtrip = 0u;
    if (!weight_id_to_param_id(weight_wid, weight_param_roundtrip)) {
      return fail_with_reason(target.matrix_name, "weight_id_to_param_id failed for weight path");
    }
    if (weight_param_roundtrip != weight_param_id) {
      return fail_with_reason(target.matrix_name, "weight_param_id roundtrip mismatch");
    }

    const QuantLinearMeta* meta_by_weight = lookup_quant_linear_meta_by_weight_param_id(weight_param_id);
    if (meta_by_weight == (const QuantLinearMeta*)0 || meta_by_weight->matrix_id != meta.matrix_id) {
      return fail_with_reason(target.matrix_name, "lookup_quant_linear_meta_by_weight_param_id continuity mismatch");
    }

    const QuantLinearMeta* meta_by_inv_sw = lookup_quant_linear_meta_by_inv_sw_param_id(inv_sw_param_id);
    if (meta_by_inv_sw == (const QuantLinearMeta*)0 || meta_by_inv_sw->matrix_id != meta.matrix_id) {
      return fail_with_reason(target.matrix_name, "lookup_quant_linear_meta_by_inv_sw_param_id continuity mismatch");
    }

    WeightId inv_sw_wid = WEIGHT_COUNT;
    if (!quant_linear_matrix_id_to_inv_sw_weight_id(meta.matrix_id, inv_sw_wid)) {
      return fail_with_reason(target.matrix_name, "quant_linear_matrix_id_to_inv_sw_weight_id failed");
    }
    uint32_t inv_sw_param_roundtrip = 0u;
    if (!weight_id_to_param_id(inv_sw_wid, inv_sw_param_roundtrip)) {
      return fail_with_reason(target.matrix_name, "weight_id_to_param_id failed for inv_sw path");
    }
    if (inv_sw_param_roundtrip != inv_sw_param_id) {
      return fail_with_reason(target.matrix_name, "inv_sw_param_id roundtrip mismatch");
    }

    std::printf(
        "[p11z][PASS] matrix=%s rows=%u cols=%u num_weights=%u payload_words_2b=%u last_word_valid_count=%u weight_param_id=%u inv_sw_param_id=%u\n",
        target.matrix_name,
        (unsigned)rows,
        (unsigned)cols,
        (unsigned)num_weights,
        (unsigned)payload_words_2b,
        (unsigned)last_word_valid_count,
        (unsigned)weight_param_id,
        (unsigned)inv_sw_param_id);
  }

  std::printf("PASS: tb_ternary_qkv_runtime_probe_p11z\n");
  return 0;
}

#endif  // __SYNTHESIS__
