// tb_ternary_loader_preview_p11d.cpp
#ifndef __SYNTHESIS__

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "weights.h"
#include "gen/WeightStreamOrder.h"

namespace {

enum class JsonType {
  kNull,
  kBool,
  kNumber,
  kString,
  kArray,
  kObject,
};

struct JsonValue {
  JsonType type = JsonType::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string str;
  std::vector<JsonValue> arr;
  std::unordered_map<std::string, JsonValue> obj;
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text), pos_(0u) {}

  bool Parse(JsonValue& out) {
    SkipWs();
    if (!ParseValue(out)) {
      return false;
    }
    SkipWs();
    if (pos_ != text_.size()) {
      return Fail("trailing content after root JSON value");
    }
    return true;
  }

  const std::string& error() const { return error_; }

 private:
  bool Fail(const char* msg) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s at byte %u", msg, (unsigned)pos_);
    error_ = buf;
    return false;
  }

  bool Fail(const std::string& msg) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s at byte %u", msg.c_str(), (unsigned)pos_);
    error_ = buf;
    return false;
  }

  void SkipWs() {
    while (pos_ < text_.size()) {
      const unsigned char c = (unsigned char)text_[pos_];
      if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  bool ParseValue(JsonValue& out) {
    if (pos_ >= text_.size()) {
      return Fail("unexpected EOF while parsing value");
    }
    const char c = text_[pos_];
    if (c == '{') {
      return ParseObject(out);
    }
    if (c == '[') {
      return ParseArray(out);
    }
    if (c == '"') {
      out.type = JsonType::kString;
      return ParseString(out.str);
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
      out.type = JsonType::kNumber;
      return ParseNumber(out.number);
    }
    if (text_.compare(pos_, 4u, "true") == 0) {
      out.type = JsonType::kBool;
      out.boolean = true;
      pos_ += 4u;
      return true;
    }
    if (text_.compare(pos_, 5u, "false") == 0) {
      out.type = JsonType::kBool;
      out.boolean = false;
      pos_ += 5u;
      return true;
    }
    if (text_.compare(pos_, 4u, "null") == 0) {
      out.type = JsonType::kNull;
      pos_ += 4u;
      return true;
    }
    return Fail("unexpected token while parsing value");
  }

  bool ParseObject(JsonValue& out) {
    if (text_[pos_] != '{') {
      return Fail("expected '{'");
    }
    ++pos_;
    out.type = JsonType::kObject;
    out.obj.clear();

    SkipWs();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return true;
    }

    while (pos_ < text_.size()) {
      SkipWs();
      if (pos_ >= text_.size() || text_[pos_] != '"') {
        return Fail("expected object key string");
      }
      std::string key;
      if (!ParseString(key)) {
        return false;
      }

      if (out.obj.find(key) != out.obj.end()) {
        return Fail(std::string("duplicate key '") + key + "'");
      }

      SkipWs();
      if (pos_ >= text_.size() || text_[pos_] != ':') {
        return Fail("expected ':' after object key");
      }
      ++pos_;

      JsonValue value;
      SkipWs();
      if (!ParseValue(value)) {
        return false;
      }
      out.obj.insert(std::make_pair(key, value));

      SkipWs();
      if (pos_ < text_.size() && text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (pos_ < text_.size() && text_[pos_] == '}') {
        ++pos_;
        return true;
      }
      return Fail("expected ',' or '}' in object");
    }

    return Fail("unexpected EOF in object");
  }

  bool ParseArray(JsonValue& out) {
    if (text_[pos_] != '[') {
      return Fail("expected '['");
    }
    ++pos_;
    out.type = JsonType::kArray;
    out.arr.clear();

    SkipWs();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return true;
    }

    while (pos_ < text_.size()) {
      JsonValue value;
      SkipWs();
      if (!ParseValue(value)) {
        return false;
      }
      out.arr.push_back(value);

      SkipWs();
      if (pos_ < text_.size() && text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (pos_ < text_.size() && text_[pos_] == ']') {
        ++pos_;
        return true;
      }
      return Fail("expected ',' or ']' in array");
    }

    return Fail("unexpected EOF in array");
  }

  bool ParseString(std::string& out) {
    if (pos_ >= text_.size() || text_[pos_] != '"') {
      return Fail("expected '\"' for string");
    }
    ++pos_;
    out.clear();
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"') {
        return true;
      }
      if ((unsigned char)c < 0x20u) {
        return Fail("control character in string");
      }
      if (c == '\\') {
        if (pos_ >= text_.size()) {
          return Fail("unexpected EOF in string escape");
        }
        const char esc = text_[pos_++];
        switch (esc) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u':
            return Fail("unicode escapes are not supported by this minimal parser");
          default:
            return Fail("invalid escape in string");
        }
      } else {
        out.push_back(c);
      }
    }
    return Fail("unexpected EOF in string");
  }

  bool ParseNumber(double& out) {
    const size_t start = pos_;

    if (text_[pos_] == '-') {
      ++pos_;
      if (pos_ >= text_.size()) {
        return Fail("incomplete number");
      }
    }

    if (text_[pos_] == '0') {
      ++pos_;
    } else {
      if (text_[pos_] < '1' || text_[pos_] > '9') {
        return Fail("invalid number");
      }
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
      }
    }

    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
        return Fail("invalid number fraction");
      }
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
      }
    }

    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
        return Fail("invalid number exponent");
      }
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
      }
    }

    const std::string token = text_.substr(start, pos_ - start);
    char* endptr = nullptr;
    out = std::strtod(token.c_str(), &endptr);
    if (!endptr || *endptr != '\0') {
      return Fail("failed to parse number");
    }
    return true;
  }

  std::string text_;
  size_t pos_;
  std::string error_;
};

static inline const JsonValue* RequireField(const JsonValue& obj,
                                            const char* key,
                                            const JsonType expected_type,
                                            std::string& err) {
  if (obj.type != JsonType::kObject) {
    err = "RequireField called with non-object";
    return nullptr;
  }
  const auto it = obj.obj.find(std::string(key));
  if (it == obj.obj.end()) {
    err = std::string("missing required field '") + key + "'";
    return nullptr;
  }
  if (it->second.type != expected_type) {
    err = std::string("field '") + key + "' has wrong type";
    return nullptr;
  }
  return &it->second;
}

static inline bool JsonNumberToU32(const JsonValue& v, uint32_t& out) {
  if (v.type != JsonType::kNumber) {
    return false;
  }
  const double n = v.number;
  if (!(n >= 0.0) || n > (double)std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  const uint32_t u = (uint32_t)n;
  if ((double)u != n) {
    return false;
  }
  out = u;
  return true;
}

static inline bool ParseHexWord(const std::string& s, uint32_t& out) {
  if (s.size() != 10u) {
    return false;
  }
  if (!(s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))) {
    return false;
  }
  uint32_t v = 0u;
  for (size_t i = 2u; i < 10u; ++i) {
    const char c = s[i];
    uint32_t nibble = 0u;
    if (c >= '0' && c <= '9') {
      nibble = (uint32_t)(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      nibble = (uint32_t)(10 + (c - 'a'));
    } else if (c >= 'A' && c <= 'F') {
      nibble = (uint32_t)(10 + (c - 'A'));
    } else {
      return false;
    }
    v = (v << 4) | nibble;
  }
  out = v;
  return true;
}

struct MatrixRecord {
  std::string matrix_id;
  uint32_t weight_param_id = 0u;
  uint32_t inv_sw_param_id = 0u;
  uint32_t rows = 0u;
  uint32_t cols = 0u;
  uint32_t num_weights = 0u;
  uint32_t payload_words_2b = 0u;
  uint32_t last_word_valid_count = 0u;
  std::vector<uint32_t> inv_sw_fp32_bits;
  std::vector<uint32_t> payload_words;
  uint32_t count_neg = 0u;
  uint32_t count_zero = 0u;
  uint32_t count_pos = 0u;
  std::string status;
};

static bool ParseHexWordArray(const JsonValue& arr, std::vector<uint32_t>& out, std::string& err) {
  if (arr.type != JsonType::kArray) {
    err = "hex word field is not array";
    return false;
  }
  out.clear();
  out.reserve(arr.arr.size());
  for (uint32_t i = 0u; i < (uint32_t)arr.arr.size(); ++i) {
    const JsonValue& elem = arr.arr[i];
    if (elem.type != JsonType::kString) {
      err = "hex array element is not string";
      return false;
    }
    uint32_t word = 0u;
    if (!ParseHexWord(elem.str, word)) {
      err = std::string("hex word parse failed: ") + elem.str;
      return false;
    }
    out.push_back(word);
  }
  return true;
}

static bool ParseMatrixRecord(const JsonValue& obj, MatrixRecord& out, std::string& err) {
  const JsonValue* f_matrix_id = RequireField(obj, "matrix_id", JsonType::kString, err);
  const JsonValue* f_weight_param_id = RequireField(obj, "weight_param_id", JsonType::kNumber, err);
  const JsonValue* f_inv_sw_param_id = RequireField(obj, "inv_sw_param_id", JsonType::kNumber, err);
  const JsonValue* f_rows = RequireField(obj, "rows", JsonType::kNumber, err);
  const JsonValue* f_cols = RequireField(obj, "cols", JsonType::kNumber, err);
  const JsonValue* f_num_weights = RequireField(obj, "num_weights", JsonType::kNumber, err);
  const JsonValue* f_payload_words_2b = RequireField(obj, "payload_words_2b", JsonType::kNumber, err);
  const JsonValue* f_last_word_valid_count = RequireField(obj, "last_word_valid_count", JsonType::kNumber, err);
  const JsonValue* f_inv_sw_fp32_hex = RequireField(obj, "inv_sw_fp32_hex", JsonType::kArray, err);
  const JsonValue* f_payload_hex_words = RequireField(obj, "payload_hex_words", JsonType::kArray, err);
  const JsonValue* f_count_neg = RequireField(obj, "count_neg", JsonType::kNumber, err);
  const JsonValue* f_count_zero = RequireField(obj, "count_zero", JsonType::kNumber, err);
  const JsonValue* f_count_pos = RequireField(obj, "count_pos", JsonType::kNumber, err);
  const JsonValue* f_status = RequireField(obj, "status", JsonType::kString, err);

  if (!f_matrix_id || !f_weight_param_id || !f_inv_sw_param_id || !f_rows || !f_cols ||
      !f_num_weights || !f_payload_words_2b || !f_last_word_valid_count ||
      !f_inv_sw_fp32_hex || !f_payload_hex_words || !f_count_neg || !f_count_zero ||
      !f_count_pos || !f_status) {
    return false;
  }

  out.matrix_id = f_matrix_id->str;
  out.status = f_status->str;

  if (!JsonNumberToU32(*f_weight_param_id, out.weight_param_id) ||
      !JsonNumberToU32(*f_inv_sw_param_id, out.inv_sw_param_id) ||
      !JsonNumberToU32(*f_rows, out.rows) ||
      !JsonNumberToU32(*f_cols, out.cols) ||
      !JsonNumberToU32(*f_num_weights, out.num_weights) ||
      !JsonNumberToU32(*f_payload_words_2b, out.payload_words_2b) ||
      !JsonNumberToU32(*f_last_word_valid_count, out.last_word_valid_count) ||
      !JsonNumberToU32(*f_count_neg, out.count_neg) ||
      !JsonNumberToU32(*f_count_zero, out.count_zero) ||
      !JsonNumberToU32(*f_count_pos, out.count_pos)) {
    err = "numeric field parse failure";
    return false;
  }

  if (!ParseHexWordArray(*f_inv_sw_fp32_hex, out.inv_sw_fp32_bits, err)) {
    return false;
  }
  if (!ParseHexWordArray(*f_payload_hex_words, out.payload_words, err)) {
    return false;
  }

  return true;
}

static inline int32_t DecodeCodeToValue(const uint32_t code) {
  if (code == (uint32_t)TERNARY_CODE_NEG) {
    return -1;
  }
  if (code == (uint32_t)TERNARY_CODE_ZERO) {
    return 0;
  }
  if (code == (uint32_t)TERNARY_CODE_POS) {
    return 1;
  }
  return 99;
}

static inline bool SourceValueToCode(const double v, uint32_t& out_code) {
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

static inline uint32_t Fp32BitsFromDouble(const double x) {
  const float f = (float)x;
  uint32_t bits = 0u;
  std::memcpy(&bits, &f, sizeof(uint32_t));
  return bits;
}

static bool LookupWeightFp64(const WeightId id, const double*& out_ptr, uint32_t& out_numel) {
  switch (id) {
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_0_weight_numel;
      out_ptr = w_decoder_layers_0_self_attn_linears_0_weight;
      return true;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_0_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_0_s_w_numel;
      out_ptr = w_decoder_layers_0_self_attn_linears_0_s_w;
      return true;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_1_weight_numel;
      out_ptr = w_decoder_layers_0_self_attn_linears_1_weight;
      return true;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_1_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_1_s_w_numel;
      out_ptr = w_decoder_layers_0_self_attn_linears_1_s_w;
      return true;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_2_weight_numel;
      out_ptr = w_decoder_layers_0_self_attn_linears_2_weight;
      return true;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_2_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_2_s_w_numel;
      out_ptr = w_decoder_layers_0_self_attn_linears_2_s_w;
      return true;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_3_weight_numel;
      out_ptr = w_decoder_layers_0_self_attn_linears_3_weight;
      return true;
    case DECODER_LAYERS_0_SELF_ATTN_LINEARS_3_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_self_attn_linears_3_s_w_numel;
      out_ptr = w_decoder_layers_0_self_attn_linears_3_s_w;
      return true;
    case DECODER_LAYERS_0_FEED_FORWARD_W_1_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_1_weight_numel;
      out_ptr = w_decoder_layers_0_feed_forward_w_1_weight;
      return true;
    case DECODER_LAYERS_0_FEED_FORWARD_W_1_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_1_s_w_numel;
      out_ptr = w_decoder_layers_0_feed_forward_w_1_s_w;
      return true;
    case DECODER_LAYERS_0_FEED_FORWARD_W_2_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_2_weight_numel;
      out_ptr = w_decoder_layers_0_feed_forward_w_2_weight;
      return true;
    case DECODER_LAYERS_0_FEED_FORWARD_W_2_S_W:
      out_numel = (uint32_t)w_decoder_layers_0_feed_forward_w_2_s_w_numel;
      out_ptr = w_decoder_layers_0_feed_forward_w_2_s_w;
      return true;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_0_weight_numel;
      out_ptr = w_decoder_layers_1_self_attn_linears_0_weight;
      return true;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_0_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_0_s_w_numel;
      out_ptr = w_decoder_layers_1_self_attn_linears_0_s_w;
      return true;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_1_weight_numel;
      out_ptr = w_decoder_layers_1_self_attn_linears_1_weight;
      return true;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_1_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_1_s_w_numel;
      out_ptr = w_decoder_layers_1_self_attn_linears_1_s_w;
      return true;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_2_weight_numel;
      out_ptr = w_decoder_layers_1_self_attn_linears_2_weight;
      return true;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_2_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_2_s_w_numel;
      out_ptr = w_decoder_layers_1_self_attn_linears_2_s_w;
      return true;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_3_weight_numel;
      out_ptr = w_decoder_layers_1_self_attn_linears_3_weight;
      return true;
    case DECODER_LAYERS_1_SELF_ATTN_LINEARS_3_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_self_attn_linears_3_s_w_numel;
      out_ptr = w_decoder_layers_1_self_attn_linears_3_s_w;
      return true;
    case DECODER_LAYERS_1_FEED_FORWARD_W_1_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_1_weight_numel;
      out_ptr = w_decoder_layers_1_feed_forward_w_1_weight;
      return true;
    case DECODER_LAYERS_1_FEED_FORWARD_W_1_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_1_s_w_numel;
      out_ptr = w_decoder_layers_1_feed_forward_w_1_s_w;
      return true;
    case DECODER_LAYERS_1_FEED_FORWARD_W_2_WEIGHT:
      out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_2_weight_numel;
      out_ptr = w_decoder_layers_1_feed_forward_w_2_weight;
      return true;
    case DECODER_LAYERS_1_FEED_FORWARD_W_2_S_W:
      out_numel = (uint32_t)w_decoder_layers_1_feed_forward_w_2_s_w_numel;
      out_ptr = w_decoder_layers_1_feed_forward_w_2_s_w;
      return true;
    default:
      out_numel = 0u;
      out_ptr = nullptr;
      return false;
  }
}

static int FailContext(const char* reason,
                       const char* matrix_id,
                       const uint32_t weight_param_id,
                       const uint32_t inv_sw_param_id,
                       const int64_t idx,
                       const int64_t word,
                       const int64_t slot,
                       const int64_t out_idx,
                       const int64_t in_idx,
                       const int64_t expected,
                       const int64_t got) {
  std::fprintf(stderr,
               "[p11d][FAIL] matrix_id=%s weight_param_id=%u inv_sw_param_id=%u idx=%lld out=%lld in=%lld word=%lld slot=%lld expected=%lld got=%lld reason=%s\n",
               matrix_id ? matrix_id : "N/A",
               (unsigned)weight_param_id,
               (unsigned)inv_sw_param_id,
               (long long)idx,
               (long long)out_idx,
               (long long)in_idx,
               (long long)word,
               (long long)slot,
               (long long)expected,
               (long long)got,
               reason ? reason : "unknown");
  return 1;
}

static int FailGlobal(const char* reason) {
  std::fprintf(stderr, "[p11d][FAIL] reason=%s\n", reason ? reason : "unknown");
  return 1;
}

}  // namespace

int main() {
  const char* const json_path = "gen/ternary_p11c_export.json";

  std::ifstream ifs(json_path, std::ios::binary);
  if (!ifs.good()) {
    return FailGlobal("cannot open gen/ternary_p11c_export.json");
  }
  std::string json_text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  if (json_text.empty()) {
    return FailGlobal("empty JSON artifact");
  }

  JsonValue root;
  JsonParser parser(json_text);
  if (!parser.Parse(root)) {
    std::fprintf(stderr, "[p11d][FAIL] JSON parse error: %s\n", parser.error().c_str());
    return 1;
  }
  if (root.type != JsonType::kObject) {
    return FailGlobal("root is not object");
  }

  std::string err;
  const JsonValue* f_version = RequireField(root, "version", JsonType::kString, err);
  const JsonValue* f_format = RequireField(root, "format", JsonType::kString, err);
  const JsonValue* f_matrix_count = RequireField(root, "matrix_count", JsonType::kNumber, err);
  const JsonValue* f_matrices = RequireField(root, "matrices", JsonType::kArray, err);
  if (!f_version || !f_format || !f_matrix_count || !f_matrices) {
    std::fprintf(stderr, "[p11d][FAIL] %s\n", err.c_str());
    return 1;
  }

  uint32_t matrix_count = 0u;
  if (!JsonNumberToU32(*f_matrix_count, matrix_count)) {
    return FailGlobal("matrix_count is not valid u32");
  }

  if (matrix_count != (uint32_t)QUANT_LINEAR_MATRIX_COUNT) {
    return FailGlobal("matrix_count mismatch with QUANT_LINEAR_MATRIX_COUNT");
  }

  if ((uint32_t)f_matrices->arr.size() != matrix_count) {
    return FailGlobal("matrices array length != matrix_count");
  }

  std::unordered_map<std::string, MatrixRecord> parsed_by_matrix_id;
  parsed_by_matrix_id.reserve(matrix_count);

  for (uint32_t i = 0u; i < matrix_count; ++i) {
    const JsonValue& obj = f_matrices->arr[i];
    if (obj.type != JsonType::kObject) {
      return FailGlobal("matrix entry is not object");
    }

    MatrixRecord rec;
    if (!ParseMatrixRecord(obj, rec, err)) {
      std::fprintf(stderr, "[p11d][FAIL] matrix[%u] parse failed: %s\n", (unsigned)i, err.c_str());
      return 1;
    }

    if (parsed_by_matrix_id.find(rec.matrix_id) != parsed_by_matrix_id.end()) {
      return FailGlobal("duplicate matrix_id in matrices[]");
    }

    parsed_by_matrix_id.insert(std::make_pair(rec.matrix_id, rec));
  }

  for (uint32_t i = 0u; i < (uint32_t)QUANT_LINEAR_MATRIX_COUNT; ++i) {
    const QuantLinearMeta& meta = kQuantLinearMeta[i];
    const char* const matrix_name = quant_linear_matrix_id_name(meta.matrix_id);

    auto it = parsed_by_matrix_id.find(std::string(matrix_name));
    if (it == parsed_by_matrix_id.end()) {
      return FailContext("missing matrix record by matrix_id", matrix_name,
                         meta.weight_param_id, meta.inv_sw_param_id,
                         -1, -1, -1, -1, -1, -1, -1);
    }
    const MatrixRecord& rec = it->second;

    if (rec.status != "PASS") {
      return FailContext("matrix status is not PASS", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1, -1, -1);
    }

    if (rec.weight_param_id != meta.weight_param_id ||
        rec.inv_sw_param_id != meta.inv_sw_param_id ||
        rec.rows != meta.rows ||
        rec.cols != meta.cols ||
        rec.num_weights != meta.num_weights ||
        rec.payload_words_2b != meta.payload_words_2b ||
        rec.last_word_valid_count != meta.last_word_valid_count) {
      return FailContext("artifact metadata mismatch vs kQuantLinearMeta", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1, -1, -1);
    }

    if (rec.last_word_valid_count < 1u || rec.last_word_valid_count > 16u) {
      return FailContext("last_word_valid_count out of range 1..16", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1, -1, (int64_t)rec.last_word_valid_count);
    }

    const uint64_t rows_cols = (uint64_t)rec.rows * (uint64_t)rec.cols;
    if (rows_cols != (uint64_t)rec.num_weights) {
      return FailContext("rows*cols != num_weights", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1, (int64_t)rows_cols, (int64_t)rec.num_weights);
    }

    const uint32_t expected_payload_words = ternary_payload_words_2b(rec.num_weights);
    const uint32_t expected_last_word_valid = ternary_last_word_valid_count(rec.num_weights);
    if (rec.payload_words_2b != expected_payload_words) {
      return FailContext("payload_words_2b mismatch", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1,
                         (int64_t)expected_payload_words, (int64_t)rec.payload_words_2b);
    }
    if (rec.last_word_valid_count != expected_last_word_valid) {
      return FailContext("last_word_valid_count mismatch", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1,
                         (int64_t)expected_last_word_valid, (int64_t)rec.last_word_valid_count);
    }

    if ((uint32_t)rec.payload_words.size() != rec.payload_words_2b) {
      return FailContext("payload_hex_words length mismatch", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1,
                         (int64_t)rec.payload_words_2b, (int64_t)rec.payload_words.size());
    }

    WeightId weight_wid = WEIGHT_COUNT;
    if (!quant_linear_weight_param_id_to_weight_id(rec.weight_param_id, weight_wid)) {
      return FailContext("quant_linear_weight_param_id_to_weight_id failed", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1, -1, -1);
    }

    WeightId inv_sw_wid = WEIGHT_COUNT;
    if (!quant_linear_matrix_id_to_inv_sw_weight_id(meta.matrix_id, inv_sw_wid)) {
      return FailContext("quant_linear_matrix_id_to_inv_sw_weight_id failed", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1, -1, -1);
    }

    const double* weight_src = nullptr;
    const double* sw_src = nullptr;
    uint32_t weight_numel = 0u;
    uint32_t sw_numel = 0u;
    if (!LookupWeightFp64(weight_wid, weight_src, weight_numel) || !weight_src) {
      return FailContext("lookup weight source failed", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1, -1, -1);
    }
    if (!LookupWeightFp64(inv_sw_wid, sw_src, sw_numel) || !sw_src) {
      return FailContext("lookup inv_s_w source failed", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1, -1, -1);
    }

    if (weight_numel != rec.num_weights) {
      return FailContext("weight source numel mismatch", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1,
                         (int64_t)rec.num_weights, (int64_t)weight_numel);
    }

    if (sw_numel < 1u) {
      return FailContext("inv_s_w source numel < 1", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1, -1, (int64_t)sw_numel);
    }

    if ((uint32_t)rec.inv_sw_fp32_bits.size() != sw_numel) {
      return FailContext("inv_sw_fp32_hex length mismatch vs source", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1,
                         (int64_t)sw_numel, (int64_t)rec.inv_sw_fp32_bits.size());
    }

    uint32_t count_neg = 0u;
    uint32_t count_zero = 0u;
    uint32_t count_pos = 0u;

    for (uint32_t idx = 0u; idx < rec.num_weights; ++idx) {
      const uint32_t word_idx = idx >> 4;
      const uint32_t slot = idx & 15u;
      const uint32_t shift = slot << 1;
      const uint32_t code = (rec.payload_words[word_idx] >> shift) & 0x3u;
      const uint32_t out_idx = idx / rec.cols;
      const uint32_t in_idx = idx % rec.cols;

      if ((out_idx * rec.cols + in_idx) != idx || out_idx >= rec.rows || in_idx >= rec.cols) {
        return FailContext("logical W[out][in] reconstruction mismatch", rec.matrix_id.c_str(),
                           rec.weight_param_id, rec.inv_sw_param_id,
                           (int64_t)idx, (int64_t)word_idx, (int64_t)slot,
                           (int64_t)out_idx, (int64_t)in_idx,
                           (int64_t)idx, (int64_t)(out_idx * rec.cols + in_idx));
      }

      uint32_t expected_code = 0u;
      if (!SourceValueToCode(weight_src[idx], expected_code)) {
        return FailContext("illegal ternary source value", rec.matrix_id.c_str(),
                           rec.weight_param_id, rec.inv_sw_param_id,
                           (int64_t)idx, (int64_t)word_idx, (int64_t)slot,
                           (int64_t)out_idx, (int64_t)in_idx,
                           -1, (int64_t)weight_src[idx]);
      }

      if (code == (uint32_t)TERNARY_CODE_RSVD) {
        return FailContext("illegal_valid_code_10", rec.matrix_id.c_str(),
                           rec.weight_param_id, rec.inv_sw_param_id,
                           (int64_t)idx, (int64_t)word_idx, (int64_t)slot,
                           (int64_t)out_idx, (int64_t)in_idx,
                           (int64_t)expected_code, (int64_t)code);
      }

      if (code != expected_code) {
        return FailContext("decode_mismatch", rec.matrix_id.c_str(),
                           rec.weight_param_id, rec.inv_sw_param_id,
                           (int64_t)idx, (int64_t)word_idx, (int64_t)slot,
                           (int64_t)out_idx, (int64_t)in_idx,
                           (int64_t)expected_code, (int64_t)code);
      }

      if (code == (uint32_t)TERNARY_CODE_NEG) {
        ++count_neg;
      } else if (code == (uint32_t)TERNARY_CODE_ZERO) {
        ++count_zero;
      } else if (code == (uint32_t)TERNARY_CODE_POS) {
        ++count_pos;
      } else {
        return FailContext("decoded code is out of ternary domain", rec.matrix_id.c_str(),
                           rec.weight_param_id, rec.inv_sw_param_id,
                           (int64_t)idx, (int64_t)word_idx, (int64_t)slot,
                           (int64_t)out_idx, (int64_t)in_idx,
                           -1, (int64_t)code);
      }

      const int32_t expected_value = DecodeCodeToValue(expected_code);
      const int32_t got_value = DecodeCodeToValue(code);
      if (expected_value != got_value) {
        return FailContext("decoded value mismatch", rec.matrix_id.c_str(),
                           rec.weight_param_id, rec.inv_sw_param_id,
                           (int64_t)idx, (int64_t)word_idx, (int64_t)slot,
                           (int64_t)out_idx, (int64_t)in_idx,
                           (int64_t)expected_value, (int64_t)got_value);
      }
    }

    if (count_neg != rec.count_neg || count_zero != rec.count_zero || count_pos != rec.count_pos) {
      return FailContext("count_neg/count_zero/count_pos mismatch", rec.matrix_id.c_str(),
                         rec.weight_param_id, rec.inv_sw_param_id,
                         -1, -1, -1, -1, -1,
                         (int64_t)(rec.count_neg + rec.count_zero + rec.count_pos),
                         (int64_t)(count_neg + count_zero + count_pos));
    }

    if (rec.last_word_valid_count < 16u) {
      const uint32_t last_word = rec.payload_words[rec.payload_words_2b - 1u];
      for (uint32_t slot = rec.last_word_valid_count; slot < 16u; ++slot) {
        const uint32_t code = (last_word >> (slot << 1)) & 0x3u;
        if (code != (uint32_t)TERNARY_CODE_ZERO) {
          return FailContext("tail_padding_non_zero", rec.matrix_id.c_str(),
                             rec.weight_param_id, rec.inv_sw_param_id,
                             -1,
                             (int64_t)(rec.payload_words_2b - 1u),
                             (int64_t)slot,
                             -1, -1,
                             (int64_t)TERNARY_CODE_ZERO,
                             (int64_t)code);
        }
      }
    }

    for (uint32_t j = 0u; j < sw_numel; ++j) {
      const double s_w = sw_src[j];
      if (s_w == 0.0) {
        return FailContext("s_w==0 while checking inv_s_w binding", rec.matrix_id.c_str(),
                           rec.weight_param_id, rec.inv_sw_param_id,
                           (int64_t)j, -1, -1, -1, -1, -1, -1);
      }
      const uint32_t expected_bits = Fp32BitsFromDouble(1.0 / s_w);
      const uint32_t got_bits = rec.inv_sw_fp32_bits[j];
      if (expected_bits != got_bits) {
        return FailContext("inv_s_w_fp32_bit_mismatch", rec.matrix_id.c_str(),
                           rec.weight_param_id, rec.inv_sw_param_id,
                           (int64_t)j, -1, -1, -1, -1,
                           (int64_t)expected_bits, (int64_t)got_bits);
      }
    }

    std::printf("[p11d][PASS] matrix_id=%s rows=%u cols=%u num_weights=%u payload_words_2b=%u last_word_valid_count=%u counts(neg,zero,pos)=(%u,%u,%u)\n",
                rec.matrix_id.c_str(),
                (unsigned)rec.rows,
                (unsigned)rec.cols,
                (unsigned)rec.num_weights,
                (unsigned)rec.payload_words_2b,
                (unsigned)rec.last_word_valid_count,
                (unsigned)count_neg,
                (unsigned)count_zero,
                (unsigned)count_pos);
  }

  std::printf("[p11d][PASS] loader-side parser/checker preview complete\n");
  return 0;
}

#else
int main() { return 0; }
#endif
