// P00-011AA: QKV formal LOAD_W bridge + READ_MEM roundtrip (local-only).
#ifndef __SYNTHESIS__

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "AecctProtocol.h"
#include "AecctTypes.h"
#include "Top.h"
#include "gen/SramMap.h"
#include "gen/include/WeightStreamOrder.h"

namespace {

struct MatrixTarget {
  const char* name;
  uint32_t matrix_id;
};

struct QkvJsonRecord {
  std::string matrix_id;
  uint32_t rows;
  uint32_t cols;
  uint32_t num_weights;
  uint32_t payload_words_2b;
  uint32_t last_word_valid_count;
  uint32_t weight_param_id;
  uint32_t inv_sw_param_id;
  std::vector<uint32_t> payload_words;
  std::vector<uint32_t> inv_sw_words;
};

struct SpanPlan {
  std::string matrix_id;
  uint32_t payload_offset;
  uint32_t payload_len;
  uint32_t inv_offset;
  uint32_t inv_len;
};

struct FormalIo {
  aecct::ctrl_ch_t ctrl_cmd;
  aecct::ctrl_ch_t ctrl_rsp;
  aecct::data_ch_t data_in;
  aecct::data_ch_t data_out;
};

static const MatrixTarget kTargets[] = {
    {"L0_WQ", (uint32_t)QLM_L0_WQ},
    {"L0_WK", (uint32_t)QLM_L0_WK},
    {"L0_WV", (uint32_t)QLM_L0_WV},
};
static const uint32_t kParamWords = (uint32_t)EXP_LEN_PARAM_WORDS;

static bool contains(const std::string& s, const char* x) { return s.find(x) != std::string::npos; }

static int fail(const std::string& why) {
  std::fprintf(stderr, "[p11aa][FAIL] %s\n", why.c_str());
  return 1;
}

static bool read_file(const char* p, std::string& out) {
  std::ifstream f(p, std::ios::in | std::ios::binary);
  if (!f.is_open()) return false;
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

static inline void skip_ws(const std::string& s, size_t& i) {
  while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
}

static bool parse_qstr(const std::string& s, size_t& i, std::string& out) {
  out.clear();
  if (i >= s.size() || s[i] != '"') return false;
  ++i;
  bool esc = false;
  while (i < s.size()) {
    char c = s[i++];
    if (esc) {
      out.push_back(c);
      esc = false;
      continue;
    }
    if (c == '\\') {
      esc = true;
      continue;
    }
    if (c == '"') return true;
    out.push_back(c);
  }
  return false;
}

static bool find_match(const std::string& s, size_t open_pos, char open_c, char close_c, size_t& out_close) {
  if (open_pos >= s.size() || s[open_pos] != open_c) return false;
  uint32_t d = 0u;
  bool in_str = false;
  bool esc = false;
  for (size_t i = open_pos; i < s.size(); ++i) {
    char c = s[i];
    if (in_str) {
      if (esc) {
        esc = false;
      } else if (c == '\\') {
        esc = true;
      } else if (c == '"') {
        in_str = false;
      }
      continue;
    }
    if (c == '"') {
      in_str = true;
      continue;
    }
    if (c == open_c) ++d;
    if (c == close_c) {
      if (d == 0u) return false;
      --d;
      if (d == 0u) {
        out_close = i;
        return true;
      }
    }
  }
  return false;
}

static bool field_str(const std::string& obj, const char* k, std::string& out, std::string& err) {
  std::string key = std::string("\"") + k + "\"";
  size_t kp = obj.find(key);
  if (kp == std::string::npos) {
    err = std::string("missing field: ") + k;
    return false;
  }
  size_t cp = obj.find(':', kp + key.size());
  if (cp == std::string::npos) {
    err = std::string("missing colon for field: ") + k;
    return false;
  }
  ++cp;
  skip_ws(obj, cp);
  if (!parse_qstr(obj, cp, out)) {
    err = std::string("invalid string field: ") + k;
    return false;
  }
  return true;
}

static bool field_u32(const std::string& obj, const char* k, uint32_t& out, std::string& err) {
  std::string key = std::string("\"") + k + "\"";
  size_t kp = obj.find(key);
  if (kp == std::string::npos) {
    err = std::string("missing field: ") + k;
    return false;
  }
  size_t cp = obj.find(':', kp + key.size());
  if (cp == std::string::npos) {
    err = std::string("missing colon for field: ") + k;
    return false;
  }
  ++cp;
  skip_ws(obj, cp);
  if (cp >= obj.size() || !std::isdigit((unsigned char)obj[cp])) {
    err = std::string("field is not u32: ") + k;
    return false;
  }
  uint64_t v = 0;
  while (cp < obj.size() && std::isdigit((unsigned char)obj[cp])) {
    v = v * 10u + (uint64_t)(obj[cp] - '0');
    if (v > 0xFFFFFFFFull) {
      err = std::string("field overflow u32: ") + k;
      return false;
    }
    ++cp;
  }
  out = (uint32_t)v;
  return true;
}

static bool parse_hex_u32(const std::string& s, uint32_t& out) {
  size_t i = 0u;
  if (s.size() >= 2u && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) i = 2u;
  if (i >= s.size()) return false;
  uint64_t v = 0u;
  for (; i < s.size(); ++i) {
    char c = s[i];
    uint32_t d = 0u;
    if (c >= '0' && c <= '9')
      d = (uint32_t)(c - '0');
    else if (c >= 'a' && c <= 'f')
      d = 10u + (uint32_t)(c - 'a');
    else if (c >= 'A' && c <= 'F')
      d = 10u + (uint32_t)(c - 'A');
    else
      return false;
    v = (v << 4u) | (uint64_t)d;
    if (v > 0xFFFFFFFFull) return false;
  }
  out = (uint32_t)v;
  return true;
}

static bool field_hex_arr(const std::string& obj,
                          const char* k,
                          bool& found,
                          std::vector<uint32_t>& out,
                          std::string& err) {
  found = false;
  out.clear();
  std::string key = std::string("\"") + k + "\"";
  size_t kp = obj.find(key);
  if (kp == std::string::npos) return true;
  found = true;
  size_t cp = obj.find(':', kp + key.size());
  if (cp == std::string::npos) {
    err = std::string("missing colon for field: ") + k;
    return false;
  }
  ++cp;
  skip_ws(obj, cp);
  if (cp >= obj.size() || obj[cp] != '[') {
    err = std::string("field is not array: ") + k;
    return false;
  }
  size_t end = 0u;
  if (!find_match(obj, cp, '[', ']', end)) {
    err = std::string("array close not found: ") + k;
    return false;
  }
  size_t i = cp + 1u;
  while (i < end) {
    skip_ws(obj, i);
    if (i >= end) break;
    if (obj[i] == ',') {
      ++i;
      continue;
    }
    if (obj[i] != '"') {
      err = std::string("array element not string: ") + k;
      return false;
    }
    std::string q;
    if (!parse_qstr(obj, i, q)) {
      err = std::string("array string parse failed: ") + k;
      return false;
    }
    uint32_t w = 0u;
    if (!parse_hex_u32(q, w)) {
      err = std::string("hex parse failed in array: ") + k;
      return false;
    }
    out.push_back(w);
  }
  return true;
}

static bool extract_matrix_objects(const std::string& json, std::vector<std::string>& objs, std::string& err) {
  objs.clear();
  size_t kp = json.find("\"matrices\"");
  if (kp == std::string::npos) {
    err = "missing key: matrices";
    return false;
  }
  size_t cp = json.find(':', kp + 10u);
  if (cp == std::string::npos) {
    err = "missing colon for matrices";
    return false;
  }
  ++cp;
  skip_ws(json, cp);
  if (cp >= json.size() || json[cp] != '[') {
    err = "matrices is not array";
    return false;
  }
  size_t end = 0u;
  if (!find_match(json, cp, '[', ']', end)) {
    err = "cannot close matrices array";
    return false;
  }
  size_t i = cp + 1u;
  while (i < end) {
    skip_ws(json, i);
    if (i >= end) break;
    if (json[i] == ',') {
      ++i;
      continue;
    }
    if (json[i] != '{') {
      err = "matrices element is not object";
      return false;
    }
    size_t oe = 0u;
    if (!find_match(json, i, '{', '}', oe)) {
      err = "cannot close matrix object";
      return false;
    }
    objs.push_back(json.substr(i, oe - i + 1u));
    i = oe + 1u;
  }
  if (objs.empty()) {
    err = "matrices array empty";
    return false;
  }
  return true;
}

static bool parse_qkv_record(const std::string& obj, QkvJsonRecord& r, std::string& err) {
  r = QkvJsonRecord();
  if (!field_str(obj, "matrix_id", r.matrix_id, err)) return false;
  if (!field_u32(obj, "rows", r.rows, err) || !field_u32(obj, "cols", r.cols, err) ||
      !field_u32(obj, "num_weights", r.num_weights, err) ||
      !field_u32(obj, "payload_words_2b", r.payload_words_2b, err) ||
      !field_u32(obj, "last_word_valid_count", r.last_word_valid_count, err) ||
      !field_u32(obj, "weight_param_id", r.weight_param_id, err) ||
      !field_u32(obj, "inv_sw_param_id", r.inv_sw_param_id, err)) {
    return false;
  }

  bool found = false;
  if (!field_hex_arr(obj, "payload_hex_words", found, r.payload_words, err)) return false;
  if (!found && !field_hex_arr(obj, "payload_words_hex", found, r.payload_words, err)) return false;
  if (!found || r.payload_words.empty()) {
    err = "schema limitation/mismatch: missing payload array";
    return false;
  }

  found = false;
  if (!field_hex_arr(obj, "inv_sw_fp32_hex", found, r.inv_sw_words, err)) return false;
  if (!found && !field_hex_arr(obj, "inv_sw_words_hex", found, r.inv_sw_words, err)) return false;
  if (!found || r.inv_sw_words.empty()) {
    err = "schema limitation/mismatch: missing inv_sw array";
    return false;
  }
  return true;
}

static bool validate_payload_semantics(const std::vector<uint32_t>& words,
                                       uint32_t num_weights,
                                       uint32_t last_valid,
                                       std::string& why) {
  if (num_weights == 0u || last_valid == 0u || last_valid > 16u) {
    why = "invalid semantic args";
    return false;
  }
  uint32_t exp_words = ternary_payload_words_2b(num_weights);
  if ((uint32_t)words.size() != exp_words) {
    why = "payload word count mismatch";
    return false;
  }
  if (last_valid != ternary_last_word_valid_count(num_weights)) {
    why = "last_word_valid_count mismatch";
    return false;
  }
  for (uint32_t w = 0; w < exp_words; ++w) {
    uint32_t x = words[w];
    uint32_t valid = (w + 1u == exp_words) ? last_valid : 16u;
    for (uint32_t i = 0; i < valid; ++i) {
      uint32_t c = (x >> (2u * i)) & 0x3u;
      if (c == 2u) {
        why = "illegal ternary code 10";
        return false;
      }
    }
    for (uint32_t i = valid; i < 16u; ++i) {
      uint32_t c = (x >> (2u * i)) & 0x3u;
      if (c != 0u) {
        why = "non-zero tail padding";
        return false;
      }
    }
  }
  why.clear();
  return true;
}

static bool collect_qkv_records(const std::vector<std::string>& objs,
                                std::vector<QkvJsonRecord>& out,
                                std::string& err) {
  out.clear();
  for (uint32_t t = 0u; t < (uint32_t)(sizeof(kTargets) / sizeof(kTargets[0])); ++t) {
    int32_t hit = -1;
    uint32_t count = 0u;
    for (uint32_t i = 0u; i < (uint32_t)objs.size(); ++i) {
      std::string id;
      std::string e;
      if (!field_str(objs[i], "matrix_id", id, e)) {
        err = std::string("matrix object malformed: ") + e;
        return false;
      }
      if (id == kTargets[t].name) {
        ++count;
        hit = (int32_t)i;
      }
    }
    if (count != 1u || hit < 0) {
      err = std::string("expected exactly one record for matrix_id=") + kTargets[t].name +
            ", found " + std::to_string(count);
      return false;
    }
    QkvJsonRecord r;
    if (!parse_qkv_record(objs[(uint32_t)hit], r, err)) return false;
    out.push_back(r);
  }
  return true;
}

static bool run_probe_semantic_cases(const std::vector<QkvJsonRecord>& records) {
  if (records.empty()) {
    std::fprintf(stderr, "[p11aa][FAIL] probe semantic input empty\n");
    return false;
  }
  std::printf("[p11aa][probe_semantic] probe-side validation only; Top formal loader remains transport-only for PARAM ingest.\n");

  for (size_t i = 0; i < records.size(); ++i) {
    std::string why;
    if (!validate_payload_semantics(records[i].payload_words,
                                    records[i].num_weights,
                                    records[i].last_word_valid_count,
                                    why)) {
      std::fprintf(stderr, "[p11aa][FAIL] baseline payload semantic check failed for matrix_id=%s: %s\n",
                   records[i].matrix_id.c_str(),
                   why.c_str());
      return false;
    }
  }

  const QkvJsonRecord& ref = records[0];
  if (ref.payload_words.empty() || ref.num_weights < 2u) {
    std::fprintf(stderr, "[p11aa][FAIL] probe semantic reference record invalid\n");
    return false;
  }

  {
    std::vector<uint32_t> w = ref.payload_words;
    w.back() = (w.back() & 0x3FFFFFFFu) | 0x40000000u;
    uint32_t n = ref.num_weights - 1u;
    uint32_t lv = ternary_last_word_valid_count(n);
    std::string why;
    bool ok = validate_payload_semantics(w, n, lv, why);
    if (ok || !contains(why, "non-zero tail padding")) {
      std::fprintf(stderr, "[p11aa][FAIL] expected non-zero tail padding rejection, got=%s\n", why.c_str());
      return false;
    }
    std::printf("[p11aa][probe_semantic][PASS] non-zero tail padding rejected by probe validator (pre-LOAD_W)\n");
  }
  {
    std::vector<uint32_t> w = ref.payload_words;
    w[0] = (w[0] & ~0x3u) | 0x2u;
    std::string why;
    bool ok = validate_payload_semantics(w, ref.num_weights, ref.last_word_valid_count, why);
    if (ok || !contains(why, "illegal ternary code 10")) {
      std::fprintf(stderr, "[p11aa][FAIL] expected illegal-code rejection, got=%s\n", why.c_str());
      return false;
    }
    std::printf("[p11aa][probe_semantic][PASS] illegal ternary code 10 rejected by probe validator (pre-LOAD_W)\n");
  }
  return true;
}

static bool build_expected_image(const std::vector<QkvJsonRecord>& records,
                                 std::vector<uint32_t>& out_words,
                                 std::vector<SpanPlan>& out_spans,
                                 std::string& err) {
  out_words.assign(kParamWords, 0u);
  out_spans.clear();

  for (uint32_t t = 0u; t < (uint32_t)(sizeof(kTargets) / sizeof(kTargets[0])); ++t) {
    const QkvJsonRecord* r = 0;
    for (size_t i = 0; i < records.size(); ++i) {
      if (records[i].matrix_id == kTargets[t].name) {
        r = &records[i];
        break;
      }
    }
    if (r == 0) {
      err = std::string("missing parsed record for matrix_id=") + kTargets[t].name;
      return false;
    }

    const QuantLinearMeta& m = kQuantLinearMeta[kTargets[t].matrix_id];
    if (m.matrix_id != kTargets[t].matrix_id || r->rows != m.rows || r->cols != m.cols || r->num_weights != m.num_weights ||
        r->payload_words_2b != m.payload_words_2b || r->last_word_valid_count != m.last_word_valid_count ||
        r->weight_param_id != m.weight_param_id || r->inv_sw_param_id != m.inv_sw_param_id) {
      err = std::string("JSON metadata mismatch against kQuantLinearMeta for ") + kTargets[t].name;
      return false;
    }
    if (r->num_weights != r->rows * r->cols) {
      err = std::string("num_weights mismatch rows*cols for ") + kTargets[t].name;
      return false;
    }
    if ((uint32_t)r->payload_words.size() != r->payload_words_2b ||
        r->payload_words_2b != ternary_payload_words_2b(r->num_weights) ||
        r->last_word_valid_count != ternary_last_word_valid_count(r->num_weights)) {
      err = std::string("payload expectation mismatch for ") + kTargets[t].name;
      return false;
    }

    WeightId wid0 = WEIGHT_COUNT, wid1 = WEIGHT_COUNT, inv_wid = WEIGHT_COUNT;
    if (!quant_linear_weight_param_id_to_weight_id(r->weight_param_id, wid0) ||
        !quant_linear_matrix_id_to_weight_id(m.matrix_id, wid1) || wid0 != wid1 ||
        !quant_linear_matrix_id_to_inv_sw_weight_id(m.matrix_id, inv_wid)) {
      err = std::string("mapping continuity failed for ") + kTargets[t].name;
      return false;
    }
    uint32_t inv_param = 0u;
    if (!weight_id_to_param_id(inv_wid, inv_param) || inv_param != r->inv_sw_param_id) {
      err = std::string("inv_sw mapping continuity failed for ") + kTargets[t].name;
      return false;
    }

    if (r->weight_param_id >= PARAM_COUNT || r->inv_sw_param_id >= PARAM_COUNT) {
      err = std::string("param id out of range for ") + kTargets[t].name;
      return false;
    }
    const ParamMeta& pm = kParamMeta[r->weight_param_id];
    const ParamMeta& im = kParamMeta[r->inv_sw_param_id];
    if (pm.len_w < r->payload_words_2b || pm.offset_w + r->payload_words_2b > kParamWords ||
        im.len_w == 0u || im.offset_w + im.len_w > kParamWords ||
        (uint32_t)r->inv_sw_words.size() > im.len_w) {
      err = std::string("span bounds failed for ") + kTargets[t].name;
      return false;
    }

    for (uint32_t i = 0; i < r->payload_words_2b; ++i) out_words[pm.offset_w + i] = r->payload_words[i];
    for (uint32_t i = 0; i < (uint32_t)r->inv_sw_words.size(); ++i) out_words[im.offset_w + i] = r->inv_sw_words[i];

    SpanPlan sp;
    sp.matrix_id = r->matrix_id;
    sp.payload_offset = pm.offset_w;
    sp.payload_len = r->payload_words_2b;
    sp.inv_offset = im.offset_w;
    sp.inv_len = im.len_w;
    out_spans.push_back(sp);
  }
  return true;
}

static inline void tick(FormalIo& io) { aecct::top(io.ctrl_cmd, io.ctrl_rsp, io.data_in, io.data_out); }

static bool pop_rsp(FormalIo& io, uint8_t& kind, uint8_t& payload) {
  aecct::u16_t w;
  if (!io.ctrl_rsp.nb_read(w)) return false;
  kind = aecct::unpack_ctrl_rsp_kind(w);
  payload = aecct::unpack_ctrl_rsp_payload(w);
  return true;
}

static bool expect_rsp(FormalIo& io, uint8_t ek, uint8_t ep, const char* tag) {
  uint8_t k = 0, p = 0;
  if (!pop_rsp(io, k, p)) {
    std::fprintf(stderr, "[p11aa][FAIL] %s: missing ctrl_rsp\n", tag);
    return false;
  }
  if (k != ek || p != ep) {
    std::fprintf(stderr,
                 "[p11aa][FAIL] %s: ctrl_rsp mismatch got(kind=%u,payload=%u) expect(kind=%u,payload=%u)\n",
                 tag,
                 (unsigned)k,
                 (unsigned)p,
                 (unsigned)ek,
                 (unsigned)ep);
    return false;
  }
  return true;
}

static bool expect_no_rsp(FormalIo& io, const char* tag) {
  uint8_t k = 0, p = 0;
  if (pop_rsp(io, k, p)) {
    std::fprintf(stderr, "[p11aa][FAIL] %s: unexpected ctrl_rsp kind=%u payload=%u\n", tag, (unsigned)k, (unsigned)p);
    return false;
  }
  return true;
}

static bool read_data(FormalIo& io, uint32_t& out, const char* tag) {
  aecct::u32_t w;
  if (!io.data_out.nb_read(w)) {
    std::fprintf(stderr, "[p11aa][FAIL] %s: missing data_out\n", tag);
    return false;
  }
  out = (uint32_t)w.to_uint();
  return true;
}

static void drive_cmd(FormalIo& io, uint8_t op) {
  io.ctrl_cmd.write(aecct::pack_ctrl_cmd(op));
  tick(io);
}

static bool do_soft_reset(FormalIo& io, const char* tag) {
  drive_cmd(io, (uint8_t)aecct::OP_SOFT_RESET);
  return expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET, tag);
}

static bool do_set_w_base(FormalIo& io, uint32_t base, uint8_t ek, uint8_t ep, const char* tag) {
  io.data_in.write((aecct::u32_t)base);
  drive_cmd(io, (uint8_t)aecct::OP_SET_W_BASE);
  return expect_rsp(io, ek, ep, tag);
}

static bool do_load_w(FormalIo& io, uint8_t ek, uint8_t ep, const char* tag) {
  drive_cmd(io, (uint8_t)aecct::OP_LOAD_W);
  return expect_rsp(io, ek, ep, tag);
}

static bool stream_param(FormalIo& io, const std::vector<uint32_t>& words) {
  if (words.size() != kParamWords) return false;
  for (uint32_t i = 0; i < kParamWords; ++i) {
    io.data_in.write((aecct::u32_t)words[i]);
    tick(io);
    if (i + 1u < kParamWords) {
      if (!expect_no_rsp(io, "stream_param")) return false;
    } else {
      if (!expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W, "stream_param_done")) return false;
    }
  }
  return true;
}

static bool read_mem(FormalIo& io, uint32_t addr, uint32_t len, std::vector<uint32_t>& out) {
  out.clear();
  out.reserve(len);
  io.data_in.write((aecct::u32_t)addr);
  io.data_in.write((aecct::u32_t)len);
  drive_cmd(io, (uint8_t)aecct::OP_READ_MEM);
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t w = 0u;
    if (!read_data(io, w, "read_mem")) return false;
    out.push_back(w);
  }
  return expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM, "read_mem_done");
}

static bool compare_span(FormalIo& io,
                         const std::vector<uint32_t>& exp,
                         uint32_t base,
                         const std::string& matrix_id,
                         const char* kind,
                         uint32_t rel_off,
                         uint32_t len) {
  std::vector<uint32_t> got;
  uint32_t off = base + rel_off;
  if (!read_mem(io, off, len, got)) return false;
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t e = exp[rel_off + i];
    uint32_t a = got[i];
    if (e != a) {
      std::printf(
          "[p11aa][SPAN][FAIL] matrix_id=%s kind=%s target_offset=%u compare_length=%u first_mismatch_index=%u expected=0x%08X actual=0x%08X\n",
          matrix_id.c_str(),
          kind,
          (unsigned)off,
          (unsigned)len,
          (unsigned)i,
          (unsigned)e,
          (unsigned)a);
      return false;
    }
  }
  std::printf("[p11aa][SPAN][PASS] matrix_id=%s kind=%s target_offset=%u compare_length=%u result=PASS\n",
              matrix_id.c_str(),
              kind,
              (unsigned)off,
              (unsigned)len);
  return true;
}

static bool run_formal_negatives(FormalIo& io, const std::vector<uint32_t>& words, uint32_t base) {
  std::printf("[p11aa][formal_negative] begin\n");
  if (!do_soft_reset(io, "neg_soft_reset_1")) return false;
  if (!do_load_w(io, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE, "neg_load_wo_base")) return false;
  std::printf("[p11aa][formal_negative][PASS] LOAD_W without SET_W_BASE rejected\n");

  if (!do_soft_reset(io, "neg_soft_reset_2")) return false;
  if (!do_set_w_base(io, base, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_SET_W_BASE, "neg_set_base")) return false;
  if (!do_load_w(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W, "neg_load_begin")) return false;

  for (uint32_t i = 0u; i < 11u; ++i) {
    io.data_in.write((aecct::u32_t)words[i]);
    tick(io);
    if (!expect_no_rsp(io, "neg_partial_stream")) return false;
  }
  tick(io);
  if (!expect_no_rsp(io, "neg_no_done_before_full_len")) return false;
  std::printf("[p11aa][formal_negative][PASS] no DONE before full expected length during incomplete LOAD_W\n");

  drive_cmd(io, (uint8_t)aecct::OP_CFG_BEGIN);
  if (!expect_rsp(io, (uint8_t)aecct::RSP_ERR, (uint8_t)aecct::ERR_BAD_STATE, "neg_followup_reject")) return false;
  std::printf("[p11aa][formal_negative][PASS] follow-up command rejected while incomplete LOAD_W active\n");

  if (!do_soft_reset(io, "neg_soft_reset_recover")) return false;
  if (!do_set_w_base(io, base, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_SET_W_BASE, "neg_recover_set_base")) return false;
  std::printf("[p11aa][formal_negative][PASS] reset cleanup recovered control path\n");
  return true;
}

static bool run_formal_roundtrip(FormalIo& io,
                                 const std::vector<uint32_t>& words,
                                 const std::vector<SpanPlan>& spans,
                                 uint32_t base) {
  std::printf("[p11aa][formal_roundtrip] begin\n");
  if (!do_soft_reset(io, "round_soft_reset")) return false;
  if (!do_set_w_base(io, base, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_SET_W_BASE, "round_set_base")) return false;
  if (!do_load_w(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W, "round_load_begin")) return false;
  if (!stream_param(io, words)) return false;

  for (size_t i = 0; i < spans.size(); ++i) {
    if (!compare_span(io, words, base, spans[i].matrix_id, "payload", spans[i].payload_offset, spans[i].payload_len))
      return false;
    if (!compare_span(io, words, base, spans[i].matrix_id, "inv_sw", spans[i].inv_offset, spans[i].inv_len)) return false;
  }
  return true;
}

}  // namespace

int main() {
  const char* json_path = "gen/ternary_p11c_export.json";
  const uint32_t base = (uint32_t)sram_map::W_REGION_BASE;

  std::string json;
  if (!read_file(json_path, json)) return fail(std::string("cannot read JSON artifact: ") + json_path);

  std::vector<std::string> objs;
  std::string err;
  if (!extract_matrix_objects(json, objs, err)) return fail(std::string("cannot parse matrices array: ") + err);

  std::vector<QkvJsonRecord> records;
  if (!collect_qkv_records(objs, records, err)) return fail(std::string("QKV record parse failed: ") + err);
  if (!run_probe_semantic_cases(records)) return 1;

  std::vector<uint32_t> expected;
  std::vector<SpanPlan> spans;
  if (!build_expected_image(records, expected, spans, err))
    return fail(std::string("expected PARAM image build failed: ") + err);

  FormalIo io;
  if (!run_formal_negatives(io, expected, base)) return 1;
  if (!run_formal_roundtrip(io, expected, spans, base)) return 1;

  std::printf("PASS: tb_qkv_formal_loadw_bridge_p11aa\n");
  return 0;
}

#endif  // __SYNTHESIS__
