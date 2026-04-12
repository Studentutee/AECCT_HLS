// Focused local-only checker:
// - Build the authoritative unified PARAM stream from weights.h / WeightStreamOrder.h.
// - Drive SET_W_BASE + LOAD_W through the real Top command path.
// - Read back PARAM_STREAM_DEFAULT / W_REGION with READ_MEM and compare exact words.
// - Keep scope on storage/load correctness, not math-path closure.
#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "AecctProtocol.h"
#include "AecctTypes.h"
#include "Top.h"
#include "gen/SramMap.h"
#include "tb/weights_streamer.h"

namespace {

struct Io {
  aecct::ctrl_ch_t ctrl_cmd;
  aecct::ctrl_ch_t ctrl_rsp;
  aecct::data_ch_t data_in;
  aecct::data_ch_t data_out;
};

static int fail(const std::string &why) {
  std::fprintf(stderr, "[loadw_smoke][FAIL] %s\n", why.c_str());
  return 1;
}

static inline void tick(Io &io) { aecct::top(io.ctrl_cmd, io.ctrl_rsp, io.data_in, io.data_out); }

static inline void drive_cmd(Io &io, uint8_t op) {
  io.ctrl_cmd.write(aecct::pack_ctrl_cmd(op));
  tick(io);
}

static bool pop_rsp(Io &io, uint8_t &kind, uint8_t &payload) {
  aecct::u16_t w;
  if (!io.ctrl_rsp.nb_read(w)) {
    return false;
  }
  kind = aecct::unpack_ctrl_rsp_kind(w);
  payload = aecct::unpack_ctrl_rsp_payload(w);
  return true;
}

static bool expect_rsp(Io &io, uint8_t exp_kind, uint8_t exp_payload, const char *tag) {
  uint8_t got_kind = 0u;
  uint8_t got_payload = 0u;
  if (!pop_rsp(io, got_kind, got_payload)) {
    std::fprintf(stderr, "[loadw_smoke][FAIL] %s: missing ctrl_rsp\n", tag);
    return false;
  }
  if (got_kind != exp_kind || got_payload != exp_payload) {
    std::fprintf(stderr,
                 "[loadw_smoke][FAIL] %s: ctrl_rsp mismatch got(kind=%u,payload=%u) expect(kind=%u,payload=%u)\n",
                 tag,
                 (unsigned)got_kind,
                 (unsigned)got_payload,
                 (unsigned)exp_kind,
                 (unsigned)exp_payload);
    return false;
  }
  return true;
}

static bool expect_no_rsp(Io &io, const char *tag) {
  uint8_t kind = 0u;
  uint8_t payload = 0u;
  if (pop_rsp(io, kind, payload)) {
    std::fprintf(stderr,
                 "[loadw_smoke][FAIL] %s: unexpected ctrl_rsp kind=%u payload=%u\n",
                 tag,
                 (unsigned)kind,
                 (unsigned)payload);
    return false;
  }
  return true;
}

static bool read_data(Io &io, uint32_t &out, const char *tag) {
  aecct::u32_t w;
  if (!io.data_out.nb_read(w)) {
    std::fprintf(stderr, "[loadw_smoke][FAIL] %s: missing data_out\n", tag);
    return false;
  }
  out = (uint32_t)w.to_uint();
  return true;
}

static bool do_soft_reset(Io &io, const char *tag) {
  drive_cmd(io, (uint8_t)aecct::OP_SOFT_RESET);
  return expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET, tag);
}

static bool do_set_w_base(Io &io, uint32_t base_w, const char *tag) {
  io.data_in.write((aecct::u32_t)base_w);
  drive_cmd(io, (uint8_t)aecct::OP_SET_W_BASE);
  return expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_SET_W_BASE, tag);
}

static bool do_load_w_begin(Io &io, const char *tag) {
  drive_cmd(io, (uint8_t)aecct::OP_LOAD_W);
  return expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W, tag);
}

static bool read_mem(Io &io, uint32_t addr_w, uint32_t len_w, std::vector<uint32_t> &out) {
  out.clear();
  out.reserve(len_w);
  io.data_in.write((aecct::u32_t)addr_w);
  io.data_in.write((aecct::u32_t)len_w);
  drive_cmd(io, (uint8_t)aecct::OP_READ_MEM);
  for (uint32_t i = 0u; i < len_w; ++i) {
    uint32_t got = 0u;
    if (!read_data(io, got, "read_mem")) {
      return false;
    }
    out.push_back(got);
  }
  return expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM, "read_mem_done");
}

static void append_zero_words(std::vector<uint32_t> &out, uint32_t n_words) {
  out.insert(out.end(), n_words, 0u);
}

static void append_fp32_words_from_fp64(std::vector<uint32_t> &out,
                                        const double *src,
                                        uint32_t src_numel,
                                        uint32_t stream_len_w) {
  for (uint32_t i = 0u; i < src_numel; ++i) {
    out.push_back(tb_fp32_bits_from_double(src[i]));
  }
  if (stream_len_w > src_numel) {
    append_zero_words(out, stream_len_w - src_numel);
  }
}

static bool append_inv_sw_words_from_fp64(std::vector<uint32_t> &out,
                                          const double *src,
                                          uint32_t src_numel,
                                          uint32_t stream_len_w,
                                          WeightId wid,
                                          std::string &err) {
  for (uint32_t i = 0u; i < src_numel; ++i) {
    const double s_w = src[i];
    if (s_w == 0.0) {
      char buf[192];
      std::snprintf(buf,
                    sizeof(buf),
                    "inv_s_w conversion failed (s_w==0), WeightId=%u, idx=%u",
                    (unsigned)wid,
                    (unsigned)i);
      err = buf;
      return false;
    }
    out.push_back(tb_fp32_bits_from_double(1.0 / s_w));
  }
  if (stream_len_w > src_numel) {
    append_zero_words(out, stream_len_w - src_numel);
  }
  return true;
}

static bool append_bitpack_words(std::vector<uint32_t> &out,
                                 const ac_int<1, false> *bits,
                                 uint32_t num_bits,
                                 uint32_t stream_len_w) {
  const uint32_t need_words = (num_bits + 31u) >> 5;
  uint32_t bit_idx = 0u;
  for (uint32_t word_i = 0u; word_i < need_words; ++word_i) {
    uint32_t word = 0u;
    for (uint32_t b = 0u; b < 32u; ++b) {
      if (bit_idx < num_bits) {
        const uint32_t bit = (uint32_t)(bits[bit_idx].to_int());
        word |= ((bit & 1u) << b);
      }
      ++bit_idx;
    }
    out.push_back(word);
  }
  if (stream_len_w > need_words) {
    append_zero_words(out, stream_len_w - need_words);
  }
  return true;
}

static bool build_expected_param_stream(std::vector<uint32_t> &out, std::string &err) {
  out.clear();
  out.reserve((uint32_t)EXP_LEN_PARAM_WORDS);

  for (uint32_t i = 0u; i < (uint32_t)BIAS_COUNT; ++i) {
    const BiasId bid = (BiasId)i;
    const TensorMeta meta = kBiasMeta[i];
    uint32_t numel = 0u;
    const double *ptr = tb_lookup_bias_fp64(bid, numel);
    if (!ptr || numel == 0u) {
      append_zero_words(out, meta.len_w);
    } else {
      append_fp32_words_from_fp64(out, ptr, numel, meta.len_w);
    }
  }

  for (uint32_t i = 0u; i < (uint32_t)WEIGHT_COUNT; ++i) {
    const WeightId wid = (WeightId)i;
    const TensorMeta meta = kWeightMeta[i];
    if (meta.dtype == 0u) {
      uint32_t numel = 0u;
      const double *ptr = tb_lookup_weight_fp64(wid, numel);
      if (!ptr || numel == 0u) {
        append_zero_words(out, meta.len_w);
      } else if (is_quant_linear_inv_sw_weight_slot(wid)) {
        if (!append_inv_sw_words_from_fp64(out, ptr, numel, meta.len_w, wid, err)) {
          return false;
        }
      } else {
        append_fp32_words_from_fp64(out, ptr, numel, meta.len_w);
      }
    } else {
      uint32_t num_bits = 0u;
      const ac_int<1, false> *bits = tb_lookup_weight_bits(wid, num_bits);
      if (!bits || num_bits == 0u) {
        append_zero_words(out, meta.len_w);
      } else if (!append_bitpack_words(out, bits, num_bits, meta.len_w)) {
        err = "bitpack append failed";
        return false;
      }
    }
  }

  if (out.size() != (size_t)EXP_LEN_PARAM_WORDS) {
    char buf[192];
    std::snprintf(buf,
                  sizeof(buf),
                  "PARAM word count mismatch built=%u expect=%u",
                  (unsigned)out.size(),
                  (unsigned)EXP_LEN_PARAM_WORDS);
    err = buf;
    return false;
  }
  return true;
}

static bool stream_param_words(Io &io, const std::vector<uint32_t> &words) {
  if (words.size() != (size_t)EXP_LEN_PARAM_WORDS) {
    return false;
  }
  for (uint32_t i = 0u; i < (uint32_t)words.size(); ++i) {
    io.data_in.write((aecct::u32_t)words[i]);
    tick(io);
    if (i + 1u < (uint32_t)words.size()) {
      if (!expect_no_rsp(io, "stream_param_words")) {
        return false;
      }
    } else {
      if (!expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W, "load_w_done")) {
        return false;
      }
    }
  }
  return true;
}

static bool compare_words(const std::vector<uint32_t> &exp,
                          const std::vector<uint32_t> &got,
                          const char *tag,
                          uint32_t base_w) {
  if (exp.size() != got.size()) {
    std::fprintf(stderr,
                 "[loadw_smoke][FAIL] %s: size mismatch exp=%u got=%u\n",
                 tag,
                 (unsigned)exp.size(),
                 (unsigned)got.size());
    return false;
  }
  for (uint32_t i = 0u; i < (uint32_t)exp.size(); ++i) {
    if (exp[i] != got[i]) {
      std::fprintf(stderr,
                   "[loadw_smoke][FAIL] %s: first mismatch addr_w=%u index=%u exp=0x%08X got=0x%08X\n",
                   tag,
                   (unsigned)(base_w + i),
                   (unsigned)i,
                   (unsigned)exp[i],
                   (unsigned)got[i]);
      return false;
    }
  }
  std::printf("[loadw_smoke][PASS] %s base_w=%u len=%u\n",
              tag,
              (unsigned)base_w,
              (unsigned)exp.size());
  return true;
}

}  // namespace

int main() {
  const sram_map::SectionDesc &param_desc = sram_map::section_desc(sram_map::SEC_PARAM_STREAM_DEFAULT);
  const sram_map::SectionDesc &w_region_desc = sram_map::section_desc(sram_map::SEC_W_REGION);
  const sram_map::SectionDesc &bias_desc = sram_map::section_desc(sram_map::SEC_BIAS_LEGACY);
  const sram_map::SectionDesc &weight_desc = sram_map::section_desc(sram_map::SEC_WEIGHT_LEGACY);

  if (!sram_map::default_param_stream_fits_w_region()) {
    return fail("default PARAM stream does not fit in W_REGION");
  }
  if (param_desc.base_w != (uint32_t)sram_map::PARAM_BASE_DEFAULT) {
    return fail("PARAM_STREAM_DEFAULT base mismatch against PARAM_BASE_DEFAULT");
  }

  std::string err;
  std::vector<uint32_t> expected_param;
  if (!build_expected_param_stream(expected_param, err)) {
    return fail(std::string("build_expected_param_stream failed: ") + err);
  }

  std::vector<uint32_t> expected_w_region((size_t)w_region_desc.words_w, 0u);
  const uint32_t param_rel = param_desc.base_w - w_region_desc.base_w;
  for (uint32_t i = 0u; i < (uint32_t)expected_param.size(); ++i) {
    expected_w_region[param_rel + i] = expected_param[i];
  }

  std::vector<uint32_t> expected_padding((size_t)(w_region_desc.words_w - (param_rel + param_desc.words_w)), 0u);

  Io io;
  if (!do_soft_reset(io, "soft_reset")) {
    return 1;
  }
  if (!do_set_w_base(io, param_desc.base_w, "set_w_base")) {
    return 1;
  }
  if (!do_load_w_begin(io, "load_w_begin")) {
    return 1;
  }
  if (!stream_param_words(io, expected_param)) {
    return 1;
  }

  std::vector<uint32_t> got_param;
  if (!read_mem(io, param_desc.base_w, param_desc.words_w, got_param)) {
    return 1;
  }
  if (!compare_words(expected_param, got_param, "PARAM_STREAM_DEFAULT_exact", param_desc.base_w)) {
    return 1;
  }

  std::vector<uint32_t> got_w_region;
  if (!read_mem(io, w_region_desc.base_w, w_region_desc.words_w, got_w_region)) {
    return 1;
  }
  if (!compare_words(expected_w_region, got_w_region, "W_REGION_exact_with_padding", w_region_desc.base_w)) {
    return 1;
  }

  std::vector<uint32_t> got_bias;
  if (!read_mem(io, bias_desc.base_w, bias_desc.payload_words_w, got_bias)) {
    return 1;
  }
  if (!compare_words(std::vector<uint32_t>(expected_w_region.begin(), expected_w_region.begin() + bias_desc.payload_words_w),
                     got_bias,
                     "BIAS_payload_exact",
                     bias_desc.base_w)) {
    return 1;
  }

  std::vector<uint32_t> got_weight;
  if (!read_mem(io, weight_desc.base_w, weight_desc.payload_words_w, got_weight)) {
    return 1;
  }
  const uint32_t weight_rel = weight_desc.base_w - w_region_desc.base_w;
  if (!compare_words(std::vector<uint32_t>(expected_w_region.begin() + weight_rel,
                                           expected_w_region.begin() + weight_rel + weight_desc.payload_words_w),
                     got_weight,
                     "WEIGHT_payload_exact",
                     weight_desc.base_w)) {
    return 1;
  }

  if (!expected_padding.empty()) {
    std::vector<uint32_t> got_padding;
    const uint32_t padding_base = param_desc.base_w + param_desc.words_w;
    if (!read_mem(io, padding_base, (uint32_t)expected_padding.size(), got_padding)) {
      return 1;
    }
    if (!compare_words(expected_padding, got_padding, "W_REGION_tail_padding_zero", padding_base)) {
      return 1;
    }
  }

  std::printf("[loadw_smoke] param_words=%u w_region_words=%u padding_words=%u\n",
              (unsigned)param_desc.words_w,
              (unsigned)w_region_desc.words_w,
              (unsigned)expected_padding.size());
  std::printf("PASS: tb_loadw_param_stream_readmem_smoke\n");
  return 0;
}

#endif  // __SYNTHESIS__
