#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "AecctProtocol.h"
#include "AecctTypes.h"
#include "Top.h"
#include "gen/SramMap.h"
#include "tb_fp16_branch_word16_common.h"

namespace {

struct Io {
  aecct::ctrl_ch_t ctrl_cmd;
  aecct::ctrl_ch_t ctrl_rsp;
  aecct::data_ch_t data_in;
  aecct::data_ch_t data_out;
};

static inline void tick(Io& io) { aecct::top(io.ctrl_cmd, io.ctrl_rsp, io.data_in, io.data_out); }

static inline void drive_cmd(Io& io, uint8_t op) {
  io.ctrl_cmd.write(aecct::pack_ctrl_cmd(op));
  tick(io);
}

static bool pop_rsp(Io& io, uint8_t& kind, uint8_t& payload) {
  aecct::u16_t w;
  if (!io.ctrl_rsp.nb_read(w)) {
    return false;
  }
  kind = aecct::unpack_ctrl_rsp_kind(w);
  payload = aecct::unpack_ctrl_rsp_payload(w);
  return true;
}

static bool expect_rsp(Io& io, uint8_t exp_kind, uint8_t exp_payload, const char* tag) {
  uint8_t kind = 0u;
  uint8_t payload = 0u;
  if (!pop_rsp(io, kind, payload)) {
    std::fprintf(stderr, "[fp16_top][FAIL] %s missing ctrl_rsp\n", tag);
    return false;
  }
  if (kind != exp_kind || payload != exp_payload) {
    std::fprintf(stderr,
                 "[fp16_top][FAIL] %s ctrl_rsp mismatch got(kind=%u,payload=%u) expect(kind=%u,payload=%u)\n",
                 tag,
                 (unsigned)kind,
                 (unsigned)payload,
                 (unsigned)exp_kind,
                 (unsigned)exp_payload);
    return false;
  }
  return true;
}

static bool expect_no_rsp(Io& io, const char* tag) {
  uint8_t kind = 0u;
  uint8_t payload = 0u;
  if (pop_rsp(io, kind, payload)) {
    std::fprintf(stderr,
                 "[fp16_top][FAIL] %s unexpected ctrl_rsp kind=%u payload=%u\n",
                 tag,
                 (unsigned)kind,
                 (unsigned)payload);
    return false;
  }
  return true;
}

static bool do_soft_reset(Io& io) {
  drive_cmd(io, (uint8_t)aecct::OP_SOFT_RESET);
  return expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET, "soft_reset");
}

static bool do_set_w_base(Io& io, uint32_t base_w) {
  io.data_in.write((aecct::u32_t)base_w);
  drive_cmd(io, (uint8_t)aecct::OP_SET_W_BASE);
  return expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_SET_W_BASE, "set_w_base");
}

static bool do_load_w_begin(Io& io) {
  drive_cmd(io, (uint8_t)aecct::OP_LOAD_W);
  return expect_rsp(io, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W, "load_w_begin");
}

static bool stream_param_words(Io& io, const std::vector<uint32_t>& words32, const uint32_t expected_words_w) {
  if (words32.size() != (size_t)expected_words_w) {
    std::fprintf(stderr,
                 "[fp16_top][FAIL] param u32 stream size mismatch got=%u expect=%u\n",
                 (unsigned)words32.size(),
                 (unsigned)expected_words_w);
    return false;
  }
  for (uint32_t i = 0u; i < (uint32_t)words32.size(); ++i) {
    io.data_in.write((aecct::u32_t)words32[i]);
    tick(io);
    if (i + 1u < (uint32_t)words32.size()) {
      if (!expect_no_rsp(io, "load_w_stream")) {
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

static bool read_mem(Io& io, uint32_t addr_w, uint32_t len_w, std::vector<uint32_t>& out) {
  out.clear();
  out.reserve(len_w);
  io.data_in.write((aecct::u32_t)addr_w);
  io.data_in.write((aecct::u32_t)len_w);
  drive_cmd(io, (uint8_t)aecct::OP_READ_MEM);
  for (uint32_t i = 0u; i < len_w; ++i) {
    aecct::u32_t w;
    if (!io.data_out.nb_read(w)) {
      std::fprintf(stderr, "[fp16_top][FAIL] READ_MEM missing data_out idx=%u\n", (unsigned)i);
      return false;
    }
    out.push_back((uint32_t)w.to_uint());
  }
  return expect_rsp(io, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_READ_MEM, "read_mem_done");
}

}  // namespace

int main() {
  if (!sram_map::fp16_baseline_default_param_stream_fits_w_region()) {
    std::fprintf(stderr, "[fp16_top][FAIL] baseline PARAM stream does not fit inside W_REGION\n");
    return 1;
  }

  std::vector<uint16_t> param_word16;
  if (!fp16_branch_tb::build_param_image_word16(param_word16)) {
    return 1;
  }

  std::vector<uint32_t> param_u32;
  fp16_branch_tb::pack_word16_to_u32_stream(param_word16, param_u32);
  const uint32_t param_base_w =
      storage_words_to_legacy_words_ceil(sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_BASE_WORD16);
  const uint32_t param_words_w =
      storage_words_to_legacy_words_ceil(sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_WORDS_WORD16);
  const uint32_t w_region_base_w =
      storage_words_to_legacy_words_ceil(sram_map::FP16_BASELINE_W_REGION_BASE_WORD16);
  const uint32_t w_region_words_w =
      storage_words_to_legacy_words_ceil(sram_map::FP16_BASELINE_W_REGION_WORDS_WORD16);

  Io io;
  if (!do_soft_reset(io)) {
    return 1;
  }
  if (!do_set_w_base(io, param_base_w)) {
    return 1;
  }
  if (!do_load_w_begin(io)) {
    return 1;
  }
  if (!stream_param_words(io, param_u32, param_words_w)) {
    return 1;
  }

  std::vector<uint32_t> got_param_u32;
  if (!read_mem(io, param_base_w, param_words_w, got_param_u32)) {
    return 1;
  }
  std::vector<uint16_t> got_param_word16;
  fp16_branch_tb::unpack_u32_to_word16_stream(got_param_u32,
                                               sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_WORDS_WORD16,
                                               got_param_word16);
  if (!fp16_branch_tb::compare_word16_vectors_exact(param_word16,
                                                    got_param_word16,
                                                    "Top_LOAD_W_READ_MEM_PARAM_word16_exact",
                                                    sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_BASE_WORD16)) {
    return 1;
  }

  std::vector<uint32_t> got_w_region_u32;
  if (!read_mem(io, w_region_base_w, w_region_words_w, got_w_region_u32)) {
    return 1;
  }
  std::vector<uint16_t> got_w_region_word16;
  fp16_branch_tb::unpack_u32_to_word16_stream(got_w_region_u32,
                                               sram_map::FP16_BASELINE_W_REGION_WORDS_WORD16,
                                               got_w_region_word16);

  std::vector<uint16_t> expected_w_region_word16(sram_map::FP16_BASELINE_W_REGION_WORDS_WORD16, 0u);
  for (uint32_t i = 0u; i < (uint32_t)param_word16.size(); ++i) {
    expected_w_region_word16[i] = param_word16[i];
  }
  if (!fp16_branch_tb::compare_word16_vectors_exact(expected_w_region_word16,
                                                    got_w_region_word16,
                                                    "Top_LOAD_W_READ_MEM_W_REGION_word16_exact",
                                                    sram_map::FP16_BASELINE_W_REGION_BASE_WORD16)) {
    return 1;
  }

  for (uint32_t pid = 0u; pid < (uint32_t)BIAS_COUNT + (uint32_t)WEIGHT_COUNT; ++pid) {
    Fp16BranchStorageDesc desc;
    if (!fp16_branch_param_storage_desc(pid, desc)) {
      std::fprintf(stderr, "[fp16_top][FAIL] storage desc lookup failed pid=%u\n", (unsigned)pid);
      return 1;
    }
    std::vector<uint16_t> exp;
    uint32_t logical = 0u;
    if (desc.is_bias) {
      if (!fp16_branch_tb::build_bias_words16((BiasId)pid, exp, logical)) {
        std::fprintf(stderr, "[fp16_top][FAIL] bias rebuild failed pid=%u\n", (unsigned)pid);
        return 1;
      }
    } else {
      if (!fp16_branch_tb::build_weight_words16((WeightId)(pid - (uint32_t)BIAS_COUNT), exp, logical)) {
        std::fprintf(stderr, "[fp16_top][FAIL] weight rebuild failed pid=%u\n", (unsigned)pid);
        return 1;
      }
    }
    std::vector<uint16_t> got(desc.words16, 0u);
    for (uint32_t j = 0u; j < desc.words16; ++j) {
      got[j] = got_param_word16[desc.offset_words16 + j];
    }
    if (!fp16_branch_tb::compare_word16_vectors_exact(exp,
                                                      got,
                                                      desc.key,
                                                      sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_BASE_WORD16 +
                                                          desc.offset_words16)) {
      return 1;
    }
  }

  std::printf("[fp16_top][SUMMARY] param_base_w=%u param_words_w=%u w_region_base_w=%u w_region_words_w=%u\n",
              (unsigned)param_base_w,
              (unsigned)param_words_w,
              (unsigned)w_region_base_w,
              (unsigned)w_region_words_w);
  std::printf("[fp16_top][SUMMARY] param_word16=%u w_region_word16=%u u32_stream=%u\n",
              (unsigned)sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_WORDS_WORD16,
              (unsigned)sram_map::FP16_BASELINE_W_REGION_WORDS_WORD16,
              (unsigned)param_u32.size());
  std::printf("PASS: tb_fp16_branch_top_loadw_readback_smoke\n");
  return 0;
}

#endif
