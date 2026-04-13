#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <vector>

#include "Top.h"
#include "gen/SramMap.h"
#include "tb_fp16_branch_word16_common.h"

namespace {

static int run_one_sample(const uint32_t sample_idx) {
  static aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
  for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
    sram[i] = (aecct::u32_t)0u;
  }

  std::vector<uint16_t> param_word16;
  std::vector<uint32_t> param_u32;
  if (!fp16_branch_tb::seed_param_image_word16_into_sram(sram, param_word16, param_u32)) {
    return 1;
  }

  std::vector<uint32_t> infer_words_u32;
  if (!fp16_branch_tb::build_infer_input_words_u32(sample_idx, infer_words_u32)) {
    return 1;
  }
  fp16_branch_tb::seed_u32_words_into_sram(sram,
                                           (uint32_t)aecct::PREPROC_IN_BASE_WORD_DEFAULT,
                                           infer_words_u32);

  aecct::TopRegs regs;
  regs.clear();
  regs.w_base_set = true;
  regs.w_base_word =
      (aecct::u32_t)storage_words_to_legacy_words_ceil(sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_BASE_WORD16);
  regs.infer_ingest_contract.start = true;
  regs.infer_ingest_contract.done = true;
  regs.infer_ingest_contract.in_base_word = (aecct::u32_t)aecct::PREPROC_IN_BASE_WORD_DEFAULT;
  regs.infer_ingest_contract.len_words_expected = (aecct::u32_t)aecct::PREPROC_IN_WORDS_EXPECTED;
  regs.infer_ingest_contract.len_words_valid = (aecct::u32_t)aecct::PREPROC_IN_WORDS_EXPECTED;
  aecct::infer_refresh_preproc_ranges(regs.infer_ingest_contract,
                                      (uint32_t)aecct::PREPROC_X_OUT_WORDS_EXPECTED);

  aecct::run_preproc_block(regs, sram);

  std::vector<uint16_t> got_x_word16;
  fp16_branch_tb::extract_x_work_word16_from_u32_sram(sram, got_x_word16);

  std::vector<uint16_t> ref_x_word16;
  if (!fp16_branch_tb::build_preproc_x_ref_word16(sample_idx, ref_x_word16)) {
    return 1;
  }

  if (!fp16_branch_tb::compare_word16_vectors_exact(ref_x_word16,
                                                    got_x_word16,
                                                    "Top_run_preproc_block_X_WORK_word16_exact",
                                                    sram_map::FP16_BASELINE_X_WORK_BASE_WORD16)) {
    return 1;
  }

  std::printf("[fp16_preproc][PASS] sample=%u x_work_word16=%u packed_u32=%u\n",
              (unsigned)sample_idx,
              (unsigned)got_x_word16.size(),
              (unsigned)storage_words_to_legacy_words_ceil(sram_map::FP16_BASELINE_X_WORK_WORDS_WORD16));
  return 0;
}

}  // namespace

int main() {
  for (uint32_t sample = 0u; sample < 3u; ++sample) {
    if (run_one_sample(sample) != 0) {
      return 1;
    }
  }
  std::printf("PASS: tb_fp16_branch_preproc_xwork_smoke\n");
  return 0;
}

#endif
