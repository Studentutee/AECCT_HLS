#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "blocks/PreprocEmbedSPE.h"
#include "tb_fp16_branch_word16_common.h"
#include "AECCT_ac_ref/include/RefModel.h"

namespace {

static inline uint32_t f32_to_bits_local(const float x) {
  uint32_t bits = 0u;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}

static bool build_ref_input_fp32(const uint32_t sample_idx,
                                 std::vector<double>& ref_input_fp32) {
  std::vector<uint32_t> infer_words_u32;
  if (!fp16_branch_tb::build_infer_input_words_u32(sample_idx, infer_words_u32)) {
    return false;
  }
  ref_input_fp32.assign(infer_words_u32.size(), 0.0);
  for (uint32_t i = 0u; i < (uint32_t)infer_words_u32.size(); ++i) {
    const aecct::u32_t bits = (aecct::u32_t)infer_words_u32[i];
    ref_input_fp32[i] = (double)aecct::fp32_from_bits(bits).to_float();
  }
  return true;
}

static int run_one_sample(const uint32_t sample_idx) {
  static aecct::u16_t sram[sram_map::FP16_BASELINE_STORAGE_WORDS_TOTAL];
  for (uint32_t i = 0u; i < (uint32_t)sram_map::FP16_BASELINE_STORAGE_WORDS_TOTAL; ++i) {
    sram[i] = (aecct::u16_t)0u;
  }

  std::vector<uint16_t> param_word16;
  if (!fp16_branch_tb::seed_param_image_word16_into_sram(sram, param_word16)) {
    return 1;
  }

  std::vector<uint32_t> infer_words_host;
  if (!fp16_branch_tb::build_infer_input_words_u32(sample_idx, infer_words_host)) {
    return 1;
  }
  std::vector<aecct::u32_t> infer_words_u32(infer_words_host.size(), (aecct::u32_t)0u);
  for (uint32_t i = 0u; i < (uint32_t)infer_words_host.size(); ++i) {
    infer_words_u32[i] = (aecct::u32_t)infer_words_host[i];
  }

  aecct::PreprocCfg cfg;
  cfg.infer_in_words = (aecct::u32_t)aecct::PREPROC_IN_WORDS_EXPECTED;
  cfg.x_out_words = (aecct::u32_t)sram_map::FP16_BASELINE_X_WORK_WORDS_WORD16;
  aecct::PreprocEmbedSPEWord16(
      sram,
      cfg,
      (aecct::u32_t)0u,
      (aecct::u32_t)sram_map::FP16_BASELINE_X_WORK_BASE_WORD16,
      infer_words_u32.data(),
      (aecct::u32_t)sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_BASE_WORD16);

  std::vector<uint16_t> got_x_word16;
  fp16_branch_tb::extract_x_work_word16_from_word16_sram(sram, got_x_word16);

  std::vector<uint16_t> ref_stage_local_word16;
  if (!fp16_branch_tb::build_preproc_x_ref_word16(sample_idx, ref_stage_local_word16)) {
    return 1;
  }
  if (!fp16_branch_tb::compare_word16_vectors_exact(
          ref_stage_local_word16,
          got_x_word16,
          "PreprocWord16_stage_local_word16_exact",
          sram_map::FP16_BASELINE_X_WORK_BASE_WORD16)) {
    return 1;
  }

  std::vector<double> ref_input_fp32;
  if (!build_ref_input_fp32(sample_idx, ref_input_fp32)) {
    return 1;
  }
  const uint32_t tensor_words = (uint32_t)N_NODES * (uint32_t)D_MODEL;
  std::vector<double> ref_logits((uint32_t)EXP_LEN_OUT_LOGITS_WORDS, 0.0);
  std::vector<aecct_ref::bit1_t> ref_xpred((uint32_t)EXP_LEN_OUT_XPRED_WORDS);
  std::vector<double> ref_layer0_attn_input(tensor_words, 0.0);

  aecct_ref::RefModel ref_model;
  ref_model.set_run_config(aecct_ref::make_fp16_experiment_run_config());
  aecct_ref::RefModelIO ref_io;
  ref_io.input_y = nullptr;
  ref_io.input_y_fp32 = ref_input_fp32.data();
  ref_io.out_logits = ref_logits.data();
  ref_io.out_x_pred = ref_xpred.data();
  ref_io.out_layer0_attn_input = ref_layer0_attn_input.data();
  ref_io.out_layer0_attn_out = nullptr;
  ref_io.out_layer0_residual_add_dut_aligned_out = nullptr;
  ref_io.B = 1;
  ref_io.N = (int)EXP_LEN_OUT_XPRED_WORDS;
  ref_model.infer_step0(ref_io);

  for (uint32_t i = 0u; i < (uint32_t)got_x_word16.size(); ++i) {
    const uint16_t exp = fp16_branch_tb::fp16_lane_from_double_tb(ref_layer0_attn_input[i]);
    const uint16_t got = got_x_word16[i];
    if (exp != got) {
      std::fprintf(stderr,
                   "[fp16_preproc_u16_ref][FAIL] sample=%u first mismatch elem=%u token=%u d=%u exp=0x%04X got=0x%04X\n",
                   (unsigned)sample_idx,
                   (unsigned)i,
                   (unsigned)(i / (uint32_t)D_MODEL),
                   (unsigned)(i % (uint32_t)D_MODEL),
                   (unsigned)exp,
                   (unsigned)got);
      return 2;
    }
  }

  std::printf("[fp16_preproc_u16_ref][PASS] sample=%u x_work_word16=%u\n",
              (unsigned)sample_idx,
              (unsigned)got_x_word16.size());
  return 0;
}

}  // namespace

int main() {
  int overall = 0;
  for (uint32_t sample = 0u; sample < 3u; ++sample) {
    const int rc = run_one_sample(sample);
    if (rc != 0 && overall == 0) {
      overall = rc;
    }
  }
  if (overall == 0) {
    std::printf("PASS: tb_fp16_preproc_u16_trace_ref_compare\n");
    return 0;
  }
  return overall;
}

#endif
