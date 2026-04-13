#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "Top.h"
#include "blocks/TransformerLayer.h"
#include "LayerParamBringup.h"
#include "LayerScratchDesc.h"
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

static bool compare_preproc_x_work(const aecct::u32_t* sram,
                                   const std::vector<double>& ref_layer0_attn_input,
                                   uint32_t& first_idx,
                                   uint16_t& exp_lane,
                                   uint16_t& got_lane) {
  std::vector<uint16_t> got_words16;
  fp16_branch_tb::extract_x_work_word16_from_u32_sram(sram, got_words16);
  if (got_words16.size() != ref_layer0_attn_input.size()) {
    std::fprintf(stderr,
                 "[fp16_trace_chain][FAIL] preproc size mismatch got=%u ref=%u\n",
                 (unsigned)got_words16.size(),
                 (unsigned)ref_layer0_attn_input.size());
    return false;
  }
  for (uint32_t i = 0u; i < (uint32_t)got_words16.size(); ++i) {
    const uint16_t exp = fp16_branch_tb::fp16_lane_from_double_tb(ref_layer0_attn_input[i]);
    const uint16_t got = got_words16[i];
    if (exp != got) {
      first_idx = i;
      exp_lane = exp;
      got_lane = got;
      return false;
    }
  }
  return true;
}

static bool compare_fp32_shadow_against_ref(const char* tag,
                                            const std::vector<double>& ref_vec,
                                            uint32_t (*peek_word)(aecct::u32_t),
                                            uint32_t& first_idx,
                                            uint32_t& exp_bits,
                                            uint32_t& got_bits) {
  for (uint32_t i = 0u; i < (uint32_t)ref_vec.size(); ++i) {
    const uint32_t exp = f32_to_bits_local((float)ref_vec[i]);
    const uint32_t got = peek_word((aecct::u32_t)i);
    if (exp != got) {
      first_idx = i;
      exp_bits = exp;
      got_bits = got;
      std::fprintf(stderr,
                   "[fp16_trace_chain][FAIL] %s first mismatch idx=%u exp=0x%08X got=0x%08X\n",
                   tag,
                   (unsigned)i,
                   (unsigned)exp,
                   (unsigned)got);
      return false;
    }
  }
  return true;
}

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
  fp16_branch_tb::seed_u32_words_into_sram(
      sram,
      (uint32_t)aecct::PREPROC_IN_BASE_WORD_DEFAULT,
      infer_words_u32);

  std::vector<double> ref_input_fp32;
  if (!build_ref_input_fp32(sample_idx, ref_input_fp32)) {
    return 1;
  }

  const uint32_t tensor_words = (uint32_t)N_NODES * (uint32_t)D_MODEL;
  std::vector<double> ref_logits((uint32_t)EXP_LEN_OUT_LOGITS_WORDS, 0.0);
  std::vector<aecct_ref::bit1_t> ref_xpred((uint32_t)EXP_LEN_OUT_XPRED_WORDS);
  std::vector<double> ref_layer0_attn_input(tensor_words, 0.0);
  std::vector<double> ref_layer0_attn_out(tensor_words, 0.0);
  std::vector<double> ref_layer0_residual_add_dut_aligned_out(tensor_words, 0.0);

  aecct_ref::RefModel ref_model;
  ref_model.set_run_config(aecct_ref::make_fp16_experiment_run_config());
  aecct_ref::RefModelIO ref_io;
  ref_io.input_y = nullptr;
  ref_io.input_y_fp32 = ref_input_fp32.data();
  ref_io.out_logits = ref_logits.data();
  ref_io.out_x_pred = ref_xpred.data();
  ref_io.out_layer0_attn_input = ref_layer0_attn_input.data();
  ref_io.out_layer0_attn_out = ref_layer0_attn_out.data();
  ref_io.out_layer0_residual_add_dut_aligned_out = ref_layer0_residual_add_dut_aligned_out.data();
  ref_io.B = 1;
  ref_io.N = (int)EXP_LEN_OUT_XPRED_WORDS;
  ref_model.infer_step0(ref_io);

  aecct::TopRegs regs;
  regs.clear();
  regs.w_base_set = true;
  regs.w_base_word =
      (aecct::u32_t)storage_words_to_legacy_words_ceil(
          sram_map::FP16_BASELINE_PARAM_STREAM_DEFAULT_BASE_WORD16);
  regs.infer_ingest_contract.start = true;
  regs.infer_ingest_contract.done = true;
  regs.infer_ingest_contract.in_base_word = (aecct::u32_t)aecct::PREPROC_IN_BASE_WORD_DEFAULT;
  regs.infer_ingest_contract.len_words_expected = (aecct::u32_t)aecct::PREPROC_IN_WORDS_EXPECTED;
  regs.infer_ingest_contract.len_words_valid = (aecct::u32_t)aecct::PREPROC_IN_WORDS_EXPECTED;
  regs.cfg_n_layers = (aecct::u32_t)1u;
  aecct::infer_refresh_preproc_ranges(
      regs.infer_ingest_contract,
      (uint32_t)aecct::PREPROC_X_OUT_WORDS_EXPECTED);

  aecct::run_preproc_block(regs, sram);

  uint32_t first_idx16 = 0u;
  uint16_t exp_lane = 0u;
  uint16_t got_lane = 0u;
  if (!compare_preproc_x_work(sram,
                              ref_layer0_attn_input,
                              first_idx16,
                              exp_lane,
                              got_lane)) {
    std::fprintf(stderr,
                 "[fp16_trace_chain][FAIL] sample=%u preproc first mismatch elem=%u token=%u d=%u exp=0x%04X got=0x%04X\n",
                 (unsigned)sample_idx,
                 (unsigned)first_idx16,
                 (unsigned)(first_idx16 / (uint32_t)D_MODEL),
                 (unsigned)(first_idx16 % (uint32_t)D_MODEL),
                 (unsigned)exp_lane,
                 (unsigned)got_lane);
    return 1;
  }

  aecct::transformer_layer_debug_clear_layer1_stage_valid();
  const aecct::CfgRegs cfg = aecct::build_layer_cfg(regs);
  const aecct::LayerScratch sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
  const aecct::LayerParamBase pb = aecct::make_layer_param_base(regs.w_base_word, (aecct::u32_t)0u);
  aecct::TransformerLayer(
      sram,
      cfg,
      (aecct::u32_t)0u,
      (aecct::u32_t)aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT,
      (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
      sc,
      pb);

  if (!aecct::transformer_layer_debug_layer0_attn_out_writeback_valid()) {
    std::fprintf(stderr,
                 "[fp16_trace_chain][FAIL] sample=%u layer0 attn out debug shadow missing\n",
                 (unsigned)sample_idx);
    return 1;
  }
  if (!aecct::transformer_layer_debug_layer0_residual0_add_out_valid()) {
    std::fprintf(stderr,
                 "[fp16_trace_chain][FAIL] sample=%u layer0 residual add debug shadow missing\n",
                 (unsigned)sample_idx);
    return 1;
  }

  uint32_t first_idx = 0u;
  uint32_t exp_bits = 0u;
  uint32_t got_bits = 0u;
  const bool attn_ok = compare_fp32_shadow_against_ref(
      "layer0_attn_out",
      ref_layer0_attn_out,
      [](aecct::u32_t idx) {
        return (uint32_t)aecct::transformer_layer_debug_peek_layer0_attn_out_writeback_word(idx).to_uint();
      },
      first_idx,
      exp_bits,
      got_bits);

  const bool residual_ok = compare_fp32_shadow_against_ref(
      "layer0_residual_add_dut_aligned_out",
      ref_layer0_residual_add_dut_aligned_out,
      [](aecct::u32_t idx) {
        return (uint32_t)aecct::transformer_layer_debug_peek_layer0_residual0_add_out_word(idx).to_uint();
      },
      first_idx,
      exp_bits,
      got_bits);

  std::printf("[fp16_trace_chain][SUMMARY] sample=%u preproc_exact=1 attn_out_exact=%u residual_exact=%u\n",
              (unsigned)sample_idx,
              attn_ok ? 1u : 0u,
              residual_ok ? 1u : 0u);

  return (attn_ok && residual_ok) ? 0 : 2;
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
    std::printf("PASS: tb_fp16_trace_chain_layer0_ref_compare\n");
    return 0;
  }
  return overall;
}

#endif
