#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <vector>

#include "Top.h"
#include "blocks/TernaryLiveQkvLeafKernel.h"
#include "gen/SramMap.h"
#include "tb_fp16_branch_word16_common.h"

namespace {

static bool build_matrix_payload_and_inv_fp32(
    const uint32_t matrix_id,
    std::vector<aecct::u32_t>& payload_words_u32,
    aecct::u32_t& inv_sw_bits_out) {
  WeightId payload_wid;
  WeightId inv_wid;
  if (!quant_linear_matrix_id_to_weight_id(matrix_id, payload_wid) ||
      !quant_linear_matrix_id_to_inv_sw_weight_id(matrix_id, inv_wid)) {
    std::fprintf(stderr,
                 "[fp16_q][FAIL] matrix id mapping failed matrix=%u (%s)\n",
                 (unsigned)matrix_id,
                 quant_linear_matrix_id_name(matrix_id));
    return false;
  }

  std::vector<uint16_t> payload_words16;
  uint32_t payload_logical = 0u;
  if (!fp16_branch_tb::build_weight_words16(payload_wid, payload_words16, payload_logical)) {
    std::fprintf(stderr,
                 "[fp16_q][FAIL] payload word16 build failed matrix=%u (%s)\n",
                 (unsigned)matrix_id,
                 quant_linear_matrix_id_name(matrix_id));
    return false;
  }

  std::vector<uint32_t> payload_u32_tmp;
  fp16_branch_tb::pack_word16_to_u32_stream(payload_words16, payload_u32_tmp);
  payload_words_u32.assign(payload_u32_tmp.size(), (aecct::u32_t)0u);
  for (uint32_t i = 0u; i < (uint32_t)payload_u32_tmp.size(); ++i) {
    payload_words_u32[i] = (aecct::u32_t)payload_u32_tmp[i];
  }

  std::vector<uint16_t> inv_words16;
  uint32_t inv_logical = 0u;
  if (!fp16_branch_tb::build_weight_words16(inv_wid, inv_words16, inv_logical) || inv_words16.empty()) {
    std::fprintf(stderr,
                 "[fp16_q][FAIL] inv_s_w word16 build failed matrix=%u (%s)\n",
                 (unsigned)matrix_id,
                 quant_linear_matrix_id_name(matrix_id));
    return false;
  }
  inv_sw_bits_out = aecct::fp32_bits_from_fp16_lane((aecct::u16_t)inv_words16[0]);
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
  aecct::infer_refresh_preproc_ranges(
      regs.infer_ingest_contract,
      (uint32_t)aecct::PREPROC_X_OUT_WORDS_EXPECTED);

  aecct::run_preproc_block(regs, sram);

  std::vector<uint16_t> x_work_word16;
  fp16_branch_tb::extract_x_work_word16_from_u32_sram(sram, x_work_word16);

  std::vector<aecct::u32_t> wq_payload_words;
  aecct::u32_t wq_inv_sw_bits = (aecct::u32_t)0u;
  if (!build_matrix_payload_and_inv_fp32((uint32_t)QLM_L0_WQ, wq_payload_words, wq_inv_sw_bits)) {
    return 1;
  }

  aecct::AttnCfg attn_cfg;
  attn_cfg.token_count = (aecct::u32_t)aecct::ATTN_TOKEN_COUNT;
  attn_cfg.d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
  attn_cfg.n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
  attn_cfg.d_head = (aecct::u32_t)aecct::ATTN_D_HEAD;
  aecct::AttnScratch sc = aecct::default_attn_scratch();
  bool fallback_taken = true;
  if (!aecct::attn_phasea_top_managed_q_mainline(
          sram,
          regs.w_base_word,
          (aecct::u32_t)aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT,
          attn_cfg,
          sc,
          fallback_taken)) {
    std::fprintf(stderr, "[fp16_q][FAIL] q mainline returned false sample=%u\n", (unsigned)sample_idx);
    return 1;
  }
  if (fallback_taken) {
    std::fprintf(stderr, "[fp16_q][FAIL] q mainline took fallback sample=%u\n", (unsigned)sample_idx);
    return 1;
  }

  uint32_t q_mismatch = 0u;
  const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
  const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
  for (uint32_t t = 0u; t < token_count; ++t) {
    aecct::u32_t x_row[aecct::kTernaryLiveL0WqCols];
    for (uint32_t d = 0u; d < d_model; ++d) {
      x_row[d] = aecct::fp32_bits_from_fp16_lane(
          (aecct::u16_t)x_work_word16[t * d_model + d]);
    }
    aecct::u32_t q_out[aecct::kTernaryLiveL0WqRows];
    aecct::u32_t q_out_act_q[aecct::kTernaryLiveL0WqRows];
    aecct::u32_t q_out_inv_sw_bits = (aecct::u32_t)0u;
    if (!aecct::ternary_live_l0_wq_materialize_row_kernel_split(
            x_row,
            wq_payload_words.data(),
            wq_inv_sw_bits,
            q_out,
            q_out_act_q,
            q_out_inv_sw_bits)) {
      std::fprintf(stderr,
                   "[fp16_q][FAIL] reference kernel failed sample=%u token=%u\n",
                   (unsigned)sample_idx,
                   (unsigned)t);
      return 1;
    }

    const uint32_t row_q_base = (uint32_t)sc.q_base_word.to_uint() + t * d_model;
    const uint32_t row_q_act_q_base = (uint32_t)sc.q_act_q_base_word.to_uint() + t * d_model;
    for (uint32_t d = 0u; d < d_model; ++d) {
      const uint32_t got_q = (uint32_t)sram[row_q_base + d].to_uint();
      const uint32_t got_q_act_q = (uint32_t)sram[row_q_act_q_base + d].to_uint();
      const uint32_t exp_q = (uint32_t)q_out[d].to_uint();
      const uint32_t exp_q_act_q = (uint32_t)q_out[d].to_uint();
      if (got_q != exp_q || got_q_act_q != exp_q_act_q) {
        if (q_mismatch == 0u) {
          std::fprintf(stderr,
                       "[fp16_q][FAIL] first mismatch sample=%u token=%u d=%u got_q=0x%08X exp_q=0x%08X got_q_act_q=0x%08X exp_q_act_q=0x%08X\n",
                       (unsigned)sample_idx,
                       (unsigned)t,
                       (unsigned)d,
                       (unsigned)got_q,
                       (unsigned)exp_q,
                       (unsigned)got_q_act_q,
                       (unsigned)exp_q_act_q);
        }
        ++q_mismatch;
      }
    }
    const uint32_t got_q_sx = (uint32_t)sram[(uint32_t)sc.q_sx_base_word.to_uint()].to_uint();
    const uint32_t exp_q_sx = (uint32_t)q_out_inv_sw_bits.to_uint();
    if (got_q_sx != exp_q_sx) {
      if (q_mismatch == 0u) {
        std::fprintf(stderr,
                     "[fp16_q][FAIL] q_sx mismatch sample=%u token=%u got=0x%08X exp=0x%08X\n",
                     (unsigned)sample_idx,
                     (unsigned)t,
                     (unsigned)got_q_sx,
                     (unsigned)exp_q_sx);
      }
      ++q_mismatch;
    }
  }

  if (q_mismatch != 0u) {
    std::fprintf(stderr,
                 "[fp16_q][FAIL] total mismatches sample=%u count=%u\n",
                 (unsigned)sample_idx,
                 (unsigned)q_mismatch);
    return 1;
  }

  std::printf("[fp16_q][PASS] sample=%u tokens=%u d_model=%u q_mismatch=0\n",
              (unsigned)sample_idx,
              (unsigned)token_count,
              (unsigned)d_model);
  return 0;
}

}  // namespace

int main() {
  for (uint32_t sample = 0u; sample < 3u; ++sample) {
    if (run_one_sample(sample) != 0) {
      return 1;
    }
  }
  std::printf("PASS: tb_fp16_branch_q_compare_smoke\n");
  return 0;
}

#endif
