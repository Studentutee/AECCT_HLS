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

  aecct::AttnCfg attn_cfg;
  attn_cfg.token_count = (aecct::u32_t)aecct::ATTN_TOKEN_COUNT;
  attn_cfg.d_model = (aecct::u32_t)aecct::ATTN_D_MODEL;
  attn_cfg.n_heads = (aecct::u32_t)aecct::ATTN_N_HEADS;
  attn_cfg.d_head = (aecct::u32_t)aecct::ATTN_D_HEAD;
  aecct::AttnScratch sc = aecct::default_attn_scratch();

  bool kv_fallback = true;
  if (!aecct::attn_phasea_top_managed_kv_mainline(
          sram,
          regs.w_base_word,
          (aecct::u32_t)aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT,
          attn_cfg,
          sc,
          kv_fallback) || kv_fallback) {
    std::fprintf(stderr, "[fp16_score][FAIL] kv mainline failed sample=%u fallback=%u\n",
                 (unsigned)sample_idx,
                 (unsigned)(kv_fallback ? 1u : 0u));
    return 1;
  }

  bool q_fallback = true;
  if (!aecct::attn_phasea_top_managed_q_mainline(
          sram,
          regs.w_base_word,
          (aecct::u32_t)aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT,
          attn_cfg,
          sc,
          q_fallback) || q_fallback) {
    std::fprintf(stderr, "[fp16_score][FAIL] q mainline failed sample=%u fallback=%u\n",
                 (unsigned)sample_idx,
                 (unsigned)(q_fallback ? 1u : 0u));
    return 1;
  }

  const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
  const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
  const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
  const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
  const aecct::quant_acc_t inv_sqrt_d_head = aecct::attn_phaseb_inv_sqrt_d_head(d_head);

  uint32_t total_mismatch = 0u;
  for (uint32_t token = 0u; token < token_count; ++token) {
    bool score_fallback = true;
    if (!aecct::attn_phaseb_top_managed_qk_score_mainline(
            sram,
            attn_cfg,
            sc,
            (aecct::u32_t)token,
            score_fallback) || score_fallback) {
      std::fprintf(stderr,
                   "[fp16_score][FAIL] score mainline failed sample=%u token=%u fallback=%u\n",
                   (unsigned)sample_idx,
                   (unsigned)token,
                   (unsigned)(score_fallback ? 1u : 0u));
      return 1;
    }

    for (uint32_t h = 0u; h < n_heads; ++h) {
      const uint32_t head_col_base = h * d_head;
      const uint32_t q_row_base = (uint32_t)sc.q_base_word.to_uint() + token * d_model;
      const uint32_t score_head_base = (uint32_t)sc.score_base_word.to_uint() + h * token_count;
      for (uint32_t key = 0u; key < token_count; ++key) {
        const uint32_t k_row_base = (uint32_t)sc.k_base_word.to_uint() + key * d_model + head_col_base;
        aecct::quant_acc_t dot = aecct::quant_acc_t(0);
        for (uint32_t i = 0u; i < d_head; ++i) {
          const aecct::quant_act_t qv = aecct::quant_act_from_bits(sram[q_row_base + head_col_base + i]);
          const aecct::quant_act_t kv = aecct::quant_act_from_bits(sram[k_row_base + i]);
          dot += aecct::quant_acc_t(qv) * aecct::quant_acc_t(kv);
        }
        const aecct::quant_acc_t scaled = dot * inv_sqrt_d_head;
        const uint32_t exp_bits = (uint32_t)aecct::quant_bits_from_acc(scaled).to_uint();
        const uint32_t got_bits = (uint32_t)sram[score_head_base + key].to_uint();
        if (got_bits != exp_bits) {
          if (total_mismatch == 0u) {
            std::fprintf(stderr,
                         "[fp16_score][FAIL] first mismatch sample=%u token=%u head=%u key=%u got=0x%08X exp=0x%08X\n",
                         (unsigned)sample_idx,
                         (unsigned)token,
                         (unsigned)h,
                         (unsigned)key,
                         (unsigned)got_bits,
                         (unsigned)exp_bits);
          }
          ++total_mismatch;
        }
      }
    }
  }

  if (total_mismatch != 0u) {
    std::fprintf(stderr,
                 "[fp16_score][FAIL] total mismatches sample=%u count=%u\n",
                 (unsigned)sample_idx,
                 (unsigned)total_mismatch);
    return 1;
  }

  std::printf("[fp16_score][PASS] sample=%u tokens=%u heads=%u score_mismatch=0\n",
              (unsigned)sample_idx,
              (unsigned)token_count,
              (unsigned)n_heads);
  return 0;
}

}  // namespace

int main() {
  for (uint32_t sample = 0u; sample < 3u; ++sample) {
    if (run_one_sample(sample) != 0) {
      return 1;
    }
  }
  std::printf("PASS: tb_fp16_branch_score_compare_smoke\n");
  return 0;
}

#endif
