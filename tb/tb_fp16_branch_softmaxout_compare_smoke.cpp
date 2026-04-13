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
    std::fprintf(stderr, "[fp16_softmax][FAIL] kv mainline failed sample=%u fallback=%u\n",
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
    std::fprintf(stderr, "[fp16_softmax][FAIL] q mainline failed sample=%u fallback=%u\n",
                 (unsigned)sample_idx,
                 (unsigned)(q_fallback ? 1u : 0u));
    return 1;
  }

  const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
  const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
  const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
  const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;
  const uint32_t out_base = (uint32_t)aecct::ATTN_OUT_BASE_WORD_DEFAULT;

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
                   "[fp16_softmax][FAIL] score mainline failed sample=%u token=%u fallback=%u\n",
                   (unsigned)sample_idx,
                   (unsigned)token,
                   (unsigned)(score_fallback ? 1u : 0u));
      return 1;
    }

    bool softmax_fallback = true;
    if (!aecct::attn_phaseb_top_managed_softmax_out_mainline(
            sram,
            attn_cfg,
            sc,
            (aecct::u32_t)token,
            (aecct::u32_t)out_base,
            softmax_fallback) || softmax_fallback) {
      std::fprintf(stderr,
                   "[fp16_softmax][FAIL] softmax/out mainline failed sample=%u token=%u fallback=%u\n",
                   (unsigned)sample_idx,
                   (unsigned)token,
                   (unsigned)(softmax_fallback ? 1u : 0u));
      return 1;
    }

    const uint32_t pre_row_base = (uint32_t)sc.pre_concat_base_word.to_uint() + token * d_model;
    const uint32_t post_row_base = (uint32_t)sc.post_concat_base_word.to_uint() + token * d_model;
    const uint32_t out_row_base = out_base + token * d_model;

    for (uint32_t h = 0u; h < n_heads; ++h) {
      const uint32_t head_col_base = h * d_head;
      const uint32_t score_head_base = (uint32_t)sc.score_base_word.to_uint() + h * token_count;

      softmax_score_t running_max = softmax_score_t(0);
      softmax_sum_t running_l = softmax_sum_t(0);
      aecct::quant_acc_t running_acc[aecct::ATTN_D_MODEL];
      for (uint32_t i = 0u; i < (uint32_t)aecct::ATTN_D_MODEL; ++i) {
        running_acc[i] = aecct::quant_acc_t(0);
      }
      bool have_state = false;

      for (uint32_t key = 0u; key < token_count; ++key) {
        const aecct::fp32_t score_fp = aecct::fp32_from_bits(sram[score_head_base + key]);
        const softmax_score_t score =
            score_fp.template convert_to_ac_fixed<18, 6, true, AC_RND, AC_SAT>(false);
        const uint32_t v_row_base = (uint32_t)sc.v_base_word.to_uint() + key * d_model + head_col_base;

        if (!have_state) {
          running_max = score;
          running_l = softmax_sum_t(1);
          for (uint32_t i = 0u; i < d_head; ++i) {
            const aecct::quant_act_t vv = aecct::quant_act_from_bits(sram[v_row_base + i]);
            running_acc[i] = aecct::quant_acc_t(vv);
          }
          have_state = true;
          continue;
        }

        if (score > running_max) {
          const softmax_x_t old_minus_new = softmax_x_t(running_max - score);
          const softmax_exp_t alpha = softmax_exp_lut(old_minus_new);
          running_l = softmax_sum_t(running_l * softmax_sum_t(alpha)) + softmax_sum_t(1);
          for (uint32_t i = 0u; i < d_head; ++i) {
            const aecct::quant_act_t vv = aecct::quant_act_from_bits(sram[v_row_base + i]);
            running_acc[i] = aecct::quant_acc_t(running_acc[i] * aecct::quant_acc_t(alpha)) + aecct::quant_acc_t(vv);
          }
          running_max = score;
        } else {
          const softmax_x_t score_minus_old = softmax_x_t(score - running_max);
          const softmax_exp_t beta = softmax_exp_lut(score_minus_old);
          running_l += softmax_sum_t(beta);
          for (uint32_t i = 0u; i < d_head; ++i) {
            const aecct::quant_act_t vv = aecct::quant_act_from_bits(sram[v_row_base + i]);
            running_acc[i] += aecct::quant_acc_t(beta) * aecct::quant_acc_t(vv);
          }
        }
      }

      const softmax_inv_t inv_l = softmax_rcp_lut(running_l);
      for (uint32_t i = 0u; i < d_head; ++i) {
        const aecct::quant_acc_t out_val = running_acc[i] * aecct::quant_acc_t(inv_l);
        const uint32_t exp_bits = (uint32_t)aecct::quant_bits_from_acc(out_val).to_uint();
        const uint32_t got_pre = (uint32_t)sram[pre_row_base + head_col_base + i].to_uint();
        const uint32_t got_post = (uint32_t)sram[post_row_base + head_col_base + i].to_uint();
        const uint32_t got_out = (uint32_t)sram[out_row_base + head_col_base + i].to_uint();
        if (got_pre != exp_bits || got_post != exp_bits || got_out != exp_bits) {
          if (total_mismatch == 0u) {
            std::fprintf(stderr,
                         "[fp16_softmax][FAIL] first mismatch sample=%u token=%u head=%u lane=%u got_pre=0x%08X got_post=0x%08X got_out=0x%08X exp=0x%08X\n",
                         (unsigned)sample_idx,
                         (unsigned)token,
                         (unsigned)h,
                         (unsigned)i,
                         (unsigned)got_pre,
                         (unsigned)got_post,
                         (unsigned)got_out,
                         (unsigned)exp_bits);
          }
          ++total_mismatch;
        }
      }
    }
  }

  if (total_mismatch != 0u) {
    std::fprintf(stderr,
                 "[fp16_softmax][FAIL] total mismatches sample=%u count=%u\n",
                 (unsigned)sample_idx,
                 (unsigned)total_mismatch);
    return 1;
  }

  std::printf("[fp16_softmax][PASS] sample=%u tokens=%u heads=%u out_mismatch=0\n",
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
  std::printf("PASS: tb_fp16_branch_softmaxout_compare_smoke\n");
  return 0;
}

#endif
