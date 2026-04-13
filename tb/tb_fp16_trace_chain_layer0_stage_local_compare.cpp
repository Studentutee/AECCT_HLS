#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <vector>

#include "Top.h"
#include "blocks/AttnLayer0.h"
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
    return false;
  }
  std::vector<uint16_t> payload_words16;
  uint32_t payload_logical = 0u;
  if (!fp16_branch_tb::build_weight_words16(payload_wid, payload_words16, payload_logical)) {
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
    return false;
  }
  inv_sw_bits_out = aecct::fp32_bits_from_fp16_lane((aecct::u16_t)inv_words16[0]);
  return true;
}

static bool compare_u32_region_exact(const char* tag,
                                     const aecct::u32_t* sram,
                                     uint32_t base_word,
                                     const std::vector<uint32_t>& exp_words) {
  for (uint32_t i = 0u; i < (uint32_t)exp_words.size(); ++i) {
    const uint32_t got = (uint32_t)sram[base_word + i].to_uint();
    const uint32_t exp = exp_words[i];
    if (got != exp) {
      std::fprintf(stderr,
                   "[fp16_trace_local][FAIL] %s first mismatch addr=%u idx=%u exp=0x%08X got=0x%08X\n",
                   tag,
                   (unsigned)(base_word + i),
                   (unsigned)i,
                   (unsigned)exp,
                   (unsigned)got);
      return false;
    }
  }
  return true;
}

static bool run_stage_local_attention_chain(uint32_t sample_idx,
                                            std::vector<uint32_t>& exp_attn_out_words) {
  static aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
  for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
    sram[i] = (aecct::u32_t)0u;
  }

  std::vector<uint16_t> param_word16;
  std::vector<uint32_t> param_u32;
  if (!fp16_branch_tb::seed_param_image_word16_into_sram(sram, param_word16, param_u32)) {
    return false;
  }

  std::vector<uint32_t> infer_words_u32;
  if (!fp16_branch_tb::build_infer_input_words_u32(sample_idx, infer_words_u32)) {
    return false;
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

  std::vector<uint16_t> exp_x_word16;
  std::vector<uint16_t> got_x_word16;
  if (!fp16_branch_tb::build_preproc_x_ref_word16(sample_idx, exp_x_word16)) {
    return false;
  }
  fp16_branch_tb::extract_x_work_word16_from_u32_sram(sram, got_x_word16);
  if (!fp16_branch_tb::compare_word16_vectors_exact(
          exp_x_word16,
          got_x_word16,
          "trace_chain_preproc_x_work",
          sram_map::FP16_BASELINE_X_WORK_BASE_WORD16)) {
    return false;
  }

  std::vector<aecct::u32_t> wk_payload_words;
  std::vector<aecct::u32_t> wv_payload_words;
  std::vector<aecct::u32_t> wq_payload_words;
  aecct::u32_t wk_inv_sw_bits = (aecct::u32_t)0u;
  aecct::u32_t wv_inv_sw_bits = (aecct::u32_t)0u;
  aecct::u32_t wq_inv_sw_bits = (aecct::u32_t)0u;
  if (!build_matrix_payload_and_inv_fp32((uint32_t)QLM_L0_WK, wk_payload_words, wk_inv_sw_bits) ||
      !build_matrix_payload_and_inv_fp32((uint32_t)QLM_L0_WV, wv_payload_words, wv_inv_sw_bits) ||
      !build_matrix_payload_and_inv_fp32((uint32_t)QLM_L0_WQ, wq_payload_words, wq_inv_sw_bits)) {
    return false;
  }

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
    std::fprintf(stderr, "[fp16_trace_local][FAIL] kv mainline failed sample=%u fallback=%u\n",
                 (unsigned)sample_idx,
                 (unsigned)(kv_fallback ? 1u : 0u));
    return false;
  }

  const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
  const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
  const uint32_t n_heads = (uint32_t)aecct::ATTN_N_HEADS;
  const uint32_t d_head = (uint32_t)aecct::ATTN_D_HEAD;

  // K/V compare
  for (uint32_t t = 0u; t < token_count; ++t) {
    aecct::u32_t x_row[aecct::kTernaryLiveL0WkCols];
    for (uint32_t d = 0u; d < d_model; ++d) {
      x_row[d] = aecct::fp32_bits_from_fp16_lane((aecct::u16_t)got_x_word16[t * d_model + d]);
    }
    aecct::u32_t k_out[aecct::kTernaryLiveL0WkRows];
    aecct::u32_t k_out_act_q[aecct::kTernaryLiveL0WkRows];
    aecct::u32_t k_out_inv_sw_bits = (aecct::u32_t)0u;
    aecct::u32_t v_out[aecct::kTernaryLiveL0WvRows];
    aecct::u32_t v_out_act_q[aecct::kTernaryLiveL0WvRows];
    aecct::u32_t v_out_inv_sw_bits = (aecct::u32_t)0u;
    if (!aecct::ternary_live_l0_wk_materialize_row_kernel_split(
            x_row,
            wk_payload_words.data(),
            wk_inv_sw_bits,
            k_out,
            k_out_act_q,
            k_out_inv_sw_bits) ||
        !aecct::ternary_live_l0_wv_materialize_row_kernel_split(
            x_row,
            wv_payload_words.data(),
            wv_inv_sw_bits,
            v_out,
            v_out_act_q,
            v_out_inv_sw_bits)) {
      return false;
    }
    const uint32_t row_k_base = (uint32_t)sc.k_base_word.to_uint() + t * d_model;
    const uint32_t row_v_base = (uint32_t)sc.v_base_word.to_uint() + t * d_model;
    for (uint32_t d = 0u; d < d_model; ++d) {
      const uint32_t got_k = (uint32_t)sram[row_k_base + d].to_uint();
      const uint32_t got_v = (uint32_t)sram[row_v_base + d].to_uint();
      if (got_k != (uint32_t)k_out[d].to_uint() || got_v != (uint32_t)v_out[d].to_uint()) {
        std::fprintf(stderr,
                     "[fp16_trace_local][FAIL] kv mismatch sample=%u token=%u d=%u\n",
                     (unsigned)sample_idx,
                     (unsigned)t,
                     (unsigned)d);
        return false;
      }
    }
  }

  bool q_fallback = true;
  if (!aecct::attn_phasea_top_managed_q_mainline(
          sram,
          regs.w_base_word,
          (aecct::u32_t)aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT,
          attn_cfg,
          sc,
          q_fallback) || q_fallback) {
    std::fprintf(stderr, "[fp16_trace_local][FAIL] q mainline failed sample=%u fallback=%u\n",
                 (unsigned)sample_idx,
                 (unsigned)(q_fallback ? 1u : 0u));
    return false;
  }

  // Q compare
  for (uint32_t t = 0u; t < token_count; ++t) {
    aecct::u32_t x_row[aecct::kTernaryLiveL0WqCols];
    for (uint32_t d = 0u; d < d_model; ++d) {
      x_row[d] = aecct::fp32_bits_from_fp16_lane((aecct::u16_t)got_x_word16[t * d_model + d]);
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
      return false;
    }
    const uint32_t row_q_base = (uint32_t)sc.q_base_word.to_uint() + t * d_model;
    for (uint32_t d = 0u; d < d_model; ++d) {
      const uint32_t got_q = (uint32_t)sram[row_q_base + d].to_uint();
      if (got_q != (uint32_t)q_out[d].to_uint()) {
        std::fprintf(stderr,
                     "[fp16_trace_local][FAIL] q mismatch sample=%u token=%u d=%u exp=0x%08X got=0x%08X\n",
                     (unsigned)sample_idx,
                     (unsigned)t,
                     (unsigned)d,
                     (unsigned)q_out[d].to_uint(),
                     (unsigned)got_q);
        return false;
      }
    }
  }

  const uint32_t out_base = (uint32_t)aecct::ATTN_OUT_BASE_WORD_DEFAULT;
  for (uint32_t token = 0u; token < token_count; ++token) {
    bool score_fallback = true;
    if (!aecct::attn_phaseb_top_managed_qk_score_mainline(
            sram,
            attn_cfg,
            sc,
            (aecct::u32_t)token,
            score_fallback) || score_fallback) {
      std::fprintf(stderr,
                   "[fp16_trace_local][FAIL] score mainline failed sample=%u token=%u fallback=%u\n",
                   (unsigned)sample_idx,
                   (unsigned)token,
                   (unsigned)(score_fallback ? 1u : 0u));
      return false;
    }

    // score compare
    for (uint32_t h = 0u; h < n_heads; ++h) {
      const uint32_t q_row_base = (uint32_t)sc.q_base_word.to_uint() + token * d_model + h * d_head;
      const uint32_t score_head_base = (uint32_t)sc.score_base_word.to_uint() + h * token_count;
      for (uint32_t key = 0u; key < token_count; ++key) {
        aecct::quant_acc_t dot = aecct::quant_acc_t(0);
        const uint32_t k_row_base = (uint32_t)sc.k_base_word.to_uint() + key * d_model + h * d_head;
        for (uint32_t dh = 0u; dh < d_head; ++dh) {
          const aecct::quant_act_t qv = aecct::quant_act_from_bits(sram[q_row_base + dh]);
          const aecct::quant_act_t kv = aecct::quant_act_from_bits(sram[k_row_base + dh]);
          dot += aecct::quant_acc_t(qv) * aecct::quant_acc_t(kv);
        }
        const aecct::quant_acc_t scaled = dot * aecct::attn_phaseb_inv_sqrt_d_head(d_head);
        const uint32_t exp_bits = (uint32_t)aecct::quant_bits_from_acc(scaled).to_uint();
        const uint32_t got_bits = (uint32_t)sram[score_head_base + key].to_uint();
        if (got_bits != exp_bits) {
          std::fprintf(stderr,
                       "[fp16_trace_local][FAIL] score mismatch sample=%u token=%u head=%u key=%u exp=0x%08X got=0x%08X\n",
                       (unsigned)sample_idx,
                       (unsigned)token,
                       (unsigned)h,
                       (unsigned)key,
                       (unsigned)exp_bits,
                       (unsigned)got_bits);
          return false;
        }
      }
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
                   "[fp16_trace_local][FAIL] softmax/out mainline failed sample=%u token=%u fallback=%u\n",
                   (unsigned)sample_idx,
                   (unsigned)token,
                   (unsigned)(softmax_fallback ? 1u : 0u));
      return false;
    }

    // softmax/out compare
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
        const softmax_score_t score = score_fp.template convert_to_ac_fixed<18, 6, true, AC_RND, AC_SAT>(false);
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
          std::fprintf(stderr,
                       "[fp16_trace_local][FAIL] softmax/out mismatch sample=%u token=%u head=%u lane=%u exp=0x%08X got_pre=0x%08X got_post=0x%08X got_out=0x%08X\n",
                       (unsigned)sample_idx,
                       (unsigned)token,
                       (unsigned)h,
                       (unsigned)i,
                       (unsigned)exp_bits,
                       (unsigned)got_pre,
                       (unsigned)got_post,
                       (unsigned)got_out);
          return false;
        }
      }
    }
  }

  exp_attn_out_words.assign(token_count * d_model, 0u);
  for (uint32_t i = 0u; i < token_count * d_model; ++i) {
    exp_attn_out_words[i] = (uint32_t)sram[out_base + i].to_uint();
  }
  std::printf("[fp16_trace_local][SUMMARY] sample=%u preproc=1 kv=1 q=1 score=1 softmax_out=1\n",
              (unsigned)sample_idx);
  return true;
}

static bool run_attnlayer0_integrated_compare(uint32_t sample_idx,
                                         const std::vector<uint32_t>& exp_attn_out_words) {
  static aecct::u32_t sram[sram_map::SRAM_WORDS_TOTAL];
  for (uint32_t i = 0u; i < (uint32_t)sram_map::SRAM_WORDS_TOTAL; ++i) {
    sram[i] = (aecct::u32_t)0u;
  }
  std::vector<uint16_t> param_word16;
  std::vector<uint32_t> param_u32;
  if (!fp16_branch_tb::seed_param_image_word16_into_sram(sram, param_word16, param_u32)) {
    return false;
  }
  std::vector<uint32_t> infer_words_u32;
  if (!fp16_branch_tb::build_infer_input_words_u32(sample_idx, infer_words_u32)) {
    return false;
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
  aecct::AttnLayer0<aecct::ATTN_STAGE_FULL>(
      sram,
      attn_cfg,
      (aecct::u32_t)aecct::PREPROC_X_OUT_BASE_WORD_DEFAULT,
      (aecct::u32_t)aecct::ATTN_OUT_BASE_WORD_DEFAULT,
      sc,
      regs.w_base_word,
      false,
      false,
      false,
      false,
      (aecct::u32_t)0u);

  for (uint32_t i = 0u; i < (uint32_t)exp_attn_out_words.size(); ++i) {
    const uint32_t got_attn = (uint32_t)sram[(uint32_t)aecct::ATTN_OUT_BASE_WORD_DEFAULT + i].to_uint();
    const uint32_t exp_attn = exp_attn_out_words[i];
    if (got_attn != exp_attn) {
      std::fprintf(stderr,
                   "[fp16_trace_local][FAIL] AttnLayer0 integrated attn_out mismatch sample=%u idx=%u exp=0x%08X got=0x%08X\n",
                   (unsigned)sample_idx,
                   (unsigned)i,
                   (unsigned)exp_attn,
                   (unsigned)got_attn);
      return false;
    }
  }
  std::printf("[fp16_trace_local][SUMMARY] sample=%u attnlayer0_integrated_out=1\n", (unsigned)sample_idx);
  return true;
}

}  // namespace

int main() {
  for (uint32_t sample = 0u; sample < 3u; ++sample) {
    std::vector<uint32_t> exp_attn_out_words;
    if (!run_stage_local_attention_chain(sample, exp_attn_out_words)) {
      return 1;
    }
  }
  std::printf("PASS: tb_fp16_trace_chain_layer0_stage_local_compare\n");
  return 0;
}

#endif
