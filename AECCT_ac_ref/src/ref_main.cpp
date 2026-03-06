#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "ac_channel.h"

#include "../include/RefMetrics.h"
#include "../include/RefStep0ShapeBridge.h"
#include "../include/RefStep0RunReport.h"
#include "../synth/RefStep0Synth.h"

#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"

namespace {

static const int kVars = ModelShapes::N_VARS;
static const int kOutDim = ModelShapes::OUT_DIM;
static const int kXPredWords = ModelShapes::XPRED_WORDS;

static void print_first_k(const char *name, const double *x, int k) {
  std::printf("%s[0:%d]:\n", name, k);
  for (int i = 0; i < k; ++i) {
    std::printf("  [%d] %.9f\n", i, x[i]);
  }
}

static inline double fp32_bits_to_double(aecct_ref::u32_word_t bits) {
  aecct_ref::fp32_ref_t x;
  x.set_data(static_cast<ac_int<32, true> >(bits));
  return static_cast<double>(x.to_float());
}

static void run_synth_step0_pattern_mode(
  const double *input_y,
  uint32_t mode,
  aecct_ref::u32_word_t *out_words,
  int max_words,
  int &got_words,
  aecct_ref::RefStep0RunReport &report
) {
  ac_channel<aecct_ref::fp32_ref_t> in_y_ch;
  ac_channel<aecct_ref::u32_word_t> data_out;

  for (int i = 0; i < kVars; ++i) {
    in_y_ch.write(aecct_ref::fp32_ref_t(static_cast<float>(input_y[i])));
  }

  aecct_ref::ref_step0_set_outmode(ac_int<2, false>(mode));
  aecct_ref::ref_step0_synth(in_y_ch, data_out);
  report = aecct_ref::ref_step0_get_last_report();

  got_words = 0;
  aecct_ref::u32_word_t w;
  while (data_out.nb_read(w)) {
    if (got_words < max_words) {
      out_words[got_words] = w;
    }
    got_words++;
  }
}

static void decode_xpred_words(
  const aecct_ref::u32_word_t *words,
  aecct_ref::bit1_t *xpred_bits
) {
  for (int var_idx = 0; var_idx < kVars; ++var_idx) {
    const int word_idx = (var_idx >> 5);
    const int bit_idx = (var_idx & 31);
    const uint32_t packed = static_cast<uint32_t>(words[word_idx].to_uint());
    const uint32_t bit = (packed >> bit_idx) & 1u;
    xpred_bits[var_idx] = aecct_ref::bit1_t(bit);
  }
}

static void print_run_report(const char *tag, const aecct_ref::RefStep0RunReport &r) {
  std::printf("=== %s RunReport ===\n", tag);
  std::printf("final_scalar_base_word         : %u\n", r.final_scalar_base_word);
  std::printf("final_scalar_words             : %u\n", r.final_scalar_words);
  std::printf("scratch_base_word              : %u\n", r.scratch_base_word);
  std::printf("scratch_words                  : %u\n", r.scratch_words);
  std::printf("final_scalar_in_scratch        : %d\n", r.final_scalar_in_scratch ? 1 : 0);
  std::printf("final_scalar_range_ok          : %d\n", r.final_scalar_range_ok ? 1 : 0);
  std::printf("final_scalar_capacity_ok       : %d\n", r.final_scalar_capacity_ok ? 1 : 0);
  std::printf("addr_overlap_scr_k             : %d\n", r.final_scalar_addr_overlap_scr_k ? 1 : 0);
  std::printf("addr_overlap_scr_v             : %d\n", r.final_scalar_addr_overlap_scr_v ? 1 : 0);
  std::printf("live_conflict_scr_k            : %d\n", r.final_scalar_live_conflict_scr_k ? 1 : 0);
  std::printf("live_conflict_scr_v            : %d\n", r.final_scalar_live_conflict_scr_v ? 1 : 0);
  std::printf("final_scalar_overlap_conflict  : %d\n", r.final_scalar_overlap_conflict ? 1 : 0);
  std::printf("final_layer_no_writeback       : %d\n", r.final_layer_no_writeback_enforced ? 1 : 0);
  std::printf("final_layer_writeback_words    : %u\n", r.final_layer_writeback_words);
  std::printf("final_head_used_page_next      : %d\n", r.final_head_used_page_next ? 1 : 0);
  std::printf("pass_b_executed                : %d\n", r.pass_b_executed ? 1 : 0);
  std::printf("output_words                   : %u\n", r.output_words);
  std::printf("has_error                      : %d\n", r.has_error ? 1 : 0);
  std::printf("error_code                     : 0x%08X\n", r.error_code);
  std::printf("error_msg                      : %s\n", aecct_ref::ref_step0_error_msg_text(r.error_msg));
}

} // anonymous namespace

int main(int argc, char **argv) {
  int b_sel = -1;
  if (argc >= 2) {
    b_sel = std::atoi(argv[1]);
  }

  const int B = trace_input_y_step0_tensor_shape[0];
  const int N = trace_input_y_step0_tensor_shape[1];
  const int logits_n = trace_output_logits_step0_tensor_shape[1];
  const int xpred_n = trace_output_x_pred_step0_tensor_shape[1];

  if (N != kVars) {
    std::printf("Unexpected input N=%d, expected ModelShapes::N_VARS=%d\n", N, kVars);
    return 1;
  }
  if (logits_n != kOutDim) {
    std::printf("Unexpected logits dim=%d, expected ModelShapes::OUT_DIM=%d\n", logits_n, kOutDim);
    return 1;
  }
  if (xpred_n != kVars) {
    std::printf("Unexpected x_pred dim=%d, expected ModelShapes::N_VARS=%d\n", xpred_n, kVars);
    return 1;
  }

  if (b_sel >= 0 && b_sel >= B) {
    std::printf("Usage: ref_sim [pattern_index]\n");
    std::printf("pattern_index must be in [0, %d)\n", B);
    return 1;
  }

  const int run_B = (b_sel >= 0) ? 1 : B;

  double *out_logits = new double[run_B * kOutDim];
  aecct_ref::bit1_t *out_xpred = new aecct_ref::bit1_t[run_B * kVars];

  aecct_ref::RefStep0RunReport report_mode0 = {};
  aecct_ref::RefStep0RunReport report_mode1 = {};
  aecct_ref::RefStep0RunReport report_mode2 = {};

  for (int rb = 0; rb < run_B; ++rb) {
    const int src_b = (b_sel >= 0) ? b_sel : rb;
    const int src_base = src_b * N;

    aecct_ref::u32_word_t mode0_words[kXPredWords + 4];
    aecct_ref::u32_word_t mode1_words[kOutDim + 4];
    aecct_ref::u32_word_t mode2_words[4];

    int got0 = 0;
    int got1 = 0;
    int got2 = 0;

    run_synth_step0_pattern_mode(
      &trace_input_y_step0_tensor[src_base],
      0u,
      mode0_words,
      kXPredWords + 4,
      got0,
      report_mode0
    );

    if (report_mode0.has_error || got0 != static_cast<int>(report_mode0.output_words) ||
        report_mode0.output_words != static_cast<uint32_t>(kXPredWords)) {
      std::printf("OUTMODE=0 failed at pattern %d: got=%d report_words=%u err=%d\n",
        src_b, got0, report_mode0.output_words, report_mode0.has_error ? 1 : 0);
      print_run_report("OUTMODE=0", report_mode0);
      delete[] out_logits;
      delete[] out_xpred;
      return 2;
    }

    decode_xpred_words(mode0_words, &out_xpred[rb * kVars]);

    run_synth_step0_pattern_mode(
      &trace_input_y_step0_tensor[src_base],
      1u,
      mode1_words,
      kOutDim + 4,
      got1,
      report_mode1
    );

    if (report_mode1.has_error || got1 != static_cast<int>(report_mode1.output_words) ||
        report_mode1.output_words != static_cast<uint32_t>(kOutDim)) {
      std::printf("OUTMODE=1 failed at pattern %d: got=%d report_words=%u err=%d\n",
        src_b, got1, report_mode1.output_words, report_mode1.has_error ? 1 : 0);
      print_run_report("OUTMODE=1", report_mode1);
      delete[] out_logits;
      delete[] out_xpred;
      return 3;
    }

    for (int i = 0; i < kOutDim; ++i) {
      out_logits[rb * kOutDim + i] = fp32_bits_to_double(mode1_words[i]);
    }

    run_synth_step0_pattern_mode(
      &trace_input_y_step0_tensor[src_base],
      2u,
      mode2_words,
      4,
      got2,
      report_mode2
    );

    if (report_mode2.has_error || got2 != 0 || report_mode2.output_words != 0u || report_mode2.pass_b_executed) {
      std::printf("OUTMODE=2 failed at pattern %d: got=%d report_words=%u pass_b=%d err=%d\n",
        src_b, got2, report_mode2.output_words, report_mode2.pass_b_executed ? 1 : 0,
        report_mode2.has_error ? 1 : 0);
      print_run_report("OUTMODE=2", report_mode2);
      delete[] out_logits;
      delete[] out_xpred;
      return 4;
    }
  }

  const double *golden_logits =
    (b_sel >= 0) ? &trace_output_logits_step0_tensor[b_sel * kOutDim]
                 : trace_output_logits_step0_tensor;

  aecct_ref::Metrics m_logits =
    aecct_ref::compute_metrics(golden_logits, out_logits, static_cast<std::size_t>(run_B * kOutDim));

  std::printf("=== Step0 logits metrics vs golden ===\n");
  std::printf("MSE     : %.6e\n", m_logits.mse);
  std::printf("RMSE    : %.6e\n", m_logits.rmse);
  std::printf("MAE     : %.6e\n", m_logits.mae);
  std::printf("MaxAbs  : %.6e\n", m_logits.max_abs);

  const double *golden_xpred =
    (b_sel >= 0) ? &trace_output_x_pred_step0_tensor[b_sel * kVars]
                 : trace_output_x_pred_step0_tensor;

  std::size_t match = 0;
  for (int i = 0; i < run_B * kVars; ++i) {
    const int g = (golden_xpred[i] != 0.0) ? 1 : 0;
    const int p = static_cast<int>(out_xpred[i].to_int());
    if (g == p) {
      match++;
    }
  }
  const double acc = (run_B * kVars > 0)
    ? (100.0 * static_cast<double>(match) / static_cast<double>(run_B * kVars))
    : 0.0;

  std::printf("x_pred match: %.2f%% (%zu / %d)\n", acc, match, run_B * kVars);
  print_first_k("golden_logits", golden_logits, 8);
  print_first_k("ref_logits   ", out_logits, 8);

  print_run_report("OUTMODE=0", report_mode0);
  print_run_report("OUTMODE=1", report_mode1);
  print_run_report("OUTMODE=2", report_mode2);

  delete[] out_logits;
  delete[] out_xpred;

  return 0;
}

