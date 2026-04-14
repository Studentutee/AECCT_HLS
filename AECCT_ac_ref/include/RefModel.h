#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "RefModelDebugTaps.h"
#include "RefModelLegacyModes.h"
#include "RefPrecisionMode.h"
#include "RefStep0RunReport.h"
#include "RefTypes.h"

namespace aecct_ref {

struct RefModelIO {
  const act_t* input_y = nullptr;
  const double* input_y_fp32 = nullptr;
  double* out_logits = nullptr;
  bit1_t* out_x_pred = nullptr;
  RefModelDebugTaps debug{};
  int B = 0;
  int N = 0;
};

struct RefDumpConfig {
  bool enabled;
  const char* dump_dir;
  int pattern_index;
};

enum class RefStep0OutputMode : uint8_t {
  X_PRED = 0,
  LOGITS = 1
};

struct RefStep0Io16Image {
  RefStep0RunReport report;
  RefStep0OutputMode output_mode = RefStep0OutputMode::X_PRED;
  std::vector<uint16_t> sram_words16;
  std::vector<uint16_t> data_out_words16;
};

struct RefRunConfig {
  // Mainline control surface. The baseline_fp32 token keeps FP32 islands in FP32
  // while native linear kernels use the AECCT ternary/int8/int16 contract.
  RefPrecisionMode precision_mode = RefPrecisionMode::BASELINE_FP32;
  // Non-mainline tuning/experiment knobs are grouped away from the core path.
  RefLegacyRunConfig legacy{};
};

bool is_fp32_baseline_mode(RefPrecisionMode mode);
bool is_fp16_experiment_mode(RefPrecisionMode mode);
RefRunConfig make_fp32_baseline_run_config();
RefRunConfig make_fp16_experiment_run_config();

class RefModel {
public:
  RefModel();

  void set_run_config(const RefRunConfig& cfg);
  RefRunConfig get_run_config() const;

  void set_dump_config(const RefDumpConfig& cfg);
  void clear_dump_config();

  // Step-0 reference path aligned to algorithm_ref.ipynb.
  void infer_step0(const RefModelIO& io) const;

  // Build a ref-only io16 image for single-pattern SRAM/readback checks.
  // This path does not change the math kernel; it only stages already-computed
  // ref outputs into 16-bit storage words and io16 data_out framing.
  bool build_step0_io16_image(const RefModelIO& io,
                              RefStep0OutputMode output_mode,
                              RefStep0Io16Image& image) const;

  static bool read_mem_words16(const RefStep0Io16Image& image,
                               uint32_t addr_word16,
                               uint32_t len_words16,
                               std::vector<uint16_t>& out_words16);

  static bool unpack_logits_from_io16(const std::vector<uint16_t>& data_out_words16,
                                      std::vector<double>& logits_out);

  static bool unpack_xpred_from_io16(const std::vector<uint16_t>& data_out_words16,
                                     int n_bits,
                                     std::vector<uint8_t>& xpred_bits_out);

private:
  RefRunConfig run_cfg_;
  RefDumpConfig dump_cfg_;
};

} // namespace aecct_ref
