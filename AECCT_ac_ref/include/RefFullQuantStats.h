#pragma once

#include <cstdint>
#include <string>

namespace aecct_ref {

struct RefIntLinearStats {
  std::uint64_t int8_clamp_count = 0;
  std::uint64_t int16_overflow_count = 0;
  std::uint64_t dequant_restore_count = 0;
  std::string first_int16_overflow_block;
};

struct RefE4M3PathStats {
  std::uint64_t roundtrip_count = 0;
  std::uint64_t roundtrip_g1_count = 0;
  std::uint64_t roundtrip_g2_count = 0;
  std::uint64_t roundtrip_g3_count = 0;
  std::uint64_t roundtrip_g4_count = 0;
  std::uint64_t roundtrip_g5_count = 0;
  std::uint64_t roundtrip_g5_embed_count = 0;
  std::uint64_t roundtrip_g5_spe_count = 0;
  std::uint64_t roundtrip_g5_preproc_assembly_count = 0;
  std::uint64_t roundtrip_g5_prelayer_handoff_count = 0;
  std::uint64_t nan_in_count = 0;
  std::uint64_t nan_out_count = 0;
  std::uint64_t inf_in_count = 0;
  std::uint64_t inf_out_count = 0;
  std::string first_nonfinite_block;
};

struct RefInt8FixedExpStats {
  std::uint64_t roundtrip_count = 0;
  std::uint64_t clamp_count = 0;
  std::uint64_t zone1_count = 0;
  std::uint64_t zone2_count = 0;
  std::uint64_t zone3_count = 0;
  std::uint64_t zone4_count = 0;
  std::uint64_t footprint_g2_count = 0;
  std::uint64_t footprint_g5_embed_count = 0;
  std::string first_clamp_block;
};

struct RefFp16PathStats {
  std::uint64_t roundtrip_count = 0;
  std::uint64_t nan_in_count = 0;
  std::uint64_t nan_out_count = 0;
  std::uint64_t inf_in_count = 0;
  std::uint64_t inf_out_count = 0;
  std::uint64_t underflow_to_zero_count = 0;
  std::string first_nonfinite_block;
};

struct RefFullQuantStats {
  RefIntLinearStats int_linear;
  RefE4M3PathStats e4m3;
  RefInt8FixedExpStats int8_fixedexp;
  RefFp16PathStats fp16;
};

inline RefFullQuantStats g_ref_full_quant_stats{};

static inline void reset_ref_full_quant_stats() {
  g_ref_full_quant_stats = RefFullQuantStats{};
}

static inline RefFullQuantStats get_ref_full_quant_stats() {
  return g_ref_full_quant_stats;
}

static inline void add_ref_full_quant_stats(const RefFullQuantStats& delta) {
  RefFullQuantStats& s = g_ref_full_quant_stats;
  s.int_linear.int8_clamp_count += delta.int_linear.int8_clamp_count;
  s.int_linear.int16_overflow_count += delta.int_linear.int16_overflow_count;
  s.int_linear.dequant_restore_count += delta.int_linear.dequant_restore_count;
  if (s.int_linear.first_int16_overflow_block.empty()) {
    s.int_linear.first_int16_overflow_block = delta.int_linear.first_int16_overflow_block;
  }

  s.e4m3.roundtrip_count += delta.e4m3.roundtrip_count;
  s.e4m3.roundtrip_g1_count += delta.e4m3.roundtrip_g1_count;
  s.e4m3.roundtrip_g2_count += delta.e4m3.roundtrip_g2_count;
  s.e4m3.roundtrip_g3_count += delta.e4m3.roundtrip_g3_count;
  s.e4m3.roundtrip_g4_count += delta.e4m3.roundtrip_g4_count;
  s.e4m3.roundtrip_g5_count += delta.e4m3.roundtrip_g5_count;
  s.e4m3.roundtrip_g5_embed_count += delta.e4m3.roundtrip_g5_embed_count;
  s.e4m3.roundtrip_g5_spe_count += delta.e4m3.roundtrip_g5_spe_count;
  s.e4m3.roundtrip_g5_preproc_assembly_count += delta.e4m3.roundtrip_g5_preproc_assembly_count;
  s.e4m3.roundtrip_g5_prelayer_handoff_count += delta.e4m3.roundtrip_g5_prelayer_handoff_count;
  s.e4m3.nan_in_count += delta.e4m3.nan_in_count;
  s.e4m3.nan_out_count += delta.e4m3.nan_out_count;
  s.e4m3.inf_in_count += delta.e4m3.inf_in_count;
  s.e4m3.inf_out_count += delta.e4m3.inf_out_count;
  if (s.e4m3.first_nonfinite_block.empty()) {
    s.e4m3.first_nonfinite_block = delta.e4m3.first_nonfinite_block;
  }

  s.int8_fixedexp.roundtrip_count += delta.int8_fixedexp.roundtrip_count;
  s.int8_fixedexp.clamp_count += delta.int8_fixedexp.clamp_count;
  s.int8_fixedexp.zone1_count += delta.int8_fixedexp.zone1_count;
  s.int8_fixedexp.zone2_count += delta.int8_fixedexp.zone2_count;
  s.int8_fixedexp.zone3_count += delta.int8_fixedexp.zone3_count;
  s.int8_fixedexp.zone4_count += delta.int8_fixedexp.zone4_count;
  s.int8_fixedexp.footprint_g2_count += delta.int8_fixedexp.footprint_g2_count;
  s.int8_fixedexp.footprint_g5_embed_count += delta.int8_fixedexp.footprint_g5_embed_count;
  if (s.int8_fixedexp.first_clamp_block.empty()) {
    s.int8_fixedexp.first_clamp_block = delta.int8_fixedexp.first_clamp_block;
  }

  s.fp16.roundtrip_count += delta.fp16.roundtrip_count;
  s.fp16.nan_in_count += delta.fp16.nan_in_count;
  s.fp16.nan_out_count += delta.fp16.nan_out_count;
  s.fp16.inf_in_count += delta.fp16.inf_in_count;
  s.fp16.inf_out_count += delta.fp16.inf_out_count;
  s.fp16.underflow_to_zero_count += delta.fp16.underflow_to_zero_count;
  if (s.fp16.first_nonfinite_block.empty()) {
    s.fp16.first_nonfinite_block = delta.fp16.first_nonfinite_block;
  }
}

} // namespace aecct_ref
