#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"

namespace {

static const char* section_name(const sram_map::SectionId id) {
  switch (id) {
    case sram_map::SEC_X_WORK: return "X_WORK";
    case sram_map::SEC_SCR_K: return "SCR_K";
    case sram_map::SEC_SCR_V: return "SCR_V";
    case sram_map::SEC_FINAL_SCALAR_BUF: return "FINAL_SCALAR_BUF";
    case sram_map::SEC_SCRATCH: return "SCRATCH";
    case sram_map::SEC_BIAS_LEGACY: return "BIAS_LEGACY";
    case sram_map::SEC_WEIGHT_LEGACY: return "WEIGHT_LEGACY";
    case sram_map::SEC_W_REGION: return "W_REGION";
    case sram_map::SEC_PARAM_STREAM_DEFAULT: return "PARAM_STREAM_DEFAULT";
    case sram_map::SEC_IO_REGION: return "IO_REGION";
    case sram_map::SEC_BACKUP_RUNTIME_SCRATCH: return "BACKUP_RUNTIME_SCRATCH";
    default: return "INVALID";
  }
}

static const char* section_kind_name(const uint8_t kind) {
  switch (kind) {
    case sram_map::SECTION_PHYSICAL: return "physical";
    case sram_map::SECTION_LOGICAL: return "logical";
    default: return "invalid";
  }
}

static const char* storage_class_name(const uint8_t cls) {
  switch (cls) {
    case sram_map::CLASS_W_REGION: return "W_REGION";
    case sram_map::CLASS_X_WORK: return "X_WORK";
    case sram_map::CLASS_SCRATCH: return "SCRATCH";
    case sram_map::CLASS_IO_REGION: return "IO_REGION";
    case sram_map::CLASS_TOKEN_LOCAL: return "TOKEN_LOCAL";
    case sram_map::CLASS_SMALL_SCRATCH: return "SMALL_SCRATCH";
    default: return "INVALID";
  }
}

static const char* alias_group_name(const uint8_t grp) {
  switch (grp) {
    case sram_map::ALIAS_W_PERSIST: return "W_PERSIST";
    case sram_map::ALIAS_X_WORK: return "X_WORK";
    case sram_map::ALIAS_SCR_K: return "SCR_K";
    case sram_map::ALIAS_SCR_V: return "SCR_V";
    case sram_map::ALIAS_FINAL_SCALAR: return "FINAL_SCALAR";
    case sram_map::ALIAS_IO_STAGING: return "IO_STAGING";
    case sram_map::ALIAS_LOCAL_ONLY: return "LOCAL_ONLY";
    default: return "INVALID";
  }
}

static const char* payload_class_name(const uint8_t payload) {
  switch (payload) {
    case sram_map::PAYLOAD_FP32_TENSOR: return "fp32_tensor";
    case sram_map::PAYLOAD_FP32_VECTOR: return "fp32_vector";
    case sram_map::PAYLOAD_BITPACK: return "bitpack";
    case sram_map::PAYLOAD_PARAM_STREAM: return "param_stream";
    case sram_map::PAYLOAD_EMPTY: return "empty";
    case sram_map::PAYLOAD_COMPAT_ALIAS: return "compat_alias";
    case sram_map::PAYLOAD_RUNTIME_SCRATCH: return "runtime_scratch";
    case sram_map::PAYLOAD_MIXED_PERSIST: return "mixed_persist";
    default: return "invalid";
  }
}

static void fail(const char* tag, unsigned lhs, unsigned rhs) {
  std::printf("FAIL: %s got=%u expect=%u\n", tag, lhs, rhs);
  std::exit(1);
}

static void fail_bool(const char* tag) {
  std::printf("FAIL: %s\n", tag);
  std::exit(1);
}

static void expect_eq_u32(const char* tag, uint32_t lhs, uint32_t rhs) {
  if (lhs != rhs) {
    fail(tag, lhs, rhs);
  }
}

static void expect_true(const char* tag, bool cond) {
  if (!cond) {
    fail_bool(tag);
  }
}

static void print_section_table() {
  std::printf("SECTION TABLE\n");
  std::printf("name,kind,storage_class,alias,payload,base_w,len_w,payload_w,base_word16,len_word16,payload_word16,flags\n");
  for (uint32_t i = 0u; i < sram_map::SECTION_COUNT; ++i) {
    const sram_map::SectionDesc& d = sram_map::kSectionTable[i];
    std::printf("%s,%s,%s,%s,%s,%u,%u,%u,%u,%u,%u,0x%08X\n",
                section_name((sram_map::SectionId)d.id),
                section_kind_name(d.kind),
                storage_class_name(d.storage_class),
                alias_group_name(d.alias_group),
                payload_class_name(d.payload_class),
                d.base_w,
                d.words_w,
                d.payload_words_w,
                d.base_word16,
                d.words_word16,
                d.payload_words_word16,
                d.flags);
  }
}

} // namespace

int main() {
  expect_eq_u32("SECTION_COUNT", sram_map::SECTION_COUNT, 11u);
  expect_eq_u32("PARAM_STREAM_DEFAULT_WORDS_vs_EXP_LEN_PARAM_WORDS",
                sram_map::PARAM_STREAM_DEFAULT_WORDS,
                EXP_LEN_PARAM_WORDS);
  expect_true("default_param_stream_fits_w_region",
              sram_map::default_param_stream_fits_w_region());
  expect_eq_u32("FINAL_SCALAR_BUF_BASE_alias",
                sram_map::FINAL_SCALAR_BUF_BASE_W,
                sram_map::SCR_FINAL_SCALAR_BASE_W);
  expect_eq_u32("FINAL_SCALAR_BUF_WORDS_alias",
                sram_map::FINAL_SCALAR_BUF_WORDS,
                sram_map::SCR_FINAL_SCALAR_WORDS);
  expect_eq_u32("BIAS_padding_words", sram_map::SIZE_BIAS_PADDING_W, 8u);
  expect_eq_u32("WEIGHT_padding_words", sram_map::SIZE_WEIGHT_PADDING_W, 8u);
  expect_eq_u32("W_REGION_padding_words", sram_map::W_REGION_PADDING_WORDS, 16u);
  expect_true("W_REGION_base_is_legacy_aligned",
              sram_map::is_legacy_word_aligned(sram_map::W_REGION_BASE, ALIGN_WORDS));
  expect_true("W_REGION_base_is_storage_beat_aligned",
              sram_map::is_storage_beat_aligned_word16(sram_map::W_REGION_BASE_WORD16));
  expect_true("X_WORK_base_is_storage_beat_aligned",
              sram_map::is_storage_beat_aligned_word16(sram_map::BASE_X_WORK_WORD16));
  expect_true("SCR_K_base_is_storage_beat_aligned",
              sram_map::is_storage_beat_aligned_word16(sram_map::BASE_SCR_K_WORD16));
  expect_true("SCR_V_base_is_storage_beat_aligned",
              sram_map::is_storage_beat_aligned_word16(sram_map::BASE_SCR_V_WORD16));
  expect_true("FINAL_SCALAR_BUF_base_is_storage_beat_aligned",
              sram_map::is_storage_beat_aligned_word16(sram_map::FINAL_SCALAR_BUF_BASE_WORD16));
  expect_true("physical_section_of_addr_scr_k",
              sram_map::physical_section_of_addr(sram_map::BASE_SCR_K_W) == sram_map::SEC_SCR_K);
  expect_true("physical_section_of_addr_scr_v",
              sram_map::physical_section_of_addr(sram_map::BASE_SCR_V_W) == sram_map::SEC_SCR_V);
  expect_true("physical_section_of_addr_final_scalar",
              sram_map::physical_section_of_addr(sram_map::FINAL_SCALAR_BUF_BASE_W) == sram_map::SEC_FINAL_SCALAR_BUF);
  expect_true("physical_section_of_addr_weight_legacy",
              sram_map::physical_section_of_addr(sram_map::BASE_W_W) == sram_map::SEC_WEIGHT_LEGACY);
  expect_true("sec_param_stream_is_load_w_critical",
              sram_map::section_affects_load_w(sram_map::SEC_PARAM_STREAM_DEFAULT));
  expect_true("sec_w_region_is_load_w_critical",
              sram_map::section_affects_load_w(sram_map::SEC_W_REGION));
  expect_true("sec_x_work_is_compare_critical",
              sram_map::section_affects_compare(sram_map::SEC_X_WORK));
  expect_true("sec_scr_k_is_compare_critical",
              sram_map::section_affects_compare(sram_map::SEC_SCR_K));
  expect_true("sec_final_scalar_is_compare_critical",
              sram_map::section_affects_compare(sram_map::SEC_FINAL_SCALAR_BUF));
  expect_true("sec_io_region_payload_fits_capacity",
              sram_map::section_payload_fits_capacity(sram_map::SEC_IO_REGION));
  expect_eq_u32("W_REGION_payload_words_word16",
                sram_map::W_REGION_PAYLOAD_WORDS_WORD16,
                sram_map::PARAM_STREAM_DEFAULT_WORDS_WORD16);

  print_section_table();
  std::printf("PASS: tb_srammap_section_layout_smoke\n");
  return 0;
}
