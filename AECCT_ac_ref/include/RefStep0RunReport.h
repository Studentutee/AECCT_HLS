#pragma once

#include <cstdint>

namespace aecct_ref {

enum RefStep0ErrorBits : uint32_t {
  REF_STEP0_ERR_NONE = 0u,
  REF_STEP0_ERR_MAP_OOB = 1u << 0,
  REF_STEP0_ERR_FINAL_SCALAR_RANGE = 1u << 1,
  REF_STEP0_ERR_FINAL_SCALAR_CAPACITY = 1u << 2,
  REF_STEP0_ERR_FINAL_SCALAR_LIVE_CONFLICT = 1u << 3,
  REF_STEP0_ERR_UNSUPPORTED_LAYER = 1u << 4,
};

enum RefStep0ErrorMsg : uint32_t {
  REF_STEP0_MSG_NONE = 0u,
  REF_STEP0_MSG_MAP_OOB = 1u,
  REF_STEP0_MSG_FINAL_SCALAR_RANGE = 2u,
  REF_STEP0_MSG_FINAL_SCALAR_CAPACITY = 3u,
  REF_STEP0_MSG_FINAL_SCALAR_LIVE_CONFLICT = 4u,
  REF_STEP0_MSG_UNSUPPORTED_LAYER = 5u,
};

struct RefStep0RunReport {
  // Legacy bridge-word fields retained for compatibility with existing reports.
  uint32_t final_scalar_base_word;
  uint32_t final_scalar_words;
  uint32_t scratch_base_word;
  uint32_t scratch_words;

  // v12.1 storage-word view (16-bit words, 8-word beats).
  uint32_t final_scalar_base_word16;
  uint32_t final_scalar_words16;
  uint32_t scratch_base_word16;
  uint32_t scratch_words16;
  uint32_t final_scalar_beats16;
  uint32_t scratch_beats16;

  bool final_scalar_in_scratch;
  bool final_scalar_range_ok;
  bool final_scalar_capacity_ok;
  bool final_scalar_addr_overlap_scr_k;
  bool final_scalar_addr_overlap_scr_v;
  bool final_scalar_live_conflict_scr_k;
  bool final_scalar_live_conflict_scr_v;
  bool final_scalar_overlap_conflict;

  bool final_layer_no_writeback_enforced;
  uint32_t final_layer_writeback_words;
  // Legacy compatibility field; expected false under single X_WORK scheduling.
  bool final_head_used_page_next;

  bool pass_b_executed;
  uint32_t output_words;
  uint32_t output_words16;
  bool final_scalar_base_word16_aligned;
  bool scratch_base_word16_aligned;

  bool has_error;
  uint32_t error_code;
  uint32_t error_msg;
};

static inline const char *ref_step0_error_msg_text(uint32_t msg) {
  switch (msg) {
    case REF_STEP0_MSG_NONE: return "NONE";
    case REF_STEP0_MSG_MAP_OOB: return "MAP_OOB";
    case REF_STEP0_MSG_FINAL_SCALAR_RANGE: return "FINAL_SCALAR_RANGE";
    case REF_STEP0_MSG_FINAL_SCALAR_CAPACITY: return "FINAL_SCALAR_CAPACITY";
    case REF_STEP0_MSG_FINAL_SCALAR_LIVE_CONFLICT: return "FINAL_SCALAR_LIVE_CONFLICT";
    case REF_STEP0_MSG_UNSUPPORTED_LAYER: return "UNSUPPORTED_LAYER";
    default: return "UNKNOWN";
  }
}

} // namespace aecct_ref
