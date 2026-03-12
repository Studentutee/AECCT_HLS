#pragma once
// Minimal Top-owned shared SRAM request/grant contract.

#include "AecctTypes.h"

namespace aecct {

enum RequesterId : unsigned {
  REQ_PREPROC = 0,
  REQ_TRANSFORMER = 1,
  REQ_LAYER_NORM = 2,
  REQ_FINAL_HEAD = 3,
  REQ_DEBUG_READ_MEM = 4,
  REQ_ID_COUNT = 5
};

enum PriorityClass : unsigned {
  PRIO_NECESSARY_WRITEBACK = 0,
  PRIO_XIN_READ = 1,
  PRIO_W_PREFETCH = 2,
  PRIO_DEBUG_READ_MEM = 3,
  PRIO_CLASS_COUNT = 4
};

struct MemReq {
  bool valid;
  RequesterId requester;
  PriorityClass prio;
  bool is_write;
  u32_t addr_word;
  u32_t len_words;
  u32_t tag;
};

struct MemGrant {
  bool valid;
  bool accept;
  RequesterId requester;
  u32_t granted_addr_word;
  u32_t granted_len_words;
  u16_t reason;
};

static inline MemReq make_empty_mem_req() {
  MemReq req;
  req.valid = false;
  req.requester = REQ_PREPROC;
  req.prio = PRIO_DEBUG_READ_MEM;
  req.is_write = false;
  req.addr_word = 0;
  req.len_words = 0;
  req.tag = 0;
  return req;
}

static inline MemGrant make_empty_mem_grant() {
  MemGrant grant;
  grant.valid = false;
  grant.accept = false;
  grant.requester = REQ_PREPROC;
  grant.granted_addr_word = 0;
  grant.granted_len_words = 0;
  grant.reason = 0;
  return grant;
}

static inline PriorityClass default_priority_of(RequesterId id) {
  switch (id) {
    case REQ_PREPROC:
      return PRIO_XIN_READ;
    case REQ_TRANSFORMER:
      return PRIO_W_PREFETCH;
    case REQ_LAYER_NORM:
      return PRIO_NECESSARY_WRITEBACK;
    case REQ_FINAL_HEAD:
      return PRIO_NECESSARY_WRITEBACK;
    case REQ_DEBUG_READ_MEM:
      return PRIO_DEBUG_READ_MEM;
    default:
      return PRIO_DEBUG_READ_MEM;
  }
}

} // namespace aecct
