#pragma once
// Minimal HLS-safe range types (half-open interval [begin, end)).

#include "AecctTypes.h"

namespace aecct {

struct TokenRange {
  u32_t begin;
  u32_t end;
};

struct TileRange {
  u32_t begin;
  u32_t end;
};

static inline TokenRange make_token_range(u32_t begin, u32_t end) {
  TokenRange r;
  r.begin = begin;
  r.end = end;
  return r;
}

static inline TileRange make_tile_range(u32_t begin, u32_t end) {
  TileRange r;
  r.begin = begin;
  r.end = end;
  return r;
}

static inline bool token_range_valid(const TokenRange& r) {
  return r.begin <= r.end;
}

static inline bool tile_range_valid(const TileRange& r) {
  return r.begin <= r.end;
}

} // namespace aecct
