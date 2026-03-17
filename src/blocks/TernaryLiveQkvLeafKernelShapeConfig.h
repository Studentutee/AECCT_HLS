#pragma once
// P00-011T: compile-time shape SSOT for current supported QKV live-cut/compile-prep build.
// Runtime metadata/config performs validation against these compile-time supported shapes only.
// This header does not imply runtime-variable top interfaces.

#include <cstdint>

namespace aecct {

// Current compile-time supported shapes for QKV split/materialize surfaces.
static constexpr uint32_t kQkvCtSupportedL0WqRows = 32u;
static constexpr uint32_t kQkvCtSupportedL0WqCols = 32u;
static constexpr uint32_t kQkvCtSupportedL0WqPayloadWords = 64u;

static constexpr uint32_t kQkvCtSupportedL0WkRows = 32u;
static constexpr uint32_t kQkvCtSupportedL0WkCols = 32u;
static constexpr uint32_t kQkvCtSupportedL0WkPayloadWords = 64u;

static constexpr uint32_t kQkvCtSupportedL0WvRows = 32u;
static constexpr uint32_t kQkvCtSupportedL0WvCols = 32u;
static constexpr uint32_t kQkvCtSupportedL0WvPayloadWords = 64u;

// Legacy-style names are re-exports from SSOT only.
// They must remain aliases and not become a second active definition point.
static constexpr uint32_t kTernaryLiveL0WqRows = kQkvCtSupportedL0WqRows;
static constexpr uint32_t kTernaryLiveL0WqCols = kQkvCtSupportedL0WqCols;
static constexpr uint32_t kTernaryLiveL0WqPayloadWords = kQkvCtSupportedL0WqPayloadWords;

static constexpr uint32_t kTernaryLiveL0WkRows = kQkvCtSupportedL0WkRows;
static constexpr uint32_t kTernaryLiveL0WkCols = kQkvCtSupportedL0WkCols;
static constexpr uint32_t kTernaryLiveL0WkPayloadWords = kQkvCtSupportedL0WkPayloadWords;

static constexpr uint32_t kTernaryLiveL0WvRows = kQkvCtSupportedL0WvRows;
static constexpr uint32_t kTernaryLiveL0WvCols = kQkvCtSupportedL0WvCols;
static constexpr uint32_t kTernaryLiveL0WvPayloadWords = kQkvCtSupportedL0WvPayloadWords;

} // namespace aecct
