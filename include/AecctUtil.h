#pragma once
// AecctUtil.h
// 共用小工具（header-only）
// - align_up
// - range check
// - bit helpers
// （中文為主，括號補英文關鍵字）

#include <stdint.h>

namespace aecct {

	// 對齊：把 x 往上對齊到 align 的倍數（align-up）
	static inline uint32_t align_up_u32(uint32_t x, uint32_t align) {
		// align 建議為 2 的冪（power-of-two），但這裡不強制
		if (align == 0u) { return x; }
		uint32_t r = x % align;
		return (r == 0u) ? x : (x + (align - r));
	}

	static inline bool in_range_u32(uint32_t x, uint32_t lo, uint32_t hi_inclusive) {
		return (x >= lo) && (x <= hi_inclusive);
	}

	static inline uint32_t mask_u32(unsigned w) {
		// w==32 -> 全 1；w==0 -> 0
		if (w >= 32u) return 0xFFFFFFFFu;
		if (w == 0u) return 0u;
		return (1u << w) - 1u;
	}

} // namespace aecct