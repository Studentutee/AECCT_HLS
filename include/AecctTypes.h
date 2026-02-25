#pragma once
// AecctTypes.h
// AC datatypes、通道型別（channel typedefs）、以及一些打包/拆包 helpers
// （M0：只需要 ctrl/data 的固定 channel 型別）

#include <ac_int.h>
#include <ac_channel.h>

namespace aecct {

	// 固定介面（Fixed interfaces）
	// ctrl_cmd/rsp: 16-bit
	// data_in/out: 32-bit
	typedef ac_int<16, false> u16_t;
	typedef ac_int<32, false> u32_t;

	typedef ac_channel<u16_t> ctrl_ch_t;
	typedef ac_channel<u32_t> data_ch_t;

	// 一些通用 raw-bit helpers（常用於後續把 float bits 當 u32 傳）
	// M0 先放著不一定用到。
	static inline u32_t u32_from_uint(uint32_t v) { return (u32_t)v; }
	static inline uint32_t uint_from_u32(const u32_t& v) { return (uint32_t)v.to_uint(); }

} // namespace aecct