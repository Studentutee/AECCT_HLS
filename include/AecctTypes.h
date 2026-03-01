#pragma once
// Core AC datatypes and channel aliases.

#include <ac_channel.h>
#include <ac_int.h>

namespace aecct {

typedef ac_int<16, false> u16_t;
typedef ac_int<32, false> u32_t;

typedef ac_channel<u16_t> ctrl_ch_t;
typedef ac_channel<u32_t> data_ch_t;

static inline u32_t u32_from_uint(uint32_t v) { return (u32_t)v; }
static inline uint32_t uint_from_u32(const u32_t& v) { return (uint32_t)v.to_uint(); }

} // namespace aecct
