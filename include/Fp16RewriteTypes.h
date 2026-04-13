#pragma once
// FP16 rewrite common types.
//
// Purpose:
// - give the clean rewrite path one explicit type/header boundary
// - keep fp16 compute helpers separate from legacy fp32/u32 carrier helpers
// - keep Top-owned SRAM word semantics visible to every new block
//
// Scope:
// - local rewrite scaffolding only
// - not Catapult closure
// - not SCVerify closure

#include <cstdint>

#include <ac_int.h>
#include <ac_std_float.h>

#include "AecctTypes.h"
#include "gen/ModelShapes.h"

namespace aecct {
namespace fp16_rewrite {

typedef ac_std_float<16, 5> fp16_t;
typedef ac_int<16, false> fp16_bits_t;

static const uint32_t FP16_BITS = 16u;
static const uint32_t FP16_WORDS_PER_ELEM = 1u;

static_assert(SRAM_STORAGE_WORD_BITS == 16u, "fp16 rewrite assumes 16-bit SRAM words");
static_assert(FP16_WORDS_PER_ELEM == 1u, "fp16 rewrite assumes 1 fp16 == 1 storage word");

static inline fp16_bits_t fp16_to_bits(const fp16_t& value) {
    return value.data_ac_int();
}

static inline fp16_t fp16_from_bits(const fp16_bits_t& bits) {
    fp16_t value;
    value.set_data(bits);
    return value;
}

static inline u16_t fp16_to_word(const fp16_t& value) {
    return (u16_t)fp16_to_bits(value);
}

static inline fp16_t fp16_from_word(const u16_t& word) {
    return fp16_from_bits((fp16_bits_t)word);
}

static inline fp16_t fp16_from_double(const double value) {
    return fp16_t((float)value);
}

static inline double fp16_to_double(const fp16_t& value) {
    return (double)value.to_double();
}

} // namespace fp16_rewrite
} // namespace aecct
