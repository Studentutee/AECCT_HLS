#pragma once

#include "ac_fixed.h"
#include "ac_int.h"

// ============================================================
// exp approximation for softmax (synth-friendly)
// - clamp x to [-8, 0]
// - LUT step = 0.5
// - linear interpolation
//
// IMPORTANT:
//   exp(x) output is in (0, 1] for x<=0, so its type should not be
//   the same as softmax denominator accumulator.
// ============================================================

namespace fx_exp {

	// Signed input type for x (logits diff)
	typedef ac_fixed<32, 16, true,  AC_RND_CONV, AC_SAT_SYM> fx_t;

	// exp(x) for x<=0 is in (0,1]; keep it unsigned with enough fraction bits
	// Range: [0, 1]
	typedef ac_fixed<24, 1, false, AC_RND_CONV, AC_SAT_SYM> ufx_t;

	// Softmax denominator needs to accumulate up to N terms (e.g., 75)
	// Range suggestion: [0, 256)
	typedef ac_fixed<32, 8, false, AC_RND_CONV, AC_SAT_SYM> denom_t;

	// Fraction type for interpolation in [0,1)
	typedef ac_fixed<18, 1, false, AC_RND_CONV, AC_SAT_SYM> frac_t;

	static const ufx_t EXP_NEG_LUT[17] = {
		ufx_t(1.0),
		ufx_t(0.6065306597126334),
		ufx_t(0.36787944117144233),
		ufx_t(0.22313016014842982),
		ufx_t(0.1353352832366127),
		ufx_t(0.0820849986238988),
		ufx_t(0.049787068367863944),
		ufx_t(0.0301973834223185),
		ufx_t(0.01831563888873418),
		ufx_t(0.011108996538242306),
		ufx_t(0.006737946999085467),
		ufx_t(0.004086771438464067),
		ufx_t(0.0024787521766663585),
		ufx_t(0.0015034391929775724),
		ufx_t(0.0009118819655545162),
		ufx_t(0.0005530843701478336),
		ufx_t(0.00033546262790251185)
	};

	// Approx exp(x) for x in [-8,0] using LUT step=0.5 with linear interpolation.
	static inline ufx_t exp_neg_approx(fx_t x) {
		// clamp to [-8,0]
		if (x > fx_t(0))   x = fx_t(0);
		if (x < fx_t(-8))  x = fx_t(-8);

		// t = -x*2 in [0..16]
		fx_t t = fx_t(-x) * fx_t(2);

		// idx = floor(t) in [0..15]
		ac_int<6,false> idx = ac_int<6,false>(t.to_int());
		if (idx > 15) idx = 15;

		// frac = t - idx in [0,1)
		frac_t frac = frac_t(t - fx_t(idx));

		ufx_t a = EXP_NEG_LUT[idx];
		ufx_t b = EXP_NEG_LUT[idx + 1];

		// linear interpolation: a + (b-a)*frac
		// (use ufx_t/frac_t to keep unsigned behavior)
		ufx_t y = ufx_t(a + ufx_t((b - a) * ufx_t(frac)));
		return y;
	}

} // namespace fx_exp
