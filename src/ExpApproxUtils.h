#pragma once

#include "ac_fixed.h"
#include "ac_int.h"

namespace fx_exp {

	typedef ac_fixed<32, 16, true,  AC_RND_CONV, AC_SAT_SYM> fx_t;

	// exp(x) output for x<=0 is in (0, 1]; keep unsigned
	typedef ac_fixed<24, 2, false, AC_RND_CONV, AC_SAT_SYM> ufx_t; // 先保留你原本的範圍(<=4)，避免你只改header就變更糟

	// denom accumulator for softmax (use this in L0_AttnCore!)
	typedef ac_fixed<32, 8, false, AC_RND_CONV, AC_SAT_SYM> denom_t; // covers up to 255

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

	// Approx exp(x) for x in [-8,0]
	static inline ufx_t exp_neg_approx(fx_t x) {
		if (x > fx_t(0))  x = fx_t(0);
		if (x < fx_t(-8)) x = fx_t(-8);

		// t = -x*2 in [0..16]
		fx_t t = fx_t(-x) * fx_t(2);

		ac_int<6,false> idx = ac_int<6,false>(t.to_int());
		if (idx > 15) idx = 15;

		fx_t frac = t - fx_t(idx); // [0,1)

		fx_t a = fx_t(EXP_NEG_LUT[idx]);
		fx_t b = fx_t(EXP_NEG_LUT[idx + 1]);

		// y = a + (b-a)*frac   (all signed math)
		fx_t y = a + (b - a) * frac;

		// cast back to unsigned exp output
		return ufx_t(y);
	}

} // namespace fx_exp
