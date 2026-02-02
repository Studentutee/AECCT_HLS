#pragma once
#include <cstdio>

#ifndef MAX_MISMATCH_PRINT
#define MAX_MISMATCH_PRINT 10
#endif

// ------------------------------------------------------------
// to_double helpers
// ------------------------------------------------------------
// 1) 一般 scalar：直接 static_cast<double>
template <typename T>
static inline double to_double(const T& x) {
    return static_cast<double>(x);
}

// 2) ac_fixed / ac_int：用 .to_double()
//    這兩個型別都有提供 to_double() 成員函式
template <int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
static inline double to_double(const ac_fixed<W, I, S, Q, O>& x) {
    return x.to_double();
}

template <int W, bool S>
static inline double to_double(const ac_int<W, S>& x) {
    return x.to_double();
}

// -------------------------------
// 核心比對：absolute error
// -------------------------------
template <typename ExpT, typename ActT>
static int compare_array_abs(
    const ExpT* expected,
    const ActT* actual,
    int n,
    double atol,
    const char* tensor_name)
{
    int mismatches = 0;
    double max_err = 0.0;
    int max_idx = -1;

    for (int i = 0; i < n; i++) {
        const double e = to_double(expected[i]);
        const double a = to_double(actual[i]);
        const double err = std::fabs(a - e);
        //printf("actual[%d]=%.10g expected[%d]=%.10g |err|=%.10g\n", i, a, i, e, err);
        if (err > max_err) { max_err = err; max_idx = i; }

        if (err > atol) {
            if (mismatches < MAX_MISMATCH_PRINT) {
                std::printf("[MIS] %s[%d]: exp=%.10g act=%.10g |err|=%.10g\n",
                    tensor_name, i, e, a, err);
            }
            mismatches++;

#if FAIL_FAST
            break;
#endif
        }
    }

    if (mismatches == 0) {
        std::printf("[OK ] %s: n=%d max|err|=%.10g @%d\n",
            tensor_name, n, max_err, max_idx);
    }
    else {
        std::printf("[BAD] %s: n=%d mismatches=%d max|err|=%.10g @%d\n",
            tensor_name, n, mismatches, max_err, max_idx);
    }

    return mismatches;
}