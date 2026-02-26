#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "gen/ModelShapes.h"
#include "SoftmaxApprox.h"
#include "VerifyTolerance.h"

static void softmax_ref(
    const float* scores,
    float* probs,
    unsigned len
) {
    float m = scores[0];
    for (unsigned i = 1; i < len; ++i) {
        if (scores[i] > m) {
            m = scores[i];
        }
    }

    float sum = 0.0f;
    for (unsigned i = 0; i < len; ++i) {
        float e = std::exp(scores[i] - m);
        probs[i] = e;
        sum += e;
    }
    for (unsigned i = 0; i < len; ++i) {
        probs[i] /= sum;
    }
}

template <unsigned MAX_LEN>
static void run_case(
    const char* name,
    const float* scores_f,
    unsigned len
) {
    softmax_score_t in_q[MAX_LEN];
    softmax_prob_t out_q[MAX_LEN];
    float ref[MAX_LEN];

    for (unsigned i = 0; i < len; ++i) {
        in_q[i] = softmax_score_t(scores_f[i]);
    }

    SoftmaxApprox<MAX_LEN>(in_q, out_q, len);
    softmax_ref(scores_f, ref, len);

    double max_abs_err = 0.0;
    unsigned max_idx = 0;
    double sum_q = 0.0;
    for (unsigned i = 0; i < len; ++i) {
        double q = out_q[i].to_double();
        double e = std::fabs(q - (double)ref[i]);
        sum_q += q;
        if (e > max_abs_err) {
            max_abs_err = e;
            max_idx = i;
        }
    }

    if (max_abs_err > EPS_SOFTMAX) {
        std::printf("ERROR: %s max_abs_err=%.9g idx=%u (tol=%.9g)\n",
            name, max_abs_err, max_idx, (double)EPS_SOFTMAX);
        std::exit(1);
    }
    if (std::fabs(sum_q - 1.0) > 5.0e-2) {
        std::printf("ERROR: %s sum(prob)=%.9g not close to 1.0\n", name, sum_q);
        std::exit(1);
    }

    std::printf("PASS: %s (max_abs_err=%.9g)\n", name, max_abs_err);
}

int main() {
    static float case0[8];
    static float case1[16];
    static float case2[N_NODES];

    for (unsigned i = 0; i < 8; ++i) {
        case0[i] = 0.2f * (float)i - 0.6f;
    }
    for (unsigned i = 0; i < 16; ++i) {
        case1[i] = ((i % 5) - 2) * 0.7f;
    }
    for (unsigned i = 0; i < (unsigned)N_NODES; ++i) {
        float v = (float)((int)(i % 13) - 6) * 0.33f;
        case2[i] = v + 0.1f * std::sin((double)i * 0.17);
    }

    run_case<8>("softmax_case0_len8", case0, 8u);
    run_case<16>("softmax_case1_len16", case1, 16u);
    run_case<N_NODES>("softmax_case2_len75", case2, (unsigned)N_NODES);

    std::printf("PASS: tb_softmax_m15a\n");
    return 0;
}

