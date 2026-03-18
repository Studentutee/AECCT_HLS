// P00-011AE-prep: QK/score scaffold (compile/run independent, local-only).

#include <cstdint>
#include <cstdio>

namespace {

struct MockScoreAdapterPacket {
    uint16_t token_q;
    uint16_t token_k;
    int32_t dot;
};

static int32_t mock_qk_dot(uint32_t seed_q, uint32_t seed_k) {
    int32_t q = (int32_t)((seed_q * 29u) & 0x7FFFu) - 8192;
    int32_t k = (int32_t)((seed_k * 31u) & 0x7FFFu) - 8192;
    return q * k;
}

} // namespace

int main() {
    MockScoreAdapterPacket p;
    p.token_q = 0u;
    p.token_k = 1u;
    p.dot = mock_qk_dot(3u, 5u);
    if (p.dot == 0) {
        std::printf("ERROR: tb_qk_score_scaffold_p11ae_prep dot invalid\n");
        return 1;
    }
    std::printf("PASS: tb_qk_score_scaffold_p11ae_prep\n");
    return 0;
}
