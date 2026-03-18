// P00-011AD-prep: Q-path scaffold (compile/run independent, local-only).

#include <cstdint>
#include <cstdio>

namespace {

struct MockQAdapterPacket {
    uint16_t token;
    uint16_t d_tile;
    uint32_t words[8];
};

static uint32_t q_path_mock_checksum(const MockQAdapterPacket& p) {
    uint32_t acc = ((uint32_t)p.token << 16) ^ (uint32_t)p.d_tile;
    for (unsigned i = 0; i < 8u; ++i) {
        acc ^= (p.words[i] + (uint32_t)(i * 131u));
    }
    return acc;
}

} // namespace

int main() {
    MockQAdapterPacket p;
    p.token = 1u;
    p.d_tile = 0u;
    for (unsigned i = 0; i < 8u; ++i) {
        p.words[i] = (uint32_t)(i + 3u) * 17u;
    }
    const uint32_t c = q_path_mock_checksum(p);
    if (c == 0u) {
        std::printf("ERROR: tb_q_path_scaffold_p11ad_prep checksum invalid\n");
        return 1;
    }
    std::printf("PASS: tb_q_path_scaffold_p11ad_prep\n");
    return 0;
}
