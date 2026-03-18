// P00-011AF-prep: softmax/output scaffold (compile/run independent, local-only).

#include <cstdint>
#include <cstdio>

namespace {

static uint32_t softmax_out_mock_accumulate(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t x = (a ^ (b << 1)) + (c * 7u);
    x ^= (x >> 3);
    return x;
}

} // namespace

int main() {
    const uint32_t v = softmax_out_mock_accumulate(7u, 11u, 13u);
    if (v == 0u) {
        std::printf("ERROR: tb_softmax_out_scaffold_p11af_prep value invalid\n");
        return 1;
    }
    std::printf("PASS: tb_softmax_out_scaffold_p11af_prep\n");
    return 0;
}
