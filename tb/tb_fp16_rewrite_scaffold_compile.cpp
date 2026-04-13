#include <cstdio>

#include "Fp16RewriteTopContract.h"
#include "Fp16WeightProvider.h"
#include "TopManagedWindowTypes.h"

int main() {
    const aecct::fp16_rewrite::TopRewriteIoContract contract =
        aecct::fp16_rewrite::make_default_top_rewrite_io_contract();
    aecct::fp16_rewrite::HeaderFp16PreprocWeightProvider provider;
    const aecct::fp16_rewrite::fp16_t embed00 = provider.src_embed(0u, 0u);
    const aecct::fp16_rewrite::fp16_t lpe00 = provider.lpe_token(0u, 0u);

    aecct::fp16_rewrite::Fp16WindowPacket packet;
    aecct::fp16_rewrite::clear_fp16_window_packet(packet);
    packet.ctx = aecct::fp16_rewrite::make_window_context(
        (uint32_t)aecct::PHASE_PREPROC,
        0u,
        0u,
        0u,
        1u,
        0u,
        1u,
        1u,
        0u,
        8u
    );
    packet.data[0] = aecct::fp16_rewrite::fp16_to_word(embed00);
    packet.data[1] = aecct::fp16_rewrite::fp16_to_word(lpe00);

    std::printf(
        "[fp16_rewrite_scaffold] ctrl_bits=%u data_bits=%u storage_word_bits=%u top_owns_sram=%u first_words=0x%04X 0x%04X\n",
        contract.ctrl_bits,
        contract.data_bits,
        contract.storage_word_bits,
        contract.top_owns_sram ? 1u : 0u,
        (unsigned)packet.data[0].to_uint(),
        (unsigned)packet.data[1].to_uint()
    );
    return 0;
}
