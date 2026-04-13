#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <vector>

#include "blocks/PreprocEmbedSPE.h"
#include "tb_fp16_branch_word16_common.h"
#include "AECCT_ac_ref/include/RefModel.h"

namespace {

static bool build_ref_input_fp32(const uint32_t sample_idx, std::vector<double>& ref_input_fp32) {
    std::vector<uint32_t> infer_words_u32;
    if (!fp16_branch_tb::build_infer_input_words_u32(sample_idx, infer_words_u32)) {
        return false;
    }
    ref_input_fp32.assign(infer_words_u32.size(), 0.0);
    for (uint32_t i = 0u; i < (uint32_t)infer_words_u32.size(); ++i) {
        ref_input_fp32[i] = (double)aecct::fp32_from_bits((aecct::u32_t)infer_words_u32[i]).to_float();
    }
    return true;
}

int run_probe(uint32_t sample_idx) {
    std::vector<uint32_t> infer_words_u32;
    if (!fp16_branch_tb::build_infer_input_words_u32(sample_idx, infer_words_u32)) {
        return 1;
    }

    aecct::u32_t input_words[CODE_N];
    for (uint32_t i = 0u; i < (uint32_t)CODE_N; ++i) {
        input_words[i] = (aecct::u32_t)infer_words_u32[i];
    }

    aecct::fp16_rewrite::HeaderFp16PreprocWeightProvider weights;
    aecct::preproc_clean::PreprocFp16Debug dbg;
    aecct::preproc_clean::run_preproc_fp16_clean_from_words(input_words, weights, dbg);

    std::vector<double> ref_input_fp32;
    if (!build_ref_input_fp32(sample_idx, ref_input_fp32)) {
        return 1;
    }
    const uint32_t tensor_words = (uint32_t)N_NODES * (uint32_t)D_MODEL;
    std::vector<double> ref_layer0_attn_input(tensor_words, 0.0);
    std::vector<double> ref_logits((uint32_t)EXP_LEN_OUT_LOGITS_WORDS, 0.0);
    std::vector<aecct_ref::bit1_t> ref_xpred((uint32_t)EXP_LEN_OUT_XPRED_WORDS);

    aecct_ref::RefModel ref_model;
    ref_model.set_run_config(aecct_ref::make_fp16_experiment_run_config());
    aecct_ref::RefModelIO ref_io;
    ref_io.input_y = nullptr;
    ref_io.input_y_fp32 = ref_input_fp32.data();
    ref_io.out_logits = ref_logits.data();
    ref_io.out_x_pred = ref_xpred.data();
    ref_io.out_layer0_attn_input = ref_layer0_attn_input.data();
    ref_io.out_layer0_attn_out = nullptr;
    ref_io.out_layer0_residual_add_dut_aligned_out = nullptr;
    ref_io.B = 1;
    ref_io.N = (int)EXP_LEN_OUT_XPRED_WORDS;
    ref_model.infer_step0(ref_io);

    const uint32_t token = 0u;
    const uint32_t d = 1u;
    const aecct::fp16_t node = dbg.node_feature[token];
    const aecct::fp16_t embed = weights.src_embed(token, d);
    const aecct::fp16_t mul = aecct::preproc_clean::preproc_mul_fp16(node, embed);
    const uint16_t got = (uint16_t)aecct::bits_from_fp16(dbg.x_work[token][d]).to_uint();
    const uint16_t stage_local = (uint16_t)fp16_branch_tb::ref_preproc_x_fp16_bits(sample_idx, token, d).to_uint();
    const uint16_t ref_model_lane = fp16_branch_tb::fp16_lane_from_double_tb(ref_layer0_attn_input[token * (uint32_t)D_MODEL + d]);

    std::printf("[preproc_gap_probe] sample=%u token=%u d=%u\n", (unsigned)sample_idx, (unsigned)token, (unsigned)d);
    std::printf("  node_bits=0x%04X embed_bits=0x%04X mul_bits=0x%04X got_bits=0x%04X\n",
                (unsigned)((uint16_t)aecct::bits_from_fp16(node).to_uint()),
                (unsigned)((uint16_t)aecct::bits_from_fp16(embed).to_uint()),
                (unsigned)((uint16_t)aecct::bits_from_fp16(mul).to_uint()),
                (unsigned)got);
    std::printf("  stage_local_bits=0x%04X refmodel_bits=0x%04X\n",
                (unsigned)stage_local,
                (unsigned)ref_model_lane);
    return 0;
}

} // namespace

int main() {
    return run_probe(0u);
}

#endif
