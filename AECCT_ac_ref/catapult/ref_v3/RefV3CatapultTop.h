#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Config.h"
#include "ref_v3/RefV3FinalPassABlock.h"
#include "ref_v3/RefV3FinalPassBBlock.h"
#include "ref_v3/RefV3Layer0AttnLnPath.h"
#include "ref_v3/RefV3Layer0FfnPath.h"
#include "ref_v3/RefV3Layer1AttnLnPath.h"
#include "ref_v3/RefV3Layer1FfnPath.h"
#include "ref_v3/RefV3LayerNormBlock.h"
#include "ref_v3/RefV3MidNormPath.h"
#include "ref_v3/RefV3PreprocBlock.h"

#if defined(__has_include)
#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#endif
#endif

#ifndef CCS_BLOCK
#define CCS_BLOCK(name) name
#endif

namespace aecct_ref {
namespace ref_v3 {

#pragma hls_design top
class RefV3CatapultTop {
public:
  RefV3CatapultTop() {}

#pragma hls_design interface
  bool CCS_BLOCK(run)(
    ac_channel<RefV3PreprocInputPayload>& in_preproc_ch,
    ac_channel<RefV3FinalOutputPayload>& out_payload_ch) {
    RefRunConfig run_cfg{};
    ac_channel<RefV3PreprocInputPayload> ch_preproc_in_main;
    ac_channel<RefV3AttentionTokenVectorPayload> ch_preproc_to_l0_attn;
    ac_channel<RefV3AttentionInputPayload> ch_xwork0_side;
    ac_channel<RefV3AttentionTokenVectorPayload> ch_l0_attn_to_ffn;
    ac_channel<RefV3AttentionTokenVectorPayload> ch_l0_ffn_to_midnorm;
    ac_channel<RefV3AttentionTokenVectorPayload> ch_midnorm_to_l1_attn;
    ac_channel<RefV3AttentionInputPayload> ch_xwork1_side;
    ac_channel<RefV3AttentionTokenVectorPayload> ch_l1_attn_to_ffn;
    ac_channel<RefV3AttentionTokenVectorPayload> ch_l1_ffn_to_endnorm;
    ac_channel<RefV3AttentionTokenVectorPayload> ch_endnorm_to_finala;
    ac_channel<RefV3FinalScalarTokenPayload> ch_finala_to_finalb;
    ac_channel<RefV3FinalInputYPayload> ch_finalb_input_y;

    // Consume preproc input once, then fan out to preproc main path and final-pass-B input.
    const RefV3PreprocInputPayload preproc_input_payload = in_preproc_ch.read();
    ch_preproc_in_main.write(preproc_input_payload);

    RefV3FinalInputYPayload final_input_y_payload;
    final_input_y_payload.var_count = preproc_input_payload.var_count;
    REFV3_CATAPULT_TOP_FINALB_INPUT_COPY_LOOP: for (int n = 0; n < REFV3_VAR_N; ++n) {
      final_input_y_payload.input_y[n] = preproc_input_payload.input_y[n];
    }
    ch_finalb_input_y.write(final_input_y_payload);

    if (!preproc_block_.run(ch_preproc_in_main, ch_preproc_to_l0_attn, ch_xwork0_side)) {
      return false;
    }

    if (!layer0_attn_ln_path_.run(run_cfg, ch_preproc_to_l0_attn, ch_xwork0_side, ch_l0_attn_to_ffn)) {
      return false;
    }
    if (!layer0_ffn_path_.run(ch_l0_attn_to_ffn, ch_l0_ffn_to_midnorm)) {
      return false;
    }
    if (!mid_norm_path_.run(run_cfg, ch_l0_ffn_to_midnorm, ch_midnorm_to_l1_attn, ch_xwork1_side)) {
      return false;
    }

    if (!layer1_attn_ln_path_.run(run_cfg, ch_midnorm_to_l1_attn, ch_xwork1_side, ch_l1_attn_to_ffn)) {
      return false;
    }
    if (!layer1_ffn_path_.run(ch_l1_attn_to_ffn, ch_l1_ffn_to_endnorm)) {
      return false;
    }
    if (!end_norm_block_.run(REFV3_LAYER1_ID, run_cfg, ch_l1_ffn_to_endnorm, ch_endnorm_to_finala)) {
      return false;
    }
    if (!final_pass_a_block_.run(ch_endnorm_to_finala, ch_finala_to_finalb)) {
      return false;
    }
    if (!final_pass_b_block_.run(ch_finala_to_finalb, ch_finalb_input_y, out_payload_ch)) {
      return false;
    }
    return true;
  }

private:
  RefV3PreprocBlock preproc_block_;
  RefV3Layer0AttnLnPath layer0_attn_ln_path_;
  RefV3Layer0FfnPath layer0_ffn_path_;
  RefV3MidNormPath mid_norm_path_;
  RefV3Layer1AttnLnPath layer1_attn_ln_path_;
  RefV3Layer1FfnPath layer1_ffn_path_;
  RefV3LayerNormBlock end_norm_block_;
  RefV3FinalPassABlock final_pass_a_block_;
  RefV3FinalPassBBlock final_pass_b_block_;
};

} // namespace ref_v3
} // namespace aecct_ref
