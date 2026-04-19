#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3FfnLinear0ReluBlock.h"
#include "ref_v3/RefV3Payload.h"

namespace aecct_ref {
namespace ref_v3 {

bool ref_v3_preproc_block_top(ac_channel<RefV3PreprocInputPayload>& in_input_ch,
                              ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch,
                              ac_channel<RefV3AttentionInputPayload>& out_xwork_ch);

bool ref_v3_atten_kv_block_top(int lid,
                               ac_channel<RefV3AttentionTokenVectorPayload>& in_x_token_ch,
                               ac_channel<RefV3AttentionKPayload>& out_k_payload_ch,
                               ac_channel<RefV3AttentionVPayload>& out_v_payload_ch);

bool ref_v3_atten_qsoftres_block_top(int lid,
                                     ac_channel<RefV3AttentionInputPayload>& in_xwork_ch,
                                     ac_channel<RefV3AttentionKPayload>& in_k_payload_ch,
                                     ac_channel<RefV3AttentionVPayload>& in_v_payload_ch,
                                     ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch);

bool ref_v3_layernorm_block_top(int lid,
                                ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                                ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch);

bool ref_v3_ffn_linear0_relu_block_top(int lid,
                                        ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                                        ac_channel<RefV3FfnHiddenTokenPayload>& out_hidden_ch);

bool ref_v3_ffn_linear1_residual_block_top(
  int lid,
  ac_channel<RefV3FfnHiddenTokenPayload>& in_hidden_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& in_residual_token_ch,
  ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch);

bool ref_v3_final_pass_a_block_top(ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
                                   ac_channel<RefV3FinalScalarTokenPayload>& out_scalar_ch);

bool ref_v3_final_pass_b_block_top(ac_channel<RefV3FinalScalarTokenPayload>& in_scalar_ch,
                                   ac_channel<RefV3FinalInputYPayload>& in_input_y_ch,
                                   ac_channel<RefV3FinalOutputPayload>& out_payload_ch);

} // namespace ref_v3
} // namespace aecct_ref
