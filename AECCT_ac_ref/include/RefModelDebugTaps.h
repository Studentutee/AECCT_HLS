#pragma once

#include <cstdint>

namespace aecct_ref {

// Optional compare/debug taps. Mainline inference only requires logits/x_pred.
struct RefModelDebugTaps {
  double* out_finalhead_s_t = nullptr;
  double* out_end_norm = nullptr;
  double* out_layer1_ffn_ln_out = nullptr;
  double* out_layer0_ffn_ln_out = nullptr;
  double* out_layer0_ffn1_out = nullptr;
  double* out_layer0_relu_out = nullptr;
  double* out_layer0_ffn2_out = nullptr;
  double* out_layer0_ffn_w2_quant_raw_out = nullptr;
  double* out_layer0_ffn_w2_quant_raw_qx = nullptr;
  double* out_layer0_ffn_w2_quant_raw_weight_scaled = nullptr;
  double* out_layer0_ffn_w2_quant_raw_weight_bits = nullptr;
  double* out_layer0_ffn_w2_quant_raw_weight_scaled_bits = nullptr;
  double* out_layer0_ffn_w2_quant_raw_bias_domain = nullptr;
  double* out_layer0_ffn_w2_quant_raw_partial_acc_focus = nullptr;
  double* out_layer0_ffn_w2_quant_raw_sx_bits = nullptr;
  double* out_layer0_ffn_w2_quant_raw_sw_bits = nullptr;
  double* out_layer0_ffn_w2_quant_raw_inv_bits = nullptr;
  bool layer0_w2_raw_scale_bits_override_valid = false;
  uint32_t layer0_w2_raw_sx_bits_override = 0u;
  uint32_t layer0_w2_raw_inv_bits_override = 0u;
  double* out_layer0_attn_input = nullptr;
  double* out_layer0_post_concat = nullptr;
  double* out_layer0_attn_out = nullptr;
  double* out_layer0_pre_ln_input = nullptr;
  double* out_layer0_ln_out = nullptr;
  double* out_layer0_residual_add_out = nullptr;
  double* out_layer0_residual_add_dut_aligned_out = nullptr;
  double* out_layer0_sublayer1_ln_in = nullptr;
  double* out_layer0_sublayer1_ln_in_dut_aligned = nullptr;
  double* out_layer0_sublayer1_ln_out_dut_aligned = nullptr;
  double* out_layer0_mid_norm_dut_aligned = nullptr;
  double* out_layer1_ffn2_out = nullptr;
  double* out_layer1_attn_out = nullptr;
  double* out_layer1_post_concat = nullptr;
  double* out_layer1_q = nullptr;
  double* out_layer1_attn_input = nullptr;
  double* out_layer1_attn_input_dut_aligned = nullptr;
  double* out_layer1_pre_ln_input = nullptr;
  double* out_layer1_pre_ln_input_dut_aligned = nullptr;
  double* out_layer1_ln_out = nullptr;
  double* out_layer1_ln0_out_dut_aligned = nullptr;
  double* out_layer1_ffn1_out = nullptr;
  double* out_layer1_relu_out = nullptr;
  double* out_layer1_sublayer1_ln_in_dut_aligned = nullptr;
  double* out_layer1_sublayer1_ln_affine_out_dut_aligned = nullptr;
  double* out_layer1_sublayer1_ln_out_dut_aligned = nullptr;
};

} // namespace aecct_ref
