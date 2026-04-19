#pragma once

#include "ref_v3/RefV3Types.h"

namespace aecct_ref {
namespace ref_v3 {

// local-only boundary cache: converts exported weight payloads to refv3_fp_t once.
const refv3_fp_t* refv3_preproc_src_embed_fp_local_only();
const refv3_fp_t* refv3_preproc_lpe_token_fp_local_only();

refv3_fp_t refv3_attn_input_s_x_fp_local_only(int lid);
refv3_fp_t refv3_attn_output_s_x_fp_local_only(int lid);
refv3_fp_t refv3_ffn_w1_s_x_fp_local_only(int lid);
refv3_fp_t refv3_ffn_w2_s_x_fp_local_only(int lid);

RefV3TernaryLinearParams refv3_attn_linear_params_fp_local_only(int lid, int linear_id);

RefV3TernaryLinearParams refv3_layernorm0_params_fp_local_only(int lid);
RefV3TernaryLinearParams refv3_layernorm1_params_fp_local_only(int lid);
RefV3TernaryLinearParams refv3_midnorm_params_fp_local_only();
RefV3TernaryLinearParams refv3_endnorm_params_fp_local_only();
RefV3TernaryLinearParams refv3_ffn_w1_params_fp_local_only(int lid);
RefV3TernaryLinearParams refv3_ffn_w2_params_fp_local_only(int lid);

const refv3_fp_t* refv3_final_embed_weight_fp_local_only();
refv3_fp_t refv3_final_embed_bias_fp_local_only();

const refv3_fp_t* refv3_out_fc_weight_fp_local_only();
const refv3_fp_t* refv3_out_fc_bias_fp_local_only();

bool refv3_src_mask_bit_local_only(int q_token, int k_token);
bool refv3_h_parity_edge_local_only(int check_idx, int var_idx);

} // namespace ref_v3
} // namespace aecct_ref
