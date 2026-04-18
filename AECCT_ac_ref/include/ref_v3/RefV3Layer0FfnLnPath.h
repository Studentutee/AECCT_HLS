#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Config.h"
#include "ref_v3/RefV3LayerNormBlock.h"
#include "ref_v3/RefV3WeightsFp16LocalOnly.h"

namespace aecct_ref {
namespace ref_v3 {

class RefV3Layer0FfnLnPath {
public:
  RefV3Layer0FfnLnPath() {}

  // Layer0 post-FFN LN stage uses layer0 sublayer_1_norm parameters.
  bool run(const RefRunConfig& run_cfg,
           ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
           ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    const RefV3TernaryLinearParams layer0_ffnln_params =
      refv3_layernorm1_params_fp_local_only(REFV3_LAYER0_ID);
    return ln_block_.run_with_params(
      REFV3_LAYER0_ID,
      layer0_ffnln_params,
      run_cfg,
      in_token_ch,
      out_token_ch);
  }

private:
  RefV3LayerNormBlock ln_block_;
};

} // namespace ref_v3
} // namespace aecct_ref
