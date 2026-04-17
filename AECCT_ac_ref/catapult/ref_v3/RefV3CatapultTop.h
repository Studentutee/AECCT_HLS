#pragma once

#include "ac_channel.h"
#include "ref_v3/RefV3Config.h"
#include "ref_v3/RefV3Layer0AttnLnPath.h"

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
    ac_channel<RefV3AttentionTokenVectorPayload>& in_token_ch,
    ac_channel<RefV3AttentionTokenVectorPayload>& out_token_ch) {
    RefRunConfig run_cfg{};
    return layer0_attn_ln_path_.run(run_cfg, in_token_ch, out_token_ch);
  }

private:
  RefV3Layer0AttnLnPath layer0_attn_ln_path_;
};

} // namespace ref_v3
} // namespace aecct_ref
