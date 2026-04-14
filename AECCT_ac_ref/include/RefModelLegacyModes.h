#pragma once

#include "RefAlgoVariant.h"
#include "RefFragGroupConfig.h"
#include "RefLayerNormMode.h"
#include "RefSoftmaxExpMode.h"
#include "RefStageConfig.h"

namespace aecct_ref {

// Legacy/non-mainline experiment knobs are grouped here so the main RefModel
// contract can keep precision_mode as the primary control surface.
struct RefLegacyRunConfig {
  RefAlgoVariant algo_variant = RefAlgoVariant::BASELINE_SPEC_FLOW;
  RefSoftmaxExpMode softmax_exp_mode = RefSoftmaxExpMode::BASELINE_NEAREST_LUT;
  RefLayerNormMode ln_mode = RefLayerNormMode::LN_BASELINE;
  RefFinalHeadExploreStage finalhead_stage = RefFinalHeadExploreStage::S0;
  RefFragGroup frag_group = RefFragGroup::NONE;
};

} // namespace aecct_ref
