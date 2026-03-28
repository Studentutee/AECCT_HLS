#pragma once

namespace aecct_ref {

enum class RefFinalHeadExploreStage : unsigned char {
  S0 = 0, // existing: s_t roundtrip before out_fc_in
  S1 = 1, // out_fc local multiplicand / pre-MAC
  S2 = 2, // S0 + S1
  S3 = 3, // oned_final_embed_out / local s_t generation output boundary
  S4 = 4  // S0 + S1 + S3 (FinalHead-local combined)
};

static inline const char* to_string(RefFinalHeadExploreStage stage) {
  switch (stage) {
    case RefFinalHeadExploreStage::S0: return "S0";
    case RefFinalHeadExploreStage::S1: return "S1";
    case RefFinalHeadExploreStage::S2: return "S2";
    case RefFinalHeadExploreStage::S3: return "S3";
    case RefFinalHeadExploreStage::S4: return "S4";
    default: return "UNKNOWN_STAGE";
  }
}

static inline bool stage_uses_island_s0(RefFinalHeadExploreStage stage) {
  return stage == RefFinalHeadExploreStage::S0 ||
         stage == RefFinalHeadExploreStage::S2 ||
         stage == RefFinalHeadExploreStage::S4;
}

static inline bool stage_uses_island_s1(RefFinalHeadExploreStage stage) {
  return stage == RefFinalHeadExploreStage::S1 ||
         stage == RefFinalHeadExploreStage::S2 ||
         stage == RefFinalHeadExploreStage::S4;
}

static inline bool stage_uses_island_s3(RefFinalHeadExploreStage stage) {
  return stage == RefFinalHeadExploreStage::S3 ||
         stage == RefFinalHeadExploreStage::S4;
}

} // namespace aecct_ref
