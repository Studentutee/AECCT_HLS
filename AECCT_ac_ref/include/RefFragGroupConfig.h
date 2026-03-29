#pragma once

namespace aecct_ref {

enum class RefFragGroup : unsigned char {
  NONE = 0,
  G1_LAYERNORM = 1,
  G2_RESIDUAL = 2,
  G3_ATTN_CONTEXT = 3,
  G4_SOFTMAX_NEIGHBORHOOD = 4,
  G5_PREPROC_EMBED = 5,
  C1_G1_G2 = 6,
  C2_G1_G3 = 7,
  C3_G2_G3 = 8,
  C4_G1_G4 = 9
};

static inline const char* to_string(RefFragGroup g) {
  switch (g) {
    case RefFragGroup::NONE: return "NONE";
    case RefFragGroup::G1_LAYERNORM: return "G1";
    case RefFragGroup::G2_RESIDUAL: return "G2";
    case RefFragGroup::G3_ATTN_CONTEXT: return "G3";
    case RefFragGroup::G4_SOFTMAX_NEIGHBORHOOD: return "G4";
    case RefFragGroup::G5_PREPROC_EMBED: return "G5";
    case RefFragGroup::C1_G1_G2: return "C1";
    case RefFragGroup::C2_G1_G3: return "C2";
    case RefFragGroup::C3_G2_G3: return "C3";
    case RefFragGroup::C4_G1_G4: return "C4";
    default: return "UNKNOWN";
  }
}

} // namespace aecct_ref

