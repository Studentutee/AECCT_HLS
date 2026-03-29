# TOP MANAGED SRAM ARCH GAP INVENTORY (2026-03-29)

## Scope
- Surveyed Top and block boundaries focused on Top-owned SRAM dispatch semantics.
- Primary files: `src/Top.h`, `src/blocks/TransformerLayer.h`, `src/blocks/PreprocEmbedSPE.h`, `src/blocks/LayerNormBlock.h`, `src/blocks/FinalHead.h`.
- Goal: identify remaining direct-SRAM or pseudo-direct-SRAM ownership patterns that block Top-managed architecture convergence.

## Gap Table
| gap_id | file | scope | direct-SRAM or pseudo-direct-SRAM pattern | risk level | can fix tonight? | reason | recommended minimal cut |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | `src/Top.h` + block wrappers | Preproc / initial LN / FinalHead dispatch | Top called wrapper entries that self-built contracts inside block (`PreprocEmbedSPE`, `LayerNormBlock`, `FinalHead`) | HIGH | YES | Directly affects ownership boundary semantics but can be fixed with local callsite changes only | Move contract assembly to Top and dispatch CoreWindow/CorePass entries with explicit Top-owned contract |
| G2 | `src/blocks/TransformerLayer.h` + `src/Top.h` | Per-layer sublayer1 LN parameter preload | Block loaded LN gamma/beta from param base (`load_layer_sublayer1_norm_params`) | HIGH | YES | Ownership boundary drift at layer boundary; fixable by adding preload flag and Top-side preload call | Preload in Top loop, pass explicit `sublayer1_norm_preloaded_by_top=true`, keep guarded fallback for compatibility |
| G3 | `src/Top.h` | Default infer loop path | Default pipeline still uses pointer-facing transformer loop path (deep bridge path exists but is not default) | MED | NO | Switching default loop path tonight risks broad behavior drift and larger verification budget | Keep default loop, but move ownership semantics forward inside current loop first |
| G4 | `src/Top.h` | CFG/PARAM ingest staging | Legacy base/shadow ingest style still coexists with newer top-managed window semantics | MED | NO | Re-architecting ingest and decode ownership is cross-cutting and not a minimal cut | Plan a dedicated ingest refactor pack with explicit region dispatch contracts |
| G5 | `src/blocks/PreprocEmbedSPE.h`, `src/blocks/LayerNormBlock.h`, `src/blocks/FinalHead.h` | Wrapper compatibility layer | Wrapper APIs still self-construct contracts for compatibility callers | LOW | NO | Removing wrappers tonight would likely break existing TB/local scripts and widen blast radius | Keep wrappers as compatibility fallback; enforce Top-owned path through static guard and Top dispatch anchors |

## Tonight Candidate Selection (Task B)
1. **Selected C1 (from G1): Top-owned contract dispatch for preproc/LN/final-head in `src/Top.h`**
   - Why selected:
     - strongest direct push toward "Top is sole shared SRAM owner"
     - no external Top formal contract change
     - local validation already available (`p11ah`, `p11aj`, purity/hygiene)
2. **Selected C2 (from G2): Top preload of sublayer1 LN params + guarded fallback in `TransformerLayer`**
   - Why selected:
     - removes block-side param ownership behavior on active loop boundary
     - additive/minimal (default compatibility preserved)
     - can be statically guarded by anchor checker

## Why Not Other Candidates Tonight
1. **G3 not selected**
   - requires broader default-path switch to deep bridge loop; higher coupling and higher regression cost.
2. **G4 not selected**
   - ingest/base-shadow refactor is multi-stage and not minimal for this overnight window.
3. **G5 not selected**
   - deleting compatibility wrappers would force broad rewiring across TB and scripts.

## Inventory Posture
- Evidence posture: local-only survey and local execution evidence.
- Closure posture: not Catapult closure; not SCVerify closure.

## Night-Batch G4 Remap Update (2026-03-29)
| gap_id | file | scope | current status | can fix tonight? | note |
| --- | --- | --- | --- | --- | --- |
| G4-A | `src/Top.h` | INFER ingest base dispatch ownership | DONE (this round) | YES | Added explicit `InferIngestContract` in Top for base/len/valid tracking. |
| G4-B | `src/Top.h` + `Preproc` dispatch callsite | Preproc infer input len/base still constant-flavored | DONE (this round) | YES | `run_preproc_block` now consumes ingest contract `in_base_word` + `len_words_valid`. |
| G4-C | `src/Top.h` + `FinalHead` dispatch callsite | FinalHead label source depended on shadow path | DONE (this round) | YES | FinalHead now consumes SRAM ingest view (`infer_label_words_view`). |
| G4-D | `src/Top.h` | INFER-side metadata harmonization (`phase/token/tile/span`) | DONE (this round) | YES | Added infer contract window metadata + OP_INFER span preflight validation. |
| G4-E | `src/Top.h` | Broader CFG/PARAM/INFER ingest unified rearchitecture | DEFERRED | NO | Cross-command unification still cross-cutting and exceeds night-batch change budget. |

### Primary / Secondary / Tertiary Ranking (Current Round)
1. Primary: G4-A/B/C combined Top-side ingest contractization (`src/Top.h`) with guard expansion.
2. Secondary: stricter shadow deprecation pass (remove mirror writes for normal infer path).
3. Tertiary: staged unified ingest rearchitecture (CFG/PARAM/INFER metadata harmonization).
