# REPORT_POST_ATTN_NEXT_SCOPE_CANDIDATE_AUDIT

Date: 2026-04-05  
Scope: post-attention non-attention bounded-candidate audit (read-only planning)

## 1) Audit Basis
- Read-only code-path scan on:
  - `src/Top.h`
  - `src/blocks/FFNLayer0.h`
  - `src/blocks/FinalHead.h`
  - `src/blocks/LayerNormBlock.h`
  - `src/blocks/PreprocEmbedSPE.h`
- Compile-backed baseline checks rerun:
  - `scripts/check_design_purity.ps1` -> `PASS: check_design_purity`
  - `scripts/check_repo_hygiene.ps1 -Phase pre` -> `PASS: check_repo_hygiene`
  - `scripts/local/run_p11aj_top_managed_sram_provenance.ps1` -> PASS
  - `scripts/local/run_p11anb_attnlayer0_boundary_seam_contract.ps1` -> PASS

## 2) Ranked Candidates (non-attention)
1. Candidate A: FFN handoff fallback tightening
- Module/files:
  - `src/Top.h`
  - `src/blocks/FFNLayer0.h`
- Why now:
  - Top already tracks FFN handoff gate/fallback counters (`p11av`/`p11aw`), so bounded telemetry and seam are present.
  - Similar bounded-cut shape to completed attention handoff work.
- Risk: Medium
- Bounded-cut suitability: High
- Parallelizable: Yes (can split checker/TB scaffolding and design-side seam updates)

2. Candidate B: FinalHead top-fed scalar/payload fallback seam hardening
- Module/files:
  - `src/Top.h`
  - `src/blocks/FinalHead.h`
- Why now:
  - Top already preloads top-fed final-scalar path and tracks payload fallback counters (`p11ax`).
  - Could reduce legacy fallback dependence in post-attention output path.
- Risk: Medium-High (output behavior surface is reviewer-visible)
- Bounded-cut suitability: Medium
- Parallelizable: Partially (analysis and checker work can be parallel; final seam cut should stay single-owner)

3. Candidate C: LayerNorm affine consume seam tightening
- Module/files:
  - `src/Top.h`
  - `src/blocks/LayerNormBlock.h`
- Why now:
  - LayerNorm already has top-fed gamma/beta consume seam and explicit affine consume trace structure.
  - Good candidate for deterministic handoff/no-fallback policy by phase.
- Risk: Medium
- Bounded-cut suitability: Medium
- Parallelizable: Yes for audit/checker scaffolding; core seam edits should be serialized

4. Candidate D: Preproc ingest top-fed-only policy tightening
- Module/files:
  - `src/Top.h`
  - `src/blocks/PreprocEmbedSPE.h`
- Why now:
  - Preproc core still keeps compatibility fallback (`topfed_in_words` missing -> local SRAM read).
  - Could be tightened with policy bit + bounded reject path.
- Risk: Low-Medium
- Bounded-cut suitability: Medium
- Parallelizable: Yes

## 3) Recommended Next Round Mainline Candidate
- Pick: Candidate A (FFN handoff fallback tightening).
- Reason:
  - Best balance of bounded scope, existing Top-side observability, and direct ownership/fallback reduction value.
  - Lowest chance to re-open already closed attention scope.

## 4) Guardrails for Next Round
- Keep Top as sole shared-SRAM owner.
- Do not change external 4-channel contract.
- No FFN/LayerNorm broad refactor; only bounded seam tightening.
- Keep claims local-only and compile-first unless real tool evidence is added.

## 5) 2026-04-06 Follow-Up Audit Result
- Candidate A (`FFN handoff fallback tightening`) was re-audited with compile-backed evidence and is currently blocked for immediate bounded cut.
- Blocking reason:
  - `FFNLayer0` strict policy/reject-stage gates already exist.
  - `TransformerLayer` still keeps descriptor-miss compatibility fallback preload behavior by design.
  - Immediate tightening would change accepted fallback semantics without a new explicit seam contract.
- Blocker report:
  - `docs/night_run/BLOCKER_POST_ATTN_FFN_HANDOFF_FALLBACK_TIGHTENING.md`
