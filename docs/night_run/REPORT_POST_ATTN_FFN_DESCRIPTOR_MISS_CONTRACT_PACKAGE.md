# REPORT_POST_ATTN_FFN_DESCRIPTOR_MISS_CONTRACT_PACKAGE

Date: 2026-04-06  
Scope: Top -> TransformerLayer -> FFNLayer0 descriptor-miss contract package (local-only, compile-first, evidence-first)

## 1) One-Line Verdict
- This round delivers a **checker package** (not tightening): descriptor-present consumes top-fed payload, descriptor-miss keeps compatibility preload fallback, and counters/markers stay consistent.

## 2) Hardware Contract Snapshot
- Input payload class:
  - `W1`: input (`topfed_w1_x_words`), weight (`topfed_w1_weight_words`), bias (`topfed_w1_bias_words`)
  - `W2`: input (`topfed_w2_input_words`), weight (`topfed_w2_weight_words`), bias (`topfed_w2_bias_words`)
- Producer/consumer boundary:
  - Producer: `Top` composes handoff descriptor and valid spans.
  - Consumer stage-0: `TransformerLayer` checks descriptor-ready and selects top-fed vs compatibility preload path.
  - Consumer stage-1: `FFNLayer0` executes stage W1/ReLU/W2 under strict gate flags.
- Intermediate/writeback:
  - `W1` output -> `sc.ffn.w1_out_base_word`
  - `ReLU` output -> `sc.ffn.relu_out_base_word`
  - `W2` output -> `sc.ffn.w2_out_base_word`
  - residual add output -> `sc.ffn.add2_base_word`, then LayerNorm consumes.

## 3) Descriptor-Miss Semantics (As-Is)
- Contract status: **hybrid state**.
- `TransformerLayer` descriptor-miss behavior: compatibility preload fallback is preserved.
  - W1 input miss: preload from `sc.attn_out_base_word` then feed FFN W1.
  - W1/W2 weight+bias miss: preload from param SRAM section then feed FFN.
  - W2 input miss: preload from ReLU scratch then feed FFN W2.
- `FFNLayer0` behavior:
  - receives strict flags (`FFN_POLICY_REQUIRE_W1_TOPFED` / `FFN_POLICY_REQUIRE_W2_TOPFED`)
  - rejects only when incoming descriptor-ready gate is not satisfied at FFN entry.
  - non-strict mode keeps fallback touch observable (`fallback_legacy_touch_counter`).

## 4) Ownership Boundary (As-Is)
- Shared SRAM owner: `Top` only.
- `TransformerLayer` and `FFNLayer0` do not own shared-SRAM arbitration semantics.
- `TransformerLayer` currently acts as seam adapter:
  - accepts Top handoff descriptor,
  - performs compatibility preload on miss,
  - passes selected payload/valid span to `FFNLayer0`.

## 5) Why No Tightening This Round
- Immediate tightening at this seam would change accepted descriptor-miss fallback semantics in `TransformerLayer`.
- Current checker acceptance explicitly relies on miss -> compatibility preload fallback behavior.
- Therefore this round intentionally freezes behavior and formalizes checkability first.

## 6) Exact Code Locations
- `src/blocks/TransformerLayer.h`
  - descriptor container: `TransformerLayerFfnTopfedHandoffDesc` (`301-316`)
  - W1 miss compatibility preload: `600-609`, `651-659`, `696-704`, `1110-1118`, `1161-1169`, `1206-1214`
  - W2 miss compatibility preload: `777-781`, `801-809`, `825-833`, `1287-1291`, `1311-1319`, `1335-1343`
  - strict FFN dispatch flags preserved: `727-749`, `887-904`, `1237-1259`, `1397-1413`
- `src/blocks/FFNLayer0.h`
  - descriptor-ready checks: `218-236`
  - strict reject gate W1/W2: `245-256`, `351-360`
  - fallback touch observability path: `37-42`, `267`, `297`, `304`, `372`, `402`, `409`
- `src/Top.h`
  - handoff gate/non-empty/fallback counters: `373-378`
  - loop-level seam accounting: `2845-2862`, `3493-3510`
  - pipeline-level mirror counters: `3655-3660`, `3677-3682`

## 7) Compile-Backed Checker Package
- New package runner:
  - `scripts/local/run_post_attn_ffn_descriptor_miss_contract_package.ps1`
- Runner composes and validates:
  - `scripts/local/run_p11au_transformerlayer_ffn_higher_level_ownership_seam.ps1`
  - `scripts/local/run_p11av_top_ffn_handoff_assembly_smoke.ps1`
  - `scripts/local/run_p11g6_ffn_fallback_observability.ps1`
- Package PASS banners:
  - `PRESENT -> TOPFED_CONSUME PASS`
  - `MISS -> FALLBACK_PRELOAD PASS`
  - `NO_FAKE_TOPFED_ACCEPT PASS`
  - `COUNTER / MARKER CONSISTENCY PASS`
  - `REJECT_STAGE_OBSERVABILITY PASS`

## 8) Governance Posture
- local-only
- compile-first
- evidence-first
- not Catapult closure
- not SCVerify closure
