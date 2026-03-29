# MORNING REVIEW QUICKSTART (2026-03-29)

## Quick Goal
- Validate helper-channel mixed-payload HOL risk reduction progress in 10 minutes.
- Confirm no overclaim: local-only evidence, not Catapult closure, not SCVerify closure.

## 10-Minute Review Order
1. **Baseline verdicts (2 min)**
   - Open `docs/handoff/HELPER_SPLIT_ACCEPTANCE_AUDIT_20260329.md`.
   - Confirm AF/AE/AD/AC all marked `ACCEPTABLE` with real run logs.
2. **Remaining hotspots (2 min)**
   - Open `docs/handoff/HELPER_CHANNEL_HOTSPOT_INVENTORY_20260329.md`.
   - Focus on residual MED/HIGH entries and candidate ranking.
3. **New split diff (3 min)**
   - Review:
     - `src/blocks/AttnPhaseATopManagedKv.h`
     - `tb/tb_kv_build_stream_stage_p11ac.cpp`
   - Confirm change scope is helper/TB-local only.
4. **Execution proof (2 min)**
   - Open `build/p11ac_impl_split_taskd/run.log`.
   - Check key lines:
     - `STREAM_ORDER PASS`
     - `MEMORY_ORDER PASS`
     - `EXACT_SCR_KV_COMPARE PASS`
     - `MAINLINE_PATH_TAKEN PASS`
     - `FALLBACK_NOT_TAKEN PASS`
5. **Regression guard (1 min)**
   - Open `build/helper_channel_guard/check_helper_channel_split_regression.log`.
   - Confirm:
     - `guard: AC legacy work-unit split anchors OK`
     - `PASS: check_helper_channel_split_regression`

## Accepted / Likely-Good
- A: acceptance audit report complete.
- B: helper split regression guard script added and passing.
- C: repo-wide helper/staging hotspot inventory complete.
- D: one low-risk helper-only split landed and validated (AC legacy work-unit path).

## Needs Human Diff Review
- `src/blocks/AttnPhaseATopManagedKv.h`
- `tb/tb_kv_build_stream_stage_p11ac.cpp`
- `scripts/check_helper_channel_split_regression.ps1`

## Deferred / Follow-Up
- Candidate C2 (`AttnPhaseATopManagedQ.h` legacy `X/WQ` work-unit split) is deferred for a follow-up pass with direct runner evidence.

## Artifact Index
- `docs/handoff/HELPER_SPLIT_ACCEPTANCE_AUDIT_20260329.md`
- `docs/handoff/HELPER_CHANNEL_REGRESSION_GUARD_20260329.md`
- `docs/handoff/HELPER_CHANNEL_HOTSPOT_INVENTORY_20260329.md`
- `docs/handoff/HELPER_LOWRISK_SPLIT_TASKD_20260329.md`
- `docs/handoff/OVERNIGHT_helper_split_and_guard_progress_20260329.md`
- `build/p11ac_impl_split_taskd/run.log`
- `build/helper_channel_guard/check_helper_channel_split_regression.log`

## Suggested GPT Prompt (copy-ready)
- `請依 docs/handoff/OVERNIGHT_helper_split_and_guard_progress_20260329.md 與 build/p11ac_impl_split_taskd/run.log，做 helper-channel mixed-payload 風險驗收，重點檢查：1) AC legacy split 是否真的消除單通道混送；2) regression guard 是否足以防回退；3) 還有哪些 residual hotspot 需要下一輪處理。請維持 local-only posture，不要 overclaim Catapult/SCVerify closure。`
