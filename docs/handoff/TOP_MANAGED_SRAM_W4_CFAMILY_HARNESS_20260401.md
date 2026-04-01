# TOP_MANAGED_SRAM_W4_CFAMILY_HARNESS_20260401

## Goal
- Consolidate SoftmaxOut C-family (`C0/C1/C2`) task-local test execution pattern into reusable family harness + runner strategy.

## Inventory (Pre-Consolidation)
- TB:
  - `tb/tb_w4c0_softmaxout_contract_probe.cpp`
  - `tb/tb_w4c1_softmaxout_head_token_contract_probe.cpp`
  - `tb/tb_w4c2_softmaxout_acc_single_later_token_bridge.cpp`
- Runner:
  - `scripts/local/run_p11w4c0_softmaxout_contract_probe.ps1`
  - `scripts/local/run_p11w4c1_softmaxout_head_token_contract_probe.ps1`
  - `scripts/local/run_p11w4c2_softmaxout_acc_single_later_token_bridge.ps1`
- Shared pattern:
  - common bootstrap flow (`AC/AD -> AE -> AF`)
  - common compile/run/verdict/manifest skeleton
  - common PASS-line gating + negative reject + anti-fallback checks

## Consolidation Decision
- Chosen scheme: **Option C**
  - shared harness + shared runner
  - existing round scripts kept as thin wrappers
- Why this is minimal:
  - preserves per-round source and PASS line contract
  - avoids broad TB rewrite
  - allows C3/C4 extension with same runner/harness layers

## Landed Assets
- Shared harness:
  - `tb/tb_w4cfamily_softmaxout_harness_common.h`
- Shared runner:
  - `scripts/local/run_p11w4cfamily_softmaxout_common.ps1`
- Thin wrappers retained:
  - C0/C1/C2 runners now delegate to shared runner with round-specific PASS lines.

## Safety Notes
- No external Top 4-channel contract change.
- No second ownership/arbitration contract.
- No hidden semantic change in design mainline for consolidation-only scope.
