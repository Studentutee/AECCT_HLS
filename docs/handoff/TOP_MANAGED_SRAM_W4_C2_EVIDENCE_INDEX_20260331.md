# TOP_MANAGED_SRAM_W4_C2_EVIDENCE_INDEX_20260331

## Evidence Root
- `build/evidence/w4c2_acc_single_later_token_bridge_20260401/`

## Structural Gates
1. `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
2. `powershell -ExecutionPolicy Bypass -File scripts/check_interface_lock.ps1`
3. `powershell -ExecutionPolicy Bypass -File scripts/check_macro_hygiene.ps1`
4. `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
5. `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
6. `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
7. `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

## Targeted C2 Runner
- Command:
  - `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4c2_softmaxout_acc_single_later_token_bridge.ps1`
- Runner log:
  - `build/evidence/w4c2_acc_single_later_token_bridge_20260401/run_p11w4c2_softmaxout_acc_single_later_token_bridge.log`
- TB raw run log:
  - `build/p11w4c2/softmaxout_acc_single_later_token_bridge/run.log`
- Build log:
  - `build/p11w4c2/softmaxout_acc_single_later_token_bridge/build.log`
- Verdict:
  - `build/p11w4c2/softmaxout_acc_single_later_token_bridge/verdict.txt`
- Manifest:
  - `build/p11w4c2/softmaxout_acc_single_later_token_bridge/file_manifest.txt`

Expected C2 PASS lines:
- `W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_BRIDGE_VISIBLE PASS`
- `W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_OWNERSHIP_CHECK PASS`
- `W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_EXPECTED_COMPARE PASS`
- `W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_LEGACY_COMPARE PASS`
- `W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_NO_SPURIOUS_TOUCH PASS`
- `W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_MISMATCH_REJECT PASS`
- `W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_ANTI_FALLBACK PASS`
- `W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_LATER_TOKEN_CONSUME_COUNT_EXACT PASS`
- `W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_TOKEN_SELECTOR_VISIBLE PASS`
- `PASS: tb_w4c2_softmaxout_acc_single_later_token_bridge`
- `PASS: run_p11w4c2_softmaxout_acc_single_later_token_bridge`

## Baseline Recheck
1. `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4c1_softmaxout_head_token_contract_probe.ps1`
2. `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4c0_softmaxout_contract_probe.ps1`
3. `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4b9_qkscore_longspan_bridge.ps1`
4. `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4b8_qkscore_family_fullhead_bridge.ps1`
5. `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4b1_phaseb_tile_bridge.ps1`

## Index Files
- `build/evidence/w4c2_acc_single_later_token_bridge_20260401/command_manifest.txt`
- `build/evidence/w4c2_acc_single_later_token_bridge_20260401/evidence_highlights.txt`
- `build/evidence/w4c2_acc_single_later_token_bridge_20260401/authority_extract_20260401.txt`
- `build/evidence/w4c2_acc_single_later_token_bridge_20260401/verdict_summary.txt`
- `build/evidence/w4c2_acc_single_later_token_bridge_20260401/git_status_pre.txt`
- `build/evidence/w4c2_acc_single_later_token_bridge_20260401/git_status_post.txt`
