# TOP_MANAGED_SRAM_W4_QKSCORE_B5_EVIDENCE_INDEX_20260331

## Scope
- local-only evidence bundle for W4-B5 QkScore family bridge generalization
- not Catapult closure
- not SCVerify closure

## Primary W4-B5 logs
- `build/p11w4b5/qkscore_family_bridge/build.log`
- `build/p11w4b5/qkscore_family_bridge/run.log`
- `build/p11w4b5/qkscore_family_bridge/verdict.txt`
- `build/p11w4b5/qkscore_family_bridge/file_manifest.txt`

## Structural / guard logs
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression_summary.txt`

## Structural gates
- `PASS: check_design_purity`
- `PASS: check_interface_lock`
- `PASS: check_macro_hygiene`
- `PASS: check_repo_hygiene (pre/post)`

## Regression safety (baseline recheck)
- `build/p11w4b3/qkscore_bridge/run.log`
- `build/p11w4b2/qkscore_tile_bridge/run.log`

## Bundle note
- current run keeps raw logs in `build/p11w4b5/...` and referenced baseline folders.
