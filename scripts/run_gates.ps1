$ErrorActionPreference = "Stop"

python scripts/check_design_purity.py --repo-root .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/check_interface_lock.py --repo-root .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/check_repo_hygiene.py --repo-root .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "PASS: all gates"
