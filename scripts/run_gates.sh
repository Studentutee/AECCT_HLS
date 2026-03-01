#!/usr/bin/env bash
set -euo pipefail

python3 scripts/check_design_purity.py --repo-root .
python3 scripts/check_interface_lock.py --repo-root .
python3 scripts/check_repo_hygiene.py --repo-root .

echo "PASS: all gates"
