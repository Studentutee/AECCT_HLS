#!/usr/bin/env bash
# Catapult Shell Runbook template launcher
#
# 使用方式：
#   bash catapult_shell_run.sh /abs/path/to/project_override.env
#
# 注意：
# - 請先登入安裝 Catapult 的遠端 Linux 主機（例如經由 SSH + ssh-ed25519 公鑰）
# - 這支腳本只負責在已登入的 shell 中執行 Catapult、保存 log、搜尋 messages.txt、輸出摘要
# - 不負責 SSH onboarding / 私鑰管理

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash catapult_shell_run.sh /abs/path/to/project_override.env

Expected override variables:
  MODE, REPO_ROOT, PROJECT_NAME, PROJECT_TCL, RUN_TAG, CATAPULT_OUTDIR
Optional:
  PROJECT_ENTRY_DESC, TOP_TARGET, ENTRY_TU, CATAPULT_BIN, ALLOW_PATH_CATAPULT,
  MESSAGE_SEARCH_ROOTS, BLOCKER_KEYWORDS, WARNING_KEYWORDS, REMOTE_HOST,
  REMOTE_USER, REMOTE_REPO_ROOT, GOVERNANCE_POSTURE_DEFAULT
USAGE
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

OVERRIDE_FILE="$1"
if [[ ! -f "$OVERRIDE_FILE" ]]; then
  echo "ERROR: override file not found: $OVERRIDE_FILE" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$OVERRIDE_FILE"

: "${MODE:?MODE is required}"
: "${REPO_ROOT:?REPO_ROOT is required}"
: "${PROJECT_NAME:?PROJECT_NAME is required}"
: "${PROJECT_TCL:?PROJECT_TCL is required}"
: "${RUN_TAG:?RUN_TAG is required}"
: "${CATAPULT_OUTDIR:?CATAPULT_OUTDIR is required}"

PROJECT_ENTRY_DESC="${PROJECT_ENTRY_DESC:-}"
TOP_TARGET="${TOP_TARGET:-}"
ENTRY_TU="${ENTRY_TU:-}"
CATAPULT_BIN="${CATAPULT_BIN:-/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult}"
ALLOW_PATH_CATAPULT="${ALLOW_PATH_CATAPULT:-0}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-}"

# normalize arrays in case override omitted them
if ! declare -p MESSAGE_SEARCH_ROOTS >/dev/null 2>&1; then
  MESSAGE_SEARCH_ROOTS=()
fi
if ! declare -p BLOCKER_KEYWORDS >/dev/null 2>&1; then
  BLOCKER_KEYWORDS=("# Error" "Compilation aborted")
fi
if ! declare -p WARNING_KEYWORDS >/dev/null 2>&1; then
  WARNING_KEYWORDS=("Warning")
fi
if ! declare -p GOVERNANCE_POSTURE_DEFAULT >/dev/null 2>&1; then
  GOVERNANCE_POSTURE_DEFAULT=("not Catapult closure" "not SCVerify closure")
fi

mkdir -p "$CATAPULT_OUTDIR"

CONSOLE_LOG="$CATAPULT_OUTDIR/catapult_console.log"
INTERNAL_LOG="$CATAPULT_OUTDIR/catapult_internal.log"
VERSION_FILE="$CATAPULT_OUTDIR/catapult_version.txt"
ENV_FILE="$CATAPULT_OUTDIR/command.env"
EXACT_CMD_FILE="$CATAPULT_OUTDIR/exact_command.sh"
MESSAGES_PATH_FILE="$CATAPULT_OUTDIR/messages_path.txt"
GREP_SUMMARY="$CATAPULT_OUTDIR/grep_summary.txt"
GREP_BLOCKERS="$CATAPULT_OUTDIR/grep_blockers.txt"
GREP_WARNINGS="$CATAPULT_OUTDIR/grep_warnings.txt"
GREP_MESSAGES_TAIL="$CATAPULT_OUTDIR/grep_messages_tail.txt"
REPORT_STUB="$CATAPULT_OUTDIR/report_stub.md"
RUN_META="$CATAPULT_OUTDIR/run_meta.txt"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

record_failure_context() {
  local reason="$1"
  {
    echo "status=failed"
    echo "reason=$reason"
    echo "override_file=$OVERRIDE_FILE"
    echo "project_name=$PROJECT_NAME"
    echo "run_tag=$RUN_TAG"
    echo "repo_root=$REPO_ROOT"
    echo "project_tcl=$PROJECT_TCL"
    echo "catapult_outdir=$CATAPULT_OUTDIR"
  } > "$RUN_META"
}

require_file() {
  local path="$1"
  local desc="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing $desc: $path" >&2
    record_failure_context "missing_$desc"
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local desc="$2"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing $desc: $path" >&2
    record_failure_context "missing_$desc"
    exit 2
  fi
}

resolve_catapult_bin() {
  if [[ -x "$CATAPULT_BIN" ]]; then
    echo "$CATAPULT_BIN"
    return 0
  fi

  if [[ "$ALLOW_PATH_CATAPULT" == "1" ]]; then
    local resolved
    resolved="$(command -v catapult || true)"
    if [[ -n "$resolved" ]]; then
      echo "$resolved"
      return 0
    fi
  fi

  return 1
}

write_env_snapshot() {
  {
    echo "MODE=$MODE"
    echo "REPO_ROOT=$REPO_ROOT"
    echo "PROJECT_NAME=$PROJECT_NAME"
    echo "PROJECT_TCL=$PROJECT_TCL"
    echo "PROJECT_ENTRY_DESC=$PROJECT_ENTRY_DESC"
    echo "TOP_TARGET=$TOP_TARGET"
    echo "ENTRY_TU=$ENTRY_TU"
    echo "RUN_TAG=$RUN_TAG"
    echo "CATAPULT_OUTDIR=$CATAPULT_OUTDIR"
    echo "REMOTE_HOST=$REMOTE_HOST"
    echo "REMOTE_USER=$REMOTE_USER"
    echo "REMOTE_REPO_ROOT=$REMOTE_REPO_ROOT"
  } > "$ENV_FILE"
}

write_exact_command() {
  cat > "$EXACT_CMD_FILE" <<CMD
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
"$RESOLVED_CATAPULT_BIN" -shell -file "$PROJECT_TCL" -logfile "$INTERNAL_LOG"
CMD
  chmod +x "$EXACT_CMD_FILE"
}

find_latest_messages() {
  local paths=()
  local root

  paths+=("$CATAPULT_OUTDIR")
  for root in "${MESSAGE_SEARCH_ROOTS[@]}"; do
    [[ -n "$root" ]] && paths+=("$root")
  done
  paths+=("$REPO_ROOT")

  # unique existing dirs only
  local existing=()
  declare -A seen=()
  for root in "${paths[@]}"; do
    if [[ -d "$root" && -z "${seen[$root]:-}" ]]; then
      existing+=("$root")
      seen[$root]=1
    fi
  done

  if [[ ${#existing[@]} -eq 0 ]]; then
    return 1
  fi

  find "${existing[@]}" -type f -name 'messages.txt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR==1 {print substr($0, index($0,$2))}'
}

count_or_zero() {
  local pattern="$1"
  local file="$2"
  grep -c -- "$pattern" "$file" 2>/dev/null || true
}

write_keyword_hits() {
  local out_file="$1"
  local file="$2"
  shift 2
  : > "$out_file"
  local keyword
  for keyword in "$@"; do
    {
      echo "===== $keyword ====="
      grep -n -- "$keyword" "$file" || true
      echo
    } >> "$out_file"
  done
}

write_report_stub() {
  local messages_txt="$1"
  local error_count="$2"
  local abort_count="$3"
  local compile_done_count="$4"

  cat > "$REPORT_STUB" <<EOF_REPORT
1. Summary
- Mode: $MODE
- Project: $PROJECT_NAME
- Entry desc: ${PROJECT_ENTRY_DESC:-N/A}
- Top target: ${TOP_TARGET:-N/A}
- Entry TU: ${ENTRY_TU:-N/A}
- Repo root: $REPO_ROOT
- Project Tcl: $PROJECT_TCL
- Catapult outdir: $CATAPULT_OUTDIR
- Latest messages.txt: ${messages_txt:-NOT_FOUND}
- Current status: $( [[ "$error_count" == "0" && "$abort_count" == "0" && "$compile_done_count" != "0" ]] && echo "compile-first may be PASS; verify excerpts before claiming" || echo "investigation needed" )

2. Exact files changed
- $( [[ "$MODE" == "run-only" ]] && echo "無" || echo "<fill actual changed files>" )

3. Exact commands run
- cd "$REPO_ROOT"
- "$RESOLVED_CATAPULT_BIN" -shell -file "$PROJECT_TCL" -logfile "$INTERNAL_LOG"

4. Actual execution evidence / log excerpt
- See: $CONSOLE_LOG
- See: $INTERNAL_LOG
- See: $messages_txt
- # Error count: $error_count
- Compilation aborted count: $abort_count
- Completed transformation 'compile' count: $compile_done_count

5. Governance posture
$(for item in "${GOVERNANCE_POSTURE_DEFAULT[@]}"; do echo "- $item"; done)
- Do not overclaim beyond the evidence captured above.
EOF_REPORT
}

require_dir "$REPO_ROOT" "repo_root"
require_file "$PROJECT_TCL" "project_tcl"

RESOLVED_CATAPULT_BIN="$(resolve_catapult_bin || true)"
if [[ -z "$RESOLVED_CATAPULT_BIN" ]]; then
  echo "ERROR: catapult binary not found or not executable: $CATAPULT_BIN" >&2
  record_failure_context "catapult_binary_not_found"
  exit 2
fi

write_env_snapshot
write_exact_command

{
  echo "status=started"
  echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "mode=$MODE"
  echo "project_name=$PROJECT_NAME"
  echo "repo_root=$REPO_ROOT"
  echo "project_tcl=$PROJECT_TCL"
  echo "catapult_bin=$RESOLVED_CATAPULT_BIN"
  echo "catapult_outdir=$CATAPULT_OUTDIR"
} > "$RUN_META"

if "$RESOLVED_CATAPULT_BIN" -version > "$VERSION_FILE" 2>&1; then
  :
else
  log "WARN: failed to record catapult version"
fi

pushd "$REPO_ROOT" >/dev/null
set +e
"$RESOLVED_CATAPULT_BIN" -shell -file "$PROJECT_TCL" -logfile "$INTERNAL_LOG" 2>&1 | tee "$CONSOLE_LOG"
CATAPULT_RC=${PIPESTATUS[0]}
set -e
popd >/dev/null

LATEST_MESSAGES="$(find_latest_messages || true)"
if [[ -n "$LATEST_MESSAGES" ]]; then
  echo "$LATEST_MESSAGES" > "$MESSAGES_PATH_FILE"
else
  echo "NOT_FOUND" > "$MESSAGES_PATH_FILE"
fi

ERROR_COUNT=0
ABORT_COUNT=0
COMPILE_DONE_COUNT=0

{
  echo "catapult_rc=$CATAPULT_RC"
  echo "messages_txt=${LATEST_MESSAGES:-NOT_FOUND}"
} >> "$RUN_META"

if [[ -n "$LATEST_MESSAGES" && -f "$LATEST_MESSAGES" ]]; then
  ERROR_COUNT="$(count_or_zero '# Error' "$LATEST_MESSAGES")"
  ABORT_COUNT="$(count_or_zero 'Compilation aborted' "$LATEST_MESSAGES")"
  COMPILE_DONE_COUNT="$(count_or_zero "Completed transformation 'compile'" "$LATEST_MESSAGES")"

  {
    echo "# Error=$ERROR_COUNT"
    echo "Compilation aborted=$ABORT_COUNT"
    echo "Completed transformation 'compile'=$COMPILE_DONE_COUNT"
  } > "$GREP_SUMMARY"

  write_keyword_hits "$GREP_BLOCKERS" "$LATEST_MESSAGES" "${BLOCKER_KEYWORDS[@]}"
  write_keyword_hits "$GREP_WARNINGS" "$LATEST_MESSAGES" "${WARNING_KEYWORDS[@]}"
  grep -n -- '# Error\|Compilation aborted\|Completed transformation\|Warning' "$LATEST_MESSAGES" | tail -n 200 > "$GREP_MESSAGES_TAIL" || true
else
  {
    echo "# Error=UNKNOWN"
    echo "Compilation aborted=UNKNOWN"
    echo "Completed transformation 'compile'=UNKNOWN"
    echo "messages.txt not found"
  } > "$GREP_SUMMARY"
  : > "$GREP_BLOCKERS"
  : > "$GREP_WARNINGS"
  : > "$GREP_MESSAGES_TAIL"
fi

write_report_stub "$LATEST_MESSAGES" "$ERROR_COUNT" "$ABORT_COUNT" "$COMPILE_DONE_COUNT"

{
  echo "status=finished"
  echo "finished_timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "error_count=$ERROR_COUNT"
  echo "abort_count=$ABORT_COUNT"
  echo "compile_done_count=$COMPILE_DONE_COUNT"
} >> "$RUN_META"

log "Runbook artifacts written to: $CATAPULT_OUTDIR"
log "Exact command: $EXACT_CMD_FILE"
log "Console log: $CONSOLE_LOG"
log "Internal log: $INTERNAL_LOG"
log "Messages path: $(cat "$MESSAGES_PATH_FILE")"
log "Summary: $GREP_SUMMARY"

if [[ "$CATAPULT_RC" -ne 0 ]]; then
  log "WARN: catapult exited with non-zero status: $CATAPULT_RC"
fi

if [[ "$ERROR_COUNT" != "0" || "$ABORT_COUNT" != "0" ]]; then
  log "WARN: messages.txt indicates blocker(s). Review grep_summary.txt / grep_blockers.txt"
fi
