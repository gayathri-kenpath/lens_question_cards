#!/bin/bash
# Run one pass of the pipeline: fetch from Postgres -> Gemini -> output/output_llm.json
#
# Cron calls this every minute; it only actually runs once PIPELINE_INTERVAL_MINUTES
# (read from .env on every call, default 30) has passed since the last run started.
# Changing the value in .env takes effect on the next tick - no reinstall needed.
#
# Usage:
#   ./run_pipeline.sh           # run if the interval has passed
#   ./run_pipeline.sh --force   # run now regardless of the interval
#
# Logs go to logs/pipeline.log. Skips the run if the previous one is still going.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/pipeline.log"
LAST_RUN_FILE="$LOG_DIR/.last_run"
LOCK_DIR="$PROJECT_DIR/.pipeline.lock"
DEFAULT_INTERVAL=30

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

# Read one key from .env without sourcing it (values like passwords may contain shell characters)
env_value() { grep -E "^$1=" .env 2>/dev/null | tail -n1 | cut -d= -f2- | tr -d '"'"'"' \r' || true; }

INTERVAL="$(env_value PIPELINE_INTERVAL_MINUTES)"
INTERVAL="${INTERVAL:-$DEFAULT_INTERVAL}"
if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    log "Invalid PIPELINE_INTERVAL_MINUTES='$INTERVAL' in .env (must be a whole number of minutes) - using $DEFAULT_INTERVAL"
    INTERVAL=$DEFAULT_INTERVAL
fi

if [[ "${1:-}" != "--force" && -f "$LAST_RUN_FILE" ]]; then
    elapsed=$(( ($(date +%s) - $(cat "$LAST_RUN_FILE")) / 60 ))
    (( elapsed < INTERVAL )) && exit 0
fi

# mkdir is atomic, so it works as a lock (macOS has no flock)
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "Previous run still in progress - skipping"
    exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

# Record the start time even if the run fails, so failures don't retry every minute
date +%s > "$LAST_RUN_FILE"

log "Run started (interval: every $INTERVAL min)"
if "$PROJECT_DIR/.venv/bin/python" process_llm.py >> "$LOG_FILE" 2>&1; then
    log "Run finished OK"
else
    status=$?
    log "Run FAILED (exit $status)"
    exit $status
fi
