#!/bin/bash
# Install a cron job that ticks every minute and calls run_pipeline.sh.
# How often the pipeline actually runs is set by PIPELINE_INTERVAL_MINUTES in .env
# (default 30) and is re-read on every tick, so you never need to reinstall to change it.
#
# Usage:
#   ./schedule_pipeline.sh install     # add the cron job
#   ./schedule_pipeline.sh uninstall   # remove it
#   ./schedule_pipeline.sh status      # show whether it is installed and the current interval
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="$PROJECT_DIR/run_pipeline.sh"
MARKER="# lensv3-qn-pipeline"
CRON_LINE="* * * * * \"$RUN_SCRIPT\" $MARKER"

current_crontab() { crontab -l 2>/dev/null || true; }
interval() {
    local v
    v="$(grep -E '^PIPELINE_INTERVAL_MINUTES=' "$PROJECT_DIR/.env" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
    echo "${v:-30 (default)}"
}

case "${1:-}" in
    install)
        chmod +x "$RUN_SCRIPT"
        { current_crontab | grep -vF "$MARKER"; echo "$CRON_LINE"; } | crontab -
        echo "Installed. Pipeline runs every $(interval) min (PIPELINE_INTERVAL_MINUTES in .env)."
        echo "Logs: $PROJECT_DIR/logs/pipeline.log"
        ;;
    uninstall)
        current_crontab | grep -vF "$MARKER" | crontab -
        echo "Removed the pipeline cron job."
        ;;
    status)
        if current_crontab | grep -qF "$MARKER"; then
            echo "Installed. Pipeline runs every $(interval) min."
        else
            echo "Not installed. (Interval in .env: $(interval) min)"
        fi
        ;;
    *)
        echo "Usage: $0 {install|uninstall|status}" >&2
        exit 1
        ;;
esac
