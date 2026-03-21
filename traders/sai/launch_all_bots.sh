#!/usr/bin/env bash
# ============================================================
# launch_all_bots.sh — Start all 4 ZDOM 0DTE bot instances
# ============================================================
# Usage:
#   ./scripts/launch_all_bots.sh            # Start all 4 bots
#   ./scripts/launch_all_bots.sh --dry-run   # Show what would run
#   ./scripts/launch_all_bots.sh --status    # Check running bots
#   ./scripts/launch_all_bots.sh --stop      # Gracefully stop all bots
#
# Each bot runs as a background process with its own log file.
# Ctrl+C sends SIGINT to all bots for graceful shutdown.
# ============================================================

set -euo pipefail

PROJECT_DIR="/Users/saiyeluru/Documents/Projects/iv-rank-scanner"
VENV_PYTHON="${PROJECT_DIR}/venv/bin/python"
LOG_DIR="${HOME}/logs/zdom_bots"
PID_DIR="${LOG_DIR}/pids"
BOT_SCRIPT="${PROJECT_DIR}/scripts/zero_dte_bot.py"

mkdir -p "$LOG_DIR" "$PID_DIR"

# --- Bot definitions ---
# Format: NAME|ENV_FILE|EXTRA_ARGS
# Env files are in traders/sai/ (this directory)
TRADER_DIR="${PROJECT_DIR}/traders/sai"
BOTS=(
    "schwab1|${TRADER_DIR}/.env.schwab1|--paper"
    "schwab2|${TRADER_DIR}/.env.schwab2|--paper"
    "tradier_live|${TRADER_DIR}/.env.tradier_live|--confirm-live"
    "tradier_paper|${TRADER_DIR}/.env.tradier_paper|--paper"
)

DATE_TAG=$(date +%Y-%m-%d)

# --- Functions ---

log() { echo "[$(date '+%H:%M:%S')] $*"; }

status() {
    log "=== Bot Status ==="
    local running=0
    for bot_def in "${BOTS[@]}"; do
        IFS='|' read -r name env_file extra_args <<< "$bot_def"
        local pidfile="${PID_DIR}/${name}.pid"
        if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            log "  ✓ ${name} — PID $(cat "$pidfile") — running"
            ((running++))
        else
            log "  ✗ ${name} — not running"
            rm -f "$pidfile"
        fi
    done
    log "=== ${running}/${#BOTS[@]} bots running ==="
}

stop_all() {
    log "Stopping all bots..."
    for bot_def in "${BOTS[@]}"; do
        IFS='|' read -r name env_file extra_args <<< "$bot_def"
        local pidfile="${PID_DIR}/${name}.pid"
        if [[ -f "$pidfile" ]]; then
            local pid
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                log "  Sending SIGINT to ${name} (PID ${pid})..."
                kill -INT "$pid" 2>/dev/null || true
            fi
            rm -f "$pidfile"
        fi
    done
    # Wait a moment then check for stragglers
    sleep 3
    for bot_def in "${BOTS[@]}"; do
        IFS='|' read -r name env_file extra_args <<< "$bot_def"
        local pidfile="${PID_DIR}/${name}.pid"
        # Check if process still exists by grepping
        local stale_pid
        stale_pid=$(pgrep -f "env-file.*${name##*_}" 2>/dev/null || true)
        if [[ -n "$stale_pid" ]]; then
            log "  Force-killing stale ${name} (PID ${stale_pid})..."
            kill -9 "$stale_pid" 2>/dev/null || true
        fi
    done
    log "All bots stopped."
}

launch() {
    local dry_run="${1:-false}"

    # Pre-flight: check env files exist
    for bot_def in "${BOTS[@]}"; do
        IFS='|' read -r name env_file extra_args <<< "$bot_def"
        local env_path="${env_file}"
        if [[ ! -f "$env_path" ]]; then
            log "ERROR: Missing env file: ${env_path}"
            log "  → Skipping ${name}"
            continue
        fi

        # Check for placeholder credentials
        if grep -q "REPLACE_WITH" "$env_path" 2>/dev/null; then
            log "WARNING: ${env_file} has placeholder credentials — skipping ${name}"
            continue
        fi
    done

    log "=== Launching ZDOM 0DTE Bots ==="
    log "Project: ${PROJECT_DIR}"
    log "Python:  ${VENV_PYTHON}"
    log ""

    local launched=0
    for bot_def in "${BOTS[@]}"; do
        IFS='|' read -r name env_file extra_args <<< "$bot_def"
        local env_path="${env_file}"
        local logfile="${LOG_DIR}/${name}_${DATE_TAG}.log"

        # Skip if env file missing or has placeholders
        if [[ ! -f "$env_path" ]]; then
            continue
        fi
        if grep -q "REPLACE_WITH" "$env_path" 2>/dev/null; then
            continue
        fi

        # Skip if already running
        local pidfile="${PID_DIR}/${name}.pid"
        if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            log "  ⊘ ${name} already running (PID $(cat "$pidfile")) — skipping"
            ((launched++))
            continue
        fi

        # Build command
        local cmd="${VENV_PYTHON} ${BOT_SCRIPT} --env-file ${env_path} ${extra_args}"

        if [[ "$dry_run" == "true" ]]; then
            log "  [DRY RUN] ${name}: ${cmd}"
            log "           Log: ${logfile}"
        else
            log "  Starting ${name}..."
            log "    Cmd: ${cmd}"
            log "    Log: ${logfile}"

            # Launch in background, capture PID
            cd "$PROJECT_DIR"
            nohup $cmd >> "$logfile" 2>&1 &
            local pid=$!
            echo "$pid" > "$pidfile"
            log "    PID: ${pid}"
            ((launched++))

            # Small delay between launches to avoid port/DB conflicts
            sleep 2
        fi
    done

    log ""
    log "=== ${launched} bots launched — logs in ${LOG_DIR} ==="
    log "Use '$0 --status' to check, '$0 --stop' to stop all"
}

# --- Main ---
case "${1:-}" in
    --status)
        status
        ;;
    --stop)
        stop_all
        ;;
    --dry-run)
        launch true
        ;;
    *)
        # Trap Ctrl+C to stop all bots
        trap 'echo ""; stop_all; exit 0' INT TERM
        launch false
        # Keep script alive so Ctrl+C can stop all bots
        log ""
        log "Bots running in background. Press Ctrl+C to stop all."
        log "Tailing logs... (Ctrl+C to stop)"
        log ""
        # Tail all active log files
        tail -f "${LOG_DIR}"/*_${DATE_TAG}.log 2>/dev/null || wait
        ;;
esac
