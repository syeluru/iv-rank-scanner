#!/bin/bash
# Daily data fetch — run after market close (4:30pm ET recommended)
# Pulls all data from ThetaData + FRED for the latest trading day
# All scripts support incremental updates — they only fetch what's missing

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON=python3
LOG="$PROJECT_DIR/logs/daily_fetch_$(date +%Y-%m-%d).log"
mkdir -p "$PROJECT_DIR/logs"

echo "=== Daily Fetch $(date) ===" | tee -a "$LOG"

# Check Theta Terminal is running
if ! curl -s -o /dev/null -w "" "http://127.0.0.1:25503/v3/index/history/eod?symbol=SPX&start_date=2026-01-01&end_date=2026-01-02&format=json" 2>/dev/null; then
    echo "ERROR: Theta Terminal not running on :25503" | tee -a "$LOG"
    exit 1
fi

echo "[1/8] SPX 1-min intraday..." | tee -a "$LOG"
$PYTHON scripts/fetch_spx_intraday.py 2>&1 | tee -a "$LOG"

echo "[2/8] VIX 1-min intraday..." | tee -a "$LOG"
$PYTHON scripts/fetch_vix_data.py 2>&1 | tee -a "$LOG"

echo "[3/8] VIX1D 1-min intraday..." | tee -a "$LOG"
$PYTHON scripts/fetch_vix1d.py 2>&1 | tee -a "$LOG"

echo "[4/8] Option chain EOD + OI..." | tee -a "$LOG"
$PYTHON scripts/fetch_option_chain.py 2>&1 | tee -a "$LOG"

echo "[5/8] Intraday greeks..." | tee -a "$LOG"
$PYTHON scripts/fetch_intraday_greeks.py 2>&1 | tee -a "$LOG"

echo "[6/8] Term structure..." | tee -a "$LOG"
$PYTHON scripts/fetch_term_structure.py 2>&1 | tee -a "$LOG"

echo "[7/8] Events + econ calendar..." | tee -a "$LOG"
$PYTHON scripts/fetch_events_data.py 2>&1 | tee -a "$LOG"
$PYTHON scripts/fetch_econ_calendar.py 2>&1 | tee -a "$LOG"

echo "[8/8] Macro + cycle features..." | tee -a "$LOG"
$PYTHON scripts/fetch_macro_features.py 2>&1 | tee -a "$LOG"
$PYTHON scripts/fetch_cycle_features.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Daily Fetch Complete $(date) ===" | tee -a "$LOG"
