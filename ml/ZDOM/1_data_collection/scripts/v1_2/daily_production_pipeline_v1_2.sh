#!/bin/bash
# ZDOM V1.2 Daily Production Pipeline
# Runs automatically on weekdays via LaunchAgent
#
# Schedule:
#   9:00 AM ET — Refresh ALL daily tables (FRED, yfinance, calendar, regime, etc.)
#   9:30-4:00   — Intraday collector runs separately (LaunchAgent)
#   4:30 PM ET — Refresh ALL daily tables again + pull ThetaData EOD + validate
#
# Usage:
#   ./daily_production_pipeline_v1_2.sh --mode morning    # 9:00 AM run
#   ./daily_production_pipeline_v1_2.sh --mode evening    # 4:30 PM run
#   ./daily_production_pipeline_v1_2.sh --mode full       # Both (manual catch-up)

# set -e  # disabled — individual failures logged, pipeline continues

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPTS_DIR="$PROJECT_DIR/1_data_collection/scripts/v1_2"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/daily_pipeline_${DATE}.log"
PYTHON=python3

# Source FRED API key (from env, SSM, or fallback)
if [ -z "$FRED_API_KEY" ]; then
    export FRED_API_KEY=$(aws ssm get-parameter --name /ironworks/options-model/FRED_API_KEY --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "dfb3f177682f9fd569755b3c0b86201d")
fi

cd "$PROJECT_DIR"

MODE="${2:-full}"
if [ "$1" = "--mode" ]; then
    MODE="$2"
fi

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

check_theta() {
    if curl -s -o /dev/null -w "" "http://127.0.0.1:25503/" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

start_theta() {
    log "Starting ThetaData terminal..."
    cd "$PROJECT_DIR/theta_terminal"
    java -jar ThetaTerminalv3.jar &>/tmp/theta_terminal.log &
    sleep 15
    if check_theta; then
        log "  ThetaData terminal started successfully"
    else
        log "  ERROR: ThetaData terminal failed to start"
        exit 1
    fi
}

# ============================================================================
# REFRESH ALL DAILY TABLES
# Runs at BOTH 9:00 AM and 4:30 PM
# ============================================================================
refresh_daily_tables() {
    log "--- Refreshing all daily tables ---"

    log "  [1/11] FRED daily features..."
    $PYTHON "$SCRIPTS_DIR/build_fred_daily_features_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "  [2/11] yfinance cross-asset..."
    $PYTHON "$SCRIPTS_DIR/build_cross_asset_daily_features_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "  [3/11] yfinance breadth..."
    $PYTHON "$SCRIPTS_DIR/build_breadth_daily_features_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "  [4/11] yfinance vol context..."
    $PYTHON "$SCRIPTS_DIR/build_vol_context_daily_features_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "  [5/11] Econ calendar..."
    $PYTHON "$SCRIPTS_DIR/build_econ_calendar_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "  [6/11] Mag7 earnings..."
    $PYTHON "$SCRIPTS_DIR/build_mag7_earnings_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "  [7/11] Regime features..."
    $PYTHON "$SCRIPTS_DIR/build_regime_features_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "  [8/11] Macro + presidential cycles..."
    $PYTHON "$SCRIPTS_DIR/build_macro_and_cycles_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "  [9/11] SPX daily (yfinance)..."
    $PYTHON "$SCRIPTS_DIR/fetch_spx_daily_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "  [10/11] VIX daily (yfinance)..."
    $PYTHON "$SCRIPTS_DIR/fetch_vix_daily_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "  [11/11] VIX1D daily..."
    $PYTHON "$SCRIPTS_DIR/fetch_vix1d_daily_v1_2.py" >> "$LOG" 2>&1 && log "    OK" || log "    FAILED"

    log "--- Daily tables refresh complete ---"
}

# ============================================================================
# VALIDATE RAW DATA
# Checks every raw parquet has data for today
# ============================================================================
validate_raw_data() {
    log "--- Validating raw data coverage ---"
    $PYTHON -c "
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

RAW = Path('$PROJECT_DIR/1_data_collection/raw/v1_2')
DATA = Path('$PROJECT_DIR/1_data_collection/data/v1_2')
today = date.today()
# For intraday files, expect today; for daily, expect at least T-1
t1 = today - timedelta(days=1)
while t1.weekday() >= 5:
    t1 -= timedelta(days=1)

all_ok = True

# Check raw files
for f in sorted(RAW.glob('*.parquet')):
    if f.name.startswith('_'): continue
    try:
        df = pd.read_parquet(f)
        for dcol in ['datetime', 'date', 'timestamp']:
            if dcol in df.columns:
                df[dcol] = pd.to_datetime(df[dcol])
                last = df[dcol].max().date()
                ok = last >= t1
                if not ok: all_ok = False
                status = 'OK' if ok else f'STALE (last={last}, need>={t1})'
                print(f'  raw  {f.name:<45} last={last}  {status}')
                break
    except Exception as e:
        print(f'  raw  {f.name:<45} ERROR: {e}')
        all_ok = False

# Check data files
for f in sorted(DATA.glob('*.parquet')):
    if f.name.startswith('_'): continue
    try:
        df = pd.read_parquet(f)
        for dcol in ['date', 'decision_datetime', 'datetime']:
            if dcol in df.columns:
                df[dcol] = pd.to_datetime(df[dcol])
                last = df[dcol].max().date()
                ok = last >= t1
                if not ok: all_ok = False
                status = 'OK' if ok else f'STALE (last={last}, need>={t1})'
                print(f'  data {f.name:<45} last={last}  {status}')
                break
    except Exception as e:
        print(f'  data {f.name:<45} ERROR: {e}')
        all_ok = False

# Check greeks daily chunks for today
chunks = list((RAW / 'greeks_daily_chunks').glob(f'greeks_{today.strftime(\"%Y%m%d\")}.parquet'))
if chunks:
    df = pd.read_parquet(chunks[0])
    print(f'  raw  greeks_{today.strftime(\"%Y%m%d\")}.parquet            {len(df):,} rows  OK')
else:
    print(f'  raw  greeks_{today.strftime(\"%Y%m%d\")}.parquet            MISSING')
    all_ok = False

if all_ok:
    print('\nALL FILES CURRENT.')
else:
    print('\nWARNING: Some files are stale or missing!')
" >> "$LOG" 2>&1
    log "--- Validation complete ---"
}

# ============================================================================
# MORNING RUN (9:00 AM ET)
# ============================================================================
morning_run() {
    log "=== MORNING RUN — $DATE ==="

    # 1. Ensure ThetaData is running
    if ! check_theta; then
        start_theta
    else
        log "  ThetaData already running"
    fi

    # 2. Refresh ALL daily tables
    refresh_daily_tables

    # 3. Validate
    validate_raw_data

    log "=== MORNING RUN COMPLETE ==="
}

# ============================================================================
# EVENING RUN (4:30 PM ET)
# ============================================================================
evening_run() {
    log "=== EVENING RUN — $DATE ==="

    # 1. Ensure ThetaData is running
    if ! check_theta; then
        start_theta
    fi

    # 2. Pull full day's intraday bars from ThetaData
    log "[1/6] Pulling intraday bars (SPX/VIX/VIX1D)..."
    $PYTHON -c "
import pandas as pd, requests
from io import StringIO
from pathlib import Path
from datetime import datetime

RAW = Path('$PROJECT_DIR/1_data_collection/raw/v1_2')
THETA = 'http://127.0.0.1:25503'
TODAY = datetime.now().strftime('%Y%m%d')

for sym, file, prefix in [('SPX','spx_1min.parquet',''), ('VIX','vix_1min.parquet','vix_'), ('VIX1D','vix1d_1min.parquet','vix1d_')]:
    existing = pd.read_parquet(RAW / file)
    existing['datetime'] = pd.to_datetime(existing['datetime'])
    resp = requests.get(f'{THETA}/v3/index/history/ohlc', params={'symbol': sym, 'start_date': TODAY, 'end_date': TODAY, 'interval': '1m'}, timeout=30)
    df = pd.read_csv(StringIO(resp.text))
    if len(df) == 0 or df['close'].sum() == 0:
        print(f'  {sym}: no data')
        continue
    df = df.rename(columns={'timestamp': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[(df['datetime'].dt.time >= pd.Timestamp('09:30').time()) & (df['datetime'].dt.time <= pd.Timestamp('16:00').time())]
    if prefix:
        df = df.rename(columns={'open': f'{prefix}open', 'high': f'{prefix}high', 'low': f'{prefix}low', 'close': f'{prefix}close'})
    keep = ['datetime'] + [c for c in df.columns if c.startswith(prefix) and c != 'datetime'] if prefix else ['datetime','open','high','low','close']
    new = df[keep]
    combined = pd.concat([existing, new], ignore_index=True).drop_duplicates('datetime').sort_values('datetime')
    combined.to_parquet(RAW / file, index=False)
    print(f'  {sym}: {len(new)} new bars, total={len(combined):,}')
" >> "$LOG" 2>&1 && log "  Intraday bars OK" || log "  Intraday bars FAILED"

    # 3. Pull intraday greeks (EOD backfill for full day)
    log "[2/6] Pulling intraday greeks..."
    $PYTHON "$SCRIPTS_DIR/fetch_intraday_greeks_v1_2.py" >> "$LOG" 2>&1 && log "  Greeks OK" || log "  Greeks FAILED"

    # 4. Pull term structure
    log "[3/6] Pulling term structure..."
    $PYTHON "$SCRIPTS_DIR/fetch_term_structure_v1_2.py" >> "$LOG" 2>&1 && log "  Term structure OK" || log "  Term structure FAILED"

    # 5. Pull OI
    log "[4/6] Pulling OI..."
    $PYTHON -c "
import pandas as pd, requests
from io import StringIO
from pathlib import Path
from datetime import datetime

RAW = Path('$PROJECT_DIR/1_data_collection/raw/v1_2')
THETA = 'http://127.0.0.1:25503'
TODAY = datetime.now().strftime('%Y%m%d')

oi = pd.read_parquet(RAW / 'spxw_0dte_oi.parquet')
oi['date'] = pd.to_datetime(oi['date'])
resp = requests.get(f'{THETA}/v3/option/history/open_interest', params={'symbol': 'SPXW', 'expiration': TODAY, 'start_date': TODAY, 'end_date': TODAY}, timeout=30)
df = pd.read_csv(StringIO(resp.text))
if len(df) > 0 and 'open_interest' in df.columns:
    df['date'] = f'{TODAY[:4]}-{TODAY[4:6]}-{TODAY[6:8]}'
    oi = pd.concat([oi, df[['date','strike','right','open_interest']]], ignore_index=True)
    oi['date'] = pd.to_datetime(oi['date'])
    oi = oi.drop_duplicates(['date','strike','right']).sort_values(['date','strike'])
    oi.to_parquet(RAW / 'spxw_0dte_oi.parquet', index=False)
    print(f'  OI: {len(df)} rows for {TODAY}')
else:
    print(f'  OI: no data for {TODAY}')
" >> "$LOG" 2>&1
    $PYTHON "$SCRIPTS_DIR/build_spxw_0dte_oi_daily_v1_2.py" >> "$LOG" 2>&1 && log "  OI daily OK" || log "  OI daily FAILED"

    # 6. Rebuild term structure daily + daily aggregates
    log "[5/6] Rebuilding term structure daily..."
    $PYTHON "$SCRIPTS_DIR/build_spxw_term_structure_daily_v1_2.py" >> "$LOG" 2>&1 && log "  Term structure daily OK" || log "  Term structure daily FAILED"

    log "[6/6] Rebuilding daily aggregates (SPX/VIX/VIX1D from 1-min)..."
    $PYTHON -c "
import pandas as pd
from pathlib import Path

RAW = Path('$PROJECT_DIR/1_data_collection/raw/v1_2')
DATA = Path('$PROJECT_DIR/1_data_collection/data/v1_2')

for sym, file, raw_file, prefix in [
    ('SPX', 'spx_daily_v1_2.parquet', 'spx_1min.parquet', 'spx_'),
    ('VIX', 'vix_daily_v1_2.parquet', 'vix_1min.parquet', 'vix_'),
    ('VIX1D', 'vix1d_daily_v1_2.parquet', 'vix1d_1min.parquet', 'vix1d_'),
]:
    bars = pd.read_parquet(RAW / raw_file)
    bars['datetime'] = pd.to_datetime(bars['datetime'])
    close_col = f'{prefix}close' if prefix != 'spx_' else 'close'
    open_col = f'{prefix}open' if prefix != 'spx_' else 'open'
    high_col = f'{prefix}high' if prefix != 'spx_' else 'high'
    low_col = f'{prefix}low' if prefix != 'spx_' else 'low'
    bars = bars[bars[close_col] > 0]
    bars['date'] = bars['datetime'].dt.date
    daily = bars.groupby('date').agg(**{
        f'{prefix}open': (open_col, 'first'),
        f'{prefix}high': (high_col, 'max'),
        f'{prefix}low': (low_col, 'min'),
        f'{prefix}close': (close_col, 'last'),
    }).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily.to_parquet(DATA / file, index=False)
    print(f'  {sym} daily: {len(daily)} days, last={daily[\"date\"].max().date()}')
" >> "$LOG" 2>&1 && log "  Daily aggregates OK" || log "  Daily aggregates FAILED"

    # 7. Refresh ALL daily tables (same as morning — picks up today's close data)
    refresh_daily_tables

    # 8. Validate everything
    validate_raw_data

    log "=== EVENING RUN COMPLETE ==="
}

# ============================================================================
# Main
# ============================================================================
log "Daily Production Pipeline v1.2 — mode=$MODE"

case "$MODE" in
    morning) morning_run ;;
    evening) evening_run ;;
    full)    morning_run; evening_run ;;
    *)       echo "Usage: $0 --mode [morning|evening|full]"; exit 1 ;;
esac
