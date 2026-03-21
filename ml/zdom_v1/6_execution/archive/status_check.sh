#!/bin/bash
echo "===== STATUS CHECK $(date) ====="
echo ""

# Theta Terminal
if curl -s "http://127.0.0.1:25503/v3/option/history/eod?symbol=SPXW&expiration=2024-01-02&start_date=2024-01-02&end_date=2024-01-02&format=json" > /dev/null 2>&1; then
  echo "✅ Theta Terminal: RUNNING"
else
  echo "❌ Theta Terminal: DOWN"
fi

# Fetch processes
FETCH_COUNT=$(ps aux | grep fetch_intraday_greeks | grep -v grep | wc -l | tr -d ' ')
if [ "$FETCH_COUNT" -gt 0 ]; then
  echo "✅ Greeks fetch: RUNNING ($FETCH_COUNT workers)"
else
  echo "❌ Greeks fetch: STOPPED"
fi

# WMS
if curl -s "http://localhost:3000" > /dev/null 2>&1; then
  echo "✅ WMS: RUNNING at http://localhost:3000"
else
  echo "❌ WMS: DOWN"
fi

# Data progress
echo ""
echo "📊 Data:"
python3 -c "
import pandas as pd, os
f = os.path.expanduser('~/ironworks/Zero_DTE_Options_Strategy/data/spxw_0dte_intraday_greeks.parquet')
if os.path.exists(f):
    df = pd.read_parquet(f)
    n_days = df['date'].nunique()
    pct = n_days / 754 * 100
    size = os.path.getsize(f) / 1e6
    print(f'  Intraday greeks: {n_days}/754 days ({pct:.1f}%) | {size:.0f}MB')
    print(f'  Last date: {df[\"date\"].max().date()}')
    print(f'  Rows: {len(df):,}')
" 2>/dev/null

echo ""
echo "📁 All data files:"
ls -lh "$HOME/ironworks/Zero_DTE_Options_Strategy/data/"*.parquet 2>/dev/null | awk '{print "  "$NF, $5}' | sed 's|.*/||'
