# Feature Audit Instructions — ZDOM V1

## What This Is
`zdom_feature_audit_final_v2.csv` is a line-by-line audit of all 291 features (284 original + 7 new) in ZDOM V1. Each feature is being reviewed to ensure training and production compute identically, with no data leakage (using same-day data not available at scoring time).

## Status
- **90 features confirmed** (Final Name and all columns filled)
- **201 features remaining** (need review)
- **Matto is reviewing from the top down**
- **Sai should review from the bottom up**

## How to Review (Line by Line)
For each unconfirmed feature (rows where "Final Name" is empty):

1. Read the **Feature** name and **Check** column (Matto's notes/questions)
2. Read the **TARS Response** column (initial response to Matto's notes)
3. Confirm or adjust each of these columns:
   - **Final Name** — new feature name (should be explicit about temporal period: intraday_, daily_prior_, prior_day_)
   - **Final Status** — SAFE or LEAK (after fixes applied)
   - **Train Source** — exact data source and computation for training. MUST match production logic.
   - **Production Source** — where the data comes from in live trading
   - **Storage** — where the data persists (parquet file, live, derived)
   - **Refresh Frequency** — how often it updates (per minute, daily, once at startup)
   - **Dependencies** — what services must be running (ThetaData, Tradier, cron, etc.)
   - **Production Computation** — exact steps to compute at scoring time

## Key Rules
1. **Train = Production**: every feature MUST compute identically in both contexts
2. **No same-day close/high/low**: anything using today's close is LEAK — shift to prior day (T-1)
3. **FRED data shifted 2 days**: FRED has 1-2 day publication delay
4. **Naming convention**: prefix with temporal period (intraday_, daily_prior_, prior_day_)
5. **VIX intraday features**: use ThetaData VIX 1-min bars, not daily aggregates
6. **IV surface features**: from 10am intraday greeks snapshot — verify if SAFE
7. **Current minute**: use bar T open (available at scoring), NOT bar T close (not yet complete)

## Data Sources in Production
- **Tradier API**: current SPX quote, option chain with greeks (bid/ask/delta/IV)
- **ThetaData (localhost:25503)**: SPX 1-min OHLC bars, VIX 1-min OHLC bars
- **Parquets (loaded at startup)**: daily features pre-computed by 4:30 PM / 9:00 AM cron
- **Derived**: calendar flags, timing features from system clock

## Examples of Confirmed Features
Look at the first 90 rows for examples of properly filled columns. Key patterns:
- Entry features (credit, time, strategy): Production Source = Tradier + system clock
- VIX intraday: Production Source = ThetaData VIX 1-min, Storage = vix_1min.parquet
- Calendar flags: Production Source = derived from date, Dependencies = None
- Daily shifted features: Production Source = parquet lookup, Production Computation = "no live calc needed"

## When Done
Once all 291 features are confirmed, this audit becomes the spec for:
1. Rebuilding feature engineering scripts (fix leakage)
2. Rebuilding model table
3. Retraining models
4. Updating live_features.py for production
