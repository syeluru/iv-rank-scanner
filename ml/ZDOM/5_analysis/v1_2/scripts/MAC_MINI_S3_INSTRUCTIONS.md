# Walk-Forward Backtest S3 — Mac Mini Execution Plan

## Context

We are running a full walk-forward backtest for ZDOM V1.2 with Matt's retrained models
(commit `626c7f3` — pct_change fix, T-1 FRED lag, retrained XGB models). The work is
split across two machines:

- **MacBook** (already running): S1 (best EV single) + S2 (diversified greedy) → `walkforward_s1s2_grid.csv`
- **Mac Mini** (this machine): S3 (knapsack optimal) → `walkforward_s3_grid.csv`

## What Changed in Matt's Update

1. DXY/gold/oil change features: `.diff()` → `.pct_change()` (percentage not absolute)
2. FRED and macro_regime features: T-2 lag → T-1 lag
3. All 9 XGB models retrained on corrected features (same date range, same splits)
4. All daily parquets rebuilt with `--clip 2026-03-13`
5. Repo restructured: canonical path is now `ml/ZDOM/` (old `ml/zdom_v1_2/` removed)
6. Master dataset: `ml/ZDOM/3_feature_engineering/v1_2/outputs/master_data_v1_2_final.parquet`

## Data & Model Details

- **Dataset**: `master_data_v1_2_final.parquet` — 1,701,426 rows × 497 cols, 742 trading days (2023-03-13 → 2026-02-27)
- **Models**: `ml/ZDOM/4_models/v1_2/tp{10-50}_target_{xgb|lgbm}_v1_2.pkl` — 18 total (9 XGB retrained, 9 LGBM original)
- **Model selection**: Best holdout AUC per TP level (XGB or LGBM)
- **285 features** per model, isotonic calibration applied
- **Train end date**: 2024-09-04 (from model `split_info`)

## Data Split (5 equal chunks post-train)

| Fold       | Date Range                    | ~Days | ~Rows    |
|------------|-------------------------------|-------|----------|
| test       | 2024-09-05 → 2024-12-18      | 74    | 168,459  |
| holdout_1  | 2024-12-19 → 2025-04-02      | 74    | 168,786  |
| holdout_2  | 2025-04-03 → 2025-07-25      | 74    | 165,547  |
| holdout_3  | 2025-07-28 → 2025-11-07      | 74    | 171,740  |
| holdout_4  | 2025-11-10 → 2026-02-27      | 75    | 177,002  |

## Permutation Grid (S3 only)

- **Strategy**: S3 (integer knapsack optimal portfolio allocation)
- **SL levels**: 10%, 15%, 20%, 25%, 30%
- **Skip rates**: 0.05 → 0.35 at 1% increments (31 levels)
- **Shadow 5D**: enabled (IC_05d_25w locks buying power, $0 PnL)
- **Slippage**: $0.20/side ($0.40 round-trip per share)
- **Fees**: $0.052/share ($5.20 RT / 100 shares)
- **Starting portfolio**: $10,000
- **Buying power per contract**: $2,500

**Total simulations**: 1 strategy × 5 SL × 31 skip × 5 folds = **775 sims**

## S3 Strategy Logic

S3 uses scipy's `milp` solver to find the optimal integer allocation of contracts
across all available candidates at each entry time, subject to:
- SL budget constraint: Σ(max_loss_i × qty_i) ≤ daily SL budget remaining
- Buying power constraint: Σ(BP × qty_i) ≤ portfolio − BP already used
- Max 50 contracts per candidate
- Fallback: greedy knapsack sorted by EV/max_loss ratio if MILP fails

## Pipeline Phases (executed by the script automatically)

1. **Data Loading** — Read master parquet, sort by date/datetime/strategy
2. **Model Loading** — Load best model per TP level by holdout AUC
3. **EV Lookup** — Precompute expected value from training data per (strategy, TP)
4. **Calibration** — Fit isotonic regression calibrators on training probabilities
5. **Per-fold loop** (5 folds):
   a. Score fold with calibrated probabilities
   b. Precompute skip-rate cutoffs per (strategy, TP, skip_rate)
   c. Build candidate pool (date → entry_time → candidates sorted by EV)
   d. Run 155 sims (5 SL × 31 skip rates) for S3
   e. Free memory between folds

## Output

- **File**: `ml/ZDOM/5_analysis/v1_2/results/backtest_walkforward/walkforward_s3_grid.csv`
- **Columns**: strategy, skip_rate, max_sl_pct, total_return_pct, max_drawdown_pct, sharpe, win_rate, n_trades, n_shadow, trades_per_day, final_portfolio, avg_pnl, period, config
- **775 rows** (one per simulation)
- Best configs printed at end, averaged across folds

## Script Location

```
ml/ZDOM/5_analysis/v1_2/scripts/backtest_walkforward_s3.py
```

## After Completion

Push the results CSV to git so the MacBook can merge S1+S2 + S3 results:
```bash
git add ml/ZDOM/5_analysis/v1_2/results/backtest_walkforward/walkforward_s3_grid.csv
git commit -m "Add S3 walkforward grid results (775 sims, $0.20/side slip)"
git push
```
