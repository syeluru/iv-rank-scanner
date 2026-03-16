# ZDOM V1 — Zero Day Options Model

ML-powered 0DTE SPX Iron Condor entry filter. Scores 81 combinations (9 TP models x 9 IC strategies) every minute and picks the highest expected value trade.

## Project Structure

```
zdom_v1/
  1_data_collection/       Raw market data + fetch scripts (ThetaData, FRED, yfinance)
  2_feature_engineering/   284 features built from raw data + reference docs
  3_data_join/             Source of truth: target_v1.parquet + model_table_v1.parquet
  4_models/v1/             18 trained models (9 TP levels x XGB + LGBM)
  5_analysis/              Training, evaluation, walk-forward backtests, results
  6_execution/             Live trading: live_orchestrator.py + live_features.py
  src/                     Shared modules (gex, iv_surface, regime, vanna_charm)
  tests/                   Unit tests
```

## Pipeline Flow

1. **Data Collection** — `1_data_collection/scripts/daily_fetch.sh` pulls SPX/VIX/options data
2. **Feature Engineering** — `2_feature_engineering/scripts/build_*.py` computes 284 features
3. **Data Join** — `2_feature_engineering/scripts/build_model_table.py` joins features + targets
4. **Training** — `5_analysis/scripts/train_v1.py` trains 18 models with Optuna tuning
5. **Evaluation** — `5_analysis/scripts/backtest_walkforward.py` runs walk-forward on holdout
6. **Execution** — `6_execution/scripts/live_orchestrator.py` runs live paper/prod trading

## Quick Start (Paper Trading)

```bash
# Prerequisites
pip install -r requirements.txt
export TRADIER_PAPER_TOKEN=your_token
export TRADIER_ACCOUNT_ID=your_account_id

# Run paper trading ($10K portfolio, 30% skip rate)
python3 6_execution/scripts/live_orchestrator.py --portfolio 10000

# Dry run (score only, no orders)
python3 6_execution/scripts/live_orchestrator.py --dry-run --portfolio 10000
```

## Key Results (V1)

- **Best model**: TP10 XGB (holdout AUC 0.7297)
- **Walk-forward backtest**: +157% return at 30% skip rate (S1), Sharpe 2.13
- **Shadow 5d filter**: when IC_05d is the best option, skip entirely (market conditions unfavorable)
- **182 tradeable days/year** after hard blockers (FOMC, CPI, VIX>35, etc.)

## Key Files

| File | What it does |
|------|-------------|
| `model_documentation_v1.md` | Full model documentation (assumptions, methodology, results) |
| `6_execution/scripts/live_orchestrator.py` | Main trading loop |
| `6_execution/scripts/live_features.py` | Builds 284-feature vector from live data |
| `5_analysis/backtests/summary_with_05d.csv` | Walk-forward backtest results |
| `5_analysis/results/ZDOM_V1_Results.pdf` | Training results PDF |
| `4_models/v1/tp10_target_xgb.pkl` | Best model (TP10 XGB) |

## Documentation

See `model_documentation_v1.md` for the full model documentation including:
- Target definition and assumptions
- Feature dictionary (284 features)
- Training methodology and evaluation results
- Execution logic and 5-delta shadow filter
- Hard blockers and risk controls
- Version roadmap (V1-V5)
- Transaction cost analysis
