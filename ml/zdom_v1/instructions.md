First inspect the existing repo in detail and do not invent files or components that are already present. Base your plan on the actual codebase state, not assumptions.

**Important:** Read `model_documentation_v1.md` for all model design decisions, assumptions, target logic, feature inventory, split methodology, and evaluation approach. That document is the source of truth for how the model works and why.

You are my lead engineer and system architect. Your job is to take the strategy and specification below and turn it into a fully buildable, step by step implementation plan inside my local codebase.

Project owner: Matto
Project path: ~/ironworks/projects/options-model/
Broker: Tradier
Data sources: ThetaData on localhost:25503, FRED, yfinance

Your role:
You are not just summarizing. You are acting like a senior engineer who must understand the entire system, inspect the existing repo, identify what is already built, what is partially built, what is broken, and what is missing, then create an execution plan to finish it correctly.

Primary objective:
Build a fully autonomous 0DTE SPX options trading system that:
1. Fetches and validates market and macro data daily
2. Builds feature sets and target variables
3. Trains and evaluates multiple machine learning models with proper time series methodology
4. Scores the current trading day at 10:00am ET
5. Decides trade or skip using model consensus plus blocker rules
6. Selects entry timing and strike structure
7. Executes paper trades first through Tradier sandbox
8. Logs all decisions and outcomes
9. Only enables live trading after explicit human approval

What I need from you:
I want you to fully understand the project and then walk me through how to build it in exact engineering order. I do not want a vague overview. I want concrete implementation steps tied to actual files, scripts, directories, functions, configs, commands, and validation checks.

Your workflow:
Follow this exact sequence.

Section 1: Repo understanding
1. Inspect the full repository structure
2. Identify all existing scripts, configs, notebooks, data folders, and make targets
3. Explain what each major file appears to do
4. Identify what parts of the pipeline already exist versus what is missing
5. Flag dead code, placeholder files, stale experiments, and possible credential issues
6. Produce a concise current state summary

Section 2: Architecture mapping
Create a clear architecture map for the full system, including:
1. Data ingestion layer
2. Data storage layer
3. Feature engineering layer
4. Target generation layer
5. Model training layer
6. Evaluation and explainability layer
7. Live scoring layer
8. Trade execution layer
9. Logging and monitoring layer
10. Environment separation for dev, UAT, and prod

For each layer, state:
1. Purpose
2. Inputs
3. Outputs
4. Main files involved
5. Missing pieces
6. Risks or design issues

Section 3: Strategy to system translation
Translate the trading strategy into explicit software requirements.

Use the following strategy rules as system requirements:
1. Trading window is 10:00am ET to 3:00pm ET only
2. Hard blockers always override model signals
3. Soft blockers require stronger model confidence
4. Five parallel models must run on different lookback windows
5. Consensus rule is 3 of 5 models must agree to trade
6. Walk forward time series CV only
7. Holdout is touched exactly once
8. Every trade and every skip must be logged with reasoning
9. Live trading is forbidden until explicit approval
10. Max daily risk and sizing constraints must be enforced in code

For each strategy rule:
1. Explain how it should be represented in software
2. State where it should live in the codebase
3. Define how it should be tested
4. Identify failure modes

Section 4: Gap analysis
Compare the desired end state to the current repo state.
Create a table with:
1. Component
2. Exists now
3. Status as complete, partial, broken, or missing
4. Required next action
5. Priority as critical, high, medium, or low

Focus especially on:
1. Data fetch reliability
2. ThetaData connectivity
3. Tradier auth and environment separation
4. Target variable quality
5. Feature table integrity
6. Model training correctness
7. Backtesting realism
8. Live scoring readiness
9. Paper execution readiness
10. Logging, monitoring, and kill switches

Section 5: Build plan
Create an exact build plan in the best engineering order.
Break it into phases and steps.
For each step include:
1. Objective
2. Why it matters
3. Files to inspect or create
4. Functions or modules to implement
5. Inputs and outputs
6. Dependencies
7. Commands to run
8. How to verify success
9. Common failure points

Make this practical and sequential so an engineer could execute it without guessing.

Section 6: File by file implementation guide
Based on the repo, propose the exact files that should exist.
For each important file, state:
1. File path
2. Purpose
3. Main classes or functions inside it
4. What it should read
5. What it should write
6. What tests should cover it

If a file already exists, explain whether to keep, refactor, or replace it.
If a file is missing, propose where it should be created.

Section 7: Data contracts
Define the expected schema for major datasets and tables.
I want explicit schemas for:
1. Raw market data
2. Options chain data
3. Greek data
4. Macro data
5. Feature tables
6. Target table
7. Final model table
8. Live scoring input
9. Trade log
10. Shadow trade log

For each schema define:
1. Primary keys
2. Required fields
3. Timezone rules
4. Data types
5. Null handling expectations
6. Validation checks

Section 8: Target variable redesign
The current target appears unusable due to extreme class imbalance.
I want you to deeply analyze the target design.

Evaluate these target options:
1. tp25 hit
2. pnl percent regression
3. multiclass win size outcome
4. risk tier classification

Tell me:
1. Which one is best for V1
2. Why
3. Tradeoffs
4. Expected class balance
5. How to simulate it correctly
6. What assumptions must be locked down

Then propose exactly how build_target.py should work step by step.

Section 9: Feature engineering review
Review the full feature set and tell me:
1. Which features are genuinely useful
2. Which are redundant
3. Which may leak future information
4. Which are too sparse or noisy
5. Which should be dropped or reworked

Then define a clean feature engineering policy:
1. Naming conventions
2. Lookback safety
3. Null thresholds
4. Winsorization or clipping if needed
5. Imputation rules
6. Feature versioning
7. Drift monitoring

Section 10: Modeling design
Design the training and evaluation framework for the five model ensemble.

I want:
1. Exact split methodology
2. Gap rules between train, test, and holdout
3. How each lookback window is trained
4. What happens if one model underperforms
5. Threshold selection method
6. Probability calibration approach
7. Consensus logic implementation
8. SHAP workflow
9. Model artifact versioning
10. Retraining policy

Also tell me whether XGBoost remains the best default or whether another model should be benchmarked.

Section 11: Backtest design
Explain how to build a realistic backtest for this system.
It must account for:
1. Time gated entries
2. Hard blockers
3. Soft blockers
4. Trade skip logic
5. Position sizing
6. Slippage assumptions
7. Profit target logic
8. Expiry outcomes
9. Alternative structure selection if added later

Define:
1. Backtest inputs
2. Backtest loop
3. Output metrics
4. Stress test scenarios
5. What would make the backtest invalid or overly optimistic

Section 12: Live system design
Design the live daily workflow from premarket to close.
Include:
1. Scheduler timing
2. Fetch sequence
3. Validation gates
4. Score generation
5. Trade decision object
6. Order placement flow
7. Monitoring loop
8. Take profit logic
9. End of day logging
10. Failure handling and abort conditions

I want this described both conceptually and as a concrete script flow.

Section 13: Risk controls
Define all the risk controls that must exist in code before live deployment.

Required controls include:
1. Hard daily max loss
2. Max position size
3. Environment lockout to prevent live token misuse
4. No live execution without approval flag
5. Broker response validation
6. Duplicate order prevention
7. Kill switch
8. Trading window enforcement
9. Missing data abort
10. Low confidence no trade rule

For each control:
1. Where it should be implemented
2. What should happen when it triggers
3. How it should be tested

Section 14: Testing plan
Create a complete testing strategy.
I want:
1. Unit tests
2. Integration tests
3. Data validation tests
4. Backtest sanity tests
5. Paper trading dry run tests
6. Production readiness checklist

For each major module, tell me what must be tested before moving forward.

Section 15: Execution roadmap
Turn all of this into a practical roadmap I can execute over the next 10 days.
For each day:
1. Specific deliverables
2. Commands to run
3. Files expected to change
4. Success criteria
5. Blockers that could prevent progress

Section 16: Immediate next actions
After reviewing the repo, end with:
1. The top 10 highest priority actions
2. The single biggest technical risk
3. The single biggest trading risk
4. The fastest path to paper trading readiness
5. The exact first command I should run

Important instructions:
1. Be highly specific and concrete
2. Do not give generic ML advice
3. Tie every recommendation to this actual system
4. Assume this is a serious trading system where mistakes are expensive
5. Prefer robust, simple, testable designs over clever designs
6. Flag any place where the spec is logically inconsistent or dangerous
7. If repo contents conflict with the strategy spec, call that out explicitly
8. If something is missing, propose the exact implementation
9. If you are uncertain, say so clearly and explain what to inspect next
10. Do not skip steps

Strategy and system spec:

# Zero DTE Options Strategy

Automated 0DTE SPX Iron Condor trading system powered by machine learning. The system collects real-time market data, runs ensemble ML models at 10am ET daily, decides whether to trade (go/no-go), selects optimal entry timing and strike deltas, opens an Iron Condor (or alternative structure), and closes at a take-profit target — fully autonomous with human override.

**Owner:** Matto
**Code:** `~/ironworks/projects/options-model/`
**Broker:** Tradier (sandbox for dev/paper, `api.tradier.com` for prod)
**Data:** ThetaData Index Standard ($30/mo) on localhost:25503 + FRED + yfinance

---

## How It Works (End State)

1. **Pre-market** (before 9:30am ET): Daily fetch cron pulls overnight data — SPX/VIX/VIX1D intraday, option chains, greeks, term structure, macro indicators
2. **10:00am ET**: V1 model runs go/no-go prediction using 249 features across 5 lookback windows. 3-of-5 consensus required to trade. Hard/soft blockers checked.
3. **If TRADE**: V2 model selects optimal entry minute (10am–3pm). V3 selects short/long delta targets.
4. **Entry**: Open Iron Condor at selected deltas. Credit collected targets ~1-2% of wing width.
5. **Management**: Monitor position. Close at take-profit (e.g., 25% of max credit) or let expire worthless.
6. **Close**: Log outcome, update microstructure features for next day's model input.

## Model Versioning

| Version | Purpose | Target | Status |
|---------|---------|--------|--------|
| V1 | Go/No-Go oracle — should I trade today? | `good_trade` (binary) or `risk_tier` (3-class) | Building |
| V2 | Entry timing — given TRADE, which minute? | Best `ic_pnl_pct` within day | Planned |
| V3 | Strike selection — what delta for short/long? | Optimal delta pair | Planned |
| V4 | Structure selection — IC vs butterfly vs jade lizard vs credit spread | Best structure for regime | Planned |

---

## 10-Day Plan

### Day 1 — Environment & Infrastructure Audit
- [ ] Verify Python env, dependencies (`pip install -r requirements.txt`)
- [ ] Confirm Theta Terminal runs on localhost:25503
- [ ] Confirm Tradier sandbox API key works (`sandbox.tradier.com`)
- [ ] Confirm all API keys are valid: ThetaData, Yahoo Finance, FRED
- [ ] Rotate any expired keys. Scrub old creds from git history (`git filter-repo`)
- [ ] Run `make fetch` end-to-end — confirm every fetch script succeeds
- [ ] Inventory existing data: what's fresh, what's stale, what's missing

### Day 2 — Data Collection & Validation
- [ ] Backfill any gaps in historical data (SPX daily, VIX, VIX1D)
- [ ] Backfill intraday greeks if incomplete
- [ ] Fetch econ calendar, FOMC dates, MAG7 earnings, term structure
- [ ] Fetch macro data from FRED (yield curve, credit spreads) — needs FRED_API_KEY
- [ ] Validate all parquet files: row counts, date ranges, null rates
- [ ] Document data coverage: start date → end date for each source

### Day 3 — Target Variable Rebuild
- [ ] **Fix the target variable** — current `ic_profitable` is 97.8% positive (useless)
- [ ] Options to evaluate:
  - `ic_tp25_hit` — did the IC hit 25% take-profit?
  - `ic_pnl_pct` — regression on P&L percentage
  - Multi-class: big win / small win / loss
- [ ] Rebuild `build_target.py` with chosen approach
- [ ] Validate class balance (target minority class 20-40%)
- [ ] Run `make target` and verify output
T
### Day 4 — Feature Engineering
- [ ] Run all feature build scripts: `make features`
  - GEX/dealer gamma, HMM regime, IV surface, Vanna/Charm
  - VIX1D, intraday momentum, SPX technicals, options microstructure
  - Breadth, cross-asset, vol expansion, macro regime, presidential cycles
- [ ] Validate feature count (target ~300 features)
- [ ] Check null rates — drop or impute features with >20% nulls
- [ ] Run `make model_table` — join all features + target into training table
- [ ] Sanity check: feature distributions, correlations, obvious leakage

### Day 5 — Model Training (Walk-Forward CV)
- [ ] Train 5 parallel XGBoost models at different lookback windows:
  - 2-Year (~500 days), 1-Year (~252), 180-Day (~126), 90-Day (~63), 7-Day (~5)
- [ ] Walk-forward time-series CV only — no random splits
- [ ] 60% train / 20% test / 20% holdout. Gaps between splits
- [ ] Run `make train` — `train_all_horizons.py`
- [ ] Log AUC per model per fold. Minimum AUC >= 0.70
- [ ] Monitor train vs test AUC gap (flag overfitting if gap > 0.15)

### Day 6 — Model Evaluation & SHAP
- [ ] Run `make eval` — `evaluate.py`
- [ ] SHAP analysis on every model: top features, stability across folds
- [ ] Compare 5 models: which agree, which diverge
- [ ] Test 3-of-5 consensus rule: what's the effective trade rate?
- [ ] Generate performance report: AUC, precision, recall, calibration curves
- [ ] If any model < 0.70 AUC, investigate — retrain or drop that horizon

### Day 7 — Backtesting
- [ ] Simulate full trading history with the 3-of-5 consensus model
- [ ] Apply all hard blockers (FOMC, CPI, VIX>35, etc.) — verify they fire correctly
- [ ] Apply soft blockers (VIX 25-35 size down, FOMC week, etc.)
- [ ] Half-Kelly position sizing: `f* = (edge / odds) * 0.5`
- [ ] Track: win rate, avg win, avg loss, max drawdown, Sharpe, Sortino
- [ ] Compare to baseline: prior bot did +26% in 4 weeks. Can the model beat it?
- [ ] Stress test: what happens on worst historical days? (Feb 5 2018, March 2020)

### Day 8 — Paper Trading Setup
- [ ] Wire up `score_live.py` to pull real-time data and score with all 5 models
- [ ] Connect to Tradier sandbox (`sandbox.tradier.com`)
- [ ] Implement `execute_trade.py` for paper orders: IC construction, layered entry
- [ ] Implement `shadow_trade_log.py` — log every TRADE and SKIP with reasoning
- [ ] Set up daily automation: Theta Terminal → fetch → score → trade/skip → log
- [ ] Verify order construction: correct strikes, widths, expirations

### Day 9 — Paper Trading Execution & Monitoring
- [ ] Run live paper trading for the full session (10:00am - 3:00pm ET)
- [ ] Monitor in real-time: model scores, consensus, trade decisions
- [ ] Verify hard/soft blockers fire correctly on live data
- [ ] Check position sizing calculations against live portfolio value
- [ ] Review shadow trade log at end of day
- [ ] Compare paper results to what backtest predicted for that day
- [ ] Fix any bugs, edge cases, data timing issues

### Day 10 — Production Prep & Go-Live (Tradier Live)
- [ ] Review all paper trading results with Matto
- [ ] **Get Matto's explicit approval before switching to live**
- [ ] Swap Tradier sandbox token for live token (`api.tradier.com`)
- [ ] Set max daily loss limit in code (hard stop)
- [ ] Set max position size limit (2% portfolio risk per day)
- [ ] First live day: reduced size (25% of normal) as a sanity check
- [ ] Monitor every trade in real-time on Day 1
- [ ] End-of-day review: P&L, fills, slippage vs backtest expectations

---

## Core Insight

Regardless of which delta you use to build an iron condor (10, 15, 25), they all return roughly the same percentage over time. The market efficiently prices volatility across the strike range. The edge comes from **when** you trade, not where you place the strikes. The model is a go/no-go oracle. Strike selection is secondary.

## Risk Philosophy

Maximize long-run expected log-wealth. Avoid the left tail. One blow-up can erase months of work.

- 0DTE Iron Condors are short gamma: collect small premium daily, can lose 5-10x on a gap move
- 95 winning days of $500 = +$47,500. 5 losing days of $5,000 = -$25,000. 1 black swan of $25,000 = net negative
- Capital preservation is existential. 25% drawdown delays the whole plan by a year

## Hard Blockers (NEVER trade)

FOMC day/eve, CPI/PPI day, NFP day, GDP day, MAG7 earnings day, 3+ S&P mega-cap earnings, major bank earnings week, VIX > 35, SPX gap > 1.5%, 3+ consecutive losses, negative GEX regime (below GEX flip level).

## Soft Blockers (ML score must be > 0.70)

VIX 25-35 (size down 50%), FOMC week (not day), month/quarter-end, Monday after 1%+ Friday, SPX near GEX flip level (within 0.5%).

## Position Sizing

Half-Kelly: `f* = (edge / odds) * 0.5`. If edge < 15%, don't trade. Max 2% portfolio risk per day. Hard cap.

## Strategy Menu

| Strategy | When |
|---|---|
| Iron Condor | Low vol, no directional bias, VIX 15-25 |
| Iron Butterfly | Very low vol, tight range expected |
| Put Credit Spread | Bullish bias detected |
| Call Credit Spread | Bearish bias, risk-off |
| Jade Lizard | Bullish with elevated put skew |
| Wide Strangle | Post-FOMC resolution, expected large move |

Layered entry is the default: open one wing first, wait for confirmation, add second wing (or don't).

## Trading Window

**10:00am – 3:00pm ET only.** No entries before 10am (chaotic open) or after 3pm (gamma decay acceleration). Hard rule.

## Model Architecture

5 parallel XGBoost models at different lookback windows:

| Window | Days | Purpose |
|---|---|---|
| 2-Year | ~500 | Full macro cycles, regime classification |
| 1-Year | ~252 | Seasonal patterns, annual cycles |
| 180-Day | ~126 | Current vol regime, primary trading signal |
| 90-Day | ~63 | Current momentum, short-term edge |
| 7-Day | ~5 | Micro-regime, final go/no-go filter |

**Decision: 3-of-5 models must agree for TRADE. Disagreement = SKIP.**

## Features (249 total, 14 feature tables)

| Category | Source File | Features | Description |
|----------|------------|----------|-------------|
| SPX technicals | `spx_features.parquet` | ~60 | Gap, range, SMAs, RSI, MACD, BB, Yang-Zhang HV, HV/IV ratios, ATR, efficiency ratio, choppiness |
| Options | `options_features.parquet` | ~33 | ATM IV, put/call skew, OI, volume ratios, IC credit at standard widths |
| IV surface | `iv_surface_features.parquet` | ~24 | Term structure slopes, skew metrics, smile curvature, vol-of-vol |
| Regime (HMM) | `regime_features.parquet` | 5 | 2-state volatility regime, transition probabilities |
| GEX regime | `gex_regime_features.parquet` | 6 | 3-state GEX regime, net GEX, dealer gamma positioning |
| Vanna/Charm | `vanna_charm_features.parquet` | 7 | Net vanna, charm exposure, gamma/vanna ratio |
| Momentum | `momentum_features.parquet` | 8 | First 15/30/60min return, range, direction, open drive strength, gap reversal |
| VIX1D | `vix1d_daily.parquet` | 4 | VIX1D close, z-score, vs-SMA ratios, VIX1D/VIX ratio |
| Macro | `macro_regime.parquet` | ~35 | Yield curve, credit spreads, gold, claims, fed funds |
| Cycles | `presidential_cycles.parquet` | ~41 | Presidential cycle, election proximity, seasonal calendar |
| Breadth | `breadth_features.parquet` | ~15 | RSP/SPY ratio, IWM momentum, breadth thrust |
| Cross-asset | `cross_asset_features.parquet` | ~20 | DXY, gold, EEM, QQQ relative strength, correlations |
| Vol expansion | `vol_expansion_features.parquet` | 8 | VVIX, vol surface metrics |
| Microstructure | `microstructure_features.parquet` | 8 | Prior IC outcome, consecutive wins/losses, streak features |
| Intraday timing | (computed in model table) | 6 | Entry hour, minute, minutes since open, time sin/cos |

## Training Rules

- Walk-forward time-series CV only. No random splits, ever.
- 60% train / 20% test / 20% holdout. Holdout touched exactly once.
- Gaps between splits >= largest rolling window used in features.
- AUC >= 0.70 minimum per model. Below that, model abstains.
- SHAP analysis on every training run.
- Retrain quarterly minimum. Log feature drift.
- Monitor train vs test AUC gap (0.95 train / 0.65 test = overfitting).

## Pipeline

```
make fetch         # Fetch all data (requires Theta Terminal on :25503)
make features      # Build all features
make target        # Simulate IC outcomes
make model_table   # Join into training table
make train         # Train XGBoost models
make eval          # Evaluate (AUC, SHAP)
make score         # Score today
make all           # Full pipeline
```

## Data Sources

| Source | Data | Cost |
|---|---|---|
| ThetaData | SPX 1-min OHLC, VIX 1-min, VIX1D 1-min, option chains (EOD + OI), intraday greeks, term structure | $30/mo |
| FRED | Yield curve (2s10s, 3m10y), credit spreads (HY, IG), unemployment claims, LEI, PMI | Free |
| yfinance | Gold (GC=F), market breadth (RSP, IWM, QQQ, EEM, DXY, HYG) | Free |

## Environments

- **Dev:** paper, subset data, Tradier sandbox
- **UAT:** paper, full historical, Tradier sandbox
- **Prod:** live, real-time, Tradier live (requires Matto's sign-off)

## Rules

1. Never use live Tradier token outside prod
2. Never execute live trades without Matto's explicit approval
3. No live trading tokens until Matto says so
4. Walk-forward CV only — no random splits
5. Gaps between train/test/holdout splits
6. Holdout touched exactly once
7. Log every TRADE and SKIP with reasoning
8. SHAP on every training run
9. AUC >= 0.70 minimum per model
10. Hard blockers override everything
11. Daily data fetch is not optional
