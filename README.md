# 0DTE SPX Iron Condor Bot

Automated 0DTE (zero days to expiration) iron condor trading system for SPX options via the Schwab API. Uses ML-based entry gating, dynamic take-profit/stop-loss management, and a real-time Dash monitoring dashboard.

## How It Works

1. **Pre-market**: ML models (v3 risk filter + v4 entry timing) evaluate whether today is safe to trade
2. **Entry** (noon ET): Builds a 10-delta iron condor with $25 wings, sizes position to target 2% daily return
3. **Monitor** (noon–3 PM): Polls positions every 30s, applies dynamic take-profit and portfolio-based stop-loss
4. **Exit** (3 PM ET): Force-closes any remaining positions before expiration

## Key Features

- **ML Entry Gating**: Random Forest classifier predicts P(catastrophic loss) — skips ~11% of dangerous days
- **Dynamic Take-Profit**: Distributes remaining daily target across open positions proportionally
- **Portfolio-Based Stop-Loss**: Caps max loss at 10% of portfolio (replaces fixed credit multiplier)
- **Live Portfolio Value**: Reads account balance from Schwab API at startup (no hardcoded values)
- **Position Sizing**: Auto-calculates contract count from portfolio value and daily target
- **Dash Dashboard**: Real-time command center with P&L charts, payoff diagrams, position tracking

## Risk Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Daily Target | 2% | Target return per day |
| Take Profit | 25% of credit | Base TP (dynamic TP adjusts for partial fills) |
| Stop Loss | 10% of portfolio | Max loss per trade |
| Short Delta | 0.10 | 10-delta short strikes |
| Wing Width | $25 | Distance between short and long strikes |
| Entry Time | 12:00 ET | Noon entry for theta decay |
| Exit Time | 15:00 ET | Forced close before expiration |
| Max Contracts | 50 | Hard cap per position |

## Project Structure

```
├── scripts/
│   ├── zero_dte_bot.py              # Main bot (entry, monitor, exit lifecycle)
│   ├── run_dashboard.py             # Launch Dash monitoring UI
│   ├── test_ml_today.py             # Pre-market ML analysis
│   ├── train_model.py               # Train v3 risk filter model
│   ├── train_entry_model.py         # Train v4 entry timing model
│   ├── build_training_dataset.py    # Build ML training data from historical trades
│   ├── backfill_candles.py          # Backfill OHLC candle data
│   ├── backfill_order_history.py    # Import Schwab order history
│   ├── backfill_minute_pnl.py       # Backfill minute-level P&L
│   ├── ingest_candles.py            # Daily candle ingestion
│   └── setup_schwab_auth.py         # Schwab OAuth setup
│
├── config/
│   └── settings.py                  # All bot parameters (BOT_* settings)
│
├── execution/
│   ├── broker_api/
│   │   ├── schwab_client.py         # Schwab API client (orders, positions, account)
│   │   ├── auth_manager.py          # OAuth token management
│   │   └── rate_limiter.py          # API rate limiting
│   └── order_manager/
│       └── ic_order_builder.py      # Iron condor strike selection & order construction
│
├── strategies/
│   └── algo/
│       └── zero_dte_iron_condor_strategy.py  # Strategy with ML integration
│
├── data/
│   ├── collectors/
│   │   ├── thetadata_client.py      # ThetaData API (options chains, greeks)
│   │   ├── schwab_collector.py      # Schwab market data
│   │   └── historical_data.py       # Historical data loading
│   ├── providers/
│   │   ├── schwab_provider.py       # Schwab data provider
│   │   └── theta_provider.py        # ThetaData provider
│   └── storage/
│       └── database.py              # DuckDB connection & queries
│
├── ml/                              # ML pipeline (separate module)
│   ├── models/predictor.py          # v3/v4 model inference
│   ├── features/pipeline.py         # Feature engineering pipeline
│   └── data/candle_loader.py        # OHLC data loader for features
│
├── monitoring/
│   ├── dashboards/
│   │   ├── command_center.py        # Main Dash app (positions, P&L, payoffs)
│   │   └── components/              # Reusable dashboard components
│   ├── services/
│   │   ├── schwab_positions.py      # Live position fetching
│   │   ├── payoff_diagram.py        # Iron condor payoff visualization
│   │   ├── pnl_calculator.py        # P&L calculations
│   │   ├── trade_grouper.py         # Group legs into strategies
│   │   └── trade_aggregator.py      # Aggregate trade statistics
│   └── alerts/
│       └── telegram_notifier.py     # Telegram trade alerts
│
└── schwab-confirm-extension/        # Chrome extension for Schwab order confirmation
```

## Setup

### Prerequisites

- Python 3.12+
- [ThetaData](https://thetadata.net/) terminal running locally (port 25503)
- Schwab developer API credentials

### Installation

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Schwab Authentication

```bash
python scripts/setup_schwab_auth.py
```

### Run the Bot

```bash
# Pre-market check
python scripts/test_ml_today.py

# Live trading (full lifecycle)
python scripts/zero_dte_bot.py --live

# Monitor existing positions only
python scripts/zero_dte_bot.py --monitor
```

### Dashboard

```bash
python scripts/run_dashboard.py
# Open http://localhost:8050
```

## Architecture

```
ThetaData API ──→ Options Chain / Greeks ──→ Strike Selection
                                                    │
Schwab API ──→ Account Balance ──→ Position Sizing ─┘
                                                    │
ML Predictor ──→ Risk Gate (v3) ──→ Entry Timing (v4)
                                                    │
                                            Iron Condor Order
                                                    │
                                         Schwab Order Execution
                                                    │
                                    Monitor Loop (30s polling)
                                      │           │          │
                                Dynamic TP    Portfolio SL   3PM Close
```

## Disclaimer

This software is for personal use and educational purposes. Trading options involves substantial risk of loss. Past performance does not guarantee future results. You are solely responsible for your trading decisions.

## License

Private use only. Not for redistribution.
