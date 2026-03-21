# ZDOM V2: Dynamic Position Management System

## Vision

V1/V1.2 optimized **entry** — 284 features, 9 ML models, hard blockers, diversified multi-IC positioning. But once a position is open, monitoring is purely mechanical P&L watching (fixed SL at 2x credit, SLTP ratchet at 30%, 30-second polling). Nothing reacts to *why* the market is moving or *whether conditions have changed*.

V2 adds **adaptive exit intelligence** — real-time regime detection, dynamic TP/SL adjustment, and an ML exit advisor that scores "should I stay or should I go" every monitoring cycle.

---

## Research Foundation

### Why Exit > Entry (Evidence Summary)

The claim that exit management dominates entry optimization is supported by 300,000+ trades across independent studies:

| Study | Trades | Key Finding |
|-------|--------|-------------|
| **ProjectFinance (2007-2017)** | 71,417 SPY iron condors | Same entries + 16 different exit combos → outcome variance dominated by exit rules, not entry |
| **ThetaProfits + Sandvand replication** | 9,100 + 1,344 | 38% win rate but 2.4x winner/loser ratio → 84.5% annual. Entry is mechanical (hourly 10-15d ICs). All edge from per-side SL = total premium |
| **Option Alpha** | 230,000 0DTE trades | Same IC, same triggers: 50% allocation = -67%, 30% allocation = +6.86%. Exit/sizing param flipped sign of returns |
| **Tastytrade (2005-2017)** | SPY 1-SD strangles | 50% TP management → $2.04/day vs hold-to-expiration → $1.18/day. 73% more per day from exit rule alone |
| **Van Tharp / Basso** | Random entries | Coin-flip entries + disciplined exits = 7% CAGR. Random entry system was profitable |
| **Brinson, Hood & Beebower (1986)** | Portfolio study | 93.6% of return variation explained by allocation (how much, when to cut), not selection (what to enter) |

### Why 0DTE Specifically Needs Adaptive Exits

**Gamma regime changes intraday** — documented by SpotGamma and academic papers:

- **Positive dealer gamma** → price reversals (safe for ICs, prices snap back)
- **Negative dealer gamma** → price momentum (dangerous, moves accelerate against you)
- Regime can flip multiple times per session (SpotGamma documented 3 regime changes in a single day)
- Impact: up to 3.3pp of annualized volatility difference between regimes (Adams, Fontaine, Ornthanalai 2024)
- **Dim, Eraker, Vilkov (2023):** Positive MM gamma strengthens reversals; negative gamma strengthens momentum

Your entry at 10:15 AM cannot predict the gamma regime at 2:30 PM. Only real-time monitoring can.

**The gamma crunch problem:** Iron condors have net negative gamma. As expiration approaches, gamma increases exponentially for ATM options. A 0DTE ATM option can swing from 0.50 to 0.80 delta on a small move. Losses accelerate non-linearly as price approaches your short strike — traditional percentage-based stops trigger too late.

**Caveat:** Option Alpha's stop-loss backtest found removing stop-losses increased total returns by 50-87% for short premium — but drawdowns were 2-3x worse. The theoretical "no stops" return is unrealizable for most accounts (a -27% drawdown typically leads to behavioral blowup). Stops make the strategy survivable, which is a form of edge.

---

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────┐
│                    V2 MONITORING LOOP                     │
│                  (replaces static P&L watch)              │
│                                                          │
│  Every 1 min while position open:                        │
│                                                          │
│  1. FEATURE STREAM (Phase 1)                             │
│     ├── Gamma/Dealer signals                             │
│     ├── Regime features (HMM, CUSUM, ATR)                │
│     ├── VIX intraday features                            │
│     └── Position-aware features                          │
│                                                          │
│  2. EXIT ADVISOR (Phase 2)                               │
│     ├── Score: P(should_exit_now)                        │
│     ├── Score: P(adverse_move_5min)                      │
│     └── Output: hold_confidence [0,1]                    │
│                                                          │
│  3. DYNAMIC TP/SL (Phase 3)                              │
│     ├── ATR-adaptive stop loss                           │
│     ├── Time-decay TP progression                        │
│     ├── Delta-based emergency exit                       │
│     └── Regime-weighted adjustments                      │
│                                                          │
│  4. ACTION                                               │
│     ├── hold_confidence > 0.6 → HOLD (existing logic)   │
│     ├── hold_confidence 0.3-0.6 → TIGHTEN (Phase 3)     │
│     └── hold_confidence < 0.3 → EXIT NOW                 │
└──────────────────────────────────────────────────────────┘
```

---

## Phase 1: Real-Time Feature Stream

### Goal
Build an intraday feature pipeline that runs **during the trade**, not just at entry. Every 1 minute while a position is open, compute ~20 features describing the current market regime and position state.

### Feature Spec

#### 1A. Gamma / Dealer Signals (4 features)

| Feature | Source | Logic |
|---------|--------|-------|
| `net_gex_sign` | Option chain OI + greeks | +1 if net dealer gamma positive (pinning/safe), -1 if negative (amplifying/danger). Compute from aggregate OI × gamma across strikes |
| `dist_to_gamma_flip` | Option chain OI + greeks | SPX price distance to the strike where dealer gamma crosses zero, in ATR units. Crossing this level = regime change |
| `short_strike_delta` | Option chain greeks | Current delta of your short call and short put. Entry delta ~0.10; if it rises to 0.20+ it's an early warning before P&L shows deterioration |
| `short_strike_gamma` | Option chain greeks | Current gamma of short strikes. Exponentially increases as price approaches, quantifies acceleration risk |

**Why these matter:** Delta leads P&L. A 10-delta strike becoming 20-delta is a red flag *before* the debit price reflects it. GEX sign tells you whether the market's natural tendency is to revert (safe) or trend (dangerous). These are the earliest possible warning signals.

**Data source:** You already fetch the full 0DTE option chain every monitoring cycle from Schwab/Tradier. Gamma and delta are available per strike. Net GEX requires aggregating `OI × gamma × 100 × spot_price` across all strikes (calls positive, puts negative for dealer perspective).

#### 1B. Regime Detection Features (6 features)

| Feature | Source | Logic |
|---------|--------|-------|
| `rolling_atr_ratio` | SPX 1-min bars | 5-min rolling ATR / session ATR. >1.5 = vol expansion, <0.7 = compression. Detects when volatility is spiking or dying |
| `hmm_state_prob_highvol` | HMM model output | P(high-vol regime) from 3-state HMM. Retrain daily on prior 20 days of 1-min bars. Use `hmmlearn` GaussianHMM |
| `hmm_state_prob_trending` | HMM model output | P(trending regime). When high, mean-reversion assumptions break down |
| `cusum_score` | SPX 1-min returns | CUSUM statistic on returns. Detects structural breaks (sudden shift in mean return) without lag. Threshold = 2σ of session returns |
| `vix_vix1d_spread_change` | VIX + VIX1D bars | Change in VIX-VIX1D spread since position entry. Widening = stress increasing. Narrowing = calming |
| `vix_intraday_range_pct` | VIX 1-min bars | (VIX high - VIX low) / VIX open since market open. High = volatile session |

**HMM implementation detail:**
```python
from hmmlearn import hmm

# Train daily before market open
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
# Features: [1-min return, 5-min rolling vol, volume_ratio]
# Fit on last 20 trading days of 1-min bars
model.fit(X_train)

# During trade: get current state probabilities
state_probs = model.predict_proba(X_current)
# Map states to regimes by emission means (lowest vol = calm, highest = crisis)
```

**CUSUM implementation detail:**
```python
def cusum_score(returns, threshold_sigma=2.0):
    """Cumulative sum change-point detection."""
    mu = returns.mean()
    sigma = returns.std()
    threshold = threshold_sigma * sigma
    s_pos, s_neg = 0.0, 0.0
    for r in returns:
        s_pos = max(0, s_pos + (r - mu) - threshold / 2)
        s_neg = max(0, s_neg - (r - mu) - threshold / 2)
    return max(s_pos, s_neg) / sigma  # normalized score
```

#### 1C. Position-Aware Features (6 features)

| Feature | Source | Logic |
|---------|--------|-------|
| `minutes_to_close` | Clock | Minutes until 3:00 PM ET forced close. Theta acceleration is non-linear — last 2 hours are qualitatively different |
| `pnl_pct_of_max` | Position state | Current unrealized P&L as % of max possible profit (full credit). Range: -∞ to 1.0 |
| `dist_to_short_call_atr` | SPX price + ATR | (Short call strike - SPX price) / rolling ATR. How many "vol units" of cushion you have on the call side |
| `dist_to_short_put_atr` | SPX price + ATR | (SPX price - Short put strike) / rolling ATR. Cushion on put side |
| `sltp_floor_pct` | SLTP state | Current SLTP locked floor as % of credit. 0 if SLTP not yet activated. Captures "how much profit is already secured" |
| `time_in_trade_minutes` | Position state | Minutes since fill. Combined with PnL, tells you if theta is working as expected |

#### 1D. Cross-Asset Intraday (3 features)

| Feature | Source | Logic |
|---------|--------|-------|
| `spx_vwap_offset_pct` | SPX 1-min bars | (SPX price - VWAP) / VWAP × 100. Far from VWAP = extended, likely to revert (good for IC) |
| `tick_cumulative` | NYSE tick (if available) | Cumulative NYSE TICK. Extreme readings (+1000/-1000) signal exhaustion |
| `put_call_ratio_change` | Option chain | Change in 0DTE put/call volume ratio since entry. Rising = fear increasing |

### Data Pipeline

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ ThetaData / Schwab  │────▶│ Feature Builder   │────▶│ Feature Vector  │
│ - SPX 1-min OHLCV   │     │ (extends v1_2     │     │ ~19 features    │
│ - VIX 1-min         │     │  live_features.py) │     │ every 1 minute  │
│ - 0DTE option chain │     │                    │     │                 │
│ - VIX1D (if avail)  │     │ + HMM model state  │     │                 │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
```

Extends the existing `ml/zdom_v1_2/6_execution/scripts/live_features.py` pattern — same data sources, just computed every minute during the trade instead of only at entry.

---

## Phase 2: ML Exit Advisor

### Goal
Train a classification model that predicts whether to hold or exit an open 0DTE iron condor, scored every monitoring cycle.

### Target Definition

**Primary target: `should_exit_now`** (binary)

For every minute of every historical IC trade trajectory:
- Compute `pnl_if_close_now = (credit - current_debit) × qty × 100 - fees`
- Compute `pnl_if_hold = actual_final_pnl` (what actually happened if you held to natural exit)
- Label = 1 if `pnl_if_close_now > pnl_if_hold` (closing now is better than holding)
- Label = 0 if holding produces better outcome

This creates a **counterfactual target** — at each minute, did the trader benefit from holding or would they have been better off exiting?

**Secondary target: `adverse_move_5min`** (binary)

- Label = 1 if the position loses >15% of remaining credit in the next 5 minutes
- Label = 0 otherwise
- This is a **danger detection** model — fires before damage occurs

**Why these targets:**
- `should_exit_now` directly optimizes the exit decision
- `adverse_move_5min` provides early warning that the primary model can use, and works independently as a "get out now" signal

### Training Data Construction

You already have the building blocks:

1. **Historical IC trajectories** from your target builder (`build_target_v1.py`):
   - Entry time, strikes, credit, quantity
   - Minute-by-minute debit prices from ThetaData
   - Final exit reason (TP, SL, close_win, close_loss)

2. **For each trajectory, at each minute**, reconstruct:
   - All Phase 1 features (from historical 1-min SPX/VIX bars + option chain snapshots)
   - Position-aware features (current debit, P&L, time remaining)
   - The counterfactual label (close now vs. hold)

3. **Expected dataset size:**
   - ~500 trading days × ~4 entries/day × ~180 minutes/trade = ~360,000 labeled rows
   - Each row has ~19 Phase 1 features + position features
   - Plenty for tree-based models

### Training Pipeline

```python
# Pseudocode for training data construction

for each historical trade in backtest:
    entry_time, strikes, credit, qty = trade.entry_info
    trajectory = get_minute_by_minute_debit(trade)  # already built in target builder

    for minute_idx, (timestamp, debit) in enumerate(trajectory):
        # Phase 1 features at this timestamp
        features = build_realtime_features(timestamp, strikes, debit, credit, ...)

        # Counterfactual label
        pnl_close_now = (credit - debit) * qty * 100 - fees
        pnl_hold = trade.final_pnl
        label_exit = 1 if pnl_close_now > pnl_hold else 0

        # Adverse move label
        if minute_idx + 5 < len(trajectory):
            future_debit = trajectory[minute_idx + 5].debit
            credit_remaining = credit - debit
            loss_pct = (future_debit - debit) / max(credit_remaining, 0.01)
            label_adverse = 1 if loss_pct > 0.15 else 0

        dataset.append({**features, 'should_exit': label_exit, 'adverse_5min': label_adverse})
```

### Model Architecture

**Algorithm:** LightGBM (same stack as v1_2, proven on your data)

**Why not LSTM/RL:**
- LightGBM handles tabular features well and you have ~360K rows — plenty
- RL (PPO/DQN) needs enormous trajectory count for stable policies, massive hyperparameter tuning, reward shaping
- LSTM adds complexity for marginal gain on tabular data
- Start simple. If LightGBM ceiling is hit, then consider sequence models

**Two models:**
1. `exit_advisor_lgbm_v2.pkl` — predicts P(should_exit_now)
2. `adverse_detector_lgbm_v2.pkl` — predicts P(adverse_move_5min)

**Validation:**
- Walk-forward split matching v1_2 (train on 2024 and prior, test on holdout)
- Metrics: AUC, precision@recall=0.5, calibration curve
- **Critical metric:** Expected P&L improvement — simulate using model's exit signals vs. baseline (current static exits) on holdout period

### Integration with Monitoring Loop

```python
# In the monitoring loop (currently every 30s, change to every 60s for feature computation)

features = build_v2_realtime_features(...)  # Phase 1

# Score both models
p_exit = exit_advisor.predict_proba(features)[1]
p_adverse = adverse_detector.predict_proba(features)[1]

# Combine into hold_confidence
hold_confidence = 1.0 - max(p_exit, p_adverse * 1.5)  # adverse gets 1.5x weight (safety bias)

# Action thresholds (tuned on holdout)
if hold_confidence > 0.6:
    # Continue with existing monitoring logic (SLTP, fixed SL)
    pass
elif hold_confidence > 0.3:
    # Tighten: apply Phase 3 dynamic SL/TP adjustments
    apply_dynamic_adjustments(...)
else:
    # Exit immediately
    close_position(reason="v2_exit_advisor")
```

---

## Phase 3: Dynamic TP/SL Adjustment

### Goal
Replace fixed TP/SL thresholds with regime-adaptive ones that tighten in danger and relax in safety.

### 3A. ATR-Adaptive Stop Loss

Current: Fixed SL at 2× credit (or portfolio-based cap).

New: SL multiplier varies with current volatility regime.

```python
def dynamic_sl_debit(credit, rolling_atr, session_atr, hmm_state_probs):
    """Regime-adaptive stop loss."""
    atr_ratio = rolling_atr / session_atr

    # Base multiplier from HMM regime
    regime_mults = {
        'low_vol': 2.5,    # wider — let theta work in calm markets
        'neutral': 2.0,    # baseline (same as current)
        'high_vol': 1.5,   # tighter — cut losses faster in volatile regime
    }
    # Weighted by regime probabilities (soft, not hard switch)
    base_mult = sum(regime_mults[r] * p for r, p in zip(regime_mults, hmm_state_probs))

    # ATR adjustment — if current vol is 2x session average, tighten further
    atr_adj = max(0.8, min(1.2, 1.0 / atr_ratio))  # clamp to [0.8, 1.2]

    sl_mult = base_mult * atr_adj
    return credit * sl_mult
```

**Effect:**
- Calm market (HMM low-vol, ATR ratio 0.5): SL at 2.5× credit → wide, avoids noise exits
- Normal market: SL at 2.0× credit → same as current
- Volatile market (HMM high-vol, ATR ratio 2.0): SL at 1.2× credit → tight, preserves capital

### 3B. Time-Decay TP Progression

Current: Fixed 25% TP target throughout the trade.

New: TP target tightens as theta accelerates in the afternoon.

```python
def dynamic_tp_pct(base_tp_pct, minutes_to_close):
    """Progressive TP tightening as expiration approaches."""
    if minutes_to_close > 180:  # before noon
        return base_tp_pct  # full target (e.g., 25%)
    elif minutes_to_close > 60:  # noon to 2 PM
        # Linear tightening: 25% → 15%
        progress = (180 - minutes_to_close) / 120
        return base_tp_pct - (base_tp_pct - 0.15) * progress
    else:  # final hour
        # Aggressive tightening: 15% → 5%
        progress = (60 - minutes_to_close) / 60
        return 0.15 - 0.10 * progress
```

**Rationale from research:** The final hour has extreme gamma acceleration. A 0DTE ATM option can swing from 0.50 to 0.80 delta on a small move. CBOE's Schwartz: *"You could be up $150 five minutes before close and one small move puts you down $400-500."* Taking 5-10% profit at 2:50 PM is better than risking a gamma spike for the full 25%.

### 3C. Delta-Based Emergency Exit

This is the single highest-impact change. No ML needed — purely rule-based.

```python
def check_delta_emergency(short_call_delta, short_put_delta, threshold=0.25):
    """Exit when short strike delta exceeds danger threshold.

    Delta leads P&L for short options. A 10-delta strike becoming 25-delta
    means the probability of your strike being breached has increased 2.5x
    since entry — and the P&L impact accelerates from here (negative gamma).
    """
    if abs(short_call_delta) > threshold or abs(short_put_delta) > threshold:
        return True, "delta_emergency"
    return False, None
```

**Why this matters:** Your current monitoring checks *debit price* (lagging indicator). Delta is a *leading indicator* — it tells you the market has moved significantly toward your short strike before the full P&L impact materializes. This fires 1-3 minutes earlier than a P&L-based stop, which for 0DTE can mean the difference between a -1.5× and -2.5× loss.

### 3D. Regime-Weighted SLTP Ratchet

Current: SLTP activates at 30% profit and ratchets every 5%.

New: Activation and ratchet step adapt to regime.

```python
def dynamic_sltp_params(hmm_state_probs):
    """Regime-adaptive SLTP parameters."""
    # High-vol regime: activate earlier, ratchet tighter (protect gains aggressively)
    # Low-vol regime: activate later, ratchet wider (let winners run)
    regime_params = {
        'low_vol':  {'activate_pct': 0.40, 'ratchet_step': 0.08, 'floor_offset': 0.10},
        'neutral':  {'activate_pct': 0.30, 'ratchet_step': 0.05, 'floor_offset': 0.05},
        'high_vol': {'activate_pct': 0.20, 'ratchet_step': 0.03, 'floor_offset': 0.03},
    }
    # Weighted blend
    params = {}
    for key in ['activate_pct', 'ratchet_step', 'floor_offset']:
        params[key] = sum(
            regime_params[r][key] * p
            for r, p in zip(regime_params, hmm_state_probs)
        )
    return params
```

---

## Phase 4: GEX-Aware Position Management (Stretch Goal)

### Approximate GEX from Existing Data

You already fetch the full 0DTE option chain. Approximate net GEX:

```python
def compute_net_gex(chain, spot_price):
    """Approximate net gamma exposure from option chain.

    Assumes dealers are short calls and long puts (standard assumption).
    Positive GEX = dealers long gamma = price stabilizing.
    Negative GEX = dealers short gamma = price destabilizing.
    """
    net_gex = 0.0
    for strike in chain:
        # Calls: dealers short → negative gamma for dealers when call OI is high
        # But standard GEX convention: call OI creates positive GEX (dealers hedge by buying dips)
        call_gex = strike.call_oi * strike.call_gamma * 100 * spot_price
        put_gex = -strike.put_oi * strike.put_gamma * 100 * spot_price  # puts flip sign
        net_gex += call_gex + put_gex
    return net_gex
```

### GEX-Based Regime Override

```python
def gex_regime_adjustment(net_gex, net_gex_history):
    """Override HMM regime when GEX signals clear danger."""
    gex_percentile = percentile_rank(net_gex, net_gex_history)

    if gex_percentile < 10:  # bottom 10% = extremely negative GEX
        return 'force_high_vol'  # override to tightest stops
    elif gex_percentile > 90:  # top 10% = strongly positive GEX
        return 'force_low_vol'   # override to widest stops (pinning likely)
    return None  # no override, use HMM
```

### Gamma Flip Level as Support/Resistance

```python
def find_gamma_flip(chain, spot_price):
    """Find the strike where cumulative GEX changes sign.

    Price crossing this level = regime change from stabilizing to destabilizing.
    """
    cumulative = 0.0
    for strike in sorted(chain, key=lambda s: s.strike):
        call_gex = strike.call_oi * strike.call_gamma * 100 * spot_price
        put_gex = -strike.put_oi * strike.put_gamma * 100 * spot_price
        prev_cumulative = cumulative
        cumulative += call_gex + put_gex
        if prev_cumulative * cumulative < 0:  # sign change
            return strike.strike
    return None
```

Use as: if SPX crosses the gamma flip level while your IC is open, immediately tighten all stops by 30%.

---

## Implementation Roadmap

### Priority Order (highest impact first)

| Priority | Component | Effort | Impact | Dependencies |
|----------|-----------|--------|--------|-------------|
| **P0** | Delta-based emergency exit (3C) | 1 day | Very High | None — can ship today |
| **P1** | Phase 1 feature stream | 1 week | High | Extends existing live_features.py |
| **P2** | HMM regime detection | 3 days | High | Phase 1 (for training data) |
| **P3** | ATR-adaptive SL (3A) | 2 days | Medium-High | Phase 1 (rolling ATR) + P2 (HMM state) |
| **P4** | Time-decay TP (3B) | 1 day | Medium | None |
| **P5** | Phase 2 training data construction | 1 week | High | Phase 1 (need features for each trajectory minute) |
| **P6** | Phase 2 model training + validation | 3 days | High | P5 |
| **P7** | Phase 2 integration with bot | 2 days | High | P6 + P3 |
| **P8** | Dynamic SLTP ratchet (3D) | 1 day | Medium | P2 (HMM state) |
| **P9** | Phase 4 GEX computation | 3 days | Medium | Existing chain data |
| **P10** | Phase 4 GEX integration | 2 days | Medium | P9 + P3 |

### Quick Wins (ship before ML is ready)

1. **Delta monitoring** (P0): Add `short_strike_delta` check to existing monitoring loop. Exit when delta > 0.25. Zero ML, zero new data, massive risk reduction.

2. **Time-decay TP** (P4): Replace fixed 25% TP with the time-aware progression. Pure math, no model needed.

3. **Rolling ATR ratio** (subset of P1): Compute 5-min ATR / session ATR during monitoring. Log it. Use it to inform manual decisions before full automation.

### What NOT to Build (Yet)

- **RL/PPO agent:** Massive complexity, needs 10x more trajectory data for stable policies, reward shaping is an open research problem. LightGBM classification gets 80% of the benefit at 20% of the cost. Revisit only if Phase 2 model hits a ceiling.

- **Order flow / Level 2 data:** You don't have this data source and it's expensive ($500+/month for quality feeds). GEX + VIX + ATR regime detection covers overlapping ground from data you already have.

- **Full GEX from SpotGamma API:** Starts at $100/month. Your own approximate GEX from option chain OI is sufficient for regime detection. Consider SpotGamma only if your approximation proves too noisy.

- **LSTM sequence model:** Your features are tabular and cross-sectional per minute. LightGBM handles this natively. LSTM would require windowed input, more engineering, and is typically marginal over gradient boosting on structured data. Only consider if you want to capture multi-minute temporal patterns that single-row features miss.

---

## Success Metrics

### Backtest Metrics (compare V2 exits vs V1.2 exits on same entries)

| Metric | V1.2 Baseline | V2 Target |
|--------|--------------|-----------|
| Max drawdown | Measure current | Reduce by 30%+ |
| Sharpe ratio | Measure current | Improve by 0.3+ |
| Worst single trade loss | Measure current | Reduce by 40%+ |
| Win rate | Measure current | Maintain ±2% |
| Total return | Measure current | Maintain ±5% (accept slightly lower return for much lower risk) |

The key insight from research: V2 should primarily **reduce tail risk and drawdowns**, not increase total return. The ThetaProfits and Option Alpha data show that risk management is what makes a strategy *survivable* — and survivability IS the edge over long horizons.

### Live Monitoring Metrics

- Count of delta emergency exits (should be rare, ~2-5% of trades)
- Distribution of dynamic SL multipliers (should cluster 1.5-2.5, not always 2.0)
- SLTP activation rate by regime (should activate earlier in high-vol)
- Exit advisor score distribution (calibration — does P=0.7 mean 70% of those trades benefit from exit?)

---

## Directory Structure

```
ml/zdom_v2/
├── PLAN.md                          # This document
├── 1_data_collection/
│   └── scripts/
│       ├── build_trajectory_features.py   # Reconstruct per-minute features for historical trades
│       └── build_hmm_training_data.py     # Daily 1-min bar data for HMM training
├── 2_feature_engineering/
│   └── scripts/
│       ├── realtime_features.py           # Phase 1 feature builder (used live + for training)
│       ├── hmm_regime.py                  # HMM regime detector (train + predict)
│       ├── cusum.py                       # CUSUM change-point detection
│       └── gex.py                         # GEX computation from option chain
├── 3_models/
│   ├── exit_advisor_lgbm_v2.pkl           # P(should_exit_now)
│   ├── adverse_detector_lgbm_v2.pkl       # P(adverse_move_5min)
│   └── hmm_regime_v2.pkl                  # 3-state HMM
├── 4_training/
│   └── scripts/
│       ├── train_exit_advisor.py          # Train + validate exit advisor
│       ├── train_adverse_detector.py      # Train + validate adverse detector
│       └── evaluate_v2_vs_v1_2.py         # Head-to-head backtest comparison
├── 5_execution/
│   └── scripts/
│       ├── v2_monitor.py                  # V2 monitoring loop (replaces static monitoring)
│       ├── dynamic_sl.py                  # ATR-adaptive SL + delta emergency
│       ├── dynamic_tp.py                  # Time-decay TP progression
│       └── dynamic_sltp.py               # Regime-adaptive SLTP ratchet
└── 6_analysis/
    └── scripts/
        ├── backtest_v2_exits.py           # Simulate V2 exit logic on historical trajectories
        └── compare_v1_2_vs_v2.py          # Generate comparison report
```

---

## Key Research References

| Source | Relevance |
|--------|-----------|
| ProjectFinance: 71,417 Iron Condors (2007-2017) | Same entries, 16 exit combos → exit rules dominate outcome variance |
| ThetaProfits + Sandvand: 9,100+ trades | 38% WR + smart exits = 84.5% annual |
| Option Alpha: 230,000 0DTE trades | Sizing alone flipped returns from -67% to +7% |
| Tastytrade: Managing Winners (2005-2017) | 50% TP management → 73% more $/day than hold-to-expiration |
| Adams, Fontaine, Ornthanalai (2024) | Dealer gamma impact up to 3.3pp annualized vol |
| Dim, Eraker, Vilkov (2023) | Positive MM gamma → reversals, negative → momentum |
| SpotGamma: Anatomy of 0DTE Market | Documented intraday gamma regime shifts |
| Cont, Kukanov, Stoikov (2014) | OFI linearly predicts short-horizon price changes |
| Van Tharp / Basso: Random Entry Study | Random entries + disciplined exits = profitable |
| CBOE: Henry Schwartz Deep Dive | "Set-and-forget" vs "10-cent bid" exit technique |
| Option Alpha: Stop-Loss Strategies | Stops reduce returns 50-87% but drawdowns 2-3x. Survivability IS edge |
