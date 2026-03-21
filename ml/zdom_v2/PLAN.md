# ZDOM V2.1: Institutional-Grade 0DTE Exit and Position Management

## Vision

V1/V1.2 optimized **entry selection**. Once a position is open, however, the live stack still behaves mostly like a debit-threshold monitor: fixed stop-loss logic, static SLTP ratchets, and account-level combined stops. That is a solid baseline, but it is not how a serious 0DTE risk engine should behave.

V2.1 upgrades exit management from a static threshold system into a **state-aware decision engine** that:

1. Estimates the **current market, vol, liquidity, and position state** every monitoring cycle.
2. Predicts **continuation value across multiple time horizons**, **short-horizon tail risk**, and **expected exit cost**.
3. Chooses among a richer set of actions than just hold vs. exit.
4. Falls back to hard risk rules whenever model confidence is weak or the regime is unfamiliar.
5. Treats **execution quality** as part of risk management, not an afterthought.

This is the design goal for an internal, institutional-quality 0DTE management stack. It is realistic, measurable, and buildable with the data and infrastructure we already have.

---

## Ground Truth

### What The Current Live System Does

The current orchestrator monitors open iron condors primarily by:

- Current debit-to-close from fresh option chain quotes
- Fixed stop-loss thresholds
- Static SLTP profit-lock tiers
- Account-level combined stop-loss

This is implemented in the existing live code:

- `scripts/orchestrator/position_manager.py`
- `execution/order_manager/ic_order_builder.py`

That means the real bottleneck is not just "more ML." The bottleneck is the full decision stack:

- better state estimation
- better labels
- better validation
- better execution modeling
- better live fail-safes

### Design Principle

For 0DTE, **the best exit system is not the fanciest model**. It is the one that survives contact with:

- spread widening
- delayed chain refreshes
- quote flicker
- regime shifts
- model uncertainty
- tail intraday moves in the final 60-90 minutes

If a proposed feature or model cannot be computed reliably and quickly in live trading, it does not belong in the critical path.

---

## What Needs To Change

### The Key Weaknesses In The Original V2 Plan

The original V2 plan was directionally right, but incomplete in five important ways:

1. **Target design was too naive.**
   A binary `should_exit_now` label based on realized final P&L is not a clean counterfactual. It teaches the model from one realized path under one historical policy.

2. **The action space was too small.**
   "Hold / tighten / exit" is not enough. A serious 0DTE manager should be able to distinguish:
   - hold
   - reduce risk
   - cut threatened side
   - flatten entire condor
   - switch to emergency execution mode

3. **Execution was under-modeled.**
   In 0DTE, the difference between quote value and fill value is often the difference between a tolerable exit and a disastrous one.

4. **Validation was too weak.**
   Ordinary holdout testing is not enough for a highly adaptive intraday strategy. We need purged / embargoed temporal validation and shadow deployment before capital promotion.

5. **Dealer gamma inference was too central.**
   Approximate GEX from open interest is useful as a weak feature. It is not reliable enough to be the sole regime spine or a dominant hard override.

---

## System Architecture

### Core Idea

V2.1 should be built as a **6-layer stack**.

### Layer 1: Live State Estimation

Estimate four state buckets every monitoring cycle:

1. **Market state**
   - price trend
   - realized volatility burst
   - distance from VWAP
   - structural break / acceleration

2. **Volatility state**
   - realized vol regime
   - IV term / intraday stress
   - short-strike gamma acceleration

3. **Liquidity state**
   - spread width
   - quote instability
   - closeable size at best quotes
   - expected fill slippage

4. **Position state**
   - debit-to-close
   - P&L
   - minutes since entry
   - minutes to forced close
   - per-side threat level
   - portfolio overlap / concentration

### Layer 2: Hard Risk Rules

These fire regardless of ML.

- short-strike delta breach
- short-strike gamma acceleration breach
- distance-to-short collapse
- severe spread dislocation
- portfolio daily loss limit
- account combined loss limit
- forced close time

Hard rules are the first line of defense and remain active even if models fail.

### Layer 3: Forecasting Models

Four continuation-value models plus hazard and exit-cost models:

1. **Continuation-value models (4 horizons)**
   Predict expected terminal P&L if we hold with no intervention to each horizon:
   - 3:00 PM ET
   - 3:30 PM ET
   - 3:45 PM ET
   - 4:00 PM ET (actual expiration)

2. **Tail-risk / hazard model**
   Predict probability of severe adverse move, stop breach, or strike-touch over short horizons.

3. **Exit-cost model**
   Predict expected slippage / fill penalty conditional on current liquidity state and urgency.

### Layer 4: Policy Engine

Choose actions under constraints:

- maximize expected value using the CV term structure
- minimize tail risk
- penalize bad execution states
- obey kill switches and capital constraints

### Layer 5: Execution Engine

Handle how we exit:

- patient profit-taking ladder
- moderate urgency marketable-limit ladder
- emergency flatten mode
- cancel/replace sequencing
- legged or net exit only where policy allows

### Layer 6: Monitoring, Drift, and Governance

- calibration tracking
- feature drift
- regime drift
- live-vs-shadow comparison
- automatic fallback to simpler rules when uncertainty rises

---

## Action Space

V2.1 should manage each iron condor as:

- one **portfolio object**
- plus two **risk units**: put spread side and call spread side

### Allowed Actions (Initial Production Set)

1. **HOLD**
   Keep current risk posture. Normal monitoring frequency.

2. **TIGHTEN**
   Concrete behavior:
   - Promote position to elevated monitoring frequency (10-15 second cycles)
   - Lower SLTP activation threshold by 10 percentage points (e.g., 30% -> 20%)
   - Reduce SL multiplier by 0.3x (e.g., 2.0x -> 1.7x)
   - If SLTP is already active, tighten the ratchet floor offset by 50%
   - Log tighten event with full state snapshot for later analysis

3. **FULL EXIT**
   Close the entire condor using Defensive execution mode.

4. **EMERGENCY EXIT**
   Immediate urgency escalation with Emergency execution mode. Aggressive pricing, immediate recheck until flat.

### Deferred Actions (Design For, Build Later)

5. **REDUCE**
   Close part of size if multiple contracts are open.

6. **SIDE-CUT**
   Exit or reduce the threatened side if supported by execution logic and margin rules.

The research framework (features, replay, labels) should support these from day one so they can be added without re-architecture.

---

## Continuation-Value Label Strategy

### Approach: Policy-Independent "No Intervention" Counterfactual

For each minute of each historical IC trajectory, compute the P&L that would result from holding with **zero intervention** (no SL, no TP, no SLTP) to four fixed time horizons.

This is possible because `spxw_0dte_intraday_greeks.parquet` contains minute-by-minute option quotes for all strikes through 4:00 PM, regardless of when the V1.2 system actually exited.

### Four Target Definitions

For training row at time `t`:

```python
cv_3pm   = (credit - debit_at_3pm)   * qty * 100  # conservative horizon
cv_330pm = (credit - debit_at_330pm) * qty * 100  # early gamma zone
cv_345pm = (credit - debit_at_345pm) * qty * 100  # late gamma inflection
cv_4pm   = (credit - debit_at_4pm)   * qty * 100  # true expiration
```

If the row's timestamp is already past a horizon (e.g., it's 3:35 PM and the target is 3:30 PM), that target is **null** for that row. The model for that horizon simply has fewer training rows from late-session entries.

### Why This Approach

- **Policy-independent.** The labels reflect what the market does, not what our old rules did. The policy engine decides what to do with that information.
- **No circular dependency.** Labels don't change when we change exit rules.
- **The 4-point term structure is more informative than any single horizon.** The *shape* (slope, curvature) tells the policy engine whether the situation is improving, stable, or deteriorating.

### Handling Tail Extremes

The 4:00 PM target will have high variance due to gamma acceleration in the final minutes. Mitigations:

- **Winsorize** cv_4pm labels at [-3x credit, +1x credit]. No single label should dominate the loss function.
- **Winsorize** cv_345pm at [-2.5x credit, +1x credit]. Slightly less extreme.
- cv_3pm and cv_330pm use lighter winsorization at [-2x credit, +1x credit].
- The policy engine weights later-horizon predictions with lower confidence (see Policy Engine section).
- The hazard model independently captures tail risk, so the CV model doesn't need to be the sole tail-risk signal.

### Training Approach

Train **four separate LightGBM regression models**, one per horizon. This allows:
- Independent evaluation of each horizon's predictive quality
- Dropping a horizon (likely 4pm) if it proves too noisy to be useful
- Different hyperparameter tuning per horizon (later horizons may benefit from stronger regularization)

---

## Hazard Model Specification

### Target Definition

Binary labels at three short horizons:

- `hazard_1m`: did the position lose >20% of remaining credit in the next 1 minute?
- `hazard_3m`: did the position lose >20% of remaining credit in the next 3 minutes?
- `hazard_5m`: did the position lose >15% of remaining credit in the next 5 minutes?

### Class Imbalance Handling

These events are rare — expected positive rate ~2-5% of minutes. Concrete mitigations:

1. **LightGBM `scale_pos_weight`**: Set to `n_negative / n_positive` (approximately 20-50x).
2. **Evaluation metric**: Use **precision-recall AUC** as the primary metric, not ROC AUC. ROC AUC is misleadingly high on imbalanced data.
3. **Minimum recall target**: 70% recall at the operating threshold. We'd rather have false alarms (exit early, miss some profit) than miss a tail event (hold through a catastrophe). False alarms cost basis points; missed tails cost multiples of credit.
4. **Threshold selection**: Use precision-recall curves on the validation set. Choose the threshold where recall >= 70% and precision is maximized subject to that constraint.
5. **Calibration**: Apply Platt scaling or isotonic regression post-training so that P(hazard) = 0.3 actually means 30% of those minutes see the adverse event.

### Training Approach

Three separate LightGBM binary classifiers. Same features as continuation-value models. Purged/embargoed temporal validation (see Validation Framework).

---

## Exit-Cost Model Specification

### The Cold-Start Problem

The exit-cost model trains on actual fill telemetry from Tradier. But V2.1 execution modes (Harvest/Defensive/Emergency) don't exist yet, so we have no fills segmented by urgency level.

### Bootstrap Strategy

**Phase A — Heuristic model (ships with hard rules, before ML):**

```python
def heuristic_slippage(net_width, urgency_mode, minutes_to_close):
    """Estimated slippage from quote mid to actual fill."""
    base = net_width * 0.3  # assume 30% of spread as baseline slippage

    urgency_mult = {
        'HARVEST': 0.7,     # patient, get better fills
        'DEFENSIVE': 1.0,   # baseline
        'EMERGENCY': 1.8,   # aggressive pricing, worse fills
    }[urgency_mode]

    # Slippage increases in final hour as liquidity thins
    time_mult = 1.0 + max(0, (60 - minutes_to_close) / 60) * 0.5

    return base * urgency_mult * time_mult
```

Calibrate the 0.3 base factor from existing V1.2 fill logs (compare fill price vs. quote mid at order time for historical closes).

**Phase B — Data collection (runs for 2-4 weeks after execution modes ship):**

Every close attempt logs: quote mid, executable proxy, order price, fill price, time-to-fill, urgency mode, width at submission. This accumulates the training set.

**Phase C — Trained model (replaces heuristic):**

LightGBM regressor predicting `fill_debit - quote_mid` given:
- net width at submission
- urgency mode (one-hot)
- minutes to close
- number of replace attempts
- VIX level
- underlying recent volatility

Retrain monthly as more fill data accumulates.

---

## Live Data Requirements

The system must distinguish between:

- **critical-path live data**: required for real-time decisions
- **research-grade historical data**: required for training, replay, and validation

### Critical-Path Live Data

#### 1. SPX / SPY 1-Minute Underlying Bars

**Needed for**

- realized vol
- ATR / range expansion
- VWAP offset
- trend / momentum features
- break detection
- time-bucket context

**Possible live sources**

- Tradier underlying quote / timesales if available
- Schwab quote endpoints
- Polygon / ThetaData if already licensed
- Internal 1-minute bar builder from streaming quotes

**Live computation**

- maintain rolling windows in memory
- update on each new minute
- no historical reload required intraday beyond warm start

#### 2. Full 0DTE SPX/SPXW Option Chain With Quotes + Greeks

**Needed for**

- debit-to-close
- short-strike delta
- short-strike gamma
- per-side spread width
- quote stability
- closeable size
- approximate intraday skew / shape

**Possible live sources**

- Schwab option chain
- Tradier option chain
- ThetaData if available

**Must have fields**

- bid / ask
- bid size / ask size if possible
- delta
- gamma
- IV
- symbol / strike / expiration / type

#### 3. Open Interest

**Needed for**

- approximate OI-weighted surface context
- rough GEX / flip-distance estimation

**Reality**

OI is usually stale intraday. It is still useful as a slow-moving contextual feature, but should not be treated as precise live positioning.

**Possible sources**

- historical daily OI from data vendor
- chain endpoint if broker includes OI
- Cboe historical datasets for research

#### 4. VIX / VIX1D / Vol Context

**Needed for**

- intraday stress regime
- vol spread changes
- end-of-day gamma/tail urgency scaling

**Possible live sources**

- broker quote endpoints if supported
- vendor feed
- daily fallback if live VIX1D is unavailable

#### 5. Order / Fill Telemetry

**Needed for**

- build slippage model
- calibrate exit-cost expectations
- compare quote mid vs. executed fill

**Source**

- your own broker order history and local logs

This is one of the most important datasets in the entire system.

### Research-Grade Historical Data

#### What we already have (confirmed in data audit)

- 3 years of 1-minute SPX bars (`spx_1min.parquet`)
- 3 years of 1-minute VIX bars (`vix_1min.parquet`)
- VIX1D 1-minute bars from April 2023 (`vix1d_1min.parquet`)
- Full 0DTE option chain with greeks every minute (`spxw_0dte_intraday_greeks.parquet`)
- Daily OI by strike (`spxw_0dte_oi_daily_v1_2.parquet`)
- Historical IC trajectory simulation from target builder
- 250+ entry features already engineered

#### What we need to build (not collect)

- Per-minute feature reconstruction for historical trajectories (compute, don't collect)
- Continuation-value labels at 4 horizons (compute from existing quote data)
- Hazard labels (compute from existing quote data)
- Exit-cost training data (accumulate from live telemetry — new data, not historical)

#### Best-in-class research add-ons (future)

- Cboe DataShop option quote intervals with Greeks and OI
- Cboe Open-Close / capacity summaries for capacity-aware context
- trade-by-trade or higher-resolution quote data if budget later allows

### Data Strategy

Use a **three-tier data plan**:

1. **Immediate**
   Broker chain + underlying bars + your own execution logs + existing ThetaData historical files

2. **Research upgrade**
   Add Cboe or vendor historical minute option quotes with Greeks and OI for proper replay (only if existing ThetaData data proves insufficient)

3. **Institutional upgrade**
   Add richer microstructure or trade-capacity datasets only if V2.1 proves edge and scaling justifies cost

---

## Feature Framework

All critical-path features must be computable from live rolling state in memory. No expensive intraday recomputation from scratch.

### Feature Group A: Position Threat Features

These are the highest-priority live features.

| Feature | Description | Live Computation |
|--------|-------------|------------------|
| `short_call_delta_abs` | absolute delta of short call | from live chain |
| `short_put_delta_abs` | absolute delta of short put | from live chain |
| `short_call_gamma` | short call gamma | from live chain |
| `short_put_gamma` | short put gamma | from live chain |
| `call_distance_pts` | short call strike minus spot | strike - spot |
| `put_distance_pts` | spot minus short put strike | spot - strike |
| `call_distance_atr` | call cushion in ATR units | `call_distance_pts / rolling_atr` |
| `put_distance_atr` | put cushion in ATR units | `put_distance_pts / rolling_atr` |
| `threatened_side` | call / put / neither | side with lower normalized cushion |
| `delta_velocity` | change in short-strike delta over last N minutes | rolling diff |
| `gamma_velocity` | change in short-strike gamma over last N minutes | rolling diff |

**Why it matters**

These features measure whether a side is becoming dangerous before the full debit expansion shows up.

### Feature Group B: Price / Vol State Features

| Feature | Description | Live Computation |
|--------|-------------|------------------|
| `ret_1m` | 1-minute return | from bar cache |
| `ret_3m` | 3-minute return | rolling |
| `ret_5m` | 5-minute return | rolling |
| `rv_5m` | realized vol over 5 minutes | rolling std of returns |
| `rv_15m` | realized vol over 15 minutes | rolling std |
| `rv_burst_ratio` | short RV / longer RV | `rv_5m / rv_30m` |
| `rolling_atr_5m` | intraday ATR | rolling high-low range |
| `session_atr_like` | session average range proxy | running average |
| `atr_ratio` | current ATR vs session ATR | `rolling_atr_5m / session_atr_like` |
| `vwap_offset_pct` | distance from VWAP | `(spot - vwap)/vwap` |
| `range_exhaustion` | current session range vs baseline | current range / trailing reference |
| `break_score` | structural break / acceleration score | CUSUM or simplified burst metric |

### Feature Group C: Liquidity / Execution Features

These are missing from the original V2 and are mandatory.

| Feature | Description | Live Computation |
|--------|-------------|------------------|
| `net_close_width` | net spread width to close condor | close ask - close bid approximation |
| `net_close_mid` | approximate net close mid | from leg mids |
| `net_close_ask` | executable net close ask proxy | shorts at ask, longs at bid |
| `width_pct_mid` | width normalized by mid | `width / max(mid, eps)` |
| `short_leg_width_max` | max width of any short leg | max per-side width |
| `quote_refresh_gap_sec` | seconds since last quote update | local clock delta |
| `size_imbalance_proxy` | bid/ask size imbalance if available | from sizes |
| `estimated_slippage` | expected fill penalty | heuristic model (Phase A) or trained model (Phase C) |

**Why it matters**

A great risk signal with bad exit liquidity is not a great signal. It changes the optimal action.

### Feature Group D: Time State Features

| Feature | Description | Live Computation |
|--------|-------------|------------------|
| `minutes_since_entry` | age of trade | clock |
| `minutes_to_close` | time until forced flatten | clock |
| `final_hour_flag` | last 60 minutes | clock |
| `final_30m_flag` | last 30 minutes | clock |
| `final_15m_flag` | last 15 minutes (gamma crunch zone) | clock |
| `time_bucket` | early / midday / late / danger | categorical |

### Feature Group E: Portfolio Features

| Feature | Description | Live Computation |
|--------|-------------|------------------|
| `account_open_positions` | number of open ICs | live account state |
| `same_side_overlap_score` | clustered tail exposure across positions | compare threatened sides + strike proximity |
| `combined_unrealized_pct` | account unrealized P&L vs portfolio value | live |
| `daily_loss_remaining` | room until kill switch | live |
| `gamma_concentration_score` | exposure concentration proxy | sum of threatened-side gamma across positions |

### Feature Group F: Context Features

Use these as weak context, not hard truth.

| Feature | Description | Live Computation |
|--------|-------------|------------------|
| `approx_net_gex_sign` | sign of rough OI-weighted gamma exposure | daily OI + live gamma |
| `dist_to_gex_flip_proxy` | spot distance to estimated gamma-flip level | chain aggregation |
| `vix_level` | live or delayed VIX level | quote feed |
| `vix1d_spread` | VIX minus VIX1D or equivalent stress spread | quote feed |
| `iv_skew_proxy` | difference between put/call wing IVs | live chain |

### What Not To Put In The Critical Path

- full-blown HMM retraining intraday
- expensive OI-heavy recalculation every few seconds
- features requiring unavailable trade-capacity data
- anything that turns one monitoring cycle into a latency risk

If HMMs are used at all, they should be pre-trained offline and served as a lightweight inference component, not retrained intraday.

---

## Live Computation Design

### Monitoring Frequency

Use **variable-frequency monitoring**, not fixed-frequency monitoring.

#### Normal mode

- every 30-60 seconds

#### Elevated-risk mode

Switch to every 10-15 seconds when any of the following occurs:

- short-strike delta above warning threshold
- delta velocity spike
- ATR burst ratio above threshold
- liquidity width deteriorates sharply
- final 45-60 minutes of session
- any position in TIGHTEN state

#### Emergency mode

Immediate re-check loop while executing emergency exit until position is confirmed closed or replaced.

### In-Memory Rolling State

Maintain:

- bar cache for the current session
- per-position quote cache
- per-position feature cache
- rolling statistics objects
- last-known execution state

No feature should require a cold scan of historical files during live management.

### Practical Calculation Notes

#### Debit-to-close

Keep the current executable approximation:

- buy shorts at ask
- sell longs at bid

But also track:

- mid-based estimate
- width
- realized fill slippage versus both

#### ATR proxy

For live use, a robust intraday proxy is sufficient:

```python
rolling_atr = mean(high_low_range over last 5-10 bars)
session_atr_like = running_mean(high_low_range since open)
atr_ratio = rolling_atr / max(session_atr_like, eps)
```

#### Delta / gamma velocity

```python
delta_velocity = short_delta_now - short_delta_3m_ago
gamma_velocity = short_gamma_now - short_gamma_3m_ago
```

#### Threatened side

```python
call_threat = call_distance_atr / max(short_call_delta_abs, eps)
put_threat  = put_distance_atr  / max(short_put_delta_abs, eps)
threatened_side = "call" if call_threat < put_threat else "put"
```

This is simple, interpretable, and live-computable.

---

## Hard Risk Rules

These should ship before any model.

### Rule 1: Delta Emergency Exit

Trigger emergency logic when:

- short call delta exceeds threshold, or
- short put delta exceeds threshold

Two-stage approach:

- warning threshold: `0.18-0.22` (promotes to TIGHTEN)
- emergency threshold: `0.25-0.30` (triggers EMERGENCY EXIT)

Thresholds must be tuned by replay, not guessed.

**Acceptance criteria:** Expected fire rate 2-5% of all position-minutes at warning level, <1% at emergency level. If either fires >3x the expected rate over a 2-week window, disable pending investigation.

### Rule 2: Gamma Acceleration Exit

Trigger when:

- short-side gamma crosses a danger threshold, or
- gamma velocity spikes while cushion is collapsing

This catches the non-linear acceleration phase that pure debit logic often sees too late.

**Acceptance criteria:** Should fire on <2% of position-minutes. Primarily fires in final 90 minutes when gamma is highest.

### Rule 3: Cushion Collapse

Trigger when:

- distance to short strike in ATR units drops below threshold

This is often easier to stabilize than raw gamma.

**Acceptance criteria:** Expected overlap with Rule 1 (delta breach) of ~60-70%. The value is catching cases where ATR is expanding (cushion shrinks in vol-adjusted terms) before delta fully reflects it.

### Rule 4: Liquidity Protection

If spread width or quote instability becomes extreme:

- switch to higher urgency
- reduce confidence in model value estimates
- do not wait for delicate profit targets

**Acceptance criteria:** Should fire rarely (<1% of cycles). Primarily during market stress or data feed issues.

### Rule 5: Time-Based Gamma Clamp

From the final 45-60 minutes:

- tighten all profit-taking logic (lower TP target by 50%)
- lower tolerance for side threat (warning thresholds drop by 20%)
- increase monitoring frequency to elevated mode

**Acceptance criteria:** Always active in the time window. Measure impact on final-hour P&L distribution versus baseline.

### Rule 6: Portfolio Kill Switches

- daily realized loss cap
- account combined loss cap
- max number of emergency exits per day (3 emergency exits in one day = halt new entries)
- regime-dislocation circuit breaker (if data quality degrades, halt all activity)

### Rule Rollback Protocol

Every hard rule ships with a feature flag and a monitoring dashboard. For each rule:

1. Track fire rate per day and per week
2. Track outcome of rule fires (was the exit actually beneficial vs. holding?)
3. If fire rate exceeds 3x expected rate for 5+ consecutive trading days, auto-disable the rule and alert
4. If <40% of rule fires result in better outcomes than holding (measured on a 20-day trailing window), flag for threshold re-tuning
5. Manual override always available

---

## Multi-Position Interaction Rules

Each position is managed **independently** with these shared constraints:

1. **Shared daily loss budget.** All positions draw from the same daily realized loss cap. When one position takes a large loss, the remaining budget shrinks for all others — making kill switches fire sooner.

2. **Correlated threat escalation.** If 2+ positions are simultaneously in TIGHTEN or worse on the **same side** (e.g., both threatened on the put side), escalate all of them to the next urgency level. Correlated tail exposure is the most dangerous scenario.

3. **Emergency cascade.** If any position triggers EMERGENCY EXIT, all other positions on the same threatened side immediately promote to elevated monitoring frequency.

4. **Entry suppression.** While any position is in EMERGENCY EXIT state, no new entries are allowed until the emergency is resolved and a 5-minute cooldown has passed.

5. **Concentration guard.** If `same_side_overlap_score` (Feature Group E) exceeds a threshold, new entries on that side are blocked regardless of entry model scores.

---

## Modeling Framework

### Model A: Continuation-Value Models (4 Horizons)

**Purpose**

Estimate expected terminal P&L if we hold with no intervention from the current state to each of four time horizons.

**Targets**

Four regression targets per training row:

- `cv_3pm`: P&L if hold to 3:00 PM ET (no SL, no TP, no SLTP)
- `cv_330pm`: P&L if hold to 3:30 PM ET
- `cv_345pm`: P&L if hold to 3:45 PM ET
- `cv_4pm`: P&L if hold to 4:00 PM ET (actual expiration)

Labels are null when the row's timestamp is past the target horizon.

**Winsorization**

| Horizon | Floor | Ceiling |
|---------|-------|---------|
| cv_3pm | -2x credit | +1x credit |
| cv_330pm | -2x credit | +1x credit |
| cv_345pm | -2.5x credit | +1x credit |
| cv_4pm | -3x credit | +1x credit |

**Algorithm**

Four separate LightGBM regressors. Start with default hyperparameters, tune on validation.

- Later horizons (3:45, 4:00) may benefit from stronger regularization (`min_child_samples`, `lambda_l2`) due to higher target variance.
- Use MAE or Huber loss (not MSE) to reduce sensitivity to extreme labels even after winsorization.

**Expected Performance**

- cv_3pm: highest R², lowest variance — most actionable
- cv_330pm: moderate R²
- cv_345pm: lower R², but captures the gamma inflection
- cv_4pm: lowest R², highest variance — useful as a directional signal, not a precise estimate

If cv_4pm R² < 0.05 on validation, consider dropping it and using the 3-point curve (3pm, 3:30, 3:45) only.

### Model B: Tail-Risk / Hazard Models

**Purpose**

Estimate short-horizon probabilities of severe adverse moves.

**Targets**

- `hazard_1m`: binary, >20% of remaining credit lost in next 1 minute
- `hazard_3m`: binary, >20% of remaining credit lost in next 3 minutes
- `hazard_5m`: binary, >15% of remaining credit lost in next 5 minutes

**Algorithm**

Three separate LightGBM binary classifiers.

- `scale_pos_weight` = n_negative / n_positive (expected ~20-50x)
- Optimize for **PR-AUC**, not ROC-AUC
- Minimum operating recall: 70%
- Post-training calibration via Platt scaling or isotonic regression
- Threshold selection: maximize precision subject to recall >= 70% on validation set

**Why three horizons:** 1-minute hazard is the "get out NOW" signal. 5-minute hazard is the "this is about to get bad" signal. 3-minute fills the gap. The policy engine can weight them differently based on current execution state (if exit takes 2 minutes to fill, the 3-minute hazard is more relevant than the 1-minute).

### Model C: Exit-Cost Model

**Purpose**

Estimate expected slippage and fill cost if we try to exit now.

**Target**

- `slippage = actual_fill_debit - quote_mid_at_submission`

**Bootstrap path** (see Exit-Cost Model Specification section above):

- Phase A: Heuristic model (ships immediately)
- Phase B: Data collection (2-4 weeks of fills across urgency modes)
- Phase C: Trained LightGBM regressor (replaces heuristic)

**Calibration dependency:** The heuristic model's base slippage factor (initially 30% of spread width) should be calibrated against existing V1.2 fill logs. Once Phase 1 telemetry is running, recalibrate weekly using actual fill-vs-quote data.

### Why This Stack Is Better Than A Single Classifier

A single `should_exit_now` classifier mixes:

- value estimation
- risk estimation
- execution estimation

Those are different problems and should not be forced into one number if we can separate them cleanly.

---

## Policy Engine

### Decision Score

At each cycle:

1. Estimate continuation value at 4 horizons
2. Estimate tail risk at 3 horizons
3. Estimate exit cost
4. Estimate model uncertainty
5. Choose action under hard constraints

### CV Term Structure Analysis

```python
cv_3   = model_3pm.predict(features)
cv_330 = model_330pm.predict(features)
cv_345 = model_345pm.predict(features)
cv_4   = model_4pm.predict(features)

# Weight later horizons with lower confidence
weighted_cv = (
    cv_3   * 0.35 +
    cv_330 * 0.30 +
    cv_345 * 0.20 +
    cv_4   * 0.15
)

# Term structure slope (P&L trend across horizons)
cv_slope_early = (cv_330 - cv_3) / 30    # per minute, 3:00-3:30
cv_slope_late  = (cv_4 - cv_345) / 15    # per minute, 3:45-4:00

# Deterioration signal: positive near-term, negative far-term
cv_divergence = cv_slope_early - cv_slope_late
```

### Policy Logic

```python
# Layer 2: Hard rules always first
if hard_risk_rule_triggered:
    return EMERGENCY_EXIT

# Layer 6: Uncertainty gate
if model_uncertainty_high:
    return SIMPLE_RULE_FALLBACK  # use only hard rules + current baseline logic

# Layer 3: Model-informed decisions
net_hold_value = weighted_cv
net_exit_value = current_pnl - expected_exit_cost

# Tail risk check
max_hazard = max(hazard_1m, hazard_3m * 0.8, hazard_5m * 0.6)
if max_hazard > tail_risk_limit:  # tuned on validation, start at 0.4
    if exit_liquidity_ok:
        return FULL_EXIT
    return EMERGENCY_EXIT

# Value comparison
if net_exit_value > net_hold_value + policy_buffer:  # buffer prevents churn
    return FULL_EXIT

# Threat detection
if threatened_side != "neither" and delta_velocity > 0:
    if cv_slope_late < 0:  # getting worse into the close
        return TIGHTEN

return HOLD
```

### Uncertainty Gate

Concrete approach using **LightGBM prediction variance**:

```python
def estimate_uncertainty(model, features):
    """Use variance across individual trees as confidence proxy."""
    leaf_predictions = []
    for tree_idx in range(model.num_trees()):
        leaf_predictions.append(model.predict(features, start_iteration=tree_idx, num_iteration=1))
    return np.std(leaf_predictions)

# Track rolling calibration error
calibration_error = rolling_mean(abs(predicted_cv - actual_cv), window=100_trades)

# Uncertainty is high when:
# 1. Tree disagreement is large (OOD or ambiguous state)
# 2. Rolling calibration has degraded
uncertainty_score = tree_std / historical_median_std + calibration_error / baseline_calibration_error

if uncertainty_score > 2.0:
    model_uncertainty_high = True  # fall back to rules-only
```

When uncertainty is high:

- reduce model authority
- increase hard-rule authority
- simplify to conservative logic (existing V1.2 baseline behavior)

---

## Execution Engine

This is a first-class part of V2.1.

### Exit Modes

#### 1. Harvest Mode

For favorable exits (profit-taking):

- use patient limit pricing (start at mid, improve slowly)
- up to 3 reprices at 30-second intervals
- max wait time: 90 seconds
- if not filled, escalate to Defensive

#### 2. Defensive Mode

For normal stop / elevated risk:

- marketable-limit (mid + 30% of width toward executable side)
- cancel/replace every 15 seconds, improving price 10% of width each time
- max wait time: 60 seconds
- if not filled, escalate to Emergency

#### 3. Emergency Mode

For tail events:

- aggressive marketable-limit (executable proxy or worse)
- cancel/replace every 5 seconds
- immediate recheck until flat
- after 3 failed attempts, use market order if broker supports

### Data To Log On Every Exit

- timestamp
- quoted net mid
- quoted executable proxy
- order price
- fill price
- time-to-fill
- quote width at submission
- urgency mode
- reason for exit
- number of cancel/replace cycles
- final slippage vs mid
- final slippage vs executable proxy

This data becomes the training set for the exit-cost model.

---

## Validation Framework

### Required Validation Standards

1. **Temporal walk-forward**
   Standard forward-only testing. Train on months 1-N, validate on month N+1.

2. **Purged / embargoed validation**
   Remove leakage from overlapping labels and neighboring observations.
   - **Purge**: remove all training rows within 60 minutes of any validation row (same day, same position)
   - **Embargo**: remove 1 full trading day buffer between train and validation periods
   This is critical because consecutive minutes of the same trade share most features.

3. **Multiple-path robustness**
   Use combinatorial-style temporal splits where feasible to reduce path luck. Minimum 5 temporal folds.

4. **Execution-aware replay**
   Evaluate with realistic slippage / width assumptions, not idealized mid fills.
   - **Optimistic tier**: mid-price fills (lower bound on slippage)
   - **Realistic tier**: calibrated from Phase 1 telemetry fill-vs-quote ratios (primary evaluation tier)
   - **Stressed tier**: 2x realistic slippage (robustness check)

   Calibration dependency: the Realistic tier starts as a heuristic (30% of width) and is recalibrated once 2+ weeks of Phase 1 fill telemetry is available.

5. **Shadow live deployment**
   Run the new engine in paper/shadow mode alongside current production before promotion.

### Metrics That Matter

Primary:

- max drawdown
- worst trade loss
- CVaR / left-tail metrics
- realized slippage
- Sharpe / Sortino after execution assumptions
- percentage of days hitting kill switches

Secondary:

- raw return
- win rate
- AUC / PR-AUC
- precision / recall at operating threshold

### Promotion Ladder

#### Stage 1: Research greenlight

- replay beats baseline on tail risk (max DD, worst trade) and maintains acceptable return
- all models pass calibration checks on holdout

#### Stage 2: Shadow greenlight

- live shadow decisions improve simulated outcomes vs current live policy
- minimum 10 trading days of shadow data
- stale-data incidents < 5% of cycles

#### Stage 3: Limited capital rollout

- one Tradier account, low size, hard fallback enabled
- minimum 10 trading days before expansion

#### Stage 4: General rollout

- only after stable calibration, execution behavior, and acceptable returns across Stage 3

---

## Implementation Roadmap

### Phase 0: Lock Scope And Control Path

**Goal**

Create the telemetry foundation for V2.1.

**Build**

- Freeze entry behavior around the current `v1_2` flow:
  - no changes to `v1_2` model selection
  - no changes to entry thresholds
  - no changes to scan cadence unless required by exit integration
- Standardize Tradier-only assumptions for V2.1 rollout
- Pick the primary live control path:
  - Build exit V2.1 on the orchestrator path, not the legacy monolithic bot
  - Use `PositionManager` + `ExecutionRouter` as the core management loop
- Define initial production action set: HOLD, TIGHTEN, FULL_EXIT, EMERGENCY_EXIT
- Explicitly defer: partial reduction, side-cutting, Schwab parity

**Exit criteria**

- One execution path is selected and documented
- No ambiguity remains about entry vs. exit ownership

### Phase 1: Telemetry Foundation

**Goal**

Capture all live state needed for research, debugging, and slippage modeling.

**Build**

1. Per-monitor-cycle structured telemetry for each open position:
   - timestamp, account name, position id
   - underlying spot
   - net close mid proxy, executable close proxy, net width
   - short call delta, short put delta
   - short call gamma, short put gamma
   - time since entry, time to forced close
   - unrealized P&L
   - active hard-rule flags
   - chosen monitoring mode

2. Per-order telemetry for every close attempt:
   - quote snapshot at order submit
   - urgency mode
   - submitted order price
   - fill price, time to fill
   - replace count
   - final slippage versus mid
   - final slippage versus executable proxy
   - exit reason

3. Persist telemetry in structured format:
   - JSONL files in `logs/v2_exit/`
   - daily roll files
   - daily merge / normalization script

**Files to create or edit**

- `scripts/orchestrator/position_manager.py`
- `scripts/orchestrator/execution_router.py`
- `ml/zdom_v2/1_data_collection/scripts/merge_execution_logs.py`

**Exit criteria**

- Every monitor cycle produces structured telemetry
- Every close attempt produces structured execution telemetry
- Daily outputs are joinable and replayable

### Phase 2: Tradier Live Data Layer

**Goal**

Build a clean Tradier-first market-state adapter for low-latency live feature computation.

**Build**

1. Confirm and document Tradier live fields:
   - underlying quote / timesales endpoints
   - option chain (bid/ask, bid/ask size, delta, gamma, IV, timestamps)
   - order status / fill endpoints

2. Build live data adapter exposing:
   - current underlying state
   - current full 0DTE option chain
   - latest per-leg Greeks
   - quote freshness metadata

3. Maintain in-memory rolling state:
   - 1-minute bars (from quotes)
   - current chain snapshot
   - per-position quote snapshots
   - rolling return / vol windows

4. Add stale-data guards:
   - if chain is stale (>90 seconds), suppress model authority
   - if underlying data is stale, force conservative fallback logic
   - log all stale-data incidents

**Files to create or edit**

- `execution/broker_api/tradier_client.py`
- `scripts/orchestrator/broker_manager.py`
- `ml/zdom_v2/2_feature_engineering/scripts/realtime_exit_features.py`

**Exit criteria**

- Live Tradier data can support all critical-path features
- Stale data is detected automatically

### Phase 3: Live Feature Engine

**Goal**

Compute the first high-value feature set directly from Tradier live data.

**Build**

1. Single feature-builder module for V2.1 exits.

2. First production feature set (Groups A-D):
   - Position threat: delta, gamma, distance, velocity, threatened side
   - Price/vol: returns, RV, ATR ratio, VWAP, break score
   - Liquidity: width, mid, executable proxy, quote freshness
   - Time: minutes since entry, minutes to close, time bucket flags

3. Weak-context features (Group F) added after core stability:
   - approximate GEX sign, GEX flip proxy
   - VIX / VIX1D context
   - IV skew proxy

4. Unit tests for feature correctness, missing data handling, stale data handling.

5. Ensure same logic can be reused by historical replay (Phase 7).

**Files to create or edit**

- `ml/zdom_v2/2_feature_engineering/scripts/realtime_exit_features.py`
- `ml/zdom_v2/2_feature_engineering/scripts/position_threat_features.py`
- `ml/zdom_v2/2_feature_engineering/scripts/liquidity_features.py`
- `tests/`

**Exit criteria**

- Live feature builder works on Tradier data
- Same logic reusable by replay engine

### Phase 4: Hard Risk Rules

**Goal**

Ship the highest-value risk improvements before ML. Reduce tail losses immediately.

**Build**

1. Delta-based warning (0.18-0.22 → TIGHTEN) and emergency (0.25-0.30 → EMERGENCY EXIT) thresholds
2. Gamma-acceleration guard
3. Cushion-collapse rule (distance-to-short in ATR units below threshold)
4. Liquidity-protection rule (extreme width → defensive/emergency logic)
5. Final-hour gamma clamp (tighter TP, lower threat tolerance, faster monitoring)
6. Stricter kill switches (daily loss cap, combined stop, max 3 emergency exits/day, data-quality circuit breaker)
7. Feature flags for all rules
8. Monitoring dashboard for fire rates and outcomes per rule
9. Rollback protocol (auto-disable if fire rate >3x expected for 5+ days)

**Files to create or edit**

- `scripts/orchestrator/position_manager.py`
- `ml/zdom_v2/5_execution/scripts/hard_risk_rules.py`

**Exit criteria**

- Hard rules can run in paper mode with no model dependency
- Emergency exits fire deterministically under known thresholds
- Fire rate monitoring is active

### Phase 5: Variable-Frequency Monitoring

**Goal**

Stop treating all market states as equally urgent.

**Build**

1. Monitoring modes:
   - normal: every 30-60 seconds
   - elevated: every 10-15 seconds
   - emergency: immediate re-check until flat or stabilized

2. Promote to elevated when:
   - delta warning threshold breached
   - delta velocity spike
   - ATR burst ratio spike
   - width deteriorates sharply
   - final 45-60 minutes
   - any position in TIGHTEN state

3. State transitions and throttling to respect Tradier rate limits.

**Files to create or edit**

- `scripts/orchestrator/orchestrator.py`
- `scripts/orchestrator/position_manager.py`

**Exit criteria**

- Bot shifts monitoring frequency based on risk state
- Tradier rate limits are not violated

### Phase 6: Exit Execution Modes

**Goal**

Make exit execution an explicit part of the risk engine.

**Build**

1. Three Tradier close modes: HARVEST, DEFENSIVE, EMERGENCY (see Execution Engine section for pricing/timing specs)

2. Handling for: working close orders, partial fills, repeated replace attempts, forced escalation

3. Centralized executable close pricing (one shared calculation path)

4. Fill-quality telemetry captured automatically

**Exit-cost bootstrap**: After this phase ships, the heuristic slippage model goes live. Run for 2-4 weeks to accumulate fills across urgency modes before training Model C.

**Files to create or edit**

- `scripts/orchestrator/execution_router.py`
- `execution/order_manager/ic_order_builder.py`
- `ml/zdom_v2/5_execution/scripts/execution_modes.py`

**Exit criteria**

- Exits run through explicit urgency modes
- Fill-quality telemetry is captured automatically
- Heuristic slippage model is active

### Phase 7: Historical Replay Engine

**Goal**

Build an honest offline environment for evaluating exit policies.

**Build**

1. Per-minute position trajectory reconstruction from existing ThetaData historical data

2. Reconstruct: underlying bars, option quotes, Greeks, OI (where available)

3. Reuse live feature engine inside replay

4. Compute continuation-value labels at 4 horizons (3:00, 3:30, 3:45, 4:00 PM) using no-intervention counterfactual:
   ```python
   for each historical trade:
       for each minute t in trajectory:
           features = build_v2_features(t, ...)
           cv_3pm   = (credit - debit_at_3pm)   * qty * 100  if t < 3:00 PM else null
           cv_330pm = (credit - debit_at_330pm) * qty * 100  if t < 3:30 PM else null
           cv_345pm = (credit - debit_at_345pm) * qty * 100  if t < 3:45 PM else null
           cv_4pm   = (credit - debit_at_4pm)   * qty * 100  if t < 4:00 PM else null
           hazard_1m = (debit_t+1 - debit_t) > 0.20 * remaining_credit
           hazard_3m = (debit_t+3 - debit_t) > 0.20 * remaining_credit
           hazard_5m = (debit_t+5 - debit_t) > 0.15 * remaining_credit
   ```

5. Support three execution assumption tiers:
   - optimistic: mid-price fills
   - realistic: calibrated from Phase 1 telemetry (heuristic until data available)
   - stressed: 2x realistic slippage

6. Support side-by-side evaluation: current baseline vs. hard-rules-only vs. V2.1 policy

**Files to create or edit**

- `ml/zdom_v2/1_data_collection/scripts/build_position_trajectories.py`
- `ml/zdom_v2/1_data_collection/scripts/build_exit_replay_dataset.py`
- `ml/zdom_v2/6_analysis/scripts/replay_v2_policy.py`

**Exit criteria**

- One command replays historical positions with live-equivalent features
- Labels at 4 CV horizons + 3 hazard horizons are generated
- Slippage tiers are configurable

### Phase 8: Model Training

**Goal**

Train the full model stack on replay data.

**Build**

1. **Continuation-value models** (4 separate LightGBM regressors):
   - Targets: cv_3pm, cv_330pm, cv_345pm, cv_4pm (winsorized per spec)
   - Loss: Huber loss
   - Later horizons use stronger regularization
   - Drop cv_4pm model if R² < 0.05 on validation

2. **Hazard models** (3 separate LightGBM classifiers):
   - Targets: hazard_1m, hazard_3m, hazard_5m
   - `scale_pos_weight` = n_neg / n_pos
   - Optimize for PR-AUC
   - Minimum recall: 70% at operating threshold
   - Post-training calibration (Platt scaling)

3. **Exit-cost model** (1 LightGBM regressor):
   - Only trainable after 2-4 weeks of Phase 6 fill telemetry
   - Until then, heuristic model remains in production
   - Target: fill_debit - quote_mid
   - Features: width, urgency mode, minutes to close, VIX, recent RV

4. Validation:
   - Purged / embargoed temporal splits (60-min purge, 1-day embargo)
   - Minimum 5 temporal folds
   - Reject models that improve return only by increasing left-tail risk
   - Calibration curves for all hazard models

**Files to create or edit**

- `ml/zdom_v2/4_training/scripts/train_continuation_value.py`
- `ml/zdom_v2/4_training/scripts/train_hazard_models.py`
- `ml/zdom_v2/4_training/scripts/train_exit_cost_model.py`
- `ml/zdom_v2/4_training/scripts/evaluate_policy_v2.py`

**Exit criteria**

- Models beat baseline on replay under realistic execution assumptions
- All models pass calibration checks
- CV term structure provides actionable signal (slope/curvature distinguish hold vs. exit states)

### Phase 9: Policy Engine

**Goal**

Combine rules, model outputs, and execution costs into a single action decision.

**Build**

1. Policy engine consuming: hard-rule signals, CV at 4 horizons, hazard at 3 horizons, expected exit cost, uncertainty state

2. CV term structure analysis: weighted CV, slope (early vs. late), divergence signal

3. Production policy (see Policy Engine section for full logic)

4. Uncertainty gate using tree prediction variance + rolling calibration error

5. All thresholds and buffers externally configurable (YAML or config module)

**Files to create or edit**

- `ml/zdom_v2/5_execution/scripts/policy_engine.py`
- `ml/zdom_v2/5_execution/scripts/uncertainty_gate.py`
- `scripts/orchestrator/position_manager.py`

**Exit criteria**

- One policy interface determines the action each monitoring cycle
- Uncertainty gate correctly falls back to rules-only under high uncertainty

### Phase 10: Shadow Mode

**Goal**

Validate V2.1 decisions live before capital deployment.

**Build**

1. Run V2.1 scoring live in shadow alongside current Tradier production exit logic

2. Log all disagreements: baseline decision vs. V2.1 recommendation

3. Daily review: missed saves, premature exits, slippage assumption accuracy, uncertainty fallback frequency, stale-data incidents

**Files to create or edit**

- `ml/zdom_v2/6_analysis/scripts/shadow_decision_audit.py`
- `scripts/orchestrator/position_manager.py`

**Exit criteria**

- Minimum 10 trading days of shadow data
- Enough evidence to judge promotability
- Stale-data incidents < 5% of cycles

### Phase 11: Controlled Tradier Rollout

**Goal**

Deploy safely with bounded capital risk.

**Build**

1. Rollout sequence:
   - Tradier paper account
   - One small live Tradier account
   - Broader Tradier deployment only after stable results

2. Permanent safety fallbacks:
   - Hard rules always active
   - Fallback to baseline exits if data is stale
   - Fallback if models fail to load
   - Fallback if uncertainty score > threshold
   - Fallback if repeated execution failures occur

3. Daily post-trade audit summaries

**Files to create or edit**

- `scripts/zdom_bot_tradier.py`
- `scripts/orchestrator/config.py`
- `scripts/orchestrator/orchestrator.py`

**Exit criteria**

- V2.1 can be toggled on per Tradier account with automatic fallback
- Minimum 10 trading days at each stage before expansion

### Phase 12: Operations And Governance

**Goal**

Keep the system observable, reviewable, and reversible.

**Build**

1. Daily reports: exit reasons, emergency exit count, avg slippage by mode, worst trade, calibration drift, stale-data incidents

2. Weekly model-health review

3. Version every live: model artifact, policy config, hard-rule config

4. Same-day rollback capability

**Exit criteria**

- The system can be monitored and rolled back safely

---

## Recommended First Build Order

1. Lock scope and control path (Phase 0)
2. Telemetry foundation (Phase 1)
3. Tradier live data layer (Phase 2)
4. Live feature engine (Phase 3)
5. Hard risk rules (Phase 4)
6. Variable-frequency monitoring (Phase 5)
7. Exit execution modes (Phase 6)
8. **[2-4 week pause: accumulate fill telemetry for exit-cost model]**
9. Historical replay engine (Phase 7)
10. Model training (Phase 8)
11. Policy engine (Phase 9)
12. Shadow mode (Phase 10)
13. Controlled Tradier rollout (Phase 11)
14. Operating cadence and governance (Phase 12)

### Earliest Practical Milestone

The first meaningful production milestone is Phases 0-6:

- structured telemetry
- live feature engine
- delta / gamma / cushion hard rules
- variable-frequency monitoring
- explicit Tradier emergency exit mode

That milestone should deliver immediate practical value even before any ML exit model is deployed.

### Second Milestone (ML)

Phases 7-9 complete the ML stack. This requires:
- Historical replay engine working
- 2-4 weeks of fill telemetry accumulated (for exit-cost model)
- All 4 CV models + 3 hazard models trained and validated

### Third Milestone (Production)

Phases 10-12 put it in production with shadow validation and governance.

---

## Directory Structure

```
ml/zdom_v2/
├── PLAN.md
├── 1_data_collection/
│   └── scripts/
│       ├── build_position_trajectories.py
│       ├── build_exit_replay_dataset.py
│       └── merge_execution_logs.py
├── 2_feature_engineering/
│   └── scripts/
│       ├── realtime_exit_features.py
│       ├── liquidity_features.py
│       ├── position_threat_features.py
│       └── regime_context_features.py
├── 3_models/
│   ├── cv_3pm_lgbm_v2.pkl
│   ├── cv_330pm_lgbm_v2.pkl
│   ├── cv_345pm_lgbm_v2.pkl
│   ├── cv_4pm_lgbm_v2.pkl
│   ├── hazard_1m_lgbm_v2.pkl
│   ├── hazard_3m_lgbm_v2.pkl
│   ├── hazard_5m_lgbm_v2.pkl
│   └── exit_cost_lgbm_v2.pkl
├── 4_training/
│   └── scripts/
│       ├── train_continuation_value.py
│       ├── train_hazard_models.py
│       ├── train_exit_cost_model.py
│       └── evaluate_policy_v2.py
├── 5_execution/
│   └── scripts/
│       ├── v2_monitor.py
│       ├── hard_risk_rules.py
│       ├── policy_engine.py
│       ├── execution_modes.py
│       └── uncertainty_gate.py
└── 6_analysis/
    └── scripts/
        ├── replay_v2_policy.py
        ├── compare_v1_2_vs_v2_1.py
        └── shadow_decision_audit.py
```

---

## What Not To Build Yet

- full RL / PPO policy learning
- sequence models (LSTM/transformer) before tree models have clearly failed
- dealer-position hard overrides from rough OI inference
- expensive order-flow feeds ($TICK, Level 2) before basic exit telemetry is fully exploited
- Option C multi-policy label blending (start with Option A, revisit if CV models plateau)
- REDUCE / SIDE-CUT actions (design supports them, defer implementation)

The next real edge here is far more likely to come from:

- better live threat measurement
- better execution
- better validation discipline

than from exotic model architecture.

---

## Success Criteria

V2.1 is a success if, versus the current live-exit baseline, it delivers:

1. materially lower worst-trade loss
2. materially lower drawdown
3. lower tail-risk concentration in the final hour
4. stable live execution quality
5. acceptable return retention after realistic slippage

Target directionally:

- reduce worst-trade loss by 25-40%
- reduce max drawdown by 20-35%
- reduce emergency late-session blowups meaningfully
- maintain similar trade frequency
- preserve enough total return to justify complexity

---

## Repo-Specific Implementation Checklist

### Assumptions

- `v1_2` remains the **entry model** for now
- Exit management is the only major strategy upgrade in scope
- Tradier is the only live broker target for V2.1
- Schwab support can be added later after the Tradier path is stable

### Current Code Paths In Scope

- `scripts/zdom_bot_tradier.py`
- `scripts/zero_dte_bot.py`
- `scripts/orchestrator/position_manager.py`
- `scripts/orchestrator/execution_router.py`
- `scripts/orchestrator/orchestrator.py`
- `execution/broker_api/tradier_client.py`
- `execution/order_manager/ic_order_builder.py`

---

## References And Design Inputs

These are guiding references, not claims of direct replication.

- Cboe research on 0DTE market impact and volatility
- Cboe 2025 options-industry state materials
- Almeida, Freire, Hizmeri on 0DTE asset-pricing characteristics
- Cartea and Wang on alpha-aware market making and adverse selection
- de Prado-style purged / embargoed validation and CPCV concepts
- adaptive conformal inference literature for uncertainty-aware online prediction

### Practical Interpretation Of The Research

The most important takeaway is not "dealer gamma explains everything."

It is:

- short-dated gamma can matter, especially in tails
- average effects are not the same as tail effects
- live execution and uncertainty management matter as much as raw predictive power

That is the design philosophy of V2.1.
