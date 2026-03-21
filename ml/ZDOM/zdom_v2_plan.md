# ZDOM V2.1: Institutional-Grade 0DTE Exit and Position Management

## Vision

V1/V1.2 optimized **entry selection**. Once a position is open, however, the live stack still behaves mostly like a debit-threshold monitor: fixed stop-loss logic, static SLTP ratchets, and account-level combined stops. That is a solid baseline, but it is not how a serious 0DTE risk engine should behave.

V2.1 upgrades exit management from a static threshold system into a **state-aware decision engine** that:

1. Estimates the **current market, vol, liquidity, and position state** every monitoring cycle.
2. Predicts **continuation value**, **short-horizon tail risk**, and **expected exit cost**.
3. Chooses among a richer set of actions than just hold vs. exit.
4. Falls back to hard risk rules whenever model confidence is weak or the regime is unfamiliar.
5. Treats **execution quality** as part of risk management, not an afterthought.

This is the design goal for an internal, institutional-quality 0DTE management stack. It is realistic, measurable, and buildable with the data and infrastructure we can assemble.

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

Three models, not one:

1. **Continuation-value model**
   Predict expected terminal P&L if we hold for the next decision horizon.

2. **Tail-risk / hazard model**
   Predict probability of severe adverse move, stop breach, or strike-touch over short horizons.

3. **Exit-cost model**
   Predict expected slippage / fill penalty conditional on current liquidity state and urgency.

### Layer 4: Policy Engine

Choose actions under constraints:

- maximize expected value
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

### Allowed Actions

1. **HOLD**
   Keep current risk posture.

2. **TIGHTEN**
   Move to tighter SLTP / tighter stop / higher urgency execution if exit triggers.

3. **REDUCE**
   Close part of size if multiple contracts are open.

4. **SIDE-CUT**
   Exit or reduce the threatened side if supported by execution logic and margin rules.

5. **FULL EXIT**
   Close the entire condor.

6. **EMERGENCY EXIT**
   Immediate urgency escalation with aggressive execution assumptions.

If side-cutting is too operationally risky at first, we should still design the research framework to support it later.

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

#### Minimum acceptable historical set

- 1-minute SPX or SPY bars
- 1-minute VIX and VIX1D if possible
- 1-minute SPXW option quotes with Greeks
- daily open interest
- your own historical fills and order outcomes

#### Best-in-class research add-ons

- Cboe DataShop option quote intervals with Greeks and OI
- Cboe Open-Close / capacity summaries for capacity-aware context
- trade-by-trade or higher-resolution quote data if budget later allows

### Data Strategy Recommendation

Use a **three-tier data plan**:

1. **Immediate**
   Broker chain + underlying bars + your own execution logs

2. **Research upgrade**
   Add Cboe or vendor historical minute option quotes with Greeks and OI for proper replay

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
| `estimated_slippage` | expected fill penalty | model or heuristic |

**Why it matters**

A great risk signal with bad exit liquidity is not a great signal. It changes the optimal action.

### Feature Group D: Time State Features

| Feature | Description | Live Computation |
|--------|-------------|------------------|
| `minutes_since_entry` | age of trade | clock |
| `minutes_to_close` | time until forced flatten | clock |
| `final_hour_flag` | last 60 minutes | clock |
| `final_30m_flag` | last 30 minutes | clock |
| `time_bucket` | early / midday / late | categorical |

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

- warning threshold: `0.18-0.22`
- emergency threshold: `0.25-0.30`

Thresholds must be tuned by replay, not guessed.

### Rule 2: Gamma Acceleration Exit

Trigger when:

- short-side gamma crosses a danger threshold, or
- gamma velocity spikes while cushion is collapsing

This catches the non-linear acceleration phase that pure debit logic often sees too late.

### Rule 3: Cushion Collapse

Trigger when:

- distance to short strike in ATR units drops below threshold

This is often easier to stabilize than raw gamma.

### Rule 4: Liquidity Protection

If spread width or quote instability becomes extreme:

- switch to higher urgency
- reduce confidence in model value estimates
- do not wait for delicate profit targets

### Rule 5: Time-Based Gamma Clamp

From the final 45-60 minutes:

- tighten all profit-taking logic
- lower tolerance for side threat
- increase monitoring frequency

### Rule 6: Portfolio Kill Switches

- daily realized loss cap
- account combined loss cap
- max number of emergency exits per day
- regime-dislocation circuit breaker

---

## Modeling Framework

### Model A: Continuation-Value Model

**Purpose**

Estimate expected terminal P&L if we continue holding from the current state, under the current policy horizon.

**Target**

Regression target:

- terminal P&L if hold from time `t` until:
  - next forced decision point,
  - stop/TP trigger,
  - forced close,
  - or chosen policy horizon

This is better than a single binary label because it preserves magnitude.

**Candidate algorithms**

- LightGBM regressor
- XGBoost regressor

Start simple.

### Model B: Tail-Risk / Hazard Model

**Purpose**

Estimate short-horizon probabilities such as:

- severe adverse move in next 1/3/5 minutes
- stop breach in next N minutes
- strike-touch probability in next N minutes

**Targets**

- binary hazard labels over multiple horizons

This should become the safety-biased model.

### Model C: Exit-Cost Model

**Purpose**

Estimate expected slippage and fill cost if we try to exit now with a given urgency.

**Target**

- actual fill debit minus quote-mid
- actual fill debit minus executable proxy

Model separately by:

- profit-taking exits
- normal-risk exits
- emergency exits

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

1. Estimate continuation value
2. Estimate tail risk
3. Estimate exit cost
4. Estimate uncertainty / out-of-distribution risk
5. Choose action under hard constraints

### Example Policy Logic

```python
if hard_risk_rule_triggered:
    return EMERGENCY_EXIT

if uncertainty_high:
    return SIMPLE_RULE_FALLBACK

net_hold_value = continuation_value
net_exit_value = current_pnl - expected_exit_cost

if tail_risk_prob > tail_risk_limit:
    if exit_liquidity_ok:
        return FULL_EXIT
    return EMERGENCY_EXIT

if net_exit_value > net_hold_value + policy_buffer:
    return FULL_EXIT

if threatened_side and threat_rising:
    return TIGHTEN

return HOLD
```

### Uncertainty Handling

Do not trust model outputs equally in all states.

Add:

- calibration tracking
- out-of-distribution distance checks
- conformal or adaptive error-band logic if feasible

When uncertainty is high:

- reduce model authority
- increase hard-rule authority
- simplify to conservative logic

---

## Execution Engine

This is a first-class part of V2.1.

### Exit Modes

#### 1. Harvest Mode

For favorable exits:

- use patient limit pricing
- small number of reprices
- low urgency

#### 2. Defensive Mode

For normal stop / elevated risk:

- marketable-limit
- tighter cancel/replace cadence
- cap waiting time

#### 3. Emergency Mode

For tail events:

- aggressive marketable-limit or market logic depending broker support and risk rules
- immediate recheck until flat

### Data To Log On Every Exit

- timestamp
- quoted net mid
- quoted executable proxy
- order price
- fill price
- time-to-fill
- quote width
- urgency mode
- reason for exit

This data becomes the training set for the exit-cost model.

---

## Validation Framework

### Required Validation Standards

1. **Temporal walk-forward**
   Standard forward-only testing.

2. **Purged / embargoed validation**
   Remove leakage from overlapping labels and neighboring observations.

3. **Multiple-path robustness**
   Use combinatorial-style temporal splits where feasible to reduce path luck.

4. **Execution-aware replay**
   Evaluate with realistic slippage / width assumptions, not idealized mid fills.

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
- AUC
- precision / recall

### Promotion Ladder

#### Stage 1: Research greenlight

- replay beats baseline on tail risk and maintains acceptable return

#### Stage 2: Shadow greenlight

- live shadow decisions improve simulated outcomes vs current live policy

#### Stage 3: Limited capital rollout

- one account, low size, hard fallback enabled

#### Stage 4: General rollout

- only after stable calibration and execution behavior

---

## Implementation Roadmap

### Phase 0: Instrument The Current Live System

**Goal**

Create the telemetry foundation for V2.1.

**Build**

- log every monitor cycle per position
- log per-leg and net quote widths
- log delta/gamma of shorts each cycle
- log exit reason and fill outcome
- log quote-to-fill slippage

**Impact**

This makes future modeling credible.

### Phase 1: Ship Hard Rules First

**Goal**

Capture the highest-value improvements without ML dependency.

**Build**

- warning + emergency delta thresholds
- gamma acceleration guard
- cushion-collapse rule
- variable-frequency monitoring
- final-hour gamma clamp
- stricter kill switches

**Expected result**

Lower tail losses and better emergency responsiveness.

### Phase 2: Build Historical Replay Engine

**Goal**

Reconstruct per-minute position state and simulate decisions honestly.

**Build**

- position trajectory builder
- per-minute live-equivalent feature builder
- replay engine with action simulation
- execution assumption layer

### Phase 3: Train The Three Core Models

**Build**

- continuation-value model
- hazard model
- exit-cost model

**Requirement**

All trained and validated on purged temporal splits.

### Phase 4: Policy Engine Integration

**Build**

- decision scorer
- uncertainty gate
- fallback logic
- risk constraint interface

### Phase 5: Shadow Deployment

**Build**

- live scoring without trading
- compare current production decisions vs V2.1 recommendations
- analyze disagreements and missed saves

### Phase 6: Controlled Production Rollout

**Build**

- feature flags
- per-account rollout control
- automatic fallback
- live calibration dashboards / summaries

---

## Concrete Data-and-Feature Build Plan

### What We Can Compute Live Immediately

From current broker chain + underlying quotes:

- debit-to-close
- short-strike delta/gamma
- per-side cushion
- spread width proxies
- VWAP offset
- rolling realized vol / ATR proxies
- time-based features
- account-level concentration features

### What Needs Historical Vendor Data To Become Strong

- robust replay at scale
- accurate minute-level option width history
- OI-based context features
- more stable exit-cost modeling
- better regime research across many market conditions

### Preferred Data Acquisition Order

1. **Use current broker live chain and your own live logs now**
2. **Backfill with existing ThetaData / local historical option datasets already in repo**
3. **Add Cboe or equivalent minute quote datasets if replay quality is not sufficient**
4. **Only later consider premium microstructure datasets**

---

## What Not To Build Yet

- full RL / PPO policy learning
- sequence models before tree models have clearly failed
- dealer-position hard overrides from rough OI inference
- expensive order-flow feeds before basic exit telemetry is fully exploited

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
│   ├── continuation_value_lgbm_v2.pkl
│   ├── hazard_1m_lgbm_v2.pkl
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

## Repo-Specific Implementation Checklist

This checklist assumes:

- `v1_2` remains the **entry model** for now
- exit management is the only major strategy upgrade in scope
- Tradier is the only live broker target for V2.1
- Schwab support can be added later after the Tradier path is stable

### Current Code Paths In Scope

The main code paths for the first implementation pass are:

- `scripts/zdom_bot_tradier.py`
- `scripts/zero_dte_bot.py`
- `scripts/orchestrator/position_manager.py`
- `scripts/orchestrator/execution_router.py`
- `scripts/orchestrator/orchestrator.py`
- `execution/broker_api/tradier_client.py`
- `execution/order_manager/ic_order_builder.py`

### Phase 0: Lock Scope And Control Path

#### Objectives

- freeze entry behavior around the current `v1_2` flow
- standardize Tradier as the only supported execution engine for V2.1 rollout
- choose one live exit-control path so implementation does not fragment

#### Tasks

1. Freeze the entry engine:
   - no changes to `v1_2` model selection
   - no changes to entry thresholds
   - no changes to scan cadence unless required by exit integration

2. Standardize Tradier-only assumptions in:
   - config defaults
   - rollout docs
   - feature flags

3. Pick the primary live control path.
   Recommendation:
   - build exit V2.1 on the orchestrator path rather than the legacy monolithic bot
   - use `PositionManager` + `ExecutionRouter` as the core management loop

4. Define the initial supported action set for production:
   - `HOLD`
   - `TIGHTEN`
   - `FULL_EXIT`
   - `EMERGENCY_EXIT`

5. Explicitly defer to later phases:
   - partial reduction
   - side-cutting / broken-condor management
   - Schwab parity

#### Exit Criteria

- one execution path is selected and documented
- no ambiguity remains about entry vs. exit ownership

### Phase 1: Telemetry Foundation

Telemetry means the bot's structured "flight recorder" data: what it saw, what it decided, and what actually happened.

#### Objectives

- capture all live state needed for research, debugging, and slippage modeling
- make future model training evidence-based

#### Tasks

1. Add per-monitor-cycle structured telemetry for each open position.

2. Log at minimum:
   - timestamp
   - account name
   - position id / label
   - underlying spot
   - net close mid proxy
   - executable close proxy
   - net width
   - short call delta
   - short put delta
   - short call gamma
   - short put gamma
   - time since entry
   - time to forced close
   - unrealized P&L
   - active hard-rule flags
   - chosen monitoring mode

3. Add per-order telemetry for every close attempt.

4. Log at minimum:
   - quote snapshot at order submit
   - urgency mode
   - submitted order price
   - fill price
   - time to fill
   - replace count
   - final slippage versus mid
   - final slippage versus executable proxy
   - exit reason

5. Persist telemetry in structured format:
   - JSONL or CSV
   - one directory such as `logs/v2_exit/`
   - daily roll files

6. Add a daily merge / normalization script so telemetry can be joined with order outcomes.

#### Files To Create Or Edit

- `scripts/orchestrator/position_manager.py`
- `scripts/orchestrator/execution_router.py`
- `ml/zdom_v2/1_data_collection/scripts/merge_execution_logs.py`

#### Exit Criteria

- every monitor cycle produces structured telemetry
- every close attempt produces structured execution telemetry
- daily outputs are easy to replay and analyze

### Phase 2: Tradier Live Data Layer

#### Objectives

- build a clean Tradier-first market-state adapter
- support low-latency live feature computation without heavy file reads

#### Tasks

1. Confirm and document Tradier live fields available from:
   - underlying quote or timesales endpoints
   - option chain endpoints
   - order status / fill endpoints

2. Verify support for:
   - bid / ask
   - bid / ask size if available
   - delta
   - gamma
   - IV
   - timestamps

3. Build a live data adapter that exposes:
   - current underlying state
   - current full 0DTE option chain
   - latest per-leg Greeks
   - latest quote freshness metadata

4. Maintain in-memory rolling state for:
   - 1-minute bars
   - current chain snapshot
   - per-position quote snapshots
   - rolling return / vol windows

5. Add stale-data guards:
   - if chain is stale, suppress model authority
   - if underlying data is stale, force conservative fallback logic

#### Files To Create Or Edit

- `execution/broker_api/tradier_client.py`
- `scripts/orchestrator/broker_manager.py`
- `ml/zdom_v2/2_feature_engineering/scripts/realtime_exit_features.py`

#### Exit Criteria

- live Tradier data can support all critical-path features
- stale data can be detected automatically

### Phase 3: Live Feature Engine

#### Objectives

- compute the first high-value feature set directly from Tradier live data
- share as much code as possible between live scoring and historical replay

#### Tasks

1. Build a single feature-builder module for V2.1 exits.

2. Implement the first production feature set:
   - short call / put absolute delta
   - short call / put gamma
   - call / put distance in points
   - call / put distance in ATR units
   - threatened side
   - delta velocity
   - gamma velocity
   - 1m / 3m / 5m returns
   - 5m / 15m realized vol
   - ATR ratio
   - VWAP offset
   - session range exhaustion
   - net close mid
   - net close executable proxy
   - width percent of mid
   - quote refresh gap
   - minutes since entry
   - minutes to close
   - account open-position count
   - combined unrealized percent

3. Add weak-context features only after core stability:
   - approximate GEX sign
   - GEX flip proxy
   - VIX / VIX1D context
   - IV skew proxy

4. Add unit tests for:
   - feature correctness
   - missing data handling
   - stale data handling

#### Files To Create Or Edit

- `ml/zdom_v2/2_feature_engineering/scripts/realtime_exit_features.py`
- `ml/zdom_v2/2_feature_engineering/scripts/position_threat_features.py`
- `ml/zdom_v2/2_feature_engineering/scripts/liquidity_features.py`
- `tests/`

#### Exit Criteria

- live feature builder works on Tradier data
- same logic can be reused by replay later

### Phase 4: Hard Risk Rules

#### Objectives

- ship the highest-value risk improvements before ML
- reduce tail losses immediately

#### Tasks

1. Implement delta-based warning and emergency thresholds.

2. Implement gamma-acceleration guard.

3. Implement cushion-collapse rule based on distance-to-short in ATR units.

4. Implement liquidity-protection rule:
   - if width or quote instability becomes extreme, shift to defensive or emergency logic

5. Implement final-hour gamma clamp:
   - tighter profit-taking
   - lower tolerance for threat
   - faster monitoring cadence

6. Implement stricter kill switches:
   - daily realized loss cap
   - combined account stop
   - max emergency exits per day
   - regime-dislocation fallback if data quality degrades

7. Add feature flags for all rules.

#### Files To Create Or Edit

- `scripts/orchestrator/position_manager.py`
- `ml/zdom_v2/5_execution/scripts/hard_risk_rules.py`

#### Exit Criteria

- hard rules can run in paper mode with no model dependency
- emergency exits fire deterministically under known thresholds

### Phase 5: Variable-Frequency Monitoring

#### Objectives

- stop treating all market states as equally urgent

#### Tasks

1. Implement monitoring modes:
   - normal: every 30-60 seconds
   - elevated: every 10-15 seconds
   - emergency: immediate re-check until flat or stabilized

2. Promote a position to elevated monitoring when:
   - delta warning threshold is breached
   - delta velocity spikes
   - ATR burst ratio spikes
   - width deteriorates sharply
   - final 45-60 minutes begin

3. Add state transitions and throttling so monitoring remains rate-limit aware under Tradier.

#### Files To Create Or Edit

- `scripts/orchestrator/orchestrator.py`
- `scripts/orchestrator/position_manager.py`

#### Exit Criteria

- the bot can shift monitoring frequency based on risk state
- Tradier rate limits are not violated

### Phase 6: Exit Execution Modes

#### Objectives

- make exit execution an explicit part of the risk engine

#### Tasks

1. Add explicit Tradier close modes:
   - `HARVEST`
   - `DEFENSIVE`
   - `EMERGENCY`

2. For each mode, define:
   - initial price logic
   - max wait time
   - cancel / replace cadence
   - escalation sequence

3. Add handling for:
   - working close orders
   - partial fills
   - repeated replace attempts
   - forced escalation under worsening threat

4. Centralize executable close pricing assumptions so there is one shared calculation path.

#### Files To Create Or Edit

- `scripts/orchestrator/execution_router.py`
- `execution/order_manager/ic_order_builder.py`
- `ml/zdom_v2/5_execution/scripts/execution_modes.py`

#### Exit Criteria

- exits run through explicit Tradier urgency modes
- fill-quality telemetry is captured automatically

### Phase 7: Historical Replay Engine

#### Objectives

- build an honest offline environment for evaluating exit policies

#### Tasks

1. Build per-minute position trajectory reconstruction.

2. Reconstruct from historical data:
   - underlying bars
   - option quotes
   - Greeks
   - open interest where available
   - your own order/fill telemetry

3. Reuse the live feature engine inside replay wherever possible.

4. Support multiple execution assumptions:
   - optimistic
   - executable proxy
   - stressed

5. Support side-by-side evaluation:
   - current baseline exits
   - hard-rule-only exits
   - later V2.1 policy exits

#### Files To Create Or Edit

- `ml/zdom_v2/1_data_collection/scripts/build_position_trajectories.py`
- `ml/zdom_v2/1_data_collection/scripts/build_exit_replay_dataset.py`
- `ml/zdom_v2/6_analysis/scripts/replay_v2_policy.py`

#### Exit Criteria

- one command can replay historical positions with live-equivalent features

### Phase 8: Model Training

#### Objectives

- train the three-model exit stack on replayable data

#### Tasks

1. Train a continuation-value regression model.

2. Train short-horizon hazard models:
   - 1-minute
   - 3-minute
   - 5-minute

3. Train an exit-cost model using live Tradier telemetry.

4. Use purged / embargoed temporal validation.

5. Evaluate:
   - calibration
   - tail-risk capture
   - policy value added
   - stability across temporal folds

6. Reject models that improve return only by increasing left-tail risk.

#### Files To Create Or Edit

- `ml/zdom_v2/4_training/scripts/train_continuation_value.py`
- `ml/zdom_v2/4_training/scripts/train_hazard_models.py`
- `ml/zdom_v2/4_training/scripts/train_exit_cost_model.py`
- `ml/zdom_v2/4_training/scripts/evaluate_policy_v2.py`

#### Exit Criteria

- models beat baseline on replay under realistic execution assumptions

### Phase 9: Policy Engine

#### Objectives

- combine rules, model outputs, and execution costs into a single action decision

#### Tasks

1. Build a policy engine that consumes:
   - hard-rule signals
   - continuation value
   - hazard probability
   - expected exit cost
   - uncertainty state

2. Implement initial production policy:
   - hard risk hit -> `EMERGENCY_EXIT`
   - high uncertainty -> conservative fallback
   - tail risk above limit -> `FULL_EXIT`
   - exit value sufficiently better than hold value -> `FULL_EXIT`
   - rising threat but not urgent -> `TIGHTEN`
   - else -> `HOLD`

3. Keep thresholds and buffers externally configurable.

#### Files To Create Or Edit

- `ml/zdom_v2/5_execution/scripts/policy_engine.py`
- `ml/zdom_v2/5_execution/scripts/uncertainty_gate.py`
- `scripts/orchestrator/position_manager.py`

#### Exit Criteria

- one policy interface determines the action each monitoring cycle

### Phase 10: Shadow Mode

#### Objectives

- validate V2.1 decisions live before capital deployment

#### Tasks

1. Run V2.1 scoring live in shadow alongside the current Tradier production exit logic.

2. Log all disagreements between:
   - baseline live decision
   - V2.1 recommended decision

3. Review each day:
   - missed saves
   - premature exits
   - slippage assumption accuracy
   - uncertainty fallback frequency
   - stale-data incidents

#### Files To Create Or Edit

- `ml/zdom_v2/6_analysis/scripts/shadow_decision_audit.py`
- `scripts/orchestrator/position_manager.py`

#### Exit Criteria

- enough live shadow evidence exists to judge whether V2.1 is promotable

### Phase 11: Controlled Tradier Rollout

#### Objectives

- deploy safely with bounded capital risk

#### Tasks

1. Roll out in order:
   - Tradier paper
   - one small live Tradier account
   - broader Tradier deployment only after stable results

2. Keep permanent safety fallbacks:
   - hard rules always active
   - fallback to baseline exits if data is stale
   - fallback if models fail to load
   - fallback if uncertainty is too high
   - fallback if repeated execution failures occur

3. Add daily post-trade audit summaries.

#### Files To Create Or Edit

- `scripts/zdom_bot_tradier.py`
- `scripts/orchestrator/config.py`
- `scripts/orchestrator/orchestrator.py`

#### Exit Criteria

- V2.1 can be toggled on per Tradier account with automatic fallback

### Phase 12: Operations And Governance

#### Objectives

- keep the system observable, reviewable, and reversible

#### Tasks

1. Add daily reports covering:
   - exit reasons
   - emergency exit count
   - average slippage by exit mode
   - worst trade
   - calibration drift
   - stale-data incidents

2. Add weekly model-health review.

3. Version every live:
   - model artifact
   - policy config
   - hard-rule config

4. Maintain same-day rollback capability.

#### Exit Criteria

- the system can be monitored and rolled back safely

### Recommended First Build Order

The recommended execution order is:

1. telemetry foundation
2. Tradier live data layer
3. live feature engine
4. hard risk rules
5. variable-frequency monitoring
6. exit execution modes
7. historical replay engine
8. model training
9. policy engine
10. shadow mode
11. controlled Tradier rollout
12. operating cadence and governance

### Earliest Practical Milestone

The first meaningful production milestone is:

- structured telemetry
- live feature engine
- delta / gamma / cushion hard rules
- variable-frequency monitoring
- explicit Tradier emergency exit mode

That milestone should deliver immediate practical value even before any ML exit model is deployed.

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
