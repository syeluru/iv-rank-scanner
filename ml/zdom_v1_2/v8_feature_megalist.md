# V8 Feature Megalist — Candidate Intraday Features for 0DTE IC Prediction

> Compiled from 5 parallel research agents covering: technical indicators, options microstructure,
> market internals, ML feature engineering, and historical data sources.

---

## SUMMARY

| Category | New Features | Data Source |
|----------|-------------|-------------|
| A. Price Action & Technical Indicators | 38 | SPX 1-min bars (already have) |
| A2. Multi-Day S/R & Structure | 30 | SPX 1-min bars (already have) |
| A3. Sloped S/R & Trendlines | 22 | SPX 5-min resampled (already have) |
| B. Volatility Surface & IV Dynamics | 20 | Option chain greeks (already have) |
| C. Dealer Positioning (GEX/DEX/Vanna/Charm) | 18 | Option chain + OI (already have / ThetaData) |
| D. Realized vs Implied Vol | 10 | SPX 1-min + option chain (already have) |
| E. VIX Complex & Term Structure | 12 | VIX/VIX1D (have) + VIX3M/VIX9D/VVIX (ThetaData) |
| F. Cross-Asset & Macro | 12 | ETF proxies via ThetaData + FRED |
| G. Market Breadth & Order Flow | 18 | IBKR / ThetaData / SqueezeMetrics |
| H. Volume & Liquidity | 12 | SPY/ES volume (ThetaData) |
| I. Statistical & Regime Detection | 14 | Computed from SPX bars |
| J. Temporal & Event Encoding | 10 | Computed from calendar |
| K. Feature Engineering Techniques | 8 | Applied to other features |
| **TOTAL** | **~221** | |

---

## A. PRICE ACTION & TECHNICAL INDICATORS (35 features)
*Source: SPX 1-min OHLCV bars (already have)*

### A1. Momentum Oscillators
| # | Feature | What It Measures | Why It Predicts IC Outcomes |
|---|---------|-----------------|---------------------------|
| 1 | `macd_histogram_5_13_9` | MACD histogram value | Momentum state not captured by RSI alone |
| 2 | `macd_hist_slope_5bar` | Rate of change of MACD histogram | Acceleration/deceleration of trend |
| 3 | `macd_zero_cross_bars` | Bars since last MACD zero-line cross | Longer = more established trend |
| 4 | `williams_r_14` | Overbought/oversold (-100 to 0) | Near -50 = neutral = good for IC |
| 5 | `stoch_in_neutral_zone` | Both %K and %D between 30-70 | Range-bound market signal |
| 6 | `imi_14` | Intraday Momentum Index (open-to-close RSI) | Different signal than close-to-close RSI |
| 7 | `roc_5` | 5-bar rate of change | Short-term momentum |
| 8 | `roc_15` | 15-bar rate of change | Medium-term momentum |
| 9 | `roc_30` | 30-bar rate of change | Longer-term momentum |
| 10 | `roc_divergence` | Sign disagreement between roc_5 and roc_30 | Short-term reversal = range-bound signal |

### A2. Bollinger Band Extensions
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 11 | `bb_position` | (close - lower) / (upper - lower) | 0-1 range; near 0.5 = mean |
| 12 | `close_vs_upper_band_pct` | Distance above upper band | Breakout risk for IC |
| 13 | `bb_width_change_rate` | Rate of BBW expansion over 10 bars | Rapid expansion = regime change |

### A3. VWAP Enhancements
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 14 | `vwap_z_score` | (price - VWAP) / rolling_std | Continuous deviation (better than binary) |
| 15 | `vwap_touch_count_30m` | VWAP crosses in last 30 bars | More crosses = choppier = IC-friendly |

### A4. Moving Average Features
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 16 | `ema_ribbon_width` | (EMA9 - EMA21) / price * 100 | Wide = trending, narrow = consolidating |
| 17 | `ma_alignment` | EMA9 vs SMA20 vs SMA50 alignment score | Strong alignment = trend = bad for IC |

### A5. ATR Extensions
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 18 | `atr_acceleration` | Change in ATR over last 10 bars | Expanding volatility = dangerous |
| 19 | `normalized_atr` | ATR / close * 100 | Absolute vol level as % of price |

### A6. Z-Score Features
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 20 | `price_zscore_20` | (close - SMA20) / std20 | Direct measure of price extension |
| 21 | `price_zscore_60` | (close - SMA60) / std60 | Z > 2 or < -2 = extreme = likely to revert |
| 22 | `return_zscore_5min` | Z-score of last 5-min return vs session | Detects abnormal moves |

### A7. Support / Resistance
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 23 | `distance_to_nearest_support_pct` | % to nearest support level | Proximity to bounce zone |
| 24 | `distance_to_nearest_resistance_pct` | % to nearest resistance level | Proximity to rejection zone |
| 25 | `sr_density` | S/R levels within +/-0.5% of price | High density = congestion = range-bound |
| 26 | `price_vs_pivot` | (close - daily pivot) / pivot * 100 | Relative to pivot point |
| 27 | `pivot_zone` | Which Camarilla zone (-3 to +3) | Structural price positioning |

**S/R Computation:** 5-day history, `scipy.signal.find_peaks` (prominence >= 0.15%, distance >= 15 bars), cluster with `AgglomerativeClustering` (threshold = 0.1%).

### A8. Candlestick Aggregates
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 28 | `doji_count_last_30` | Doji candles in last 30 bars | Indecision = range-bound = IC-friendly |
| 29 | `avg_body_ratio_30` | Average |close-open|/(high-low) over 30 bars | Small bodies = indecision |
| 30 | `directional_candle_ratio` | (bullish - bearish) / total in 30 bars | Near 0 = balanced |
| 31 | `max_candle_body_pct` | Largest single-bar body as % of price | Large = strong conviction = IC risk |
| 32 | `wick_ratio_avg` | Average (upper+lower wick) / body | High = rejection/indecision |

### A9. Multi-Timeframe
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 33 | `trend_alignment_score` | Count of timeframes where price > SMA20 (0-4) | Near 2 = mixed = IC-friendly |
| 34 | `rsi_spread_1m_15m` | RSI(14) on 1-min minus RSI(14) on 15-min | Timeframe divergence |
| 35 | `adx_max_across_tf` | max(ADX_5m, ADX_15m, ADX_30m) | Any timeframe trending = IC at risk |

### A10. Chart Pattern Derivatives
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 36 | `local_extrema_count_30m` | Peaks + valleys in 30 bars | More = choppier = IC-friendly |
| 37 | `higher_highs_count` | Consecutive higher highs | > 3 = strong uptrend |
| 38 | `lower_lows_count` | Consecutive lower lows | > 3 = strong downtrend |

**Libraries:** `pandas-ta` (130+ indicators), `scipy.signal.find_peaks` for S/R and patterns, `PatternPy` for chart patterns.

---

## A2. MULTI-DAY S/R & STRUCTURE (30 features)
*Source: SPX 1-min bars resampled to 5-min/15-min/daily (already have)*

### A2.1 Prior-Day Structural Levels
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `prior_day_high` | Yesterday's high | Most-watched level by all traders |
| 2 | `prior_day_low` | Yesterday's low | Key support reference |
| 3 | `dist_to_prior_day_high_pct` | % distance to yesterday's high | Proximity to breakout/rejection |
| 4 | `dist_to_prior_day_low_pct` | % distance to yesterday's low | Proximity to breakdown/bounce |
| 5 | `inside_prior_day_range` | Price within yesterday's H/L (binary) | Range day vs trend day signal |
| 6 | `prior_day_close_dist_pct` | Distance from yesterday's close | Gap/reversion context |
| 7 | `prior_day_range_pct` | Yesterday's H-L as % of price | Sets expected range context |

### A2.2 Multi-Day Range
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 8 | `rolling_5d_high` | 5-day high | Weekly resistance |
| 9 | `rolling_5d_low` | 5-day low | Weekly support |
| 10 | `dist_to_5d_high_pct` | % distance to 5-day high | Proximity to weekly breakout |
| 11 | `dist_to_5d_low_pct` | % distance to 5-day low | Proximity to weekly breakdown |
| 12 | `breakout_from_5d_range` | Price exceeded 5-day H or L (binary) | Trend day confirmation |
| 13 | `days_in_range` | Consecutive days with overlapping H/L | Compression → imminent breakout |
| 14 | `consolidation_tightness` | 2-day range / 5-day range | Small = compressing |
| 15 | `range_contraction_streak` | Consecutive days of shrinking range | Precedes breakouts |

### A2.3 Peak/Valley S/R from Prior Days
*Computed: 5-day 5-min bars → `find_peaks(prominence>=0.15%, distance>=10)` → cluster with `AgglomerativeClustering(threshold=0.1%)`*
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 16 | `nearest_support_dist_pct` | % to nearest clustered support level | Proximity to bounce zone |
| 17 | `nearest_resistance_dist_pct` | % to nearest clustered resistance level | Proximity to rejection zone |
| 18 | `sr_level_strength` | Touch count of nearest S/R level | More touches = stronger level |
| 19 | `sr_density_nearby` | S/R levels within 0.3% of price | High density = congestion = range-bound |

### A2.4 Fibonacci Retracements
*Computed from most recent 5-day significant swing (high→low or low→high)*
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 20 | `dist_to_fib_382_pct` | Distance to 38.2% retracement | Shallow pullback level |
| 21 | `dist_to_fib_500_pct` | Distance to 50% retracement | Mean-reversion midpoint |
| 22 | `dist_to_fib_618_pct` | Distance to 61.8% retracement | Deep pullback level |
| 23 | `nearest_fib_level_dist` | Distance to whichever fib is closest | General fib proximity |

### A2.5 Multi-Day Pivots
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 24 | `price_vs_pivot` | (close - daily pivot) / pivot * 100 | Relative to pivot point |
| 25 | `camarilla_zone` | Which Camarilla zone (-3 to +3) | Expected intraday range structure |
| 26 | `price_vs_camarilla_r3_pct` | Distance to Camarilla R3 | Beyond R3 = range breakout likely |

### A2.6 Multi-Day Swing Structure
*Computed from 5-day 15-min bars, `find_peaks(distance=15, prominence=0.1%)`*
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 27 | `higher_highs_multiday` | Successive higher swing highs (count) | Multi-day uptrend strength |
| 28 | `lower_lows_multiday` | Successive lower swing lows (count) | Multi-day downtrend strength |
| 29 | `swing_trend_score` | +1 per HH/HL, -1 per LH/LL (net) | Trend direction + strength |
| 30 | `last_swing_magnitude_pct` | Size of most recent swing | Large = volatile, small = range-bound |

---

## A3. SLOPED S/R & TRENDLINES (22 features)
*Source: SPX 5-min bars resampled from 1-min (already have)*

**Why sloped S/R matters:** Horizontal S/R captures WHERE buyers/sellers step in. Sloped S/R captures the RATE at which they're willing to pay more (ascending support) or accept less (descending resistance). A break of a well-tested ascending support line means the buyer group has exhausted — qualitatively different from breaking a horizontal level.

**Evidence:** Lo, Mamaysky & Wang (2000, *Journal of Finance*) found trendlines have "marginal predictive value" as binary rules, but when encoded as continuous features (slope, distance, R², touch count), they capture momentum rate and mean-reversion boundaries useful for ML. The slope itself encodes momentum strength — steep ascending support = aggressive buying, shallow = weakening.

### A3.1 Regression Trendlines on Swing Points (5-Day)
*Fit linear regression on last 3-5 swing highs (resistance) and swing lows (support) from 5-day 5-min bars*
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `support_slope_5d` | Rate of ascending/descending support (pts/hr) | Positive = buyers increasingly aggressive |
| 2 | `resistance_slope_5d` | Rate of ascending/descending resistance (pts/hr) | Negative = sellers defending lower prices |
| 3 | `dist_to_support_trendline_5d_pct` | How close price is to support trendline | Near = likely to bounce (IC-friendly) |
| 4 | `dist_to_resistance_trendline_5d_pct` | How close price is to resistance trendline | Near = likely to reject (IC-friendly) |
| 5 | `support_trendline_r2_5d` | R² of support line fit | High R² = well-respected, clean line |
| 6 | `resistance_trendline_r2_5d` | R² of resistance line fit | High R² = well-respected, clean line |
| 7 | `support_touch_count_5d` | Times price touched support trendline and bounced | More touches = more reliable |
| 8 | `resistance_touch_count_5d` | Times price touched resistance trendline and bounced | More touches = more reliable |

### A3.2 Intraday Trendlines (Today Only)
*Same algorithm on today's 5-min bars*
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 9 | `support_slope_1d` | Today's intraday support slope | Current session momentum |
| 10 | `resistance_slope_1d` | Today's intraday resistance slope | Current session ceiling drift |
| 11 | `dist_to_support_trendline_1d_pct` | Distance to today's support line | Intraday bounce proximity |
| 12 | `dist_to_resistance_trendline_1d_pct` | Distance to today's resistance line | Intraday rejection proximity |

### A3.3 Trendline Convergence (Triangle/Wedge Detection)
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 13 | `trendline_convergence_rate_5d` | Rate upper/lower 5d trendlines approach each other | Converging = compression → imminent breakout |
| 14 | `bars_to_apex_5d` | Estimated bars until 5d trendlines intersect | Breakout timing estimate |
| 15 | `channel_slope_5d` | Average of support + resistance slopes | Overall multi-day trend direction |
| 16 | `channel_angle_diff_5d` | Difference in slopes | Converging, parallel, or diverging |

### A3.4 Multi-Timeframe Trendline Agreement
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 17 | `support_slope_agreement` | Do 1d and 5d support slopes agree in direction? | Aligned = stronger support |
| 18 | `trendline_tf_divergence` | |1d slope - 5d slope| magnitude | Large = conflicting timeframes |

### A3.5 Pattern Proxies (Continuous Features)
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 19 | `symmetry_score` | Correlation of recent price with its mirror | Detects double tops/bottoms continuously |
| 20 | `triangle_convergence_pct` | Channel width as % of price, decreasing | Wedge/triangle tightness |
| 21 | `avg_bounce_magnitude_pct` | Average size of bounces off trendlines | Large bounces = strong levels |
| 22 | `touch_recency_decay` | Exp-weighted touch score (recent > old) | Stale trendlines = less reliable |

---

## B. VOLATILITY SURFACE & IV DYNAMICS (20 features)
*Source: Option chain greeks from spxw_0dte_intraday_greeks.parquet (already have)*

### B1. IV Rate of Change
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `atm_iv_velocity` | ATM IV change over last 5-15 min | Rising IV = increasing uncertainty |
| 2 | `atm_iv_acceleration` | Rate of change of IV velocity | Regime transitions |
| 3 | `atm_iv_change_since_open` | % change in ATM IV from session open | Intraday IV drift |

### B2. IV Skew
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 4 | `risk_reversal_25d` | IV(25d put) - IV(25d call) | Steep = market pricing downside |
| 5 | `skew_rate_of_change` | Change in skew over 30 min | Rapidly steepening = emerging fear |
| 6 | `iv_butterfly` | (IV_25d_put + IV_25d_call)/2 - IV_atm | Tail risk premium (curvature) |
| 7 | `put_call_iv_spread_at_strikes` | IV diff at IC short strikes | Asymmetric wing pricing |

### B3. IV Surface
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 8 | `skew_slope` | Linear regression of IV vs moneyness | Overall smirk steepness |
| 9 | `skew_curvature` | Quadratic term of IV vs moneyness fit | Smile shape |
| 10 | `wing_iv_richness` | avg(IV at wings) / IV_atm | Rich wings = better credit per risk |
| 11 | `moneyness_weighted_iv` | OI-weighted average IV across strikes | Aggregate vol level |

### B4. IV Regime
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 12 | `iv_rank_52w` | ATM IV percentile vs 52-week range | High = rich premiums |
| 13 | `iv_percentile_20d` | ATM IV percentile vs 20-day range | Short-term IV positioning |
| 14 | `vol_of_vol_30m` | Std of ATM IV changes over 30 min | High = IV could spike anytime |

### B5. Term Structure
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 15 | `iv_0dte_vs_1dte_ratio` | 0DTE ATM IV / 1DTE ATM IV | Inverted = acute intraday risk |
| 16 | `iv_0dte_vs_vix_ratio` | 0DTE ATM IV / VIX | Short vs medium-term vol expectations |
| 17 | `skew_0dte_vs_1dte` | Skew difference across expirations | Steeper 0DTE = more tail risk today |

### B6. IV at IC Strikes
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 18 | `short_put_iv` | IV at IC short put strike | Direct risk pricing of that wing |
| 19 | `short_call_iv` | IV at IC short call strike | Direct risk pricing of that wing |
| 20 | `iv_asymmetry` | |short_put_iv - short_call_iv| | How skewed the risk is |

---

## C. DEALER POSITIONING — GEX/DEX/VANNA/CHARM (18 features)
*Source: Option chain with OI + greeks (ThetaData, partially have)*

### C1. Gamma Exposure (GEX)
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `net_gex` | Net dealer gamma exposure ($B) | Positive = mean-reverting (IC-friendly), negative = momentum-amplifying |
| 2 | `gex_normalized` | GEX / spot^2 | Cross-regime comparable |
| 3 | `gex_regime` | +1 positive / -1 negative | Binary regime flag |
| 4 | `gamma_flip_distance_pct` | Distance from spot to zero-gamma strike | How much room before negative gamma |
| 5 | `gex_0dte_share` | Fraction of total GEX from 0DTE | High = intraday gamma dynamics dominate |
| 6 | `gex_concentration_hhi` | Herfindahl index of GEX by strike | Concentrated = strong pin effects |
| 7 | `max_gamma_strike_dist` | Distance to strike with highest |GEX| | Near = price gets "pinned" |

**GEX Formula:** `call_gex = gamma * OI * 100 * spot^2 * 0.01` per strike, `put_gex = -gamma * OI * 100 * spot^2 * 0.01`, Net GEX = sum(call_gex) + sum(put_gex).

**Evidence:** Markets close within GEX-estimated range ~78% of the time. Goldman research: 1 std dev increase in gamma imbalance reduces absolute returns by 20+ bps.

### C2. Delta Exposure (DEX)
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 8 | `net_dex` | Net dealer delta exposure | Directional hedging pressure |
| 9 | `dex_change_30m` | DEX change over 30 min | Shifting dealer positioning |

### C3. Vanna & Charm Exposure
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 10 | `vanna_exposure` | Aggregate vanna × OI across chain | IV change → forced delta hedging |
| 11 | `charm_exposure` | Aggregate charm × OI across chain | Time decay → forced delta hedging (huge in 0DTE) |

**Data:** ThetaData `greeks_second_order` endpoint provides vanna, charm per contract.

### C4. Structural Levels
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 12 | `call_wall_distance_pct` | Distance to strike with max call gamma×OI | Resistance from dealer hedging |
| 13 | `put_wall_distance_pct` | Distance to strike with max put gamma×OI | Support from dealer hedging |
| 14 | `ic_short_put_gamma` | GEX at IC short put strike | Dealer protection of your put wing |
| 15 | `ic_short_call_gamma` | GEX at IC short call strike | Dealer protection of your call wing |

### C5. GEX Shape
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 16 | `gex_skewness` | Skewness of GEX distribution by strike | Asymmetric dealer positioning |
| 17 | `gex_kurtosis` | Peaked vs flat GEX distribution | Peaked = strong pin, flat = no clear influence |
| 18 | `abs_gamma_total` | Total absolute gamma regardless of sign | High = massive hedging flows both ways |

---

## D. REALIZED VS IMPLIED VOL (10 features)
*Source: SPX 1-min bars + option chain (already have)*

| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `vrp_intraday` | ATM IV (annualized) - GK RV 30m (annualized) | Positive VRP = IC collecting more than moves justify |
| 2 | `vrp_rate_of_change` | Change in VRP over 15 min | Collapsing VRP = moves accelerating vs pricing |
| 3 | `gk_rv_30m` | Garman-Klass RV over 30 min (annualized) | Efficient vol estimator (7.4x better than close-to-close) |
| 4 | `yz_rv_30m` | Yang-Zhang RV over 30 min | Handles opening jumps (14x more efficient) |
| 5 | `rv_acceleration` | RV change over 15 min | Moves getting larger = early warning |
| 6 | `rv_5m_vs_30m_ratio` | Short-window RV / long-window RV | > 1 = expanding vol, < 1 = calming |
| 7 | `realized_skewness_30m` | Skewness of 30-min returns | Negative = fat left tail (put wing risk) |
| 8 | `realized_kurtosis_30m` | Kurtosis of 30-min returns | > 5 = outsized moves threatening wings |
| 9 | `gk_rv_vs_atm_iv` | Direct ratio of realized to implied | **Already proven top-5 feature in v4/v5** |
| 10 | `rv_percentile_20d` | Where current RV sits vs 20-day history | Context for current vol level |

---

## E. VIX COMPLEX & TERM STRUCTURE (12 features)
*Source: VIX/VIX1D (have) + VIX3M/VIX9D/VVIX (ThetaData — test endpoints)*

| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `vix1d_vix_ratio` | VIX1D / VIX | > 1 = today expected more volatile than avg |
| 2 | `vix_vix3m_spread` | VIX - VIX3M | Positive = backwardation = stress |
| 3 | `vix_term_slope` | Linear fit across VIX1D/VIX/VIX9D/VIX3M | Full term structure gradient |
| 4 | `vix_term_curvature` | Quadratic term of VIX term structure | Convexity of vol expectations |
| 5 | `vvix_level` | VVIX absolute value | > 110 precedes VIX spikes; < 80 = safe |
| 6 | `vvix_roc_30m` | VVIX rate of change over 30 min | Rising VVIX = increasing tail risk |
| 7 | `vix_velocity_30m` | VIX change over 30 min (%) | 2+ point move = emergency signal |
| 8 | `vix1d_intraday_range` | VIX1D high-low range normalized | Vol-of-vol proxy from 1-day index |
| 9 | `vix_mean_reversion_dist` | VIX distance from 10-day SMA | How stretched VIX is |
| 10 | `vix_rsi_14` | RSI(14) on 1-min VIX bars | VIX overbought/oversold |
| 11 | `vix_contango_flag` | 1 if VIX1D < VIX < VIX3M (normal) | Normal structure = calm |
| 12 | `vix1d_percentile_20d` | VIX1D percentile rank vs 20 days | Context for intraday vol level |

**Data availability:** Test ThetaData endpoints:
- `http://127.0.0.1:25503/v3/index/history/ohlc?symbol=VIX9D&interval=1m`
- `http://127.0.0.1:25503/v3/index/history/ohlc?symbol=VIX3M&interval=1m`
- `http://127.0.0.1:25503/v3/index/history/ohlc?symbol=VVIX&interval=1m`

---

## F. CROSS-ASSET & MACRO (12 features)
*Source: ETF proxies via ThetaData + FRED for daily*

| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `tlt_intraday_return` | 20Y+ treasury ETF intraday return | Bond stress → equity stress |
| 2 | `tlt_spx_corr_30m` | Rolling 30-min TLT-SPX correlation | Positive = stress (both selling off) |
| 3 | `shy_tlt_spread_chg` | Change in SHY-TLT spread intraday | Yield curve intraday dynamics |
| 4 | `uup_intraday_return` | Dollar index ETF intraday return | DXY vol → SPX vol |
| 5 | `gld_intraday_return` | Gold ETF intraday return | Risk-off signal |
| 6 | `gld_spx_corr_30m` | Rolling 30-min gold-SPX correlation | Positive = risk-off flow |
| 7 | `risk_off_composite` | gold_up + dxy_up + yields_down + vix_up | Aggregate risk-off signal |
| 8 | `sector_dispersion` | Cross-sectional std of sector ETF returns | High = rotation, low = trend |
| 9 | `avg_sector_correlation` | Avg pairwise correlation among sectors | High = systematic risk = larger SPX moves |
| 10 | `dix_level` | SqueezeMetrics Dark Index (prior day) | Dark pool sentiment |
| 11 | `squeezemetrics_gex` | SqueezeMetrics GEX (prior day) | Market-wide gamma regime |
| 12 | `yield_curve_2s10s` | 2Y-10Y treasury spread (FRED, daily) | Macro regime context |

**Data sources:**
- ThetaData: TLT, SHY, UUP, GLD, USO, XLF/XLK/XLE/XLV sector ETFs (same API, no extra cost)
- SqueezeMetrics: Free CSV download from squeezemetrics.com/monitor/dix
- FRED: DGS2, DGS10, T10Y2Y (free API, daily)

---

## G. MARKET BREADTH & ORDER FLOW (18 features)
*Source: IBKR ($TICK, $ADVN, $DECL) + ThetaData + SqueezeMetrics*

### G1. NYSE TICK (requires IBKR or Databento)
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `tick_current` | Current NYSE TICK reading | Instantaneous breadth |
| 2 | `tick_cumulative` | Cumulative TICK since open | Directional pressure accumulation |
| 3 | `tick_range_30min` | TICK range over 30 min | Breadth volatility |
| 4 | `tick_extreme_count` | Readings > +800 or < -800 today | Program trading events |
| 5 | `cum_tick_slope` | Slope of cumulative TICK | Breadth momentum |

### G2. Advance/Decline (requires IBKR)
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 6 | `ad_ratio` | Advances / Declines ratio | Breadth confirmation |
| 7 | `ad_cumulative` | Cumulative A/D since open | Breadth trend |
| 8 | `mcclellan_oscillator` | EMA19(A/D) - EMA39(A/D) | Smoothed breadth momentum |

### G3. Options Order Flow
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 9 | `pc_volume_ratio_30m` | Rolling 30-min put/call volume ratio | Fear/hedging demand |
| 10 | `pc_ratio_roc` | Rate of change of P/C ratio | Onset of panic buying |
| 11 | `net_premium_flow` | Dollar call premium - put premium | Conviction-weighted sentiment |
| 12 | `volume_surge_ratio` | Current vol vs same-TOD historical avg | New information arriving |
| 13 | `otm_volume_concentration` | Volume in deep OTM (>2 sigma) / total | Tail risk hedging activity |
| 14 | `sweep_ratio` | Intermarket sweep volume / total | Urgency and informed trading |

### G4. Cumulative Volume Delta (requires ES tick data)
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 15 | `cvd_normalized` | Cumulative (ask vol - bid vol) / total | Net buying/selling pressure |
| 16 | `cvd_slope_30m` | CVD linear regression slope over 30 min | Order flow momentum |
| 17 | `cvd_price_divergence` | CVD direction disagrees with price | Hidden directional pressure |
| 18 | `aggressive_buy_ratio` | Volume at ask / total volume | Buy-side aggression |

---

## H. VOLUME & LIQUIDITY (12 features)
*Source: SPY/ES 1-min bars with volume (ThetaData)*

### H1. Volume Profile
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `poc_distance_pct` | Distance from price to Point of Control | Price at POC = fair value, away = overextended |
| 2 | `in_value_area` | Price within 70% volume area (binary) | Inside = mean-reverting, outside = breakout |
| 3 | `va_width_pct` | Value area width as % of price | Narrow = tight range, wide = volatile |
| 4 | `volume_node_density` | High-volume nodes near current price | Support/resistance from volume |

### H2. OBV & Volume Momentum
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 5 | `obv_slope_30m` | OBV linear regression slope over 30 min | Volume-confirmed momentum |
| 6 | `obv_price_divergence` | OBV direction disagrees with price | Hidden accumulation/distribution |
| 7 | `volume_momentum` | Current volume vs 20-bar avg | Activity level |

### H3. Liquidity
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 8 | `kyles_lambda` | Price impact per unit of signed volume | Illiquid = larger moves per trade |
| 9 | `kyles_lambda_change` | Lambda change vs first hour | Deteriorating liquidity |
| 10 | `amihud_illiquidity` | avg |return| / dollar_volume | Market depth proxy |
| 11 | `spread_widening_rate` | ATM option spread change vs session start | Market maker stress |
| 12 | `vpin` | Volume-sync'd P(informed trading) | Informed flow = vol spike risk |

---

## I. STATISTICAL & REGIME DETECTION (14 features)
*Source: Computed from SPX 1-min bars*

### I1. Time-Series Statistics
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `hurst_exponent_60` | Hurst exponent over 60 bars | < 0.5 = mean-reverting (IC-friendly), > 0.5 = trending |
| 2 | `sample_entropy_60` | Sample entropy over 60 bars | Lower = more predictable |
| 3 | `permutation_entropy` | Ordinal pattern entropy | Captures up/down sequence complexity |
| 4 | `autocorr_lag1` | Rolling autocorrelation at lag 1 | Positive = momentum, negative = reversion |
| 5 | `autocorr_lag5` | Rolling autocorrelation at lag 5 | Intermediate-term persistence |
| 6 | `partial_autocorr_lag1` | Partial autocorrelation at lag 1 | Pure lag-1 effect (no intermediates) |

### I2. Regime Detection
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 7 | `hmm_regime_prob_calm` | P(calm regime) from 3-state HMM | Direct regime classification |
| 8 | `hmm_regime_prob_volatile` | P(volatile regime) | High = IC dangerous |
| 9 | `bars_since_changepoint` | Bars since last detected change-point | Fresh regime = uncertain |
| 10 | `changepoints_last_60` | Change-points in last 60 bars | Many = unstable market |

### I3. Distribution Features
| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 11 | `rolling_skewness_30` | Skewness of 30-bar returns | Directional bias in distribution |
| 12 | `rolling_kurtosis_30` | Kurtosis of 30-bar returns | Tail risk in recent returns |
| 13 | `range_expansion_ratio` | Current true range / ATR(14) | > 1 = expanding vol |
| 14 | `rv_percentile_rank` | Where current RV sits in 20-day dist | Context for volatility level |

**Libraries:** `nolds` (Hurst), `antropy` (entropy), `hmmlearn` (HMM), `ruptures` (change-points), `fracdiff` (fractional differentiation).

---

## J. TEMPORAL & EVENT ENCODING (10 features)
*Source: Computed from calendar + existing macro date tables*

| # | Feature | What It Measures | Why |
|---|---------|-----------------|-----|
| 1 | `theta_decay_proxy` | 1 / sqrt(minutes_to_close) | Approximates theta acceleration |
| 2 | `session_fraction_squared` | (minutes_elapsed / 390)^2 | Captures acceleration near close |
| 3 | `fomc_proximity_kernel` | exp(-0.5*(hours_to_fomc/24)^2) | Gaussian decay around FOMC |
| 4 | `cpi_proximity_kernel` | exp(-0.5*(hours_to_cpi/24)^2) | Gaussian decay around CPI |
| 5 | `nfp_proximity_kernel` | exp(-0.5*(hours_to_nfp/24)^2) | Gaussian decay around NFP |
| 6 | `post_fomc_hours` | Hours since last FOMC (0 if >48h) | Post-FOMC drift effect |
| 7 | `post_cpi_hours` | Hours since last CPI (0 if >24h) | CPI reaction unfolds over hours |
| 8 | `event_vol_premium` | Expected vol multiplier from nearest event | FOMC = ~1.5x avg vol |
| 9 | `pre_event_compression` | Low vol + event < 24h away (binary) | Vol compression resolves violently |
| 10 | `lunch_hour_flag` | 12:00-13:00 ET (binary) | Known low-vol period |

---

## K. FEATURE ENGINEERING TECHNIQUES (8 meta-features)
*Applied to other raw features above*

| # | Technique | What It Does | Apply To |
|---|-----------|-------------|----------|
| 1 | `fractional_diff_d` | Fractional differentiation (d≈0.3-0.7) | SPX price, VIX — preserves memory while achieving stationarity |
| 2 | `ewma_vol_ratio` | EWMA(span=5).std / EWMA(span=30).std | Returns — short vs long vol comparison |
| 3 | `expanding_zscore` | Z-score using expanding (not full-day) stats | All features — avoid look-ahead |
| 4 | `percentile_rank_60` | Non-parametric rank over 60 bars | ATM IV, RV, VIX — robust to outliers |
| 5 | `quantile_transform` | Map to normal distribution | All features — normalize for ML |
| 6 | `interaction_constraints` | XGBoost group-level interaction limits | Prevent spurious cross-group interactions |
| 7 | `purged_cv` | Purged K-Fold cross-validation | Training — prevent label leakage |
| 8 | `boruta_shap_selection` | All-relevant feature selection | Final feature set — keep only robust features |

---

## DATA ACQUISITION PRIORITY

### Tier 1 — FREE, compute from data we already have
These require NO new data ingestion. Pure computation from existing parquets.

| Data Source | Features Enabled | Count |
|-------------|-----------------|-------|
| SPX 1-min bars | Section A (technicals, S/R, patterns) | 38 |
| SPX 1-min bars (resampled) | Section A2 (multi-day S/R, structure) | 30 |
| SPX 1-min bars (resampled) | Section A3 (sloped S/R, trendlines) | 22 |
| SPX 1-min bars | All Section I (statistics, regime) | 14 |
| Option chain greeks | All Section B (IV surface, skew) | 20 |
| Option chain + OI | Most of Section C (GEX/DEX/walls) | ~12 |
| SPX + option chain | All Section D (RV vs IV) | 10 |
| VIX + VIX1D 1-min | Most of Section E | ~8 |
| Calendar | All Section J (temporal, events) | 10 |
| **Subtotal** | | **~164** |

### Tier 2 — FREE, requires ThetaData pulls (same API we already use)
| Data Source | Endpoint | Features Enabled | Count |
|-------------|----------|-----------------|-------|
| VIX3M, VIX9D, VVIX 1-min | `/v3/index/history/ohlc` | Rest of Section E (term structure) | 4 |
| SPY 1-min (has volume!) | `/v3/stock/history/ohlc?symbol=SPY` | Section H (volume profile, OBV, liquidity) | 12 |
| TLT, SHY, UUP, GLD 1-min | `/v3/stock/history/ohlc` | Section F (cross-asset) | 7 |
| Sector ETFs 1-min | `/v3/stock/history/ohlc` | Sector dispersion | 2 |
| 2nd-order greeks | `/v2/hist/option/greeks_second_order` | Vanna, charm exposure | 2 |
| **Subtotal** | | | **~27** |

### Tier 3 — FREE external downloads
| Source | Data | Features | Count |
|--------|------|----------|-------|
| SqueezeMetrics | DIX + GEX CSV | `dix_level`, `squeezemetrics_gex` | 2 |
| CBOE | Put/call ratio archive CSV | Daily P/C ratio context | 1 |
| FRED | DGS2, DGS10, T10Y2Y | `yield_curve_2s10s` | 1 |
| **Subtotal** | | | **4** |

### Tier 4 — Requires paid data / IBKR streaming
| Source | Data | Features | Count |
|--------|------|----------|-------|
| IBKR | $TICK, $ADVN, $DECL | NYSE TICK, breadth (Section G1-G2) | 8 |
| IBKR/Databento | ES tick data | CVD (Section G4) | 4 |
| **Subtotal** | | | **12** |

---

## FEATURE SELECTION STRATEGY

With ~169 candidate features, overfitting is a real risk. Recommended pipeline:

1. **Compute all Tier 1 features** (~112) from existing data
2. **Filter with Mutual Information** — reduce to ~80
3. **Run Boruta-SHAP** on the ~80 — keeps only "all-relevant" features
4. **Final SHAP analysis** — prune to 40-60 features based on mean |SHAP value|
5. **Stability check** — run selection on 5 non-overlapping time windows, keep only features selected in 4/5+
6. **XGBoost with constraints** — use `interaction_constraints` to prevent spurious cross-category interactions, `max_depth=4-5`, `colsample_bytree=0.6`

### Key Pitfalls to Avoid
- **Look-ahead bias:** Use `.shift(1)` after rolling calculations, expanding (not full-day) z-scores
- **Feature count:** Keep final set < sqrt(n_samples). For ~40K daily rows: max ~200 features safe. For 1-min rows (~100K): ~300 safe, but less is more.
- **Purged cross-validation:** Use Lopez de Prado's purged CV (mlfinlab), NOT standard k-fold
- **Daily leakage:** Features computed with full-day data (e.g., day-high) cannot be used intraday without expanding window

---

## ACADEMIC REFERENCES

1. Todorov & Zhang (2026). "Intraday Volatility Patterns from Short-Dated Options." *J. Econometrics*
2. Albers (2025). "Does VIX1D Render Common Vol Forecasting Models Obsolete?" *J. Futures Markets*
3. Carr & Wu (2009). "Variance Risk Premia." *Review of Financial Studies*
4. Easley, Lopez de Prado, O'Hara (2012). "Flow Toxicity and Liquidity." *J. Portfolio Mgmt*
5. Lopez de Prado (2018). *Advances in Financial Machine Learning*. Wiley.
6. Garman & Klass (1980). "On the Estimation of Security Price Volatilities."
7. Yang & Zhang (2000). "Drift-Independent Volatility Estimation."
8. Luxenberg & Boyd (2024). "EWMA Models." Stanford.
9. CBOE (2024). "0DTE Index Options and Market Volatility."
10. Goldman Sachs. "Gamma Imbalance and Absolute Returns."
