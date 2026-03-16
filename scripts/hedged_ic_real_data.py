"""
Hedged Iron Condor Analysis Using Real Option Chain Data

Picks a random trading day from the 0DTE greeks dataset, constructs hedged ICs
at multiple delta levels (5, 10, 15, 20), and computes EV tables.

Structure:
  Short IC: inner, tighter wings (e.g., $20 wide)
  Long IC:  outer, wider wings (e.g., $25 wide), with a $5 gap between short IC wing and long IC wing
  Net = short IC credit - long IC debit (must be positive)
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_PATH = "data/spxw_0dte_intraday_greeks.parquet"
SHORT_IC_WIDTH = 20   # $20 wings on short IC
LONG_IC_WIDTH = 25    # $25 wings on long IC
GAP = 5              # $5 gap between short IC wing and long IC wing
ENTRY_TIME = "12:00"  # noon entry
FEE_PER_LEG = 0.65
NUM_LEGS = 8
TOTAL_FEES = FEE_PER_LEG * NUM_LEGS  # $5.20

TARGET_DELTAS = [0.05, 0.10, 0.15, 0.20]


def load_day(date):
    """Load and clean one day of option data."""
    df = pd.read_parquet(DATA_PATH, filters=[("date", "=", pd.Timestamp(date))])
    # Clean
    df = df[(df["implied_vol"] > 0) & (df["implied_vol"] < 5.0) &
            (df["underlying_price"] > 0) & ((df["bid"] > 0) | (df["ask"] > 0))]
    df["right"] = df["right"].str.upper().str[0]
    df["mid"] = np.where((df["bid"] > 0) & (df["ask"] > 0),
                         (df["bid"] + df["ask"]) / 2,
                         np.maximum(df["bid"], df["ask"]))
    df["minute_key"] = df["timestamp"].dt.strftime("%H:%M")
    return df


def find_strike_by_delta(chain, target_delta, right):
    """Find the strike closest to target |delta| for a given right (C/P)."""
    sub = chain[chain["right"] == right].copy()
    if sub.empty:
        return None, None, None, None
    sub["delta_dist"] = (sub["delta"].abs() - target_delta).abs()
    best = sub.loc[sub["delta_dist"].idxmin()]
    return best["strike"], best["delta"], best["mid"], best["bid"]


def build_hedged_ic(chain, target_delta, spx_price):
    """
    Build a hedged IC at the given target delta.

    Short IC: sell put/call at ~target_delta, buy wings SHORT_IC_WIDTH away
    Long IC:  buy put/call GAP points outside short IC wings, sell wings LONG_IC_WIDTH away from long IC long legs

    Put side:
      Long IC short put (furthest OTM) ... Long IC long put ... [GAP] ... Short IC long put ... Short IC short put
    Call side:
      Short IC short call ... Short IC long call ... [GAP] ... Long IC long call ... Long IC short call (furthest OTM)
    """
    # Short IC: find short strikes at target delta
    short_put_strike, short_put_delta, short_put_mid, _ = find_strike_by_delta(chain, target_delta, "P")
    short_call_strike, short_call_delta, short_call_mid, _ = find_strike_by_delta(chain, target_delta, "C")

    if short_put_strike is None or short_call_strike is None:
        return None

    # Short IC long wings
    si_long_put_strike = short_put_strike - SHORT_IC_WIDTH
    si_long_call_strike = short_call_strike + SHORT_IC_WIDTH

    # Long IC long legs (GAP outside short IC wings)
    li_long_put_strike = si_long_put_strike - GAP
    li_long_call_strike = si_long_call_strike + GAP

    # Long IC short legs (LONG_IC_WIDTH outside long IC long legs)
    li_short_put_strike = li_long_put_strike - LONG_IC_WIDTH
    li_short_call_strike = li_long_call_strike + LONG_IC_WIDTH

    # Look up prices for all 8 legs
    def get_leg(strike, right):
        sub = chain[(chain["strike"] == strike) & (chain["right"] == right)]
        if sub.empty:
            # Try nearest available strike
            all_strikes = chain[chain["right"] == right]["strike"].unique()
            if len(all_strikes) == 0:
                return None
            nearest = all_strikes[np.argmin(np.abs(all_strikes - strike))]
            sub = chain[(chain["strike"] == nearest) & (chain["right"] == right)]
            if sub.empty:
                return None
            strike = nearest
        row = sub.iloc[0]
        return {
            "strike": strike,
            "delta": row["delta"],
            "mid": row["mid"],
            "bid": row["bid"],
            "ask": row["ask"],
            "iv": row["implied_vol"],
        }

    legs = {
        "si_short_put": get_leg(short_put_strike, "P"),
        "si_long_put": get_leg(si_long_put_strike, "P"),
        "si_short_call": get_leg(short_call_strike, "C"),
        "si_long_call": get_leg(si_long_call_strike, "C"),
        "li_long_put": get_leg(li_long_put_strike, "P"),
        "li_short_put": get_leg(li_short_put_strike, "P"),
        "li_long_call": get_leg(li_long_call_strike, "C"),
        "li_short_call": get_leg(li_short_call_strike, "C"),
    }

    if any(v is None for v in legs.values()):
        return None

    # Short IC credit (sell short strikes, buy long wings) — use mid prices
    si_credit = (legs["si_short_put"]["mid"] + legs["si_short_call"]["mid"]
                 - legs["si_long_put"]["mid"] - legs["si_long_call"]["mid"])

    # Long IC debit (buy long legs, sell short legs)
    li_debit = (legs["li_long_put"]["mid"] + legs["li_long_call"]["mid"]
                - legs["li_short_put"]["mid"] - legs["li_short_call"]["mid"])

    net_credit = si_credit - li_debit

    return {
        "target_delta": target_delta,
        "spx_price": spx_price,
        "legs": legs,
        "si_credit": si_credit,
        "li_debit": li_debit,
        "net_credit": net_credit,
        "net_credit_after_fees": net_credit * 100 - TOTAL_FEES,
        # Strike layout
        "si_short_put": legs["si_short_put"]["strike"],
        "si_long_put": legs["si_long_put"]["strike"],
        "si_short_call": legs["si_short_call"]["strike"],
        "si_long_call": legs["si_long_call"]["strike"],
        "li_long_put": legs["li_long_put"]["strike"],
        "li_short_put": legs["li_short_put"]["strike"],
        "li_long_call": legs["li_long_call"]["strike"],
        "li_short_call": legs["li_short_call"]["strike"],
    }


def compute_payoff(ic, spx):
    """Compute per-share P&L at expiration for a given SPX price."""
    l = ic["legs"]

    # Short IC: we sold it
    si_pnl = (max(l["si_long_put"]["strike"] - spx, 0)       # long put (we own)
              - max(l["si_short_put"]["strike"] - spx, 0)     # short put (we sold)
              - max(spx - l["si_short_call"]["strike"], 0)    # short call (we sold)
              + max(spx - l["si_long_call"]["strike"], 0))    # long call (we own)

    # Long IC: we bought it
    li_pnl = (-max(l["li_short_put"]["strike"] - spx, 0)     # short put (we sold)
              + max(l["li_long_put"]["strike"] - spx, 0)      # long put (we own)
              + max(spx - l["li_long_call"]["strike"], 0)     # long call (we own)
              - max(spx - l["li_short_call"]["strike"], 0))   # short call (we sold)

    return si_pnl + li_pnl + ic["net_credit"]


def compute_ev(ic):
    """Compute EV using delta-implied probabilities from real chain data."""
    l = ic["legs"]

    # All strikes sorted
    all_strikes = sorted(set([
        l["li_short_put"]["strike"], l["li_long_put"]["strike"],
        l["si_long_put"]["strike"], l["si_short_put"]["strike"],
        l["si_short_call"]["strike"], l["si_long_call"]["strike"],
        l["li_long_call"]["strike"], l["li_short_call"]["strike"],
    ]))

    # Put side: P(SPX < K) ≈ |put delta|
    # Call side: P(SPX > K) ≈ |call delta|
    put_strikes = [s for s in all_strikes if s <= ic["si_short_put"]]
    call_strikes = [s for s in all_strikes if s >= ic["si_short_call"]]

    # Build probability boundaries
    zones = []

    # Put side zones (from lowest to short put)
    p_below = {}
    for s in put_strikes:
        # Find this strike's put delta
        for leg_name in ["li_short_put", "li_long_put", "si_long_put", "si_short_put"]:
            if l[leg_name]["strike"] == s:
                p_below[s] = abs(l[leg_name]["delta"])
                break

    # Call side zones
    p_above = {}
    for s in call_strikes:
        for leg_name in ["si_short_call", "si_long_call", "li_long_call", "li_short_call"]:
            if l[leg_name]["strike"] == s:
                p_above[s] = abs(l[leg_name]["delta"])
                break

    # Ensure monotonic
    put_sorted = sorted(p_below.items())  # ascending strike, ascending P(below)
    call_sorted = sorted(p_above.items())  # ascending strike, descending P(above)

    # Force monotonic for puts (P(below) should increase with strike)
    for i in range(1, len(put_sorted)):
        if put_sorted[i][1] < put_sorted[i-1][1]:
            put_sorted[i] = (put_sorted[i][0], put_sorted[i-1][1])

    # Force monotonic for calls (P(above) should decrease with strike)
    for i in range(1, len(call_sorted)):
        if call_sorted[i][1] > call_sorted[i-1][1]:
            call_sorted[i] = (call_sorted[i][0], call_sorted[i-1][1])

    ev_zones = []

    # Zone: below lowest strike
    if put_sorted:
        p = put_sorted[0][1]
        midpoint = put_sorted[0][0] - 20
        pnl = compute_payoff(ic, midpoint) * 100 - TOTAL_FEES
        ev_zones.append((f"SPX < {put_sorted[0][0]:.0f}", p, pnl))

    # Put inter-strike zones
    for i in range(len(put_sorted) - 1):
        s1, p1 = put_sorted[i]
        s2, p2 = put_sorted[i + 1]
        p = p2 - p1
        if p < 0:
            p = 0
        midpoint = (s1 + s2) / 2
        pnl = compute_payoff(ic, midpoint) * 100 - TOTAL_FEES
        ev_zones.append((f"{s1:.0f} ≤ SPX < {s2:.0f}", p, pnl))

    # Middle zone
    if put_sorted and call_sorted:
        p_left = put_sorted[-1][1]
        p_right = call_sorted[0][1]
        p_middle = max(0, 1 - p_left - p_right)
        midpoint = (put_sorted[-1][0] + call_sorted[0][0]) / 2
        pnl = compute_payoff(ic, midpoint) * 100 - TOTAL_FEES
        ev_zones.append((f"{put_sorted[-1][0]:.0f} ≤ SPX < {call_sorted[0][0]:.0f}", p_middle, pnl))

    # Call inter-strike zones
    for i in range(len(call_sorted) - 1):
        s1, p1 = call_sorted[i]
        s2, p2 = call_sorted[i + 1]
        p = p1 - p2
        if p < 0:
            p = 0
        midpoint = (s1 + s2) / 2
        pnl = compute_payoff(ic, midpoint) * 100 - TOTAL_FEES
        ev_zones.append((f"{s1:.0f} ≤ SPX < {s2:.0f}", p, pnl))

    # Zone: above highest strike
    if call_sorted:
        p = call_sorted[-1][1]
        midpoint = call_sorted[-1][0] + 20
        pnl = compute_payoff(ic, midpoint) * 100 - TOTAL_FEES
        ev_zones.append((f"SPX ≥ {call_sorted[-1][0]:.0f}", p, pnl))

    return ev_zones


def print_ic_analysis(ic):
    """Print full analysis for one hedged IC."""
    l = ic["legs"]

    print(f"\n{'='*80}")
    print(f"  TARGET DELTA: {ic['target_delta']:.0%}   |   SPX: {ic['spx_price']:.2f}")
    print(f"{'='*80}")

    print(f"\n  Short IC (credit ${ic['si_credit']:.2f}/share = ${ic['si_credit']*100:.0f}/contract):")
    print(f"    Buy  {l['si_long_put']['strike']:.0f}P  δ={l['si_long_put']['delta']:+.4f}  mid=${l['si_long_put']['mid']:.2f}")
    print(f"    Sell {l['si_short_put']['strike']:.0f}P  δ={l['si_short_put']['delta']:+.4f}  mid=${l['si_short_put']['mid']:.2f}")
    print(f"    Sell {l['si_short_call']['strike']:.0f}C  δ={l['si_short_call']['delta']:+.4f}  mid=${l['si_short_call']['mid']:.2f}")
    print(f"    Buy  {l['si_long_call']['strike']:.0f}C  δ={l['si_long_call']['delta']:+.4f}  mid=${l['si_long_call']['mid']:.2f}")

    print(f"\n  Long IC (debit ${ic['li_debit']:.2f}/share = ${ic['li_debit']*100:.0f}/contract):")
    print(f"    Sell {l['li_short_put']['strike']:.0f}P  δ={l['li_short_put']['delta']:+.4f}  mid=${l['li_short_put']['mid']:.2f}")
    print(f"    Buy  {l['li_long_put']['strike']:.0f}P  δ={l['li_long_put']['delta']:+.4f}  mid=${l['li_long_put']['mid']:.2f}")
    print(f"    Buy  {l['li_long_call']['strike']:.0f}C  δ={l['li_long_call']['delta']:+.4f}  mid=${l['li_long_call']['mid']:.2f}")
    print(f"    Sell {l['li_short_call']['strike']:.0f}C  δ={l['li_short_call']['delta']:+.4f}  mid=${l['li_short_call']['mid']:.2f}")

    print(f"\n  Net credit:      ${ic['net_credit']:.2f}/share = ${ic['net_credit']*100:.0f}/contract")
    print(f"  Fees (8 legs):   ${TOTAL_FEES:.2f}")
    print(f"  Net after fees:  ${ic['net_credit_after_fees']:.2f}/contract")

    # Payoff table at key strikes
    all_strikes = sorted(set([
        l["li_short_put"]["strike"], l["li_long_put"]["strike"],
        l["si_long_put"]["strike"], l["si_short_put"]["strike"],
        l["si_short_call"]["strike"], l["si_long_call"]["strike"],
        l["li_long_call"]["strike"], l["li_short_call"]["strike"],
    ]))
    key_prices = sorted(set(
        [all_strikes[0] - 30] + all_strikes +
        [(all_strikes[i] + all_strikes[i+1]) / 2 for i in range(len(all_strikes)-1)] +
        [all_strikes[-1] + 30]
    ))

    print(f"\n  {'SPX':>8}  {'P&L/sh':>8}  {'P&L/ctr':>9}  {'After fees':>11}")
    print(f"  {'-'*42}")
    for price in key_prices:
        pnl = compute_payoff(ic, price)
        pnl_ctr = pnl * 100
        pnl_af = pnl_ctr - TOTAL_FEES
        print(f"  {price:>8.0f}  {pnl:>+8.2f}  {pnl_ctr:>+9.0f}  {pnl_af:>+11.2f}")

    # EV
    ev_zones = compute_ev(ic)
    print(f"\n  {'Zone':<30} {'Prob':>7} {'P&L':>9} {'EV':>9}")
    print(f"  {'-'*58}")
    total_ev = 0
    total_p = 0
    for zone_name, prob, pnl in ev_zones:
        ev = prob * pnl
        total_ev += ev
        total_p += prob
        print(f"  {zone_name:<30} {prob:>6.2%} {pnl:>+9.0f} {ev:>+9.2f}")
    print(f"  {'-'*58}")
    print(f"  {'TOTAL':<30} {total_p:>6.2%} {'':>9} {total_ev:>+9.2f}")
    print(f"\n  ** EV per contract (after fees): ${total_ev:+.2f} **")

    return total_ev


def make_chart(ics, date_str):
    """Create combined payoff chart for all delta levels."""
    colors = ['#ff6b6b', '#ffa726', '#66bb6a', '#42a5f5']
    fig = go.Figure()

    for ic, color in zip(ics, colors):
        all_strikes = sorted(set([
            ic["legs"][k]["strike"] for k in ic["legs"]
        ]))
        lo = all_strikes[0] - 50
        hi = all_strikes[-1] + 50
        spx_range = np.arange(lo, hi, 0.5)
        pnl_values = np.array([compute_payoff(ic, s) * 100 - TOTAL_FEES for s in spx_range])

        fig.add_trace(go.Scatter(
            x=spx_range, y=pnl_values,
            name=f'{ic["target_delta"]:.0%} delta (net ${ic["net_credit"]:.2f})',
            line=dict(color=color, width=2.5),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
    fig.update_layout(
        title=dict(text=f"Hedged Iron Condor Payoff — {date_str} (after fees)",
                   font=dict(size=16)),
        xaxis_title="SPX at Expiration",
        yaxis_title="P&L per Contract ($)",
        template="plotly_dark",
        height=600, width=1100,
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified',
    )
    fig.write_html("output/hedged_ic_real_data.html")
    print(f"\nChart saved to output/hedged_ic_real_data.html")


def main():
    # Pick a random recent trading day (or accept one as argument)
    print("Loading dataset metadata...")
    dates_df = pd.read_parquet(DATA_PATH, columns=["date"])
    all_dates = sorted(dates_df["date"].unique())
    # Pick from the middle-to-recent range for representative data
    np.random.seed(42)
    if len(sys.argv) > 1:
        target_date = pd.Timestamp(sys.argv[1])
    else:
        # Pick a random date from the last year of data
        recent_dates = [d for d in all_dates if d >= pd.Timestamp("2025-06-01")]
        if not recent_dates:
            recent_dates = all_dates[-100:]
        target_date = np.random.choice(recent_dates)

    date_str = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    print(f"Selected date: {date_str}")
    print(f"Loading option chain for {date_str}...")

    df = load_day(target_date)
    print(f"  Loaded {len(df):,} rows")

    # Filter to entry time
    entry = df[df["minute_key"] == ENTRY_TIME]
    if entry.empty:
        # Try nearest time
        times = sorted(df["minute_key"].unique())
        nearest = min(times, key=lambda t: abs(pd.Timestamp(f"2000-01-01 {t}") - pd.Timestamp(f"2000-01-01 {ENTRY_TIME}")))
        print(f"  No data at {ENTRY_TIME}, using {nearest}")
        entry = df[df["minute_key"] == nearest]

    spx_price = entry["underlying_price"].iloc[0]
    print(f"  SPX at entry: {spx_price:.2f}")
    print(f"  Available strikes: {entry['strike'].nunique()} puts, calls")

    # Build hedged ICs at each delta level
    results = []
    for target_delta in TARGET_DELTAS:
        ic = build_hedged_ic(entry, target_delta, spx_price)
        if ic is None:
            print(f"\n  *** Could not build IC at {target_delta:.0%} delta — skipping ***")
            continue

        if ic["net_credit"] <= 0:
            print(f"\n  *** {target_delta:.0%} delta IC has negative net credit "
                  f"(${ic['net_credit']:.2f}) — skipping ***")
            continue

        ev = print_ic_analysis(ic)
        results.append(ic)

    # Summary table
    if results:
        print(f"\n\n{'='*80}")
        print(f"  SUMMARY — {date_str} — SPX {spx_price:.2f}")
        print(f"{'='*80}")
        print(f"\n  {'Delta':>6} {'SI Credit':>10} {'LI Debit':>10} {'Net Cr':>8} "
              f"{'After Fee':>10} {'Max Loss':>9} {'P(mid)':>8}")
        print(f"  {'-'*64}")

        for ic in results:
            all_strikes = sorted(set([ic["legs"][k]["strike"] for k in ic["legs"]]))
            max_loss = min(compute_payoff(ic, s) * 100 - TOTAL_FEES
                          for s in np.arange(all_strikes[0], all_strikes[-1], 1))
            # P(middle) = 1 - P(below short put) - P(above short call)
            p_mid = 1 - abs(ic["legs"]["si_short_put"]["delta"]) - abs(ic["legs"]["si_short_call"]["delta"])

            print(f"  {ic['target_delta']:>5.0%} "
                  f" ${ic['si_credit']*100:>7.0f} "
                  f" ${ic['li_debit']*100:>7.0f} "
                  f" ${ic['net_credit']*100:>5.0f} "
                  f" ${ic['net_credit_after_fees']:>8.2f} "
                  f" ${max_loss:>7.0f} "
                  f" {p_mid:>7.1%}")

        make_chart(results, date_str)


if __name__ == "__main__":
    main()
