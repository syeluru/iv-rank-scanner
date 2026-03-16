"""
Hedged Iron Condor Payoff Analysis

Short IC (credit $3.15): 6430P/6450P/6755C/6775C (width $20)
Long IC (debit $2.50):   6400P/6425P/6780C/6805C (width $25)
Net credit: $0.65
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Trade Parameters ──────────────────────────────────────────────
# Short IC legs (we SELL this IC, receive credit)
short_ic = {
    'put_long': 6430,   # buy put  (delta 0.0891)
    'put_short': 6450,  # sell put (delta 0.1061)
    'call_short': 6755, # sell call (delta 0.1050)
    'call_long': 6775,  # buy call (delta 0.0735)
    'credit': 3.15,
}

# Long IC legs (we BUY this IC, pay debit)
long_ic = {
    'put_short': 6400,  # sell put (delta 0.0675)
    'put_long': 6425,   # buy put  (delta 0.0858)
    'call_long': 6780,  # buy call (delta 0.0669)
    'call_short': 6805, # sell call (delta 0.0428)
    'debit': 2.50,
}

net_credit = short_ic['credit'] - long_ic['debit']  # $0.65

# Deltas for probability estimation
deltas = {
    6400: 0.0675,
    6425: 0.0858,
    6430: 0.0891,
    6450: 0.1061,
    6755: 0.1050,
    6775: 0.0735,
    6780: 0.0669,
    6805: 0.0428,
}

# ── Payoff Function ──────────────────────────────────────────────
def payoff_at_expiry(spx: float) -> float:
    """Compute combined P&L per share at expiration."""
    # Short IC P&L (we sold it, so positive credit)
    si_put_long = max(short_ic['put_long'] - spx, 0)    # we own
    si_put_short = -max(short_ic['put_short'] - spx, 0)  # we sold
    si_call_short = -max(spx - short_ic['call_short'], 0) # we sold
    si_call_long = max(spx - short_ic['call_long'], 0)    # we own
    short_ic_pnl = si_put_long + si_put_short + si_call_short + si_call_long

    # Long IC P&L (we bought it, so negative debit)
    li_put_short = -max(long_ic['put_short'] - spx, 0)   # we sold
    li_put_long = max(long_ic['put_long'] - spx, 0)       # we own
    li_call_long = max(spx - long_ic['call_long'], 0)     # we own
    li_call_short = -max(spx - long_ic['call_short'], 0)  # we sold
    long_ic_pnl = li_put_short + li_put_long + li_call_long + li_call_short

    return short_ic_pnl + long_ic_pnl + net_credit


# ── Compute Payoff Table ─────────────────────────────────────────
strikes = sorted([
    short_ic['put_long'], short_ic['put_short'],
    short_ic['call_short'], short_ic['call_long'],
    long_ic['put_short'], long_ic['put_long'],
    long_ic['call_long'], long_ic['call_short'],
])

# Key price points for the table
key_prices = sorted(set(
    [s - 50 for s in strikes[:1]] +  # well below
    strikes +
    [s + 50 for s in strikes[-1:]] +  # well above
    # midpoints between consecutive strikes
    [(strikes[i] + strikes[i+1]) / 2 for i in range(len(strikes)-1)] +
    # middle of profit zone
    [int((short_ic['put_short'] + short_ic['call_short']) / 2)]
))

print("=" * 70)
print("HEDGED IRON CONDOR — PAYOFF AT EXPIRATION")
print("=" * 70)
print(f"\nShort IC: {short_ic['put_long']}P/{short_ic['put_short']}P/"
      f"{short_ic['call_short']}C/{short_ic['call_long']}C  credit=${short_ic['credit']:.2f}")
print(f"Long  IC: {long_ic['put_short']}P/{long_ic['put_long']}P/"
      f"{long_ic['call_long']}C/{long_ic['call_short']}C  debit=${long_ic['debit']:.2f}")
print(f"Net credit: ${net_credit:.2f}")
print(f"\nShort IC width: ${short_ic['put_short'] - short_ic['put_long']} (puts), "
      f"${short_ic['call_long'] - short_ic['call_short']} (calls)")
print(f"Long  IC width: ${long_ic['put_long'] - long_ic['put_short']} (puts), "
      f"${long_ic['call_short'] - long_ic['call_long']} (calls)")

print(f"\n{'SPX':>8}  {'P&L/share':>10}  {'P&L (1 lot)':>12}  {'Zone':>30}")
print("-" * 70)

for price in key_prices:
    pnl = payoff_at_expiry(price)
    pnl_dollar = pnl * 100  # 1 contract = 100 multiplier

    # Determine zone
    if price <= long_ic['put_short']:
        zone = "Both ICs max loss (puts)"
    elif price <= long_ic['put_long']:
        zone = "Long IC put spread active"
    elif price <= short_ic['put_long']:
        zone = "Short IC put spread losing"
    elif price <= short_ic['put_short']:
        zone = "Short IC put spread partial"
    elif price <= short_ic['call_short']:
        zone = "Profit zone (all OTM)"
    elif price <= long_ic['call_long']:
        zone = "Short IC call spread partial"
    elif price <= short_ic['call_long']:
        zone = "Both call spreads active"
    elif price <= long_ic['call_short']:
        zone = "Short IC maxed, long IC gaining"
    else:
        zone = "Both ICs max loss (calls)"

    print(f"{price:>8.0f}  {pnl:>+10.2f}  {pnl_dollar:>+12.0f}  {zone:>30}")


# ── Continuous payoff for chart ──────────────────────────────────
spx_range = np.arange(6350, 6860, 0.5)
pnl_values = np.array([payoff_at_expiry(s) for s in spx_range])

# Also compute individual IC payoffs for decomposition
def short_ic_payoff(spx):
    p = max(short_ic['put_long'] - spx, 0) - max(short_ic['put_short'] - spx, 0) \
        - max(spx - short_ic['call_short'], 0) + max(spx - short_ic['call_long'], 0)
    return p + short_ic['credit']

def long_ic_payoff(spx):
    p = -max(long_ic['put_short'] - spx, 0) + max(long_ic['put_long'] - spx, 0) \
        + max(spx - long_ic['call_long'], 0) - max(spx - long_ic['call_short'], 0)
    return p - long_ic['debit']

short_pnl = np.array([short_ic_payoff(s) for s in spx_range])
long_pnl = np.array([long_ic_payoff(s) for s in spx_range])


# ── EV Calculation using delta-implied probabilities ─────────────
print("\n" + "=" * 70)
print("EXPECTED VALUE ANALYSIS (delta-implied probabilities)")
print("=" * 70)

# Put side probabilities (P(SPX < K) ≈ put delta)
p_below_6400 = deltas[6400]       # 6.75%
p_below_6425 = deltas[6425]       # 8.58%
p_below_6430 = deltas[6430]       # 8.91%
p_below_6450 = deltas[6450]       # 10.61%

# Call side probabilities (P(SPX > K) ≈ call delta)
# Strikes in order: 6755 (short IC short), 6775 (short IC long), 6780 (long IC long), 6805 (long IC short)
p_above_6755 = deltas[6755]       # 10.50%
p_above_6775 = deltas[6775]       # 7.35%
p_above_6780 = deltas[6780]       # 6.69%
p_above_6805 = deltas[6805]       # 4.28%

# These are monotonically decreasing — good
zones = [
    ("SPX < 6400",           p_below_6400,                                    payoff_at_expiry(6380)),
    ("6400 ≤ SPX < 6425",    p_below_6425 - p_below_6400,                    payoff_at_expiry(6412.5)),
    ("6425 ≤ SPX < 6430",    p_below_6430 - p_below_6425,                    payoff_at_expiry(6427.5)),
    ("6430 ≤ SPX < 6450",    p_below_6450 - p_below_6430,                    payoff_at_expiry(6440)),
    ("6450 ≤ SPX < 6755",    1 - p_below_6450 - p_above_6755,               payoff_at_expiry(6600)),
    ("6755 ≤ SPX < 6775",    p_above_6755 - p_above_6775,                   payoff_at_expiry(6765)),
    ("6775 ≤ SPX < 6780",    p_above_6775 - p_above_6780,                   payoff_at_expiry(6777.5)),
    ("6780 ≤ SPX < 6805",    p_above_6780 - p_above_6805,                   payoff_at_expiry(6792.5)),
    ("SPX ≥ 6805",           p_above_6805,                                   payoff_at_expiry(6820)),
]

print(f"\n{'Zone':<25} {'Probability':>11} {'P&L/share':>10} {'P&L×100':>10} {'EV Contrib':>11}")
print("-" * 70)

total_ev = 0
total_prob = 0
for zone_name, prob, pnl in zones:
    ev_contrib = prob * pnl * 100
    total_ev += ev_contrib
    total_prob += prob
    print(f"{zone_name:<25} {prob:>10.2%} {pnl:>+10.2f} {pnl*100:>+10.0f} {ev_contrib:>+11.2f}")

print("-" * 70)
print(f"{'TOTAL':<25} {total_prob:>10.2%} {'':>10} {'':>10} {total_ev:>+11.2f}")
print(f"\nExpected P&L per contract: ${total_ev:+.2f}")
print(f"Expected P&L as % of net credit: {total_ev/(net_credit*100)*100:+.1f}%")

# Key metrics
max_profit_tails = payoff_at_expiry(6350) * 100
max_profit_middle = payoff_at_expiry(6600) * 100
max_loss_puts = min(payoff_at_expiry(s) for s in np.arange(6400, 6451, 1)) * 100
max_loss_calls = min(payoff_at_expiry(s) for s in np.arange(6755, 6806, 1)) * 100

print(f"\n{'KEY METRICS':}")
print(f"  Max profit (tails):     ${max_profit_tails:+.0f} per contract")
print(f"  Max profit (middle):    ${max_profit_middle:+.0f} per contract")
print(f"  Max loss (put side):    ${max_loss_puts:+.0f} per contract")
print(f"  Max loss (call side):   ${max_loss_calls:+.0f} per contract")
print(f"  Breakeven (put side):   ~{short_ic['put_short'] - net_credit:.0f}")
print(f"  Breakeven (call side):  ~{short_ic['call_short'] + net_credit:.0f}")

# Probability of profit
p_profit_middle = 1 - p_below_6450 - p_above_6755
p_profit_tails = p_below_6400 + p_above_6805

print(f"\n  P(max profit zone):     {p_profit_middle:.1%} (middle)")
print(f"  P(tail profit zone):    {p_profit_tails:.1%} (extremes)")
print(f"  P(put loss zone):       {p_below_6450 - p_below_6400:.1%}")
print(f"  P(call loss zone):      {p_above_6755 - p_above_6805:.1%}")


# ── Chart ─────────────────────────────────────────────────────────
fig = make_subplots(rows=1, cols=1)

# Combined payoff
fig.add_trace(go.Scatter(
    x=spx_range, y=pnl_values * 100,
    name='Combined (Hedged IC)',
    line=dict(color='#00ff88', width=3),
    fill='tozeroy',
    fillcolor='rgba(0,255,136,0.1)',
))

# Individual IC payoffs (dashed)
fig.add_trace(go.Scatter(
    x=spx_range, y=short_pnl * 100,
    name='Short IC only',
    line=dict(color='#ff6b6b', width=1.5, dash='dash'),
    opacity=0.7,
))

fig.add_trace(go.Scatter(
    x=spx_range, y=long_pnl * 100,
    name='Long IC only',
    line=dict(color='#4dabf7', width=1.5, dash='dash'),
    opacity=0.7,
))

# Zero line
fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)

# Strike markers
for strike in strikes:
    fig.add_vline(x=strike, line_dash="dot", line_color="gray", opacity=0.3)

# Annotations for key levels
fig.add_annotation(x=6600, y=net_credit*100 + 20, text=f"Net credit: ${net_credit:.2f}",
                   showarrow=False, font=dict(color='#00ff88', size=12))

fig.update_layout(
    title=dict(text="Hedged Iron Condor — Payoff at Expiration (per contract)",
               font=dict(size=16)),
    xaxis_title="SPX at Expiration",
    yaxis_title="P&L per Contract ($)",
    template="plotly_dark",
    height=600,
    width=1100,
    legend=dict(x=0.01, y=0.99),
    margin=dict(t=50, b=50),
    hovermode='x unified',
)

fig.write_html("output/hedged_ic_payoff.html")
print(f"\nChart saved to output/hedged_ic_payoff.html")
