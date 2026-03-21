"""
Build the target variable by simulating SPXW 0DTE Iron Condor outcomes.

Strategy simulated:
  - Enter ~9:45am (proxied by option open prices from ThetaData EOD)
  - Sell call + put at ATM ± short_offset (configurable, default 100 pts)
  - Buy call + put at ATM ± wing_offset (50 pts further, default 150 pts)
  - Hold to expiration (4pm)
  - Outcome: based on where SPX closes relative to short strikes

Target variables generated:
  ic_profitable    (1/0) — SPX closed inside short strikes (max profit day)
  ic_pnl_per_share — actual P&L per share (signed)
  ic_credit        — entry credit received per share
  ic_pnl_pct       — P&L as % of max profit (credit)
  ic_tp25_hit      — 1 if PnL >= 25% of max profit (25% TP hit, incl. partial)

Strike selection:
  ATM = round(SPX_open / 5) * 5
  short_call = ATM + short_offset
  short_put  = ATM - short_offset
  long_call  = ATM + wing_offset
  long_put   = ATM - wing_offset

NOTE: This uses EOD option open prices as a proxy for entry prices.
For higher fidelity, replace with minute-level quote data from
fetch_minute_quotes.py (requires OPTION.PRO tier on ThetaData).

Inputs:
  data/spxw_0dte_eod.parquet
  data/spx_daily.parquet

Output:
  data/target.parquet   — one row per trading day
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# ── IC configuration ──────────────────────────────────────────────────────────
# Adjust these to match the live bot's actual strike selection
SHORT_OFFSET = 100   # points from ATM to short strikes
WING_OFFSET  = 150   # points from ATM to long (wing) strikes
# ─────────────────────────────────────────────────────────────────────────────


def round_strike(price, step=5):
    return round(price / step) * step


def get_open_price(eod_day, strike, right):
    """Get the opening option price for a specific strike/right. Returns NaN if missing."""
    row = eod_day[(eod_day["strike"] == strike) & (eod_day["right"] == right)]
    if row.empty:
        return np.nan
    val = row["open"].iloc[0]
    return float(val) if val > 0 else np.nan


def simulate_ic(spx_open, spx_close, short_call_open, short_put_open,
                long_call_open, long_put_open, short_offset, wing_offset):
    """
    Simulate Iron Condor P&L.

    Entry credit = (short_call_open + short_put_open) - (long_call_open + long_put_open)

    Exit at expiration:
      Short call loss = max(0, spx_close - short_call_strike)
      Short put loss  = max(0, short_put_strike - spx_close)
      Long call gain  = max(0, spx_close - long_call_strike)
      Long put gain   = max(0, long_put_strike - spx_close)

    PnL per share = credit - (short_call_loss + short_put_loss - long_call_gain - long_put_gain)
    """
    short_call_strike = spx_open + short_offset  # approx — use ATM as base
    short_put_strike  = spx_open - short_offset
    long_call_strike  = spx_open + wing_offset
    long_put_strike   = spx_open - wing_offset

    # Entry
    if any(pd.isna(x) for x in [short_call_open, short_put_open, long_call_open, long_put_open]):
        return dict(
            ic_credit=np.nan, ic_pnl_per_share=np.nan,
            ic_pnl_pct=np.nan, ic_profitable=np.nan, ic_tp25_hit=np.nan
        )

    credit = (short_call_open + short_put_open) - (long_call_open + long_put_open)
    if credit <= 0:
        # Bad fill / data issue — skip this day
        return dict(
            ic_credit=np.nan, ic_pnl_per_share=np.nan,
            ic_pnl_pct=np.nan, ic_profitable=np.nan, ic_tp25_hit=np.nan
        )

    # Exit (at expiration — intrinsic value only)
    short_call_loss = max(0.0, spx_close - short_call_strike)
    short_put_loss  = max(0.0, short_put_strike - spx_close)
    long_call_gain  = max(0.0, spx_close - long_call_strike)
    long_put_gain   = max(0.0, long_put_strike - spx_close)

    exit_debit = short_call_loss + short_put_loss - long_call_gain - long_put_gain
    pnl = credit - exit_debit

    # Cap to [-(wing_offset - short_offset), credit] — max loss is wing width - credit
    max_loss = -(wing_offset - short_offset - credit)
    pnl = max(pnl, max_loss)

    pnl_pct = pnl / credit * 100

    return dict(
        ic_credit=round(credit, 2),
        ic_pnl_per_share=round(pnl, 2),
        ic_pnl_pct=round(pnl_pct, 2),
        ic_profitable=1 if pnl > 0 else 0,
        ic_tp25_hit=1 if pnl_pct >= 25 else 0,  # kept ≥25% of max credit
    )


def main():
    print(f"IC config: short ±{SHORT_OFFSET}pts, wings ±{WING_OFFSET}pts")
    print(f"Loading data...")

    eod = pd.read_parquet(DATA_DIR / "spxw_0dte_eod.parquet")
    spx = pd.read_parquet(DATA_DIR / "spx_daily.parquet")

    eod["date"] = pd.to_datetime(eod["date"])
    spx["date"] = pd.to_datetime(spx["date"])

    price_map_open  = spx.set_index("date")["spx_open"].to_dict()
    price_map_close = spx.set_index("date")["spx_close"].to_dict()

    trading_days = sorted(eod["date"].unique())
    print(f"  {len(trading_days)} trading days to simulate")

    rows = []
    skipped = 0

    for date in trading_days:
        spx_open  = price_map_open.get(date, np.nan)
        spx_close = price_map_close.get(date, np.nan)

        if pd.isna(spx_open) or pd.isna(spx_close):
            skipped += 1
            continue

        atm = round_strike(spx_open)

        eod_day = eod[eod["date"] == date]

        # Look up IC legs
        short_call_open = get_open_price(eod_day, atm + SHORT_OFFSET, "call")
        short_put_open  = get_open_price(eod_day, atm - SHORT_OFFSET, "put")
        long_call_open  = get_open_price(eod_day, atm + WING_OFFSET,  "call")
        long_put_open   = get_open_price(eod_day, atm - WING_OFFSET,  "put")

        result = simulate_ic(
            spx_open, spx_close,
            short_call_open, short_put_open, long_call_open, long_put_open,
            SHORT_OFFSET, WING_OFFSET
        )

        row = {
            "date": date,
            "spx_open": spx_open,
            "spx_close": spx_close,
            "atm_strike": atm,
            "short_call_strike": atm + SHORT_OFFSET,
            "short_put_strike":  atm - SHORT_OFFSET,
            "long_call_strike":  atm + WING_OFFSET,
            "long_put_strike":   atm - WING_OFFSET,
            "short_call_open": short_call_open,
            "short_put_open":  short_put_open,
            "long_call_open":  long_call_open,
            "long_put_open":   long_put_open,
        }
        row.update(result)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    print(f"\n{'='*60}")
    print(f"Simulated {len(df)} trading days ({skipped} skipped — missing SPX price)")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Summary stats
    valid = df.dropna(subset=["ic_profitable"])
    print(f"\nValid IC simulations: {len(valid)} days")
    print(f"  Win rate (profitable):  {valid['ic_profitable'].mean():.1%}")
    print(f"  TP25 hit rate:          {valid['ic_tp25_hit'].mean():.1%}")
    print(f"  Avg credit:             ${valid['ic_credit'].mean():.2f}/share")
    print(f"  Avg PnL:                ${valid['ic_pnl_per_share'].mean():.2f}/share")
    print(f"  Avg PnL %:              {valid['ic_pnl_pct'].mean():.1f}%")

    missing = df["ic_profitable"].isna().sum()
    if missing > 0:
        print(f"\n  Warning: {missing} days with no valid IC (missing option data)")

    # Class balance
    print(f"\nClass balance (ic_profitable):")
    print(f"  Win (1):  {valid['ic_profitable'].sum():3.0f} days ({valid['ic_profitable'].mean():.1%})")
    print(f"  Loss (0): {(1-valid['ic_profitable']).sum():3.0f} days ({(1-valid['ic_profitable'].mean()):.1%})")

    outfile = DATA_DIR / "target.parquet"
    df.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile} ({outfile.stat().st_size / 1e6:.2f} MB)")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
