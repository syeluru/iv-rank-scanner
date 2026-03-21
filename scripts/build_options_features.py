"""
Build per-day options-derived features from ThetaData.

Inputs:
  data/spxw_0dte_eod.parquet       — SPXW option chain EOD (bid/ask/OHLC per strike)
  data/spxw_0dte_oi.parquet        — Open interest per strike
  data/spxw_term_structure.parquet — ATM term structure (bid/ask at multiple DTEs)
  data/spx_daily.parquet           — SPX daily OHLC (for ATM reference)

Output:
  data/options_features.parquet    — One row per trading day

Features:
  Volume / OI:
    total_call_volume, total_put_volume, pc_volume_ratio
    total_call_oi, total_put_oi, pc_oi_ratio
    oi_concentration (Herfindahl-like index of OI across strikes)

  Max Pain:
    max_pain_strike             — strike minimizing option holder payout
    max_pain_distance_pts       — ATM - max_pain (signed)
    max_pain_distance_pct       — as % of SPX open

  ATM pricing:
    atm_call_open, atm_put_open     — opening option prices at ATM
    atm_call_close, atm_put_close   — closing option prices at ATM
    atm_straddle_open               — call_open + put_open (same-day IV proxy)

  Chain skew:
    skew_25    — put_mid_25otm / call_mid_25otm  (25-pt wings)
    skew_50    — put_mid_50otm / call_mid_50otm  (50-pt wings)
    skew_100   — put_mid_100otm / call_mid_100otm (100-pt wings)

  IC credit approximation (ATM ± 100/150 pt wings):
    ic_credit_100   — approx IC credit entering at ±100/150 strikes
    ic_credit_150   — approx IC credit entering at ±150/200 strikes

  Term structure (if available):
    ts_0dte_atm_mid    — 0DTE ATM mid price
    ts_7dte_atm_mid    — ~7DTE ATM mid
    ts_30dte_atm_mid   — ~30DTE ATM mid
    ts_slope_0_30      — log(ts_30 / ts_0) — term structure slope
    ts_slope_0_7       — log(ts_7 / ts_0)
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def round_strike(price, step=5):
    return round(price / step) * step


def compute_max_pain(eod_day, spx_open):
    """
    Max pain: strike where total option-holder payout (at expiry) is minimized.
    Returns (max_pain_strike, distance_pts, distance_pct).
    """
    strikes = sorted(eod_day["strike"].unique())
    if not strikes:
        return np.nan, np.nan, np.nan

    call_oi = eod_day[eod_day["right"] == "call"].groupby("strike")["open_interest"].sum()
    put_oi = eod_day[eod_day["right"] == "put"].groupby("strike")["open_interest"].sum()

    min_pain = np.inf
    max_pain_strike = np.nan

    for s in strikes:
        # Payout if expires at strike s
        call_pain = sum(
            oi * max(0, k - s)
            for k, oi in call_oi.items()
        )
        put_pain = sum(
            oi * max(0, s - k)
            for k, oi in put_oi.items()
        )
        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            max_pain_strike = s

    dist_pts = spx_open - max_pain_strike
    dist_pct = dist_pts / spx_open * 100 if spx_open > 0 else np.nan
    return max_pain_strike, dist_pts, dist_pct


def get_option_price(eod_day, strike, right, price_col="open"):
    """Get option price at a specific strike and right. Returns NaN if not found."""
    row = eod_day[(eod_day["strike"] == strike) & (eod_day["right"] == right)]
    if row.empty:
        return np.nan
    val = row[price_col].iloc[0]
    return val if val > 0 else np.nan


def get_mid_price(eod_day, strike, right):
    """Get bid/ask mid at a strike."""
    row = eod_day[(eod_day["strike"] == strike) & (eod_day["right"] == right)]
    if row.empty:
        return np.nan
    bid = row["bid"].iloc[0]
    ask = row["ask"].iloc[0]
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return np.nan


def compute_ic_credit(eod_day, atm, short_offset, wing_offset, price_col="open"):
    """
    Approximate Iron Condor credit:
    - Sell call at ATM + short_offset, buy call at ATM + wing_offset
    - Sell put at ATM - short_offset, buy put at ATM - wing_offset
    Returns credit per share (not per contract). NaN if any leg missing.
    """
    short_call = get_option_price(eod_day, atm + short_offset, "call", price_col)
    long_call  = get_option_price(eod_day, atm + wing_offset,  "call", price_col)
    short_put  = get_option_price(eod_day, atm - short_offset, "put",  price_col)
    long_put   = get_option_price(eod_day, atm - wing_offset,  "put",  price_col)

    if any(pd.isna(x) for x in [short_call, long_call, short_put, long_put]):
        return np.nan
    if any(x <= 0 for x in [short_call, long_call, short_put, long_put]):
        return np.nan

    credit = (short_call + short_put) - (long_call + long_put)
    return credit if credit > 0 else np.nan


def compute_term_structure_features(ts_day):
    """
    From the term structure for a single trade date, extract ATM mid prices
    at different DTE buckets.
    Returns dict of features.
    """
    feats = {}

    if ts_day.empty:
        return {
            "ts_0dte_atm_mid": np.nan, "ts_7dte_atm_mid": np.nan,
            "ts_30dte_atm_mid": np.nan, "ts_slope_0_30": np.nan,
            "ts_slope_0_7": np.nan,
        }

    # Use ATM strike (moneyness closest to 0) and mid price
    for target_dte, label in [(0, "0dte"), (7, "7dte"), (30, "30dte")]:
        bucket = ts_day[(ts_day["dte"] >= target_dte - 2) & (ts_day["dte"] <= target_dte + 5)]
        if bucket.empty:
            feats[f"ts_{label}_atm_mid"] = np.nan
        else:
            # Pick row closest to ATM (moneyness ~ 0) at nearest DTE
            bucket = bucket.copy()
            bucket["abs_moneyness"] = bucket["moneyness"].abs()
            bucket = bucket.sort_values(["abs_moneyness", "dte"])
            feats[f"ts_{label}_atm_mid"] = bucket["mid"].iloc[0]

    # Term structure slopes
    ts0  = feats.get("ts_0dte_atm_mid", np.nan)
    ts7  = feats.get("ts_7dte_atm_mid", np.nan)
    ts30 = feats.get("ts_30dte_atm_mid", np.nan)

    feats["ts_slope_0_30"] = (
        np.log(ts30 / ts0) if (ts0 and ts30 and ts0 > 0 and ts30 > 0) else np.nan
    )
    feats["ts_slope_0_7"] = (
        np.log(ts7 / ts0) if (ts0 and ts7 and ts0 > 0 and ts7 > 0) else np.nan
    )
    return feats


def main():
    print("Loading data...")
    eod = pd.read_parquet(DATA_DIR / "spxw_0dte_eod.parquet")
    oi  = pd.read_parquet(DATA_DIR / "spxw_0dte_oi.parquet")
    spx = pd.read_parquet(DATA_DIR / "spx_daily.parquet")

    eod["date"] = pd.to_datetime(eod["date"])
    oi["date"]  = pd.to_datetime(oi["date"])
    spx["date"] = pd.to_datetime(spx["date"])

    # Merge OI into EOD
    eod = eod.merge(
        oi[["date", "expiration", "strike", "right", "open_interest"]],
        on=["date", "expiration", "strike", "right"], how="left"
    )
    eod["open_interest"] = eod["open_interest"].fillna(0)

    # Load term structure if available
    ts_file = DATA_DIR / "spxw_term_structure.parquet"
    has_ts = ts_file.exists()
    if has_ts:
        ts = pd.read_parquet(ts_file)
        ts["date"] = pd.to_datetime(ts["date"])
        print(f"  Term structure: {ts.shape}")
    else:
        ts = pd.DataFrame()
        print("  Term structure: not found, skipping")

    # Price map
    price_map = spx.set_index("date")["spx_open"].to_dict()

    trading_days = sorted(eod["date"].unique())
    print(f"  EOD data: {len(eod)} rows, {len(trading_days)} trading days")

    rows = []
    for i, date in enumerate(trading_days):
        if (i + 1) % 50 == 0:
            print(f"  Processing day {i+1}/{len(trading_days)}...")

        spx_open = price_map.get(date, np.nan)
        if pd.isna(spx_open):
            continue

        atm = round_strike(spx_open)
        eod_day = eod[eod["date"] == date]
        ts_day  = ts[ts["date"] == date] if has_ts else pd.DataFrame()

        row = {"date": date, "spx_open": spx_open, "atm_strike": atm}

        # ── Volume & OI ──
        calls = eod_day[eod_day["right"] == "call"]
        puts  = eod_day[eod_day["right"] == "put"]

        row["total_call_volume"] = calls["volume"].sum()
        row["total_put_volume"]  = puts["volume"].sum()
        tot_vol = row["total_call_volume"] + row["total_put_volume"]
        row["pc_volume_ratio"] = (
            row["total_put_volume"] / row["total_call_volume"]
            if row["total_call_volume"] > 0 else np.nan
        )

        row["total_call_oi"] = calls["open_interest"].sum()
        row["total_put_oi"]  = puts["open_interest"].sum()
        row["pc_oi_ratio"] = (
            row["total_put_oi"] / row["total_call_oi"]
            if row["total_call_oi"] > 0 else np.nan
        )

        # OI concentration (Herfindahl index across strikes — higher = more concentrated)
        total_oi = eod_day["open_interest"].sum()
        if total_oi > 0:
            strike_oi_shares = eod_day.groupby("strike")["open_interest"].sum() / total_oi
            row["oi_concentration"] = (strike_oi_shares ** 2).sum()
        else:
            row["oi_concentration"] = np.nan

        # ── Max Pain ──
        mp_strike, mp_dist_pts, mp_dist_pct = compute_max_pain(eod_day, spx_open)
        row["max_pain_strike"]       = mp_strike
        row["max_pain_distance_pts"] = mp_dist_pts
        row["max_pain_distance_pct"] = mp_dist_pct

        # ── ATM pricing ──
        row["atm_call_open"]  = get_option_price(eod_day, atm, "call", "open")
        row["atm_put_open"]   = get_option_price(eod_day, atm, "put",  "open")
        row["atm_call_close"] = get_option_price(eod_day, atm, "call", "close")
        row["atm_put_close"]  = get_option_price(eod_day, atm, "put",  "close")

        atm_call_open = row["atm_call_open"]
        atm_put_open  = row["atm_put_open"]
        row["atm_straddle_open"] = (
            atm_call_open + atm_put_open
            if not (pd.isna(atm_call_open) or pd.isna(atm_put_open)) else np.nan
        )

        # ── Chain skew (put/call ratio at equal distance from ATM) ──
        for offset in [25, 50, 100]:
            call_mid = get_mid_price(eod_day, atm + offset, "call")
            put_mid  = get_mid_price(eod_day, atm - offset, "put")
            row[f"call_mid_{offset}otm"] = call_mid
            row[f"put_mid_{offset}otm"]  = put_mid
            row[f"skew_{offset}"] = (
                put_mid / call_mid
                if (call_mid and put_mid and call_mid > 0) else np.nan
            )

        # ── IC credit approximation ──
        # IC_100: short ±100 from ATM, wings ±150
        row["ic_credit_100"] = compute_ic_credit(eod_day, atm, 100, 150, "open")
        # IC_150: short ±150 from ATM, wings ±200
        row["ic_credit_150"] = compute_ic_credit(eod_day, atm, 150, 200, "open")
        # IC_50: tighter, short ±50, wings ±100
        row["ic_credit_50"]  = compute_ic_credit(eod_day, atm, 50,  100, "open")

        # ── Term structure ──
        ts_feats = compute_term_structure_features(ts_day)
        row.update(ts_feats)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    print(f"\n{'='*60}")
    print(f"Options features: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    null_pcts = df.isna().mean() * 100
    print("\nNull rates:")
    for col in df.columns:
        if null_pcts[col] > 5:
            print(f"  {col:35s} {null_pcts[col]:.1f}%")

    outfile = DATA_DIR / "options_features.parquet"
    df.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile} ({outfile.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
