"""
Build V1 target for 0DTE SPX Iron Condor ML models.

For each entry (every 1 min from 9:30-15:00 ET, 9 delta strategies):
  - Delta-match strikes fresh at entry time (locked for the trade)
  - Credit = IC mid value at entry (what you'd collect selling it)
  - Walk forward to 15:00 for NINE separate TP vs SL races:
      TP10 vs SL, TP15 vs SL, ... TP50 vs SL — whichever hits first wins
  - SL = 2× credit (debit >= SL → loss)
  - TP = credit × (1 - pct) (debit <= TP → win)
  - SL checked first each bar (SL wins ties)
  - If neither TP nor SL hits by 3pm: check final debit vs credit
      profit → 1 (close_win), loss → 0 (close_loss)
  - Entry window generates rows from 9:30 but training filters to 10:00+
  - Last entry is 14:59 (15:00 is close only, no new trades)

Output columns per TP level (9 levels × 4 cols = 36):
  tp{pct}_target (0/1), tp{pct}_exit_reason, tp{pct}_exit_time, tp{pct}_exit_debit

Inputs:
  data/spxw_0dte_intraday_greeks.parquet — bid/ask per strike per minute

Output:
  data/target_v1.parquet — one row per strategy per 1-min entry per day
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# ── Exit parameters ───────────────────────────────────────────────────────
SL_MULT = 2.0                     # SL at 2× credit
TP_LEVELS = [i / 100 for i in range(10, 55, 5)]  # TP10, TP15, ..., TP50
CLOSE_TIME = "15:00"

# ── Entry window ──────────────────────────────────────────────────────────
ENTRY_START = "09:30"
ENTRY_END = "15:00"
INTERVAL_MINUTES = 1

# ── Strategy universe (5d to 45d in steps of 5, all $25 wings) ───────────
STRATEGIES = [
    {"name": f"IC_{d:02d}d_25w", "short_delta": d / 100, "wing_width": 25}
    for d in range(5, 50, 5)
]


def find_strike_by_delta(greeks_at_time, target_delta, right):
    """Find strike closest to target_delta. For puts, target_delta should be negative."""
    subset = greeks_at_time[greeks_at_time["right"] == right].copy()
    if subset.empty:
        return None

    if right == "call":
        subset = subset[(subset["delta"] > 0) & (subset["delta"] < 1) & (subset["bid"] > 0)]
    else:
        subset = subset[(subset["delta"] < 0) & (subset["delta"] > -1) & (subset["bid"] > 0)]

    if subset.empty:
        return None

    subset = subset.copy()
    subset["delta_diff"] = (subset["delta"] - target_delta).abs()
    best = subset.loc[subset["delta_diff"].idxmin()]
    return best


def find_strike_by_price(greeks_at_time, target_strike, right):
    """Find the option row closest to a specific strike price."""
    subset = greeks_at_time[greeks_at_time["right"] == right].copy()
    if subset.empty:
        return None

    subset = subset[subset["bid"] > 0].copy()
    if subset.empty:
        return None

    subset["strike_diff"] = (subset["strike"] - target_strike).abs()
    best = subset.loc[subset["strike_diff"].idxmin()]

    if best["strike_diff"] > 10:
        return None
    return best


def build_iron_condor(greeks_now, strategy, underlying):
    """Build an IC using mid prices for entry (assume fill at mid)."""
    delta = strategy["short_delta"]
    wing = strategy["wing_width"]

    short_put = find_strike_by_delta(greeks_now, -delta, "put")
    short_call = find_strike_by_delta(greeks_now, delta, "call")
    if short_put is None or short_call is None:
        return None

    if not (short_put["strike"] < underlying < short_call["strike"]):
        return None

    long_put = find_strike_by_price(greeks_now, short_put["strike"] - wing, "put")
    long_call = find_strike_by_price(greeks_now, short_call["strike"] + wing, "call")
    if long_put is None or long_call is None:
        return None

    # Entry assumes fill at mid price
    sc_mid = (short_call["bid"] + short_call["ask"]) / 2
    sp_mid = (short_put["bid"] + short_put["ask"]) / 2
    lc_mid = (long_call["bid"] + long_call["ask"]) / 2
    lp_mid = (long_put["bid"] + long_put["ask"]) / 2
    credit = (sc_mid + sp_mid) - (lc_mid + lp_mid)

    if credit <= 0:
        return None

    call_wing = long_call["strike"] - short_call["strike"]
    put_wing = short_put["strike"] - long_put["strike"]
    if call_wing <= 0 or put_wing <= 0:
        return None

    return {
        "short_call": short_call["strike"],
        "short_put": short_put["strike"],
        "long_call": long_call["strike"],
        "long_put": long_put["strike"],
        "sc_delta": short_call["delta"],
        "sp_delta": short_put["delta"],
        "sc_iv": short_call["implied_vol"],
        "sp_iv": short_put["implied_vol"],
        "credit": credit,
        "call_wing": call_wing,
        "put_wing": put_wing,
    }


def simulate_trade(trade, price_index, entry_ts, close_ts):
    """Walk forward for 9 separate TP scenarios, each a race against SL.

    For each TP level (10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%):
      - Walk entry+1 through 3:00 PM
      - SL checked first each bar: debit >= 2× credit
      - TP checked second: debit <= credit × (1 - tp_pct)
      - First one to trigger wins the race
      - If neither by close: check final debit vs credit → win or loss

    Returns dict with per-scenario targets and flags, or None if no data.
    """
    credit = trade["credit"]
    sl_debit = credit * SL_MULT

    sc = price_index.get(("call", trade["short_call"]))
    sp = price_index.get(("put", trade["short_put"]))
    lc = price_index.get(("call", trade["long_call"]))
    lp = price_index.get(("put", trade["long_put"]))

    if any(x is None for x in [sc, sp, lc, lp]):
        return None

    combined = pd.DataFrame({"sc": sc, "sp": sp, "lc": lc, "lp": lp})
    combined = combined[(combined.index > entry_ts) & (combined.index <= close_ts)]
    combined = combined.dropna()

    if combined.empty:
        return None

    combined["debit"] = (combined["sc"] + combined["sp"]) - (combined["lc"] + combined["lp"])

    # Run 3 separate races
    result = {}
    debit_arr = combined["debit"].values
    ts_arr = combined.index.values

    for pct in TP_LEVELS:
        key = f"tp{int(pct * 100)}"
        tp_debit = credit * (1 - pct)
        resolved = False

        for j in range(len(debit_arr)):
            d = debit_arr[j]
            t = ts_arr[j]

            # SL check first
            if d >= sl_debit:
                result[f"{key}_target"] = 0
                result[f"{key}_exit_reason"] = "sl"
                result[f"{key}_exit_time"] = str(t)
                result[f"{key}_exit_debit"] = round(float(d), 4)
                resolved = True
                break

            # TP check
            if d <= tp_debit:
                result[f"{key}_target"] = 1
                result[f"{key}_exit_reason"] = "tp"
                result[f"{key}_exit_time"] = str(t)
                result[f"{key}_exit_debit"] = round(float(d), 4)
                resolved = True
                break

        # Neither hit — evaluate at close
        if not resolved:
            last_d = round(float(debit_arr[-1]), 4)
            last_t = ts_arr[-1]
            if last_d < round(credit, 4):
                result[f"{key}_target"] = 1
                result[f"{key}_exit_reason"] = "close_win"
            else:
                result[f"{key}_target"] = 0
                result[f"{key}_exit_reason"] = "close_loss"
            result[f"{key}_exit_time"] = str(last_t)
            result[f"{key}_exit_debit"] = round(last_d, 4)

    time_in_trade = int((pd.Timestamp(ts_arr[-1]) - entry_ts).total_seconds() / 60)
    result["credit"] = round(credit, 4)
    result["time_to_close_min"] = time_in_trade

    return result


def simulate_day(day_greeks, date):
    """Simulate all IC strategies at every 1-min entry for one trading day."""
    rows = []

    day_greeks = day_greeks.copy()
    day_greeks["mid"] = (day_greeks["bid"] + day_greeks["ask"]) / 2

    # Pre-index: (right, strike) → Series of mid prices indexed by timestamp
    price_index = {}
    for (right, strike), group in day_greeks.groupby(["right", "strike"]):
        ts_series = group.set_index("timestamp")["mid"].sort_index()
        price_index[(right, strike)] = ts_series

    entry_start_ts = pd.Timestamp(f"{date} {ENTRY_START}")
    entry_end_ts = pd.Timestamp(f"{date} {ENTRY_END}")
    close_ts = pd.Timestamp(f"{date} {CLOSE_TIME}")

    all_timestamps = sorted(day_greeks["timestamp"].unique())
    entry_timestamps = [
        pd.Timestamp(t) for t in all_timestamps
        if entry_start_ts <= pd.Timestamp(t) <= entry_end_ts
    ]

    seen_minutes = set()
    filtered_ts = []
    for ts in entry_timestamps:
        minute_key = ts.strftime("%H:%M")
        if minute_key not in seen_minutes:
            seen_minutes.add(minute_key)
            filtered_ts.append(ts)

    for ts in filtered_ts:
        greeks_now = day_greeks[day_greeks["timestamp"] == ts]
        if len(greeks_now) < 10:
            continue

        underlying = greeks_now["underlying_price"].median()

        for strategy in STRATEGIES:
            trade = build_iron_condor(greeks_now, strategy, underlying)
            if trade is None:
                continue

            outcome = simulate_trade(trade, price_index, ts, close_ts)
            if outcome is None:
                continue

            rows.append({
                "datetime": ts,
                "date": date,
                "strategy": strategy["name"],
                "spx_at_entry": round(underlying, 2),
                "short_call": trade["short_call"],
                "short_put": trade["short_put"],
                "long_call": trade["long_call"],
                "long_put": trade["long_put"],
                "call_wing_width": trade["call_wing"],
                "put_wing_width": trade["put_wing"],
                "sc_delta": round(trade["sc_delta"], 4),
                "sp_delta": round(trade["sp_delta"], 4),
                "sc_iv": round(trade["sc_iv"], 4),
                "sp_iv": round(trade["sp_iv"], 4),
                **outcome,
            })

    return rows


def main():
    print("Building V1 target: Iron Condors, TP vs SL race logic")
    print(f"  SL: {SL_MULT:.0f}× credit")
    print(f"  TP levels: {', '.join(f'{int(p*100)}%' for p in TP_LEVELS)}")
    print(f"  Close: {CLOSE_TIME} ET")
    print(f"  Entry: {ENTRY_START}-{ENTRY_END} ET, every {INTERVAL_MINUTES} min")
    print(f"  Strategies: {len(STRATEGIES)} ({', '.join(s['name'] for s in STRATEGIES)})")
    print()

    greeks_file = DATA_DIR / "spxw_0dte_intraday_greeks.parquet"
    if not greeks_file.exists():
        print("ERROR: Missing greeks file. Run fetch scripts first.")
        return

    greeks_dates_df = pq.read_table(greeks_file, columns=["date"]).to_pandas()
    greeks_dates_df["date"] = pd.to_datetime(greeks_dates_df["date"]).dt.date
    greeks_dates = sorted(greeks_dates_df["date"].unique())
    del greeks_dates_df
    print(f"  Greeks: {len(greeks_dates)} dates available")

    # Resume support
    outfile = DATA_DIR / "target_v1.parquet"
    existing_dates = set()
    all_rows = []

    if outfile.exists():
        existing_df = pd.read_parquet(outfile)
        if "tp10_target" in existing_df.columns:
            existing_dates = set(pd.to_datetime(existing_df["date"]).dt.date)
            all_rows = existing_df.to_dict("records")
            print(f"  Resuming: {len(existing_dates)} dates already done")
        else:
            print("  Old schema detected — starting fresh")

    remaining = [d for d in greeks_dates if d not in existing_dates]
    print(f"  Remaining: {len(remaining)} dates to process\n")

    if not remaining:
        print("All dates already processed!")
        return

    total = len(remaining)
    errors = []
    save_every = 25

    for i, date in enumerate(remaining):
        pct = (i + 1) / total * 100
        sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {date}...")
        sys.stdout.flush()

        try:
            date_ts = pd.Timestamp(date)
            day_greeks = pq.read_table(
                greeks_file,
                filters=[("date", "=", date_ts)]
            ).to_pandas()

            if len(day_greeks) == 0:
                continue

            day_greeks["timestamp"] = pd.to_datetime(day_greeks["timestamp"])

            day_rows = simulate_day(day_greeks, date)
            all_rows.extend(day_rows)

            sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {date}: {len(day_rows)} entries    ")

        except Exception as e:
            errors.append((date, str(e)))
            sys.stdout.write(f"\r[{i+1}/{total}] ({pct:.1f}%) {date}: ERROR - {str(e)[:60]}    ")

        if (i + 1) % save_every == 0 or (i + 1) == total:
            if all_rows:
                df = pd.DataFrame(all_rows)
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values(["datetime", "strategy"]).reset_index(drop=True)
                df.to_parquet(outfile, index=False)
                sys.stdout.write(f" [saved {len(df):,} rows]")
            sys.stdout.write("\n")

    # Final stats
    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["datetime", "strategy"]).reset_index(drop=True)
    df.to_parquet(outfile, index=False)

    print(f"\n{'='*60}")
    print(f"Generated {len(df):,} rows")
    print(f"  Trading days: {df['date'].nunique()}")
    print(f"  Strategies: {df['strategy'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Avg rows/day: {len(df) / df['date'].nunique():.0f}")

    for tp in TP_LEVELS:
        key = f"tp{int(tp*100)}"
        print(f"\n{key.upper()} scenario:")
        print(f"  Target=1 (enter): {df[f'{key}_target'].mean():.1%}")
        print(f"  Exit reasons:")
        for reason in df[f"{key}_exit_reason"].value_counts().index:
            n = (df[f"{key}_exit_reason"] == reason).sum()
            print(f"    {reason:12s} {n:>8,} ({n/len(df):.1%})")

    print(f"\nPer strategy:")
    for strat in sorted(df["strategy"].unique()):
        s = df[df["strategy"] == strat]
        print(f"  {strat:15s}  n={len(s):7,}  "
              f"tp10={s['tp10_target'].mean():.1%}  "
              f"tp25={s['tp25_target'].mean():.1%}  "
              f"tp50={s['tp50_target'].mean():.1%}  "
              f"avg_credit=${s['credit'].mean():.2f}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for d, err in errors[:10]:
            print(f"  {d}: {err}")

    print(f"\nSaved to {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
