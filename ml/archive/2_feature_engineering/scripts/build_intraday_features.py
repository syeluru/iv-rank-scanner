"""
Build intraday features computed per-minute from SPX 1-min bars.

These features require live data in production and change every minute.
All rolling windows are <= 30 minutes (the max lookback at 10:00 AM entry).

Output:
  data/intraday_features.parquet — one row per minute (10:00-15:00), per trading day

Features computed:
  Momentum (returns over trailing windows):
    spx_ret_5m, spx_ret_10m, spx_ret_15m, spx_ret_30m

  Intraday MAs & ratios:
    spx_vs_vwap          — price vs intraday VWAP
    spx_vs_intraday_ma5  — price vs 5-min MA
    spx_vs_intraday_ma15 — price vs 15-min MA
    spx_vs_intraday_ma30 — price vs 30-min MA

  Cross-timeframe ratios (intraday vs daily):
    spx_vs_sma_7d, spx_vs_sma_20d, spx_vs_sma_50d, spx_vs_sma_180d
    intraday_ma15_vs_sma_7d, intraday_ma15_vs_sma_20d

  Intraday volatility:
    rvol_5m, rvol_10m, rvol_15m, rvol_30m — rolling realized vol (annualized)
    rvol_intraday_vs_5d   — today's running RV vs 5-day avg daily RV
    rvol_intraday_vs_20d  — today's running RV vs 20-day avg daily RV
    garman_klass_rv_30m   — GK estimator over trailing 30 bars
    range_exhaustion      — today's range so far / 14d ATR

  Intraday range & structure:
    intraday_pctB         — where in today's range (0=low, 1=high)
    orb_range_pct         — opening range (9:30-10:00) as % of price
    orb_containment       — 1 if still within opening range, 0 if broken out
    orb_breakout_dir      — +1 above ORB high, -1 below ORB low, 0 inside

  Squeeze / vol compression:
    ttm_squeeze           — 1 if intraday BB inside Keltner (vol squeeze)
    bb_squeeze_duration   — consecutive minutes in squeeze

  IV context (from options chain — daily, but useful ratios):
    iv_rank_30d           — ATM IV percentile within last 30 trading days
    iv_rv_ratio           — ATM IV / 20d realized vol
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"

ENTRY_START = "10:00"
ENTRY_END = "15:00"


def compute_daily_rvol(daily_df, windows=[5, 10, 20]):
    """Compute daily realized vol for comparison with intraday."""
    daily_df = daily_df.sort_values("date").copy()
    log_ret = np.log(daily_df["spx_close"] / daily_df["spx_close"].shift(1))
    result = {}
    for w in windows:
        result[f"daily_rv_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)
    result["atr_14d"] = (daily_df["spx_high"] - daily_df["spx_low"]).rolling(14).mean()
    result["date"] = daily_df["date"]
    return pd.DataFrame(result)


def compute_daily_smas(daily_df, windows=[7, 20, 50, 180]):
    """Compute daily SMAs."""
    daily_df = daily_df.sort_values("date").copy()
    result = {"date": daily_df["date"].values}
    for w in windows:
        result[f"sma_{w}d"] = daily_df["spx_close"].rolling(w, min_periods=max(1, w//2)).mean().values
    return pd.DataFrame(result)


def compute_iv_rank(iv_surface_df):
    """Compute IV rank (percentile within trailing 30 trading days)."""
    if iv_surface_df is None:
        return None
    df = iv_surface_df.sort_values("date").copy()
    if "atm_iv" not in df.columns:
        return None
    df["iv_rank_30d"] = df["atm_iv"].rolling(30, min_periods=10).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x) * 100, raw=False
    )
    return df[["date", "iv_rank_30d", "atm_iv"]]


def process_day(day_bars, date, daily_rv, daily_smas, atr_14d, iv_info):
    """Compute all intraday features for one trading day."""
    day_bars = day_bars.sort_values("datetime").reset_index(drop=True).copy()
    day_bars["log_ret"] = np.log(day_bars["close"] / day_bars["close"].shift(1))

    # VWAP proxy (using typical price * volume approximation — we don't have volume,
    # so use cumulative typical price as a proxy)
    day_bars["typical"] = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3
    day_bars["cum_typical"] = day_bars["typical"].expanding().mean()

    # Intraday MAs
    day_bars["ma5"] = day_bars["close"].rolling(5, min_periods=1).mean()
    day_bars["ma15"] = day_bars["close"].rolling(15, min_periods=5).mean()
    day_bars["ma30"] = day_bars["close"].rolling(30, min_periods=10).mean()

    # Rolling realized vol (annualized, based on 1-min bars, 390 bars/day)
    for w in [5, 10, 15, 30]:
        day_bars[f"rvol_{w}m"] = day_bars["log_ret"].rolling(w, min_periods=max(2, w//2)).std() * np.sqrt(390 * 252)

    # Garman-Klass 30-min
    log_hl = np.log(day_bars["high"] / day_bars["low"])
    log_co = np.log(day_bars["close"] / day_bars["open"])
    gk_component = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    day_bars["garman_klass_rv_30m"] = gk_component.rolling(30, min_periods=10).mean().apply(
        lambda x: np.sqrt(x * 390 * 252) if x > 0 else np.nan
    )

    # Running intraday range
    day_bars["running_high"] = day_bars["high"].expanding().max()
    day_bars["running_low"] = day_bars["low"].expanding().min()
    day_bars["running_range"] = day_bars["running_high"] - day_bars["running_low"]

    # Opening range (9:30-10:00)
    orb = day_bars[day_bars["datetime"] < pd.Timestamp(f"{date} 10:00:00")]
    if len(orb) > 0:
        orb_high = orb["high"].max()
        orb_low = orb["low"].min()
        orb_range = orb_high - orb_low
    else:
        orb_high = orb_low = orb_range = np.nan

    # BB and Keltner for TTM squeeze (20-bar, intraday)
    day_bars["bb_mid"] = day_bars["close"].rolling(20, min_periods=10).mean()
    bb_std = day_bars["close"].rolling(20, min_periods=10).std()
    day_bars["bb_upper"] = day_bars["bb_mid"] + 2 * bb_std
    day_bars["bb_lower"] = day_bars["bb_mid"] - 2 * bb_std
    atr_20 = (day_bars["high"] - day_bars["low"]).rolling(20, min_periods=10).mean()
    day_bars["kc_upper"] = day_bars["bb_mid"] + 1.5 * atr_20
    day_bars["kc_lower"] = day_bars["bb_mid"] - 1.5 * atr_20

    # ── T-1 SHIFT: All rolling features use prior bar's value ──
    # At scoring time T, the current bar hasn't closed. We only have T-1.
    # Shift all pre-computed rolling features by 1 bar so each row sees
    # the value from the prior completed bar.
    shift_cols = ["cum_typical", "ma5", "ma15", "ma30",
                  "rvol_5m", "rvol_10m", "rvol_15m", "rvol_30m",
                  "garman_klass_rv_30m",
                  "running_high", "running_low", "running_range",
                  "bb_mid", "bb_upper", "bb_lower",
                  "kc_upper", "kc_lower"]
    for col in shift_cols:
        if col in day_bars.columns:
            day_bars[col] = day_bars[col].shift(1)

    # Filter to entry window
    start_ts = pd.Timestamp(f"{date} {ENTRY_START}")
    end_ts = pd.Timestamp(f"{date} {ENTRY_END}")
    window = day_bars[(day_bars["datetime"] >= start_ts) & (day_bars["datetime"] <= end_ts)].copy()

    if window.empty:
        return []

    rows = []
    squeeze_count = 0

    for _, bar in window.iterrows():
        ts = bar["datetime"]

        # Use T-1 close as "current price" — matches production where
        # the current bar hasn't closed yet at scoring time.
        idx = day_bars.index[day_bars["datetime"] == ts]
        if len(idx) > 0 and idx[0] > 0:
            price = day_bars.loc[idx[0] - 1, "close"]  # T-1 close
        else:
            price = bar["open"]  # fallback for first bar

        # Momentum
        feats = {"datetime": ts, "date": date}
        for w in [5, 10, 15, 30]:
            if len(idx) > 0:
                i = idx[0]
                # T-1 close vs T-(w+1) close
                if i >= w + 1 and i - w - 1 >= 0:
                    prev_price = day_bars.loc[i - w - 1, "close"]
                    feats[f"spx_ret_{w}m"] = (price - prev_price) / prev_price * 100 if prev_price > 0 else np.nan
                else:
                    feats[f"spx_ret_{w}m"] = np.nan
            else:
                feats[f"spx_ret_{w}m"] = np.nan

        # Intraday MAs vs price
        feats["spx_vs_vwap"] = (price - bar["cum_typical"]) / bar["cum_typical"] * 100 if bar["cum_typical"] > 0 else np.nan
        feats["spx_vs_intraday_ma5"] = (price - bar["ma5"]) / bar["ma5"] * 100 if bar["ma5"] > 0 else np.nan
        feats["spx_vs_intraday_ma15"] = (price - bar["ma15"]) / bar["ma15"] * 100 if bar["ma15"] > 0 else np.nan
        feats["spx_vs_intraday_ma30"] = (price - bar["ma30"]) / bar["ma30"] * 100 if bar["ma30"] > 0 else np.nan

        # Cross-timeframe: price vs daily SMAs
        for w in [7, 20, 50, 180]:
            sma_val = daily_smas.get(f"sma_{w}d", np.nan)
            if not pd.isna(sma_val) and sma_val > 0:
                feats[f"spx_vs_sma_{w}d"] = (price - sma_val) / sma_val * 100
            else:
                feats[f"spx_vs_sma_{w}d"] = np.nan

        # Cross-timeframe: intraday MA vs daily SMA
        for sma_w in [7, 20]:
            sma_val = daily_smas.get(f"sma_{sma_w}d", np.nan)
            if not pd.isna(bar["ma15"]) and not pd.isna(sma_val) and sma_val > 0:
                feats[f"intraday_ma15_vs_sma_{sma_w}d"] = (bar["ma15"] - sma_val) / sma_val * 100
            else:
                feats[f"intraday_ma15_vs_sma_{sma_w}d"] = np.nan

        # Realized vol
        for w in [5, 10, 15, 30]:
            feats[f"rvol_{w}m"] = bar.get(f"rvol_{w}m", np.nan)
        feats["garman_klass_rv_30m"] = bar.get("garman_klass_rv_30m", np.nan)

        # Intraday RV vs daily RV
        running_rv = bar.get("rvol_30m", np.nan)
        for dw in [5, 20]:
            drv = daily_rv.get(f"daily_rv_{dw}d", np.nan)
            if not pd.isna(running_rv) and not pd.isna(drv) and drv > 0:
                feats[f"rvol_intraday_vs_{dw}d"] = running_rv / drv
            else:
                feats[f"rvol_intraday_vs_{dw}d"] = np.nan

        # Range exhaustion
        if not pd.isna(atr_14d) and atr_14d > 0:
            feats["range_exhaustion"] = bar["running_range"] / atr_14d
        else:
            feats["range_exhaustion"] = np.nan

        # Intraday %B
        rng = bar["running_high"] - bar["running_low"]
        if rng > 0:
            feats["intraday_pctB"] = (price - bar["running_low"]) / rng
        else:
            feats["intraday_pctB"] = 0.5

        # ORB features
        if not pd.isna(orb_range) and price > 0:
            feats["orb_range_pct"] = orb_range / price * 100
            feats["orb_containment"] = 1 if orb_low <= price <= orb_high else 0
            if price > orb_high:
                feats["orb_breakout_dir"] = 1
            elif price < orb_low:
                feats["orb_breakout_dir"] = -1
            else:
                feats["orb_breakout_dir"] = 0
        else:
            feats["orb_range_pct"] = np.nan
            feats["orb_containment"] = np.nan
            feats["orb_breakout_dir"] = np.nan

        # TTM Squeeze
        in_squeeze = (not pd.isna(bar["bb_upper"]) and
                      bar["bb_upper"] < bar["kc_upper"] and
                      bar["bb_lower"] > bar["kc_lower"])
        feats["ttm_squeeze"] = 1 if in_squeeze else 0
        squeeze_count = squeeze_count + 1 if in_squeeze else 0
        feats["bb_squeeze_duration"] = squeeze_count

        # IV context
        feats["iv_rank_30d"] = iv_info.get("iv_rank_30d", np.nan) if iv_info else np.nan
        if iv_info and not pd.isna(iv_info.get("atm_iv", np.nan)):
            drv20 = daily_rv.get("daily_rv_20d", np.nan)
            if not pd.isna(drv20) and drv20 > 0:
                feats["iv_rv_ratio"] = iv_info["atm_iv"] / drv20
            else:
                feats["iv_rv_ratio"] = np.nan
        else:
            feats["iv_rv_ratio"] = np.nan

        rows.append(feats)

    return rows


def main():
    print("Building intraday features (per-minute, 10:00-15:00)...")

    spx_1m = pd.read_parquet(DATA_DIR / "spx_1min.parquet")
    spx_1m["datetime"] = pd.to_datetime(spx_1m["datetime"])
    spx_1m["date"] = spx_1m["datetime"].dt.date
    print(f"  SPX 1-min: {len(spx_1m):,} bars, {spx_1m['date'].nunique()} days")

    spx_daily = pd.read_parquet(DATA_DIR / "spx_daily.parquet")
    spx_daily["date"] = pd.to_datetime(spx_daily["date"]).dt.date
    print(f"  SPX daily: {len(spx_daily)} days")

    # Daily RV and SMAs
    daily_rv_df = compute_daily_rvol(spx_daily)
    daily_rv_df["date"] = spx_daily["date"].values
    daily_sma_df = compute_daily_smas(spx_daily)
    daily_sma_df["date"] = spx_daily["date"].values

    # IV surface for IV rank
    iv_path = DATA_DIR / "iv_surface_features.parquet"
    iv_rank_df = None
    if iv_path.exists():
        iv_surface = pd.read_parquet(iv_path)
        iv_surface["date"] = pd.to_datetime(iv_surface["date"]).dt.date
        iv_rank_df = compute_iv_rank(iv_surface)
        if iv_rank_df is not None:
            iv_rank_df["date"] = iv_rank_df["date"].values
            print(f"  IV rank: {len(iv_rank_df)} days")

    dates = sorted(spx_1m["date"].unique())
    print(f"\n  Processing {len(dates)} days...")

    all_rows = []
    for i, date in enumerate(dates):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(dates)}] {date}")

        day_bars = spx_1m[spx_1m["date"] == date]
        if len(day_bars) < 30:
            continue

        # Get daily context
        drv_row = daily_rv_df[daily_rv_df["date"] == date]
        daily_rv = drv_row.iloc[0].to_dict() if len(drv_row) > 0 else {}

        dsma_row = daily_sma_df[daily_sma_df["date"] == date]
        daily_smas = dsma_row.iloc[0].to_dict() if len(dsma_row) > 0 else {}

        atr_14d = daily_rv.get("atr_14d", np.nan)

        iv_info = None
        if iv_rank_df is not None:
            iv_row = iv_rank_df[iv_rank_df["date"] == date]
            if len(iv_row) > 0:
                iv_info = iv_row.iloc[0].to_dict()

        rows = process_day(day_bars, date, daily_rv, daily_smas, atr_14d, iv_info)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime").reset_index(drop=True)

    outfile = DATA_DIR / "intraday_features.parquet"
    df.to_parquet(outfile, index=False)

    feature_cols = [c for c in df.columns if c not in ["datetime", "date"]]
    print(f"\n{'='*60}")
    print(f"Intraday features: {len(df):,} rows x {len(feature_cols)} features")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Trading days: {df['date'].nunique()}")
    print(f"Avg rows/day: {len(df) / df['date'].nunique():.0f}")

    print(f"\nFeatures ({len(feature_cols)}):")
    for f in sorted(feature_cols):
        null_pct = df[f].isna().mean() * 100
        print(f"  {f:35s} null={null_pct:5.1f}%")

    print(f"\nSaved to {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
