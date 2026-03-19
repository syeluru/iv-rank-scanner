"""
Build technical indicator features on top of the merged SPX table.
Reads: data/spx_merged.parquet
Saves: data/spx_features.parquet

Features:
  - Daily aggregates: gap_pct, intraday_range, open_to_close return
  - Moving averages: 5/10/20/50/100/200 SMA, 5/20 EMA
  - RSI (14)
  - MACD (12/26/9)
  - Bollinger Bands (20,2) + bandwidth + %B
  - Yang-Zhang historical volatility: 5D/10D/20D/30D/60D
  - HV/IV ratios (Yang-Zhang vs VIX)
  - VIX SMAs + distance features
  - Volume features
  - ATR (14)
  - Efficiency ratio, choppiness index
  - Calendar features (day of week, month, etc.)
  - Intraday features (minutes to close, time sin/cos)
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def yang_zhang_vol(daily, window):
    """Yang-Zhang volatility estimator. Returns annualized vol in %."""
    log_ho = np.log(daily["d_high"] / daily["d_open"])
    log_lo = np.log(daily["d_low"] / daily["d_open"])
    log_co = np.log(daily["d_close"] / daily["d_open"])
    log_oc = np.log(daily["d_open"] / daily["d_close"].shift(1))
    log_cc = np.log(daily["d_close"] / daily["d_close"].shift(1))

    rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    cc_var = log_cc ** 2
    oc_var = log_oc ** 2

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    yz_var = oc_var.rolling(n).mean() + k * cc_var.rolling(n).mean() + (1 - k) * rs_var.rolling(n).mean()
    return np.sqrt(yz_var * 252) * 100


def main():
    print("Loading merged table...")
    df = pd.read_parquet(DATA_DIR / "spx_merged.parquet")
    print(f"  Shape: {df.shape}")

    # ═══════════════════════════════════════════════
    # STEP 1: Daily aggregates (computed once per day, broadcast to all minutes)
    # ═══════════════════════════════════════════════
    print("\nAggregating daily OHLCV...")
    agg_dict = {
        "d_open": ("open", "first"),
        "d_high": ("high", "max"),
        "d_low": ("low", "min"),
        "d_close": ("close", "last"),
    }
    if "volume" in df.columns:
        agg_dict["d_volume"] = ("volume", "sum")
    daily = df.groupby("date").agg(**agg_dict).reset_index().sort_values("date")

    # ── Price features ──
    print("Computing daily price features...")
    daily["prev_close"] = daily["d_close"].shift(1)
    daily["gap_pct"] = (daily["d_open"] - daily["prev_close"]) / daily["prev_close"] * 100
    daily["intraday_range_pct"] = (daily["d_high"] - daily["d_low"]) / daily["d_open"] * 100
    daily["open_to_close_pct"] = (daily["d_close"] - daily["d_open"]) / daily["d_open"] * 100
    daily["close_to_close_pct"] = daily["d_close"].pct_change() * 100

    # ── Moving averages ──
    print("Computing moving averages...")
    for w in [5, 10, 20, 50, 100, 200]:
        daily[f"sma_{w}"] = daily["d_close"].rolling(w).mean()
        daily[f"close_vs_sma_{w}"] = (daily["d_close"] - daily[f"sma_{w}"]) / daily[f"sma_{w}"] * 100

    # EMAs
    for w in [5, 20]:
        daily[f"ema_{w}"] = daily["d_close"].ewm(span=w, adjust=False).mean()
        daily[f"close_vs_ema_{w}"] = (daily["d_close"] - daily[f"ema_{w}"]) / daily[f"ema_{w}"] * 100

    # ── RSI (14) ──
    print("Computing RSI...")
    delta = daily["d_close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    daily["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD (12, 26, 9) ──
    print("Computing MACD...")
    ema_12 = daily["d_close"].ewm(span=12, adjust=False).mean()
    ema_26 = daily["d_close"].ewm(span=26, adjust=False).mean()
    daily["macd_line"] = ema_12 - ema_26
    daily["macd_signal"] = daily["macd_line"].ewm(span=9, adjust=False).mean()
    daily["macd_hist"] = daily["macd_line"] - daily["macd_signal"]

    # ── Bollinger Bands (20, 2) ──
    print("Computing Bollinger Bands...")
    bb_sma = daily["d_close"].rolling(20).mean()
    bb_std = daily["d_close"].rolling(20).std()
    daily["bb_upper"] = bb_sma + 2 * bb_std
    daily["bb_lower"] = bb_sma - 2 * bb_std
    daily["bb_bandwidth"] = (daily["bb_upper"] - daily["bb_lower"]) / bb_sma * 100
    daily["bb_pct_b"] = (daily["d_close"] - daily["bb_lower"]) / (daily["bb_upper"] - daily["bb_lower"])
    daily["bb_bandwidth_pctile"] = daily["bb_bandwidth"].rolling(100).rank(pct=True)

    # ── Yang-Zhang Historical Volatility ──
    print("Computing Yang-Zhang HV...")
    for w in [5, 10, 20, 30, 60]:
        daily[f"yz_hv_{w}d"] = yang_zhang_vol(daily, w)

    # ── HV/IV ratios ──
    print("Computing HV/IV ratios...")
    # Get VIX close per day (take from first row per day in merged data)
    vix_daily = df.groupby("date")["vix_close"].first().reset_index()
    daily = daily.merge(vix_daily, on="date", how="left")

    for w in [5, 10, 20, 30]:
        daily[f"hv_iv_ratio_{w}d"] = daily[f"yz_hv_{w}d"] / daily["vix_close"]

    # ── VIX features ──
    print("Computing VIX features...")
    daily["vix_sma_5"] = daily["vix_close"].rolling(5).mean()
    daily["vix_sma_10"] = daily["vix_close"].rolling(10).mean()
    daily["vix_sma_20"] = daily["vix_close"].rolling(20).mean()
    daily["vix_vs_sma_10"] = (daily["vix_close"] - daily["vix_sma_10"]) / daily["vix_sma_10"] * 100
    daily["vix_vs_sma_20"] = (daily["vix_close"] - daily["vix_sma_20"]) / daily["vix_sma_20"] * 100

    # ── Volume features ──
    if "d_volume" in daily.columns:
        print("Computing volume features...")
        daily["vol_sma_20"] = daily["d_volume"].rolling(20).mean()
        daily["vol_ratio"] = daily["d_volume"] / daily["vol_sma_20"]
    else:
        print("No volume data (index) — skipping volume features")

    # ── ATR (14) ──
    print("Computing ATR...")
    tr = pd.concat([
        daily["d_high"] - daily["d_low"],
        (daily["d_high"] - daily["d_close"].shift(1)).abs(),
        (daily["d_low"] - daily["d_close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    daily["atr_14"] = tr.rolling(14).mean()
    daily["atr_pct"] = daily["atr_14"] / daily["d_close"] * 100

    # ── Efficiency Ratio (10) ──
    print("Computing efficiency ratio...")
    net_change = (daily["d_close"] - daily["d_close"].shift(10)).abs()
    sum_changes = daily["d_close"].diff().abs().rolling(10).sum()
    daily["efficiency_ratio"] = net_change / sum_changes

    # ── Choppiness Index (14) ──
    print("Computing choppiness index...")
    atr_sum = tr.rolling(14).sum()
    high_14 = daily["d_high"].rolling(14).max()
    low_14 = daily["d_low"].rolling(14).min()
    daily["choppiness_14"] = 100 * np.log10(atr_sum / (high_14 - low_14)) / math.log10(14)

    # ── Calendar features ──
    print("Computing calendar features...")
    daily["day_of_week"] = pd.to_datetime(daily["date"]).dt.dayofweek
    daily["month"] = pd.to_datetime(daily["date"]).dt.month
    daily["is_monday"] = (daily["day_of_week"] == 0).astype(int)
    daily["is_friday"] = (daily["day_of_week"] == 4).astype(int)
    daily["is_month_end"] = pd.to_datetime(daily["date"]).dt.is_month_end.astype(int)
    daily["is_quarter_end"] = pd.to_datetime(daily["date"]).dt.is_quarter_end.astype(int)

    # Drop helper columns before merge
    daily.drop(columns=["prev_close", "vix_close"], inplace=True)

    print(f"  Daily features computed: {daily.shape[1]} columns")

    # ═══════════════════════════════════════════════
    # STEP 2: Merge daily features back to 1-min table
    # ═══════════════════════════════════════════════
    print("\nMerging daily features onto 1-min table...")
    df = df.merge(daily, on="date", how="left")
    print(f"  After daily merge: {df.shape}")

    # ═══════════════════════════════════════════════
    # STEP 3: Intraday features (per-row, time-based)
    # ═══════════════════════════════════════════════
    print("Computing intraday features...")
    df["time"] = df["datetime"].dt.time
    df["minutes_since_open"] = (df["datetime"].dt.hour * 60 + df["datetime"].dt.minute) - (9 * 60 + 30)
    df["minutes_to_close"] = (16 * 60) - (df["datetime"].dt.hour * 60 + df["datetime"].dt.minute)

    # Time cyclical encoding (sin/cos for time of day)
    total_minutes = 390  # 9:30 to 4:00
    df["time_sin"] = np.sin(2 * np.pi * df["minutes_since_open"] / total_minutes)
    df["time_cos"] = np.cos(2 * np.pi * df["minutes_since_open"] / total_minutes)

    # Drop helper
    df.drop(columns=["time"], inplace=True)

    # ═══════════════════════════════════════════════
    # STEP 4: Save
    # ═══════════════════════════════════════════════
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Final feature table: {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        print(f"  {col:30s} nulls: {null_pct:.1f}%")

    outfile = DATA_DIR / "spx_features.parquet"
    df.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile}")
    print(f"File size: {outfile.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
