"""
Live feature vector builder for ZDOM V1.

Combines:
  1. Daily features (from most recent parquets - 239 features)
  2. Intraday features (computed from accumulated 1-min bars - 30 features)
  3. Entry features (strategy one-hots, timing, credit - 15 features)

Total: 284 features matching the trained model.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"


class LiveFeatureBuilder:
    """Builds feature vectors for live scoring."""

    def __init__(self, data_dir=None):
        if data_dir is not None:
            global DATA_DIR
            DATA_DIR = Path(data_dir)
        self.daily_features = None
        self.daily_smas = {}
        self.daily_rvols = {}
        self.atr_14d = None
        self.bars = []  # accumulated 1-min bars today
        self.orb_high = None
        self.orb_low = None
        self.orb_range = None
        self.squeeze_count = 0

    def load_daily_features(self):
        """Load the most recent daily features from all parquets."""
        daily = {}

        # SPX daily features
        try:
            df = pd.read_parquet(DATA_DIR / "spx_features_daily.parquet")
            df["date"] = pd.to_datetime(df["date"])
            row = df.sort_values("date").iloc[-1]
            for col in df.columns:
                if col != "date":
                    daily[col] = row[col]
            # Store SMAs for cross-timeframe
            for w in [7, 20, 50, 180]:
                col = f"close_vs_sma_{w}" if f"close_vs_sma_{w}" in df.columns else None
                # Actually we need the raw SMA values for intraday comparison
            # Get ATR
            self.atr_14d = row.get("atr_pct", None)
        except Exception as e:
            print(f"  [warn] spx_features_daily: {e}")

        # Load each feature parquet and take last row
        parquets = [
            "options_features.parquet", "iv_surface_features.parquet",
            "regime_features.parquet", "gex_regime_features.parquet",
            "vanna_charm_features.parquet", "momentum_features.parquet",
            "breadth_features.parquet", "cross_asset_features.parquet",
            "vol_expansion_features.parquet", "microstructure_features.parquet",
        ]
        for pf in parquets:
            try:
                df = pd.read_parquet(DATA_DIR / pf)
                df["date"] = pd.to_datetime(df["date"])
                row = df.sort_values("date").iloc[-1]
                for col in df.columns:
                    if col != "date":
                        daily[col] = row[col]
            except Exception as e:
                print(f"  [warn] {pf}: {e}")

        # VIX daily (optional — VIX features also in options_features/iv_surface)
        vix_path = DATA_DIR / "vix_daily.parquet"
        if vix_path.exists():
            try:
                df = pd.read_parquet(vix_path)
                df["date"] = pd.to_datetime(df["date"])
                row = df.sort_values("date").iloc[-1]
                for col in ["vix_open", "vix_high", "vix_low", "vix_close"]:
                    if col in df.columns:
                        daily[col] = row[col]
                    elif col.replace("vix_", "") in df.columns:
                        daily[col] = row[col.replace("vix_", "")]
            except Exception:
                pass

        # VIX1D
        try:
            df = pd.read_parquet(DATA_DIR / "vix1d_daily.parquet")
            df["date"] = pd.to_datetime(df["date"])
            row = df.sort_values("date").iloc[-1]
            for col in df.columns:
                if col != "date" and "vix1d" in col:
                    daily[col] = row[col]
        except:
            pass

        # Load raw SMA values for intraday cross-timeframe
        try:
            df = pd.read_parquet(DATA_DIR / "spx_features_daily.parquet")
            df["date"] = pd.to_datetime(df["date"])
            spx_daily = pd.read_parquet(DATA_DIR / "spx_daily.parquet")
            spx_daily["date"] = pd.to_datetime(spx_daily["date"])
            spx_daily = spx_daily.sort_values("date")
            for w in [7, 20, 50, 180]:
                self.daily_smas[w] = spx_daily["spx_close"].rolling(w, min_periods=max(1, w // 2)).mean().iloc[-1]

            # Daily RVs for intraday comparison
            log_ret = np.log(spx_daily["spx_close"] / spx_daily["spx_close"].shift(1))
            for w in [5, 20]:
                self.daily_rvols[w] = log_ret.rolling(w).std().iloc[-1] * np.sqrt(252)

            # ATR for range exhaustion
            atr = (spx_daily["spx_high"] - spx_daily["spx_low"]).rolling(14).mean().iloc[-1]
            self.atr_14d = atr
        except:
            pass

        self.daily_features = daily
        print(f"  Loaded {len(daily)} daily features")
        return daily

    def add_bar(self, bar):
        """Add a 1-min bar: {'datetime': ts, 'open': x, 'high': x, 'low': x, 'close': x}"""
        self.bars.append(bar)

        # Update ORB (9:30-10:00)
        ts = bar["datetime"]
        if ts.hour == 9 or (ts.hour == 10 and ts.minute == 0):
            if self.orb_high is None:
                self.orb_high = bar["high"]
                self.orb_low = bar["low"]
            else:
                self.orb_high = max(self.orb_high, bar["high"])
                self.orb_low = min(self.orb_low, bar["low"])
            self.orb_range = self.orb_high - self.orb_low

    def add_bar_from_quote(self, quote):
        """Build a synthetic bar from a Tradier quote."""
        now = datetime.now()
        bar = {
            "datetime": now,
            "open": quote.get("last", 0),
            "high": quote.get("last", 0),
            "low": quote.get("last", 0),
            "close": quote.get("last", 0),
        }
        self.add_bar(bar)
        return bar

    def compute_intraday_features(self):
        """Compute all 30 intraday features from accumulated bars."""
        feats = {}
        if len(self.bars) < 2:
            return {f: np.nan for f in [
                'spx_ret_5m', 'spx_ret_10m', 'spx_ret_15m', 'spx_ret_30m',
                'spx_vs_vwap', 'spx_vs_intraday_ma5', 'spx_vs_intraday_ma15', 'spx_vs_intraday_ma30',
                'spx_vs_sma_7d', 'spx_vs_sma_20d', 'spx_vs_sma_50d', 'spx_vs_sma_180d',
                'intraday_ma15_vs_sma_7d', 'intraday_ma15_vs_sma_20d',
                'rvol_5m', 'rvol_10m', 'rvol_15m', 'rvol_30m',
                'garman_klass_rv_30m', 'rvol_intraday_vs_5d', 'rvol_intraday_vs_20d',
                'range_exhaustion', 'intraday_pctB',
                'orb_range_pct', 'orb_containment', 'orb_breakout_dir',
                'ttm_squeeze', 'bb_squeeze_duration', 'iv_rank_30d', 'iv_rv_ratio',
            ]}

        closes = [b["close"] for b in self.bars]
        highs = [b["high"] for b in self.bars]
        lows = [b["low"] for b in self.bars]
        price = closes[-1]
        n = len(closes)

        # Momentum returns
        for w in [5, 10, 15, 30]:
            if n > w:
                prev = closes[-(w + 1)]
                feats[f"spx_ret_{w}m"] = (price - prev) / prev * 100 if prev > 0 else np.nan
            else:
                feats[f"spx_ret_{w}m"] = np.nan

        # VWAP (cumulative typical price)
        typicals = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        vwap = np.mean(typicals)
        feats["spx_vs_vwap"] = (price - vwap) / vwap * 100 if vwap > 0 else np.nan

        # Intraday MAs
        for w, name in [(5, "ma5"), (15, "ma15"), (30, "ma30")]:
            if n >= w:
                ma = np.mean(closes[-w:])
                feats[f"spx_vs_intraday_{name}"] = (price - ma) / ma * 100 if ma > 0 else np.nan
            else:
                feats[f"spx_vs_intraday_{name}"] = np.nan

        # Cross-timeframe: price vs daily SMAs
        for w in [7, 20, 50, 180]:
            sma = self.daily_smas.get(w)
            if sma and sma > 0:
                feats[f"spx_vs_sma_{w}d"] = (price - sma) / sma * 100
            else:
                feats[f"spx_vs_sma_{w}d"] = np.nan

        # Intraday MA15 vs daily SMAs
        ma15 = np.mean(closes[-15:]) if n >= 15 else np.nan
        for w in [7, 20]:
            sma = self.daily_smas.get(w)
            if not np.isnan(ma15) and sma and sma > 0:
                feats[f"intraday_ma15_vs_sma_{w}d"] = (ma15 - sma) / sma * 100
            else:
                feats[f"intraday_ma15_vs_sma_{w}d"] = np.nan

        # Realized vol
        log_rets = [np.log(closes[i] / closes[i - 1]) for i in range(1, n) if closes[i - 1] > 0]
        for w in [5, 10, 15, 30]:
            if len(log_rets) >= w:
                feats[f"rvol_{w}m"] = np.std(log_rets[-w:]) * np.sqrt(390 * 252)
            else:
                feats[f"rvol_{w}m"] = np.nan

        # Garman-Klass 30m
        if n >= 30:
            gk_vals = []
            for i in range(-30, 0):
                h, l, c, o = highs[i], lows[i], closes[i], self.bars[i]["open"]
                if h > 0 and l > 0 and c > 0 and o > 0:
                    log_hl = np.log(h / l)
                    log_co = np.log(c / o)
                    gk_vals.append(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2)
            feats["garman_klass_rv_30m"] = np.sqrt(np.mean(gk_vals) * 390 * 252) if gk_vals else np.nan
        else:
            feats["garman_klass_rv_30m"] = np.nan

        # RV intraday vs daily
        rv30 = feats.get("rvol_30m")
        for dw in [5, 20]:
            drv = self.daily_rvols.get(dw)
            if rv30 and not np.isnan(rv30) and drv and drv > 0:
                feats[f"rvol_intraday_vs_{dw}d"] = rv30 / drv
            else:
                feats[f"rvol_intraday_vs_{dw}d"] = np.nan

        # Range exhaustion
        running_high = max(highs)
        running_low = min(lows)
        running_range = running_high - running_low
        if self.atr_14d and self.atr_14d > 0:
            feats["range_exhaustion"] = running_range / self.atr_14d
        else:
            feats["range_exhaustion"] = np.nan

        # Intraday %B
        rng = running_high - running_low
        feats["intraday_pctB"] = (price - running_low) / rng if rng > 0 else 0.5

        # ORB features
        if self.orb_range and price > 0:
            feats["orb_range_pct"] = self.orb_range / price * 100
            feats["orb_containment"] = 1 if self.orb_low <= price <= self.orb_high else 0
            if price > self.orb_high:
                feats["orb_breakout_dir"] = 1
            elif price < self.orb_low:
                feats["orb_breakout_dir"] = -1
            else:
                feats["orb_breakout_dir"] = 0
        else:
            feats["orb_range_pct"] = np.nan
            feats["orb_containment"] = np.nan
            feats["orb_breakout_dir"] = np.nan

        # TTM Squeeze (simplified)
        if n >= 20:
            bb_mid = np.mean(closes[-20:])
            bb_std = np.std(closes[-20:])
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            atr_20 = np.mean([highs[i] - lows[i] for i in range(-20, 0)])
            kc_upper = bb_mid + 1.5 * atr_20
            kc_lower = bb_mid - 1.5 * atr_20
            in_squeeze = bb_upper < kc_upper and bb_lower > kc_lower
            feats["ttm_squeeze"] = 1 if in_squeeze else 0
            self.squeeze_count = self.squeeze_count + 1 if in_squeeze else 0
            feats["bb_squeeze_duration"] = self.squeeze_count
        else:
            feats["ttm_squeeze"] = np.nan
            feats["bb_squeeze_duration"] = np.nan

        # IV features (from daily - don't change intraday in V1)
        feats["iv_rank_30d"] = self.daily_features.get("iv_rank_30d", np.nan) if self.daily_features else np.nan
        feats["iv_rv_ratio"] = self.daily_features.get("iv_rv_ratio", np.nan) if self.daily_features else np.nan

        return feats

    def build_feature_vector(self, strategy, credit, feature_cols):
        """Build a complete 284-feature vector for scoring.

        Args:
            strategy: e.g. "IC_25d_25w"
            credit: IC credit at mid
            feature_cols: list of feature names the model expects
        Returns:
            pd.DataFrame with 1 row and all feature columns
        """
        now = datetime.now()
        row = {}

        # 1. Daily features
        if self.daily_features:
            row.update(self.daily_features)

        # 2. Intraday features
        intraday = self.compute_intraday_features()
        row.update(intraday)

        # 3. Entry features
        # Strategy one-hots
        for d in range(5, 50, 5):
            col = f"strat_IC_{d:02d}d_25w"
            row[col] = 1 if strategy == f"IC_{d:02d}d_25w" else 0

        row["entry_hour"] = now.hour
        row["entry_minute"] = now.minute
        minutes_since_open = (now.hour - 9) * 60 + now.minute - 30
        row["minutes_since_open"] = minutes_since_open
        row["time_sin"] = np.sin(2 * np.pi * minutes_since_open / 390)
        row["time_cos"] = np.cos(2 * np.pi * minutes_since_open / 390)
        row["credit"] = credit

        # Other daily features that might be needed
        row["spx_open"] = self.bars[0]["open"] if self.bars else np.nan
        row["atm_strike"] = round(row.get("spx_open", 0) / 5) * 5

        # Build DataFrame with exact column order
        df = pd.DataFrame([row])

        # Ensure all expected columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan

        return df[feature_cols]
