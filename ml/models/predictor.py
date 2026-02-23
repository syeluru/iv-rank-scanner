"""
Real-time ML predictions for iron condor trades.

Supports four model types:
  - v2 (LightGBM): Predicts P(win). Trade if confidence >= threshold.
  - v3 (RandomForest): Predicts P(big_loss). Skip if risk >= threshold.
  - v4 (RandomForest): Entry timing — predicts P(profitable) at each 5-min slot.
  - v5 (RandomForest): TP prediction — predicts P(hit 25%/50% TP) at each 5-min slot.

v2/v3 return a "trade confidence" score (0-1) via predict().
v4 returns P(profitable) via predict_v4() using intraday technical indicators.
v5 returns P(hit TP) via predict_v5() using same features + economic calendar.
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
from config.settings import settings
from loguru import logger


# ── 2026 Economic Calendar (hardcoded — update yearly) ───────────────────────
# FOMC announcement dates (2-day meetings, announcement on day 2)
FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

# CPI release dates (Bureau of Labor Statistics)
CPI_DATES_2026 = [
    "2026-01-14", "2026-02-12", "2026-03-11", "2026-04-14",
    "2026-05-13", "2026-06-10", "2026-07-15", "2026-08-12",
    "2026-09-16", "2026-10-14", "2026-11-12", "2026-12-09",
]

# NFP release dates (first Friday of each month)
NFP_DATES_2026 = [
    "2026-01-02", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]

# Pre-parsed for fast lookups
_FOMC_DATE_SET = set(pd.Timestamp(d).date() for d in FOMC_DATES_2026)
_FOMC_WEEKS = set()
for d in FOMC_DATES_2026:
    ts = pd.Timestamp(d)
    _FOMC_WEEKS.add((ts.isocalendar()[0], ts.isocalendar()[1]))  # (year, week)
_CPI_DATE_SET = set(pd.Timestamp(d).date() for d in CPI_DATES_2026)
_NFP_DATE_SET = set(pd.Timestamp(d).date() for d in NFP_DATES_2026)


def get_economic_calendar_features(dt: datetime = None) -> dict:
    """
    Return economic calendar features for a given date.

    Returns dict with: is_fomc_day, is_fomc_week, is_cpi_day, is_nfp_day.
    All values are 0 or 1 (float for ML compatibility).
    """
    if dt is None:
        dt = datetime.now()
    d = dt.date() if hasattr(dt, 'date') else dt

    iso = pd.Timestamp(d).isocalendar()
    is_fomc_week = 1.0 if (iso[0], iso[1]) in _FOMC_WEEKS else 0.0

    return {
        "is_fomc_day": 1.0 if d in _FOMC_DATE_SET else 0.0,
        "is_fomc_week": is_fomc_week,
        "is_cpi_day": 1.0 if d in _CPI_DATE_SET else 0.0,
        "is_nfp_day": 1.0 if d in _NFP_DATE_SET else 0.0,
    }


class MLPredictor:
    """Real-time ML predictions for IC trades."""

    # v2 FEATURE_MAP: pipeline feature names → model feature names
    FEATURE_MAP_V2 = {
        'vix_level': 'vix_level',
        'iv_skew': 'iv_skew',
        'put_call_ratio': 'put_call_ratio',
        'hy_oas': 'hy_oas',
        'intraday_trend': 'intraday_trend',
        'iv_put_atm': 'entry_iv_put',
        'iv_call_atm': 'entry_iv_call',
        'iv_put_otm_10': 'avg_bid_ask_spread',
        'rv_5d': 'realized_vol_5d',
        'rv_20d': 'realized_vol_20d',
        'rv_ratio_5_20': 'realized_vol_ratio',
        'garman_klass_vol_10d': 'realized_vol_intraday',
        'yield_curve_10y_2y': 'yield_curve_10y2y',
        'vix_change_1d': 'vix_change_from_open',
        'vix_change_5d': 'vix_mean_reversion',
        'vvix_level': 'variance_risk_premium',
        'market_stress_index': 'vix_percentile_30d',
        'orb_range_pct': 'morning_range_pct',
        'high_low_range_pct': 'first_hour_range_pct',
        'range_exhaustion_ratio': 'prior_day_range_ratio',
        'close_vs_open_pct': 'gap_strength',
        'price_vs_vwap': 'spx_ema9_dist',
        'range_rate': 'spx_range_expansion_rate',
        'divergence_score': 'spx_return_autocorr_30min',
        'spx_vix_correlation_30min': 'spx_roc_15min',
        'spx_up_vix_up': 'spx_roc_5min',
        'minutes_since_open': 'dist_from_session_high',
        'time_of_day_hour': 'dist_from_session_low',
        'trend_regime': 'adx_14',
        'market_breadth': 'dix_level',
        'vol_regime': 'gex_level',
        'term_structure_slope': 'vix_roc_15min',
        'vix9d_vix': 'vix_roc_5min',
        'volume_vs_avg': 'overnight_range',
        'macro_event_proximity': 'prior_day_ic_win',
    }

    # v3 features: maps pipeline feature name → v3 model feature name
    # Direct matches from the real-time feature pipeline
    PIPELINE_TO_V3 = {
        'vix_level': 'vix_level',
        'orb_range_pct': 'orb_range_pct',
        'vix_change_5d': 'vix_sma_10d_dist',    # Proxy: 5d change ≈ dist from SMA
        'rv_5d': 'rv_close_5d',
        'rv_10d': 'rv_close_10d',
        'rv_20d': 'rv_close_20d',
        'term_structure_slope': 'vix_term_slope',
    }

    # v3 rule-based filters (from training script)
    RULE_FILTERS = {
        "vix_sma_10d_dist": (">", 15),
        "gap_vs_atr": (">", 1.5),
        "rv_5d_20d_ratio": (">", 1.5),
        "prior_day_range_pct": (">", 2.0),
    }

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.ML_MODEL_PATH
        self.model = None
        self.feature_names = None
        self.model_version = None
        self.model_type = None  # "classifier_win" (v2), "risk_filter" (v3), or "entry_timing" (v4)
        self.rule_filters = {}
        self.skip_percentile = 85  # Default: skip top 15% risk
        self._daily_cache = {}
        self._daily_cache_date = None

        # v4 entry timing model (separate artifact)
        self.v4_model = None
        self.v4_feature_names = None
        self.v4_derived_features = {}

        # v5 take-profit models (separate artifacts)
        self.v5_tp25_model = None
        self.v5_tp25_feature_names = None
        self.v5_tp50_model = None
        self.v5_tp50_feature_names = None

        if self.model_path.exists():
            self._load_model()
        else:
            logger.warning(f"ML model not found at {self.model_path}")

        # Try loading v4 entry timing model
        v4_path = self.model_path.parent / "entry_timing_v4.joblib"
        if v4_path.exists():
            self._load_v4_model(v4_path)

        # Try loading v5 TP models
        v5_tp25_path = self.model_path.parent / "entry_timing_v5_tp25.joblib"
        v5_tp50_path = self.model_path.parent / "entry_timing_v5_tp50.joblib"
        if v5_tp25_path.exists():
            self._load_v5_model(v5_tp25_path, "tp25")
        if v5_tp50_path.exists():
            self._load_v5_model(v5_tp50_path, "tp50")

    def _load_model(self):
        """Load model artifact and detect version."""
        try:
            artifact = joblib.load(self.model_path)

            if isinstance(artifact, dict):
                self.model = artifact['model']
                self.feature_names = artifact.get('feature_names', [])
                label = artifact.get('label', '')

                if artifact.get('model_type') == 'RandomForestClassifier' or label == 'avoid_trade':
                    self.model_version = "v3"
                    self.model_type = "risk_filter"
                    self.rule_filters = artifact.get('rule_filters', self.RULE_FILTERS)
                    self.skip_percentile = artifact.get('skip_percentile', 85)
                    logger.info(
                        f"ML model loaded: {self.model_path.name} "
                        f"(v3 risk filter, {len(self.feature_names)} features, "
                        f"skip top {100 - self.skip_percentile}%)"
                    )
                else:
                    self.model_version = "v2"
                    self.model_type = "classifier_win"
                    logger.info(
                        f"ML model loaded: {self.model_path.name} "
                        f"(v2 win classifier, {len(self.feature_names)} features)"
                    )
            else:
                self.model = artifact
                self.feature_names = artifact.feature_name_
                self.model_version = "v2"
                self.model_type = "classifier_win"
                logger.info(f"ML model loaded: {self.model_path.name} (v2)")

        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            raise

    def _get_daily_data(self) -> dict:
        """Fetch and cache daily SPX/VIX data for feature computation (v3)."""
        today = datetime.now().date()
        if self._daily_cache_date == today and self._daily_cache:
            return self._daily_cache

        try:
            import yfinance as yf
            end = today + timedelta(days=1)
            start = today - timedelta(days=60)

            spx = yf.download('^GSPC', start=str(start), end=str(end),
                              progress=False, auto_adjust=True)
            vix = yf.download('^VIX', start=str(start), end=str(end),
                              progress=False, auto_adjust=True)
            vix3m = yf.download('^VIX3M', start=str(start), end=str(end),
                                progress=False, auto_adjust=True)

            self._daily_cache = {'spx': spx, 'vix': vix, 'vix3m': vix3m}
            self._daily_cache_date = today
            logger.debug(f"Cached daily data: SPX {len(spx)} days, VIX {len(vix)} days")
            return self._daily_cache

        except Exception as e:
            logger.warning(f"Failed to fetch daily data: {e}")
            return {}

    def compute_v3_features(self, pipeline_features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute v3 model features from pipeline output + daily data.

        The v3 model expects 14 curated features. We compute them from:
        1. Existing pipeline features (direct mapping)
        2. Daily data from yfinance (gap, prior day, ATR, VIX SMA)
        3. Derived ratios
        """
        v3 = {}
        daily = self._get_daily_data()
        spx = daily.get('spx')
        vix_df = daily.get('vix')
        vix3m_df = daily.get('vix3m')

        # ── From pipeline (direct mappings) ──
        v3['vix_level'] = pipeline_features.get('vix_level', 18.5)
        v3['orb_range_pct'] = pipeline_features.get('orb_range_pct', 0.0)

        # momentum_30min: use pipeline's range_rate or compute from trend
        v3['momentum_30min'] = pipeline_features.get('range_rate',
                               pipeline_features.get('intraday_trend', 0.0))

        # atm_iv: average of put and call ATM IV
        iv_put = pipeline_features.get('iv_put_atm', 0)
        iv_call = pipeline_features.get('iv_call_atm', 0)
        if iv_put > 0 and iv_call > 0:
            v3['atm_iv'] = (iv_put + iv_call) / 2
        elif iv_put > 0:
            v3['atm_iv'] = iv_put
        elif iv_call > 0:
            v3['atm_iv'] = iv_call
        else:
            v3['atm_iv'] = v3['vix_level']  # Fallback to VIX

        # RV from pipeline
        rv_5d = pipeline_features.get('rv_5d', 0)
        rv_10d = pipeline_features.get('garman_klass_vol_10d',
                 pipeline_features.get('rv_10d', 0))
        rv_20d = pipeline_features.get('rv_20d', 0)

        # ── From daily data ──
        if spx is not None and len(spx) >= 2:
            closes = spx['Close'].values.flatten()
            highs = spx['High'].values.flatten()
            lows = spx['Low'].values.flatten()
            opens = spx['Open'].values.flatten()

            # Gap: today's open vs yesterday's close
            if len(closes) >= 2:
                v3['gap_pct'] = (opens[-1] / closes[-2] - 1) * 100
                v3['prior_day_range_pct'] = (highs[-2] - lows[-2]) / closes[-2] * 100
                v3['prior_day_return'] = (closes[-2] / closes[-3] - 1) * 100 if len(closes) >= 3 else 0

            # ATR 14d
            if len(closes) >= 15:
                trs = []
                for i in range(-14, 0):
                    tr = max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i-1]),
                        abs(lows[i] - closes[i-1])
                    )
                    trs.append(tr)
                atr = np.mean(trs) / closes[-1] * 100
            else:
                atr = 1.0

            # RV from daily data (if pipeline didn't provide)
            if rv_5d == 0 and len(closes) >= 6:
                log_rets = np.diff(np.log(closes[-6:]))
                rv_5d = np.std(log_rets) * np.sqrt(252) * 100
            if rv_10d == 0 and len(closes) >= 11:
                log_rets = np.diff(np.log(closes[-11:]))
                rv_10d = np.std(log_rets) * np.sqrt(252) * 100
            if rv_20d == 0 and len(closes) >= 21:
                log_rets = np.diff(np.log(closes[-21:]))
                rv_20d = np.std(log_rets) * np.sqrt(252) * 100
        else:
            v3.setdefault('gap_pct', 0)
            v3.setdefault('prior_day_range_pct', 1.0)
            v3.setdefault('prior_day_return', 0)
            atr = 1.0

        # VIX SMA 10d distance
        if vix_df is not None and len(vix_df) >= 10:
            vix_closes = vix_df['Close'].values.flatten()
            vix_sma_10 = np.mean(vix_closes[-10:])
            vix_now = vix_closes[-1]
            v3['vix_sma_10d_dist'] = (vix_now / vix_sma_10 - 1) * 100
        else:
            v3['vix_sma_10d_dist'] = pipeline_features.get('vix_change_5d', 0)

        # VIX term slope
        if vix_df is not None and vix3m_df is not None and len(vix_df) >= 1 and len(vix3m_df) >= 1:
            vix_now = vix_df['Close'].values.flatten()[-1]
            vix3m_now = vix3m_df['Close'].values.flatten()[-1]
            v3['vix_term_slope'] = (vix3m_now - vix_now) / vix_now if vix_now > 0 else 0
        else:
            v3['vix_term_slope'] = pipeline_features.get('term_structure_slope', 0.1)

        # ── Derived features ──
        rv_20d_safe = max(rv_20d, 1.0)
        atr_safe = max(atr, 0.01)

        v3['iv_rv_ratio'] = v3['atm_iv'] / rv_20d_safe
        v3['rv_5d_20d_ratio'] = rv_5d / rv_20d_safe if rv_5d > 0 else 1.0
        v3['rv_acceleration'] = rv_5d - rv_10d
        v3['gap_vs_atr'] = abs(v3.get('gap_pct', 0)) / atr_safe
        v3['orb_vs_atr'] = v3['orb_range_pct'] / atr_safe

        return v3

    def check_rules(self, features: Dict[str, float]) -> bool:
        """Check rule-based filters. Returns True if ANY rule says skip."""
        for feature, (op, threshold) in self.rule_filters.items():
            val = features.get(feature, 0)
            if op == ">" and val > threshold:
                return True
            elif op == "<" and val < threshold:
                return True
        return False

    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict trade confidence (0-1). Higher = safer to trade.

        For v2: returns P(win) directly.
        For v3: returns 1 - P(big_loss). Also checks rule filters.
        """
        if self.model is None:
            raise ValueError("ML model not loaded.")

        if self.model_type == "risk_filter":
            return self._predict_v3(features)
        else:
            return self._predict_v2(features)

    def _predict_v2(self, features: Dict[str, float]) -> float:
        """v2 prediction: map pipeline features to model features, return P(win)."""
        mapped = {}
        for pipeline_name, model_name in self.FEATURE_MAP_V2.items():
            if pipeline_name in features:
                val = features[pipeline_name]
                mapped[model_name] = 0.0 if val != val else val

        for name in self.feature_names:
            if name in features and name not in mapped:
                val = features[name]
                mapped[name] = 0.0 if val != val else val

        missing = set(self.feature_names) - set(mapped.keys())
        for name in missing:
            mapped[name] = 0.0

        feature_array = np.array([mapped[n] for n in self.feature_names]).reshape(1, -1)
        return float(self.model.predict_proba(feature_array)[0][1])

    def _predict_v3(self, features: Dict[str, float]) -> float:
        """
        v3 prediction: compute risk features, return trade confidence.

        Returns 1 - P(big_loss). Also applies rule filters as override.
        If rules say skip, returns 0.0 (maximum risk).
        """
        # Compute v3 features from pipeline output + daily data
        v3_features = self.compute_v3_features(features)

        # Check rule-based filters first
        if self.check_rules(v3_features):
            triggered = []
            for feat, (op, thresh) in self.rule_filters.items():
                val = v3_features.get(feat, 0)
                if (op == ">" and val > thresh) or (op == "<" and val < thresh):
                    triggered.append(f"{feat}={val:.2f}{op}{thresh}")
            logger.warning(f"Rule filter triggered: {', '.join(triggered)}")
            return 0.0  # Maximum risk → don't trade

        # ML prediction
        feature_array = np.array([
            v3_features.get(name, 0.0) for name in self.feature_names
        ]).reshape(1, -1)

        risk_prob = float(self.model.predict_proba(feature_array)[0][1])

        # Convert to trade confidence: 1 - risk
        confidence = 1.0 - risk_prob
        return confidence

    def predict_with_explanation(self, features: Dict[str, float]) -> Dict:
        """Predict with feature importance explanation."""
        if self.model_type == "risk_filter":
            v3_features = self.compute_v3_features(features)
            confidence = self.predict(features)
            risk_prob = 1.0 - confidence

            importances = self.model.feature_importances_
            feature_info = [
                (name, v3_features.get(name, 0.0), imp)
                for name, imp in zip(self.feature_names, importances)
            ]
            feature_info.sort(key=lambda x: x[2], reverse=True)

            # Check which rules triggered
            rules_triggered = []
            for feat, (op, thresh) in self.rule_filters.items():
                val = v3_features.get(feat, 0)
                if (op == ">" and val > thresh) or (op == "<" and val < thresh):
                    rules_triggered.append(f"{feat}={val:.2f}{op}{thresh}")

            return {
                'probability': confidence,
                'risk_probability': risk_prob,
                'rules_triggered': rules_triggered,
                'top_features': feature_info[:10],
                'model_version': self.model_version,
                'v3_features': v3_features,
            }
        else:
            # v2 path
            mapped = {}
            for pipeline_name, model_name in self.FEATURE_MAP_V2.items():
                if pipeline_name in features:
                    val = features[pipeline_name]
                    mapped[model_name] = 0.0 if val != val else val
            for name in self.feature_names:
                if name in features and name not in mapped:
                    val = features[name]
                    mapped[name] = 0.0 if val != val else val
                elif name not in mapped:
                    mapped[name] = 0.0

            prob = self.predict(features)
            importances = self.model.feature_importances_
            feature_info = [
                (name, mapped.get(name, 0.0), imp)
                for name, imp in zip(self.feature_names, importances)
            ]
            feature_info.sort(key=lambda x: x[2], reverse=True)

            return {
                'probability': prob,
                'top_features': feature_info[:10],
                'model_version': self.model_version,
            }

    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def v4_ready(self) -> bool:
        return self.v4_model is not None

    @property
    def num_features(self) -> int:
        return len(self.feature_names) if self.feature_names else 0

    def _load_v4_model(self, path: Path):
        """Load v4 entry timing model artifact."""
        try:
            artifact = joblib.load(path)
            self.v4_model = artifact["model"]
            self.v4_feature_names = artifact.get("feature_names", [])
            self.v4_derived_features = artifact.get("derived_features", {})
            logger.info(
                f"v4 entry timing model loaded: {path.name} "
                f"({len(self.v4_feature_names)} features)"
            )
        except Exception as e:
            logger.warning(f"Failed to load v4 model: {e}")

    def compute_v4_features(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
    ) -> Dict[str, float]:
        """
        Compute v4 feature vector from 5-min candles + daily context.

        Args:
            candles_5min: DataFrame with open/high/low/close columns,
                          rows from 9:30 up to current time.
            daily_data: Optional dict with pre-computed day-level features
                        (vix_sma_10d_dist, gap_pct, prior_day_range_pct, etc.)
            option_atm_iv: ATM implied volatility from option chain.

        Returns:
            Dict of all v4 features.
        """
        from ml.features.technical_indicators import compute_all_technical_indicators

        features = {}

        # ── Technical indicators from candles ──
        tech = compute_all_technical_indicators(candles_5min, orb_bars=6)
        features.update(tech)

        # ── Intraday features from candles ──
        if candles_5min is not None and len(candles_5min) >= 2:
            closes = candles_5min["close"].values.astype(float)
            highs = candles_5min["high"].values.astype(float)
            lows = candles_5min["low"].values.astype(float)

            log_ret = np.diff(np.log(closes[closes > 0]))
            features["intraday_rv"] = float(np.sqrt(np.sum(log_ret**2) * 252) * 100) if len(log_ret) >= 2 else 10.0

            session_high = float(np.max(highs))
            session_low = float(np.min(lows))
            open_price = float(candles_5min.iloc[0]["open"])
            if open_price > 0:
                features["high_low_range_pct"] = ((session_high - session_low) / open_price) * 100
            else:
                features["high_low_range_pct"] = 0.0
        else:
            features["intraday_rv"] = 10.0
            features["high_low_range_pct"] = 0.0

        # ── Day-level features ──
        if daily_data is None:
            daily_data = {}
            daily = self._get_daily_data()
            spx = daily.get("spx")
            vix_df = daily.get("vix")

            if spx is not None and len(spx) >= 21:
                closes_d = spx["Close"].values.flatten()
                highs_d = spx["High"].values.flatten()
                lows_d = spx["Low"].values.flatten()
                opens_d = spx["Open"].values.flatten()

                # gap_pct
                if len(closes_d) >= 2:
                    daily_data["gap_pct"] = (opens_d[-1] / closes_d[-2] - 1) * 100
                    daily_data["prior_day_range_pct"] = (highs_d[-2] - lows_d[-2]) / closes_d[-2] * 100

                # rv_close_20d for iv_rv_ratio
                if len(closes_d) >= 21:
                    log_rets = np.diff(np.log(closes_d[-21:]))
                    daily_data["rv_close_20d"] = np.std(log_rets) * np.sqrt(252) * 100

                # ATR for range_exhaustion_pct
                if len(closes_d) >= 15:
                    trs = []
                    for i in range(-14, 0):
                        tr = max(highs_d[i] - lows_d[i],
                                 abs(highs_d[i] - closes_d[i-1]),
                                 abs(lows_d[i] - closes_d[i-1]))
                        trs.append(tr)
                    daily_data["atr_14d_pct"] = np.mean(trs) / closes_d[-1] * 100

            if vix_df is not None and len(vix_df) >= 10:
                vix_closes = vix_df["Close"].values.flatten()
                vix_sma_10 = np.mean(vix_closes[-10:])
                vix_now = vix_closes[-1]
                daily_data["vix_sma_10d_dist"] = (vix_now / vix_sma_10 - 1) * 100
                daily_data["vix_level"] = float(vix_now)

        features["vix_sma_10d_dist"] = daily_data.get("vix_sma_10d_dist", 0.0)
        features["gap_pct"] = daily_data.get("gap_pct", 0.0)
        features["prior_day_range_pct"] = daily_data.get("prior_day_range_pct", 1.0)
        features["atm_iv"] = option_atm_iv

        # iv_rv_ratio (derived)
        rv_20d = max(daily_data.get("rv_close_20d", 15.0), 1.0)
        features["iv_rv_ratio"] = option_atm_iv / rv_20d

        # ── Derived cross-features ──
        gk_rv = features.get("garman_klass_rv")
        if gk_rv is not None and not np.isnan(gk_rv) and option_atm_iv > 0:
            features["gk_rv_vs_atm_iv"] = gk_rv / option_atm_iv
        else:
            features["gk_rv_vs_atm_iv"] = 0.0

        vix_level = daily_data.get("vix_level", 18.5)
        hl_range = features.get("high_low_range_pct", 0)
        if vix_level > 0 and hl_range > 0:
            features["range_vs_vix_range"] = hl_range / (vix_level / np.sqrt(252))
        else:
            features["range_vs_vix_range"] = 0.0

        atr_pct = daily_data.get("atr_14d_pct", 0.5)
        if atr_pct > 0:
            features["range_exhaustion_pct"] = hl_range / atr_pct
        else:
            features["range_exhaustion_pct"] = 0.0

        # ── Time features ──
        now = datetime.now()
        minutes_since_open = (now.hour - 9) * 60 + (now.minute - 30)
        frac = minutes_since_open / 390.0
        features["time_sin"] = float(np.sin(2 * np.pi * frac))
        features["time_cos"] = float(np.cos(2 * np.pi * frac))
        features["minutes_to_close"] = float((16 - now.hour) * 60 - now.minute)

        return features

    def predict_v4(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
    ) -> float:
        """
        Predict P(profitable) for entry timing.

        Args:
            candles_5min: 5-min OHLC candles from 9:30 to now.
            daily_data: Optional pre-computed day-level features.
            option_atm_iv: ATM implied volatility.

        Returns:
            P(profitable) score (0-1). Higher = better entry time.
        """
        if self.v4_model is None:
            raise ValueError("v4 entry timing model not loaded")

        features = self.compute_v4_features(candles_5min, daily_data, option_atm_iv)

        # Build feature vector in model order
        feature_array = np.array([
            features.get(name, 0.0) if not np.isnan(features.get(name, 0.0))
            else 0.0
            for name in self.v4_feature_names
        ]).reshape(1, -1)

        return float(self.v4_model.predict_proba(feature_array)[0][1])

    # ── v5 Take-Profit Models ──────────────────────────────────────────────────

    @property
    def v5_tp25_ready(self) -> bool:
        return self.v5_tp25_model is not None

    @property
    def v5_tp50_ready(self) -> bool:
        return self.v5_tp50_model is not None

    @property
    def v5_ready(self) -> bool:
        return self.v5_tp25_ready or self.v5_tp50_ready

    def _load_v5_model(self, path: Path, target: str):
        """Load a v5 TP model artifact (tp25 or tp50)."""
        try:
            artifact = joblib.load(path)
            model = artifact["model"]
            feature_names = artifact.get("feature_names", [])
            label = artifact.get("label", target)

            if target == "tp25":
                self.v5_tp25_model = model
                self.v5_tp25_feature_names = feature_names
            else:
                self.v5_tp50_model = model
                self.v5_tp50_feature_names = feature_names

            logger.info(
                f"v5 {target} model loaded: {path.name} "
                f"({len(feature_names)} features, label={label})"
            )
        except Exception as e:
            logger.warning(f"Failed to load v5 {target} model: {e}")

    def compute_v5_features(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
    ) -> Dict[str, float]:
        """
        Compute v5 feature vector = v4 features + economic calendar features.

        Args:
            candles_5min: DataFrame with open/high/low/close columns.
            daily_data: Optional dict with pre-computed day-level features.
            option_atm_iv: ATM implied volatility from option chain.

        Returns:
            Dict of all v5 features (v4 features + is_fomc_day, is_fomc_week).
        """
        # Start with all v4 features
        features = self.compute_v4_features(candles_5min, daily_data, option_atm_iv)

        # Add economic calendar features
        cal = get_economic_calendar_features()
        features["is_fomc_day"] = cal["is_fomc_day"]
        features["is_fomc_week"] = cal["is_fomc_week"]

        return features

    def predict_v5(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
        target: str = "tp25",
    ) -> float:
        """
        Predict P(hit TP) for entry timing.

        Args:
            candles_5min: 5-min OHLC candles from 9:30 to now.
            daily_data: Optional pre-computed day-level features.
            option_atm_iv: ATM implied volatility.
            target: "tp25" for P(hit 25% TP) or "tp50" for P(hit 50% TP).

        Returns:
            P(hit TP) score (0-1). Higher = more likely to hit take-profit.
        """
        if target == "tp25":
            model = self.v5_tp25_model
            feature_names = self.v5_tp25_feature_names
        elif target == "tp50":
            model = self.v5_tp50_model
            feature_names = self.v5_tp50_feature_names
        else:
            raise ValueError(f"Invalid v5 target: {target}. Use 'tp25' or 'tp50'.")

        if model is None:
            raise ValueError(f"v5 {target} model not loaded")

        features = self.compute_v5_features(candles_5min, daily_data, option_atm_iv)

        # Build feature vector in model order
        feature_array = np.array([
            features.get(name, 0.0) if not np.isnan(features.get(name, 0.0))
            else 0.0
            for name in feature_names
        ]).reshape(1, -1)

        return float(model.predict_proba(feature_array)[0][1])
