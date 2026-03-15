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
        # Default (10-delta) models
        self.v5_tp25_model = None
        self.v5_tp25_feature_names = None
        self.v5_tp50_model = None
        self.v5_tp50_feature_names = None
        # Delta-specific models: {delta_float: (model, feature_names)}
        self._v5_delta_models = {}  # keyed by (target, delta), e.g. ("tp25", 0.15)

        # v6 ensemble models (4 windows per target = 8 models total)
        self.v6_ensemble = {
            'tp25': {},  # {'1y': model, '6m': model, '3m': model, '1m': model}
            'tp50': {}
        }
        self.v6_feature_names = None  # Same features across all v6 models

        # v7 long IC models: {delta_float: (model, feature_names)}
        self._v7_models = {}  # keyed by delta, e.g. {0.10: (model, features), ...}

        # v8 XGBoost models (200+ features from v8 feature store)
        # Short IC: {(target, delta): model} where target in (tp25, tp50, big_loss)
        # Long IC: {(target, delta): model} where target in (profitable, big_move)
        self._v8_short_models = {}
        self._v8_long_models = {}
        self._v8_feature_names = None  # Ordered list of feature names (shared)

        # v8 ensemble: {(strategy, target, window, delta): model}
        # e.g. ("short_ic", "tp25", "3yr", 0.10): XGBClassifier
        self._v8_ens_models = {}
        self._v8_ens_weights = {}       # window_name -> weight
        self._v8_ens_feature_names = None

        if self.model_path.exists():
            self._load_model()
        else:
            logger.warning(f"ML model not found at {self.model_path}")

        # Try loading v4 entry timing model
        v4_path = self.model_path.parent / "entry_timing_v4.joblib"
        if v4_path.exists():
            self._load_v4_model(v4_path)

        # Try loading v5 TP models (10-delta defaults)
        v5_tp25_path = self.model_path.parent / "entry_timing_v5_tp25.joblib"
        v5_tp50_path = self.model_path.parent / "entry_timing_v5_tp50.joblib"
        if v5_tp25_path.exists():
            self._load_v5_model(v5_tp25_path, "tp25")
        if v5_tp50_path.exists():
            self._load_v5_model(v5_tp50_path, "tp50")

        # Try loading delta-specific v5 models (15-delta, 20-delta)
        for delta, suffix in [(0.15, "_d15"), (0.20, "_d20")]:
            for target_name in ["tp25", "tp50"]:
                path = self.model_path.parent / f"entry_timing_v5_{target_name}{suffix}.joblib"
                if path.exists():
                    self._load_v5_delta_model(path, target_name, delta)

        # Try loading v6 ensemble models (1y, 6m, 3m, 1m windows)
        self._load_v6_ensemble()

        # Try loading v7 long IC models (per delta)
        self._load_v7_models()

        # Try loading v8 XGBoost models
        self._load_v8_models()

        # Try loading v8 ensemble models (6-timeframe weighted)
        self._load_v8_ensemble()

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

    @property
    def available_v5_deltas(self) -> list:
        """Return sorted list of deltas with loaded v5 models (at least tp25)."""
        deltas = set()
        # Default 10-delta models
        if self.v5_tp25_model is not None:
            deltas.add(0.10)
        # Delta-specific models
        for (target, delta) in self._v5_delta_models:
            if target == "tp25":
                deltas.add(delta)
        return sorted(deltas)

    @property
    def v6_ready(self) -> bool:
        """Check if v6 ensemble is loaded (need all 6 windows for both targets)."""
        return (
            len(self.v6_ensemble['tp25']) == self.V6_EXPECTED_MODELS and
            len(self.v6_ensemble['tp50']) == self.V6_EXPECTED_MODELS
        )

    @property
    def v6_tp25_ready(self) -> bool:
        """Check if v6 tp25 models are loaded (all 6 windows)."""
        return len(self.v6_ensemble['tp25']) == self.V6_EXPECTED_MODELS

    @property
    def v6_tp50_ready(self) -> bool:
        """Check if v6 tp50 models are loaded (all 6 windows)."""
        return len(self.v6_ensemble['tp50']) == self.V6_EXPECTED_MODELS

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

    def _load_v5_delta_model(self, path: Path, target: str, delta: float):
        """Load a delta-specific v5 model (e.g. 15-delta, 20-delta)."""
        try:
            artifact = joblib.load(path)
            model = artifact["model"]
            feature_names = artifact.get("feature_names", [])
            self._v5_delta_models[(target, delta)] = (model, feature_names)
            logger.info(
                f"v5 {target} d{int(delta*100)} model loaded: {path.name} "
                f"({len(feature_names)} features)"
            )
        except Exception as e:
            logger.warning(f"Failed to load v5 {target} d{int(delta*100)} model: {e}")

    # v6 ensemble windows — order matters for weighted prediction
    V6_WINDOWS = ['3y', '2y', '1y', '6m', '3m', '1m']
    V6_EXPECTED_MODELS = len(V6_WINDOWS)  # per target

    def _load_v6_ensemble(self):
        """Load v6 ensemble models (6 windows per target = 12 models total)."""
        base_path = self.model_path.parent / "v6" / "ensemble"
        if not base_path.exists():
            logger.debug("v6 ensemble directory not found, skipping")
            return

        loaded_count = 0
        total_expected = self.V6_EXPECTED_MODELS * 2

        for target in ['tp25', 'tp50']:
            for window in self.V6_WINDOWS:
                # Try with month tag first (e.g., entry_timing_v6_tp25_202603_1y.joblib)
                # Fall back to no month tag
                patterns = [
                    f"entry_timing_v6_{target}_*_{window}.joblib",
                    f"entry_timing_v6_{target}_{window}.joblib"
                ]

                path = None
                for pattern in patterns:
                    import glob
                    matches = list(base_path.glob(pattern))
                    if matches:
                        # Use most recent if multiple matches
                        path = Path(sorted(matches, reverse=True)[0])
                        break

                if path and path.exists():
                    try:
                        artifact = joblib.load(path)
                        model = artifact["model"]

                        self.v6_ensemble[target][window] = model
                        # Use XGBoost's internal feature names (authoritative)
                        # over artifact metadata (may be stale/wrong)
                        if self.v6_feature_names is None:
                            if hasattr(model, 'feature_names_in_'):
                                self.v6_feature_names = list(model.feature_names_in_)
                            else:
                                self.v6_feature_names = artifact.get("feature_names", [])

                        loaded_count += 1
                        logger.debug(f"v6 {target} {window} loaded: {path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load v6 {target} {window}: {e}")

        if loaded_count > 0:
            logger.info(
                f"v6 ensemble loaded: {loaded_count}/{total_expected} models "
                f"({len(self.v6_feature_names)} features)"
            )

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

    def compute_v6_features(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
    ) -> Dict[str, float]:
        """
        Compute full v6 feature vector (41 features).

        Extends v5 features with additional daily, VIX, intraday, and calendar
        features that the v6 XGBoost ensemble was trained on.
        """
        # Start with v5 features (v4 + economic calendar)
        features = self.compute_v5_features(candles_5min, daily_data, option_atm_iv)

        # Fetch daily data for additional features
        daily = self._get_daily_data()
        spx = daily.get("spx")
        vix_df = daily.get("vix")
        vix3m_df = daily.get("vix3m")

        # --- Daily realized volatility windows ---
        if spx is not None and len(spx) >= 21:
            closes_d = spx["Close"].values.flatten()
            highs_d = spx["High"].values.flatten()
            lows_d = spx["Low"].values.flatten()
            opens_d = spx["Open"].values.flatten()

            for window, key in [(5, "rv_close_5d"), (10, "rv_close_10d"), (20, "rv_close_20d")]:
                if len(closes_d) >= window + 1:
                    log_rets = np.diff(np.log(closes_d[-(window+1):]))
                    features[key] = float(np.std(log_rets) * np.sqrt(252) * 100)
                else:
                    features.setdefault(key, 15.0)

            # atr_14d_pct (may already be in daily_data from v4)
            if "atr_14d_pct" not in features:
                if len(closes_d) >= 15:
                    trs = []
                    for i in range(-14, 0):
                        tr = max(highs_d[i] - lows_d[i],
                                 abs(highs_d[i] - closes_d[i-1]),
                                 abs(lows_d[i] - closes_d[i-1]))
                        trs.append(tr)
                    features["atr_14d_pct"] = float(np.mean(trs) / closes_d[-1] * 100)
                else:
                    features.setdefault("atr_14d_pct", 0.5)

            # prior_day_return
            if len(closes_d) >= 2:
                features.setdefault("prior_day_return",
                                    float((closes_d[-2] / closes_d[-3] - 1) * 100) if len(closes_d) >= 3 else 0.0)

            # sma_20d_dist
            if len(closes_d) >= 20:
                sma20 = float(np.mean(closes_d[-20:]))
                features.setdefault("sma_20d_dist", float((closes_d[-1] / sma20 - 1) * 100))
            else:
                features.setdefault("sma_20d_dist", 0.0)
        else:
            for key in ["rv_close_5d", "rv_close_10d", "rv_close_20d"]:
                features.setdefault(key, 15.0)
            features.setdefault("atr_14d_pct", 0.5)
            features.setdefault("prior_day_return", 0.0)
            features.setdefault("sma_20d_dist", 0.0)

        # --- VIX features ---
        if vix_df is not None and len(vix_df) >= 10:
            vix_closes = vix_df["Close"].values.flatten()
            features.setdefault("vix_level", float(vix_closes[-1]))
            features.setdefault("vix_change_1d",
                                float(vix_closes[-1] - vix_closes[-2]) if len(vix_closes) >= 2 else 0.0)
            features.setdefault("vix_change_5d",
                                float(vix_closes[-1] - vix_closes[-6]) if len(vix_closes) >= 6 else 0.0)
            if len(vix_closes) >= 30:
                rank = float(np.sum(vix_closes[-30:] <= vix_closes[-1]) / 30.0 * 100)
                features.setdefault("vix_rank_30d", rank)
            else:
                features.setdefault("vix_rank_30d", 50.0)
        else:
            features.setdefault("vix_level", 18.5)
            features.setdefault("vix_change_1d", 0.0)
            features.setdefault("vix_change_5d", 0.0)
            features.setdefault("vix_rank_30d", 50.0)

        # vix_term_slope (VIX3M vs VIX)
        vix_now = features.get("vix_level", 18.5)
        if vix3m_df is not None and len(vix3m_df) >= 1:
            vix3m_now = float(vix3m_df["Close"].values.flatten()[-1])
            features.setdefault("vix_term_slope",
                                float((vix3m_now / vix_now - 1)) if vix_now > 0 else 0.0)
        else:
            features.setdefault("vix_term_slope", 0.0)

        # --- Intraday features ---
        if candles_5min is not None and len(candles_5min) >= 2:
            opens = candles_5min["open"].values.astype(float)
            closes = candles_5min["close"].values.astype(float)
            highs = candles_5min["high"].values.astype(float)
            lows = candles_5min["low"].values.astype(float)
            open_price = float(opens[0])

            if open_price > 0:
                features.setdefault("move_from_open_pct",
                                    float((closes[-1] / open_price - 1) * 100))
            else:
                features.setdefault("move_from_open_pct", 0.0)

            # ORB features (first 30 min = 6 bars)
            orb_bars = min(6, len(candles_5min))
            orb_high = float(np.max(highs[:orb_bars]))
            orb_low = float(np.min(lows[:orb_bars]))
            if open_price > 0:
                features.setdefault("orb_range_pct", float((orb_high - orb_low) / open_price * 100))
            else:
                features.setdefault("orb_range_pct", 0.0)

            features.setdefault("orb_contained",
                                1.0 if closes[-1] <= orb_high and closes[-1] >= orb_low else 0.0)

            # range_exhaustion (session range / ATR)
            session_range = float(np.max(highs) - np.min(lows))
            atr_pct = features.get("atr_14d_pct", 0.5)
            if atr_pct > 0 and open_price > 0:
                session_range_pct = session_range / open_price * 100
                features.setdefault("range_exhaustion", float(session_range_pct / atr_pct))
            else:
                features.setdefault("range_exhaustion", 0.0)

            # momentum_30min (last 6 bars return)
            lookback = min(6, len(closes))
            if closes[-lookback] > 0:
                features.setdefault("momentum_30min",
                                    float((closes[-1] / closes[-lookback] - 1) * 100))
            else:
                features.setdefault("momentum_30min", 0.0)

            # trend_slope_norm (linear regression slope of closes, normalized)
            if len(closes) >= 3:
                x = np.arange(len(closes))
                slope = float(np.polyfit(x, closes, 1)[0])
                features.setdefault("trend_slope_norm",
                                    float(slope / np.std(closes)) if np.std(closes) > 0 else 0.0)
            else:
                features.setdefault("trend_slope_norm", 0.0)
        else:
            for key in ["move_from_open_pct", "orb_range_pct", "orb_contained",
                        "range_exhaustion", "momentum_30min", "trend_slope_norm"]:
                features.setdefault(key, 0.0)

        # --- Day-of-week features ---
        now = datetime.now()
        dow = now.weekday()  # 0=Mon, 4=Fri
        features.setdefault("dow_sin", float(np.sin(2 * np.pi * dow / 5)))
        features.setdefault("dow_cos", float(np.cos(2 * np.pi * dow / 5)))

        # --- days_since_fomc ---
        today_date = now.date()
        days_since = 14  # default
        for fomc_date in sorted(_FOMC_DATE_SET, reverse=True):
            if fomc_date <= today_date:
                days_since = (today_date - fomc_date).days
                break
        features.setdefault("days_since_fomc", float(days_since))

        return features

    def predict_v5(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
        target: str = "tp25",
        delta: float = 0.10,
    ) -> float:
        """
        Predict P(hit TP) for entry timing.

        Args:
            candles_5min: 5-min OHLC candles from 9:30 to now.
            daily_data: Optional pre-computed day-level features.
            option_atm_iv: ATM implied volatility.
            target: "tp25" for P(hit 25% TP) or "tp50" for P(hit 50% TP).
            delta: Short delta level (0.10, 0.15, or 0.20). Uses delta-specific
                   model if available, falls back to 10-delta model.

        Returns:
            P(hit TP) score (0-1). Higher = more likely to hit take-profit.
        """
        # Try delta-specific model first (for non-default deltas)
        if delta != 0.10 and (target, delta) in self._v5_delta_models:
            model, feature_names = self._v5_delta_models[(target, delta)]
        elif target == "tp25":
            model = self.v5_tp25_model
            feature_names = self.v5_tp25_feature_names
        elif target == "tp50":
            model = self.v5_tp50_model
            feature_names = self.v5_tp50_feature_names
        else:
            raise ValueError(f"Invalid v5 target: {target}. Use 'tp25' or 'tp50'.")

        if model is None:
            raise ValueError(f"v5 {target} model not loaded (delta={delta})")

        features = self.compute_v5_features(candles_5min, daily_data, option_atm_iv)

        # Build feature vector in model order
        feature_array = np.array([
            features.get(name, 0.0) if not np.isnan(features.get(name, 0.0))
            else 0.0
            for name in feature_names
        ]).reshape(1, -1)

        return float(model.predict_proba(feature_array)[0][1])

    def predict_v5_all_deltas(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
    ) -> Dict[float, Dict[str, float]]:
        """
        Score all loaded delta models. Computes features ONCE.

        Returns:
            {delta: {"tp25": score, "tp50": score}} for each loaded delta.
            Missing models get None values.
        """
        deltas = self.available_v5_deltas
        if not deltas:
            return {}

        # Compute features once (shared across all deltas)
        features = self.compute_v5_features(candles_5min, daily_data, option_atm_iv)

        results = {}
        for delta in deltas:
            scores = {}
            for target in ["tp25", "tp50"]:
                # Select model for this (target, delta)
                if delta != 0.10 and (target, delta) in self._v5_delta_models:
                    model, feature_names = self._v5_delta_models[(target, delta)]
                elif delta == 0.10 and target == "tp25":
                    model = self.v5_tp25_model
                    feature_names = self.v5_tp25_feature_names
                elif delta == 0.10 and target == "tp50":
                    model = self.v5_tp50_model
                    feature_names = self.v5_tp50_feature_names
                else:
                    model = None
                    feature_names = None

                if model is not None and feature_names is not None:
                    try:
                        feature_array = np.array([
                            features.get(name, 0.0) if not np.isnan(features.get(name, 0.0))
                            else 0.0
                            for name in feature_names
                        ]).reshape(1, -1)
                        scores[target] = float(model.predict_proba(feature_array)[0][1])
                    except Exception:
                        scores[target] = None
                else:
                    scores[target] = None

            results[delta] = scores

        return results

    def predict_v6_ensemble(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
        target: str = "tp25",
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, any]:
        """
        Predict using v6 ensemble (4-window weighted average).

        Args:
            candles_5min: 5-min OHLC candles from 9:30 to now.
            daily_data: Optional pre-computed day-level features.
            option_atm_iv: ATM implied volatility.
            target: "tp25" or "tp50".
            weights: Optional custom weights. Default: {'1y': 0.4, '6m': 0.3, '3m': 0.2, '1m': 0.1}

        Returns:
            Dict with:
                - ensemble_prob: Weighted ensemble prediction (0-1)
                - individual_probs: Dict of {window: prob} for each window
                - consensus_count: Number of models predicting >= 0.5
                - disagreement: True if std(probs) > 0.15
                - std: Standard deviation of individual predictions
        """
        if weights is None:
            weights = {'3y': 0.20, '2y': 0.20, '1y': 0.20, '6m': 0.15, '3m': 0.15, '1m': 0.10}

        # Check if target models are loaded
        if target not in ['tp25', 'tp50']:
            raise ValueError(f"Invalid target: {target}. Use 'tp25' or 'tp50'.")

        models = self.v6_ensemble.get(target, {})
        if len(models) != self.V6_EXPECTED_MODELS:
            raise ValueError(
                f"v6 {target} ensemble incomplete: {len(models)}/{self.V6_EXPECTED_MODELS} models loaded"
            )

        # Compute full v6 feature set (41 features)
        features = self.compute_v6_features(candles_5min, daily_data, option_atm_iv)

        # Build feature vector
        feature_array = np.array([
            features.get(name, 0.0) if not np.isnan(features.get(name, 0.0))
            else 0.0
            for name in self.v6_feature_names
        ]).reshape(1, -1)

        # Get predictions from all windows
        probs = {}
        for window in self.V6_WINDOWS:
            model = models[window]
            probs[window] = float(model.predict_proba(feature_array)[0][1])

        # Compute ensemble
        ensemble_prob = sum(probs[w] * weights[w] for w in weights)
        consensus_count = sum(1 for p in probs.values() if p >= 0.5)
        std = np.std(list(probs.values()))
        disagreement = std > 0.15

        return {
            'ensemble_prob': ensemble_prob,
            'individual_probs': probs,
            'consensus_count': consensus_count,
            'disagreement': disagreement,
            'std': std
        }

    # ── v7 Long IC Models ────────────────────────────────────────────────────

    def _load_v7_models(self):
        """Load v7 long IC models from ml/artifacts/models/v7/ directory."""
        v7_dir = self.model_path.parent / "v7"
        if not v7_dir.exists():
            logger.debug("v7 long IC model directory not found, skipping")
            return

        for delta, suffix in [(0.10, "d10"), (0.15, "d15"), (0.20, "d20")]:
            path = v7_dir / f"long_ic_{suffix}.joblib"
            if path.exists():
                try:
                    artifact = joblib.load(path)
                    model = artifact["model"]
                    feature_names = artifact.get("feature_names", [])
                    self._v7_models[delta] = (model, feature_names)
                    logger.info(
                        f"v7 long IC {suffix} model loaded: {path.name} "
                        f"({len(feature_names)} features)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load v7 {suffix} model: {e}")

    @property
    def v7_ready(self) -> bool:
        """Check if any v7 long IC model is loaded."""
        return len(self._v7_models) > 0

    @property
    def available_v7_deltas(self) -> list:
        """Return sorted list of deltas with loaded v7 models."""
        return sorted(self._v7_models.keys())

    def predict_v7(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
        delta: float = 0.20,
        v3_confidence: Optional[float] = None,
    ) -> float:
        """
        Predict P(long IC profitable) for a given delta.

        Uses the same v4/v6 feature pipeline plus v3_confidence as meta-feature.

        Args:
            candles_5min: 5-min OHLC candles from 9:30 to now.
            daily_data: Optional pre-computed day-level features.
            option_atm_iv: ATM implied volatility.
            delta: Target delta (0.10, 0.15, or 0.20).
            v3_confidence: Pre-computed v3 confidence score (meta-feature).

        Returns:
            P(long IC profitable) score (0-1). Higher = better for long IC.
        """
        if delta not in self._v7_models:
            raise ValueError(
                f"v7 model not loaded for delta={delta}. "
                f"Available: {self.available_v7_deltas}"
            )

        model, feature_names = self._v7_models[delta]

        # Compute features using existing v4 pipeline (same features)
        features = self.compute_v4_features(candles_5min, daily_data, option_atm_iv)

        # Add v3_confidence as meta-feature
        if v3_confidence is not None:
            features["v3_confidence"] = v3_confidence
        else:
            features["v3_confidence"] = 0.5  # Neutral default

        # Add any additional features that might be in the v7 model but not in v4
        # These come from the training data columns
        if daily_data:
            for key in ["rv_close_5d", "rv_close_10d", "rv_close_20d",
                        "vix_change_1d", "vix_change_5d", "vix_rank_30d",
                        "vix_level", "sma_20d_dist", "prior_day_return",
                        "iv_skew_10d", "move_from_open_pct",
                        "orb_contained", "range_exhaustion",
                        "is_friday", "days_since_fomc"]:
                if key in daily_data and key not in features:
                    features[key] = daily_data[key]

        # Economic calendar features
        from ml.models.predictor import get_economic_calendar_features
        econ = get_economic_calendar_features()
        features.update(econ)

        # Day-of-week features
        from datetime import datetime as dt_cls
        now = dt_cls.now()
        dow = now.weekday()
        features.setdefault("dow_sin", float(np.sin(2 * np.pi * dow / 5)))
        features.setdefault("dow_cos", float(np.cos(2 * np.pi * dow / 5)))
        features.setdefault("is_friday", 1.0 if dow == 4 else 0.0)

        # Build feature vector in model order
        feature_array = np.array([
            features.get(name, 0.0)
            if not (isinstance(features.get(name, 0.0), float) and np.isnan(features.get(name, 0.0)))
            else 0.0
            for name in feature_names
        ]).reshape(1, -1)

        return float(model.predict_proba(feature_array)[0][1])

    def predict_v7_all_deltas(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
        v3_confidence: Optional[float] = None,
    ) -> Dict[float, float]:
        """
        Score all loaded v7 deltas and return {delta: P(long_IC_profit)}.

        Returns:
            Dict mapping delta to v7 score, e.g. {0.10: 0.45, 0.15: 0.52, 0.20: 0.61}
        """
        scores = {}
        for delta in self.available_v7_deltas:
            try:
                scores[delta] = self.predict_v7(
                    candles_5min, daily_data, option_atm_iv, delta, v3_confidence
                )
            except Exception as e:
                logger.warning(f"v7 prediction failed for delta={delta}: {e}")
        return scores

    # ── Unified scoring for TradeDecisionEngine ─────────────────────────

    def predict_all_strategies(
        self,
        candles_5min: pd.DataFrame,
        daily_data: Optional[dict] = None,
        option_atm_iv: float = 15.0,
        v3_confidence: Optional[float] = None,
    ) -> dict:
        """
        Gather all model scores needed by the unified TradeDecisionEngine.

        Returns dict with:
            "v3_confidence": float,
            "v5_scores": {delta: {"tp25": float, "tp50": float}, ...} or None,
            "v7_scores": {delta: float, ...} or None,
        """
        # Get v3 confidence if not provided
        if v3_confidence is None:
            if self.model_ready:
                v3_confidence = self.predict(daily_data or {})
            else:
                v3_confidence = 0.5  # neutral default

        result = {"v3_confidence": v3_confidence, "v5_scores": None, "v7_scores": None}

        # Score v5 per-delta (short IC TP probabilities)
        if self.v5_ready:
            try:
                v5_all = self.predict_v5_all_deltas(candles_5min, daily_data, option_atm_iv)
                if v5_all:
                    result["v5_scores"] = v5_all
            except Exception as e:
                logger.warning(f"v5 scoring failed: {e}")

        # Score v7 per-delta (long IC profitability)
        if self.v7_ready:
            try:
                v7_all = self.predict_v7_all_deltas(candles_5min, daily_data, option_atm_iv, v3_confidence)
                if v7_all:
                    result["v7_scores"] = v7_all
            except Exception as e:
                logger.warning(f"v7 scoring failed: {e}")

        return result

    # ── v8 XGBoost models (200+ features) ─────────────────────────────────

    def _load_v8_models(self):
        """Load v8 XGBoost models and feature names from ml/artifacts/models/v8/."""
        v8_dir = self.model_path.parent / "v8"
        if not v8_dir.exists():
            logger.debug("v8 model directory not found, skipping")
            return

        # Load feature names (shared across all v8 models)
        feature_names_path = v8_dir / "short_ic" / "feature_names.json"
        if not feature_names_path.exists():
            feature_names_path = v8_dir / "long_ic" / "feature_names.json"
        if feature_names_path.exists():
            import json as _json
            with open(feature_names_path) as f:
                self._v8_feature_names = _json.load(f)
            logger.info(f"v8 feature names loaded: {len(self._v8_feature_names)} features")
        else:
            logger.warning("v8 feature_names.json not found — cannot load v8 models")
            return

        # Load short IC models
        short_dir = v8_dir / "short_ic"
        if short_dir.exists():
            for target in ["tp25", "tp50", "big_loss"]:
                for delta_str, delta_val in [("d10", 0.10), ("d15", 0.15), ("d20", 0.20)]:
                    path = short_dir / f"v8_short_ic_{delta_str}_{target}_full.joblib"
                    if path.exists():
                        try:
                            model = joblib.load(path)
                            self._v8_short_models[(target, delta_val)] = model
                            logger.info(f"v8 short IC {delta_str}/{target} loaded: {path.name}")
                        except Exception as e:
                            logger.warning(f"Failed to load v8 short {delta_str}/{target}: {e}")

        # Load long IC models
        long_dir = v8_dir / "long_ic"
        if long_dir.exists():
            for target in ["profitable", "big_move"]:
                for delta_str, delta_val in [("d10", 0.10), ("d15", 0.15), ("d20", 0.20)]:
                    path = long_dir / f"v8_long_ic_{delta_str}_{target}_full.joblib"
                    if path.exists():
                        try:
                            model = joblib.load(path)
                            self._v8_long_models[(target, delta_val)] = model
                            logger.info(f"v8 long IC {delta_str}/{target} loaded: {path.name}")
                        except Exception as e:
                            logger.warning(f"Failed to load v8 long {delta_str}/{target}: {e}")

        if self._v8_short_models or self._v8_long_models:
            logger.info(
                f"v8 models ready: {len(self._v8_short_models)} short, "
                f"{len(self._v8_long_models)} long"
            )

    @property
    def v8_ready(self) -> bool:
        """Check if any v8 model is loaded with feature names."""
        return self._v8_feature_names is not None and (
            len(self._v8_short_models) > 0 or len(self._v8_long_models) > 0
        )

    @property
    def v8_short_ready(self) -> bool:
        return self._v8_feature_names is not None and len(self._v8_short_models) > 0

    @property
    def v8_long_ready(self) -> bool:
        return self._v8_feature_names is not None and len(self._v8_long_models) > 0

    @property
    def available_v8_short_deltas(self) -> list:
        """Return sorted list of deltas with loaded v8 short IC tp25 models."""
        return sorted(set(d for (t, d) in self._v8_short_models if t == "tp25"))

    @property
    def available_v8_long_deltas(self) -> list:
        """Return sorted list of deltas with loaded v8 long IC profitable models."""
        return sorted(set(d for (t, d) in self._v8_long_models if t == "profitable"))

    def _v8_features_to_array(self, features_dict: dict) -> np.ndarray:
        """Convert v8 feature dict to ordered numpy array matching training order."""
        arr = np.array([
            features_dict.get(name, -999.0)
            for name in self._v8_feature_names
        ], dtype=np.float64)
        # Replace NaN with -999 (same as training preprocessing)
        arr = np.where(np.isnan(arr), -999.0, arr)
        arr = np.where(np.isinf(arr), -999.0, arr)
        return arr.reshape(1, -1)

    def predict_v8_short(
        self,
        v8_features: dict,
        delta: float = 0.10,
        target: str = "tp25",
    ) -> float:
        """
        Predict using v8 short IC model.

        Args:
            v8_features: dict from compute_all_features() in v8 feature store.
            delta: Target delta (0.10, 0.15, 0.20).
            target: "tp25", "tp50", or "big_loss".

        Returns:
            P(target) score (0-1).
        """
        key = (target, delta)
        if key not in self._v8_short_models:
            raise ValueError(f"v8 short model not loaded for {target}@{delta}δ")

        model = self._v8_short_models[key]
        X = self._v8_features_to_array(v8_features)
        return float(model.predict_proba(X)[0][1])

    def predict_v8_long(
        self,
        v8_features: dict,
        delta: float = 0.10,
        target: str = "profitable",
    ) -> float:
        """
        Predict using v8 long IC model.

        Args:
            v8_features: dict from compute_all_features() in v8 feature store.
            delta: Target delta (0.10, 0.15, 0.20).
            target: "profitable" or "big_move".

        Returns:
            P(target) score (0-1).
        """
        key = (target, delta)
        if key not in self._v8_long_models:
            raise ValueError(f"v8 long model not loaded for {target}@{delta}δ")

        model = self._v8_long_models[key]
        X = self._v8_features_to_array(v8_features)
        return float(model.predict_proba(X)[0][1])

    def predict_v8_all_strategies(
        self,
        v8_features: dict,
        v3_confidence: Optional[float] = None,
    ) -> dict:
        """
        Score all v8 strategies and return scores for the decision engine.

        Returns dict with:
            "v3_confidence": float (from v8 big_loss model or fallback to v3),
            "v5_scores": {delta: {"tp25": float, "tp50": float}, ...},
            "v7_scores": {delta: float, ...},
        """
        # v3 confidence: use v8 big_loss model if available, else fallback
        if v3_confidence is None:
            v3_confidence = 0.5

        # Try to get better v3 from v8 big_loss model
        for delta in self.available_v8_short_deltas:
            if ("big_loss", delta) in self._v8_short_models:
                try:
                    p_big_loss = self.predict_v8_short(v8_features, delta, "big_loss")
                    v3_confidence = 1.0 - p_big_loss  # v3 = 1 - P(big_loss)
                    logger.debug(f"v8 big_loss → v3_confidence={v3_confidence:.3f}")
                except Exception as e:
                    logger.warning(f"v8 big_loss prediction failed: {e}")
                break  # Only need one delta for the risk score

        result = {
            "v3_confidence": v3_confidence,
            "v5_scores": None,
            "v7_scores": None,
        }

        # Short IC scores (replacing v5)
        v5_scores = {}
        for delta in self.available_v8_short_deltas:
            entry = {}
            try:
                entry["tp25"] = self.predict_v8_short(v8_features, delta, "tp25")
            except Exception:
                continue
            try:
                entry["tp50"] = self.predict_v8_short(v8_features, delta, "tp50")
            except Exception:
                entry["tp50"] = 0.0
            v5_scores[delta] = entry
        if v5_scores:
            result["v5_scores"] = v5_scores

        # Long IC scores (replacing v7)
        v7_scores = {}
        for delta in self.available_v8_long_deltas:
            try:
                v7_scores[delta] = self.predict_v8_long(v8_features, delta, "profitable")
            except Exception:
                pass
        if v7_scores:
            result["v7_scores"] = v7_scores

        return result

    # ── v8 Ensemble (6-timeframe weighted) ────────────────────────────────

    def _load_v8_ensemble(self):
        """Load v8 ensemble models from ml/artifacts/models/v8_ensemble/."""
        ens_dir = self.model_path.parent / "v8_ensemble"
        if not ens_dir.exists():
            logger.debug("v8_ensemble directory not found, skipping")
            return

        # Load config
        config_path = ens_dir / "ensemble_config.json"
        if config_path.exists():
            import json as _json
            with open(config_path) as f:
                config = _json.load(f)
            self._v8_ens_weights = config.get("weights", {})
            logger.info(f"v8 ensemble config loaded: {len(self._v8_ens_weights)} windows")
        else:
            logger.warning("v8 ensemble config not found")
            return

        # Load feature names
        fn_path = ens_dir / "feature_names.json"
        if fn_path.exists():
            import json as _json
            with open(fn_path) as f:
                self._v8_ens_feature_names = _json.load(f)
            logger.info(f"v8 ensemble feature names: {len(self._v8_ens_feature_names)}")
        else:
            # Fall back to v8 single model feature names
            if self._v8_feature_names:
                self._v8_ens_feature_names = self._v8_feature_names
            else:
                logger.warning("v8 ensemble feature names not found")
                return

        # Load per-window models
        window_names = list(self._v8_ens_weights.keys())

        # Short IC targets
        for target in ["tp10", "tp25", "tp50", "big_loss"]:
            for window in window_names:
                for delta_str, delta_val in [("d10", 0.10), ("d15", 0.15), ("d20", 0.20)]:
                    path = ens_dir / window / f"v8_short_ic_{delta_str}_{target}.joblib"
                    if path.exists():
                        try:
                            model = joblib.load(path)
                            self._v8_ens_models[("short_ic", target, window, delta_val)] = model
                        except Exception as e:
                            logger.warning(f"Failed to load v8 ens short {window}/{target}: {e}")

        # Long IC targets
        for target in ["tp10", "tp25", "tp50"]:
            for window in window_names:
                for delta_str, delta_val in [("d10", 0.10), ("d15", 0.15), ("d20", 0.20)]:
                    path = ens_dir / window / f"v8_long_ic_{delta_str}_{target}.joblib"
                    if path.exists():
                        try:
                            model = joblib.load(path)
                            self._v8_ens_models[("long_ic", target, window, delta_val)] = model
                        except Exception as e:
                            logger.warning(f"Failed to load v8 ens long {window}/{target}: {e}")

        if self._v8_ens_models:
            n_short = sum(1 for k in self._v8_ens_models if k[0] == "short_ic")
            n_long = sum(1 for k in self._v8_ens_models if k[0] == "long_ic")
            windows_loaded = set(k[2] for k in self._v8_ens_models)
            logger.info(
                f"v8 ensemble ready: {n_short} short + {n_long} long models "
                f"across {len(windows_loaded)} windows ({sorted(windows_loaded)})"
            )

    @property
    def v8_ensemble_ready(self) -> bool:
        """Check if v8 ensemble is loaded."""
        return (
            self._v8_ens_feature_names is not None
            and len(self._v8_ens_models) > 0
            and len(self._v8_ens_weights) > 0
        )

    def _v8_ens_features_to_array(self, features_dict: dict) -> np.ndarray:
        """Convert feature dict to array using ensemble feature names."""
        arr = np.array([
            features_dict.get(name, -999.0)
            for name in self._v8_ens_feature_names
        ], dtype=np.float64)
        arr = np.where(np.isnan(arr), -999.0, arr)
        arr = np.where(np.isinf(arr), -999.0, arr)
        return arr.reshape(1, -1)

    def _predict_v8_ensemble_single(
        self,
        X: np.ndarray,
        strategy: str,
        target: str,
        delta: float,
    ) -> float:
        """
        Weighted average prediction across all loaded timeframe models.

        Automatically re-normalizes weights when some windows are missing.
        Returns P(target) as float.
        """
        predictions = {}
        for (s, t, window, d), model in self._v8_ens_models.items():
            if s == strategy and t == target and d == delta:
                try:
                    prob = float(model.predict_proba(X)[0][1])
                    predictions[window] = prob
                except Exception:
                    pass

        if not predictions:
            raise ValueError(
                f"No v8 ensemble models for {strategy}/{target}@{delta}δ"
            )

        # Weighted average with re-normalization
        active_weights = {w: self._v8_ens_weights[w]
                          for w in predictions if w in self._v8_ens_weights}
        total_w = sum(active_weights.values())
        if total_w <= 0:
            # Equal weight fallback
            return sum(predictions.values()) / len(predictions)

        weighted_sum = sum(
            predictions[w] * (active_weights[w] / total_w)
            for w in predictions if w in active_weights
        )

        logger.debug(
            f"v8 ens {strategy}/{target}@{delta}δ: "
            f"{len(predictions)} windows, "
            f"probs={[f'{w}={p:.3f}' for w, p in sorted(predictions.items())]}, "
            f"weighted={weighted_sum:.4f}"
        )

        return weighted_sum

    def predict_v8_ensemble_all_strategies(
        self,
        v8_features: dict,
        v3_confidence: Optional[float] = None,
    ) -> dict:
        """
        Score all v8 ensemble strategies. Returns dict compatible with
        TradeDecisionEngine.decide().

        Ensemble predictions are weighted averages across timeframe models.
        Falls back to single v8 models for any missing strategy/delta.
        """
        X = self._v8_ens_features_to_array(v8_features)

        if v3_confidence is None:
            v3_confidence = 0.5

        # Risk score from ensemble big_loss model
        short_deltas = sorted(set(
            d for (s, t, w, d) in self._v8_ens_models
            if s == "short_ic" and t == "big_loss"
        ))
        for delta in short_deltas:
            try:
                p_big_loss = self._predict_v8_ensemble_single(X, "short_ic", "big_loss", delta)
                v3_confidence = 1.0 - p_big_loss
                logger.debug(f"v8 ens big_loss → v3_confidence={v3_confidence:.3f}")
            except Exception:
                pass
            break

        result = {"v3_confidence": v3_confidence, "v5_scores": None, "v7_scores": None}

        # Short IC scores (ensemble replaces v5)
        # Find all deltas that have at least tp25 models
        short_deltas = sorted(set(
            d for (s, t, w, d) in self._v8_ens_models
            if s == "short_ic" and t == "tp25"
        ))
        v5_scores = {}
        for delta in short_deltas:
            entry = {}
            try:
                entry["tp25"] = self._predict_v8_ensemble_single(X, "short_ic", "tp25", delta)
            except Exception:
                continue
            for tp_target in ["tp10", "tp50"]:
                try:
                    entry[tp_target] = self._predict_v8_ensemble_single(X, "short_ic", tp_target, delta)
                except Exception:
                    pass
            v5_scores[delta] = entry
        if v5_scores:
            result["v5_scores"] = v5_scores

        # Long IC scores (ensemble replaces v7)
        # Use tp25 as the primary "profitable" signal for v7_scores compatibility
        long_deltas = sorted(set(
            d for (s, t, w, d) in self._v8_ens_models
            if s == "long_ic" and t in ("tp25", "tp10")
        ))
        v7_scores = {}
        v7_detail = {}
        for delta in long_deltas:
            detail = {}
            for tp_target in ["tp10", "tp25", "tp50"]:
                try:
                    detail[tp_target] = self._predict_v8_ensemble_single(
                        X, "long_ic", tp_target, delta
                    )
                except Exception:
                    pass
            if detail:
                # v7_scores expects a single float per delta — use tp25 as primary
                v7_scores[delta] = detail.get("tp25", detail.get("tp10", 0.0))
                v7_detail[delta] = detail
        if v7_scores:
            result["v7_scores"] = v7_scores
            result["v7_detail"] = v7_detail

        return result
