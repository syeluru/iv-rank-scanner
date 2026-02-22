"""
Technical indicator calculations using pandas-TA.

This module manages rolling DataFrames for symbols and calculates
technical indicators on-demand for strategy condition checking.

Includes live IV (Implied Volatility) calculation integration.
"""

import pandas as pd
import pandas_ta as ta
import warnings
import asyncio
from typing import Dict, Optional, List
from datetime import datetime, time, date
from loguru import logger

from config.settings import settings


class TechnicalIndicators:
    """
    Manages technical indicator calculations for symbols.

    Maintains rolling DataFrames and calculates indicators on each bar update.
    Provides fast access to latest indicator values for strategy logic.
    Includes live IV data from options chain.
    """

    def __init__(self, lookback_bars: int = None):
        """
        Initialize technical indicators manager.

        Args:
            lookback_bars: Number of historical bars to maintain (default from settings)
        """
        self.lookback_bars = lookback_bars or settings.MONITORING_LOOKBACK_BARS
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._market_open_time = time(9, 30)  # 9:30 AM ET
        self._opening_range_minutes = settings.ORB_OPENING_RANGE_MINUTES

        # FIX #1: Separate storage for daily reference data (prevents data amnesia)
        self._daily_data: Dict[str, pd.DataFrame] = {}  # Keeps ALL data from market open
        self._cached_levels: Dict[str, Dict] = {}  # Cached opening range levels per symbol
        self._last_reset_date: Optional[date] = None

        # FIX #2: 5-minute bar aggregation (prevents fakeout risk)
        self._current_5min_bars: Dict[str, List[Dict]] = {}  # Accumulates 5-sec bars into 5-min
        self._completed_5min_bars: Dict[str, pd.DataFrame] = {}  # Completed 5-min bars

        # FIX #3: Volume tracking for proper calculation
        self._minute_volume: Dict[str, Dict] = {}  # Track volume by minute for proper aggregation

        # Live IV data storage
        self._live_iv: Dict[str, float] = {}  # Current IV per symbol
        self._iv_percentile: Dict[str, float] = {}  # IV percentile per symbol
        self._iv_last_update: Dict[str, datetime] = {}  # Last IV update time
        self._iv_update_interval = 60  # Seconds between IV updates

    def update_bar(self, symbol: str, bar: Dict[str, any]):
        """
        Add new bar to DataFrame and recalculate indicators.

        FIX #1: Maintains separate daily_data for reference levels (opening range)
        FIX #2: Aggregates 5-second bars into 5-minute bars
        FIX #3: Tracks volume properly by minute

        Args:
            symbol: Stock symbol
            bar: Dict with keys: timestamp, open, high, low, close, volume
        """
        try:
            # Track current symbol for indicator calculations (used for IV lookup)
            self._current_symbol = symbol

            # Normalize timestamp to ensure consistency
            bar_timestamp = bar['timestamp']
            if isinstance(bar_timestamp, str):
                bar_timestamp = pd.to_datetime(bar_timestamp)
            elif not isinstance(bar_timestamp, pd.Timestamp):
                bar_timestamp = pd.Timestamp(bar_timestamp)

            # Ensure timezone-aware (use UTC if not already set)
            if bar_timestamp.tz is None:
                bar_timestamp = bar_timestamp.tz_localize('UTC')

            bar_date = bar_timestamp.date()

            # Reset daily data if new trading day
            if self._last_reset_date != bar_date:
                self._reset_daily_data(bar_date)

            # === FIX #1: Store in daily_data (NEVER truncated) ===
            if symbol not in self._daily_data:
                self._daily_data[symbol] = pd.DataFrame()

            daily_df = self._daily_data[symbol]

            new_row_dict = {
                'timestamp': bar_timestamp,
                'open': float(bar['open']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'close': float(bar['close']),
                'volume': int(bar['volume'])
            }

            new_row = pd.DataFrame([new_row_dict])
            daily_df = pd.concat([daily_df, new_row], ignore_index=True)
            self._daily_data[symbol] = daily_df

            # === FIX #2: Aggregate into 5-minute bars ===
            self._aggregate_5min_bar(symbol, new_row_dict)

            # === FIX #3: Track minute-level volume ===
            self._track_minute_volume(symbol, bar_timestamp, bar['volume'])

            # === Update rolling window (for fast indicators like EMA) ===
            if symbol not in self._dataframes:
                self._dataframes[symbol] = pd.DataFrame()

            df = self._dataframes[symbol]
            df = pd.concat([df, new_row], ignore_index=True)

            # Keep only last N bars (for indicator calculation efficiency)
            if len(df) > self.lookback_bars:
                df = df.iloc[-self.lookback_bars:].copy()

            # Calculate indicators on rolling window
            df = self._calculate_all_indicators(df)

            # === FIX #1: Calculate and cache opening range levels ===
            if symbol not in self._cached_levels:
                self._calculate_opening_range_levels(symbol)

            # Add cached levels to latest indicators
            if symbol in self._cached_levels:
                # Add to last row only (not entire DF - performance)
                df.loc[df.index[-1], 'opening_range_high'] = self._cached_levels[symbol].get('high', 0)
                df.loc[df.index[-1], 'opening_range_low'] = self._cached_levels[symbol].get('low', 0)

            self._dataframes[symbol] = df

            logger.debug(
                f"Updated {symbol}: rolling={len(df)} bars, "
                f"daily={len(daily_df)} bars, last close=${df['close'].iloc[-1]:.2f}"
            )

        except Exception as e:
            logger.error(f"Error updating bar for {symbol}: {e}")

    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators using pandas-TA.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with indicators added
        """
        if len(df) < 2:
            return df

        try:
            # === Trend Indicators ===
            df['ema_9'] = ta.ema(df['close'], length=settings.INDICATOR_EMA_FAST)
            df['ema_20'] = ta.ema(df['close'], length=20)
            df['ema_21'] = ta.ema(df['close'], length=settings.INDICATOR_EMA_SLOW)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)

            # === Momentum Indicators ===
            df['rsi'] = ta.rsi(df['close'], length=settings.INDICATOR_RSI_PERIOD)

            # MACD
            macd = ta.macd(
                df['close'],
                fast=settings.INDICATOR_MACD_FAST,
                slow=settings.INDICATOR_MACD_SLOW,
                signal=settings.INDICATOR_MACD_SIGNAL
            )
            if macd is not None:
                df['macd'] = macd[f'MACD_{settings.INDICATOR_MACD_FAST}_{settings.INDICATOR_MACD_SLOW}_{settings.INDICATOR_MACD_SIGNAL}']
                df['macd_signal'] = macd[f'MACDs_{settings.INDICATOR_MACD_FAST}_{settings.INDICATOR_MACD_SLOW}_{settings.INDICATOR_MACD_SIGNAL}']
                df['macd_hist'] = macd[f'MACDh_{settings.INDICATOR_MACD_FAST}_{settings.INDICATOR_MACD_SLOW}_{settings.INDICATOR_MACD_SIGNAL}']

            # Stochastic RSI
            stochrsi = ta.stochrsi(df['close'], length=14)
            if stochrsi is not None:
                df['stochrsi_k'] = stochrsi['STOCHRSIk_14_14_3_3']
                df['stochrsi_d'] = stochrsi['STOCHRSId_14_14_3_3']

            # === Volatility Indicators ===
            # Bollinger Bands
            bbands = ta.bbands(df['close'], length=20, std=2)
            if bbands is not None and not bbands.empty:
                # Get column names dynamically (pandas-TA column names can vary)
                bb_cols = bbands.columns.tolist()
                for col in bb_cols:
                    if 'BBU' in col:
                        df['bb_upper'] = bbands[col]
                    elif 'BBM' in col:
                        df['bb_middle'] = bbands[col]
                    elif 'BBL' in col:
                        df['bb_lower'] = bbands[col]
                    elif 'BBB' in col:
                        df['bb_width'] = bbands[col]

            # ATR
            atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
            if atr_result is not None:
                df['atr'] = atr_result

            # === Volume Indicators ===
            df['volume_sma'] = ta.sma(df['volume'], length=20)

            # VWAP - Use simple calculation (avoids DatetimeIndex issues)
            # Formula: Cumulative(Price × Volume) / Cumulative(Volume)
            # This is mathematically correct and more robust than pandas-ta version
            if len(df) > 0 and 'volume' in df.columns:
                # Use typical price (HLC/3) for VWAP calculation
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            else:
                df['vwap'] = df['close']  # Fallback if no volume data

            # === Custom Indicators ===
            # Opening Range High/Low (first N minutes of trading day)
            df = self._calculate_opening_range(df)

            # Swing highs and lows for divergence detection
            df = self._calculate_swing_points(df)

            # SuperTrend (only calculate if we have enough data)
            if len(df) >= 10:
                supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
                if supertrend is not None and not supertrend.empty:
                    # Get column names dynamically
                    st_cols = supertrend.columns.tolist()
                    for col in st_cols:
                        if 'SUPERT_' in col and 'd' not in col:
                            df['supertrend'] = supertrend[col]
                        elif 'SUPERTd_' in col:
                            df['supertrend_direction'] = supertrend[col]

            # Keltner Channels (for TTM Squeeze)
            if len(df) >= 20:
                kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
                if kc is not None and not kc.empty:
                    kc_cols = kc.columns.tolist()
                    for col in kc_cols:
                        if 'KCU' in col:
                            df['kc_upper'] = kc[col]
                        elif 'KCM' in col:
                            df['kc_middle'] = kc[col]
                        elif 'KCL' in col:
                            df['kc_lower'] = kc[col]

            # TTM Squeeze indicator (BB inside KC = squeeze ON)
            if 'bb_upper' in df.columns and 'kc_upper' in df.columns:
                df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])

            # ADX (trend strength)
            if len(df) >= 14:
                adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
                if adx_result is not None and not adx_result.empty:
                    adx_cols = adx_result.columns.tolist()
                    for col in adx_cols:
                        if col.startswith('ADX_'):
                            df['adx'] = adx_result[col]

            # === Additional SMAs for various strategies ===
            df['sma_5'] = ta.sma(df['close'], length=5)

            # === Bollinger Band Width Average (for Gamma Scalping) ===
            if 'bb_width' in df.columns:
                df['bb_width_avg'] = ta.sma(df['bb_width'], length=20)

            # === Rolling High/Low (for various strategies) ===
            if len(df) >= 5:
                df['high_5d'] = df['high'].rolling(window=5).max()
                df['low_5d'] = df['low'].rolling(window=5).min()

            # === Prior Day High/Low (simplified - uses rolling) ===
            # In production, this should come from daily bars
            # For intraday, we approximate with previous session data
            if len(df) >= 78:  # ~6.5 hours of 5-min bars
                # Use bars from roughly yesterday (simplified)
                df['prior_day_high'] = df['high'].shift(78).rolling(window=78).max()
                df['prior_day_low'] = df['low'].shift(78).rolling(window=78).min()
            else:
                df['prior_day_high'] = df['high'].iloc[0] if len(df) > 0 else 0
                df['prior_day_low'] = df['low'].iloc[0] if len(df) > 0 else 0

            # === Open and Previous Close ===
            if len(df) > 0:
                df['open'] = df['open']  # Already exists, just ensure it's accessible
                df['prev_close'] = df['close'].shift(1)

            # === Live IV Data ===
            # Uses real IV from options chain when available, otherwise placeholder
            # IV is updated asynchronously via update_iv() method
            symbol = df['timestamp'].iloc[0] if 'timestamp' in df.columns else None

            # Try to get symbol from the calling context (stored when update_bar is called)
            if hasattr(self, '_current_symbol'):
                symbol = self._current_symbol

            if symbol and symbol in self._live_iv:
                df['iv'] = self._live_iv[symbol]
                df['iv_percentile'] = self._iv_percentile.get(symbol, 50.0)
                # IV from 5 days ago (approximated from history if available)
                df['iv_percentile_5d'] = self._iv_percentile.get(f"{symbol}_5d", self._iv_percentile.get(symbol, 50.0))
            else:
                # Placeholder values when no live IV available
                df['iv'] = 25.0  # Neutral IV
                df['iv_percentile'] = 50.0
                df['iv_percentile_5d'] = 50.0

            # Candlestick patterns
            df = self._detect_candlestick_patterns(df)

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return df

    def _calculate_opening_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate opening range high and low.

        Opening range is the high/low of the first N minutes after market open.

        Args:
            df: DataFrame with timestamp and OHLC data

        Returns:
            DataFrame with opening_range_high and opening_range_low columns
        """
        if 'timestamp' not in df.columns or len(df) < 2:
            df['opening_range_high'] = df.get('high', 0)
            df['opening_range_low'] = df.get('low', 0)
            return df

        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Find market open for each day
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.time

            # Initialize columns
            df['opening_range_high'] = 0.0
            df['opening_range_low'] = 0.0

            # Calculate for each trading day
            for date_val in df['date'].unique():
                day_mask = df['date'] == date_val
                day_df = df[day_mask].copy()

                # Find bars within opening range period
                # Assuming market opens at 9:30 AM ET
                opening_range_end = datetime.combine(
                    date_val,
                    time(9, 30)
                ) + pd.Timedelta(minutes=self._opening_range_minutes)

                # Convert to timezone-aware if df timestamps are timezone-aware
                if pd.api.types.is_datetime64tz_dtype(day_df['timestamp']):
                    # Make opening_range_end timezone-aware (UTC)
                    opening_range_end = pd.Timestamp(opening_range_end).tz_localize('UTC')

                or_mask = day_df['timestamp'] <= opening_range_end

                if or_mask.any():
                    or_high = day_df.loc[or_mask, 'high'].max()
                    or_low = day_df.loc[or_mask, 'low'].min()

                    # Apply to all bars in the day
                    df.loc[day_mask, 'opening_range_high'] = or_high
                    df.loc[day_mask, 'opening_range_low'] = or_low

            # Clean up temporary columns
            df = df.drop(['date', 'time'], axis=1, errors='ignore')

        except Exception as e:
            logger.error(f"Error calculating opening range: {e}")
            df['opening_range_high'] = df.get('high', 0)
            df['opening_range_low'] = df.get('low', 0)

        return df

    def _calculate_swing_points(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Calculate swing highs and lows for divergence detection.

        Args:
            df: DataFrame with OHLC data
            window: Number of bars on each side to confirm swing point

        Returns:
            DataFrame with swing_high and swing_low columns
        """
        if len(df) < window * 2 + 1:
            df['swing_high'] = False
            df['swing_low'] = False
            return df

        try:
            df['swing_high'] = False
            df['swing_low'] = False

            # Identify swing highs
            for i in range(window, len(df) - window):
                is_swing_high = True
                is_swing_low = True

                center_high = df['high'].iloc[i]
                center_low = df['low'].iloc[i]

                # Check if it's a swing high
                for j in range(i - window, i + window + 1):
                    if j != i and df['high'].iloc[j] >= center_high:
                        is_swing_high = False
                        break

                # Check if it's a swing low
                for j in range(i - window, i + window + 1):
                    if j != i and df['low'].iloc[j] <= center_low:
                        is_swing_low = False
                        break

                df.loc[df.index[i], 'swing_high'] = is_swing_high
                df.loc[df.index[i], 'swing_low'] = is_swing_low

        except Exception as e:
            logger.error(f"Error calculating swing points: {e}")

        return df

    def get_latest_indicators(self, symbol: str) -> Optional[Dict[str, any]]:
        """
        Get latest indicator values for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict of indicator values or None if not available
        """
        if symbol not in self._dataframes:
            return None

        df = self._dataframes[symbol]
        if df.empty:
            return None

        # Return last row as dict
        return df.iloc[-1].to_dict()

    def get_dataframe(self, symbol: str, bars: int = None) -> Optional[pd.DataFrame]:
        """
        Get DataFrame for symbol.

        Args:
            symbol: Stock symbol
            bars: Number of recent bars (None for all)

        Returns:
            DataFrame or None
        """
        if symbol not in self._dataframes:
            return None

        df = self._dataframes[symbol]

        if bars and len(df) > bars:
            return df.iloc[-bars:].copy()

        return df.copy()

    def get_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Get current RSI value."""
        indicators = self.get_latest_indicators(symbol)
        return indicators.get('rsi') if indicators else None

    def get_ema(self, symbol: str, period: int = 9) -> Optional[float]:
        """Get current EMA value."""
        indicators = self.get_latest_indicators(symbol)
        return indicators.get(f'ema_{period}') if indicators else None

    def get_vwap(self, symbol: str) -> Optional[float]:
        """Get current VWAP value."""
        indicators = self.get_latest_indicators(symbol)
        return indicators.get('vwap') if indicators else None

    def get_atr(self, symbol: str) -> Optional[float]:
        """Get current ATR value."""
        indicators = self.get_latest_indicators(symbol)
        return indicators.get('atr') if indicators else None

    def get_bollinger_bands(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current Bollinger Bands values."""
        indicators = self.get_latest_indicators(symbol)
        if not indicators:
            return None

        return {
            'upper': indicators.get('bb_upper'),
            'middle': indicators.get('bb_middle'),
            'lower': indicators.get('bb_lower'),
            'width': indicators.get('bb_width')
        }

    def check_rsi_divergence(
        self,
        symbol: str,
        lookback: int = 20,
        divergence_type: str = 'bullish'
    ) -> bool:
        """
        Check for RSI divergence.

        Args:
            symbol: Stock symbol
            lookback: Number of bars to look back
            divergence_type: 'bullish' or 'bearish'

        Returns:
            True if divergence detected
        """
        df = self.get_dataframe(symbol, bars=lookback)
        if df is None or len(df) < lookback:
            return False

        try:
            # Get swing lows/highs
            swing_lows = df[df['swing_low'] == True]
            swing_highs = df[df['swing_high'] == True]

            if divergence_type == 'bullish':
                # Need at least 2 swing lows
                if len(swing_lows) < 2:
                    return False

                # Get last 2 swing lows
                last_two = swing_lows.tail(2)
                first_low = last_two.iloc[0]
                second_low = last_two.iloc[1]

                # Price makes lower low, but RSI makes higher low
                price_lower_low = second_low['low'] < first_low['low']
                rsi_higher_low = second_low['rsi'] > first_low['rsi']

                return price_lower_low and rsi_higher_low

            elif divergence_type == 'bearish':
                # Need at least 2 swing highs
                if len(swing_highs) < 2:
                    return False

                # Get last 2 swing highs
                last_two = swing_highs.tail(2)
                first_high = last_two.iloc[0]
                second_high = last_two.iloc[1]

                # Price makes higher high, but RSI makes lower high
                price_higher_high = second_high['high'] > first_high['high']
                rsi_lower_high = second_high['rsi'] < first_high['rsi']

                return price_higher_high and rsi_lower_high

        except Exception as e:
            logger.error(f"Error checking divergence: {e}")

        return False

    def has_sufficient_data(self, symbol: str, min_bars: int = 50) -> bool:
        """
        Check if symbol has sufficient data for indicator calculations.

        Args:
            symbol: Stock symbol
            min_bars: Minimum required bars

        Returns:
            True if sufficient data available
        """
        if symbol not in self._dataframes:
            return False

        return len(self._dataframes[symbol]) >= min_bars

    def clear_symbol(self, symbol: str):
        """
        Clear data for a symbol.

        Args:
            symbol: Stock symbol
        """
        if symbol in self._dataframes:
            del self._dataframes[symbol]
            logger.info(f"Cleared data for {symbol}")

    def clear_all(self):
        """Clear all symbol data."""
        self._dataframes.clear()
        self._daily_data.clear()
        self._cached_levels.clear()
        self._current_5min_bars.clear()
        self._completed_5min_bars.clear()
        self._minute_volume.clear()
        logger.info("Cleared all indicator data")

    # === FIX #1: Opening Range Level Management ===

    def _reset_daily_data(self, new_date: datetime.date):
        """
        Reset daily data structures for new trading day.

        Args:
            new_date: The new trading date
        """
        logger.info(f"Resetting daily data for {new_date}")
        self._last_reset_date = new_date
        self._daily_data.clear()
        self._cached_levels.clear()
        self._current_5min_bars.clear()
        self._minute_volume.clear()

    def _calculate_opening_range_levels(self, symbol: str):
        """
        Calculate and cache opening range high/low.

        FIX #1: Uses daily_data (never truncated) instead of rolling window.
        Only calculates once after opening range period completes.

        Args:
            symbol: Stock symbol
        """
        if symbol not in self._daily_data:
            return

        daily_df = self._daily_data[symbol]
        if daily_df.empty:
            return

        # Calculate opening range only after enough time has passed
        now = datetime.now().time()
        market_open = time(9, 30)  # 9:30 AM ET
        or_end_minutes = self._opening_range_minutes

        # Check if we're past the opening range period
        first_timestamp = daily_df['timestamp'].iloc[0]
        latest_timestamp = daily_df['timestamp'].iloc[-1]

        time_since_open = (latest_timestamp - first_timestamp).total_seconds() / 60

        if time_since_open < or_end_minutes:
            # Not enough data yet
            return

        # If already cached, don't recalculate (saves CPU)
        if symbol in self._cached_levels:
            return

        # Filter bars within opening range period
        or_end_time = first_timestamp + pd.Timedelta(minutes=or_end_minutes)

        # Ensure timezone compatibility
        if pd.api.types.is_datetime64tz_dtype(daily_df['timestamp']) and not hasattr(or_end_time, 'tz'):
            # Convert or_end_time to timezone-aware if needed
            or_end_time = pd.Timestamp(or_end_time)

        or_bars = daily_df[daily_df['timestamp'] <= or_end_time]

        if len(or_bars) < 2:
            return

        # Calculate levels from daily_data (which has ALL bars from market open)
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()

        # Cache the levels (CRITICAL: prevents recalculation and data loss)
        self._cached_levels[symbol] = {
            'high': or_high,
            'low': or_low,
            'calculated_at': latest_timestamp,
            'bar_count': len(or_bars)
        }

        logger.info(
            f"{symbol} Opening Range calculated: High=${or_high:.2f}, Low=${or_low:.2f} "
            f"(from {len(or_bars)} bars)"
        )

    def get_opening_range(self, symbol: str) -> Optional[Dict]:
        """
        Get cached opening range levels.

        Returns:
            Dict with 'high' and 'low' keys, or None if not calculated yet
        """
        return self._cached_levels.get(symbol)

    def get_daily_open(self, symbol: str) -> Optional[float]:
        """
        Get today's opening price for a symbol.

        Returns:
            Opening price or None if not available
        """
        if symbol not in self._daily_data:
            return None

        daily_df = self._daily_data[symbol]
        if daily_df.empty:
            return None

        # Return the open price of the first bar of the day
        return float(daily_df['open'].iloc[0])

    def get_prior_day_levels(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get prior day's high, low, and close.

        Returns:
            Dict with 'high', 'low', 'close' keys or None
        """
        # Check if we have cached previous day data
        if hasattr(self, '_prev_day_data') and symbol in self._prev_day_data:
            return self._prev_day_data[symbol]

        # Fallback to indicator values
        inds = self.get_latest_indicators(symbol)
        if not inds:
            return None

        return {
            'high': inds.get('prior_day_high', 0),
            'low': inds.get('prior_day_low', 0),
            'close': inds.get('prev_close', 0)
        }

    def set_prior_day_data(self, symbol: str, high: float, low: float, close: float):
        """
        Store prior day's OHLC data for accurate level calculations.

        Args:
            symbol: Stock symbol
            high: Prior day high
            low: Prior day low
            close: Prior day close
        """
        if not hasattr(self, '_prev_day_data'):
            self._prev_day_data = {}

        self._prev_day_data[symbol] = {
            'high': high,
            'low': low,
            'close': close
        }
        logger.debug(f"Set prior day data for {symbol}: H=${high:.2f} L=${low:.2f} C=${close:.2f}")

    # === FIX #2: 5-Minute Bar Aggregation ===

    def _aggregate_5min_bar(self, symbol: str, bar: Dict):
        """
        Aggregate 5-second bars into 5-minute bars.

        FIX #2: Prevents triggering on 5-second wicks.
        Only strategy checks should use completed 5-minute bars.

        Args:
            symbol: Stock symbol
            bar: 5-second bar data
        """
        if symbol not in self._current_5min_bars:
            self._current_5min_bars[symbol] = []

        # Add this 5-second bar to current accumulation
        self._current_5min_bars[symbol].append(bar)

        # Check if 5 minutes have passed (60 bars × 5 seconds = 300 seconds = 5 minutes)
        bars = self._current_5min_bars[symbol]

        if len(bars) >= 60:  # 60 five-second bars = 5 minutes
            # Aggregate into single 5-minute bar
            aggregated_bar = {
                'timestamp': bars[-1]['timestamp'],  # Use close time
                'open': bars[0]['open'],  # First bar's open
                'high': max(b['high'] for b in bars),
                'low': min(b['low'] for b in bars),
                'close': bars[-1]['close'],  # Last bar's close
                'volume': sum(b['volume'] for b in bars)  # Sum all volume
            }

            # Store completed 5-min bar
            if symbol not in self._completed_5min_bars:
                self._completed_5min_bars[symbol] = pd.DataFrame()

            completed_df = self._completed_5min_bars[symbol]
            new_5min_row = pd.DataFrame([aggregated_bar])
            completed_df = pd.concat([completed_df, new_5min_row], ignore_index=True)

            # Keep reasonable lookback for 5-min bars (e.g., 100 bars = ~8 hours)
            if len(completed_df) > 100:
                completed_df = completed_df.iloc[-100:].copy()

            self._completed_5min_bars[symbol] = completed_df

            logger.debug(
                f"{symbol} 5-min bar completed: Close=${aggregated_bar['close']:.2f}, "
                f"Vol={aggregated_bar['volume']:,}"
            )

            # Clear accumulation for next 5-minute period
            self._current_5min_bars[symbol] = []

    def get_latest_5min_bar(self, symbol: str) -> Optional[Dict]:
        """
        Get the most recent completed 5-minute bar.

        FIX #2: Strategies should check against 5-min bar closes, not 5-sec bars.

        Returns:
            Dict with OHLCV data of latest 5-min bar, or None
        """
        if symbol not in self._completed_5min_bars:
            return None

        df = self._completed_5min_bars[symbol]
        if df.empty:
            return None

        return df.iloc[-1].to_dict()

    def get_completed_5min_bars(self, symbol: str, bars: int = None) -> Optional[pd.DataFrame]:
        """
        Get multiple completed 5-minute bars.

        Args:
            symbol: Stock symbol
            bars: Number of bars to return (None = all bars)

        Returns:
            DataFrame with OHLCV data of completed 5-min bars, or None
        """
        if symbol not in self._completed_5min_bars:
            return None

        df = self._completed_5min_bars[symbol]
        if df.empty:
            return None

        if bars is None:
            return df.copy()

        return df.tail(bars).copy()

    def has_new_5min_bar(self, symbol: str, last_checked_time: Optional[datetime] = None) -> bool:
        """
        Check if a new 5-minute bar has completed since last check.

        Args:
            symbol: Stock symbol
            last_checked_time: Timestamp of last check

        Returns:
            True if new 5-min bar available
        """
        latest_bar = self.get_latest_5min_bar(symbol)
        if not latest_bar:
            return False

        if last_checked_time is None:
            return True

        return latest_bar['timestamp'] > last_checked_time

    # === FIX #3: Proper Volume Tracking ===

    def _track_minute_volume(self, symbol: str, timestamp: datetime, volume: int):
        """
        Track volume at minute-level granularity.

        FIX #3: Prevents false triggers from single large trades.
        Aggregates 5-second volumes into minute volumes for comparison.

        Args:
            symbol: Stock symbol
            timestamp: Bar timestamp
            volume: Bar volume
        """
        if symbol not in self._minute_volume:
            self._minute_volume[symbol] = {}

        # Round down to minute
        minute_key = timestamp.replace(second=0, microsecond=0)

        if minute_key not in self._minute_volume[symbol]:
            self._minute_volume[symbol][minute_key] = 0

        # Accumulate volume for this minute
        self._minute_volume[symbol][minute_key] += volume

    def get_minute_volume_avg(self, symbol: str, lookback_minutes: int = 20) -> Optional[float]:
        """
        Get average volume per minute over lookback period.

        FIX #3: Proper volume calculation for comparison.

        Args:
            symbol: Stock symbol
            lookback_minutes: Number of minutes to average

        Returns:
            Average volume per minute, or None if insufficient data
        """
        if symbol not in self._minute_volume:
            return None

        volumes = list(self._minute_volume[symbol].values())
        if len(volumes) < lookback_minutes:
            return None

        # Get last N minutes
        recent_volumes = volumes[-lookback_minutes:]
        return sum(recent_volumes) / len(recent_volumes)

    def get_current_minute_volume(self, symbol: str) -> int:
        """
        Get accumulated volume for current minute.

        Returns:
            Volume in current minute (sum of 5-second bars)
        """
        if symbol not in self._minute_volume:
            return 0

        if not self._minute_volume[symbol]:
            return 0

        # Get most recent minute
        latest_minute = max(self._minute_volume[symbol].keys())
        return self._minute_volume[symbol][latest_minute]

    # === Additional Helper Methods ===

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect candlestick patterns.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with pattern columns added
        """
        if len(df) < 2:
            df['bullish_engulfing'] = False
            df['bearish_engulfing'] = False
            df['hammer'] = False
            df['shooting_star'] = False
            return df

        try:
            # Initialize pattern columns
            df['bullish_engulfing'] = False
            df['bearish_engulfing'] = False
            df['hammer'] = False
            df['shooting_star'] = False

            for i in range(1, len(df)):
                current = df.iloc[i]
                prev = df.iloc[i-1]

                # Bullish Engulfing
                if (prev['close'] < prev['open'] and  # Previous bearish
                    current['close'] > current['open'] and  # Current bullish
                    current['open'] <= prev['close'] and  # Opens at/below prev close
                    current['close'] >= prev['open']):  # Closes at/above prev open
                    df.loc[df.index[i], 'bullish_engulfing'] = True

                # Bearish Engulfing
                if (prev['close'] > prev['open'] and  # Previous bullish
                    current['close'] < current['open'] and  # Current bearish
                    current['open'] >= prev['close'] and  # Opens at/above prev close
                    current['close'] <= prev['open']):  # Closes at/below prev open
                    df.loc[df.index[i], 'bearish_engulfing'] = True

                # Hammer (bullish reversal)
                body_size = abs(current['close'] - current['open'])
                lower_wick = min(current['open'], current['close']) - current['low']
                upper_wick = current['high'] - max(current['open'], current['close'])

                if body_size > 0:  # Avoid division by zero
                    if lower_wick > 2 * body_size and upper_wick < body_size:
                        df.loc[df.index[i], 'hammer'] = True

                    # Shooting Star (bearish reversal)
                    if upper_wick > 2 * body_size and lower_wick < body_size:
                        df.loc[df.index[i], 'shooting_star'] = True

        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")

        return df

    def check_bullish_candle_pattern(self, symbol: str) -> bool:
        """Check if latest candle is a bullish pattern."""
        inds = self.get_latest_indicators(symbol)
        if not inds:
            return False

        return inds.get('bullish_engulfing', False) or inds.get('hammer', False)

    def check_bearish_candle_pattern(self, symbol: str) -> bool:
        """Check if latest candle is a bearish pattern."""
        inds = self.get_latest_indicators(symbol)
        if not inds:
            return False

        return inds.get('bearish_engulfing', False) or inds.get('shooting_star', False)

    def check_ema_crossover(self, symbol: str, fast_period: int = 9, slow_period: int = 21) -> Optional[str]:
        """
        Check for EMA crossover.

        Args:
            symbol: Stock symbol
            fast_period: Fast EMA period
            slow_period: Slow EMA period

        Returns:
            'bullish', 'bearish', or None
        """
        df = self.get_dataframe(symbol, bars=3)
        if df is None or len(df) < 2:
            return None

        try:
            fast_key = f'ema_{fast_period}'
            slow_key = f'ema_{slow_period}'

            if fast_key not in df.columns or slow_key not in df.columns:
                return None

            # Check last two bars
            prev_fast = df[fast_key].iloc[-2]
            prev_slow = df[slow_key].iloc[-2]
            curr_fast = df[fast_key].iloc[-1]
            curr_slow = df[slow_key].iloc[-1]

            # Bullish crossover: fast crosses above slow
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                return 'bullish'

            # Bearish crossover: fast crosses below slow
            if prev_fast >= prev_slow and curr_fast < curr_slow:
                return 'bearish'

        except Exception as e:
            logger.error(f"Error checking EMA crossover: {e}")

        return None

    def check_stochrsi_crossover(self, symbol: str) -> Optional[str]:
        """
        Check for Stochastic RSI %K/%D crossover in extreme zones.

        Returns:
            'bullish' if crossover in oversold (<20), 'bearish' if in overbought (>80), None otherwise
        """
        df = self.get_dataframe(symbol, bars=3)
        if df is None or len(df) < 2:
            return None

        try:
            if 'stochrsi_k' not in df.columns or 'stochrsi_d' not in df.columns:
                return None

            prev_k = df['stochrsi_k'].iloc[-2]
            prev_d = df['stochrsi_d'].iloc[-2]
            curr_k = df['stochrsi_k'].iloc[-1]
            curr_d = df['stochrsi_d'].iloc[-1]

            # Bullish crossover in oversold zone
            if prev_k <= prev_d and curr_k > curr_d and curr_k < 30:
                return 'bullish'

            # Bearish crossover in overbought zone
            if prev_k >= prev_d and curr_k < curr_d and curr_k > 70:
                return 'bearish'

        except Exception as e:
            logger.error(f"Error checking StochRSI crossover: {e}")

        return None

    def check_squeeze_fired(self, symbol: str) -> Optional[str]:
        """
        Check if TTM Squeeze has fired (BB exits KC).

        Returns:
            'bullish' or 'bearish' based on momentum direction, None if no fire
        """
        df = self.get_dataframe(symbol, bars=10)
        if df is None or len(df) < 6:
            return None

        try:
            if 'squeeze_on' not in df.columns or 'macd_hist' not in df.columns:
                return None

            # Check if squeeze was on for at least 6 bars
            recent_squeeze = df['squeeze_on'].iloc[-7:-1]  # Last 6 bars before current
            if not recent_squeeze.all():
                return None  # Squeeze wasn't sustained

            # Check if squeeze is now off
            current_squeeze = df['squeeze_on'].iloc[-1]
            if current_squeeze:
                return None  # Still in squeeze

            # Squeeze fired! Check momentum direction
            current_momentum = df['macd_hist'].iloc[-1]
            prev_momentum = df['macd_hist'].iloc[-2]

            if current_momentum > 0 and current_momentum > prev_momentum:
                return 'bullish'
            elif current_momentum < 0 and current_momentum < prev_momentum:
                return 'bearish'

        except Exception as e:
            logger.error(f"Error checking squeeze fire: {e}")

        return None

    def is_bandwidth_at_low(self, symbol: str, lookback: int = 30) -> bool:
        """
        Check if Bollinger BandWidth is at lowest point in lookback period.

        Args:
            symbol: Stock symbol
            lookback: Number of periods to check

        Returns:
            True if current bandwidth is minimum
        """
        df = self.get_dataframe(symbol, bars=lookback + 1)
        if df is None or 'bb_width' not in df.columns or len(df) < lookback:
            return False

        try:
            current_width = df['bb_width'].iloc[-1]
            min_width = df['bb_width'].iloc[-lookback:].min()

            return current_width == min_width

        except Exception as e:
            logger.error(f"Error checking bandwidth: {e}")
            return False

    def detect_overnight_gap(self, symbol: str, min_gap_pct: float = 2.0) -> Optional[Dict]:
        """
        Detect overnight gap from previous close.

        Args:
            symbol: Stock symbol
            min_gap_pct: Minimum gap percentage to register

        Returns:
            Dict with gap info or None
        """
        if symbol not in self._daily_data:
            return None

        daily_df = self._daily_data[symbol]
        if len(daily_df) < 2:
            return None

        try:
            # Get first bar of today (opening price)
            first_bar = daily_df.iloc[0]
            open_price = first_bar['open']

            # Get yesterday's close (would need to cache this between days)
            # For now, check if we have previous day data
            # This is a simplified version - production would track prev_close properly
            if hasattr(self, '_prev_day_close') and symbol in self._prev_day_close:
                prev_close = self._prev_day_close[symbol]

                gap_pct = ((open_price - prev_close) / prev_close) * 100

                if abs(gap_pct) >= min_gap_pct:
                    return {
                        'gap_pct': gap_pct,
                        'gap_type': 'up' if gap_pct > 0 else 'down',
                        'open_price': open_price,
                        'prev_close': prev_close
                    }

        except Exception as e:
            logger.error(f"Error detecting gap: {e}")

        return None

    def set_previous_day_close(self, symbol: str, close_price: float):
        """
        Store previous day's close for gap detection.

        Args:
            symbol: Stock symbol
            close_price: Closing price
        """
        if not hasattr(self, '_prev_day_close'):
            self._prev_day_close = {}

        self._prev_day_close[symbol] = close_price

    # === Live IV Methods ===

    async def update_iv(self, symbol: str, force: bool = False) -> Optional[float]:
        """
        Update live IV for a symbol from options chain.

        Uses the IV data provider to fetch and calculate real IV.

        Args:
            symbol: Stock symbol
            force: Force update even if recently updated

        Returns:
            Current IV or None if unavailable
        """
        # Check if update is needed
        now = datetime.now()
        last_update = self._iv_last_update.get(symbol)

        if not force and last_update:
            if (now - last_update).total_seconds() < self._iv_update_interval:
                return self._live_iv.get(symbol)

        try:
            # Import here to avoid circular imports
            from strategies.indicators.iv_data_provider import iv_data_provider

            # Fetch live IV
            iv = await iv_data_provider.get_live_iv(symbol)

            if iv is not None:
                self._live_iv[symbol] = iv
                self._iv_last_update[symbol] = now

                # Also get percentile
                percentile = await iv_data_provider.get_iv_percentile(symbol)
                if percentile is not None:
                    self._iv_percentile[symbol] = percentile
                    logger.debug(f"Updated IV for {symbol}: {iv:.2f}% (percentile: {percentile:.1f}%)")
                else:
                    logger.debug(f"Updated IV for {symbol}: {iv:.2f}% (percentile: N/A)")
                return iv

        except Exception as e:
            logger.error(f"Failed to update IV for {symbol}: {e}")

        return self._live_iv.get(symbol)

    async def update_iv_batch(self, symbols: List[str]) -> Dict[str, float]:
        """
        Update IV for multiple symbols.

        Args:
            symbols: List of symbols to update

        Returns:
            Dict mapping symbol to IV
        """
        results = {}

        for symbol in symbols:
            iv = await self.update_iv(symbol)
            if iv is not None:
                results[symbol] = iv

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.3)

        return results

    def get_iv(self, symbol: str) -> Optional[float]:
        """
        Get cached IV for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Cached IV or None
        """
        return self._live_iv.get(symbol)

    def get_iv_percentile_cached(self, symbol: str) -> Optional[float]:
        """
        Get cached IV percentile for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Cached IV percentile or None
        """
        return self._iv_percentile.get(symbol)

    def set_iv(self, symbol: str, iv: float, percentile: float = None):
        """
        Manually set IV for a symbol (useful for testing or external data sources).

        Args:
            symbol: Stock symbol
            iv: Implied volatility (as percentage)
            percentile: IV percentile (optional)
        """
        self._live_iv[symbol] = iv
        self._iv_last_update[symbol] = datetime.now()

        if percentile is not None:
            self._iv_percentile[symbol] = percentile

        logger.debug(f"Set IV for {symbol}: {iv:.2f}%")

    async def get_vix(self) -> Optional[float]:
        """
        Get current VIX level.

        Returns:
            Current VIX or None
        """
        try:
            from strategies.indicators.iv_data_provider import iv_data_provider
            return await iv_data_provider.get_vix_level()
        except Exception as e:
            logger.error(f"Failed to get VIX: {e}")
            return None

    def get_iv_stats(self, symbol: str) -> Optional[Dict]:
        """
        Get comprehensive IV statistics for a symbol.

        Returns:
            Dict with IV stats or None
        """
        try:
            from strategies.indicators.iv_calculator import iv_calculator
            return iv_calculator.get_iv_stats(symbol)
        except Exception:
            return None
