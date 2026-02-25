#!/usr/bin/env python3
"""
0DTE Iron Condor Automated Trading Bot.

Single-purpose bot that runs the full daily lifecycle:
  Pre-flight → Entry (v5/v4 dynamic or fixed-time) → Monitor → Exit

Entry modes:
  v5 dynamic (default): Scans 10:00-14:00 ET every 5 min using ML TP prediction
                         models. Enters when P(hit 25% TP) >= threshold.
                         Also shows P(hit 50% TP) for information.
  v4 dynamic (fallback): Same scan loop but predicts P(profitable at 3PM close).
                          Used when v5 models not loaded or --no-v5 flag set.
  Fixed-time (fallback): Enters at BOT_ENTRY_TIME (noon ET) with v3 risk filter.
                          Used when v4 model not loaded or --no-v4 flag set.

Exit via: take-profit (native Schwab limit), stop-loss (bot-monitored),
          SLTP trailing lock, or 3PM forced close (bot-monitored).

Monitor mode: auto-detects multiple ICs from positions, monitors each with
tiered TP/SL alerts (sound notifications + Schwab orders).

Usage:
  python scripts/zero_dte_bot.py --dry-run         # Paper: v5 scan or fixed-time
  python scripts/zero_dte_bot.py --dry-run --skip-wait --skip-ml  # Instant test
  python scripts/zero_dte_bot.py --dry-run --no-v5  # Force v4 or fixed-time entry
  python scripts/zero_dte_bot.py --dry-run --no-v4  # Force fixed-time entry
  python scripts/zero_dte_bot.py --dry-run --manual-approve  # Prompt when score < threshold
  python scripts/zero_dte_bot.py --live             # Real money (careful!)
  python scripts/zero_dte_bot.py --monitor          # Auto-detect credits from orders
  python scripts/zero_dte_bot.py --monitor --credits 2.60,3.00  # Manual credits
"""

import sys
import os
import math
import select
import asyncio
import argparse
import json
import subprocess
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytz
from loguru import logger

from config.settings import settings
from execution.order_manager.ic_order_builder import IronCondorOrderBuilder


# Timezone
ET = pytz.timezone('US/Eastern')

# macOS sounds
TP_SOUND = "/System/Library/Sounds/Glass.aiff"
SL_SOUND = "/System/Library/Sounds/Sosumi.aiff"


class AlertLevel(Enum):
    """Tiered alert levels for IC monitoring."""
    TP_25 = "TP_25"        # debit <= 75% of credit -> sound
    TP_50 = "TP_50"        # debit <= 50% of credit -> place limit order
    SL_WARNING = "SL_WARN" # debit halfway to SL exit -> sound warning
    SL_EXIT = "SL_EXIT"    # debit >= portfolio-based SL -> MARKET CLOSE


@dataclass
class TrackedIronCondor:
    """Tracks one iron condor's state during monitoring."""
    ic_id: int
    label: str
    strikes: Dict           # same format as select_strikes() output
    quantity: int
    fill_credit: float
    portfolio_value: float  # live portfolio value from Schwab

    # Mutable state
    alerts_fired: Set[AlertLevel] = field(default_factory=set)
    tp_order_id: Optional[str] = None
    exit_reason: Optional[str] = None
    is_closed: bool = False

    # Computed thresholds (set in __post_init__)
    tp_25_debit: float = 0.0
    tp_50_debit: float = 0.0
    sl_warning_debit: float = 0.0  # halfway to SL exit -> sound
    sl_exit_debit: float = 0.0     # portfolio-based SL -> MARKET CLOSE

    # Dynamic TP (set externally after daily P&L is computed)
    target_tp_debit: float = 0.0

    # Tiered SLTP (trailing profit lock): ratcheting tiers from 1% to 5%
    # Each tuple: (activation_pct, floor_pct, activation_debit, floor_debit)
    sltp_tiers: List[Tuple[float, float, float, float]] = field(default_factory=list)
    sltp_active_tier_idx: int = -1        # Index of highest activated tier (-1 = none)
    sltp_floor_debit: float = 0.0         # Current active floor debit (from highest tier)

    def __post_init__(self):
        # Round-trip transaction cost: 4 legs open + 4 legs close = 8 legs
        # Convert to per-share terms (same units as credit/debit prices)
        fee = (settings.BOT_COST_PER_LEG * 8) / 100  # $0.052
        fee_dollars = settings.BOT_COST_PER_LEG * 8   # $5.20

        # TP: max debit to keep X% of credit after fees
        # profit = credit - debit - fee, so debit = credit * (1 - X%) - fee
        self.tp_25_debit = round(self.fill_credit * 0.75 - fee, 2)
        self.tp_50_debit = round(self.fill_credit * 0.50 - fee, 2)

        # SL: portfolio-based. Max loss = BOT_STOP_LOSS_PCT × portfolio.
        # loss = qty × ((debit - credit) × 100 + fees)
        # Solve: debit = credit + (max_loss - qty × fees) / (qty × 100)
        if self.quantity > 0:
            max_loss = self.portfolio_value * settings.BOT_STOP_LOSS_PCT
            closing_fees = self.quantity * 4 * settings.BOT_COST_PER_LEG
            self.sl_exit_debit = round(
                self.fill_credit + (max_loss - closing_fees) / (self.quantity * 100), 2
            )
            # Warning at halfway between credit and SL exit
            self.sl_warning_debit = round(
                self.fill_credit + (self.sl_exit_debit - self.fill_credit) / 2, 2
            )

        # Tiered SLTP: build tiers from START to MAX in STEP increments
        # Each tier: activate at pct% gain, lock-in floor at (pct - GAP)%
        # debit = credit - (portfolio * pct / qty + fee_dollars) / 100
        if self.fill_credit > 0 and self.quantity > 0:
            self.sltp_tiers = []
            portfolio = self.portfolio_value
            pct = settings.BOT_SLTP_START_PCT
            while pct <= settings.BOT_SLTP_MAX_PCT + 1e-9:
                floor_pct = pct - settings.BOT_SLTP_GAP_PCT
                act_debit = round(
                    self.fill_credit - (portfolio * pct / self.quantity + fee_dollars) / 100, 2
                )
                flr_debit = round(
                    self.fill_credit - (portfolio * floor_pct / self.quantity + fee_dollars) / 100, 2
                )
                self.sltp_tiers.append((pct, floor_pct, act_debit, flr_debit))
                pct = round(pct + settings.BOT_SLTP_STEP_PCT, 4)
            self.sltp_active_tier_idx = -1
            self.sltp_floor_debit = 0.0


@dataclass
class DailyPnLState:
    """Tracks daily P&L from Schwab account balances for dynamic TP sizing."""
    sod_balance: float          # initialBalances.liquidationValue
    current_balance: float      # currentBalances.liquidationValue
    total_daily_pnl: float      # current - sod
    open_ic_unrealized: float   # sum of currentDayProfitLoss for open SPX positions
    realized_today: float       # total_daily_pnl - open_ic_unrealized
    remaining_to_target: float  # daily_target - realized_today
    daily_target: float         # BOT_PORTFOLIO_VALUE * BOT_DAILY_TARGET_PCT


def play_sound(path: str):
    """Play a macOS sound file (non-blocking)."""
    try:
        subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def send_notification(title: str, message: str):
    """Send a macOS notification via osascript."""
    try:
        script = f'display notification "{message}" with title "{title}"'
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def alert_tp(ic: TrackedIronCondor, level: AlertLevel, debit: float):
    """Fire a take-profit alert (sound + notification + log)."""
    pct = round((1 - debit / ic.fill_credit) * 100)
    msg = f"{ic.label}: TP {pct}% (debit ${debit:.2f} vs credit ${ic.fill_credit:.2f})"
    logger.info(f"  ALERT {level.value}: {msg}")
    play_sound(TP_SOUND)
    send_notification(f"TP Alert - {ic.label}", msg)


def alert_sl(ic: TrackedIronCondor, level: AlertLevel, debit: float):
    """Fire a stop-loss alert (sound + notification + log)."""
    mult = round(debit / ic.fill_credit, 1)
    msg = f"{ic.label}: SL {mult}x (debit ${debit:.2f} vs credit ${ic.fill_credit:.2f})"
    logger.warning(f"  ALERT {level.value}: {msg}")
    play_sound(SL_SOUND)
    send_notification(f"SL Alert - {ic.label}", msg)


class ZeroDTEBot:
    """Automated 0DTE iron condor trading bot."""

    def __init__(self, dry_run: bool = True, skip_wait: bool = False, skip_ml: bool = False,
                 manual_strikes: tuple = None, monitor_mode: bool = False,
                 monitor_credit: float = None, monitor_credits: list = None,
                 no_v4: bool = False, no_v5: bool = False, manual_approve: bool = False):
        self.dry_run = dry_run
        self.skip_wait = skip_wait
        self.skip_ml = skip_ml
        self.no_v4 = no_v4
        self.no_v5 = no_v5
        self.manual_approve = manual_approve
        self.manual_strikes = manual_strikes  # (short_put, short_call) or None
        self.monitor_mode = monitor_mode
        self.monitor_credit = monitor_credit
        self.monitor_credits = monitor_credits  # list of floats for multi-IC

        self.builder = IronCondorOrderBuilder()

        # State
        self.already_traded_today = False
        self.entry_order_id = None
        self.tp_order_id = None
        self.strikes = None
        self.quantity = 0
        self.fill_credit = 0.0
        self.exit_reason = None
        self.portfolio_value = 0.0  # populated from Schwab at startup
        self._entry_filled = False  # True when _entry_scan handles fill internally

        # v4 entry timing state
        self._v3_cached_confidence = None  # cache v3 day-level pre-screen
        self._v4_daily_data_cache = None   # cache daily data for v4 features

        # Lazy-loaded components
        self._schwab_client = None
        self._predictor = None
        self._candle_loader = None
        self._feature_pipeline = None

        mode = "MONITOR" if monitor_mode else ("DRY-RUN" if dry_run else "LIVE")
        logger.info(f"ZeroDTEBot initialized in {mode} mode")
        if skip_wait:
            logger.info("Skip-wait enabled: bypassing time-based sleeping")
        if skip_ml:
            logger.info("Skip-ML enabled: bypassing ML confidence check")
        if no_v5:
            logger.info("v5 disabled: falling back to v4 or fixed-time entry")
        if no_v4:
            logger.info("v4 disabled: using fixed-time entry")
        if manual_approve:
            logger.info("Manual-approve enabled: will prompt when score is below threshold")

    @property
    def schwab_client(self):
        if self._schwab_client is None:
            from execution.broker_api.schwab_client import schwab_client
            self._schwab_client = schwab_client
        return self._schwab_client

    @property
    def predictor(self):
        if self._predictor is None:
            from ml.models.predictor import MLPredictor
            self._predictor = MLPredictor()
        return self._predictor

    @property
    def candle_loader(self):
        if self._candle_loader is None:
            from ml.data.candle_loader import CandleLoader
            self._candle_loader = CandleLoader()
        return self._candle_loader

    @property
    def feature_pipeline(self):
        if self._feature_pipeline is None:
            from ml.features.pipeline import feature_pipeline
            self._feature_pipeline = feature_pipeline
        return self._feature_pipeline

    def _now_et(self) -> datetime:
        """Current time in Eastern."""
        return datetime.now(ET)

    def _parse_time(self, time_str: str) -> time:
        """Parse HH:MM string to time object."""
        h, m = map(int, time_str.split(':'))
        return time(h, m)

    def _extract_portfolio_value(self, account: dict) -> float:
        """
        Extract live portfolio value (liquidation value) from Schwab account data.

        Uses currentBalances.liquidationValue (most accurate real-time value).
        Falls back to initialBalances.liquidationValue, then BOT_PORTFOLIO_VALUE.
        """
        sec = account.get('securitiesAccount', {})

        # Prefer current (real-time) liquidation value
        current = sec.get('currentBalances', {}).get('liquidationValue', 0.0)
        if current > 0:
            return current

        # Fallback to start-of-day snapshot
        initial = sec.get('initialBalances', {}).get('liquidationValue', 0.0)
        if initial > 0:
            return initial

        # Last resort: config setting
        logger.warning(f"  Could not read liquidationValue from Schwab, "
                       f"falling back to BOT_PORTFOLIO_VALUE=${settings.BOT_PORTFOLIO_VALUE:,.0f}")
        return settings.BOT_PORTFOLIO_VALUE

    def _reset_entry_state(self):
        """Reset entry-related state after a failed fill so the bot can retry."""
        self.already_traded_today = False
        self.entry_order_id = None
        self.strikes = None
        self.quantity = 0
        self.fill_credit = 0.0

    def _prompt_sync(self, prompt: str, timeout: int = 60) -> str:
        """Blocking stdin prompt with timeout using select(). Returns '' on timeout."""
        sys.stdout.write(prompt)
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip()
        print()  # newline after timeout
        return ''

    async def _async_prompt(self, prompt: str, timeout: int = 60) -> str:
        """Non-blocking async wrapper around _prompt_sync."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._prompt_sync, prompt, timeout)

    async def _compute_entry_preview(self, chain: dict) -> Optional[dict]:
        """Compute order preview (strikes, sizing, TP/SL) without placing any orders."""
        try:
            # Select strikes
            if self.manual_strikes:
                sp_strike, sc_strike = self.manual_strikes
                strikes = self.builder.select_strikes_manual(
                    chain,
                    short_put_strike=sp_strike,
                    short_call_strike=sc_strike,
                    wing_width=settings.BOT_WING_WIDTH,
                )
            else:
                strikes = self.builder.select_strikes(
                    chain,
                    target_delta=settings.BOT_SHORT_DELTA,
                    wing_width=settings.BOT_WING_WIDTH,
                )
            if not strikes:
                return None

            credit = strikes['net_credit']
            if credit < settings.BOT_MIN_CREDIT:
                return None

            # Size the position
            buying_power = await self.schwab_client.get_buying_power() if not self.dry_run else 1_000_000
            quantity = self.builder.calculate_position_size(
                credit=credit,
                portfolio_value=self.portfolio_value,
                daily_target_pct=settings.BOT_DAILY_TARGET_PCT,
                tp_pct=settings.BOT_TAKE_PROFIT_PCT,
                cost_per_leg=settings.BOT_COST_PER_LEG,
                wing_width=settings.BOT_WING_WIDTH,
                buying_power=buying_power,
                max_contracts=settings.BOT_MAX_CONTRACTS,
            )
            if quantity <= 0:
                return None

            # Compute TP/SL debits: 5% ceiling TP (SLTP tiers are primary exit)
            fee_dollars = settings.BOT_COST_PER_LEG * 8
            max_gain = self.portfolio_value * settings.BOT_SLTP_MAX_PCT
            tp_debit = credit - (max_gain / quantity + fee_dollars) / 100
            tp_debit = round(math.ceil(tp_debit * 20) / 20, 2)
            tp_debit = max(tp_debit, 0.05)
            closing_fees = quantity * 4 * settings.BOT_COST_PER_LEG
            max_loss = self.portfolio_value * settings.BOT_STOP_LOSS_PCT
            sl_debit = round(credit + (max_loss - closing_fees) / (quantity * 100), 2)

            # Estimated P&L (5% ceiling TP)
            tp_profit = (credit - tp_debit) * quantity * 100 - closing_fees
            sl_loss = (sl_debit - credit) * quantity * 100 + closing_fees

            return {
                'strikes': strikes,
                'quantity': quantity,
                'credit': credit,
                'tp_debit': tp_debit,
                'sl_debit': sl_debit,
                'tp_profit': tp_profit,
                'sl_loss': sl_loss,
            }
        except Exception as e:
            logger.debug(f"  Preview computation failed: {e}")
            return None

    async def run(self):
        """Main bot lifecycle."""
        logger.info("=" * 60)
        logger.info("0DTE Iron Condor Bot Starting")
        logger.info(f"Date: {date.today()}")
        logger.info(f"Mode: {'DRY-RUN' if self.dry_run else 'LIVE'}")
        logger.info("=" * 60)

        try:
            # Monitor mode: attach to existing position
            if self.monitor_mode:
                await self.run_monitor_mode()
                return

            # Step 1: Pre-flight check
            if not await self.pre_flight_check():
                logger.error("Pre-flight check failed. Aborting.")
                return

            # Step 2+3: Entry — v5/v4 dynamic scanning or fixed-time
            # Priority: v5 (TP models) → v4 (profitability) → fixed-time
            use_v5 = (
                settings.BOT_V5_ENABLED
                and not self.no_v5
                and not self.skip_ml
                and self.predictor.v5_ready
            )
            use_v4 = (
                not use_v5
                and settings.BOT_V4_ENABLED
                and not self.no_v4
                and not self.skip_ml
                and self.predictor.v4_ready
            )

            if use_v5 or use_v4:
                version = "v5" if use_v5 else "v4"
                logger.info(f"{version} entry timing enabled — scanning for optimal entry")
                if not self.skip_wait:
                    await self._wait_until(settings.BOT_V4_ENTRY_START)
                entered = await self._entry_scan(use_v5=use_v5)
                if not entered:
                    logger.warning(f"{version} scan found no entry window. No trade today.")
                    return
            else:
                # Fixed-time entry (original flow)
                if not self.skip_ml:
                    logger.info("ML scan models not loaded — using fixed-time entry")
                if not self.skip_wait:
                    await self._wait_until(settings.BOT_ENTRY_TIME)
                if not await self.attempt_entry():
                    logger.warning("Entry attempt failed. No trade today.")
                    return

            # Step 4: Wait for entry fill (scan path handles this internally)
            if not self._entry_filled:
                filled = await self.wait_for_entry_fill()
                if not filled:
                    logger.warning("Entry did not fill within timeout. Cancelling.")
                    await self._cancel_order_safe(self.entry_order_id)
                    return

            # Step 5: Place take-profit order
            await self.place_take_profit_order()

            # Step 6: Monitoring loop
            await self.monitoring_loop()

        except KeyboardInterrupt:
            logger.warning("Bot interrupted by user (Ctrl+C)")
            if self.entry_order_id and not self.exit_reason:
                logger.warning("Attempting emergency close due to interrupt...")
                await self.emergency_close("keyboard_interrupt")
        except Exception as e:
            logger.error(f"Unhandled error in bot lifecycle: {e}", exc_info=True)
            if self.entry_order_id and not self.exit_reason:
                try:
                    await self.emergency_close("unhandled_error")
                except Exception as e2:
                    logger.critical(f"Emergency close also failed: {e2}")
        finally:
            self.log_trade_result()
            logger.info("Bot shutdown complete.")

    def _ensure_thetadata_running(self) -> bool:
        """
        Check if ThetaData Terminal is reachable. If not, start it and wait.

        Returns True if ThetaData is responsive, False after all retries exhausted.
        """
        HEALTH_URL = "http://127.0.0.1:25503/v3/option/list/expirations?symbol=SPXW"
        THETA_JAR = os.path.expanduser("~/Downloads/ThetaTerminalv3.jar")
        CREDS_FILE = "/tmp/theta_creds.txt"
        STARTUP_TIMEOUT = 90  # seconds to wait for ThetaData to come up

        # Quick health check
        def _is_alive():
            try:
                req = urllib.request.urlopen(HEALTH_URL, timeout=5)
                return req.status == 200
            except Exception:
                return False

        if _is_alive():
            logger.info("  ThetaData Terminal: running - OK")
            return True

        # Not running — try to start it
        logger.warning("  ThetaData Terminal not responding. Starting...")

        if not os.path.exists(THETA_JAR):
            logger.error(f"  ThetaData jar not found at {THETA_JAR}")
            return False

        # Ensure creds file exists
        if not os.path.exists(CREDS_FILE):
            logger.info(f"  Creating {CREDS_FILE} from .env credentials")
            try:
                with open(CREDS_FILE, 'w') as f:
                    f.write(f"{settings.THETADATA_USERNAME}\n{settings.THETADATA_PASSWORD}\n")
            except Exception as e:
                logger.error(f"  Failed to write creds file: {e}")
                return False

        try:
            subprocess.Popen(
                ["java", "-jar", THETA_JAR, "--creds-file", CREDS_FILE],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(THETA_JAR),
            )
            logger.info(f"  ThetaData Terminal process launched from {THETA_JAR}")
        except Exception as e:
            logger.error(f"  Failed to start ThetaData Terminal: {e}")
            return False

        # Wait for it to come up
        import time as _time
        for i in range(STARTUP_TIMEOUT // 5):
            _time.sleep(5)
            if _is_alive():
                logger.info(f"  ThetaData Terminal ready after {(i+1)*5}s")
                return True
            logger.info(f"  Waiting for ThetaData... ({(i+1)*5}s/{STARTUP_TIMEOUT}s)")

        logger.error(f"  ThetaData Terminal did not respond after {STARTUP_TIMEOUT}s")
        return False

    async def pre_flight_check(self) -> bool:
        """
        Verify all prerequisites before trading.

        Checks: Schwab connection, ML model, buying power, trading day,
        ThetaData Terminal, no existing SPX positions.
        """
        logger.info("--- Pre-flight Check ---")
        now = self._now_et()

        # Check trading day (weekday)
        if now.weekday() >= 5:
            logger.error(f"Not a trading day (weekday={now.weekday()})")
            return False
        logger.info(f"  Trading day: {now.strftime('%A')} - OK")

        # Check ThetaData Terminal (start if needed — must be up before ML)
        if not self.skip_ml:
            if not self._ensure_thetadata_running():
                logger.error("ThetaData Terminal required for ML features. Aborting.")
                return False
        else:
            logger.info("  ThetaData: skipped (--skip-ml)")

        # Check Schwab credentials configured
        if not settings.validate_schwab_credentials():
            logger.error("Schwab credentials not configured in .env")
            return False
        logger.info("  Schwab credentials: configured - OK")

        # Check Schwab connection + extract live portfolio value
        try:
            account = await self.schwab_client.get_account()
            bp = (
                account.get('securitiesAccount', {})
                .get('currentBalances', {})
                .get('buyingPower', 0.0)
            )
            self.portfolio_value = self._extract_portfolio_value(account)
            logger.info(f"  Schwab connection: OK (buying power: ${bp:,.2f})")
            logger.info(f"  Portfolio value: ${self.portfolio_value:,.2f} (live from Schwab)")

            if bp < 5000:
                logger.error(f"Insufficient buying power: ${bp:,.2f}")
                return False
        except Exception as e:
            logger.error(f"  Schwab connection failed: {e}")
            return False

        # Check ML model
        if not self.skip_ml:
            if not self.predictor.is_ready():
                logger.error("ML model not loaded")
                return False
            logger.info(f"  ML v3 model: loaded ({self.predictor.num_features} features) - OK")

            # v5 TP model status
            if self.predictor.v5_ready and not self.no_v5 and settings.BOT_V5_ENABLED:
                tp25_status = f"TP25={len(self.predictor.v5_tp25_feature_names)}f" if self.predictor.v5_tp25_ready else "TP25=N/A"
                tp50_status = f"TP50={len(self.predictor.v5_tp50_feature_names)}f" if self.predictor.v5_tp50_ready else "TP50=N/A"
                logger.info(f"  ML v5 model: loaded ({tp25_status}, {tp50_status}) - "
                           f"scanning {settings.BOT_V4_ENTRY_START}-{settings.BOT_V4_ENTRY_END} ET")
                # FOMC gate status
                from ml.models.predictor import get_economic_calendar_features
                cal = get_economic_calendar_features()
                fomc_str = "FOMC DAY" if cal["is_fomc_day"] else ("FOMC week" if cal["is_fomc_week"] else "no FOMC")
                logger.info(f"  FOMC gate: {'enabled' if settings.BOT_FOMC_GATE_ENABLED else 'disabled'} "
                           f"({fomc_str}, skip_day={'ON' if settings.BOT_FOMC_SKIP_DAY else 'OFF'})")
            elif self.no_v5:
                logger.info("  ML v5 model: disabled (--no-v5 flag)")
            elif not settings.BOT_V5_ENABLED:
                logger.info("  ML v5 model: disabled (BOT_V5_ENABLED=False)")
            else:
                logger.info("  ML v5 model: not loaded — falling back to v4/fixed-time")

            # v4 entry timing status
            if self.predictor.v4_ready and not self.no_v4 and settings.BOT_V4_ENABLED:
                logger.info(f"  ML v4 model: loaded ({len(self.predictor.v4_feature_names)} features) - "
                           f"scanning {settings.BOT_V4_ENTRY_START}-{settings.BOT_V4_ENTRY_END} ET")
            elif self.no_v4:
                logger.info("  ML v4 model: disabled (--no-v4 flag)")
            elif not settings.BOT_V4_ENABLED:
                logger.info("  ML v4 model: disabled (BOT_V4_ENABLED=False)")
            else:
                logger.info("  ML v4 model: not loaded — using fixed-time entry")
        else:
            logger.info("  ML model: skipped (--skip-ml)")

        # Check no existing SPX positions
        try:
            positions = await self.schwab_client.get_account_positions()
            spx_positions = [
                p for p in positions
                if 'SPX' in p.get('instrument', {}).get('symbol', '').upper()
            ]
            if spx_positions:
                logger.error(f"Existing SPX positions found: {len(spx_positions)}")
                for p in spx_positions:
                    sym = p.get('instrument', {}).get('symbol', '?')
                    qty = p.get('longQuantity', 0) - p.get('shortQuantity', 0)
                    logger.error(f"    {sym}: {qty}")
                return False
            logger.info("  No existing SPX positions - OK")
        except Exception as e:
            logger.error(f"  Position check failed: {e}")
            return False

        logger.info("--- Pre-flight Check PASSED ---")
        return True

    async def attempt_entry(self, chain: dict = None) -> bool:
        """
        Full entry flow: fetch chain -> ML prediction -> select strikes -> size -> place order.

        Args:
            chain: Optional pre-fetched option chain (from v4 scan). If None, fetches fresh.

        Returns True if entry order was placed, False otherwise.
        """
        logger.info("--- Attempting Entry ---")

        if self.already_traded_today:
            logger.warning("Already traded today. One-and-done.")
            return False

        # Check if daily target already met (skip in dry-run)
        if not self.dry_run:
            daily_pnl = await self._fetch_daily_pnl(open_ics=[])
            if daily_pnl.remaining_to_target <= 0:
                logger.info(f"Daily target already met: +${daily_pnl.realized_today:.2f}. No new entry.")
                return False
            logger.info(f"  Daily P&L: +${daily_pnl.realized_today:.2f}, "
                        f"need ${daily_pnl.remaining_to_target:.2f} more")

        # Fetch 0DTE chain (skip if pre-fetched by v4 scan)
        if chain is None:
            chain = await self._fetch_0dte_chain()
            if not chain:
                return False

        # ML prediction (skip if v4 already validated — v3 was pre-screened in _v4_entry_scan)
        if self._v3_cached_confidence is not None:
            logger.info(f"  ML v3 pre-screen: {self._v3_cached_confidence:.3f} (cached) - PASS")
        elif not self.skip_ml:
            confidence = await self._run_ml_prediction(chain)
            if confidence is None:
                logger.error("ML prediction failed")
                return False
            if confidence < settings.BOT_MIN_CONFIDENCE:
                logger.warning(
                    f"ML confidence {confidence:.3f} < "
                    f"threshold {settings.BOT_MIN_CONFIDENCE}. Skipping."
                )
                return False
            logger.info(f"  ML confidence: {confidence:.3f} - PASS")
        else:
            logger.info("  ML check: skipped (--skip-ml)")

        # Select strikes
        if self.manual_strikes:
            sp_strike, sc_strike = self.manual_strikes
            logger.info(f"  Using manual strikes: {sp_strike}p / {sc_strike}c")
            self.strikes = self.builder.select_strikes_manual(
                chain,
                short_put_strike=sp_strike,
                short_call_strike=sc_strike,
                wing_width=settings.BOT_WING_WIDTH,
            )
        else:
            self.strikes = self.builder.select_strikes(
                chain,
                target_delta=settings.BOT_SHORT_DELTA,
                wing_width=settings.BOT_WING_WIDTH,
            )
        if not self.strikes:
            logger.error("Strike selection failed")
            return False

        # Check minimum credit
        if self.strikes['net_credit'] < settings.BOT_MIN_CREDIT:
            logger.warning(
                f"Credit ${self.strikes['net_credit']:.2f} < "
                f"min ${settings.BOT_MIN_CREDIT:.2f}. Skipping."
            )
            return False

        # Get buying power for sizing
        buying_power = await self.schwab_client.get_buying_power()

        # Size the position
        self.quantity = self.builder.calculate_position_size(
            credit=self.strikes['net_credit'],
            portfolio_value=self.portfolio_value,
            daily_target_pct=settings.BOT_DAILY_TARGET_PCT,
            tp_pct=settings.BOT_TAKE_PROFIT_PCT,
            cost_per_leg=settings.BOT_COST_PER_LEG,
            wing_width=settings.BOT_WING_WIDTH,
            buying_power=buying_power,
            max_contracts=settings.BOT_MAX_CONTRACTS,
        )

        if self.quantity <= 0:
            logger.error("Position sizing returned 0 contracts")
            return False

        logger.info(f"  Position size: {self.quantity} contracts")

        # Build entry order
        entry_order = self.builder.build_entry_order(self.strikes, self.quantity)

        # Log the order
        self._log_order("ENTRY", entry_order)

        if self.dry_run:
            logger.info("[DRY-RUN] Entry order NOT placed. Simulating fill.")
            self.entry_order_id = "DRY_RUN_ENTRY"
            self.fill_credit = self.strikes['net_credit']
            self.already_traded_today = True
            return True

        # Place the order
        try:
            result = await self.schwab_client.place_order(entry_order)
            self.entry_order_id = result.get('orderId')
            if not self.entry_order_id:
                logger.error(f"No order ID returned: {result}")
                return False
            logger.info(f"  Entry order placed: ID={self.entry_order_id}")
            self.already_traded_today = True
            return True
        except Exception as e:
            logger.error(f"Failed to place entry order: {e}")
            return False

    async def _fetch_0dte_chain(self) -> dict:
        """Fetch nearest-expiration option chain for the bot symbol."""
        today = date.today()
        # Search up to 5 days ahead to find the nearest expiration
        # (handles weekends, holidays where today has no 0DTE)
        end_date = today + timedelta(days=5)
        try:
            chain = await self.schwab_client.get_option_chain(
                symbol=settings.BOT_SYMBOL,
                contract_type='ALL',
                from_date=today,
                to_date=end_date,
                include_underlying_quote=True,
            )
            underlying_price = chain.get('underlyingPrice', 0)
            logger.info(f"  Chain fetched: {settings.BOT_SYMBOL} @ ${underlying_price:,.2f}")
            return chain
        except Exception as e:
            logger.error(f"Failed to fetch option chain: {e}")
            return None

    async def _run_ml_prediction(self, chain: dict) -> float:
        """Run ML prediction on current market conditions with full analysis output."""
        try:
            now = self._now_et()
            underlying_price = chain.get('underlyingPrice', 0)

            print(f"\n{'='*60}")
            print(f"  ML MODEL ANALYSIS")
            print(f"{'='*60}")
            print(f"  Timestamp:   {now.strftime('%I:%M:%S %p ET')}")
            print(f"  Underlying:  $SPX @ ${underlying_price:,.2f}")
            print(f"  Model:       {self.predictor.model_version} ({self.predictor.model_type})")
            print()

            # Extract features
            print("  Extracting features...")
            features = await self.feature_pipeline.extract_features_realtime(
                underlying_symbol='SPX',
                timestamp=now,
                candle_loader=self.candle_loader,
                option_chain=chain,
                underlying_price=underlying_price,
            )

            total = len(features)
            non_zero = sum(1 for v in features.values() if v != 0 and v == v)
            print(f"  Features extracted: {total} ({non_zero} non-zero, {total - non_zero} defaulted)")
            print()

            # Run prediction with explanation
            result = self.predictor.predict_with_explanation(features)
            confidence = result['probability']
            threshold = settings.BOT_MIN_CONFIDENCE

            # Verdict box (semantics differ by model version)
            is_v3 = result.get('model_version') == 'v3'
            verdict = "TRADE" if confidence >= threshold else "NO TRADE"
            mark = ">>>" if confidence >= threshold else "XXX"

            print(f"  ┌──────────────────────────────────────────┐")
            if is_v3:
                risk_prob = result.get('risk_probability', 0)
                print(f"  |  RISK SCORE:    {risk_prob:>6.1%}  (P of big loss)   |")
                print(f"  |  CONFIDENCE:    {confidence:>6.1%}  (1 - risk)        |")
            else:
                print(f"  |  ML CONFIDENCE:  {confidence:>6.1%}                  |")
            print(f"  |  Threshold:      {threshold:>6.1%}                  |")
            print(f"  |  Decision:       {mark} {verdict:<22} |")
            print(f"  └──────────────────────────────────────────┘")

            # Show rule filter status for v3
            if is_v3:
                rules = result.get('rules_triggered', [])
                if rules:
                    print(f"\n  RULE FILTERS TRIGGERED:")
                    for rule in rules:
                        print(f"    - {rule}")
                else:
                    print(f"\n  Rule filters: all clear")

            print()

            # Top features driving the prediction
            print(f"  Top Features Driving Prediction:")
            print(f"  {'Feature':<35} {'Value':>10} {'Importance':>12}")
            print(f"  {'-'*35} {'-'*10} {'-'*12}")
            for name, value, importance in result['top_features']:
                imp_str = f"{importance:.4f}" if isinstance(importance, float) and importance < 1 else f"{importance:.0f}"
                print(f"  {name:<35} {value:>10.4f} {imp_str:>12}")
            print()

            # Show v3 computed features
            if is_v3 and 'v3_features' in result:
                print(f"  Risk Model Feature Values:")
                for name, val in sorted(result['v3_features'].items()):
                    print(f"    {name:<30} = {val:>12.4f}")
            else:
                # All pipeline feature values for v2
                print(f"  All Feature Values:")
                top_5_names = [f[0] for f in result['top_features'][:5]]
                for name in sorted(features.keys()):
                    val = features[name]
                    marker = " *" if name in top_5_names else ""
                    print(f"    {name:<40} = {val:>12.4f}{marker}")
            print(f"{'='*60}\n")

            return confidence
        except Exception as e:
            logger.error(f"ML prediction error: {e}", exc_info=True)
            return None

    # ===== v4 Entry Timing Methods =====

    def _fetch_spx_5min_candles(self) -> 'pd.DataFrame | None':
        """
        Fetch today's SPX 5-min candles from ThetaData for v4 feature computation.

        Uses the same /v3/index/history/ohlc endpoint as build_training_dataset.py.
        Returns DataFrame with open/high/low/close columns, or None on error.
        """
        import pandas as pd
        from datetime import date

        today = date.today()

        # Skip weekends (prevent forward-fill bug per memory notes)
        if today.weekday() >= 5:
            logger.warning("  Weekend: no intraday candles available")
            return None

        try:
            url = "http://127.0.0.1:25503/v3/index/history/ohlc"
            params = {
                "symbol": "SPX",
                "start_date": today.strftime("%Y%m%d"),
                "end_date": today.strftime("%Y%m%d"),
                "interval": "5m",
                "start_time": "09:30:00",
                "end_time": self._now_et().strftime("%H:%M:%S"),
                "format": "json",
            }

            # Build URL with query params
            query = "&".join(f"{k}={v}" for k, v in params.items())
            full_url = f"{url}?{query}"

            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            response = data.get("response", data if isinstance(data, list) else [])

            # Handle nested format
            if response and isinstance(response[0], dict) and "data" in response[0]:
                bars = []
                for entry in response:
                    for dp in entry.get("data", []):
                        bars.append(dp)
                response = bars

            if not response:
                logger.warning("  ThetaData returned no candle data")
                return None

            df = pd.DataFrame(response)

            # Normalize column names (ThetaData may return various formats)
            col_map = {}
            for col in df.columns:
                lower = col.lower()
                if lower in ('open', 'high', 'low', 'close'):
                    col_map[col] = lower
            if col_map:
                df = df.rename(columns=col_map)

            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close']
            if not all(c in df.columns for c in required):
                logger.warning(f"  Candle data missing columns. Have: {list(df.columns)}")
                return None

            # Convert to float
            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop NaN rows
            df = df.dropna(subset=required)

            if len(df) < 2:
                logger.warning(f"  Only {len(df)} candle(s) — need at least 2")
                return None

            logger.debug(f"  Fetched {len(df)} 5-min SPX candles")
            return df[required].reset_index(drop=True)

        except Exception as e:
            logger.warning(f"  Failed to fetch SPX candles from ThetaData: {e}")
            return None

    def _extract_atm_iv(self, chain: dict) -> float:
        """
        Extract ATM implied volatility from the option chain.

        Averages the mid-IV of the nearest put and call to the underlying price.
        Returns IV as a percentage (e.g. 15.0 for 15%).
        """
        underlying = chain.get('underlyingPrice', 0)
        if not underlying:
            return 15.0  # fallback

        # Find nearest expiration
        put_map = chain.get('putExpDateMap', {})
        call_map = chain.get('callExpDateMap', {})
        if not put_map or not call_map:
            return 15.0

        # Use first (nearest) expiration
        put_strikes = next(iter(put_map.values()), {})
        call_strikes = next(iter(call_map.values()), {})

        # Find ATM strike (closest to underlying)
        all_strikes = sorted(set(
            [float(k) for k in put_strikes.keys()] +
            [float(k) for k in call_strikes.keys()]
        ))
        if not all_strikes:
            return 15.0

        atm_strike = min(all_strikes, key=lambda s: abs(s - underlying))
        atm_str = str(atm_strike)

        # Get IV from put and call at ATM
        ivs = []
        for strike_map in [put_strikes, call_strikes]:
            contracts = strike_map.get(atm_str, [])
            if contracts:
                contract = contracts[0] if isinstance(contracts, list) else contracts
                iv = contract.get('volatility', 0)
                if iv and iv > 0:
                    ivs.append(iv)

        if ivs:
            return sum(ivs) / len(ivs)
        return 15.0

    async def _v4_entry_scan(self) -> bool:
        """Legacy wrapper — delegates to _entry_scan(use_v5=False)."""
        return await self._entry_scan(use_v5=False)

    async def _entry_scan(self, use_v5: bool = False) -> bool:
        """
        Dynamic entry scanning loop (supports v5 and v4).

        Scans every V4_SCAN_INTERVAL minutes from ENTRY_START to ENTRY_END:
        1. v3 pre-screen (run once, cache result)
        2. FOMC gate check (v5 only, if BOT_FOMC_SKIP_DAY enabled)
        3. Fetch 5-min SPX candles
        4. Compute v5/v4 features + predict
        5. If score >= threshold → attempt_entry()
        6. If no entry by ENTRY_END → skip day

        Returns True if entry was placed, False otherwise.
        """
        import pandas as pd
        from ml.models.predictor import get_economic_calendar_features

        scan_start = self._parse_time(settings.BOT_V4_ENTRY_START)
        scan_end = self._parse_time(settings.BOT_V4_ENTRY_END)
        interval = settings.BOT_V4_SCAN_INTERVAL
        version = "v5" if use_v5 else "v4"

        if use_v5:
            threshold = settings.BOT_V5_TP25_THRESHOLD
        else:
            threshold = settings.BOT_V4_MIN_CONFIDENCE

        logger.info(f"--- {version} Entry Scan ---")
        logger.info(f"  Window: {settings.BOT_V4_ENTRY_START} - {settings.BOT_V4_ENTRY_END} ET")
        logger.info(f"  Interval: {interval} min | Threshold: {threshold:.0%}")

        # FOMC gate check (v5 only)
        if use_v5 and settings.BOT_FOMC_GATE_ENABLED:
            cal = get_economic_calendar_features()
            if cal["is_fomc_day"] and settings.BOT_FOMC_SKIP_DAY:
                logger.info("  FOMC day detected — skipping per BOT_FOMC_SKIP_DAY=True")
                return False
            if cal["is_fomc_day"]:
                logger.info("  FOMC day detected — proceeding (BOT_FOMC_SKIP_DAY=False)")
            elif cal["is_fomc_week"]:
                logger.info("  FOMC week — no gate action")

        # Total number of scan slots for progress display
        start_minutes = scan_start.hour * 60 + scan_start.minute
        end_minutes = scan_end.hour * 60 + scan_end.minute
        total_slots = (end_minutes - start_minutes) // interval
        scan_count = 0

        while True:
            now = self._now_et()

            # Check if past scan window
            if now.time() >= scan_end and not self.skip_wait:
                logger.info(f"  {version} scan window closed at {settings.BOT_V4_ENTRY_END} ET. No entry today.")
                return False

            scan_count += 1
            slot_time = now.strftime('%H:%M')
            progress = f"[{scan_count}/{total_slots}]" if total_slots > 0 else f"[{scan_count}]"

            # Step 1: Fetch 0DTE chain
            chain = await self._fetch_0dte_chain()
            if not chain:
                logger.warning(f"  {progress} {slot_time} — chain unavailable, skipping slot")
                if self.skip_wait:
                    return False
                await asyncio.sleep(interval * 60)
                continue

            # Step 2: v3 pre-screen (run once per day, cache result)
            if self._v3_cached_confidence is None and not self.skip_ml:
                logger.info(f"  Running v3 day-level pre-screen...")
                v3_conf = await self._run_ml_prediction(chain)
                self._v3_cached_confidence = v3_conf

                if v3_conf is None:
                    logger.error(f"  v3 prediction failed — skipping day")
                    return False
                if v3_conf < settings.BOT_MIN_CONFIDENCE:
                    logger.warning(
                        f"  v3 pre-screen FAILED: {v3_conf:.3f} < {settings.BOT_MIN_CONFIDENCE} "
                        f"— skipping entire day"
                    )
                    return False
                logger.info(f"  v3 pre-screen PASSED: {v3_conf:.3f}")

            # Step 3: Fetch 5-min SPX candles
            candles = self._fetch_spx_5min_candles()
            if candles is None or len(candles) < 6:
                logger.info(f"  {progress} {slot_time} — insufficient candles ({0 if candles is None else len(candles)}), waiting...")
                if self.skip_wait:
                    # In skip-wait mode, try entry with fixed-time fallback
                    logger.info("  [SKIP-WAIT] Falling back to direct entry attempt")
                    return await self.attempt_entry(chain=chain)
                await asyncio.sleep(interval * 60)
                continue

            # Step 4: Extract ATM IV and compute prediction
            atm_iv = self._extract_atm_iv(chain)
            underlying = chain.get('underlyingPrice', 0)

            try:
                if use_v5:
                    # v5: predict both TP25 and TP50
                    tp25_score = self.predictor.predict_v5(
                        candles_5min=candles,
                        daily_data=self._v4_daily_data_cache,
                        option_atm_iv=atm_iv,
                        target="tp25",
                    ) if self.predictor.v5_tp25_ready else None

                    tp50_score = self.predictor.predict_v5(
                        candles_5min=candles,
                        daily_data=self._v4_daily_data_cache,
                        option_atm_iv=atm_iv,
                        target="tp50",
                    ) if self.predictor.v5_tp50_ready else None

                    primary_score = tp25_score if tp25_score is not None else (tp50_score or 0.0)
                else:
                    # v4: predict P(profitable)
                    primary_score = self.predictor.predict_v4(
                        candles_5min=candles,
                        daily_data=self._v4_daily_data_cache,
                        option_atm_iv=atm_iv,
                    )
                    tp25_score = None
                    tp50_score = None

                # Cache daily data from predictor after first call
                if self._v4_daily_data_cache is None and self.predictor._daily_cache:
                    self._v4_daily_data_cache = self.predictor._daily_cache
            except Exception as e:
                logger.warning(f"  {progress} {slot_time} — {version} prediction error: {e}")
                if self.skip_wait:
                    return False
                await asyncio.sleep(interval * 60)
                continue

            # Step 5: Display score(s) and check threshold
            bar_len = 20
            pass_mark = " >>> ENTRY" if primary_score >= threshold else ""

            if use_v5 and tp25_score is not None:
                # Dual-score display for v5
                tp25_filled = int(tp25_score * bar_len)
                tp25_bar = "█" * tp25_filled + "░" * (bar_len - tp25_filled)
                tp50_str = ""
                if tp50_score is not None:
                    tp50_filled = int(tp50_score * bar_len)
                    tp50_bar = "█" * tp50_filled + "░" * (bar_len - tp50_filled)
                    tp50_str = f" TP50={tp50_score:.3f} |{tp50_bar}|"

                logger.info(
                    f"  {progress} {slot_time}  "
                    f"v5: TP25={tp25_score:.3f} |{tp25_bar}|{tp50_str}  "
                    f"SPX=${underlying:,.0f}"
                    f"{pass_mark}"
                )
            else:
                # Single-score display for v4
                filled = int(primary_score * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                logger.info(
                    f"  {progress} {slot_time}  "
                    f"v4={primary_score:.3f} |{bar}| "
                    f"ATM-IV={atm_iv:.1f}  SPX=${underlying:,.0f}"
                    f"{pass_mark}"
                )

            if primary_score >= threshold:
                logger.info(f"  {version} TRIGGERED at {slot_time} (score {primary_score:.3f} >= {threshold})")
                # Pass the already-fetched chain to attempt_entry
                result = await self.attempt_entry(chain=chain)
                if not result:
                    # Hard failure (bad strikes, sizing, etc.) — skip this slot
                    logger.warning(f"  Entry attempt failed. Will retry at next scan slot.")
                    self._reset_entry_state()
                    await asyncio.sleep(interval * 60)
                    continue

                # Order placed — wait for fill
                filled = await self.wait_for_entry_fill()
                if filled:
                    self._entry_filled = True
                    return True  # Order placed AND filled

                # Fill timeout — cancel order, reset state, and keep scanning
                logger.warning("  Entry did not fill within timeout. Cancelling and re-scanning.")
                await self._cancel_order_safe(self.entry_order_id)
                self._reset_entry_state()
                # wait_for_entry_fill already consumed ~5 min, so we're at the next slot
                continue

            # Manual approval: below threshold but user might override
            if self.manual_approve and not self.skip_wait:
                preview = await self._compute_entry_preview(chain)
                if preview:
                    s = preview['strikes']
                    gap = threshold - primary_score
                    logger.info(f"  --- MANUAL APPROVAL ---")
                    logger.info(f"  Score: {primary_score:.3f}  (threshold {threshold:.3f}, gap {gap:.3f})")
                    if use_v5 and tp25_score is not None:
                        tp50_str = f"  TP50={tp50_score:.3f}" if tp50_score is not None else ""
                        logger.info(f"  TP25={tp25_score:.3f}{tp50_str}")
                    logger.info(f"  Put spread:  {s.get('long_put_strike',0):.0f}/{s.get('short_put_strike',0):.0f}p"
                                f"  (delta {s.get('short_put_delta',0):.3f})")
                    logger.info(f"  Call spread: {s.get('short_call_strike',0):.0f}/{s.get('long_call_strike',0):.0f}c"
                                f"  (delta {s.get('short_call_delta',0):.3f})")
                    logger.info(f"  Credit: ${preview['credit']:.2f}  x{preview['quantity']} contracts")
                    logger.info(f"  TP: buy back ${preview['tp_debit']:.2f}  "
                                f"(keep {settings.BOT_TAKE_PROFIT_PCT:.0%} -> +${preview['tp_profit']:,.0f})")
                    logger.info(f"  SL: close at ${preview['sl_debit']:.2f}  "
                                f"({settings.BOT_STOP_LOSS_PCT:.0%} portfolio -> -${preview['sl_loss']:,.0f})")

                    answer = await self._async_prompt("  Override entry? [y/N] (60s timeout): ", timeout=60)
                    if answer.lower() in ('y', 'yes'):
                        logger.info(f"  Manual override APPROVED at {slot_time}")
                        result = await self.attempt_entry(chain=chain)
                        if not result:
                            logger.warning(f"  Entry attempt failed after manual approval.")
                            self._reset_entry_state()
                            await asyncio.sleep(interval * 60)
                            continue

                        filled = await self.wait_for_entry_fill()
                        if filled:
                            self._entry_filled = True
                            return True

                        logger.warning("  Entry did not fill within timeout. Cancelling and re-scanning.")
                        await self._cancel_order_safe(self.entry_order_id)
                        self._reset_entry_state()
                        continue
                    else:
                        reason = "timeout" if answer == '' else "declined"
                        logger.info(f"  Manual override {reason}. Continuing scan.")

            # In skip-wait mode, only do one scan
            if self.skip_wait:
                logger.info(f"  [SKIP-WAIT] {version} score {primary_score:.3f} < {threshold}. "
                           f"Falling back to direct entry attempt.")
                return await self.attempt_entry(chain=chain)

            # Wait for next scan interval
            await asyncio.sleep(interval * 60)

    async def wait_for_entry_fill(self) -> bool:
        """
        Poll entry order status until filled or timeout.

        Returns True if filled, False otherwise.
        """
        if self.dry_run:
            logger.info("[DRY-RUN] Entry fill simulated immediately.")
            return True

        logger.info("Waiting for entry fill...")
        start = datetime.now()
        timeout = timedelta(seconds=settings.BOT_ENTRY_TIMEOUT)

        while datetime.now() - start < timeout:
            try:
                order_data = await self.schwab_client.get_order(self.entry_order_id)
                status = order_data.get('status', '')

                if status == 'FILLED':
                    # Extract actual fill credit from order data
                    self.fill_credit = float(order_data.get('price', self.strikes['net_credit']))
                    logger.info(f"  Entry FILLED at ${self.fill_credit:.2f}")
                    return True
                elif status in ('CANCELED', 'REJECTED', 'EXPIRED'):
                    reason = order_data.get('statusDescription', 'No reason provided')
                    logger.warning(f"  Entry order {status}: {reason}")
                    return False
                else:
                    logger.debug(f"  Entry status: {status}")

            except Exception as e:
                logger.warning(f"Error polling entry order: {e}")

            await asyncio.sleep(settings.BOT_ENTRY_POLL_INTERVAL)

        logger.warning("Entry fill timeout reached")
        return False

    async def place_take_profit_order(self):
        """Place take-profit BUY_TO_CLOSE limit order at 5% portfolio gain ceiling."""
        # Compute 5% ceiling TP: same formula as SLTP max tier
        fee_dollars = settings.BOT_COST_PER_LEG * 8
        max_gain = self.portfolio_value * settings.BOT_SLTP_MAX_PCT
        tp_debit = self.fill_credit - (max_gain / self.quantity + fee_dollars) / 100
        tp_debit = round(math.ceil(tp_debit * 20) / 20, 2)
        tp_debit = max(tp_debit, 0.05)

        logger.info(
            f"  TP target: buy back at ${tp_debit:.2f} debit "
            f"(5% portfolio = ${max_gain:.0f}, "
            f"keeping ${self.fill_credit - tp_debit:.2f} = "
            f"{((self.fill_credit - tp_debit) / self.fill_credit * 100):.0f}% of credit)"
        )

        tp_order = self.builder.build_close_order(self.strikes, self.quantity, tp_debit)
        self._log_order("TAKE-PROFIT", tp_order)

        if self.dry_run:
            logger.info("[DRY-RUN] TP order NOT placed.")
            self.tp_order_id = "DRY_RUN_TP"
            return

        try:
            result = await self.schwab_client.place_order(tp_order)
            self.tp_order_id = result.get('orderId')
            logger.info(f"  TP order placed: ID={self.tp_order_id}")
        except Exception as e:
            logger.error(f"Failed to place TP order: {e}")
            # Continue monitoring even without TP - bot will handle exit

    async def monitoring_loop(self):
        """
        Main monitoring loop. Every BOT_MONITOR_INTERVAL seconds:
        1. Check if TP order filled
        2. Fetch chain and check stop-loss
        3. Check if past exit time
        """
        # Portfolio-based SL: max loss = BOT_STOP_LOSS_PCT × portfolio
        max_loss = self.portfolio_value * settings.BOT_STOP_LOSS_PCT
        closing_fees = self.quantity * 4 * settings.BOT_COST_PER_LEG
        sl_threshold = round(
            self.fill_credit + (max_loss - closing_fees) / (self.quantity * 100), 2
        )

        exit_time = self._parse_time(settings.BOT_EXIT_TIME)

        # Build a TrackedIronCondor for SLTP tier tracking
        sltp_ic = TrackedIronCondor(
            ic_id=0, label="IC-1", strikes=self.strikes,
            quantity=self.quantity, fill_credit=self.fill_credit,
            portfolio_value=self.portfolio_value,
        )

        logger.info("--- Monitoring Loop Started ---")
        logger.info(f"  Credit: ${self.fill_credit:.2f}")
        logger.info(f"  Exit strategy: SLTP tiers (primary) | 5% ceiling TP on Schwab | 3PM close")
        logger.info(f"  SL threshold: ${sl_threshold:.2f} (debit to close = {settings.BOT_STOP_LOSS_PCT:.0%} portfolio loss)")
        logger.info(f"  Max loss: ${max_loss:,.0f} ({self.quantity} contracts)")
        logger.info(f"  Exit time: {settings.BOT_EXIT_TIME} ET")
        logger.info(f"  Check interval: {settings.BOT_MONITOR_INTERVAL}s")
        if sltp_ic.sltp_tiers:
            logger.info(f"  SLTP Tiers ({len(sltp_ic.sltp_tiers)} tiers):")
            for act_pct, flr_pct, act_debit, flr_debit in sltp_ic.sltp_tiers:
                logger.info(f"    {act_pct:.1%} -> {flr_pct:.1%}: activate @ ${act_debit:.2f}, floor @ ${flr_debit:.2f}")

        while True:
            try:
                now = self._now_et()

                # Check 1: TP filled?
                if await self._check_tp_filled():
                    self.exit_reason = "take_profit"
                    logger.info("=== EXIT: Take-profit filled ===")
                    break

                # Check 2: Time exit?
                if now.time() >= exit_time and not self.skip_wait:
                    self.exit_reason = "time_exit"
                    logger.info(f"=== EXIT: Past {settings.BOT_EXIT_TIME} ET ===")
                    await self.emergency_close("time_exit")
                    break

                # Check 3: Stop-loss?
                current_debit = await self._get_current_position_value()
                if current_debit is not None:
                    logger.info(
                        f"  [{now.strftime('%H:%M:%S')}] "
                        f"Debit to close: ${current_debit:.2f} "
                        f"(SL @ ${sl_threshold:.2f})"
                    )

                    if current_debit >= sl_threshold:
                        self.exit_reason = "stop_loss"
                        logger.warning(
                            f"=== EXIT: Stop-loss triggered "
                            f"(${current_debit:.2f} >= ${sl_threshold:.2f}) ==="
                        )
                        await self.emergency_close("stop_loss")
                        break

                    # Check 4: Tiered SLTP
                    if sltp_ic.sltp_tiers:
                        # Check if we've reached a higher tier
                        for idx in range(len(sltp_ic.sltp_tiers) - 1, sltp_ic.sltp_active_tier_idx, -1):
                            act_pct, flr_pct, act_debit, flr_debit = sltp_ic.sltp_tiers[idx]
                            if current_debit <= act_debit:
                                sltp_ic.sltp_active_tier_idx = idx
                                sltp_ic.sltp_floor_debit = flr_debit
                                logger.info(
                                    f"  >>> SLTP TIER {act_pct:.1%} ACTIVATED "
                                    f"(debit ${current_debit:.2f} <= ${act_debit:.2f})"
                                )
                                logger.info(
                                    f"      Floor raised to ${flr_debit:.2f} (locks in {flr_pct:.1%} gain)"
                                )
                                play_sound(TP_SOUND)
                                send_notification(
                                    f"Profit Lock {act_pct:.1%}",
                                    f"Floor ${flr_debit:.2f} ({flr_pct:.1%})"
                                )
                                break

                        # Check if floor breached
                        if sltp_ic.sltp_active_tier_idx >= 0 and current_debit >= sltp_ic.sltp_floor_debit:
                            flr_pct = sltp_ic.sltp_tiers[sltp_ic.sltp_active_tier_idx][1]
                            logger.warning(
                                f"=== EXIT: SLTP TRIGGERED at {flr_pct:.1%} floor "
                                f"(debit ${current_debit:.2f} >= floor ${sltp_ic.sltp_floor_debit:.2f}) ==="
                            )
                            play_sound(SL_SOUND)
                            await self.emergency_close(f"sltp_lock_{flr_pct:.1%}")
                            self.exit_reason = f"sltp_lock_{flr_pct:.1%}"
                            break
                else:
                    logger.debug(f"  [{now.strftime('%H:%M:%S')}] Could not get position value")

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                # Continue monitoring - don't crash on transient errors

            if self.skip_wait:
                # In skip-wait mode, simulate one check and exit
                logger.info("[SKIP-WAIT] Exiting monitoring loop after one check.")
                self.exit_reason = "skip_wait_test"
                break

            await asyncio.sleep(settings.BOT_MONITOR_INTERVAL)

    async def _check_tp_filled(self) -> bool:
        """Check if the take-profit order has filled."""
        if self.dry_run:
            return False  # In dry-run, TP never auto-fills

        if not self.tp_order_id:
            return False

        try:
            order_data = await self.schwab_client.get_order(self.tp_order_id)
            status = order_data.get('status', '')
            if status == 'FILLED':
                return True
        except Exception as e:
            logger.warning(f"Error checking TP order: {e}")

        return False

    async def _get_current_position_value(self) -> float:
        """Fetch fresh chain and calculate current debit to close."""
        if self.dry_run:
            # Simulate a value between 0 and credit (healthy position)
            return round(self.fill_credit * 0.6, 2)

        try:
            chain = await self._fetch_0dte_chain()
            if not chain:
                return None
            return self.builder.get_current_position_value(chain, self.strikes)
        except Exception as e:
            logger.warning(f"Error getting position value: {e}")
            return None

    async def emergency_close(self, reason: str):
        """Cancel TP order and market-close the position."""
        logger.warning(f"Emergency close triggered: {reason}")

        # Cancel TP order
        if self.tp_order_id and not self.dry_run:
            try:
                await self.schwab_client.cancel_order(self.tp_order_id)
                logger.info(f"  TP order {self.tp_order_id} cancelled")
            except Exception as e:
                logger.warning(f"  Failed to cancel TP order: {e}")

        # Place market close order
        close_order = self.builder.build_market_close_order(self.strikes, self.quantity)
        self._log_order("EMERGENCY_CLOSE", close_order)

        if self.dry_run:
            logger.info("[DRY-RUN] Emergency close NOT placed.")
            return

        try:
            result = await self.schwab_client.place_order(close_order)
            close_id = result.get('orderId', '?')
            logger.info(f"  Market close order placed: ID={close_id}")
        except Exception as e:
            logger.critical(f"FAILED TO PLACE EMERGENCY CLOSE ORDER: {e}")
            logger.critical("MANUAL INTERVENTION REQUIRED!")

    def log_trade_result(self):
        """Write trade result to JSON log file."""
        if not self.strikes:
            logger.info("No trade to log (no entry).")
            return

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        result = {
            "date": str(date.today()),
            "timestamp": self._now_et().isoformat(),
            "mode": "dry_run" if self.dry_run else "live",
            "symbol": settings.BOT_SYMBOL,
            "strikes": {
                "long_put": self.strikes['long_put']['strike'],
                "short_put": self.strikes['short_put']['strike'],
                "short_call": self.strikes['short_call']['strike'],
                "long_call": self.strikes['long_call']['strike'],
            },
            "expiration": self.strikes['expiration'],
            "quantity": self.quantity,
            "credit": self.fill_credit,
            "exit_reason": self.exit_reason,
            "settings": {
                "target_delta": settings.BOT_SHORT_DELTA,
                "wing_width": settings.BOT_WING_WIDTH,
                "tp_pct": settings.BOT_TAKE_PROFIT_PCT,
                "sl_pct": settings.BOT_STOP_LOSS_PCT,
                "portfolio_value": self.portfolio_value,
                "daily_target_pct": settings.BOT_DAILY_TARGET_PCT,
            }
        }

        log_path = log_dir / f"zero_dte_bot_{date.today()}.json"
        with open(log_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Trade result logged: {log_path}")

    # ===== Monitor Mode (Multi-IC) =====

    async def run_monitor_mode(self):
        """
        Persistent multi-IC monitor daemon.

        Outer loop runs forever:
        1. Check Schwab connection
        2. If SPX positions exist → reconstruct ICs, assign credits, monitor
        3. When all ICs close → loop back to waiting
        4. If no positions → sleep and re-check
        5. Only exits on Ctrl+C
        """
        IDLE_POLL = 60  # seconds between checks when no positions

        logger.info("=" * 60)
        logger.info("Monitor Mode - Persistent Daemon")
        logger.info("=" * 60)

        # Verify Schwab connection + extract live portfolio value
        try:
            account = await self.schwab_client.get_account()
            bp = (
                account.get('securitiesAccount', {})
                .get('currentBalances', {})
                .get('buyingPower', 0.0)
            )
            self.portfolio_value = self._extract_portfolio_value(account)
            logger.info(f"  Schwab connection: OK (buying power: ${bp:,.2f})")
            logger.info(f"  Portfolio value: ${self.portfolio_value:,.2f} (live from Schwab)")
        except Exception as e:
            logger.error(f"  Schwab connection failed: {e}")
            logger.error("  Cannot start monitor daemon without Schwab. Aborting.")
            return

        logger.info(f"  Polling for positions every {IDLE_POLL}s when idle")
        logger.info("  Press Ctrl+C to stop")
        logger.info("")

        consecutive_failures = 0
        MAX_FAILURE_BACKOFF = 300  # cap backoff at 5 minutes

        while True:
            try:
                now = self._now_et()

                # Check for SPX positions
                positions = await self._get_spx_positions()
                if not positions:
                    consecutive_failures = 0
                    # Show what a 10-delta IC would look like right now
                    try:
                        chain = await self._fetch_0dte_chain()
                        if chain:
                            strikes = self.builder.select_strikes(
                                chain,
                                target_delta=settings.BOT_SHORT_DELTA,
                                wing_width=settings.BOT_WING_WIDTH,
                            )
                            if strikes:
                                underlying = chain.get('underlyingPrice', 0)
                                logger.info(
                                    f"  [{now.strftime('%H:%M:%S')}] No positions. "
                                    f"Current IC @ {settings.BOT_SHORT_DELTA:.0%}Δ: "
                                    f"{strikes['long_put']['strike']:.0f}/{strikes['short_put']['strike']:.0f}p - "
                                    f"{strikes['short_call']['strike']:.0f}/{strikes['long_call']['strike']:.0f}c  "
                                    f"Credit: ${strikes['net_credit']:.2f}  "
                                    f"SPX: ${underlying:,.2f}"
                                )
                            else:
                                logger.info(f"  [{now.strftime('%H:%M:%S')}] No positions. Could not find IC strikes.")
                        else:
                            logger.info(f"  [{now.strftime('%H:%M:%S')}] No positions. Chain unavailable.")
                    except Exception as e:
                        logger.debug(f"  [{now.strftime('%H:%M:%S')}] No positions. Quote fetch failed: {e}")
                    await asyncio.sleep(IDLE_POLL)
                    continue

                # Positions found — enter active monitoring cycle
                logger.info(f"  [{now.strftime('%H:%M:%S')}] SPX positions detected! Starting monitor cycle.")
                success = await self._run_monitor_cycle()
                logger.info(f"  [{self._now_et().strftime('%H:%M:%S')}] Monitor cycle ended. Returning to idle.")
                logger.info("")
                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    backoff = min(IDLE_POLL * consecutive_failures, MAX_FAILURE_BACKOFF)
                    logger.info(f"  Setup failed ({consecutive_failures}x). Retrying in {backoff}s.")
                    logger.info(f"  Hint: use --monitor --credits <credit1,credit2,...> for manually-placed positions.")
                    await asyncio.sleep(backoff)

            except KeyboardInterrupt:
                raise  # let it propagate to exit
            except Exception as e:
                consecutive_failures += 1
                backoff = min(IDLE_POLL * consecutive_failures, MAX_FAILURE_BACKOFF)
                logger.error(f"  Monitor daemon error: {e}", exc_info=True)
                logger.info(f"  Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

    async def _fetch_daily_pnl(self, open_ics: List[TrackedIronCondor] = None) -> DailyPnLState:
        """
        Fetch daily P&L from Schwab account balances.

        Uses initialBalances.liquidationValue (SOD snapshot) vs currentBalances.liquidationValue
        to compute total daily P&L, then subtracts open IC unrealized to get realized P&L.
        """
        portfolio = self.portfolio_value or settings.BOT_PORTFOLIO_VALUE
        daily_target = portfolio * settings.BOT_DAILY_TARGET_PCT

        if self.dry_run:
            return DailyPnLState(
                sod_balance=portfolio,
                current_balance=portfolio,
                total_daily_pnl=0.0,
                open_ic_unrealized=0.0,
                realized_today=0.0,
                remaining_to_target=daily_target,
                daily_target=daily_target,
            )

        try:
            account = await self.schwab_client.get_account()
            sec_account = account.get('securitiesAccount', {})

            # SOD balance (start of day snapshot)
            initial_balances = sec_account.get('initialBalances', {})
            sod_balance = initial_balances.get('liquidationValue', 0.0)
            if not sod_balance:
                sod_balance = portfolio
                logger.warning(f"  initialBalances.liquidationValue unavailable, using portfolio=${sod_balance:,.0f}")

            # Current balance (live)
            current_balances = sec_account.get('currentBalances', {})
            current_balance = current_balances.get('liquidationValue', sod_balance)

            total_daily_pnl = current_balance - sod_balance

            # Sum unrealized P&L from open SPX option positions
            open_ic_unrealized = 0.0
            positions = sec_account.get('positions', [])
            for pos in positions:
                symbol = pos.get('instrument', {}).get('symbol', '')
                if 'SPX' in symbol.upper():
                    open_ic_unrealized += float(pos.get('currentDayProfitLoss', 0.0))

            realized_today = total_daily_pnl - open_ic_unrealized
            remaining = daily_target - realized_today

            return DailyPnLState(
                sod_balance=sod_balance,
                current_balance=current_balance,
                total_daily_pnl=total_daily_pnl,
                open_ic_unrealized=open_ic_unrealized,
                realized_today=realized_today,
                remaining_to_target=remaining,
                daily_target=daily_target,
            )
        except Exception as e:
            logger.warning(f"  Failed to fetch daily P&L: {e}. Using full target.")
            return DailyPnLState(
                sod_balance=portfolio,
                current_balance=portfolio,
                total_daily_pnl=0.0,
                open_ic_unrealized=0.0,
                realized_today=0.0,
                remaining_to_target=daily_target,
                daily_target=daily_target,
            )

    async def _get_spx_positions(self) -> list:
        """Fetch SPX option positions. Returns list (empty if none)."""
        try:
            positions = await self.schwab_client.get_account_positions()
            return [
                p for p in positions
                if 'SPX' in p.get('instrument', {}).get('symbol', '').upper()
            ]
        except Exception as e:
            logger.warning(f"  Failed to fetch positions: {e}")
            return []

    async def _run_monitor_cycle(self) -> bool:
        """
        One full monitor cycle: reconstruct ICs, assign credits, monitor until done.
        Returns True on successful cycle, False on setup failure.
        """
        # Log positions
        positions = await self._get_spx_positions()
        if not positions:
            return False

        if len(positions) % 4 != 0:
            logger.warning(
                f"  {len(positions)} SPX legs (not a multiple of 4). "
                f"Waiting for complete IC..."
            )
            return False

        for p in positions:
            sym = p.get('instrument', {}).get('symbol', '?')
            lq = p.get('longQuantity', 0)
            sq = p.get('shortQuantity', 0)
            logger.info(f"    {sym}: long={lq} short={sq}")

        # Reconstruct all ICs from positions
        tracked_ics = await self.reconstruct_multiple_ics_from_positions()
        if not tracked_ics:
            logger.warning("  Could not reconstruct ICs from positions. Will retry.")
            return False

        # Assign credits: manual --credits/--credit, or auto-detect from orders
        if not await self._assign_credits(tracked_ics):
            logger.error(
                "  Credit detection failed. If positions were placed manually "
                "(not via this bot), use: --monitor --credits <credit1,credit2,...>"
            )
            return False

        # Log all detected ICs with thresholds
        for ic in tracked_ics:
            logger.info(f"  {ic.label}: "
                        f"{ic.strikes['long_put']['strike']}/{ic.strikes['short_put']['strike']}p - "
                        f"{ic.strikes['short_call']['strike']}/{ic.strikes['long_call']['strike']}c "
                        f"x{ic.quantity}  Credit: ${ic.fill_credit:.2f}")
            logger.info(f"    TP 25%: debit <= ${ic.tp_25_debit:.2f}  -> sound")
            logger.info(f"    TP 50%: debit <= ${ic.tp_50_debit:.2f}  -> limit order")
            logger.info(f"    SL warn:  debit >= ${ic.sl_warning_debit:.2f} -> sound")
            logger.info(f"    SL exit:  debit >= ${ic.sl_exit_debit:.2f} -> MARKET CLOSE "
                        f"({settings.BOT_STOP_LOSS_PCT:.0%} portfolio)")
            if ic.sltp_tiers:
                logger.info(f"    SLTP Tiers ({len(ic.sltp_tiers)} tiers):")
                for act_pct, flr_pct, act_debit, flr_debit in ic.sltp_tiers:
                    logger.info(f"      {act_pct:.1%} -> {flr_pct:.1%}: activate @ ${act_debit:.2f}, floor @ ${flr_debit:.2f}")

        # Detect existing TP orders per IC, place 5% ceiling TP if missing
        for ic in tracked_ics:
            existing_tp = await self._detect_tp_order_for_ic(ic)
            if existing_tp:
                ic.tp_order_id = existing_tp
                logger.info(f"  {ic.label}: existing TP order found: ID={existing_tp}")
            else:
                # Compute 5% ceiling TP: same formula as SLTP max tier
                fee_dollars = settings.BOT_COST_PER_LEG * 8
                max_gain = ic.portfolio_value * settings.BOT_SLTP_MAX_PCT
                tp_debit = ic.fill_credit - (max_gain / ic.quantity + fee_dollars) / 100
                tp_debit = round(math.ceil(tp_debit * 20) / 20, 2)
                tp_debit = max(tp_debit, 0.05)
                ic.target_tp_debit = tp_debit
                logger.info(f"  {ic.label}: 5% ceiling TP ${tp_debit:.2f} "
                            f"(SLTP tiers are primary exit)")
                await self._place_tp_limit_order(ic, tp_debit)

        # Enter multi-IC monitoring loop (returns when all ICs close)
        await self.multi_ic_monitoring_loop(tracked_ics)

        # Log results
        logger.info("--- Monitor Cycle Results ---")
        for ic in tracked_ics:
            status = ic.exit_reason or ("open" if not ic.is_closed else "closed")
            logger.info(f"  {ic.label}: {status}")

        return True

    async def pre_flight_check_monitor(self) -> bool:
        """
        Pre-flight for monitor mode. Like pre_flight_check but REQUIRES existing SPX positions.
        """
        logger.info("--- Monitor Pre-flight Check ---")
        now = self._now_et()

        # Check trading day
        if now.weekday() >= 5:
            logger.error(f"Not a trading day (weekday={now.weekday()})")
            return False
        logger.info(f"  Trading day: {now.strftime('%A')} - OK")

        # Check Schwab connection
        try:
            account = await self.schwab_client.get_account()
            bp = (
                account.get('securitiesAccount', {})
                .get('currentBalances', {})
                .get('buyingPower', 0.0)
            )
            logger.info(f"  Schwab connection: OK (buying power: ${bp:,.2f})")
        except Exception as e:
            logger.error(f"  Schwab connection failed: {e}")
            return False

        # REQUIRE existing SPX positions (opposite of normal pre-flight)
        try:
            positions = await self.schwab_client.get_account_positions()
            spx_positions = [
                p for p in positions
                if 'SPX' in p.get('instrument', {}).get('symbol', '').upper()
            ]
            if not spx_positions:
                logger.error("No existing SPX option positions found. Nothing to monitor.")
                for p in positions:
                    sym = p.get('instrument', {}).get('symbol', '?')
                    asset_type = p.get('instrument', {}).get('assetType', '?')
                    logger.info(f"  Non-SPX position found: {sym} (type={asset_type})")
                return False
            logger.info(f"  Found {len(spx_positions)} SPX positions - OK")
            for p in spx_positions:
                sym = p.get('instrument', {}).get('symbol', '?')
                long_qty = p.get('longQuantity', 0)
                short_qty = p.get('shortQuantity', 0)
                logger.info(f"    {sym}: long={long_qty} short={short_qty}")
        except Exception as e:
            logger.error(f"  Position check failed: {e}")
            return False

        logger.info("--- Monitor Pre-flight Check PASSED ---")
        return True

    async def reconstruct_multiple_ics_from_positions(self) -> Optional[List[TrackedIronCondor]]:
        """
        Fetch all SPX positions from Schwab, group into multiple iron condors.

        Algorithm:
        1. Parse all OCC symbols, classify into long_puts, short_puts, short_calls, long_calls
        2. Sort each bucket by strike ascending
        3. Pair index-by-index: bucket[0] from each = IC#1, bucket[1] = IC#2, etc.
        4. Validate per-IC: LP < SP < SC < LC, matching quantities

        Returns list of TrackedIronCondor (without credits assigned yet), or None on error.
        """
        try:
            positions = await self.schwab_client.get_account_positions()
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return None

        spx_positions = [
            p for p in positions
            if 'SPX' in p.get('instrument', {}).get('symbol', '').upper()
        ]

        if not spx_positions:
            logger.error("No SPX option positions found.")
            return None

        if len(spx_positions) % 4 != 0:
            logger.error(
                f"Expected a multiple of 4 SPX legs, found {len(spx_positions)}. "
                f"Cannot group into iron condors."
            )
            return None

        # Parse each position
        legs = []
        for p in spx_positions:
            symbol = p.get('instrument', {}).get('symbol', '')
            parsed = self.builder.parse_option_symbol(symbol)
            if not parsed:
                logger.error(f"Failed to parse symbol: {symbol}")
                return None

            long_qty = p.get('longQuantity', 0)
            short_qty = p.get('shortQuantity', 0)
            net_qty = long_qty - short_qty

            parsed['symbol'] = symbol
            parsed['net_qty'] = net_qty
            parsed['abs_qty'] = abs(net_qty)
            parsed['avg_price'] = float(p.get('averagePrice', 0.0))
            legs.append(parsed)

        # Group legs by quantity first — legs from the same IC order share a quantity.
        # This avoids mis-pairing when strikes of different ICs interleave.
        from collections import defaultdict
        qty_groups = defaultdict(list)
        for leg in legs:
            qty_groups[leg['abs_qty']].append(leg)

        def _make_leg(parsed_leg):
            return {
                'symbol': parsed_leg['symbol'],
                'strike': parsed_leg['strike'],
                'avg_price': parsed_leg.get('avg_price', 0.0),
                'delta': 0,
                'bid': 0, 'ask': 0, 'mid': 0,
                'open_interest': 0, 'volume': 0,
            }

        tracked_ics = []
        ic_num = 0
        for qty, group_legs in sorted(qty_groups.items()):
            # Within this quantity group, bucket by type/direction
            lps = sorted([l for l in group_legs if l['option_type'] == 'P' and l['net_qty'] > 0],
                         key=lambda x: x['strike'])
            sps = sorted([l for l in group_legs if l['option_type'] == 'P' and l['net_qty'] < 0],
                         key=lambda x: x['strike'])
            scs = sorted([l for l in group_legs if l['option_type'] == 'C' and l['net_qty'] < 0],
                         key=lambda x: x['strike'])
            lcs = sorted([l for l in group_legs if l['option_type'] == 'C' and l['net_qty'] > 0],
                         key=lambda x: x['strike'])

            num_in_group = len(group_legs) // 4
            if len(group_legs) % 4 != 0 or not (len(lps) == len(sps) == len(scs) == len(lcs) == num_in_group):
                logger.error(
                    f"Quantity group {int(qty)}x has {len(group_legs)} legs that don't form ICs. "
                    f"LP={len(lps)}, SP={len(sps)}, SC={len(scs)}, LC={len(lcs)}"
                )
                return None

            for i in range(num_in_group):
                ic_num += 1
                lp, sp, sc, lc = lps[i], sps[i], scs[i], lcs[i]

                # Validate structure
                if not (lp['strike'] < sp['strike'] < sc['strike'] < lc['strike']):
                    logger.error(
                        f"IC#{ic_num} strikes don't form valid IC: "
                        f"{lp['strike']} < {sp['strike']} < {sc['strike']} < {lc['strike']}"
                    )
                    return None

                exp_date = sp['expiration']

                strikes = {
                    'long_put': _make_leg(lp),
                    'short_put': _make_leg(sp),
                    'short_call': _make_leg(sc),
                    'long_call': _make_leg(lc),
                    'net_credit': 0,  # assigned later
                    'expiration': exp_date,
                    'exp_key': f"{exp_date}:0",
                }

                label = (f"IC#{ic_num} ({int(lp['strike'])}/{int(sp['strike'])}p - "
                         f"{int(sc['strike'])}/{int(lc['strike'])}c)")

                tracked_ics.append(TrackedIronCondor(
                    ic_id=ic_num,
                    label=label,
                    strikes=strikes,
                    quantity=int(qty),
                    fill_credit=0,  # assigned later
                    portfolio_value=self.portfolio_value,
                ))

        logger.info(f"  Detected {len(tracked_ics)} iron condor(s) from {len(spx_positions)} positions")
        return tracked_ics

    async def _assign_credits(self, tracked_ics: List[TrackedIronCondor]) -> bool:
        """
        Assign fill credits to tracked ICs from CLI args or auto-detect from order history.

        Priority:
        1. --credits (comma-separated, multi-IC)
        2. --credit (single IC backward compat)
        3. Auto-detect from recent FILLED IRON_CONDOR orders
        """
        if self.monitor_credits:
            # --credits provided
            if len(self.monitor_credits) != len(tracked_ics):
                logger.error(
                    f"--credits has {len(self.monitor_credits)} values but "
                    f"found {len(tracked_ics)} ICs. Must match."
                )
                return False
            for ic, credit in zip(tracked_ics, self.monitor_credits):
                ic.fill_credit = credit
                ic.strikes['net_credit'] = credit
                ic.__post_init__()  # recompute thresholds
            logger.info(f"  Credits assigned from --credits: {self.monitor_credits}")
            return True

        if self.monitor_credit is not None:
            # --credit (single) backward compat
            if len(tracked_ics) != 1:
                logger.error(
                    f"--credit is for single IC but found {len(tracked_ics)} ICs. "
                    f"Use --credits with comma-separated values."
                )
                return False
            tracked_ics[0].fill_credit = self.monitor_credit
            tracked_ics[0].strikes['net_credit'] = self.monitor_credit
            tracked_ics[0].__post_init__()
            logger.info(f"  Credit assigned from --credit: ${self.monitor_credit:.2f}")
            return True

        # Auto-detect from order history (primary: traces entries + rolls)
        logger.info("  Auto-detecting credits from order history...")
        if await self._detect_credits_from_orders(tracked_ics):
            return True

        # Fallback: compute net credit from position averagePrice fields
        logger.info("  Order history match failed. Falling back to position averagePrice...")
        if self._detect_credits_from_avg_price(tracked_ics):
            return True

        logger.error(
            "Could not auto-detect credits from order history or position data. "
            "Use --credits 2.60,3.00 to specify manually."
        )
        return False

    def _detect_credits_from_avg_price(self, tracked_ics: List[TrackedIronCondor]) -> bool:
        """
        Compute net credit from position averagePrice fields.

        For each IC, net credit = sum(short leg avg prices) - sum(long leg avg prices).
        This works for any position regardless of how it was opened (manually, via bot, etc.).

        Returns True if all ICs got a valid credit assigned.
        """
        all_assigned = True
        for ic in tracked_ics:
            sp_avg = ic.strikes['short_put'].get('avg_price', 0.0)
            sc_avg = ic.strikes['short_call'].get('avg_price', 0.0)
            lp_avg = ic.strikes['long_put'].get('avg_price', 0.0)
            lc_avg = ic.strikes['long_call'].get('avg_price', 0.0)

            net_credit = round((sp_avg + sc_avg) - (lp_avg + lc_avg), 2)

            if net_credit <= 0:
                logger.warning(
                    f"  {ic.label}: averagePrice yields non-positive credit ${net_credit:.2f} "
                    f"(SP={sp_avg}, SC={sc_avg}, LP={lp_avg}, LC={lc_avg}). Skipping."
                )
                all_assigned = False
                continue

            ic.fill_credit = net_credit
            ic.strikes['net_credit'] = net_credit
            ic.__post_init__()  # recompute thresholds
            logger.info(
                f"  {ic.label}: credit ${net_credit:.2f} from position avgPrice "
                f"(SP={sp_avg:.2f} + SC={sc_avg:.2f} - LP={lp_avg:.2f} - LC={lc_avg:.2f})"
            )

        return all_assigned

    def _classify_order(self, order: dict) -> str:
        """
        Classify a filled order as ENTRY, ROLL, or CLOSE.

        - ENTRY: has SELL_TO_OPEN legs, no BUY_TO_CLOSE legs (original IC open)
        - ROLL: has both SELL_TO_OPEN and BUY_TO_CLOSE legs (adjusted a spread)
        - CLOSE: has BUY_TO_CLOSE legs, no SELL_TO_OPEN legs (full close)
        """
        legs = order.get('orderLegCollection', [])
        instructions = {l.get('instruction', '') for l in legs}
        has_sto = 'SELL_TO_OPEN' in instructions
        has_btc = 'BUY_TO_CLOSE' in instructions
        if has_sto and has_btc:
            return 'ROLL'
        elif has_sto:
            return 'ENTRY'
        elif has_btc:
            return 'CLOSE'
        return 'OTHER'

    async def _detect_credits_from_orders(self, tracked_ics: List[TrackedIronCondor]) -> bool:
        """
        Roll-aware credit detection from order history.

        Algorithm:
        1. Fetch all FILLED orders (last 3 days)
        2. Classify each as ENTRY, ROLL, or CLOSE
        3. Build tracked IC states from ENTRY orders (initial symbols + credit)
        4. Apply ROLLs chronologically: match BUY_TO_CLOSE symbols to an IC's
           current symbols, replace with SELL_TO_OPEN symbols, accumulate credit
        5. Match final tracked states to current positions by symbol set
        6. Assign effective credits (entry + sum of rolls)

        Returns True if all ICs got a credit assigned.
        """
        try:
            now_utc = datetime.now(pytz.utc)
            lookback_utc = now_utc - timedelta(days=3)
            all_orders = await self.schwab_client.get_orders_for_account(
                from_entered_datetime=lookback_utc,
                to_entered_datetime=now_utc,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch order history: {e}")
            return False

        if not all_orders:
            logger.warning("No orders found in last 3 days.")
            return False

        # Filter to FILLED orders
        filled_orders = [o for o in all_orders if o.get('status', '').upper() == 'FILLED']
        if not filled_orders:
            logger.warning(f"Found {len(all_orders)} orders but none with FILLED status.")
            return False

        logger.info(f"    Found {len(filled_orders)} filled orders (of {len(all_orders)} total)")

        # Sort by enteredTime ascending so rolls apply in chronological order
        def _order_time(o):
            t = o.get('enteredTime', '')
            try:
                return datetime.fromisoformat(t.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return datetime.min.replace(tzinfo=pytz.utc)

        filled_orders.sort(key=_order_time)

        # Classify all filled orders
        entries = []
        rolls = []
        for order in filled_orders:
            kind = self._classify_order(order)
            legs = order.get('orderLegCollection', [])
            order_qty = max((l.get('quantity', 0) for l in legs), default=0)

            if kind == 'ENTRY':
                # Collect all 4 symbols from the entry
                order_symbols = {
                    l.get('instrument', {}).get('symbol', '') for l in legs
                }
                entries.append({
                    'symbols': order_symbols,
                    'price': float(order.get('price', 0)),
                    'quantity': order_qty,
                    'order_id': order.get('orderId'),
                    'time': _order_time(order),
                })
                logger.info(f"    ENTRY order {order.get('orderId')}: "
                            f"{order_qty}x ${order.get('price'):.2f} "
                            f"type={order.get('complexOrderStrategyType')}")

            elif kind == 'ROLL':
                # Separate BUY_TO_CLOSE (old legs) from SELL_TO_OPEN (new legs)
                btc_symbols = set()
                sto_symbols = set()
                btc_short_strikes = []  # old short strikes being closed
                sto_short_strikes = []  # new short strikes being opened
                for l in legs:
                    sym = l.get('instrument', {}).get('symbol', '')
                    inst = l.get('instruction', '')
                    if inst == 'BUY_TO_CLOSE':
                        btc_symbols.add(sym)
                        parsed = self.builder.parse_option_symbol(sym)
                        if parsed:
                            btc_short_strikes.append(parsed)
                    elif inst == 'SELL_TO_OPEN':
                        sto_symbols.add(sym)
                        parsed = self.builder.parse_option_symbol(sym)
                        if parsed:
                            sto_short_strikes.append(parsed)

                # Determine if roll is closer-to-ATM (credit) or further (debit)
                # by comparing old short strike to new short strike
                roll_is_credit = True  # default: treat as credit
                if btc_short_strikes and sto_short_strikes:
                    # Match by option type (C or P)
                    for new_leg in sto_short_strikes:
                        for old_leg in btc_short_strikes:
                            if new_leg['option_type'] == old_leg['option_type']:
                                if new_leg['option_type'] == 'C':
                                    # Call: new < old = rolled down = closer to ATM = credit
                                    roll_is_credit = new_leg['strike'] < old_leg['strike']
                                else:
                                    # Put: new > old = rolled up = closer to ATM = credit
                                    roll_is_credit = new_leg['strike'] > old_leg['strike']
                                break
                        break  # only need to check one pair

                rolls.append({
                    'btc_symbols': btc_symbols,
                    'sto_symbols': sto_symbols,
                    'price': float(order.get('price', 0)),
                    'is_credit': roll_is_credit,
                    'quantity': order_qty,
                    'order_id': order.get('orderId'),
                    'time': _order_time(order),
                })
                direction = "credit" if roll_is_credit else "debit"
                logger.info(f"    ROLL order {order.get('orderId')}: "
                            f"{order_qty}x ${order.get('price'):.2f} ({direction}) "
                            f"close={btc_symbols} open={sto_symbols}")

            elif kind == 'CLOSE':
                logger.debug(f"    CLOSE order {order.get('orderId')}: skipped")

        if not entries:
            logger.warning("No entry orders found in filled orders.")
            return False

        # Build tracked IC states from entries, then apply rolls
        # Each "ic_state" tracks current symbols and cumulative credit
        ic_states = []
        for entry in entries:
            ic_states.append({
                'current_symbols': set(entry['symbols']),
                'cumulative_credit': entry['price'],
                'quantity': entry['quantity'],
                'order_id': entry['order_id'],
                'rolls_applied': 0,
            })

        # Apply rolls chronologically (already sorted)
        for roll in rolls:
            matched_state = None
            for state in ic_states:
                # Roll's BTC symbols must be a subset of the IC's current symbols
                # AND quantity must match
                if (roll['btc_symbols'].issubset(state['current_symbols'])
                        and roll['quantity'] == state['quantity']):
                    matched_state = state
                    break

            if matched_state is None:
                logger.warning(f"    Roll {roll['order_id']} didn't match any IC state "
                               f"(btc={roll['btc_symbols']}, qty={roll['quantity']})")
                continue

            # Replace closed symbols with new symbols
            matched_state['current_symbols'] -= roll['btc_symbols']
            matched_state['current_symbols'] |= roll['sto_symbols']

            # Accumulate credit/debit
            if roll['is_credit']:
                matched_state['cumulative_credit'] += roll['price']
            else:
                matched_state['cumulative_credit'] -= roll['price']

            matched_state['rolls_applied'] += 1
            logger.info(f"    Applied roll {roll['order_id']} to entry {matched_state['order_id']}: "
                        f"cumulative credit now ${matched_state['cumulative_credit']:.2f} "
                        f"(+{matched_state['rolls_applied']} rolls)")

        # Match final IC states to current tracked ICs by symbol set
        matched = 0
        for ic in tracked_ics:
            ic_symbols = {
                ic.strikes['long_put']['symbol'],
                ic.strikes['short_put']['symbol'],
                ic.strikes['short_call']['symbol'],
                ic.strikes['long_call']['symbol'],
            }
            for state in ic_states:
                if (state['current_symbols'] == ic_symbols
                        and state['quantity'] == ic.quantity):
                    ic.fill_credit = round(state['cumulative_credit'], 2)
                    ic.strikes['net_credit'] = ic.fill_credit
                    ic.__post_init__()  # recompute thresholds
                    rolls_note = f" (+{state['rolls_applied']} rolls)" if state['rolls_applied'] else ""
                    logger.info(f"    {ic.label}: effective credit ${ic.fill_credit:.2f}{rolls_note} "
                                f"(entry order {state['order_id']})")
                    matched += 1
                    break

        if matched != len(tracked_ics):
            logger.warning(f"Only matched {matched}/{len(tracked_ics)} ICs to order history.")
            # Debug: show what we expected vs what we have
            for ic in tracked_ics:
                if ic.fill_credit == 0:
                    ic_syms = {
                        ic.strikes['long_put']['symbol'],
                        ic.strikes['short_put']['symbol'],
                        ic.strikes['short_call']['symbol'],
                        ic.strikes['long_call']['symbol'],
                    }
                    logger.warning(f"    Unmatched {ic.label} (qty={ic.quantity}): {ic_syms}")
            for state in ic_states:
                logger.warning(f"    IC state (entry {state['order_id']}, qty={state['quantity']}): "
                               f"{state['current_symbols']}")
            return False

        return True

    async def _detect_tp_order_for_ic(self, ic: TrackedIronCondor) -> Optional[str]:
        """Find a working BUY_TO_CLOSE IRON_CONDOR order matching this IC's symbols."""
        try:
            orders = await self.schwab_client.get_orders_for_account(status='WORKING')
        except Exception as e:
            logger.warning(f"Failed to fetch working orders: {e}")
            return None

        if not orders:
            return None

        ic_symbols = {
            ic.strikes['long_put']['symbol'],
            ic.strikes['short_put']['symbol'],
            ic.strikes['short_call']['symbol'],
            ic.strikes['long_call']['symbol'],
        }

        for order in orders:
            if order.get('complexOrderStrategyType') != 'IRON_CONDOR':
                continue
            legs = order.get('orderLegCollection', [])
            has_btc = any(l.get('instruction') == 'BUY_TO_CLOSE' for l in legs)
            if not has_btc:
                continue
            order_symbols = {
                l.get('instrument', {}).get('symbol', '') for l in legs
            }
            if order_symbols == ic_symbols:
                return str(order.get('orderId', ''))

        return None

    async def multi_ic_monitoring_loop(self, tracked_ics: List[TrackedIronCondor]):
        """
        Monitor multiple ICs with tiered TP/SL alerts.

        Each cycle:
        1. Fetch chain ONCE (shared across all ICs)
        2. For each non-closed IC: check TP fill, evaluate debit, fire alerts
        3. Time exit: close all remaining ICs at BOT_EXIT_TIME
        4. Exit when all ICs closed
        """
        exit_time = self._parse_time(settings.BOT_EXIT_TIME)

        logger.info("--- Multi-IC Monitoring Loop Started ---")
        logger.info(f"  Monitoring {len(tracked_ics)} IC(s)")
        logger.info(f"  Exit time: {settings.BOT_EXIT_TIME} ET")
        logger.info(f"  Check interval: {settings.BOT_MONITOR_INTERVAL}s")

        while True:
            try:
                now = self._now_et()
                open_ics = [ic for ic in tracked_ics if not ic.is_closed]

                if not open_ics:
                    logger.info("=== All ICs closed. Monitoring complete. ===")
                    break

                # Time exit check
                if now.time() >= exit_time and not self.skip_wait:
                    logger.info(f"=== TIME EXIT: Past {settings.BOT_EXIT_TIME} ET ===")
                    for ic in open_ics:
                        await self._close_ic(ic, "time_exit")
                    break

                # Fetch chain once for all ICs
                chain = await self._fetch_0dte_chain()
                if not chain:
                    logger.warning("  Could not fetch chain this cycle, retrying...")
                    await asyncio.sleep(settings.BOT_MONITOR_INTERVAL)
                    continue

                for ic in open_ics:
                    # Check if TP limit order filled
                    if ic.tp_order_id:
                        if await self._check_order_filled(ic.tp_order_id):
                            ic.exit_reason = "take_profit_50"
                            ic.is_closed = True
                            logger.info(f"  === {ic.label}: TP limit order FILLED ===")
                            continue

                    # Get current debit
                    debit = self.builder.get_current_position_value(chain, ic.strikes)
                    if debit is None:
                        logger.debug(f"  {ic.label}: could not get position value")
                        continue

                    pnl_pct = round((1 - debit / ic.fill_credit) * 100) if ic.fill_credit > 0 else 0
                    if ic.sltp_active_tier_idx >= 0:
                        act_pct = ic.sltp_tiers[ic.sltp_active_tier_idx][0]
                        flr_pct = ic.sltp_tiers[ic.sltp_active_tier_idx][1]
                        sltp_tag = f" [SLTP {act_pct:.1%}->{flr_pct:.1%} floor@${ic.sltp_floor_debit:.2f}]"
                    else:
                        sltp_tag = ""
                    logger.info(
                        f"  [{now.strftime('%H:%M:%S')}] {ic.label}: "
                        f"debit ${debit:.2f}  P/L {pnl_pct:+d}%{sltp_tag}"
                    )

                    # --- Tiered SLTP: ratcheting profit lock ---
                    if ic.sltp_tiers:
                        # Check if we've reached a higher tier (scan from highest to current+1)
                        for idx in range(len(ic.sltp_tiers) - 1, ic.sltp_active_tier_idx, -1):
                            act_pct, flr_pct, act_debit, flr_debit = ic.sltp_tiers[idx]
                            if debit <= act_debit:
                                ic.sltp_active_tier_idx = idx
                                ic.sltp_floor_debit = flr_debit
                                logger.info(
                                    f"  >>> {ic.label}: SLTP TIER {act_pct:.1%} ACTIVATED "
                                    f"(debit ${debit:.2f} <= ${act_debit:.2f})"
                                )
                                logger.info(
                                    f"      Floor raised to ${flr_debit:.2f} (locks in {flr_pct:.1%} gain)"
                                )
                                play_sound(TP_SOUND)
                                send_notification(
                                    f"Profit Lock {act_pct:.1%} - {ic.label}",
                                    f"Floor ${flr_debit:.2f} ({flr_pct:.1%})"
                                )
                                break  # Only need highest matching tier

                        # Check if floor breached (debit reversed up past lock-in)
                        if ic.sltp_active_tier_idx >= 0 and debit >= ic.sltp_floor_debit:
                            flr_pct = ic.sltp_tiers[ic.sltp_active_tier_idx][1]
                            logger.warning(
                                f"  === {ic.label}: SLTP TRIGGERED at {flr_pct:.1%} floor "
                                f"(debit ${debit:.2f} >= floor ${ic.sltp_floor_debit:.2f}) ==="
                            )
                            play_sound(SL_SOUND)
                            await self._close_ic(ic, f"sltp_lock_{flr_pct:.1%}")
                            continue

                    # TP checks (debit falling = good)
                    if debit <= ic.tp_50_debit and AlertLevel.TP_50 not in ic.alerts_fired:
                        ic.alerts_fired.add(AlertLevel.TP_50)
                        ic.alerts_fired.add(AlertLevel.TP_25)  # skip 25 if already at 50
                        alert_tp(ic, AlertLevel.TP_50, debit)

                    elif debit <= ic.tp_25_debit and AlertLevel.TP_25 not in ic.alerts_fired:
                        ic.alerts_fired.add(AlertLevel.TP_25)
                        alert_tp(ic, AlertLevel.TP_25, debit)

                    # SL checks (debit rising = bad) — exit first, then warning
                    if debit >= ic.sl_exit_debit and AlertLevel.SL_EXIT not in ic.alerts_fired:
                        ic.alerts_fired.add(AlertLevel.SL_EXIT)
                        ic.alerts_fired.add(AlertLevel.SL_WARNING)
                        alert_sl(ic, AlertLevel.SL_EXIT, debit)
                        await self._close_ic(ic, "stop_loss_portfolio")

                    elif debit >= ic.sl_warning_debit and AlertLevel.SL_WARNING not in ic.alerts_fired:
                        ic.alerts_fired.add(AlertLevel.SL_WARNING)
                        alert_sl(ic, AlertLevel.SL_WARNING, debit)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)

            if self.skip_wait:
                logger.info("[SKIP-WAIT] Exiting monitoring loop after one check.")
                break

            await asyncio.sleep(settings.BOT_MONITOR_INTERVAL)

    async def _place_tp_limit_order(self, ic: TrackedIronCondor, debit: float):
        """Place a BUY_TO_CLOSE limit order for a specific IC."""
        # Ceil to $0.05 (fill-friendly: we're buying back, higher = more likely to fill)
        debit = round(math.ceil(debit * 20) / 20, 2)
        debit = max(debit, 0.05)

        close_order = self.builder.build_close_order(ic.strikes, ic.quantity, debit)
        self._log_order(f"TP-LIMIT {ic.label}", close_order)

        if self.dry_run:
            logger.info(f"  [DRY-RUN] TP limit order NOT placed for {ic.label}")
            return

        try:
            result = await self.schwab_client.place_order(close_order)
            ic.tp_order_id = result.get('orderId')
            logger.info(f"  {ic.label}: TP limit order placed: ID={ic.tp_order_id} at ${debit:.2f}")
        except Exception as e:
            logger.error(f"  {ic.label}: Failed to place TP limit order: {e}")

    async def _close_ic(self, ic: TrackedIronCondor, reason: str):
        """Cancel working TP (if any) and market-close an IC."""
        logger.warning(f"  {ic.label}: Closing position ({reason})")
        ic.exit_reason = reason
        ic.is_closed = True

        # Cancel existing TP order
        if ic.tp_order_id and not self.dry_run:
            try:
                await self.schwab_client.cancel_order(ic.tp_order_id)
                logger.info(f"  {ic.label}: TP order {ic.tp_order_id} cancelled")
            except Exception as e:
                logger.warning(f"  {ic.label}: Failed to cancel TP order: {e}")

        # Place market close
        close_order = self.builder.build_market_close_order(ic.strikes, ic.quantity)
        self._log_order(f"MARKET-CLOSE {ic.label}", close_order)

        if self.dry_run:
            logger.info(f"  [DRY-RUN] Market close NOT placed for {ic.label}")
            return

        try:
            result = await self.schwab_client.place_order(close_order)
            close_id = result.get('orderId', '?')
            logger.info(f"  {ic.label}: Market close order placed: ID={close_id}")
        except Exception as e:
            logger.critical(f"  {ic.label}: FAILED TO PLACE MARKET CLOSE: {e}")
            logger.critical("  MANUAL INTERVENTION REQUIRED!")

    async def _check_order_filled(self, order_id: str) -> bool:
        """Check if an order has filled."""
        if self.dry_run or not order_id:
            return False
        try:
            order_data = await self.schwab_client.get_order(order_id)
            return order_data.get('status', '') == 'FILLED'
        except Exception as e:
            logger.warning(f"Error checking order {order_id}: {e}")
            return False

    async def _cancel_order_safe(self, order_id: str):
        """Cancel an order, ignoring errors."""
        if self.dry_run or not order_id:
            return
        try:
            await self.schwab_client.cancel_order(order_id)
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")

    async def _wait_until(self, time_str: str):
        """Sleep until the specified ET time."""
        target = self._parse_time(time_str)
        now = self._now_et()
        target_dt = now.replace(
            hour=target.hour, minute=target.minute, second=0, microsecond=0
        )

        if now >= target_dt:
            logger.info(f"Already past {time_str} ET, proceeding immediately.")
            return

        wait_seconds = (target_dt - now).total_seconds()
        logger.info(f"Waiting until {time_str} ET ({wait_seconds:.0f}s)...")

        # Sleep in 60s chunks for responsiveness to Ctrl+C
        while True:
            now = self._now_et()
            if now >= target_dt:
                break
            remaining = (target_dt - now).total_seconds()
            await asyncio.sleep(min(remaining, 60))

        logger.info(f"Reached {time_str} ET. Proceeding.")

    def _log_order(self, label: str, order: dict):
        """Log an order structure for debugging."""
        logger.info(f"  [{label}] Order:")
        logger.info(f"    Type: {order.get('orderType')} / {order.get('complexOrderStrategyType')}")
        if 'price' in order:
            logger.info(f"    Price: ${order['price']}")
        for leg in order.get('orderLegCollection', []):
            sym = leg.get('instrument', {}).get('symbol', '?')
            inst = leg.get('instruction', '?')
            qty = leg.get('quantity', 0)
            logger.info(f"    {inst} {qty}x {sym}")


def main():
    parser = argparse.ArgumentParser(description="0DTE Iron Condor Trading Bot")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--dry-run', action='store_true', help='Paper trading mode')
    mode_group.add_argument('--live', action='store_true', help='Real money trading')
    mode_group.add_argument('--monitor', action='store_true',
                            help='Monitor existing position(s) with tiered TP/SL alerts')
    parser.add_argument('--credit', type=float, default=None,
                        help='Fill credit for single-IC --monitor mode')
    parser.add_argument('--credits', type=str, default=None,
                        help='Comma-separated credits for multi-IC --monitor (e.g. "2.60,3.00"). '
                             'Auto-detected from order history if omitted.')
    parser.add_argument('--skip-wait', action='store_true', help='Skip time-based waiting')
    parser.add_argument('--skip-ml', action='store_true', help='Bypass ML confidence check')
    parser.add_argument('--no-v5', action='store_true',
                        help='Disable v5 TP models — fall back to v4 or fixed-time entry')
    parser.add_argument('--no-v4', action='store_true',
                        help='Disable v4 entry timing — use fixed-time entry at BOT_ENTRY_TIME')
    parser.add_argument('--manual-approve', action='store_true',
                        help='Prompt for manual approval when ML score is below threshold')
    parser.add_argument('--strikes', type=str, default=None,
                        help='Manual short strikes as "PUT,CALL" (e.g. "6750,6920"). Bypasses delta selection.')

    args = parser.parse_args()

    # Parse --credits into list of floats
    monitor_credits = None
    if args.credits:
        try:
            monitor_credits = [float(x.strip()) for x in args.credits.split(',')]
        except ValueError:
            parser.error(f"Invalid --credits format: '{args.credits}'. Use comma-separated numbers, e.g. '2.60,3.00'")

    # Parse manual strikes
    manual_strikes = None
    if args.strikes:
        try:
            parts = args.strikes.split(',')
            manual_strikes = (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            print(f"Invalid --strikes format: {args.strikes}. Use 'PUT,CALL' e.g. '6750,6920'")
            return

    # Safety check for live mode
    if args.live:
        confirm = input(
            "\n*** LIVE TRADING MODE ***\n"
            "This will place REAL orders with REAL money.\n"
            f"Symbol: {settings.BOT_SYMBOL}\n"
            f"Max contracts: {settings.BOT_MAX_CONTRACTS}\n"
            f"Portfolio value: live from Schwab (fallback: ${settings.BOT_PORTFOLIO_VALUE:,.0f})\n"
            "\nType 'YES' to confirm: "
        )
        if confirm != 'YES':
            print("Aborted.")
            return

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
    log_path = Path("logs") / f"zero_dte_bot_{date.today()}.log"
    log_path.parent.mkdir(exist_ok=True)
    logger.add(str(log_path), level="DEBUG", rotation="10 MB")

    bot = ZeroDTEBot(
        dry_run=args.dry_run if not args.monitor else False,
        skip_wait=args.skip_wait,
        skip_ml=args.skip_ml,
        no_v4=args.no_v4,
        no_v5=args.no_v5,
        manual_strikes=manual_strikes,
        monitor_mode=args.monitor,
        monitor_credit=args.credit,
        monitor_credits=monitor_credits,
        manual_approve=args.manual_approve,
    )

    asyncio.run(bot.run())


if __name__ == '__main__':
    main()
