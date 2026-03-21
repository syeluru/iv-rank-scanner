#!/usr/bin/env python3
"""
0DTE Iron Condor Automated Trading Bot — Persistent 24/7 Mode.

Runs continuously: sleeps overnight/weekends, wakes for each trading day.
Supports re-entry after a trade closes (if ML still signals and within entry window).
Daily circuit breakers: 5% portfolio gain OR 10% portfolio loss stops trading for the day.

Single adaptive flow that auto-detects Schwab state:
  - No positions → ML scan entry → monitor
  - Positions exist → reconstruct ICs → monitor
  - Adapts to manual changes (rolls, closes, new ICs) mid-monitoring
  - After trade closes: re-enters if within scan window and circuit breakers allow

Entry modes (automatic fallback chain):
  v5 dynamic (default): Scans 10:00-14:00 ET every 5 min using ML TP prediction
                         models. Enters when P(hit 25% TP) >= threshold.
  v4 dynamic (fallback): Same scan loop but predicts P(profitable at 3PM close).
                          Used when v5 models not loaded.
  Fixed-time (fallback): Enters at BOT_ENTRY_TIME (noon ET) with v3 risk filter.
                          Used when v4 model not loaded.

Exit via: SLTP trailing lock (primary), 5% ceiling TP (Schwab limit),
          stop-loss (bot-monitored), or 3PM forced close.

Manual-approve is always on: prompts when ML score is below threshold.

Usage:
  python scripts/zero_dte_bot.py                          # Live persistent mode (prompts YES)
  python scripts/zero_dte_bot.py --paper                   # Paper persistent mode
  python scripts/zero_dte_bot.py --paper --skip-wait --skip-ml  # Quick test (single cycle)
  python scripts/zero_dte_bot.py --strikes 6875,6945       # Manual strikes
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

# ZDOM V1 replaces old ML chain when BOT_ZDOM_ENABLED=True
# Old imports kept for fallback when ZDOM disabled
try:
    from ml.models.trade_decision import TradeDecisionEngine, TradeAction
    from ml.models.strategy_scorer import StrategyScorer  # noqa: E402
    _OLD_ML_AVAILABLE = True
except ImportError:
    _OLD_ML_AVAILABLE = False


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

    # ZDOM: parameterized TP percentage (None = use default 25%)
    zdom_tp_pct: Optional[float] = None
    # ZDOM: strategy name (e.g. "IC_15d_25w") for dedup during continuous scanning
    zdom_strategy: Optional[str] = None

    # Mutable state
    alerts_fired: Set[AlertLevel] = field(default_factory=set)
    tp_order_id: Optional[str] = None
    exit_reason: Optional[str] = None
    is_closed: bool = False

    # Computed thresholds (set in __post_init__)
    tp_25_debit: float = 0.0
    tp_50_debit: float = 0.0
    sl_warning_debit: float = 0.0
    sl_exit_debit: float = 0.0

    # Fixed 25% TP debit (placed as limit order)
    target_tp_debit: float = 0.0

    # Credit-based SLTP (trailing profit lock): activates at 30%, locks 25%, trails every 5%
    # Each tuple: (activation_pct, floor_pct, activation_debit, floor_debit)
    sltp_tiers: List[Tuple[float, float, float, float]] = field(default_factory=list)
    sltp_active_tier_idx: int = -1        # Index of highest activated tier (-1 = none)
    sltp_floor_debit: float = 0.0         # Current active floor debit (from highest tier)

    def __post_init__(self):
        # Round-trip transaction cost: 4 legs open + 4 legs close = 8 legs
        fee = (settings.BOT_COST_PER_LEG * 8) / 100  # per-share fee
        fee_dollars = settings.BOT_COST_PER_LEG * 8   # $5.20

        # TP: max debit to keep X% of credit after fees
        self.tp_25_debit = round(self.fill_credit * 0.75 - fee, 2)
        self.tp_50_debit = round(self.fill_credit * 0.50 - fee, 2)

        # SL: dual approach when ZDOM enabled
        #   - ZDOM: 2x credit SL (matches model training)
        #   - Portfolio SL: hard floor safety
        #   - Use min(ZDOM_SL, portfolio_SL)
        if self.quantity > 0:
            # Portfolio-based SL
            max_loss = self.portfolio_value * settings.BOT_STOP_LOSS_PCT
            closing_fees = self.quantity * 4 * settings.BOT_COST_PER_LEG
            portfolio_sl = round(
                self.fill_credit + (max_loss - closing_fees) / (self.quantity * 100), 2
            )

            # ZDOM SL: min($cap, 2x credit) as max loss, converted to debit level
            if settings.BOT_ZDOM_ENABLED and self.zdom_tp_pct is not None:
                zdom_max_loss = round(min(
                    settings.BOT_ZDOM_SL_CAP,
                    self.fill_credit * settings.BOT_ZDOM_SL_MULT,
                ), 2)
                zdom_sl_debit = round(self.fill_credit + zdom_max_loss, 2)
                self.sl_exit_debit = min(zdom_sl_debit, portfolio_sl)
            else:
                self.sl_exit_debit = portfolio_sl

            self.sl_warning_debit = round(
                self.fill_credit + (self.sl_exit_debit - self.fill_credit) / 2, 2
            )

        # Credit-based SLTP: activates at 30% profit, locks 25%, trails every 5%
        if self.fill_credit > 0 and self.quantity > 0:
            self.sltp_tiers = []
            act_pct = settings.BOT_SLTP_ACTIVATE_PCT  # 0.30
            step = settings.BOT_SLTP_STEP_PCT          # 0.05
            offset = settings.BOT_SLTP_LOCK_OFFSET_PCT  # 0.05
            while act_pct <= 1.0 + 1e-9:
                floor_pct = act_pct - offset
                act_debit = round(self.fill_credit * (1.0 - act_pct), 2)
                flr_debit = round(self.fill_credit * (1.0 - floor_pct), 2)
                self.sltp_tiers.append((act_pct, floor_pct, act_debit, flr_debit))
                act_pct = round(act_pct + step, 4)
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
    """Automated 0DTE iron condor trading bot — unified adaptive mode."""

    def __init__(self, dry_run: bool = True, skip_wait: bool = False,
                 skip_ml: bool = False, manual_strikes: tuple = None):
        self.dry_run = dry_run
        self.skip_wait = skip_wait
        self.skip_ml = skip_ml
        self.manual_strikes = manual_strikes  # (short_put, short_call) or None

        # Sync global PAPER_TRADING setting with bot's dry_run flag
        settings.PAPER_TRADING = dry_run

        self.builder = IronCondorOrderBuilder()
        # (long IC builder removed — see zero_dte_all_strategies.py)

        # State
        self.already_traded_today = False
        self.entry_order_id = None
        self.tp_order_id = None
        self.strikes = None
        self.quantity = 0
        self.fill_credit = 0.0
        self.exit_reason = None
        self.portfolio_value = 0.0  # populated from Schwab at startup

        # v4/v5 entry timing state (used when ZDOM disabled)
        self._v3_cached_confidence = None
        self._v4_daily_data_cache = None
        self._chosen_delta = None

        # Unified strategy decision state (used when ZDOM disabled)
        self._trade_decision = None
        if _OLD_ML_AVAILABLE and not settings.BOT_ZDOM_ENABLED:
            self._decision_engine = TradeDecisionEngine(
                min_ev=settings.BOT_MIN_EV_THRESHOLD
                if hasattr(settings, 'BOT_MIN_EV_THRESHOLD') else 0.10,
                min_short_v3=0.45,
                long_entry_start=settings.BOT_LONG_IC_ENTRY_START
                if hasattr(settings, 'BOT_LONG_IC_ENTRY_START') else "11:00",
                long_entry_end=settings.BOT_LONG_IC_ENTRY_END
                if hasattr(settings, 'BOT_LONG_IC_ENTRY_END') else "12:30",
            )
        else:
            self._decision_engine = None

        # ZDOM V1 predictor
        self._zdom_predictor = None
        self._zdom_blocker_checked = False

        # Lazy-loaded components
        self._schwab_client = None
        self._predictor = None
        self._candle_loader = None
        self._feature_pipeline = None

        mode = "PAPER" if dry_run else "LIVE"
        logger.info(f"ZeroDTEBot initialized in {mode} mode")
        if skip_wait:
            logger.info("Skip-wait enabled: bypassing time-based sleeping")
        if skip_ml:
            logger.info("Skip-ML enabled: bypassing ML confidence check")

    @property
    def schwab_client(self):
        if self._schwab_client is None:
            if settings.BROKER.lower() == "tradier":
                from execution.broker_api.tradier_client import get_tradier_client
                self._schwab_client = get_tradier_client()
            else:
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
    def zdom_predictor(self):
        if self._zdom_predictor is None:
            from ml.zdom_v1.predictor import ZdomV1Predictor
            self._zdom_predictor = ZdomV1Predictor(
                models_dir=settings.BOT_ZDOM_MODELS_DIR,
                data_dir=settings.BOT_ZDOM_DATA_DIR,
                model_suffix=getattr(settings, 'BOT_ZDOM_MODEL_SUFFIX', ''),
            )
        return self._zdom_predictor

    # (long_ic_builder property removed — see zero_dte_all_strategies.py)

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
        logger.warning(f"  Could not read liquidationValue from broker, "
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

    async def _compute_entry_preview(self, chain: dict, target_delta: float = None) -> Optional[dict]:
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
                    target_delta=target_delta or settings.BOT_SHORT_DELTA,
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

            # Compute TP/SL debits: ceiling TP at daily gain limit (SLTP tiers are primary exit)
            fee_dollars = settings.BOT_COST_PER_LEG * 8
            max_gain = self.portfolio_value * settings.BOT_DAILY_GAIN_LIMIT_PCT
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

    # ===== Persistent Bot Lifecycle =====

    async def run(self):
        """Persistent bot lifecycle: sleep → wake → trade → repeat forever."""
        logger.info("=" * 60)
        logger.info("0DTE Iron Condor Bot Starting (PERSISTENT MODE)")
        logger.info(f"Mode: {'PAPER' if self.dry_run else 'LIVE'}")
        logger.info(f"Daily loss limit: -{settings.BOT_DAILY_LOSS_LIMIT_PCT:.0%}")
        logger.info("=" * 60)

        try:
            while True:
                try:
                    # Step 1: Wait for market open (sleeps overnight/weekends)
                    await self._wait_for_market_open()

                    # Step 2: Reset all daily state
                    self._reset_for_new_day()

                    # Step 3: Pre-flight (Schwab, ThetaData, ML)
                    if not await self.pre_flight():
                        logger.error("Pre-flight check failed. Sleeping until next day.")
                        await self._sleep_past_exit()
                        continue

                    # Step 4: Run the trading day (inner loop allows multiple trades)
                    await self._run_trading_day()

                    # Step 5: Sleep past exit time to avoid re-triggering same day
                    await self._sleep_past_exit()

                except KeyboardInterrupt:
                    raise  # propagate to outer handler
                except Exception as e:
                    logger.error(f"Unhandled error in daily cycle: {e}", exc_info=True)
                    if self.skip_wait:
                        break
                    logger.info("Sleeping 60s before retry...")
                    await asyncio.sleep(60)

                # In skip-wait mode, exit after one cycle (preserves test behavior)
                if self.skip_wait:
                    break

        except KeyboardInterrupt:
            logger.warning("Bot interrupted by user (Ctrl+C)")
        finally:
            self.log_trade_result()
            # Close DuckDB connections to release file locks
            if self._candle_loader is not None:
                try:
                    self._candle_loader.db.close()
                    logger.info("Closed CandleLoader DuckDB connection.")
                except Exception:
                    pass
            logger.info("Bot shutdown complete.")

    async def _run_trading_day(self):
        """
        Inner day loop: allows multiple trades within a single trading day.

        After a trade closes (TP/SL/SLTP), checks circuit breakers and re-enters
        if ML still signals and within the entry scan window.
        """
        exit_time = self._parse_time(settings.BOT_EXIT_TIME)
        trade_num = 0

        while True:
            now = self._now_et()

            # Time gate
            if now.time() >= exit_time and not self.skip_wait:
                logger.info(f"Past exit time {settings.BOT_EXIT_TIME} ET. Done for the day.")
                break

            # Circuit breaker check
            cb_reason = await self._check_daily_circuit_breaker()
            if cb_reason:
                logger.warning(f"=== CIRCUIT BREAKER: {cb_reason} ===")
                logger.info("No more trades today.")
                play_sound(SL_SOUND)
                send_notification("0DTE Bot — Circuit Breaker", cb_reason)
                break

            # Detect existing positions
            positions = await self._get_spx_positions()

            if positions:
                # MONITOR PATH: reconstruct ICs from live positions
                tracked_ics = await self._reconstruct_and_assign(positions)
                if not tracked_ics:
                    logger.error("Failed to reconstruct ICs from positions")
                    break

                trade_num += 1
                logger.info(f"Trade #{trade_num}: Detected {len(tracked_ics)} existing IC(s) — monitoring")
                self._log_ic_details(tracked_ics)
            else:
                # ENTRY PATH: double-check positions are truly empty
                # (guards against API returning stale empty data briefly)
                await asyncio.sleep(3)
                positions_recheck = await self._get_spx_positions()
                if positions_recheck:
                    logger.warning("Positions appeared on recheck — looping back to monitor path")
                    continue

                # Also check for any working STO iron condor orders (entry in flight)
                try:
                    working = await self.schwab_client.get_orders_for_account(status='WORKING')
                    has_pending_entry = any(
                        o.get('complexOrderStrategyType') == 'IRON_CONDOR'
                        and any(l.get('instruction') == 'SELL_TO_OPEN'
                                for l in o.get('orderLegCollection', []))
                        for o in (working or [])
                    )
                    if has_pending_entry:
                        logger.warning("Working STO iron condor order detected — skipping entry")
                        continue
                except Exception as e:
                    logger.debug(f"Working order check failed: {e}")

                # ML scan -> entry -> build IC(s)
                entry_ics = await self._entry_phase()

                if not entry_ics:
                    # Check if unified engine recommended LONG_ONLY (legacy only)
                    if (_OLD_ML_AVAILABLE
                            and self._trade_decision
                            and self._trade_decision.action == TradeAction.LONG_ONLY):
                        logger.info(f"=== LONG IC RECOMMENDED at "
                                    f"{int(self._trade_decision.long_delta*100)}δ "
                                    f"(EV=${self._trade_decision.long_ev:+.2f}) ===")
                        logger.info(f"  Entry window: "
                                    f"{self._trade_decision.long_recommended_entry_start}"
                                    f"-{self._trade_decision.long_recommended_entry_end} ET")
                        logger.info(f"  Long IC execution is not yet automated.")
                        logger.info(f"  Reasons:")
                        for r in self._trade_decision.reasons:
                            logger.info(f"    → {r}")
                        play_sound(TP_SOUND)
                        send_notification(
                            "0DTE Bot — LONG IC Signal",
                            f"Long IC @ {int(self._trade_decision.long_delta*100)}δ "
                            f"EV=${self._trade_decision.long_ev:+.2f}"
                        )

                    # Don't quit for the day — keep looping to detect positions
                    # that may appear (manual entries, delayed fills, etc.)
                    # and to re-scan if still within the entry window.
                    if settings.BOT_ZDOM_ENABLED:
                        scan_end_t = self._parse_time(settings.BOT_ZDOM_ENTRY_END)
                        scan_interval = settings.BOT_ZDOM_SCAN_INTERVAL
                    else:
                        scan_end_t = self._parse_time(settings.BOT_V4_ENTRY_END)
                        scan_interval = settings.BOT_V4_SCAN_INTERVAL
                    now_after = self._now_et()
                    if now_after.time() < scan_end_t and not self.skip_wait:
                        logger.info("No entry signal yet — will re-scan next interval.")
                        await asyncio.sleep(scan_interval * 60)
                        continue
                    elif now_after.time() < exit_time and not self.skip_wait:
                        logger.info("Entry scan window closed — monitoring for positions until exit time.")
                        await asyncio.sleep(60)
                        continue
                    else:
                        logger.info("No entry signal and past exit time. Done for the day.")
                        break

                trade_num += 1
                logger.info(f"Trade #{trade_num}: {len(entry_ics)} position(s) entered — monitoring")
                tracked_ics = entry_ics

            # Sync TP orders: detect existing, cancel+replace if quantity mismatch
            for ic in tracked_ics:
                await self._sync_tp_for_ic(ic)

            # Monitor until all ICs close or time exit
            await self._monitoring_loop(tracked_ics)

            # Log results for each IC
            for ic in tracked_ics:
                self._log_trade_result_for_ic(ic, trade_num)

            logger.info(f"Trade #{trade_num} monitoring complete.")

            # In skip-wait mode, exit after one trade
            if self.skip_wait:
                break

            # Reset for potential re-entry
            self._reset_for_new_trade()
            await self._refresh_portfolio_value()

            # Brief pause before re-checking
            await asyncio.sleep(5)

    def _log_ic_details(self, tracked_ics: list):
        """Log IC details with leg-level pricing for monitoring start."""
        for ic in tracked_ics:
            tp_info = f"  ZDOM TP: {ic.zdom_tp_pct:.0%}" if ic.zdom_tp_pct else ""
            logger.info(f"  {ic.label}: "
                        f"{ic.strikes['long_put']['strike']}/{ic.strikes['short_put']['strike']}p - "
                        f"{ic.strikes['short_call']['strike']}/{ic.strikes['long_call']['strike']}c "
                        f"x{ic.quantity}  Credit: ${ic.fill_credit:.2f}{tp_info}")

            # Leg detail table
            legs = [
                ("Long Put (BTO)",  ic.strikes['long_put'],  "BUY"),
                ("Short Put (STO)", ic.strikes['short_put'], "SELL"),
                ("Short Call (STO)", ic.strikes['short_call'], "SELL"),
                ("Long Call (BTO)", ic.strikes['long_call'], "BUY"),
            ]
            logger.info(f"    {'Leg':<20s} {'Strike':>7s} {'Delta':>7s} "
                        f"{'Bid':>7s} {'Ask':>7s} {'Mid':>7s} {'Action':>6s}")
            logger.info(f"    {'─'*62}")
            total_credit = 0.0
            for label, leg, action in legs:
                strike = leg.get('strike', 0)
                delta = leg.get('delta', 0)
                bid = leg.get('bid', 0)
                ask = leg.get('ask', 0)
                mid = leg.get('mid', 0)
                if action == "SELL":
                    total_credit += mid
                else:
                    total_credit -= mid
                logger.info(f"    {label:<20s} {strike:>7.0f} {delta:>+7.3f} "
                            f"${bid:>6.2f} ${ask:>6.2f} ${mid:>6.2f} {action:>6s}")
            logger.info(f"    {'─'*62}")
            logger.info(f"    {'Net Credit (mid)':<20s} {'':>7s} {'':>7s} "
                        f"{'':>7s} {'':>7s} ${total_credit:>6.2f}")
            logger.info(f"    Fill Credit: ${ic.fill_credit:.2f}  |  "
                        f"Qty: {ic.quantity}  |  "
                        f"Notional: ${ic.fill_credit * ic.quantity * 100:,.0f}")

            sl_source = (f"min(${settings.BOT_ZDOM_SL_CAP:.0f}, "
                         f"{settings.BOT_ZDOM_SL_MULT}x credit)"
                         if ic.zdom_tp_pct else f"{settings.BOT_STOP_LOSS_PCT:.0%} portfolio")
            logger.info(f"    SL exit:  debit >= ${ic.sl_exit_debit:.2f} -> MARKET CLOSE "
                        f"({sl_source})")
            if ic.sltp_tiers:
                first = ic.sltp_tiers[0]
                last = ic.sltp_tiers[-1]
                step = settings.BOT_SLTP_STEP_PCT
                offset = settings.BOT_SLTP_LOCK_OFFSET_PCT
                logger.info(f"    SLTP: {len(ic.sltp_tiers)} tiers, {step:.2%} steps, {offset:.1%} trailing offset")
                logger.info(f"      First: {first[0]:.2%} activate @ ${first[2]:.2f}, floor @ ${first[3]:.2f}")
                logger.info(f"      Last:  {last[0]:.2%} activate @ ${last[2]:.2f}, floor @ ${last[3]:.2f}")

    # ===== Persistent Mode Helpers =====

    async def _wait_for_market_open(self):
        """Sleep until the next trading window. Returns immediately if already in window."""
        now = self._now_et()
        wake_time = self._parse_time(settings.BOT_WAKE_TIME)
        exit_time = self._parse_time(settings.BOT_EXIT_TIME)

        # If skip-wait, return immediately
        if self.skip_wait:
            return

        # Already within today's trading window?
        if now.weekday() < 5 and wake_time <= now.time() < exit_time:
            logger.info(f"Market is open ({now.strftime('%A %H:%M ET')}). Proceeding.")
            return

        # Compute next wake target
        target = self._next_wake_datetime(now)
        delta = target - now
        hours = delta.total_seconds() / 3600

        logger.info(f"Market closed. Sleeping until {target.strftime('%A %Y-%m-%d %H:%M ET')} "
                     f"({hours:.1f} hours)")
        send_notification("0DTE Bot — Sleeping", f"Next wake: {target.strftime('%A %H:%M ET')}")

        await self._sleep_until(target)
        logger.info(f"Woke up at {self._now_et().strftime('%A %H:%M ET')}")

    def _next_wake_datetime(self, now: datetime) -> datetime:
        """Compute the next wake datetime (next weekday at BOT_WAKE_TIME)."""
        wake_time = self._parse_time(settings.BOT_WAKE_TIME)
        exit_time = self._parse_time(settings.BOT_EXIT_TIME)

        # If before wake time on a weekday, wake today
        if now.weekday() < 5 and now.time() < wake_time:
            return now.replace(hour=wake_time.hour, minute=wake_time.minute, second=0, microsecond=0)

        # Otherwise, find next weekday
        target_date = now.date() + timedelta(days=1)
        while target_date.weekday() >= 5:  # skip Sat/Sun
            target_date += timedelta(days=1)

        return ET.localize(datetime.combine(target_date, wake_time))

    async def _sleep_until(self, target_dt: datetime):
        """Async sleep in 60s chunks until target datetime (Ctrl+C responsive)."""
        while True:
            now = self._now_et()
            if now >= target_dt:
                break
            remaining = (target_dt - now).total_seconds()
            await asyncio.sleep(min(remaining, 60))

    async def _sleep_past_exit(self):
        """Sleep until after BOT_EXIT_TIME to prevent same-day re-trigger."""
        if self.skip_wait:
            return

        now = self._now_et()
        exit_time = self._parse_time(settings.BOT_EXIT_TIME)
        exit_dt = now.replace(hour=exit_time.hour, minute=exit_time.minute, second=0, microsecond=0)

        # Add a small buffer (1 minute past exit)
        exit_dt += timedelta(minutes=1)

        if now < exit_dt:
            wait_secs = (exit_dt - now).total_seconds()
            logger.info(f"Sleeping until past exit time ({settings.BOT_EXIT_TIME} ET, {wait_secs:.0f}s)...")
            await self._sleep_until(exit_dt)

    async def _check_daily_circuit_breaker(self) -> Optional[str]:
        """
        Check if daily P&L has hit circuit breaker limits.

        Returns a reason string if breaker tripped, None if OK to continue.
        Fail-open: returns None if Schwab API fails.
        """
        if self.dry_run:
            return None

        try:
            daily_pnl = await self._fetch_daily_pnl(open_ics=[])
            if daily_pnl.sod_balance <= 0:
                return None  # can't compute percentage

            pnl_pct = daily_pnl.total_daily_pnl / daily_pnl.sod_balance

            if pnl_pct <= -settings.BOT_DAILY_LOSS_LIMIT_PCT:
                return (f"Daily loss {pnl_pct:.1%} >= -{settings.BOT_DAILY_LOSS_LIMIT_PCT:.0%} limit "
                        f"(-${abs(daily_pnl.total_daily_pnl):,.0f})")

            logger.info(f"  Circuit breaker check: {pnl_pct:+.2%} (limit: "
                        f"-{settings.BOT_DAILY_LOSS_LIMIT_PCT:.0%})")
            return None

        except Exception as e:
            logger.warning(f"  Circuit breaker check failed (fail-open): {e}")
            return None

    def _reset_for_new_day(self):
        """Reset all state between trading days."""
        logger.info("--- Resetting state for new trading day ---")
        self.already_traded_today = False
        self.entry_order_id = None
        self.tp_order_id = None
        self.strikes = None
        self.quantity = 0
        self.fill_credit = 0.0
        self.exit_reason = None
        self.portfolio_value = 0.0
        self._v3_cached_confidence = None
        self._v4_daily_data_cache = None
        self._chosen_delta = None
        self._trade_decision = None
        self._v8_cross_cache = None  # Reset cross-asset data for new day
        self._zdom_blocker_checked = False
        # Reset ZDOM bar accumulation for fresh day
        if self._zdom_predictor is not None:
            self._zdom_predictor.reset_bars()

    def _reset_for_new_trade(self):
        """Reset entry state between trades within the same day.

        Keeps day-level caches (v3, v4 daily data, portfolio value).
        """
        logger.info("--- Resetting state for potential re-entry ---")
        self.already_traded_today = False
        self.entry_order_id = None
        self.tp_order_id = None
        self.strikes = None
        self.quantity = 0
        self.fill_credit = 0.0
        self.exit_reason = None
        self._chosen_delta = None

    async def _refresh_portfolio_value(self):
        """Re-fetch portfolio value from Schwab after a trade closes."""
        try:
            account = await self.schwab_client.get_account()
            old_val = self.portfolio_value
            self.portfolio_value = self._extract_portfolio_value(account)
            delta = self.portfolio_value - old_val
            logger.info(f"  Portfolio refreshed: ${self.portfolio_value:,.2f} "
                        f"({'+' if delta >= 0 else ''}{delta:,.2f} from last)")
        except Exception as e:
            logger.warning(f"  Portfolio refresh failed: {e} (keeping ${self.portfolio_value:,.2f})")

    def _log_trade_result_for_ic(self, ic: TrackedIronCondor, trade_num: int):
        """Append per-IC trade result to JSONL log file."""
        # Track consecutive losses for ZDOM hard blockers
        if settings.BOT_ZDOM_ENABLED and self._zdom_predictor is not None:
            is_win = ic.exit_reason in ("take_profit", "sltp_lock_25%", "sltp_lock_30%",
                                        "sltp_lock_35%", "sltp_lock_40%", "sltp_lock_45%",
                                        "sltp_lock_50%") or (
                ic.exit_reason and ic.exit_reason.startswith("sltp_lock_"))
            self._zdom_predictor.record_trade_result(is_win)

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        result = {
            "date": str(date.today()),
            "timestamp": self._now_et().isoformat(),
            "trade_num": trade_num,
            "mode": "dry_run" if self.dry_run else "live",
            "symbol": settings.BOT_SYMBOL,
            "label": ic.label,
            "strikes": {
                "long_put": ic.strikes['long_put']['strike'],
                "short_put": ic.strikes['short_put']['strike'],
                "short_call": ic.strikes['short_call']['strike'],
                "long_call": ic.strikes['long_call']['strike'],
            },
            "quantity": ic.quantity,
            "credit": ic.fill_credit,
            "exit_reason": ic.exit_reason,
            "sltp_tier_reached": ic.sltp_active_tier_idx,
            "portfolio_value": self.portfolio_value,
            "chosen_delta": self._chosen_delta or settings.BOT_SHORT_DELTA,
            "zdom_tp_pct": ic.zdom_tp_pct,
            "entry_mode": "zdom_v1" if settings.BOT_ZDOM_ENABLED else "legacy",
        }

        log_path = log_dir / f"zero_dte_bot_{date.today()}.jsonl"
        with open(log_path, 'a') as f:
            f.write(json.dumps(result, default=str) + '\n')

        logger.info(f"  {ic.label} result logged: {ic.exit_reason or 'unknown'} → {log_path}")

    async def pre_flight(self) -> bool:
        """
        Merged pre-flight check for both entry and monitoring paths.

        Checks: weekday, ThetaData, broker connection + portfolio value, ML models.
        Does NOT gate on existing/missing positions (unified flow handles that).
        """
        logger.info("--- Pre-flight Check ---")

        # Kill stale bot instances and release DuckDB locks before anything else
        self._kill_stale_processes()

        now = self._now_et()
        broker_name = settings.BROKER.upper()

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

        # Check broker credentials configured
        if settings.BROKER.lower() == "tradier":
            if not settings.validate_tradier_credentials():
                logger.error("Tradier credentials not configured in .env")
                return False
        else:
            if not settings.validate_schwab_credentials():
                logger.error("Schwab credentials not configured in .env")
                return False
        logger.info(f"  {broker_name} credentials: configured - OK")

        # Check broker connection + extract live portfolio value
        try:
            account = await self.schwab_client.get_account()
            bp = (
                account.get('securitiesAccount', {})
                .get('currentBalances', {})
                .get('buyingPower', 0.0)
            )
            self.portfolio_value = self._extract_portfolio_value(account)
            logger.info(f"  {broker_name} connection: OK (buying power: ${bp:,.2f})")
            logger.info(f"  Portfolio value: ${self.portfolio_value:,.2f} (live from {broker_name})")

            if bp < 5000:
                logger.error(f"Insufficient buying power: ${bp:,.2f}")
                return False
        except Exception as e:
            err_str = str(e)
            if ('refresh_token' in err_str or 'unsupported_token_type' in err_str) \
                    and hasattr(self.schwab_client, 're_authenticate'):
                logger.warning(f"  {broker_name} token expired — launching re-authentication...")
                play_sound("/System/Library/Sounds/Sosumi.aiff")
                send_notification("0DTE Bot — Token Expired",
                                  f"{broker_name} refresh token expired. Browser opening for re-auth.")
                if self.schwab_client.re_authenticate():
                    # Retry after re-auth
                    try:
                        account = await self.schwab_client.get_account()
                        bp = (
                            account.get('securitiesAccount', {})
                            .get('currentBalances', {})
                            .get('buyingPower', 0.0)
                        )
                        self.portfolio_value = self._extract_portfolio_value(account)
                        logger.info(f"  {broker_name} connection: OK after re-auth (buying power: ${bp:,.2f})")
                        logger.info(f"  Portfolio value: ${self.portfolio_value:,.2f} (live from {broker_name})")
                    except Exception as e2:
                        logger.error(f"  {broker_name} connection failed after re-auth: {e2}")
                        return False
                else:
                    logger.error("  Re-authentication failed. Aborting.")
                    return False
            else:
                logger.error(f"  {broker_name} connection failed: {e}")
                return False

        # ZDOM V1 models (primary when enabled)
        if not self.skip_ml and settings.BOT_ZDOM_ENABLED:
            logger.info("  Loading ZDOM V1 models...")
            if not self.zdom_predictor.load_models():
                logger.error("  ZDOM V1 model loading failed")
                return False
            self.zdom_predictor.load_thresholds()
            if not self.zdom_predictor.init_features():
                logger.error("  ZDOM V1 feature initialization failed")
                return False

            # Staleness check
            stale_warnings = self.zdom_predictor.check_staleness()
            for w in stale_warnings:
                logger.warning(f"  ZDOM stale data: {w}")

            logger.info(f"  ZDOM V1: ready — scanning {settings.BOT_ZDOM_ENTRY_START}-"
                        f"{settings.BOT_ZDOM_ENTRY_END} ET every {settings.BOT_ZDOM_SCAN_INTERVAL}min")
            logger.info(f"  ZDOM skip rate: {settings.BOT_ZDOM_SKIP_RATE:.0%}")
            logger.info(f"  ZDOM hard blockers: {'ON' if settings.BOT_ZDOM_HARD_BLOCKERS else 'OFF'}")
            logger.info(f"  ZDOM SL: min(${settings.BOT_ZDOM_SL_CAP:.0f}, "
                        f"{settings.BOT_ZDOM_SL_MULT}x credit)")

        # Legacy ML models (fallback when ZDOM disabled)
        if not self.skip_ml and not settings.BOT_ZDOM_ENABLED:
            n_windows = self.predictor.V6_EXPECTED_MODELS

            # v6 ensemble is the sole go/no-go gate
            if self.predictor.v6_ready and settings.BOT_V6_ENABLED:
                tp25_models = len(self.predictor.v6_ensemble['tp25'])
                tp50_models = len(self.predictor.v6_ensemble['tp50'])
                logger.info(f"  ML v6 ensemble: loaded (TP25={tp25_models}/{n_windows}, TP50={tp50_models}/{n_windows}, "
                           f"{len(self.predictor.v6_feature_names)}f) - "
                           f"scanning {settings.BOT_V4_ENTRY_START}-{settings.BOT_V4_ENTRY_END} ET")
                logger.info(f"  v6 consensus filter: enabled (require ≥{settings.BOT_V6_MIN_CONSENSUS}/{n_windows} models >= 0.5)")
                logger.info(f"  v6 is SOLE entry gate (v3/v4/v5 shown as informational only)")
                # FOMC gate status
                from ml.models.predictor import get_economic_calendar_features
                cal = get_economic_calendar_features()
                fomc_str = "FOMC DAY" if cal["is_fomc_day"] else ("FOMC week" if cal["is_fomc_week"] else "no FOMC")
                logger.info(f"  FOMC gate: {'enabled' if settings.BOT_FOMC_GATE_ENABLED else 'disabled'} "
                           f"({fomc_str}, skip_day={'ON' if settings.BOT_FOMC_SKIP_DAY else 'OFF'})")
            elif not settings.BOT_V6_ENABLED:
                logger.info("  ML v6 ensemble: disabled (BOT_V6_ENABLED=False) — using fixed-time entry")
            elif not self.predictor.v6_ready:
                logger.info("  ML v6 ensemble: not loaded — using fixed-time entry")

            # v8 ensemble (highest priority)
            if self.predictor.v8_ensemble_ready and settings.BOT_V8_ENABLED:
                n_models = len(self.predictor._v8_ens_models)
                n_windows = len(set(k[2] for k in self.predictor._v8_ens_models))
                n_feat = len(self.predictor._v8_ens_feature_names) if self.predictor._v8_ens_feature_names else 0
                weights = self.predictor._v8_ens_weights
                logger.info(f"  ML v8 ensemble: loaded ({n_models} models, {n_windows} windows, {n_feat} features)")
                logger.info(f"    Windows: {list(weights.keys())}")
                logger.info(f"    Weights: {', '.join(f'{w}={v:.2f}' for w, v in weights.items())}")
            elif self.predictor.v8_ready and settings.BOT_V8_ENABLED:
                n_short = len(self.predictor._v8_short_models)
                n_long = len(self.predictor._v8_long_models)
                n_feat = len(self.predictor._v8_feature_names) if self.predictor._v8_feature_names else 0
                logger.info(f"  ML v8 single: loaded ({n_short} short + {n_long} long models, {n_feat} features)")
                logger.info(f"    Short deltas: {self.predictor.available_v8_short_deltas}")
                logger.info(f"    Long deltas: {self.predictor.available_v8_long_deltas}")
            elif settings.BOT_V8_ENABLED and not self.predictor.v8_ready:
                logger.info("  ML v8: not loaded — falling back to v5/v7")

            # Informational: v3/v5/v4 status (no longer gate entry)
            if self.predictor.is_ready():
                logger.info(f"  ML v3 model: loaded ({self.predictor.num_features} features) - informational only")
            if self.predictor.v5_ready:
                logger.info(f"  ML v5 model: loaded - informational only")
            if self.predictor.v4_ready:
                logger.info(f"  ML v4 model: loaded - informational only")

        if self.skip_ml:
            logger.info("  ML model: skipped (--skip-ml)")

        # Backfill candle database for ML features (requires ThetaData)
        if not self.skip_ml and not settings.BOT_ZDOM_ENABLED:
            self._backfill_candle_db()

        logger.info("--- Pre-flight Check PASSED ---")
        return True

    async def _entry_phase(self) -> Optional[List[TrackedIronCondor]]:
        """Run ML scan entry, wait for fill, return list of TrackedIronCondor or None.

        When ZDOM V1 enabled: S2 diversified strategy — enters multiple positions
        across different deltas simultaneously.
        Otherwise falls back to legacy v8/v6/v5/v4 chain (single position).
        """
        # ZDOM V1 entry (primary when enabled)
        if settings.BOT_ZDOM_ENABLED and not self.skip_ml and self.zdom_predictor.ready:
            logger.info("ZDOM V1 entry scan enabled (S2 diversified)")
            if not self.skip_wait:
                await self._wait_until(settings.BOT_ZDOM_ENTRY_START)
            return await self._zdom_entry_scan()

        # Legacy: unified strategy selection
        use_unified = (
            _OLD_ML_AVAILABLE
            and settings.BOT_UNIFIED_STRATEGY
            and not self.skip_ml
            and (self.predictor.v8_ensemble_ready or self.predictor.v8_ready or self.predictor.v5_ready or self.predictor.v7_ready)
        )

        if use_unified:
            logger.info("Unified strategy selection enabled — scanning all strategies")
            if not self.skip_wait:
                await self._wait_until(settings.BOT_V4_ENTRY_START)
            ic = await self._unified_entry_scan()
            return [ic] if ic else None

        # Fallback: v6 ensemble
        use_v6 = (
            _OLD_ML_AVAILABLE
            and settings.BOT_V6_ENABLED
            and not self.skip_ml
            and self.predictor.v6_ready
        )

        entered = False
        entry_filled = False

        if use_v6:
            logger.info("v6 entry timing enabled — scanning for optimal entry")
            if not self.skip_wait:
                await self._wait_until(settings.BOT_V4_ENTRY_START)
            entered = await self._entry_scan(use_v6=True)
            entry_filled = entered
        else:
            if not self.skip_ml:
                logger.info("No ML models available — using fixed-time entry")
            if not self.skip_wait:
                await self._wait_until(settings.BOT_ENTRY_TIME)
            entered = await self.attempt_entry()
            if entered:
                filled = await self.wait_for_entry_fill()
                if not filled:
                    logger.warning("Entry did not fill within timeout. Cancelling.")
                    await self._cancel_order_safe(self.entry_order_id)
                    return None
                entry_filled = True

        if not entry_filled or self.fill_credit <= 0:
            return None

        return [TrackedIronCondor(
            ic_id=0, label="IC-1", strikes=self.strikes,
            quantity=self.quantity, fill_credit=self.fill_credit,
            portfolio_value=self.portfolio_value,
        )]

    async def _zdom_entry_scan(self) -> Optional[List[TrackedIronCondor]]:
        """
        ZDOM V1 S2 diversified entry scan — enters multiple positions across deltas.

        Flow per scan:
        1. Check hard blockers (cached per day, VIX checked fresh)
        2. Fetch latest 1-min SPX bar → add to feature builder
        3. Fetch Schwab 0DTE chain
        4. Build ICs for all 9 strategies at target deltas
        5. Display scorecard, get diversified candidates (best TP per strategy)
        6. If shadow (5-delta best) → skip
        7. Enter all surviving candidates, each as a separate IC order
        """
        scan_start = self._parse_time(settings.BOT_ZDOM_ENTRY_START)
        scan_end = self._parse_time(settings.BOT_ZDOM_ENTRY_END)
        interval = settings.BOT_ZDOM_SCAN_INTERVAL
        skip_rate = settings.BOT_ZDOM_SKIP_RATE

        start_minutes = scan_start.hour * 60 + scan_start.minute
        end_minutes = scan_end.hour * 60 + scan_end.minute
        total_slots = (end_minutes - start_minutes) // interval
        scan_count = 0

        logger.info(f"--- ZDOM V1 S2 Diversified Entry Scan ---")
        logger.info(f"  Window: {settings.BOT_ZDOM_ENTRY_START} - {settings.BOT_ZDOM_ENTRY_END} ET")
        logger.info(f"  Interval: {interval} min | Skip rate: {skip_rate:.0%}")
        logger.info(f"  Models: {len(self.zdom_predictor.models)} TP levels")
        qty_label = (f"{settings.BOT_ZDOM_QTY_PER_POSITION} per position"
                     if settings.BOT_ZDOM_QTY_PER_POSITION > 0 else "auto-size")
        max_label = (f", max {settings.BOT_ZDOM_MAX_POSITIONS} positions"
                     if settings.BOT_ZDOM_MAX_POSITIONS > 0 else "")
        logger.info(f"  Strategy: S2 diversified | Qty: {qty_label}{max_label}")

        while True:
            now = self._now_et()
            if now.time() >= scan_end and not self.skip_wait:
                logger.info(f"  ZDOM scan window closed at {settings.BOT_ZDOM_ENTRY_END} ET.")
                return None

            scan_count += 1
            slot_time = now.strftime('%H:%M')
            progress = f"[{scan_count}/{total_slots}]" if total_slots > 0 else f"[{scan_count}]"

            # 1. Hard blockers (calendar cached, VIX/gap checked fresh)
            if settings.BOT_ZDOM_HARD_BLOCKERS and not self._zdom_blocker_checked:
                vix_close = None
                gap_pct = None
                if self.zdom_predictor.feature_builder and self.zdom_predictor.feature_builder.daily_features:
                    vix_close = self.zdom_predictor.feature_builder.daily_features.get("vix_close")
                    gap_pct = self.zdom_predictor.feature_builder.daily_features.get("gap_pct")

                blocked, reason = self.zdom_predictor.check_hard_blockers(
                    vix_close=vix_close,
                    gap_pct=gap_pct,
                    kill_switch=settings.BOT_ZDOM_KILL_SWITCH,
                )
                if blocked:
                    logger.warning(f"  ZDOM HARD BLOCKER: {reason}")
                    play_sound(SL_SOUND)
                    send_notification("ZDOM — Blocked", reason)
                    return None
                self._zdom_blocker_checked = True
                logger.info(f"  Hard blockers: CLEAR")

            # 2. Fetch latest 1-min SPX + VIX bars
            spx_bar = self._fetch_spx_latest_bar()
            if spx_bar:
                self.zdom_predictor.add_bar(spx_bar)
            else:
                logger.debug(f"  {progress} {slot_time} — no SPX bar available")
            vix_bar = self._fetch_vix_latest_bar()
            if vix_bar:
                self.zdom_predictor.add_vix_bar(vix_bar)

            # 3. Fetch 0DTE chain
            chain = await self._fetch_0dte_chain()
            if not chain:
                logger.warning(f"  {progress} {slot_time} — chain unavailable")
                if self.skip_wait:
                    return None
                await asyncio.sleep(interval * 60)
                continue

            # 4. Build ICs for all 9 strategies (include 5d for scorecard even if credit < min)
            chain_ics = []
            for strat, delta in self.zdom_predictor.DELTA_MAP.items():
                try:
                    strikes = self.builder.select_strikes(
                        chain,
                        target_delta=delta,
                        wing_width=settings.BOT_WING_WIDTH,
                    )
                    if not strikes:
                        continue
                    credit = strikes.get('net_credit', 0)
                    # Include for scoring/display; min credit filter applied later for trading
                    chain_ics.append({
                        "strategy": strat,
                        "credit": credit,
                        "delta": delta,
                        "ic_data": strikes,
                        "below_min_credit": credit < settings.BOT_MIN_CREDIT,
                    })
                except Exception as e:
                    logger.debug(f"  Could not build IC for {strat}: {e}")

            if not chain_ics:
                logger.info(f"  {progress} {slot_time} — no valid ICs")
                if self.skip_wait:
                    return None
                await asyncio.sleep(interval * 60)
                continue

            # 5. Score all strategies + display scorecard
            scorecard_lines, _ = self.zdom_predictor.build_scorecard(chain_ics, skip_rate)
            logger.info(f"  {progress} {slot_time} ZDOM Scorecard:")
            for line in scorecard_lines:
                logger.info(line)

            # 6. Get diversified candidates (best TP per strategy, shadows filtered)
            candidates = self.zdom_predictor.get_diversified_candidates(chain_ics, skip_rate)

            if not candidates:
                # Check if it was a shadow situation
                best = self.zdom_predictor.get_best_candidate(chain_ics, skip_rate)
                if best and best.get("is_shadow"):
                    logger.info(
                        f"  >>> SHADOW: {best['strategy']} x {best['tp_level']} "
                        f"EV=${best['ev']:.2f} P={best['prob']:.3f} — skipping all"
                    )
                if self.skip_wait:
                    return None
                await asyncio.sleep(interval * 60)
                continue

            # 7. S2 Diversified Entry — enter all candidates
            logger.info(f"  >>> S2 DIVERSIFIED: {len(candidates)} positions to enter")
            for c in candidates:
                logger.info(
                    f"    {c['strategy']} x {c['tp_level']} | "
                    f"credit=${c['credit']:.2f} | EV=${c['ev']:.2f} | P={c['prob']:.3f}"
                )

            # Cap number of positions
            max_pos = settings.BOT_ZDOM_MAX_POSITIONS
            if max_pos > 0:
                candidates = candidates[:max_pos]

            # Determine per-position quantity
            fixed_qty = settings.BOT_ZDOM_QTY_PER_POSITION
            if fixed_qty > 0:
                # Fixed qty per position (e.g. --qty 1)
                logger.info(f"  Qty: {fixed_qty} per position (fixed)")
            else:
                # Auto-size: distribute buying power evenly
                buying_power = await self.schwab_client.get_buying_power()
                total_bp_contracts = int(buying_power / (settings.BOT_WING_WIDTH * 100))
                if settings.BOT_MAX_CONTRACTS > 0:
                    total_bp_contracts = min(total_bp_contracts, settings.BOT_MAX_CONTRACTS)
                fixed_qty = 0  # computed per position below

            k = len(candidates)

            filled_ics = []
            for i, cand in enumerate(candidates):
                if settings.BOT_ZDOM_QTY_PER_POSITION > 0:
                    qty = settings.BOT_ZDOM_QTY_PER_POSITION
                else:
                    base_qty = total_bp_contracts // k
                    extra = total_bp_contracts % k
                    qty = base_qty + (1 if i < extra else 0)
                if qty <= 0:
                    continue

                delta = cand["delta"]
                tp_pct = cand["tp_pct"]
                ic_label = f"IC-{i+1}"

                logger.info(f"  Placing {ic_label}: {cand['strategy']} x{qty} at {int(delta*100)}δ")

                entered = await self.attempt_entry(chain=chain, target_delta=delta, target_qty=qty)
                if not entered:
                    logger.warning(f"  {ic_label} entry failed — skipping")
                    self._reset_entry_state()
                    continue

                filled = await self.wait_for_entry_fill()
                if not filled:
                    logger.warning(f"  {ic_label} did not fill — cancelling")
                    await self._cancel_order_safe(self.entry_order_id)
                    self._reset_entry_state()
                    continue

                filled_ics.append(TrackedIronCondor(
                    ic_id=i, label=ic_label,
                    strikes=self.strikes,
                    quantity=self.quantity,
                    fill_credit=self.fill_credit,
                    portfolio_value=self.portfolio_value,
                    zdom_tp_pct=tp_pct,
                    zdom_strategy=cand["strategy"],
                ))
                logger.info(
                    f"  {ic_label} FILLED: {cand['strategy']} x{self.quantity} "
                    f"credit=${self.fill_credit:.2f} TP={int(tp_pct*100)}%"
                )

            if filled_ics:
                return filled_ics

            logger.warning("  No S2 positions filled.")
            self._reset_entry_state()
            if self.skip_wait:
                return None
            await asyncio.sleep(interval * 60)

    def _fetch_spx_latest_bar(self) -> Optional[dict]:
        """Fetch the latest 1-min SPX bar from ThetaData for ZDOM feature builder."""
        try:
            now = self._now_et()
            url = "http://127.0.0.1:25503/v3/index/history/ohlc"
            # Get last 2 minutes to ensure we have at least 1 complete bar
            start_time = (now - timedelta(minutes=2)).strftime("%H:%M:%S")
            end_time = now.strftime("%H:%M:%S")
            params = {
                "symbol": "SPX",
                "start_date": date.today().strftime("%Y%m%d"),
                "end_date": date.today().strftime("%Y%m%d"),
                "interval": "1m",
                "start_time": start_time,
                "end_time": end_time,
                "format": "json",
            }
            query = "&".join(f"{k}={v}" for k, v in params.items())
            full_url = f"{url}?{query}"

            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            response = data.get("response", data if isinstance(data, list) else [])
            if response and isinstance(response[0], dict) and "data" in response[0]:
                bars = []
                for entry in response:
                    for dp in entry.get("data", []):
                        bars.append(dp)
                response = bars

            if not response:
                return None

            # Take the last bar
            bar_data = response[-1]
            return {
                "datetime": now,
                "open": float(bar_data.get("open", bar_data.get("Open", 0))),
                "high": float(bar_data.get("high", bar_data.get("High", 0))),
                "low": float(bar_data.get("low", bar_data.get("Low", 0))),
                "close": float(bar_data.get("close", bar_data.get("Close", 0))),
            }
        except Exception as e:
            logger.debug(f"  Failed to fetch SPX 1-min bar: {e}")
            return None

    def _fetch_vix_latest_bar(self) -> Optional[dict]:
        """Fetch the latest 1-min VIX bar from ThetaData for ZDOM feature builder."""
        try:
            now = self._now_et()
            url = "http://127.0.0.1:25503/v3/index/history/ohlc"
            start_time = (now - timedelta(minutes=2)).strftime("%H:%M:%S")
            end_time = now.strftime("%H:%M:%S")
            params = {
                "symbol": "VIX",
                "start_date": date.today().strftime("%Y%m%d"),
                "end_date": date.today().strftime("%Y%m%d"),
                "interval": "1m",
                "start_time": start_time,
                "end_time": end_time,
                "format": "json",
            }
            query = "&".join(f"{k}={v}" for k, v in params.items())
            full_url = f"{url}?{query}"

            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            response = data.get("response", data if isinstance(data, list) else [])
            if response and isinstance(response[0], dict) and "data" in response[0]:
                bars = []
                for entry in response:
                    for dp in entry.get("data", []):
                        bars.append(dp)
                response = bars

            if not response:
                return None

            bar_data = response[-1]
            return {
                "datetime": now,
                "open": float(bar_data.get("open", bar_data.get("Open", 0))),
                "high": float(bar_data.get("high", bar_data.get("High", 0))),
                "low": float(bar_data.get("low", bar_data.get("Low", 0))),
                "close": float(bar_data.get("close", bar_data.get("Close", 0))),
            }
        except Exception as e:
            logger.debug(f"  Failed to fetch VIX 1-min bar: {e}")
            return None

    async def _unified_entry_scan(self) -> Optional[TrackedIronCondor]:
        """
        Unified EV-based entry scan — scores all 6 strategies every scan interval.

        For each scan slot:
        1. Fetch chain + 5-min candles
        2. Score v3, v5 (per-delta), v7 (per-delta)
        3. Run TradeDecisionEngine.decide() → picks best strategy by EV
        4. Display full strategy scorecard
        5. If SHORT_ONLY or BOTH → attempt short IC entry at recommended delta
        6. If LONG_ONLY → store decision for _run_trading_day to handle
        7. If NEITHER → wait for next scan slot

        Returns TrackedIronCondor for short entry, or None.
        Sets self._trade_decision so _run_trading_day can check for long IC.
        """
        import pandas as pd

        scan_start = self._parse_time(settings.BOT_V4_ENTRY_START)
        scan_end = self._parse_time(settings.BOT_V4_ENTRY_END)
        interval = settings.BOT_V4_SCAN_INTERVAL

        use_v8_ens = settings.BOT_V8_ENABLED and self.predictor.v8_ensemble_ready
        use_v8 = settings.BOT_V8_ENABLED and self.predictor.v8_ready and not use_v8_ens

        logger.info(f"--- Unified Strategy Scan ---")
        logger.info(f"  Window: {settings.BOT_V4_ENTRY_START} - {settings.BOT_V4_ENTRY_END} ET")
        logger.info(f"  Interval: {interval} min | Min EV: ${settings.BOT_MIN_EV_THRESHOLD:.2f}")
        if use_v8_ens:
            n_windows = len(set(k[2] for k in self.predictor._v8_ens_models))
            logger.info(f"  Models: v8 ensemble ({n_windows} windows, "
                         f"{len(self.predictor._v8_ens_models)} models)")
        elif use_v8:
            logger.info(f"  Models: v8 single "
                         f"(short={len(self.predictor._v8_short_models)} long={len(self.predictor._v8_long_models)})")
        else:
            logger.info(f"  Models: v5={'YES' if self.predictor.v5_ready else 'no'} "
                         f"v7={'YES' if self.predictor.v7_ready else 'no'}")

        start_minutes = scan_start.hour * 60 + scan_start.minute
        end_minutes = scan_end.hour * 60 + scan_end.minute
        total_slots = (end_minutes - start_minutes) // interval
        scan_count = 0

        while True:
            now = self._now_et()
            if now.time() >= scan_end and not self.skip_wait:
                logger.info(f"  Scan window closed at {settings.BOT_V4_ENTRY_END} ET. No entry today.")
                return None

            scan_count += 1
            slot_time = now.strftime('%H:%M')
            progress = f"[{scan_count}/{total_slots}]" if total_slots > 0 else f"[{scan_count}]"

            # Fetch chain
            chain = await self._fetch_0dte_chain()
            if not chain:
                logger.warning(f"  {progress} {slot_time} — chain unavailable, skipping slot")
                if self.skip_wait:
                    return None
                await asyncio.sleep(interval * 60)
                continue

            # Run v3 pre-screen (cached for the day)
            if self._v3_cached_confidence is None and not self.skip_ml and self.predictor.is_ready():
                v3_conf = await self._run_ml_prediction(chain)
                self._v3_cached_confidence = v3_conf

            v3_conf = self._v3_cached_confidence or 0.5

            # Fetch 5-min candles
            candles = self._fetch_spx_5min_candles()
            if candles is None or len(candles) < 6:
                logger.info(f"  {progress} {slot_time} — insufficient candles, waiting...")
                if self.skip_wait:
                    return None
                await asyncio.sleep(interval * 60)
                continue

            atm_iv = self._extract_atm_iv(chain)

            # Score all strategies
            try:
                # Use v8 ensemble > v8 single > v5/v7 fallback
                if use_v8_ens or use_v8:
                    v8_features = self._compute_v8_features(chain)
                    if v8_features and use_v8_ens:
                        all_scores = self.predictor.predict_v8_ensemble_all_strategies(
                            v8_features, v3_confidence=v3_conf,
                        )
                        n_windows = len(set(k[2] for k in self.predictor._v8_ens_models))
                        logger.info(f"    [v8 ensemble] {len(v8_features)} features, "
                                    f"{n_windows} windows")
                    elif v8_features and use_v8:
                        all_scores = self.predictor.predict_v8_all_strategies(
                            v8_features, v3_confidence=v3_conf,
                        )
                        logger.info(f"    [v8 single] {len(v8_features)} features → "
                                    f"short deltas={self.predictor.available_v8_short_deltas} "
                                    f"long deltas={self.predictor.available_v8_long_deltas}")
                    else:
                        logger.info("    v8 features unavailable, falling back to v5/v7")
                        all_scores = self.predictor.predict_all_strategies(
                            candles_5min=candles,
                            daily_data=self._v4_daily_data_cache,
                            option_atm_iv=atm_iv,
                            v3_confidence=v3_conf,
                        )
                else:
                    all_scores = self.predictor.predict_all_strategies(
                        candles_5min=candles,
                        daily_data=self._v4_daily_data_cache,
                        option_atm_iv=atm_iv,
                        v3_confidence=v3_conf,
                    )

                # Cache daily data
                if self._v4_daily_data_cache is None and self.predictor._daily_cache:
                    self._v4_daily_data_cache = self.predictor._daily_cache

                # Run decision engine
                decision = self._decision_engine.decide(
                    v3_confidence=all_scores["v3_confidence"],
                    v5_scores=all_scores.get("v5_scores"),
                    v7_scores=all_scores.get("v7_scores"),
                )
                self._trade_decision = decision

                # Display scorecard
                logger.info(f"  {progress} {slot_time} | Decision: {decision.action.value}")
                scorecard = self._decision_engine.scorer.display_scorecard(
                    decision.all_candidates,
                    self._decision_engine.min_ev,
                )
                for line in scorecard.split("\n"):
                    logger.info(f"    {line}")
                logger.info(f"  {decision.summary()}")
                for r in decision.reasons:
                    logger.info(f"    → {r}")

                # Act on decision
                if decision.has_short:
                    short_delta = decision.short_delta or settings.BOT_SHORT_DELTA
                    self._chosen_delta = short_delta
                    logger.info(f"    >>> SHORT IC ENTRY at {int(short_delta*100)}δ "
                                f"(EV=${decision.short_ev:+.2f})")
                    entered = await self.attempt_entry(chain=chain, target_delta=short_delta)
                    if entered:
                        filled = await self.wait_for_entry_fill()
                        if filled:
                            return TrackedIronCondor(
                                ic_id=0, label="IC-1", strikes=self.strikes,
                                quantity=self.quantity, fill_credit=self.fill_credit,
                                portfolio_value=self.portfolio_value,
                            )
                        logger.warning("    Entry did not fill. Cancelling and re-scanning.")
                        await self._cancel_order_safe(self.entry_order_id)
                        self._reset_entry_state()

                elif decision.action == TradeAction.LONG_ONLY:
                    logger.info(f"    >>> LONG IC recommended at {int(decision.long_delta*100)}δ "
                                f"(EV=${decision.long_ev:+.2f})")
                    logger.info(f"    Short IC blocked — returning None for _run_trading_day to handle long IC")
                    return None

                elif decision.action == TradeAction.NEITHER:
                    logger.info(f"    Standing aside — no positive EV strategies")

            except Exception as e:
                logger.warning(f"  {progress} {slot_time} — strategy scoring error: {e}")

            if self.skip_wait:
                return None
            await asyncio.sleep(interval * 60)

    async def _place_ceiling_tp(self, ic: TrackedIronCondor):
        """Place TP limit order for an IC (uses ZDOM TP% or default 25%)."""
        if ic.zdom_tp_pct is not None:
            # ZDOM: use the model-chosen TP percentage
            fee = (settings.BOT_COST_PER_LEG * 8) / 100
            tp_debit = round(ic.fill_credit * (1 - ic.zdom_tp_pct) - fee, 2)
            tp_label = f"{ic.zdom_tp_pct:.0%}"
        else:
            tp_debit = ic.tp_25_debit
            tp_label = "25%"
        tp_debit = round(math.ceil(tp_debit * 20) / 20, 2)
        tp_debit = max(tp_debit, 0.05)
        ic.target_tp_debit = tp_debit

        logger.info(
            f"  {ic.label}: {tp_label} TP ${tp_debit:.2f} x{ic.quantity} "
            f"(SLTP activates at 30% profit)"
        )
        await self._place_tp_limit_order(ic, tp_debit)

    async def _sync_tp_for_ic(self, ic: TrackedIronCondor):
        """
        Ensure IC has a TP order with correct quantity.

        Detects existing TP, compares quantity, cancels+replaces if mismatched.
        """
        tp_id, tp_qty = await self._detect_tp_order_for_ic(ic)

        if tp_id and tp_qty == ic.quantity:
            # TP exists with correct quantity
            ic.tp_order_id = tp_id
            logger.info(f"  {ic.label}: existing TP order: ID={tp_id} x{tp_qty}")
        elif tp_id and tp_qty != ic.quantity:
            # TP exists but wrong quantity — cancel and replace
            logger.warning(
                f"  {ic.label}: TP quantity mismatch (TP={tp_qty} vs position={ic.quantity}) "
                f"— cancelling TP {tp_id} and placing new"
            )
            await self._cancel_order_safe(tp_id)
            ic.tp_order_id = None
            await self._place_ceiling_tp(ic)
        else:
            # No TP found — place new
            await self._place_ceiling_tp(ic)

    async def _reconstruct_and_assign(self, positions=None) -> Optional[List[TrackedIronCondor]]:
        """
        Reconstruct ICs and assign credits.

        Primary: order-based reconstruction (handles netted positions correctly).
        Fallback: position-based reconstruction + credit detection.
        """
        # Primary: reconstruct from order history (avoids netting ambiguity)
        logger.info("  Attempting order-based IC reconstruction...")
        try:
            order_ics = await self.reconstruct_ics_from_orders()
            if order_ics:
                logger.info(f"  Order-based reconstruction: {len(order_ics)} IC(s)")
                return order_ics
            logger.info("  Order-based reconstruction returned None")
        except Exception as e:
            logger.warning(f"  Order-based reconstruction error: {e}", exc_info=True)

        # Fallback: position-based reconstruction
        logger.info("  Falling back to position-based reconstruction...")
        if positions is None:
            positions = await self._get_spx_positions()
        if not positions:
            return None
        tracked_ics = await self.reconstruct_multiple_ics_from_positions(positions)
        if not tracked_ics:
            return None
        if not await self._assign_credits(tracked_ics):
            logger.error("Credit detection failed for reconstructed ICs.")
            return None
        return tracked_ics

    # ===== Unified Monitoring Loop =====

    def _get_tracked_symbols(self, ics: List[TrackedIronCondor]) -> frozenset:
        """Get set of all option symbols across tracked ICs."""
        symbols = set()
        for ic in ics:
            for leg in ['long_put', 'short_put', 'short_call', 'long_call']:
                symbols.add(ic.strikes[leg]['symbol'])
        return frozenset(symbols)

    def _get_ic_identity_keys(self, ics: List[TrackedIronCondor]) -> frozenset:
        """Get a stable identity key per IC (strikes + quantity) for change detection."""
        keys = set()
        for ic in ics:
            key = (
                ic.strikes['long_put']['symbol'],
                ic.strikes['short_put']['symbol'],
                ic.strikes['short_call']['symbol'],
                ic.strikes['long_call']['symbol'],
                ic.quantity,
            )
            keys.add(key)
        return frozenset(keys)

    async def _get_current_position_symbols(self) -> frozenset:
        """Get set of current SPX option symbols from Schwab positions."""
        positions = await self._get_spx_positions()
        return frozenset(p.get('instrument', {}).get('symbol', '') for p in positions)

    def _merge_sltp_state(self, old_ics: List[TrackedIronCondor],
                          new_ics: List[TrackedIronCondor]):
        """Carry forward SLTP state for ICs that haven't changed."""
        old_by_symbols = {}
        for ic in old_ics:
            key = self._get_tracked_symbols([ic])
            old_by_symbols[key] = ic
        for new_ic in new_ics:
            key = self._get_tracked_symbols([new_ic])
            if key in old_by_symbols:
                old = old_by_symbols[key]
                new_ic.sltp_active_tier_idx = old.sltp_active_tier_idx
                new_ic.sltp_floor_debit = old.sltp_floor_debit
                new_ic.alerts_fired = old.alerts_fired
                new_ic.tp_order_id = old.tp_order_id

    async def _monitoring_loop(self, tracked_ics: List[TrackedIronCondor]):
        """Unified monitoring loop for 1 or N ICs with position change detection.

        When ZDOM is enabled and capacity remains (open ICs < max_positions),
        scans for new entry opportunities every BOT_ZDOM_SCAN_INTERVAL minutes.
        """
        exit_time = self._parse_time(settings.BOT_EXIT_TIME)

        # Track IC identity keys for change detection (avoids symbol netting issues)
        last_ic_keys = self._get_ic_identity_keys(tracked_ics)
        # Snapshot position symbols for lightweight change detection trigger
        last_pos_symbols = await self._get_current_position_symbols()

        # Print throttle: check every BOT_MONITOR_INTERVAL (1s) but print every BOT_MONITOR_PRINT_INTERVAL (30s)
        import time as _time
        last_print_time = 0.0

        # Continuous entry scan timer (ZDOM S2: check for new positions while monitoring)
        last_scan_time = 0.0
        scan_interval_secs = settings.BOT_ZDOM_SCAN_INTERVAL * 60
        max_positions = settings.BOT_ZDOM_MAX_POSITIONS if settings.BOT_ZDOM_MAX_POSITIONS > 0 else 999
        scan_end_time = self._parse_time(settings.BOT_ZDOM_ENTRY_END) if settings.BOT_ZDOM_ENABLED else None

        logger.info("--- Monitoring Loop Started ---")
        logger.info(f"  Monitoring {len(tracked_ics)} IC(s)")
        logger.info(f"  Exit time: {settings.BOT_EXIT_TIME} ET")
        logger.info(f"  Check interval: {settings.BOT_MONITOR_INTERVAL}s "
                     f"(print every {settings.BOT_MONITOR_PRINT_INTERVAL}s)")
        if settings.BOT_ZDOM_ENABLED:
            logger.info(f"  Continuous scan: every {settings.BOT_ZDOM_SCAN_INTERVAL}min "
                         f"until {settings.BOT_ZDOM_ENTRY_END} ET (max {max_positions} positions)")

        while True:
            try:
                # Lightweight check: did position symbols change?
                current_pos_symbols = await self._get_current_position_symbols()
                if not current_pos_symbols:
                    # Double-check: API may briefly return empty during settlement
                    await asyncio.sleep(3)
                    recheck = await self._get_current_position_symbols()
                    if not recheck:
                        logger.info("No positions remaining (confirmed). Exiting monitoring.")
                        return
                    else:
                        logger.warning("Positions reappeared on recheck — continuing monitoring")
                        last_pos_symbols = recheck
                        continue

                if current_pos_symbols != last_pos_symbols:
                    # Position symbols changed — re-reconstruct from orders
                    # (handles netted positions correctly)
                    logger.info("Position change detected — re-reconstructing ICs from orders")
                    last_pos_symbols = current_pos_symbols
                    new_ics = await self._reconstruct_and_assign()
                    if new_ics:
                        new_keys = self._get_ic_identity_keys(new_ics)
                        if new_keys != last_ic_keys:
                            logger.info(f"IC structure changed — {len(tracked_ics)} -> {len(new_ics)} IC(s)")
                            self._merge_sltp_state(tracked_ics, new_ics)
                            tracked_ics = new_ics
                            last_ic_keys = new_keys
                            for ic in tracked_ics:
                                await self._sync_tp_for_ic(ic)
                            continue
                    else:
                        logger.warning("Re-reconstruction failed. Continuing with existing state.")

                # Filter to open ICs
                open_ics = [ic for ic in tracked_ics if not ic.is_closed]
                if not open_ics:
                    logger.info("=== All ICs closed. Monitoring complete. ===")
                    return

                now = self._now_et()

                # Time exit check
                if now.time() >= exit_time and not self.skip_wait:
                    logger.info(f"=== TIME EXIT: Past {settings.BOT_EXIT_TIME} ET ===")
                    for ic in open_ics:
                        await self._close_ic(ic, "time_exit")
                    return

                # Fetch chain once for all ICs
                chain = await self._fetch_0dte_chain()
                if not chain:
                    logger.warning("  Could not fetch chain this cycle, retrying...")
                    await asyncio.sleep(settings.BOT_MONITOR_INTERVAL)
                    continue

                # Track per-IC unrealized P&L for cumulative summary
                ic_unrealized = {}  # ic.ic_id -> unrealized $ amount

                # Throttle status prints (check every 1s, print every 30s)
                should_print = (_time.time() - last_print_time) >= settings.BOT_MONITOR_PRINT_INTERVAL

                for ic in open_ics:
                    # Check if TP limit order filled
                    if ic.tp_order_id:
                        if await self._check_order_filled(ic.tp_order_id):
                            ic.exit_reason = "take_profit"
                            ic.is_closed = True
                            logger.info(f"  === {ic.label}: TP limit order FILLED ===")
                            continue

                    await self._monitor_short_ic(ic, chain, now, ic_unrealized,
                                                  should_print=should_print)

                if should_print:
                    last_print_time = _time.time()

                # --- Cumulative multi-IC summary and combined SL check ---
                still_open = [ic for ic in open_ics if not ic.is_closed]
                if len(still_open) >= 2 and ic_unrealized:
                    total_unrealized = sum(
                        ic_unrealized[ic.ic_id] for ic in still_open
                        if ic.ic_id in ic_unrealized
                    )
                    combined_sl_limit = self.portfolio_value * settings.BOT_STOP_LOSS_PCT
                    if should_print:
                        sign = "+" if total_unrealized >= 0 else ""
                        logger.info(
                            f"  [{now.strftime('%H:%M:%S')}] TOTAL: {len(still_open)} ICs  |  "
                            f"net unrealized {sign}${total_unrealized:,.0f}  |  "
                            f"combined SL at -${combined_sl_limit:,.0f}"
                        )

                    # Combined SL: close ALL open ICs if total unrealized loss exceeds portfolio SL
                    if total_unrealized <= -combined_sl_limit:
                        logger.warning(
                            f"  === COMBINED SL TRIGGERED: "
                            f"${total_unrealized:,.0f} <= -${combined_sl_limit:,.0f} "
                            f"({settings.BOT_STOP_LOSS_PCT:.0%} of ${self.portfolio_value:,.0f}) ==="
                        )
                        play_sound(SL_SOUND)
                        send_notification(
                            "COMBINED STOP LOSS",
                            f"Total loss ${total_unrealized:,.0f} — closing all {len(still_open)} ICs"
                        )
                        for ic in still_open:
                            if not ic.is_closed:
                                await self._close_ic(ic, "combined_stop_loss")

                # --- Continuous ZDOM entry scan during monitoring ---
                if (settings.BOT_ZDOM_ENABLED
                        and self._zdom_predictor is not None
                        and scan_end_time is not None
                        and now.time() < scan_end_time):

                    still_open_now = [ic for ic in tracked_ics if not ic.is_closed]
                    capacity = max_positions - len(still_open_now)
                    time_for_scan = (_time.time() - last_scan_time) >= scan_interval_secs

                    if capacity > 0 and time_for_scan:
                        last_scan_time = _time.time()
                        new_ics = await self._zdom_monitor_scan(
                            chain, tracked_ics, capacity, should_print=True,
                        )
                        if new_ics:
                            for new_ic in new_ics:
                                tracked_ics.append(new_ic)
                                await self._sync_tp_for_ic(new_ic)
                            # Update position tracking keys
                            last_ic_keys = self._get_ic_identity_keys(tracked_ics)
                            last_pos_symbols = await self._get_current_position_symbols()
                            logger.info(
                                f"  Now monitoring {len([ic for ic in tracked_ics if not ic.is_closed])} IC(s)"
                            )

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)

            if self.skip_wait:
                logger.info("[SKIP-WAIT] Exiting monitoring loop after one check.")
                break

            await asyncio.sleep(settings.BOT_MONITOR_INTERVAL)

    async def _zdom_monitor_scan(
        self,
        chain: dict,
        tracked_ics: List[TrackedIronCondor],
        capacity: int,
        should_print: bool = True,
    ) -> List[TrackedIronCondor]:
        """
        Single ZDOM scan cycle during monitoring — enters new positions for
        strategies not already active, up to `capacity` new positions.

        Args:
            chain: Already-fetched 0DTE option chain.
            tracked_ics: Current list of tracked ICs (for strategy dedup).
            capacity: Max new positions to open this scan.
            should_print: Whether to log the scorecard.

        Returns:
            List of newly filled TrackedIronCondor objects (may be empty).
        """
        skip_rate = settings.BOT_ZDOM_SKIP_RATE
        now = self._now_et()
        slot_time = now.strftime('%H:%M')

        # Strategies already active (not closed)
        active_strategies = {
            ic.zdom_strategy for ic in tracked_ics
            if not ic.is_closed and ic.zdom_strategy
        }

        # Fetch latest SPX + VIX bars for features
        spx_bar = self._fetch_spx_latest_bar()
        if spx_bar:
            self.zdom_predictor.add_bar(spx_bar)
        vix_bar = self._fetch_vix_latest_bar()
        if vix_bar:
            self.zdom_predictor.add_vix_bar(vix_bar)

        # Build ICs for all 9 strategies
        chain_ics = []
        for strat, delta in self.zdom_predictor.DELTA_MAP.items():
            try:
                strikes = self.builder.select_strikes(
                    chain,
                    target_delta=delta,
                    wing_width=settings.BOT_WING_WIDTH,
                )
                if not strikes:
                    continue
                credit = strikes.get('net_credit', 0)
                chain_ics.append({
                    "strategy": strat,
                    "credit": credit,
                    "delta": delta,
                    "ic_data": strikes,
                    "below_min_credit": credit < settings.BOT_MIN_CREDIT,
                })
            except Exception as e:
                logger.debug(f"  Could not build IC for {strat}: {e}")

        if not chain_ics:
            if should_print:
                logger.debug(f"  [{slot_time}] Monitor scan: no valid ICs")
            return []

        # Score + display scorecard
        scorecard_lines, _ = self.zdom_predictor.build_scorecard(chain_ics, skip_rate)
        if should_print:
            logger.info(f"  [{slot_time}] ZDOM Monitor Scorecard "
                         f"({len(active_strategies)} strategies active, "
                         f"{capacity} slot(s) available):")
            for line in scorecard_lines:
                logger.info(line)

        # Get diversified candidates, exclude already-active strategies
        candidates = self.zdom_predictor.get_diversified_candidates(chain_ics, skip_rate)

        if not candidates:
            if should_print:
                # Check for shadow
                best = self.zdom_predictor.get_best_candidate(chain_ics, skip_rate)
                if best and best.get("is_shadow"):
                    logger.info(
                        f"  >>> SHADOW: {best['strategy']} x {best['tp_level']} "
                        f"EV=${best['ev']:.2f} P={best['prob']:.3f} — skipping"
                    )
                else:
                    logger.info(f"  [{slot_time}] No candidates above threshold")
            return []

        # Filter out strategies already held
        new_candidates = [c for c in candidates if c["strategy"] not in active_strategies]
        if not new_candidates:
            if should_print:
                logger.info(f"  [{slot_time}] All passing strategies already active — no new entries")
            return []

        # Cap to available capacity
        new_candidates = new_candidates[:capacity]

        logger.info(f"  >>> MONITOR SCAN: {len(new_candidates)} new position(s) to enter")
        for c in new_candidates:
            logger.info(
                f"    {c['strategy']} x {c['tp_level']} | "
                f"credit=${c['credit']:.2f} | EV=${c['ev']:.2f} | P={c['prob']:.3f}"
            )

        # Determine per-position quantity
        qty = settings.BOT_ZDOM_QTY_PER_POSITION if settings.BOT_ZDOM_QTY_PER_POSITION > 0 else 1

        # Assign ic_ids continuing from existing tracked ICs
        next_id = max((ic.ic_id for ic in tracked_ics), default=-1) + 1

        filled_ics = []
        for i, cand in enumerate(new_candidates):
            delta = cand["delta"]
            tp_pct = cand["tp_pct"]
            ic_id = next_id + i
            ic_label = f"IC-{ic_id + 1}"

            logger.info(f"  Placing {ic_label}: {cand['strategy']} x{qty} at {int(delta*100)}δ")

            entered = await self.attempt_entry(chain=chain, target_delta=delta, target_qty=qty)
            if not entered:
                logger.warning(f"  {ic_label} entry failed — skipping")
                self._reset_entry_state()
                continue

            filled = await self.wait_for_entry_fill()
            if not filled:
                logger.warning(f"  {ic_label} did not fill — cancelling")
                await self._cancel_order_safe(self.entry_order_id)
                self._reset_entry_state()
                continue

            new_ic = TrackedIronCondor(
                ic_id=ic_id, label=ic_label,
                strikes=self.strikes,
                quantity=self.quantity,
                fill_credit=self.fill_credit,
                portfolio_value=self.portfolio_value,
                zdom_tp_pct=tp_pct,
                zdom_strategy=cand["strategy"],
            )
            filled_ics.append(new_ic)
            self._log_ic_details([new_ic])
            logger.info(
                f"  {ic_label} FILLED: {cand['strategy']} x{self.quantity} "
                f"credit=${self.fill_credit:.2f} TP={int(tp_pct*100)}%"
            )

        if not filled_ics:
            logger.info(f"  [{slot_time}] Monitor scan: no positions filled")

        return filled_ics

    # ===== Per-IC Monitoring Helpers =====

    async def _monitor_short_ic(self, ic: TrackedIronCondor, chain: dict,
                                now, ic_unrealized: dict, should_print: bool = True):
        """Monitor a SHORT iron condor. Debit falling = profit."""
        debit = self.builder.get_current_position_value(chain, ic.strikes)
        if debit is None:
            logger.debug(f"  {ic.label}: could not get position value")
            return

        # Unrealized P&L: profit = (credit - debit) × qty × 100 - fees
        fee_per_contract = settings.BOT_COST_PER_LEG * 8
        unrealized_dollars = (ic.fill_credit - debit) * ic.quantity * 100 - fee_per_contract * ic.quantity
        ic_unrealized[ic.ic_id] = unrealized_dollars

        pnl_pct = round((1 - debit / ic.fill_credit) * 100) if ic.fill_credit > 0 else 0
        if ic.sltp_active_tier_idx >= 0:
            act_pct = ic.sltp_tiers[ic.sltp_active_tier_idx][0]
            flr_pct = ic.sltp_tiers[ic.sltp_active_tier_idx][1]
            sltp_tag = f" [SLTP {act_pct:.1%}->{flr_pct:.1%} floor@${ic.sltp_floor_debit:.2f}]"
        else:
            sltp_tag = ""
        if should_print:
            sign = "+" if unrealized_dollars >= 0 else ""
            logger.info(
                f"  [{now.strftime('%H:%M:%S')}] {ic.label}: "
                f"debit ${debit:.2f}  P/L {pnl_pct:+d}%  |  "
                f"unrealized {sign}${unrealized_dollars:,.0f} "
                f"({ic.quantity}x ${debit:.2f} vs ${ic.fill_credit:.2f} credit)"
                f"{sltp_tag}"
            )

        # --- Tiered SLTP: ratcheting profit lock ---
        # For short ICs: debit FALLING = profit → lower debit activates higher tiers
        if ic.sltp_tiers:
            prev_tier_idx = ic.sltp_active_tier_idx
            for idx in range(len(ic.sltp_tiers) - 1, ic.sltp_active_tier_idx, -1):
                act_pct, flr_pct, act_debit, flr_debit = ic.sltp_tiers[idx]
                if debit <= act_debit:
                    ic.sltp_active_tier_idx = idx
                    ic.sltp_floor_debit = flr_debit
                    logger.debug(
                        f"  SLTP ratchet: {ic.label} floor -> {flr_pct:.2%} "
                        f"(debit ${debit:.2f} <= ${act_debit:.2f})"
                    )
                    if idx > prev_tier_idx:
                        logger.info(
                            f"  >>> {ic.label}: SLTP {act_pct:.0%} profit ACTIVATED "
                            f"(lock {flr_pct:.0%} @ ${flr_debit:.2f})"
                        )
                        play_sound(TP_SOUND)
                        send_notification(
                            f"Lock {flr_pct:.0%} Profit - {ic.label}",
                            f"Floor ${flr_debit:.2f}"
                        )
                    break

            # Floor breached: debit reversed up past lock-in
            if ic.sltp_active_tier_idx >= 0 and debit >= ic.sltp_floor_debit:
                flr_pct = ic.sltp_tiers[ic.sltp_active_tier_idx][1]
                logger.warning(
                    f"  === {ic.label}: SLTP TRIGGERED — locking {flr_pct:.0%} profit "
                    f"(debit ${debit:.2f} >= floor ${ic.sltp_floor_debit:.2f}) ==="
                )
                play_sound(SL_SOUND)
                await self._close_ic(ic, f"sltp_lock_{flr_pct:.0%}")
                return

        # TP checks (debit falling = good)
        if debit <= ic.tp_50_debit and AlertLevel.TP_50 not in ic.alerts_fired:
            ic.alerts_fired.add(AlertLevel.TP_50)
            ic.alerts_fired.add(AlertLevel.TP_25)
            alert_tp(ic, AlertLevel.TP_50, debit)

        elif debit <= ic.tp_25_debit and AlertLevel.TP_25 not in ic.alerts_fired:
            ic.alerts_fired.add(AlertLevel.TP_25)
            alert_tp(ic, AlertLevel.TP_25, debit)

        # SL checks (debit rising = bad)
        if debit >= ic.sl_exit_debit and AlertLevel.SL_EXIT not in ic.alerts_fired:
            ic.alerts_fired.add(AlertLevel.SL_EXIT)
            ic.alerts_fired.add(AlertLevel.SL_WARNING)
            alert_sl(ic, AlertLevel.SL_EXIT, debit)
            await self._close_ic(ic, "stop_loss_portfolio")

        elif debit >= ic.sl_warning_debit and AlertLevel.SL_WARNING not in ic.alerts_fired:
            ic.alerts_fired.add(AlertLevel.SL_WARNING)
            alert_sl(ic, AlertLevel.SL_WARNING, debit)

    # ===== Process Cleanup =====

    def _kill_stale_processes(self):
        """
        Kill stale processes that could block this bot instance.

        1. Other zero_dte_bot.py processes for the SAME account (not self, not sibling bots)
        2. Processes holding DuckDB file locks in data_store/
        3. Stale ThetaData Terminal processes (will be restarted by _ensure_thetadata_running)
        """
        my_pid = os.getpid()

        # --- 1. Kill other bot instances for the SAME account ---
        # Build account identifier to match only same-account bots
        if settings.BROKER.lower() == "schwab":
            acct_id = settings.SCHWAB_ACCOUNT_ID
        else:
            acct_id = settings.TRADIER_ACCOUNT_ID
        try:
            result = subprocess.run(
                ["pgrep", "-af", "zero_dte_bot.py"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split(None, 1)
                    if len(parts) < 2:
                        continue
                    pid = int(parts[0])
                    cmdline = parts[1]
                    if pid == my_pid:
                        continue
                    # Only kill if same account: check env-file contains our account ID
                    # or if no env-file and we're using default .env (legacy single-bot mode)
                    env_file_in_cmd = None
                    if "--env-file" in cmdline:
                        idx = cmdline.index("--env-file")
                        env_parts = cmdline[idx:].split()
                        if len(env_parts) >= 2:
                            env_file_in_cmd = env_parts[1]
                    # Determine if this other process serves the same account
                    same_account = False
                    if env_file_in_cmd and hasattr(settings, '_env_file_path'):
                        # Both use env-files: only match if same file
                        same_account = (env_file_in_cmd == str(getattr(settings, '_env_file_path', '')))
                    elif not env_file_in_cmd and not hasattr(settings, '_env_file_path'):
                        # Both use default .env: same account
                        same_account = True
                    # else: one uses env-file, other doesn't — different accounts, leave alone

                    if same_account:
                        logger.warning(f"  Killing stale bot instance PID {pid} (same account: {acct_id})")
                        try:
                            os.kill(pid, 9)
                        except ProcessLookupError:
                            pass
                    else:
                        logger.info(f"  Skipping sibling bot PID {pid} (different account)")
                import time as _t
                _t.sleep(1)
        except Exception as e:
            logger.debug(f"  Bot process cleanup skipped: {e}")

        # --- 2. Kill processes holding DuckDB locks ---
        db_dir = Path("data_store")
        if db_dir.exists():
            duckdb_files = list(db_dir.glob("*.duckdb"))
            for db_file in duckdb_files:
                try:
                    result = subprocess.run(
                        ["lsof", str(db_file)],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.stdout.strip():
                        lines = result.stdout.strip().split('\n')[1:]  # skip header
                        for line in lines:
                            parts = line.split()
                            if len(parts) >= 2:
                                pid = int(parts[1])
                                if pid != my_pid:
                                    logger.warning(f"  Killing PID {pid} holding lock on {db_file.name}")
                                    try:
                                        os.kill(pid, 9)
                                    except ProcessLookupError:
                                        pass
                except Exception as e:
                    logger.debug(f"  DuckDB lock cleanup for {db_file.name} skipped: {e}")

            # Also remove .wal files that can hold stale locks
            for wal_file in db_dir.glob("*.duckdb.wal"):
                try:
                    wal_file.unlink()
                    logger.info(f"  Removed stale WAL file: {wal_file.name}")
                except Exception:
                    pass

        # Brief pause to let OS release resources
        import time as _t
        _t.sleep(1)

    # ===== ThetaData =====

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

        # Not running — kill any stale instances before launching fresh
        logger.warning("  ThetaData Terminal not responding. Killing stale processes...")
        try:
            result = subprocess.run(
                ["pkill", "-9", "-f", "ThetaTerminal"],
                capture_output=True, timeout=5,
            )
            subprocess.run(
                ["pkill", "-9", "-f", "202602131.jar"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                logger.info("  Killed stale ThetaData processes")
                import time as _time_kill
                _time_kill.sleep(2)  # let port release
        except Exception:
            pass

        logger.info("  Starting fresh ThetaData Terminal...")

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

    # ===== Entry Methods =====

    async def attempt_entry(self, chain: dict = None, target_delta: float = None,
                            target_qty: int = 0) -> bool:
        """
        Full entry flow: fetch chain -> ML prediction -> select strikes -> size -> place order.

        Args:
            chain: Optional pre-fetched option chain (from v4 scan). If None, fetches fresh.
            target_delta: Override delta for strike selection (from multi-delta scoring).
            target_qty: Override quantity (for S2 diversified). 0 = auto-size.

        Returns True if entry order was placed, False otherwise.
        """
        logger.info("--- Attempting Entry ---")

        if self.already_traded_today:
            # S2 diversified: allow multiple entries (0=unlimited, >1=capped)
            if settings.BOT_ZDOM_ENABLED and settings.BOT_ZDOM_MAX_POSITIONS != 1:
                pass  # allow re-entry
            else:
                logger.warning("Already traded today. One-and-done.")
                return False

        # Check if daily target already met (skip in dry-run)
        if not self.dry_run:
            daily_pnl = await self._fetch_daily_pnl(open_ics=[])
            if daily_pnl.remaining_to_target <= 0:
                logger.info(f"  *** DAILY GOAL MET: +${daily_pnl.realized_today:.2f} "
                            f"(target: ${daily_pnl.daily_target:.2f}) ***")
                play_sound("/System/Library/Sounds/Glass.aiff")
                send_notification(
                    "0DTE Bot — Daily Goal Met",
                    f"+${daily_pnl.realized_today:.2f} realized. Override to enter another trade?"
                )
                answer = await self._async_prompt(
                    "  Daily goal already met. Enter another trade anyway? [y/N] (60s): ", timeout=60
                )
                if answer.lower() not in ('y', 'yes'):
                    logger.info("  Skipping entry — daily goal met, no override.")
                    return False
                logger.info("  Manual override APPROVED — entering despite daily goal met.")
            else:
                logger.info(f"  Daily P&L: +${daily_pnl.realized_today:.2f}, "
                            f"need ${daily_pnl.remaining_to_target:.2f} more")

        # Fetch 0DTE chain (skip if pre-fetched by v4 scan)
        if chain is None:
            chain = await self._fetch_0dte_chain()
            if not chain:
                return False

        # v3 shown for information only (skipped when ZDOM enabled)
        if not settings.BOT_ZDOM_ENABLED:
            if self._v3_cached_confidence is not None:
                label = "PASS" if self._v3_cached_confidence >= settings.BOT_MIN_CONFIDENCE else "FAIL"
                logger.info(f"  ML v3: {self._v3_cached_confidence:.3f} [{label}] (informational)")
            elif not self.skip_ml and self.predictor.is_ready():
                confidence = await self._run_ml_prediction(chain)
                if confidence is not None:
                    self._v3_cached_confidence = confidence
                    label = "PASS" if confidence >= settings.BOT_MIN_CONFIDENCE else "FAIL"
                    logger.info(f"  ML v3: {confidence:.3f} [{label}] (informational)")
            else:
                logger.info("  ML check: skipped (--skip-ml)")

        # Select strikes
        effective_delta = target_delta or settings.BOT_SHORT_DELTA
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
                target_delta=effective_delta,
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

        # Size the position
        if target_qty > 0:
            # S2 diversified: use pre-allocated quantity
            self.quantity = target_qty
        else:
            # Auto-size based on portfolio and buying power
            buying_power = await self.schwab_client.get_buying_power()
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
            logger.debug(f"  Chain fetched: {settings.BOT_SYMBOL} @ ${underlying_price:,.2f}")
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

    # ===== Candle DB Backfill =====

    def _backfill_candle_db(self):
        """
        Populate intraday_candles table from ThetaData if empty/stale.

        Fetches ~45 days of 5-min SPX and VIX candles so that the v3 feature
        extractors (RealizedVol, MarketFeatures, Intraday, Regime) have data.
        Runs once at startup in pre_flight(); skips if data is already fresh.
        """
        import pandas as pd

        loader = self.candle_loader  # triggers table creation via _ensure_table_exists

        for symbol in ("SPX", "VIX"):
            try:
                existing = loader.get_available_date_range(symbol)
                today = date.today()

                if existing:
                    last_date = existing[1].date() if hasattr(existing[1], 'date') else existing[1]
                    # If last data is from today or yesterday (non-weekend), skip
                    gap = (today - last_date).days
                    if gap <= 1 or (gap <= 3 and today.weekday() == 0):
                        logger.info(f"  Candle DB: {symbol} data is fresh (last: {last_date}) — skip backfill")
                        continue
                    # Partial backfill: only fetch missing days
                    start = last_date + timedelta(days=1)
                    logger.info(f"  Candle DB: {symbol} stale (last: {last_date}) — backfilling {start}..{today}")
                else:
                    start = today - timedelta(days=60)  # ~45 trading days
                    logger.info(f"  Candle DB: {symbol} empty — backfilling {start}..{today}")

                df = self._fetch_thetadata_candles(symbol, start, today)
                if df is not None and not df.empty:
                    inserted = loader.insert_candles(symbol, df)
                    logger.info(f"  Candle DB: {symbol} backfilled {inserted} rows")
                else:
                    logger.warning(f"  Candle DB: {symbol} — no data returned from ThetaData")

            except Exception as e:
                logger.warning(f"  Candle DB backfill failed for {symbol}: {e}")

    def _fetch_thetadata_candles(self, symbol: str, start: date, end: date) -> 'pd.DataFrame | None':
        """
        Fetch OHLC candles from ThetaData v3 for a date range.

        Uses /v3/index/history/ohlc with 5-min interval (good balance of
        resolution vs data size). Returns DataFrame with timestamp/open/high/low/close/volume.
        """
        import pandas as pd

        try:
            url = "http://127.0.0.1:25503/v3/index/history/ohlc"
            params = {
                "symbol": symbol,
                "start_date": start.strftime("%Y%m%d"),
                "end_date": end.strftime("%Y%m%d"),
                "interval": "5m",
                "start_time": "09:30:00",
                "end_time": "16:00:00",
                "format": "json",
            }

            query = "&".join(f"{k}={v}" for k, v in params.items())
            full_url = f"{url}?{query}"

            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=30) as resp:
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
                return None

            df = pd.DataFrame(response)

            # Normalize column names
            col_map = {}
            for col in df.columns:
                lower = col.lower()
                if lower in ('open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime', 'date', 'time'):
                    col_map[col] = lower
            if col_map:
                df = df.rename(columns=col_map)

            # Build timestamp from date + time columns if no timestamp column
            if 'timestamp' not in df.columns and 'datetime' not in df.columns:
                if 'date' in df.columns and 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(
                        df['date'].astype(str) + ' ' + df['time'].astype(str),
                        errors='coerce'
                    )
                elif 'date' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
                else:
                    # Use ms_of_day if available (ThetaData format)
                    if 'ms_of_day' in df.columns and 'date' not in df.columns:
                        # Try to find a date-like column
                        for c in df.columns:
                            if 'date' in c.lower():
                                df['timestamp'] = pd.to_datetime(df[c].astype(str), format='%Y%m%d', errors='coerce')
                                break
                    if 'timestamp' not in df.columns:
                        df['timestamp'] = pd.Timestamp.now()  # fallback
            elif 'datetime' in df.columns and 'timestamp' not in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'], errors='coerce')

            # Ensure OHLC columns
            required = ['open', 'high', 'low', 'close']
            if not all(c in df.columns for c in required):
                logger.warning(f"  ThetaData {symbol} missing OHLC columns. Have: {list(df.columns)}")
                return None

            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'volume' not in df.columns:
                df['volume'] = 0

            # Filter weekends (ThetaData forward-fills Sat/Sun)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'].dt.weekday < 5]

            # Drop rows with NaN OHLC
            df = df.dropna(subset=required)

            result_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            return df[result_cols].reset_index(drop=True)

        except Exception as e:
            logger.warning(f"  ThetaData fetch failed for {symbol}: {e}")
            return None

    # ===== v4/v5 Entry Timing Methods =====

    async def _show_all_model_scores(self, chain: dict, candles=None, prefix: str = "  "):
        """Unified scorecard: v3 through v6 with thresholds and pass/fail status.

        Args:
            chain: 0DTE option chain dict
            candles: pre-fetched 5-min candles (fetched if None)
            prefix: log line prefix for indentation
        """
        try:
            if candles is None:
                candles = self._fetch_spx_5min_candles()
            has_candles = candles is not None and len(candles) >= 6

            atm_iv = self._extract_atm_iv(chain) if has_candles else None
            underlying = chain.get('underlyingPrice', 0)
            bar_len = 20

            def bar(score):
                filled = int(score * bar_len)
                return "\u2588" * filled + "\u2591" * (bar_len - filled)

            def status(score, thresh):
                return "PASS" if score >= thresh else "FAIL"

            logger.info(f"{prefix}{'='*56}")
            logger.info(f"{prefix}  MODEL SCORECARD  |  SPX=${underlying:,.0f}  |  ATM-IV={atm_iv:.1f}" if atm_iv else
                        f"{prefix}  MODEL SCORECARD  |  SPX=${underlying:,.0f}")
            logger.info(f"{prefix}{'='*56}")

            # --- v3: day-level risk filter ---
            v3_thresh = settings.BOT_MIN_CONFIDENCE
            if self._v3_cached_confidence is not None:
                v3 = self._v3_cached_confidence
                v3_st = status(v3, v3_thresh)
                logger.info(f"{prefix}  v3 risk filter:  {v3:.3f} |{bar(v3)}|  thresh={v3_thresh:.2f}  [{v3_st}]")
            elif self.predictor.is_ready():
                logger.info(f"{prefix}  v3 risk filter:  (not yet scored)          thresh={v3_thresh:.2f}")
            else:
                logger.info(f"{prefix}  v3 risk filter:  (model not loaded)")

            # --- v4: entry timing ---
            v4_thresh = settings.BOT_V4_MIN_CONFIDENCE
            if has_candles and self.predictor.v4_ready:
                v4_score = self.predictor.predict_v4(
                    candles_5min=candles,
                    daily_data=self._v4_daily_data_cache,
                    option_atm_iv=atm_iv,
                )
                if v4_score is not None:
                    v4_st = status(v4_score, v4_thresh)
                    logger.info(f"{prefix}  v4 entry timing: {v4_score:.3f} |{bar(v4_score)}|  thresh={v4_thresh:.2f}  [{v4_st}]")
                else:
                    logger.info(f"{prefix}  v4 entry timing: (prediction failed)       thresh={v4_thresh:.2f}")
            elif not self.predictor.v4_ready:
                logger.info(f"{prefix}  v4 entry timing: (model not loaded)")
            else:
                logger.info(f"{prefix}  v4 entry timing: (insufficient candles)    thresh={v4_thresh:.2f}")

            # --- v5: TP probability ---
            v5_thresh = settings.BOT_V5_TP25_THRESHOLD
            if has_candles and self.predictor.v5_ready:
                multi_delta = (
                    settings.BOT_V5_MULTI_DELTA
                    and len(self.predictor.available_v5_deltas) > 1
                )
                if multi_delta:
                    all_delta_scores = self._score_all_deltas(candles, atm_iv, chain)
                    if all_delta_scores:
                        best = all_delta_scores[0]
                        tp25_v = best["tp25"]
                        tp50_v = best.get("tp50")
                        v5_st = status(tp25_v, v5_thresh)
                        tp50_str = f"  TP50={tp50_v:.3f}" if tp50_v is not None else ""
                        logger.info(f"{prefix}  v5 TP (d{int(best['delta']*100)}):   TP25={tp25_v:.3f} |{bar(tp25_v)}|  thresh={v5_thresh:.3f}  [{v5_st}]{tp50_str}")
                        # Show other deltas compactly
                        if len(all_delta_scores) > 1:
                            others = "  ".join(
                                f"d{int(ds['delta']*100)}={ds['tp25']:.3f}"
                                for ds in all_delta_scores[1:]
                            )
                            logger.info(f"{prefix}    other deltas: {others}")
                else:
                    tp25_score = tp50_score = None
                    if self.predictor.v5_tp25_ready:
                        tp25_score = self.predictor.predict_v5(
                            candles_5min=candles,
                            daily_data=self._v4_daily_data_cache,
                            option_atm_iv=atm_iv,
                            target="tp25",
                        )
                    if self.predictor.v5_tp50_ready:
                        tp50_score = self.predictor.predict_v5(
                            candles_5min=candles,
                            daily_data=self._v4_daily_data_cache,
                            option_atm_iv=atm_iv,
                            target="tp50",
                        )
                    if tp25_score is not None:
                        v5_st = status(tp25_score, v5_thresh)
                        tp50_str = f"  TP50={tp50_score:.3f}" if tp50_score is not None else ""
                        logger.info(f"{prefix}  v5 TP (d10):    TP25={tp25_score:.3f} |{bar(tp25_score)}|  thresh={v5_thresh:.3f}  [{v5_st}]{tp50_str}")
                    else:
                        logger.info(f"{prefix}  v5 TP:           (prediction failed)       thresh={v5_thresh:.3f}")
            elif not self.predictor.v5_ready:
                logger.info(f"{prefix}  v5 TP:           (model not loaded)")
            else:
                logger.info(f"{prefix}  v5 TP:           (insufficient candles)    thresh={v5_thresh:.3f}")

            # --- v6: ensemble ---
            v6_thresh = settings.BOT_V6_TP25_THRESHOLD
            v6_consensus_req = settings.BOT_V6_MIN_CONSENSUS
            if has_candles and self.predictor.v6_ready:
                result_tp25 = self.predictor.predict_v6_ensemble(
                    candles_5min=candles,
                    daily_data=self._v4_daily_data_cache,
                    option_atm_iv=atm_iv,
                    target="tp25"
                )
                result_tp50 = self.predictor.predict_v6_ensemble(
                    candles_5min=candles,
                    daily_data=self._v4_daily_data_cache,
                    option_atm_iv=atm_iv,
                    target="tp50"
                )
                tp25 = result_tp25['ensemble_prob']
                tp50 = result_tp50['ensemble_prob']
                cons = result_tp25['consensus_count']
                score_pass = tp25 >= v6_thresh
                cons_pass = cons >= v6_consensus_req
                overall = "PASS" if (score_pass and cons_pass) else "FAIL"
                n_windows = self.predictor.V6_EXPECTED_MODELS
                logger.info(f"{prefix}  v6 ensemble:     TP25={tp25:.3f} |{bar(tp25)}|  thresh={v6_thresh:.3f}  [{overall}]  << GATE")
                indiv_str = "  ".join(
                    f"{w}={result_tp25['individual_probs'][w]:.3f}"
                    for w in self.predictor.V6_WINDOWS
                    if w in result_tp25['individual_probs']
                )
                logger.info(f"{prefix}    TP50={tp50:.3f}  |  consensus={cons}/{n_windows} "
                           f"{'PASS' if cons_pass else 'FAIL'}  |  {indiv_str}")
                if result_tp25['disagreement']:
                    logger.warning(f"{prefix}    HIGH DISAGREEMENT (std={result_tp25['std']:.3f})")
                # Cache daily data
                if self._v4_daily_data_cache is None and self.predictor._daily_cache:
                    self._v4_daily_data_cache = self.predictor._daily_cache
            elif not self.predictor.v6_ready:
                logger.info(f"{prefix}  v6 ensemble:     (model not loaded)")
            else:
                logger.info(f"{prefix}  v6 ensemble:     (insufficient candles)    thresh={v6_thresh:.3f}")

            logger.info(f"{prefix}{'='*56}")

            # Cache daily data from any model that populated it
            if self._v4_daily_data_cache is None and self.predictor._daily_cache:
                self._v4_daily_data_cache = self.predictor._daily_cache

        except Exception as e:
            logger.warning(f"{prefix}Could not compute model scores: {e}")

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

    # ── v8 feature computation helpers ────────────────────────────────────

    def _fetch_spx_1min_candles(self) -> 'pd.DataFrame | None':
        """Fetch today's SPX 1-min candles from ThetaData for v8 features."""
        import pandas as pd
        from datetime import date

        today = date.today()
        if today.weekday() >= 5:
            return None

        try:
            url = "http://127.0.0.1:25503/v3/index/history/ohlc"
            params = {
                "symbol": "SPX",
                "start_date": today.strftime("%Y%m%d"),
                "end_date": today.strftime("%Y%m%d"),
                "interval": "1m",
                "start_time": "09:30:00",
                "end_time": self._now_et().strftime("%H:%M:%S"),
                "format": "json",
            }
            query = "&".join(f"{k}={v}" for k, v in params.items())
            full_url = f"{url}?{query}"

            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            response = data.get("response", data if isinstance(data, list) else [])
            if response and isinstance(response[0], dict) and "data" in response[0]:
                bars = []
                for entry in response:
                    for dp in entry.get("data", []):
                        bars.append(dp)
                response = bars

            if not response:
                return None

            df = pd.DataFrame(response)
            col_map = {}
            for col in df.columns:
                lower = col.lower()
                if lower in ('open', 'high', 'low', 'close', 'timestamp', 'datetime', 'time'):
                    col_map[col] = lower
            if col_map:
                df = df.rename(columns=col_map)

            required = ['open', 'high', 'low', 'close']
            if not all(c in df.columns for c in required):
                return None

            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=required)

            # Add timestamp column if missing (synthesize from index)
            if 'timestamp' not in df.columns:
                import pytz as _pytz
                eastern = _pytz.timezone('US/Eastern')
                base = eastern.localize(datetime.combine(today, time(9, 30)))
                df['timestamp'] = [base + timedelta(minutes=i) for i in range(len(df))]

            # Ensure timestamp is datetime (required for resample in v8 feature store)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            logger.debug(f"  Fetched {len(df)} 1-min SPX candles for v8")
            return df

        except Exception as e:
            logger.warning(f"  Failed to fetch 1-min SPX candles: {e}")
            return None

    def _fetch_vix_1min_candles(self) -> 'pd.DataFrame | None':
        """Fetch today's VIX 1-min candles from ThetaData."""
        import pandas as pd
        from datetime import date

        today = date.today()
        if today.weekday() >= 5:
            return None

        try:
            url = "http://127.0.0.1:25503/v3/index/history/ohlc"
            params = {
                "symbol": "VIX",
                "start_date": today.strftime("%Y%m%d"),
                "end_date": today.strftime("%Y%m%d"),
                "interval": "1m",
                "start_time": "09:30:00",
                "end_time": self._now_et().strftime("%H:%M:%S"),
                "format": "json",
            }
            query = "&".join(f"{k}={v}" for k, v in params.items())
            full_url = f"{url}?{query}"

            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            response = data.get("response", data if isinstance(data, list) else [])
            if response and isinstance(response[0], dict) and "data" in response[0]:
                bars = []
                for entry in response:
                    for dp in entry.get("data", []):
                        bars.append(dp)
                response = bars

            if not response:
                return None

            df = pd.DataFrame(response)
            col_map = {}
            for col in df.columns:
                lower = col.lower()
                if lower in ('open', 'high', 'low', 'close', 'timestamp'):
                    col_map[col] = lower
            if col_map:
                df = df.rename(columns=col_map)

            required = ['open', 'high', 'low', 'close']
            if not all(c in df.columns for c in required):
                return None

            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'timestamp' not in df.columns:
                import pytz as _pytz
                eastern = _pytz.timezone('US/Eastern')
                base = eastern.localize(datetime.combine(today, time(9, 30)))
                df['timestamp'] = [base + timedelta(minutes=i) for i in range(len(df))]

            # Ensure timestamp is datetime (required for resample in v8 feature store)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df

        except Exception as e:
            logger.debug(f"  VIX 1-min candles unavailable: {e}")
            return None

    def _schwab_chain_to_v8(self, chain: dict) -> 'tuple[pd.DataFrame, pd.DataFrame]':
        """
        Convert Schwab option chain dict to v8 format DataFrames.

        Returns (chain_df, chain_oi_df) suitable for v8 feature store.
        """
        import pandas as pd

        underlying = chain.get('underlyingPrice', 0)
        rows = []
        oi_rows = []

        for right_key, right_label in [('putExpDateMap', 'P'), ('callExpDateMap', 'C')]:
            exp_map = chain.get(right_key, {})
            if not exp_map:
                continue
            # Use first (nearest) expiration only
            strikes_map = next(iter(exp_map.values()), {})
            for strike_str, contracts in strikes_map.items():
                try:
                    strike = float(strike_str)
                except (ValueError, TypeError):
                    continue
                contract = contracts[0] if isinstance(contracts, list) else contracts
                if not isinstance(contract, dict):
                    continue

                rows.append({
                    'strike': strike,
                    'right': right_label,
                    'bid': contract.get('bid', 0),
                    'ask': contract.get('ask', 0),
                    'delta': abs(contract.get('delta', 0)),
                    'gamma': contract.get('gamma', 0),
                    'theta': contract.get('theta', 0),
                    'vega': contract.get('vega', 0),
                    'implied_vol': contract.get('volatility', 0),
                    'underlying_price': underlying,
                    'bid_size': contract.get('bidSize', 0),
                    'ask_size': contract.get('askSize', 0),
                })
                oi_rows.append({
                    'strike': strike,
                    'right': right_label,
                    'open_interest': contract.get('openInterest', 0),
                })

        chain_df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['strike', 'right', 'bid', 'ask', 'delta', 'gamma',
                     'theta', 'vega', 'implied_vol', 'underlying_price']
        )
        oi_df = pd.DataFrame(oi_rows) if oi_rows else pd.DataFrame(
            columns=['strike', 'right', 'open_interest']
        )
        return chain_df, oi_df

    def _build_cross_asset_data(self) -> dict:
        """Build cross-asset data dict for v8 features from yfinance."""
        try:
            import yfinance as yf
            from datetime import date as _date

            today = _date.today()
            start = today - timedelta(days=30)

            vix = yf.download('^VIX', start=str(start), end=str(today + timedelta(days=1)),
                              progress=False, auto_adjust=True)
            vix9d = yf.download('^VIX9D', start=str(start), end=str(today + timedelta(days=1)),
                                progress=False, auto_adjust=True)
            tlt = yf.download('TLT', start=str(start), end=str(today + timedelta(days=1)),
                              progress=False, auto_adjust=True)

            cross = {}

            if not vix.empty and len(vix) >= 2:
                vals = vix['Close'].values.flatten()
                cross['vix_level'] = float(vals[-1])
                cross['vix_change_1d'] = float(vals[-1] - vals[-2])
                cross['vix_change_1d_pct'] = float((vals[-1] / vals[-2] - 1) * 100) if vals[-2] > 0 else 0
                if len(vals) >= 10:
                    sma10 = float(vals[-10:].mean())
                    cross['vix_sma_10d_dist'] = float((vals[-1] - sma10) / sma10 * 100) if sma10 > 0 else 0
                if len(vals) >= 20:
                    sma20 = float(vals[-20:].mean())
                    cross['vix_sma_20d_dist'] = float((vals[-1] - sma20) / sma20 * 100) if sma20 > 0 else 0
                    pctile = float((vals[-1:] <= vals[-252:]).mean() * 100) if len(vals) >= 252 else 50.0
                    cross['vix_percentile_1y'] = pctile

            if not vix9d.empty and not vix.empty:
                v9d = float(vix9d['Close'].values.flatten()[-1])
                cross['vix9d_vs_vix'] = v9d / cross.get('vix_level', 20) if cross.get('vix_level', 0) > 0 else 1.0
                cross['vvix_level'] = v9d  # Proxy

            if not tlt.empty and len(tlt) >= 2:
                tlt_vals = tlt['Close'].values.flatten()
                cross['yield_10y'] = float(tlt_vals[-1])
                cross['yield_10y_change'] = float(tlt_vals[-1] - tlt_vals[-2])

            return cross
        except Exception as e:
            logger.debug(f"  Cross-asset data unavailable: {e}")
            return {}

    def _compute_v8_features(self, chain: dict) -> 'dict | None':
        """
        Compute full v8 feature vector from live data.

        Gathers SPX 1-min candles, VIX candles, option chain, cross-asset
        data, and runs the v8 feature store pipeline.
        """
        import pandas as pd

        try:
            # Fetch live data
            spx_1m = self._fetch_spx_1min_candles()
            if spx_1m is None or len(spx_1m) < 5:
                logger.warning("  v8: insufficient 1-min candles")
                return None

            vix_1m = self._fetch_vix_1min_candles()
            chain_df, chain_oi = self._schwab_chain_to_v8(chain)
            spx_price = chain.get('underlyingPrice', 0)

            # Cross-asset data (cached for the day)
            if not hasattr(self, '_v8_cross_cache') or self._v8_cross_cache is None:
                self._v8_cross_cache = self._build_cross_asset_data()
            cross_data = self._v8_cross_cache or {}

            # Build data dict for v8 feature store
            data = {
                'spx_1m': spx_1m,
                'vix_1m': vix_1m if vix_1m is not None else pd.DataFrame(),
                'chain': chain_df,
                'chain_oi': chain_oi,
                'cross_data': cross_data,
                'spx_price': spx_price,
                'entry_time': self._now_et(),
            }

            # Compute features
            from ml.features.v8.feature_store import compute_all_features
            features = compute_all_features(data)

            logger.debug(f"  v8 features computed: {len(features)} features")
            return features

        except Exception as e:
            logger.warning(f"  v8 feature computation failed: {e}")
            return None

    def _score_all_deltas(self, candles, atm_iv, chain) -> list:
        """
        Score all loaded delta models and estimate credit per delta.

        Returns list of dicts sorted by TP25 descending:
          [{"delta": 0.15, "tp25": 0.756, "tp50": 0.612, "credit": 3.80}, ...]
        """
        all_scores = self.predictor.predict_v5_all_deltas(
            candles_5min=candles,
            daily_data=self._v4_daily_data_cache,
            option_atm_iv=atm_iv,
        )

        results = []
        for delta, scores in all_scores.items():
            tp25 = scores.get("tp25")
            if tp25 is None:
                continue

            # Estimate credit for this delta
            credit = 0.0
            try:
                strikes = self.builder.select_strikes(
                    chain,
                    target_delta=delta,
                    wing_width=settings.BOT_WING_WIDTH,
                )
                if strikes:
                    credit = strikes.get('net_credit', 0.0)
            except Exception:
                pass

            results.append({
                "delta": delta,
                "tp25": tp25,
                "tp50": scores.get("tp50"),
                "credit": credit,
            })

        # Sort by TP25 descending, TP50 as tiebreaker
        results.sort(key=lambda r: (r["tp25"], r.get("tp50") or 0), reverse=True)
        return results

    async def _entry_scan(self, use_v6: bool = True, **kwargs) -> bool:
        """
        Dynamic entry scanning loop — v6 ensemble is the sole go/no-go gate.

        Scans every V4_SCAN_INTERVAL minutes from ENTRY_START to ENTRY_END:
        1. FOMC gate check (if BOT_FOMC_SKIP_DAY enabled)
        2. Fetch 5-min SPX candles
        3. Compute v6 features + predict (v3/v4/v5 shown in scorecard only)
        4. If ensemble >= threshold AND consensus >= min -> attempt_entry()
        5. If no entry by ENTRY_END -> skip day

        Manual-approve is always on: prompts when score < threshold.

        Returns True if entry was placed and filled, False otherwise.
        """
        import pandas as pd
        from ml.models.predictor import get_economic_calendar_features

        scan_start = self._parse_time(settings.BOT_V4_ENTRY_START)
        scan_end = self._parse_time(settings.BOT_V4_ENTRY_END)
        interval = settings.BOT_V4_SCAN_INTERVAL
        threshold = settings.BOT_V6_TP25_THRESHOLD

        logger.info(f"--- v6 Entry Scan ---")
        logger.info(f"  Window: {settings.BOT_V4_ENTRY_START} - {settings.BOT_V4_ENTRY_END} ET")
        logger.info(f"  Interval: {interval} min | Threshold: {threshold:.0%}")
        logger.info(f"  Consensus: ≥{settings.BOT_V6_MIN_CONSENSUS}/{self.predictor.V6_EXPECTED_MODELS} models")

        # FOMC gate check
        if settings.BOT_FOMC_GATE_ENABLED:
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
                logger.info(f"  v6 scan window closed at {settings.BOT_V4_ENTRY_END} ET. No entry today.")
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

            # Step 2: Run v3 for informational scorecard (no gating)
            if self._v3_cached_confidence is None and not self.skip_ml and self.predictor.is_ready():
                v3_conf = await self._run_ml_prediction(chain)
                self._v3_cached_confidence = v3_conf
                if v3_conf is not None:
                    label = "PASS" if v3_conf >= settings.BOT_MIN_CONFIDENCE else "FAIL"
                    logger.info(f"  v3 risk filter: {v3_conf:.3f} [{label}] (informational only)")

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

            # Show unified scorecard every scan slot (v3/v4/v5 informational)
            logger.info(f"  {progress} {slot_time}")
            await self._show_all_model_scores(chain, candles=candles, prefix="    ")

            try:
                # v6 ensemble prediction
                result_tp25 = self.predictor.predict_v6_ensemble(
                    candles_5min=candles,
                    daily_data=self._v4_daily_data_cache,
                    option_atm_iv=atm_iv,
                    target="tp25"
                )
                result_tp50 = self.predictor.predict_v6_ensemble(
                    candles_5min=candles,
                    daily_data=self._v4_daily_data_cache,
                    option_atm_iv=atm_iv,
                    target="tp50"
                )

                # Cache daily data after first call
                if self._v4_daily_data_cache is None and self.predictor._daily_cache:
                    self._v4_daily_data_cache = self.predictor._daily_cache

                tp25_score = result_tp25['ensemble_prob']
                tp50_score = result_tp50['ensemble_prob']

                # Check consensus filter
                consensus_pass = result_tp25['consensus_count'] >= settings.BOT_V6_MIN_CONSENSUS

                if result_tp25['disagreement']:
                    play_sound("/System/Library/Sounds/Glass.aiff")

                n_windows = self.predictor.V6_EXPECTED_MODELS
                # Entry decision: ensemble >= threshold AND consensus >= min
                if tp25_score >= threshold and consensus_pass:
                    logger.info(f"    >>> ENTRY (ensemble={tp25_score:.3f}, consensus={result_tp25['consensus_count']}/{n_windows})")
                    entered = await self.attempt_entry(chain=chain)
                    if entered:
                        filled = await self.wait_for_entry_fill()
                        if filled:
                            return True
                elif tp25_score >= threshold and not consensus_pass:
                    logger.warning(f"    Ensemble score meets threshold but consensus FAILED "
                                 f"({result_tp25['consensus_count']}/{settings.BOT_V6_MIN_CONSENSUS}) — SKIP")
                else:
                    logger.info(f"    Score below threshold — waiting for next slot")

                # Manual approval for v6 below-threshold
                if not self.skip_wait and (tp25_score < threshold or not consensus_pass):
                    preview = await self._compute_entry_preview(chain)
                    if preview:
                        s = preview['strikes']
                        gap = threshold - tp25_score
                        logger.info(f"    --- MANUAL APPROVAL ---")
                        logger.info(f"    v6 Ensemble: {tp25_score:.3f}  (threshold {threshold:.3f}, gap {gap:.3f})")
                        logger.info(f"    TP50: {tp50_score:.3f}  |  Consensus: {result_tp25['consensus_count']}/{n_windows}")
                        logger.info(f"    Put spread:  {s.get('long_put_strike',0):.0f}/{s.get('short_put_strike',0):.0f}p"
                                    f"  (delta {s.get('short_put_delta',0):.3f})")
                        logger.info(f"    Call spread: {s.get('short_call_strike',0):.0f}/{s.get('long_call_strike',0):.0f}c"
                                    f"  (delta {s.get('short_call_delta',0):.3f})")
                        logger.info(f"    Credit: ${preview['credit']:.2f}  x{preview['quantity']} contracts")
                        logger.info(f"    TP: buy back ${preview['tp_debit']:.2f}  "
                                    f"(keep {settings.BOT_TAKE_PROFIT_PCT:.0%} -> +${preview['tp_profit']:,.0f})")
                        logger.info(f"    SL: close at ${preview['sl_debit']:.2f}  "
                                    f"({settings.BOT_STOP_LOSS_PCT:.0%} portfolio -> -${preview['sl_loss']:,.0f})")

                        play_sound("/System/Library/Sounds/Glass.aiff")
                        send_notification("0DTE Bot — v6 Approval Needed",
                                          f"Ensemble {tp25_score:.3f} | ${preview['credit']:.2f} x{preview['quantity']}")
                        answer = await self._async_prompt("    Override entry? [y/N] (60s timeout): ", timeout=60)
                        if answer.lower() in ('y', 'yes'):
                            logger.info(f"    Manual override APPROVED at {slot_time}")
                            result = await self.attempt_entry(chain=chain)
                            if result:
                                filled = await self.wait_for_entry_fill()
                                if filled:
                                    return True
                                logger.warning("    Entry did not fill. Cancelling and re-scanning.")
                                await self._cancel_order_safe(self.entry_order_id)
                                self._reset_entry_state()
                            else:
                                logger.warning("    Entry attempt failed after manual approval.")
                                self._reset_entry_state()
                        else:
                            reason = "timeout" if answer == '' else "declined"
                            logger.info(f"    Manual override {reason}. Continuing scan.")

                # Next scan slot or exit
                if self.skip_wait:
                    return False
                await asyncio.sleep(interval * 60)
                continue

            except Exception as e:
                logger.warning(f"  {progress} {slot_time} — v6 prediction error: {e}")
                if self.skip_wait:
                    return False
                await asyncio.sleep(interval * 60)
                continue

    async def wait_for_entry_fill(self) -> bool:
        """
        Poll entry order status until filled or timeout.

        Handles manual order edits: when Schwab cancels the original order
        and creates a replacement (e.g., user changed limit price), the bot
        detects the replacement by matching IC leg symbols and continues polling.

        Returns True if filled, False otherwise.
        """
        if self.dry_run:
            logger.info("[DRY-RUN] Entry fill simulated immediately.")
            return True

        logger.info("Waiting for entry fill...")
        start = datetime.now()
        timeout = timedelta(seconds=settings.BOT_ENTRY_TIMEOUT)

        # Extract expected leg symbols from the original order for replacement matching
        original_order = await self.schwab_client.get_order(self.entry_order_id)
        expected_symbols = set()
        for leg in original_order.get('orderLegCollection', []):
            sym = leg.get('instrument', {}).get('symbol', '')
            if sym:
                expected_symbols.add(sym)

        while datetime.now() - start < timeout:
            try:
                order_data = await self.schwab_client.get_order(self.entry_order_id)
                status = order_data.get('status', '')

                if status == 'FILLED':
                    # Extract actual fill credit from order data
                    self.fill_credit = float(order_data.get('price', self.strikes['net_credit']))
                    logger.info(f"  Entry FILLED at ${self.fill_credit:.2f}")
                    return True
                elif status == 'CANCELED':
                    # Order was canceled — could be manual edit (replacement) or actual cancel
                    replacement = await self._find_replacement_order(expected_symbols)
                    if replacement:
                        new_id = str(replacement.get('orderId', ''))
                        new_price = replacement.get('price', '?')
                        new_status = replacement.get('status', '')
                        logger.info(f"  Original order canceled — found replacement order {new_id} "
                                   f"(price=${new_price}, status={new_status})")
                        self.entry_order_id = new_id

                        if new_status == 'FILLED':
                            self.fill_credit = float(replacement.get('price', self.strikes['net_credit']))
                            logger.info(f"  Entry FILLED at ${self.fill_credit:.2f} (via replacement)")
                            return True
                        # Continue polling the replacement order
                        continue
                    else:
                        reason = order_data.get('statusDescription', 'No reason provided')
                        logger.warning(f"  Entry order CANCELED (no replacement found): {reason}")
                        return False
                elif status in ('REJECTED', 'EXPIRED'):
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

    async def _find_replacement_order(self, expected_symbols: set) -> Optional[dict]:
        """
        Search recent orders for a replacement that matches the same IC legs.

        When a user manually edits an order on Schwab (e.g., changes limit price),
        Schwab cancels the original and creates a new order with a new ID but
        the same leg symbols. This finds that replacement.
        """
        try:
            from_dt = datetime.now() - timedelta(hours=1)
            orders = await self.schwab_client.get_orders_for_account(
                from_entered_datetime=from_dt,
            )

            for order in orders:
                order_id = str(order.get('orderId', ''))
                status = order.get('status', '')

                # Skip the original canceled order and terminal states
                if order_id == self.entry_order_id:
                    continue
                if status in ('CANCELED', 'REJECTED', 'EXPIRED'):
                    continue

                # Match by leg symbols
                order_symbols = set()
                for leg in order.get('orderLegCollection', []):
                    sym = leg.get('instrument', {}).get('symbol', '')
                    if sym:
                        order_symbols.add(sym)

                if order_symbols == expected_symbols:
                    return order

        except Exception as e:
            logger.warning(f"Error searching for replacement order: {e}")

        return None

    # ===== Daily P&L =====

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

    # ===== Position Management =====

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

    async def reconstruct_multiple_ics_from_positions(self, positions: list = None) -> Optional[List[TrackedIronCondor]]:
        """
        Group SPX positions into multiple iron condors.

        Algorithm:
        1. Parse all OCC symbols, classify into long_puts, short_puts, short_calls, long_calls
        2. Sort each bucket by strike ascending
        3. Pair index-by-index: bucket[0] from each = IC#1, bucket[1] = IC#2, etc.
        4. Validate per-IC: LP < SP < SC < LC, matching quantities

        Returns list of TrackedIronCondor (without credits assigned yet), or None on error.
        """
        if positions is None:
            try:
                all_positions = await self.schwab_client.get_account_positions()
            except Exception as e:
                logger.error(f"Failed to fetch positions: {e}")
                return None
            positions = [
                p for p in all_positions
                if 'SPX' in p.get('instrument', {}).get('symbol', '').upper()
            ]

        if not positions:
            logger.error("No SPX option positions found.")
            return None

        if len(positions) % 4 != 0:
            logger.error(
                f"Expected a multiple of 4 SPX legs, found {len(positions)}. "
                f"Cannot group into iron condors."
            )
            return None

        # Parse each position
        legs = []
        for p in positions:
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

        logger.info(f"  Detected {len(tracked_ics)} iron condor(s) from {len(positions)} positions")
        return tracked_ics

    async def _assign_credits(self, tracked_ics: List[TrackedIronCondor]) -> bool:
        """
        Assign fill credits to tracked ICs via auto-detection from order history
        or position averagePrice fallback.
        """
        # Auto-detect from order history (primary: traces entries + rolls)
        logger.info("  Auto-detecting credits from order history...")
        if await self._detect_credits_from_orders(tracked_ics):
            return True

        # Fallback: compute net credit from position averagePrice fields
        logger.info("  Order history match failed. Falling back to position averagePrice...")
        if self._detect_credits_from_avg_price(tracked_ics):
            return True

        logger.error(
            "Could not auto-detect credits from order history or position data."
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

    def _extract_fill_price(self, order: dict) -> float:
        """
        Extract actual fill price from execution legs (not the order-level limit price).

        For multi-leg orders, computes net credit:
          net = sum(SELL leg fills) - sum(BUY leg fills)

        Falls back to order-level 'price' if execution legs are unavailable.
        Uses legId (1-indexed) to map execution legs to order legs.
        """
        activities = order.get('orderActivityCollection', [])
        legs = order.get('orderLegCollection', [])

        if not activities or not legs:
            return float(order.get('price', 0) or 0)

        # Build instruction map: legId (1-indexed) -> instruction
        instruction_by_leg_id = {}
        for i, leg in enumerate(legs):
            instruction_by_leg_id[i + 1] = leg.get('instruction', '')

        # Sum fill prices by direction (use first activity for multi-fill)
        sell_total = 0.0
        buy_total = 0.0
        for activity in activities:
            for exec_leg in activity.get('executionLegs', []):
                leg_id = exec_leg.get('legId', 0)
                price = float(exec_leg.get('price', 0))
                instruction = instruction_by_leg_id.get(leg_id, '')
                if instruction in ('SELL_TO_OPEN', 'SELL_TO_CLOSE'):
                    sell_total += price
                elif instruction in ('BUY_TO_OPEN', 'BUY_TO_CLOSE'):
                    buy_total += price
            break  # use first activity only

        net = round(sell_total - buy_total, 2)
        if net != 0:
            return abs(net)

        # Fallback to order-level price
        return float(order.get('price', 0) or 0)

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
                fill_price = self._extract_fill_price(order)
                entries.append({
                    'symbols': order_symbols,
                    'price': fill_price,
                    'quantity': order_qty,
                    'order_id': order.get('orderId'),
                    'time': _order_time(order),
                })
                logger.info(f"    ENTRY order {order.get('orderId')}: "
                            f"{order_qty}x ${fill_price:.2f} "
                            f"type={order.get('complexOrderStrategyType')}")

            elif kind == 'ROLL':
                # Separate BUY_TO_CLOSE (old legs) from SELL_TO_OPEN (new legs)
                btc_symbols = set()   # all legs being closed (BTC + STC)
                sto_symbols = set()   # all legs being opened (STO + BTO)
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
                    elif inst == 'SELL_TO_CLOSE':
                        # Long leg being closed (e.g., 4-leg spread roll)
                        btc_symbols.add(sym)
                    elif inst == 'BUY_TO_OPEN':
                        # Long leg being opened (e.g., 4-leg spread roll)
                        sto_symbols.add(sym)

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

                roll_fill_price = self._extract_fill_price(order)
                rolls.append({
                    'btc_symbols': btc_symbols,
                    'sto_symbols': sto_symbols,
                    'price': roll_fill_price,
                    'is_credit': roll_is_credit,
                    'quantity': order_qty,
                    'order_id': order.get('orderId'),
                    'time': _order_time(order),
                })
                direction = "credit" if roll_is_credit else "debit"
                logger.info(f"    ROLL order {order.get('orderId')}: "
                            f"{order_qty}x ${roll_fill_price:.2f} ({direction}) "
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

    async def reconstruct_ics_from_orders(self) -> Optional[List[TrackedIronCondor]]:
        """
        Build TrackedIronCondor objects directly from FILLED ENTRY orders
        (not positions). This avoids netting ambiguity when two ICs share a strike.

        Algorithm:
        1. Fetch filled orders (last 3 days), classify as ENTRY/ROLL/CLOSE
        2. Build IC states from ENTRY orders (symbols + credit)
        3. Apply ROLLs chronologically
        4. Filter out closed ICs (explicit CLOSE orders + expiration check)
        5. Validate remaining ICs against current positions
        6. Return TrackedIronCondor list with credits already assigned
        """
        try:
            now_utc = datetime.now(pytz.utc)
            lookback_utc = now_utc - timedelta(days=3)
            all_orders = await self.schwab_client.get_orders_for_account(
                from_entered_datetime=lookback_utc,
                to_entered_datetime=now_utc,
            )
        except Exception as e:
            logger.warning(f"Order-based recon: failed to fetch orders: {e}")
            return None

        if not all_orders:
            logger.info("Order-based recon: no orders in last 3 days")
            return None

        filled_orders = [o for o in all_orders if o.get('status', '').upper() == 'FILLED']
        if not filled_orders:
            logger.info(f"Order-based recon: {len(all_orders)} orders but none FILLED")
            return None

        logger.info(f"    {len(filled_orders)} filled orders (of {len(all_orders)} total)")

        def _order_time(o):
            t = o.get('enteredTime', '')
            try:
                return datetime.fromisoformat(t.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return datetime.min.replace(tzinfo=pytz.utc)

        filled_orders.sort(key=_order_time)

        # --- Classify all filled orders ---
        entries = []
        rolls = []
        closes = []
        for order in filled_orders:
            kind = self._classify_order(order)
            legs = order.get('orderLegCollection', [])
            if not legs:
                continue
            order_qty = max((l.get('quantity', 0) for l in legs), default=0)

            if kind == 'ENTRY':
                leg_details = []
                order_symbols = set()
                for l in legs:
                    sym = l.get('instrument', {}).get('symbol', '')
                    inst = l.get('instruction', '')
                    parsed = self.builder.parse_option_symbol(sym)
                    if parsed:
                        parsed['symbol'] = sym
                        parsed['instruction'] = inst
                        leg_details.append(parsed)
                    order_symbols.add(sym)

                fill_price = self._extract_fill_price(order)
                entries.append({
                    'symbols': order_symbols,
                    'leg_details': leg_details,
                    'price': fill_price,
                    'quantity': order_qty,
                    'order_id': order.get('orderId'),
                    'time': _order_time(order),
                })

            elif kind == 'ROLL':
                btc_symbols = set()
                sto_symbols = set()
                sto_leg_details = []
                btc_short_strikes = []
                sto_short_strikes = []
                for l in legs:
                    sym = l.get('instrument', {}).get('symbol', '')
                    inst = l.get('instruction', '')
                    parsed = self.builder.parse_option_symbol(sym)
                    if inst == 'BUY_TO_CLOSE':
                        btc_symbols.add(sym)
                        if parsed:
                            btc_short_strikes.append(parsed)
                    elif inst == 'SELL_TO_OPEN':
                        sto_symbols.add(sym)
                        if parsed:
                            parsed['symbol'] = sym
                            parsed['instruction'] = inst
                            sto_short_strikes.append(parsed)
                            sto_leg_details.append(parsed)
                    elif inst == 'SELL_TO_CLOSE':
                        btc_symbols.add(sym)
                    elif inst == 'BUY_TO_OPEN':
                        sto_symbols.add(sym)
                        if parsed:
                            parsed['symbol'] = sym
                            parsed['instruction'] = inst
                            sto_leg_details.append(parsed)

                roll_is_credit = True
                if btc_short_strikes and sto_short_strikes:
                    for new_leg in sto_short_strikes:
                        for old_leg in btc_short_strikes:
                            if new_leg['option_type'] == old_leg['option_type']:
                                if new_leg['option_type'] == 'C':
                                    roll_is_credit = new_leg['strike'] < old_leg['strike']
                                else:
                                    roll_is_credit = new_leg['strike'] > old_leg['strike']
                                break
                        break

                roll_fill_price = self._extract_fill_price(order)
                rolls.append({
                    'btc_symbols': btc_symbols,
                    'sto_symbols': sto_symbols,
                    'sto_leg_details': sto_leg_details,
                    'price': roll_fill_price,
                    'is_credit': roll_is_credit,
                    'quantity': order_qty,
                    'order_id': order.get('orderId'),
                    'time': _order_time(order),
                })

            elif kind == 'CLOSE':
                close_symbols = set()
                for l in legs:
                    close_symbols.add(l.get('instrument', {}).get('symbol', ''))
                closes.append({
                    'symbols': close_symbols,
                    'quantity': order_qty,
                    'order_id': order.get('orderId'),
                    'time': _order_time(order),
                })

        if not entries:
            logger.info("Order-based recon: no ENTRY orders found")
            return None

        logger.info(f"    Classified: {len(entries)} entries, {len(rolls)} rolls, {len(closes)} closes")
        for e in entries:
            logger.info(f"      ENTRY {e['order_id']}: {e['quantity']}x ${e['price']:.2f} "
                        f"legs={len(e['leg_details'])} syms={e['symbols']}")

        # --- Build IC states from entries ---
        ic_states = []
        for entry in entries:
            ic_states.append({
                'current_symbols': set(entry['symbols']),
                'leg_details': list(entry['leg_details']),  # copy to avoid mutation
                'cumulative_credit': entry['price'],
                'quantity': entry['quantity'],
                'order_id': entry['order_id'],
                'rolls_applied': 0,
            })

        # --- Apply rolls chronologically ---
        for roll in rolls:
            matched_state = None
            for state in ic_states:
                if (roll['btc_symbols'].issubset(state['current_symbols'])
                        and roll['quantity'] == state['quantity']):
                    matched_state = state
                    break

            if matched_state is None:
                logger.debug(f"    Roll {roll['order_id']} didn't match any IC state")
                continue

            matched_state['current_symbols'] -= roll['btc_symbols']
            matched_state['current_symbols'] |= roll['sto_symbols']

            # Update leg_details: remove closed legs, add new ones
            matched_state['leg_details'] = [
                ld for ld in matched_state['leg_details']
                if ld['symbol'] not in roll['btc_symbols']
            ]
            matched_state['leg_details'].extend(roll['sto_leg_details'])

            if roll['is_credit']:
                matched_state['cumulative_credit'] += roll['price']
            else:
                matched_state['cumulative_credit'] -= roll['price']
            matched_state['rolls_applied'] += 1
            logger.info(f"      Applied roll {roll['order_id']} to entry {matched_state['order_id']}")

        # --- Filter out closed ICs ---
        # Method 1: explicit CLOSE orders
        for close in closes:
            for state in ic_states:
                if state.get('closed'):
                    continue
                if (close['symbols'].issubset(state['current_symbols'])
                        and close['quantity'] == state['quantity']):
                    state['closed'] = True
                    logger.info(f"      Entry {state['order_id']}: closed by order {close['order_id']}")
                    break

        # Method 2: expiration check — 0DTE entries from previous days are expired
        today_str = self._now_et().strftime('%Y-%m-%d')
        for state in ic_states:
            if state.get('closed'):
                continue
            # Check expiration from any parsed leg
            for ld in state['leg_details']:
                exp = ld.get('expiration', '')
                if exp and exp < today_str:
                    state['closed'] = True
                    logger.info(f"      Entry {state['order_id']}: expired ({exp} < {today_str})")
                    break

        open_states = [s for s in ic_states if not s.get('closed')]
        logger.info(f"    After close/expiry filter: {len(open_states)} of {len(ic_states)} IC states remain")

        if not open_states:
            logger.info("Order-based recon: all entry orders are closed/expired")
            return None

        # --- Validate against current positions ---
        positions = await self._get_spx_positions()
        if not positions:
            logger.info("Order-based recon: no SPX positions found")
            return None

        # Build position map: symbol -> net quantity
        pos_map = {}
        for p in positions:
            sym = p.get('instrument', {}).get('symbol', '')
            long_qty = p.get('longQuantity', 0)
            short_qty = p.get('shortQuantity', 0)
            pos_map[sym] = pos_map.get(sym, 0) + (long_qty - short_qty)

        logger.info(f"    Positions ({len(pos_map)} symbols): "
                    + ", ".join(f"{s.split()[-1] if ' ' in s else s}={q}"
                               for s, q in sorted(pos_map.items())))

        # Keep IC states where ALL symbols appear in positions
        active_states = []
        for state in open_states:
            missing = state['current_symbols'] - set(pos_map.keys())
            if not missing:
                active_states.append(state)
            else:
                logger.info(f"      Entry {state['order_id']}: filtered (missing {missing})")

        if not active_states:
            logger.info("Order-based recon: no ICs match current positions")
            return None

        logger.info(f"    {len(active_states)} IC(s) pass position filter")

        # Cross-check: expected vs actual net quantities per symbol
        expected_net = {}
        for state in active_states:
            for ld in state['leg_details']:
                sym = ld['symbol']
                inst = ld.get('instruction', '')
                if inst == 'SELL_TO_OPEN':
                    expected_net[sym] = expected_net.get(sym, 0) - state['quantity']
                elif inst == 'BUY_TO_OPEN':
                    expected_net[sym] = expected_net.get(sym, 0) + state['quantity']

        for sym, exp_qty in expected_net.items():
            actual_qty = pos_map.get(sym, 0)
            if actual_qty != exp_qty:
                logger.warning(f"    Qty mismatch {sym}: expected={exp_qty} actual={actual_qty}")

        # --- Build TrackedIronCondor objects ---
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
        for idx, state in enumerate(active_states):
            # Classify legs by instruction + option_type
            long_put = short_put = short_call = long_call = None
            for ld in state['leg_details']:
                inst = ld.get('instruction', '')
                otype = ld.get('option_type', '')
                if inst == 'SELL_TO_OPEN' and otype == 'P':
                    short_put = ld
                elif inst == 'BUY_TO_OPEN' and otype == 'P':
                    long_put = ld
                elif inst == 'SELL_TO_OPEN' and otype == 'C':
                    short_call = ld
                elif inst == 'BUY_TO_OPEN' and otype == 'C':
                    long_call = ld

            if not all([long_put, short_put, short_call, long_call]):
                logger.warning(f"    Entry {state['order_id']}: incomplete IC — "
                               f"LP={'Y' if long_put else 'N'} SP={'Y' if short_put else 'N'} "
                               f"SC={'Y' if short_call else 'N'} LC={'Y' if long_call else 'N'} "
                               f"(legs: {[(ld.get('instruction'), ld.get('option_type'), ld.get('strike')) for ld in state['leg_details']]})")
                continue

            if not (long_put['strike'] < short_put['strike'] < short_call['strike'] < long_call['strike']):
                logger.warning(f"    Entry {state['order_id']}: bad strike order "
                               f"{long_put['strike']}/{short_put['strike']}/"
                               f"{short_call['strike']}/{long_call['strike']}")
                continue

            ic_num = idx + 1
            exp_date = short_put.get('expiration', '')
            strikes = {
                'long_put': _make_leg(long_put),
                'short_put': _make_leg(short_put),
                'short_call': _make_leg(short_call),
                'long_call': _make_leg(long_call),
                'net_credit': round(state['cumulative_credit'], 2),
                'expiration': exp_date,
                'exp_key': f"{exp_date}:0",
            }

            label = (f"IC#{ic_num} ({int(long_put['strike'])}/{int(short_put['strike'])}p - "
                     f"{int(short_call['strike'])}/{int(long_call['strike'])}c)")

            tracked_ics.append(TrackedIronCondor(
                ic_id=ic_num,
                label=label,
                strikes=strikes,
                quantity=int(state['quantity']),
                fill_credit=round(state['cumulative_credit'], 2),
                portfolio_value=self.portfolio_value,
            ))

            rolls_note = f" (+{state['rolls_applied']} rolls)" if state['rolls_applied'] else ""
            logger.info(f"    {label}: {state['quantity']}x ${state['cumulative_credit']:.2f}"
                        f"{rolls_note} (order {state['order_id']})")

        if not tracked_ics:
            logger.warning("Order-based recon: no valid ICs built")
            return None

        logger.info(f"  Order-based recon: {len(tracked_ics)} IC(s) reconstructed")
        return tracked_ics

    async def _detect_tp_order_for_ic(self, ic: TrackedIronCondor) -> Tuple[Optional[str], int]:
        """
        Find a working BUY_TO_CLOSE IRON_CONDOR order matching this IC's symbols.

        Returns:
            (order_id, order_quantity) if found, (None, 0) otherwise.
        """
        try:
            orders = await self.schwab_client.get_orders_for_account(status='WORKING')
        except Exception as e:
            logger.warning(f"Failed to fetch working orders: {e}")
            return None, 0

        if not orders:
            return None, 0

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
                order_id = str(order.get('orderId', ''))
                order_qty = max(
                    (l.get('quantity', 0) for l in legs), default=0
                )
                return order_id, int(order_qty)

        return None, 0

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
    parser.add_argument('--paper', action='store_true',
                        help='Paper/dry-run mode (default: live)')
    parser.add_argument('--skip-wait', action='store_true',
                        help='Testing: skip time-based waiting')
    parser.add_argument('--skip-ml', action='store_true',
                        help='Testing: skip ML/ThetaData')
    parser.add_argument('--strikes', type=str, default=None,
                        help='Manual short strikes as "PUT,CALL" (e.g. "6750,6920"). Bypasses delta selection.')
    parser.add_argument('--qty', type=int, default=None,
                        help='Contracts per S2 position (e.g. --qty 1)')
    parser.add_argument('--max-positions', type=int, default=None,
                        help='Max S2 positions to open (e.g. --max-positions 4)')
    parser.add_argument('--env-file', type=str, default=None,
                        help='Path to alternate .env file (e.g. .env.account2) for multi-account support')
    parser.add_argument('--broker', type=str, default=None, choices=['schwab', 'tradier'],
                        help='Override broker (default: from .env BROKER setting)')
    parser.add_argument('--confirm-live', action='store_true',
                        help='Skip interactive YES confirmation for live mode (for automated launchers)')

    args = parser.parse_args()

    # Reload settings from alternate .env file if specified
    if args.env_file:
        from config.settings import Settings
        import config.settings as settings_module
        env_path = Path(args.env_file)
        if not env_path.exists():
            print(f"Error: env file not found: {env_path}")
            return
        new_settings = Settings(_env_file=str(env_path))
        settings_module.settings = new_settings
        # Re-import so our local `settings` ref points to the new object
        globals()['settings'] = new_settings
        # Propagate new settings to all modules that cached the import
        import execution.order_manager.ic_order_builder as ic_mod
        ic_mod.settings = new_settings
        # Broker-specific propagation
        if new_settings.BROKER.lower() == "schwab":
            import execution.broker_api.schwab_client as schwab_mod
            import execution.broker_api.auth_manager as auth_mgr_mod
            schwab_mod.settings = new_settings
            auth_mgr_mod.settings = new_settings
            from execution.broker_api.auth_manager import AuthManager
            new_auth = AuthManager(
                api_key=new_settings.SCHWAB_API_KEY,
                app_secret=new_settings.SCHWAB_API_SECRET,
                token_path=new_settings.SCHWAB_TOKEN_PATH,
            )
            auth_mgr_mod.auth_manager = new_auth
        else:
            import execution.broker_api.tradier_client as tradier_mod
            tradier_mod.settings = new_settings
        broker_name = new_settings.BROKER.upper()
        acct_id = (new_settings.SCHWAB_ACCOUNT_ID if new_settings.BROKER.lower() == "schwab"
                    else new_settings.TRADIER_ACCOUNT_ID)
        print(f"Loaded settings from {env_path} (broker: {broker_name}, account: {acct_id})")

    # Apply CLI overrides to settings
    if args.broker is not None:
        settings.BROKER = args.broker
    if args.qty is not None:
        settings.BOT_ZDOM_QTY_PER_POSITION = args.qty
    if args.max_positions is not None:
        settings.BOT_ZDOM_MAX_POSITIONS = args.max_positions

    # Parse manual strikes
    manual_strikes = None
    if args.strikes:
        try:
            parts = args.strikes.split(',')
            manual_strikes = (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            print(f"Invalid --strikes format: {args.strikes}. Use 'PUT,CALL' e.g. '6750,6920'")
            return

    # Resolve broker and account info for display
    broker_name = settings.BROKER.upper()
    if settings.BROKER.lower() == "tradier":
        acct_id = settings.TRADIER_ACCOUNT_ID
        sandbox_mode = settings.TRADIER_SANDBOX
        if args.paper:
            settings.TRADIER_SANDBOX = True
            sandbox_mode = True
    else:
        acct_id = settings.SCHWAB_ACCOUNT_ID
        sandbox_mode = False

    # Safety confirmation for live mode (default)
    if not args.paper:
        if args.confirm_live:
            print(
                f"\n*** PERSISTENT LIVE TRADING MODE ({broker_name}) ***\n"
                f"Broker: {broker_name} | Account: {acct_id}\n"
                f"Auto-confirmed via --confirm-live flag.\n"
            )
        else:
            confirm = input(
                f"\n*** PERSISTENT LIVE TRADING MODE ({broker_name}) ***\n"
                "This bot will run 24/7: trade, sleep overnight, wake for next day.\n"
                "It will place REAL orders with REAL money.\n"
                f"Broker: {broker_name} | Account: {acct_id}\n"
                f"Symbol: {settings.BOT_SYMBOL}\n"
                f"Max contracts: {settings.BOT_MAX_CONTRACTS}\n"
                f"Daily limits: +{settings.BOT_DAILY_GAIN_LIMIT_PCT:.0%} gain / "
                f"-{settings.BOT_DAILY_LOSS_LIMIT_PCT:.0%} loss\n"
                f"Portfolio value: live from broker (fallback: ${settings.BOT_PORTFOLIO_VALUE:,.0f})\n"
                "\nType 'YES' to confirm: "
            )
            if confirm != 'YES':
                print("Aborted.")
                return

    # Configure logging (rotation handles multi-day persistence)
    logger.remove()
    acct_suffix = f"_{acct_id}" if (args.env_file or args.broker) else ""
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
    log_path = Path("logs") / f"zero_dte_bot{acct_suffix}.log"
    log_path.parent.mkdir(exist_ok=True)
    logger.add(str(log_path), level="DEBUG", rotation="10 MB", retention="30 days")

    bot = ZeroDTEBot(
        dry_run=args.paper,
        skip_wait=args.skip_wait,
        skip_ml=args.skip_ml,
        manual_strikes=manual_strikes,
    )

    asyncio.run(bot.run())


if __name__ == '__main__':
    main()
