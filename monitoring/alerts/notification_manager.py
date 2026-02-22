"""
Notification manager for trading signals.

Manages notifications across multiple channels when signals are generated.
"""

from typing import List
from loguru import logger

from strategies.base_strategy import Signal
from config.settings import settings


class NotificationManager:
    """
    Manages notifications across multiple channels.

    Sends alerts when ALL strategy conditions are met.
    """

    def __init__(self):
        """Initialize notification manager."""
        self.enabled_channels = []
        self._initialize_channels()

    def _initialize_channels(self):
        """Initialize enabled notification channels."""
        # Telegram (optional)
        if settings.ENABLE_TELEGRAM_ALERTS and settings.TELEGRAM_BOT_TOKEN:
            try:
                from monitoring.alerts.telegram_notifier import TelegramNotifier
                self.enabled_channels.append(TelegramNotifier())
                logger.info("Telegram notifications enabled")
            except ImportError:
                logger.warning("Telegram support not available (python-telegram-bot not installed)")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram notifier: {e}")

        if not self.enabled_channels:
            logger.info("No external notification channels enabled (terminal only)")

    async def notify_signal(self, strategy_name: str, signals: List[Signal]):
        """
        Send notification for new signals.

        Args:
            strategy_name: Name of strategy that generated signal
            signals: List of Signal objects
        """
        if not settings.NOTIFY_ON_SIGNAL:
            return

        for signal in signals:
            # Check confidence threshold
            if signal.confidence < settings.NOTIFY_MIN_CONFIDENCE:
                logger.debug(
                    f"Skipping notification for {signal.symbol}: "
                    f"confidence {signal.confidence:.2f} < {settings.NOTIFY_MIN_CONFIDENCE:.2f}"
                )
                continue

            # Format message
            message = self._format_signal_message(strategy_name, signal)

            # Terminal beep (always enabled)
            print('\a', end='', flush=True)

            # Log notification
            logger.info(f"📢 NOTIFICATION: {message}")

            # Send to all enabled channels
            for channel in self.enabled_channels:
                try:
                    await channel.send(message)
                except Exception as e:
                    logger.error(f"Notification failed on {channel.__class__.__name__}: {e}")

    def _format_signal_message(self, strategy: str, signal: Signal) -> str:
        """
        Format signal for notification.

        Args:
            strategy: Strategy name
            signal: Signal object

        Returns:
            Formatted message string
        """
        # Build message
        lines = [
            "🔔 *ALL CONDITIONS MET*",
            "",
            f"*Strategy:* {strategy.replace('_', ' ').title()}",
            f"*Symbol:* {signal.symbol}",
            f"*Direction:* {signal.direction.value}",
            f"*Entry:* ${signal.entry_price:.2f}",
        ]

        if signal.stop_loss:
            lines.append(f"*Stop Loss:* ${signal.stop_loss:.2f}")

        if signal.confidence:
            lines.append(f"*Confidence:* {signal.confidence:.0%}")

        if signal.notes:
            lines.append("")
            lines.append(f"_{signal.notes}_")

        # Add metadata if available
        if signal.metadata:
            lines.append("")
            lines.append("*Details:*")

            for key, value in signal.metadata.items():
                if isinstance(value, float):
                    lines.append(f"  • {key}: {value:.2f}")
                else:
                    lines.append(f"  • {key}: {value}")

        return "\n".join(lines)

    async def notify_error(self, error_message: str):
        """
        Send error notification.

        Args:
            error_message: Error description
        """
        message = f"⚠️ *ERROR*\n\n{error_message}"

        logger.error(f"Error notification: {error_message}")

        for channel in self.enabled_channels:
            try:
                await channel.send(message)
            except Exception as e:
                logger.error(f"Failed to send error notification: {e}")

    async def notify_system_status(self, status: str):
        """
        Send system status notification.

        Args:
            status: Status message
        """
        message = f"ℹ️ *SYSTEM STATUS*\n\n{status}"

        logger.info(f"Status notification: {status}")

        for channel in self.enabled_channels:
            try:
                await channel.send(message)
            except Exception as e:
                logger.error(f"Failed to send status notification: {e}")
