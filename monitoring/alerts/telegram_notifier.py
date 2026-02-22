"""
Telegram notification channel.

Sends notifications via Telegram bot.
"""

from typing import Optional
from loguru import logger

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

from config.settings import settings


class TelegramNotifier:
    """
    Telegram notification channel.

    Sends messages via Telegram bot to configured chat.
    """

    def __init__(self):
        """Initialize Telegram notifier."""
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot not installed. "
                "Install with: pip install python-telegram-bot"
            )

        if not settings.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN not configured")

        if not settings.TELEGRAM_CHAT_ID:
            raise ValueError("TELEGRAM_CHAT_ID not configured")

        self.bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        self.chat_id = settings.TELEGRAM_CHAT_ID

        logger.info("Telegram notifier initialized")

    async def send(self, message: str):
        """
        Send message via Telegram.

        Args:
            message: Message text (supports Markdown)
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )

            logger.debug(f"Sent Telegram notification: {message[:50]}...")

        except TelegramError as e:
            logger.error(f"Telegram send failed: {e}")
            raise

    async def test_connection(self) -> bool:
        """
        Test Telegram connection.

        Returns:
            True if connection successful
        """
        try:
            await self.send("✅ Telegram connection test successful")
            return True
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
