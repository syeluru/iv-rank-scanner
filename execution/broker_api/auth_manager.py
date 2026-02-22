"""
Schwab API authentication and token management.

This module handles OAuth authentication flow and token refresh for Schwab API.
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from loguru import logger

try:
    import schwab
    from schwab import auth
    from schwab.client import Client as SchwabAPIClient
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False
    SchwabAPIClient = None
    logger.warning("schwab-py not installed. Install with: pip install schwab-py")

from config.settings import settings


class AuthManager:
    """
    Manages Schwab API authentication and token lifecycle.

    Handles initial OAuth flow, token storage, and automatic token refresh.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        token_path: Optional[Path] = None
    ):
        """
        Initialize authentication manager.

        Args:
            api_key: Schwab API key (defaults to settings)
            app_secret: Schwab app secret (defaults to settings)
            token_path: Path to store token (defaults to settings)
        """
        if not SCHWAB_AVAILABLE:
            raise ImportError(
                "schwab-py is required for Schwab API access. "
                "Install with: pip install schwab-py"
            )

        self.api_key = api_key or settings.SCHWAB_API_KEY
        self.app_secret = app_secret or settings.SCHWAB_API_SECRET
        self.token_path = token_path or settings.SCHWAB_TOKEN_PATH

        if not self.api_key or not self.app_secret:
            raise ValueError(
                "Schwab API credentials not configured. "
                "Set SCHWAB_API_KEY and SCHWAB_API_SECRET environment variables."
            )

        logger.info(f"AuthManager initialized. Token path: {self.token_path}")

    def authenticate_interactive(self, redirect_uri: str = "https://127.0.0.1:8182") -> None:
        """
        Perform interactive OAuth authentication flow.

        This opens a browser for user to log in and authorize the app.
        Run this once to obtain initial tokens.

        Args:
            redirect_uri: OAuth redirect URI (must match app settings in Schwab Developer Portal)
        """
        logger.info("Starting interactive OAuth flow...")
        logger.info(f"Redirect URI: {redirect_uri}")

        try:
            # Use schwab-py's easy authentication helper
            client = auth.easy_client(
                api_key=self.api_key,
                app_secret=self.app_secret,
                callback_url=redirect_uri,
                token_path=str(self.token_path)
            )

            logger.info("✓ Authentication successful!")
            logger.info(f"✓ Token saved to: {self.token_path}")

            return client

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def get_client(self) -> SchwabAPIClient:
        """
        Get authenticated Schwab client.

        Loads token from file and creates client. The client will
        automatically refresh the token when needed.

        Returns:
            Authenticated Schwab client

        Raises:
            FileNotFoundError: If token file doesn't exist
            ValueError: If token is invalid
        """
        if not self.token_path.exists():
            raise FileNotFoundError(
                f"Token file not found: {self.token_path}\n"
                "Run setup_schwab_auth.py first to authenticate."
            )

        try:
            # Create client from stored token
            client = auth.client_from_token_file(
                token_path=str(self.token_path),
                api_key=self.api_key,
                app_secret=self.app_secret
            )

            logger.info("✓ Schwab client created from token file")
            return client

        except Exception as e:
            logger.error(f"Failed to create client from token: {e}")
            raise

    def is_token_valid(self) -> bool:
        """
        Check if stored token exists and is valid.

        Returns:
            True if token exists and appears valid
        """
        if not self.token_path.exists():
            return False

        try:
            with open(self.token_path, 'r') as f:
                token_data = json.load(f)

            # Check if token has required fields
            required_fields = ['access_token', 'refresh_token', 'expires_at']
            if not all(field in token_data for field in required_fields):
                logger.warning("Token missing required fields")
                return False

            # Check if token is expired (with 5 minute buffer)
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if expires_at < datetime.now() + timedelta(minutes=5):
                logger.warning("Token is expired or expiring soon")
                return False

            logger.info("Token is valid")
            return True

        except Exception as e:
            logger.error(f"Error checking token validity: {e}")
            return False

    def token_info(self) -> Optional[dict]:
        """
        Get information about stored token.

        Returns:
            Dictionary with token info, or None if no token exists
        """
        if not self.token_path.exists():
            return None

        try:
            with open(self.token_path, 'r') as f:
                token_data = json.load(f)

            expires_at = datetime.fromisoformat(token_data.get('expires_at', ''))
            time_remaining = expires_at - datetime.now()

            return {
                'exists': True,
                'valid': self.is_token_valid(),
                'expires_at': expires_at.isoformat(),
                'time_remaining': str(time_remaining),
                'has_refresh_token': 'refresh_token' in token_data
            }

        except Exception as e:
            logger.error(f"Error reading token info: {e}")
            return {'exists': True, 'valid': False, 'error': str(e)}


# Global auth manager instance
auth_manager = AuthManager()
