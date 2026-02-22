#!/usr/bin/env python3
"""Interactive Schwab OAuth authentication. Run to refresh tokens."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from schwab import auth

client = auth.easy_client(
    api_key=settings.SCHWAB_API_KEY,
    app_secret=settings.SCHWAB_API_SECRET,
    callback_url="https://127.0.0.1:8182",
    token_path=str(settings.SCHWAB_TOKEN_PATH),
)
print(f"Token saved to {settings.SCHWAB_TOKEN_PATH}")
