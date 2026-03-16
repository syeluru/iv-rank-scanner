"""
Central configuration management using Pydantic settings.

This module provides a single source of truth for all application settings,
including API credentials, trading parameters, and strategy configurations.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application-wide settings with environment variable support."""

    # ===== Schwab API Configuration =====
    SCHWAB_API_KEY: str = ""
    SCHWAB_API_SECRET: str = ""
    SCHWAB_ACCOUNT_ID: str = ""
    SCHWAB_TOKEN_PATH: Path = Path.home() / ".schwab_token.json"

    # ===== Interactive Brokers Configuration =====
    IBKR_HOST: str = "127.0.0.1"
    IBKR_PORT: int = 7497  # 7497=paper, 7496=live, 4002=gateway paper
    IBKR_CLIENT_ID: int = 1
    IBKR_ACCOUNT_ID: str = ""
    IBKR_ENABLE_STREAMING: bool = True

    # ===== Alpaca Configuration =====
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_PAPER_TRADING: bool = True  # Use paper trading endpoint
    ALPACA_ENABLE_STREAMING: bool = True

    # ===== Tradier Configuration =====
    TRADIER_API_TOKEN: str = ""
    TRADIER_ACCOUNT_ID: str = ""
    TRADIER_SANDBOX: bool = True  # Use sandbox endpoint for paper trading
    TRADIER_RATE_LIMIT_CALLS: int = 120  # 120/min production, 60/min sandbox
    TRADIER_RATE_LIMIT_PERIOD: int = 60

    # ===== Broker Selection =====
    # Choose broker: 'schwab' or 'tradier'
    BROKER: str = "schwab"

    # ===== ThetaData Configuration =====
    THETADATA_USERNAME: str = ""
    THETADATA_PASSWORD: str = ""

    # ===== Market Data Provider =====
    # Choose provider: 'ibkr' or 'alpaca'
    # Alpaca recommended for free market data access
    MARKET_DATA_PROVIDER: str = "alpaca"

    # ===== Trading Mode =====
    PAPER_TRADING: bool = False  # Overridden by bot's --paper flag

    # ===== Manual Approval System =====
    ENABLE_MANUAL_APPROVAL: bool = True  # Toggle approval system on/off
    APPROVAL_TIMEOUT_SECONDS: int = 300  # 5 minutes
    APPROVAL_AUTO_REJECT_EXPIRED: bool = True  # Reject vs keep expired signals

    # ===== Terminal Display =====
    TERMINAL_REFRESH_RATE: float = 0.5  # Seconds between UI updates
    TERMINAL_THEME: str = "dark"  # UI color scheme

    # ===== Risk Management =====
    MAX_POSITION_SIZE: float = 0.10  # Maximum 10% of portfolio per position
    MAX_POSITIONS: int = 10  # Maximum number of concurrent positions
    DAILY_LOSS_LIMIT: float = 0.03  # Stop trading if daily loss exceeds 3%
    MAX_PORTFOLIO_LEVERAGE: float = 1.0  # No leverage by default

    # ===== Wheel Strategy Parameters =====
    WHEEL_MIN_IV_RANK: float = 50.0  # Minimum IV Rank for entry (0-100 scale)
    WHEEL_PUT_DELTA: float = 0.25  # Target delta for cash-secured puts
    WHEEL_CALL_DELTA: float = 0.30  # Target delta for covered calls
    WHEEL_MIN_DTE: int = 30  # Minimum days to expiration for entry
    WHEEL_ROLL_DTE: int = 21  # Roll at this many DTE
    WHEEL_PROFIT_TARGET: float = 0.50  # Close at 50% profit
    WHEEL_EARNINGS_BUFFER_DAYS: int = 14  # Avoid earnings within this window

    # ===== Data Collection =====
    DATA_STORE_PATH: Path = Path("data_store")
    MARKET_DATA_DB: str = "market_data.duckdb"
    TRADES_DB: str = "trades.duckdb"
    POSITIONS_DB: str = "positions.duckdb"
    IV_HISTORY_LOOKBACK_DAYS: int = 252  # 1 year for IV Rank calculation

    # ===== ML Model Configuration =====
    ML_ENABLED: bool = True
    ML_MODEL_PATH: Path = Path("ml/artifacts/models/lightgbm_v3.joblib")
    ML_MIN_WIN_PROBABILITY: float = 0.65  # Entry threshold
    ML_FEATURES_CACHE_SECONDS: int = 300  # Cache features for 5 minutes

    # ===== Data Sources for ML =====
    ML_DATA_DB: str = "ml_data.duckdb"
    FRED_API_KEY: str = ""  # For macro data (optional)
    CANDLE_DATA_PATH: Path = Path("data_store/parquet")

    # ===== Live Trading Configuration =====
    LIVE_TRADING_ENABLED: bool = False  # Enable automated trading
    LIVE_ENTRY_TIME: str = "11:30"  # HH:MM format
    LIVE_INITIAL_CONTRACTS: int = 10  # Initial position size
    LIVE_SCALE_CONTRACTS: int = 5  # Scale-in size
    LIVE_MAX_CONTRACTS: int = 25  # Maximum total contracts per day
    LIVE_SCALE_INTERVAL_MINUTES: int = 30  # Minutes between scale-ins
    LIVE_CUTOFF_TIME: str = "14:00"  # No entries after this time (HH:MM)
    LIVE_MIN_CONFIDENCE: float = 0.80  # Minimum ML confidence for entry
    LIVE_MAX_LOSS_PER_DAY: float = 5000.0  # Maximum daily loss ($)
    LIVE_PREFERRED_VIX_MIN: float = 15.0  # Preferred VIX minimum
    LIVE_PREFERRED_VIX_MAX: float = 25.0  # Preferred VIX maximum
    LIVE_AUTO_APPROVE: bool = False  # Auto-approve signals (use with caution!)

    # ===== 0DTE Bot Configuration =====
    BOT_PORTFOLIO_VALUE: float = 28000.0  # Portfolio value for sizing
    BOT_DAILY_TARGET_PCT: float = 0.02  # 2% daily profit target
    BOT_TAKE_PROFIT_PCT: float = 0.25  # Close at 25% of credit received
    BOT_STOP_LOSS_PCT: float = 0.10  # Max loss per trade as % of portfolio (10% = $8,700 on $87K)
    BOT_DAILY_GAIN_LIMIT_PCT: float = 0.05  # 5% daily gain → stop trading for day
    BOT_DAILY_LOSS_LIMIT_PCT: float = 0.10  # 10% daily loss → stop trading for day
    BOT_ENTRY_TIME: str = "12:00"  # HH:MM ET entry time
    BOT_EXIT_TIME: str = "15:00"  # HH:MM ET forced close time
    BOT_WAKE_TIME: str = "10:20"  # HH:MM ET pre-flight check time (before V4_ENTRY_START)
    BOT_SHORT_DELTA: float = 0.10  # Target delta for short strikes
    BOT_WING_WIDTH: float = 25.0  # Wing width in dollars
    BOT_SYMBOL: str = "$SPX"  # Underlying symbol
    BOT_COST_PER_LEG: float = 0.65  # Transaction cost per leg per contract
    # v3 model: confidence = 1 - P(big_loss). 0.50 = skip if >50% risk.
    # v2 model: confidence = P(win). 0.80 = trade only if >80% win probability.
    BOT_MIN_CONFIDENCE: float = 0.50  # Minimum ML confidence for entry
    BOT_MONITOR_INTERVAL: int = 5  # Seconds between monitoring checks
    BOT_ENTRY_POLL_INTERVAL: int = 30  # Seconds between entry fill polls
    BOT_ENTRY_TIMEOUT: int = 300  # Seconds to wait for entry fill
    BOT_MIN_CREDIT: float = 1.00  # Minimum net credit to enter trade
    BOT_MAX_CONTRACTS: int = 50  # Hard cap on contracts

    # v4 Entry Timing
    BOT_V4_ENABLED: bool = True               # Enable v4 dynamic entry (False = fixed-time entry)
    BOT_V4_ENTRY_START: str = "10:30"         # Start scanning for entry (HH:MM ET)
    BOT_V4_ENTRY_END: str = "14:00"           # Stop scanning (HH:MM ET)
    BOT_V4_SCAN_INTERVAL: int = 5             # Minutes between scans
    BOT_V4_MIN_CONFIDENCE: float = 0.55       # v4 P(profitable) threshold for entry

    # v5 Take-Profit Models
    BOT_V5_ENABLED: bool = True                # Enable v5 TP-based models (falls back to v4 if False)
    BOT_V5_TP25_THRESHOLD: float = 0.635       # Min P(hit 25% TP) to enter
    BOT_V5_TP50_THRESHOLD: float = 0.60        # Min P(hit 50% TP) — informational/future
    BOT_FOMC_GATE_ENABLED: bool = True         # Enable FOMC-day awareness
    BOT_FOMC_SKIP_DAY: bool = False            # If True, skip FOMC announcement days entirely
    BOT_V5_MULTI_DELTA: bool = True            # Score all loaded delta models, pick best for entry

    # v6 Ensemble Models
    BOT_V6_ENABLED: bool = True                # Enable v6 ensemble (falls back to v5/v4 if False)
    BOT_V6_TP25_THRESHOLD: float = 0.58        # Min ensemble P(hit 25% TP) to enter
    BOT_V6_TP50_THRESHOLD: float = 0.62        # Min ensemble P(hit 50% TP) to enter
    BOT_V6_MIN_CONSENSUS: int = 4              # Require ≥4/6 models >= 0.5

    # Unified Strategy Selection
    BOT_UNIFIED_STRATEGY: bool = True          # Enable EV-based multi-delta strategy selection
    BOT_MIN_EV_THRESHOLD: float = 0.10         # Min EV per contract ($) to enter a strategy

    # v8 XGBoost Models (200+ features, replaces v5/v7 when available)
    BOT_V8_ENABLED: bool = True                # Enable v8 models (falls back to v5/v7 if unavailable)

    # v7 Long IC Models
    BOT_V7_ENABLED: bool = True                # Enable v7 long IC decision system
    BOT_V7_THRESHOLD: float = 0.50             # Min P(long IC profitable) to enter long IC
    BOT_V7_PREFERRED_DELTA: float = 0.20       # Preferred delta for long ICs (highest gamma)
    BOT_LONG_IC_MAX_RISK_PCT: float = 0.02     # Max portfolio risk per long IC trade (2%)
    BOT_LONG_IC_TP_PCT: float = 0.50           # Take profit at 50% of debit as profit
    BOT_LONG_IC_SL_PCT: float = 0.40           # Stop loss at 40% loss of debit
    BOT_LONG_IC_ENTRY_START: str = "11:00"     # Long IC entry window start
    BOT_LONG_IC_ENTRY_END: str = "12:30"       # Long IC entry window end

    # Credit-based SLTP (trailing profit lock on debit)
    BOT_SLTP_ACTIVATE_PCT: float = 0.30   # SLTP activates at 30% profit (debit = credit * 0.70)
    BOT_SLTP_LOCK_OFFSET_PCT: float = 0.05  # Lock-in is 5% below activation (30% → lock 25%)
    BOT_SLTP_STEP_PCT: float = 0.05       # Trail up every 5% profit increment

    # ===== API Rate Limiting =====
    SCHWAB_RATE_LIMIT_CALLS: int = 120  # Schwab allows 120 calls per minute
    SCHWAB_RATE_LIMIT_PERIOD: int = 60  # Period in seconds

    # ===== Continuous Monitoring =====
    ENABLE_CONTINUOUS_MONITORING: bool = True
    MONITORING_INTERVAL_SECONDS: float = 5.0
    MONITORING_LOOKBACK_BARS: int = 200

    # ===== Technical Indicators =====
    INDICATOR_RSI_PERIOD: int = 14
    INDICATOR_EMA_FAST: int = 9
    INDICATOR_EMA_SLOW: int = 21
    INDICATOR_MACD_FAST: int = 12
    INDICATOR_MACD_SLOW: int = 26
    INDICATOR_MACD_SIGNAL: int = 9

    # ===== Strategy Configuration =====
    # Opening Range Breakout
    ORB_OPENING_RANGE_MINUTES: int = 60
    ORB_VOLUME_THRESHOLD: float = 1.5
    ORB_MIN_ATR: float = 1.0

    # VWAP Breakout + Pullback
    VWAP_PULLBACK_THRESHOLD: float = 0.02  # 2%
    VWAP_MIN_VOLUME_RATIO: float = 1.5

    # RSI Divergence Reversal
    RSI_DIVERGENCE_THRESHOLD: float = 5.0
    RSI_OVERSOLD: float = 30
    RSI_OVERBOUGHT: float = 70

    # ===== Monitoring & Alerts =====
    NOTIFY_ON_SIGNAL: bool = True
    NOTIFY_MIN_CONFIDENCE: float = 0.70
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    ENABLE_TELEGRAM_ALERTS: bool = False

    # ===== Logging =====
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: Path = Path("logs/trading_system.log")

    # ===== Watchlist =====
    # OPTION 2: Reduced watchlist for paper trading (symbols with free/delayed data)
    # Uncomment to use limited list while testing
    # DEFAULT_SYMBOLS: list[str] = ["SPY", "QQQ", "IWM"]  # Major ETFs only

    DEFAULT_SYMBOLS: list[str] = [
        # Large Cap Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # Major Indices ETFs
        "SPY", "QQQ", "IWM", "DIA",
        # Other High IV Candidates
        "AMD", "NFLX", "BABA", "PLTR", "COIN", "RIOT",
        "BA", "DIS", "UBER", "LYFT", "SHOP", "SQ",
        "MRNA", "PFE", "JNJ", "XOM", "CVX",
        # Semiconductors
        "INTC", "MU", "QCOM", "AVGO",
        # Finance
        "JPM", "BAC", "GS", "MS", "C",
        # Consumer
        "NKE", "SBUX", "MCD", "CMG", "HD", "LOW",
        # Energy
        "USO", "XLE", "OXY", "COP",
        # Volatility
        "UVXY", "VXX",
        # Additional high-liquidity names
        "PYPL", "ADBE", "CRM", "ORCL", "IBM", "T", "VZ"
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directories exist
        self.DATA_STORE_PATH.mkdir(parents=True, exist_ok=True)
        self.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    @property
    def market_data_db_path(self) -> Path:
        """Full path to market data database."""
        return self.DATA_STORE_PATH / self.MARKET_DATA_DB

    @property
    def trades_db_path(self) -> Path:
        """Full path to trades database."""
        return self.DATA_STORE_PATH / self.TRADES_DB

    @property
    def positions_db_path(self) -> Path:
        """Full path to positions database."""
        return self.DATA_STORE_PATH / self.POSITIONS_DB

    @property
    def ml_data_db_path(self) -> Path:
        """Full path to ML data database."""
        return self.DATA_STORE_PATH / self.ML_DATA_DB

    def validate_schwab_credentials(self) -> bool:
        """Check if Schwab API credentials are configured."""
        return bool(
            self.SCHWAB_API_KEY
            and self.SCHWAB_API_SECRET
            and self.SCHWAB_ACCOUNT_ID
        )

    def validate_tradier_credentials(self) -> bool:
        """Check if Tradier API credentials are configured."""
        return bool(self.TRADIER_API_TOKEN and self.TRADIER_ACCOUNT_ID)

    def validate_alpaca_credentials(self) -> bool:
        """Check if Alpaca API credentials are configured."""
        return bool(
            self.ALPACA_API_KEY
            and self.ALPACA_SECRET_KEY
        )


# Global settings instance
settings = Settings()
