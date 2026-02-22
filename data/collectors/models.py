"""
Data models for ThetaData API responses.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List


@dataclass
class OptionQuote:
    """Single option quote with Greeks."""
    symbol: str
    underlying: str
    expiration: date
    strike: float
    option_type: str  # 'CALL' or 'PUT'
    bid: float
    ask: float
    last: float
    mid: float
    volume: int
    open_interest: int
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    iv: Optional[float] = None
    underlying_price: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class OptionsChain:
    """Full options chain for an expiration."""
    underlying: str
    expiration: date
    underlying_price: float
    options: List[OptionQuote]
    timestamp: datetime

    def __len__(self):
        return len(self.options)

    def __iter__(self):
        return iter(self.options)

    def __getitem__(self, index):
        return self.options[index]


@dataclass
class Settlement:
    """Settlement price for an option or underlying."""
    symbol: str
    date: date
    settlement_price: float
    underlying_price: Optional[float] = None
