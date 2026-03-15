"""
Base feature extractor interface.

All feature extractors inherit from this base class to ensure
consistent API.
"""

from abc import ABC, abstractmethod
from typing import Dict


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors."""

    @abstractmethod
    async def extract(self, *args, **kwargs) -> Dict[str, float]:
        """
        Extract features.

        Returns:
            Dict mapping feature names to values
        """
        pass
