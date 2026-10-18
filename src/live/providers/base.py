"""
ChronosGuard: Base Interface for Real-World Live Data Providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

class BaseLiveProvider(ABC):
    """
    Abstract interface for streaming real-world data sources.
    Handles baseline initialization, real-time live ingestion, and ground-truth resolution.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this stream provider (e.g., 'crypto_btc', 'weather_grid')"""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name"""
        pass

    @property
    @abstractmethod
    def feature_columns(self) -> List[str]:
        """List of feature names feeding into the model"""
        pass

    @property
    @abstractmethod
    def target_column(self) -> str:
        """Binary target label (0 or 1)"""
        pass

    @property
    @abstractmethod
    def cost_parameters(self) -> Dict[str, float]:
        """Domain financial cost matrix (cost_fn, cost_fp, retrain_expense)"""
        pass

    @abstractmethod
    def fetch_baseline_data(self, n_samples: int = 500) -> pd.DataFrame:
        """
        Fetches historical reference dataset to train initial calibrated baseline model T0.
        """
        pass

    @abstractmethod
    def fetch_live_tick(self) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Fetches the single latest live record from the real-world stream.
        Returns:
            (features_dict, raw_metadata_dict)
        """
        pass

    @abstractmethod
    def resolve_delayed_ground_truth(
        self,
        tick_id: int,
        features: Dict[str, float],
        age_seconds: float
    ) -> Optional[int]:
        """
        Checks whether enough real time has elapsed to compute the ground-truth label (0 or 1).
        Returns None if label is still pending, or 0/1 once resolved.
        """
        pass
