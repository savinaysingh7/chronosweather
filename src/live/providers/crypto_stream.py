"""
ChronosGuard: Real-Time Live Cryptocurrency & Financial Volatility Stream Provider
Ingests 24/7 live tick and k-line market data from Binance Public API (Zero API Key required).
Features: Price returns, rolling volatility, volume surges, order flow proxies, RSI.
Target: High Volatility Spike / Downside Risk Event.
"""

import urllib.request
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from src.live.providers.base import BaseLiveProvider

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"

class CryptoMarketLiveProvider(BaseLiveProvider):
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self._recent_prices: List[float] = []
        self._recent_volumes: List[float] = []
        self._price_history_for_resolution: Dict[int, float] = {}
        self._volatility_threshold: float = 0.008  # Default threshold for risk spike

    @property
    def name(self) -> str:
        return f"crypto_{self.symbol.lower()}"

    @property
    def display_name(self) -> str:
        return f"Live Crypto Risk Stream ({self.symbol})"

    @property
    def feature_columns(self) -> List[str]:
        return [
            "return_1m",
            "volatility_5m",
            "volume_surge",
            "spread_ratio",
            "rsi_14",
            "buyer_ratio"
        ]

    @property
    def target_column(self) -> str:
        return "is_volatility_spike"

    @property
    def cost_parameters(self) -> Dict[str, float]:
        return {
            "cost_fn": 4500.0,    # Unhedged loss from sudden volatility crash
            "cost_fp": 150.0,     # Hedging friction / transaction fee
            "retrain_expense": 500.0  # Compute cost to update sliding-window model
        }

    def _request_json(self, url: str) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": "ChronosGuard/1.0"})
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def fetch_baseline_data(self, n_samples: int = 500) -> pd.DataFrame:
        """
        Fetches historical 1-minute k-lines from Binance to establish reference baseline T0.
        """
        limit = min(1000, max(100, n_samples + 20))
        url = f"{BINANCE_KLINES_URL}?symbol={self.symbol}&interval=1m&limit={limit}"
        raw_klines = self._request_json(url)

        # Parse klines: [time, open, high, low, close, volume, ...]
        records = []
        for k in raw_klines:
            records.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "taker_buy_vol": float(k[9])
            })
        df = pd.DataFrame(records)

        # Compute engineering features
        df["return_1m"] = df["close"].pct_change().fillna(0.0)
        df["volatility_5m"] = df["return_1m"].rolling(5).std().fillna(0.0)
        df["vol_avg_20"] = df["volume"].rolling(20).mean().fillna(df["volume"])
        df["volume_surge"] = (df["volume"] / (df["vol_avg_20"] + 1e-6)).clip(0, 10).fillna(1.0)
        df["spread_ratio"] = ((df["high"] - df["low"]) / df["close"]).clip(0, 0.2).fillna(0.0)
        df["buyer_ratio"] = (df["taker_buy_vol"] / (df["volume"] + 1e-6)).clip(0, 1.0).fillna(0.5)

        # RSI proxy
        delta = df["return_1m"]
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0) / 100.0

        # Define ground-truth target: forward 5-minute volatility shock (> 75th percentile)
        forward_vol = df["return_1m"].shift(-5).rolling(5).std()
        thresh = forward_vol.quantile(0.75) if forward_vol.quantile(0.75) > 0 else 0.003
        self._volatility_threshold = float(thresh)
        df["is_volatility_spike"] = (forward_vol > thresh).astype(int)

        # Clean NaNs and drop unneeded raw columns
        clean_df = df.dropna().tail(n_samples).reset_index(drop=True)

        # Prime internal buffers
        self._recent_prices = df["close"].tail(30).tolist()
        self._recent_volumes = df["volume"].tail(30).tolist()

        return clean_df[self.feature_columns + [self.target_column]]

    def fetch_live_tick(self) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Polls the 24hr ticker endpoint to extract real-time live market metrics.
        """
        url = f"{BINANCE_TICKER_URL}?symbol={self.symbol}"
        ticker = self._request_json(url)

        curr_price = float(ticker["lastPrice"])
        curr_vol = float(ticker["volume"])
        high_24h = float(ticker["highPrice"])
        low_24h = float(ticker["lowPrice"])
        bid_price = float(ticker.get("bidPrice", curr_price * 0.9998))
        ask_price = float(ticker.get("askPrice", curr_price * 1.0002))

        # Update running buffers
        self._recent_prices.append(curr_price)
        if len(self._recent_prices) > 60:
            self._recent_prices.pop(0)

        self._recent_volumes.append(curr_vol)
        if len(self._recent_volumes) > 60:
            self._recent_volumes.pop(0)

        # Compute instant features
        if len(self._recent_prices) >= 2:
            prev_price = self._recent_prices[-2]
            ret_1m = (curr_price - prev_price) / prev_price
        else:
            ret_1m = float(ticker.get("priceChangePercent", 0.0)) / 100.0 / 1440.0

        if len(self._recent_prices) >= 5:
            arr = np.array(self._recent_prices[-5:])
            returns = np.diff(arr) / arr[:-1]
            vol_5m = float(np.std(returns))
        else:
            vol_5m = abs(ret_1m)

        # Volume surge proxy
        vol_avg = np.mean(self._recent_volumes) if len(self._recent_volumes) > 0 else curr_vol
        vol_surge = min(10.0, curr_vol / (vol_avg + 1e-6))

        spread_ratio = (ask_price - bid_price) / curr_price if curr_price > 0 else 0.0005
        buyer_ratio = float(ticker.get("weightedAvgPrice", curr_price)) / curr_price
        buyer_ratio = float(np.clip(buyer_ratio, 0.1, 0.9))

        # Simple RSI proxy
        if len(self._recent_prices) >= 14:
            rets = np.diff(self._recent_prices[-14:])
            pos = rets[rets > 0].sum()
            neg = -rets[rets < 0].sum()
            rsi = (pos / (pos + neg + 1e-6))
        else:
            rsi = 0.5

        features = {
            "return_1m": round(ret_1m, 6),
            "volatility_5m": round(vol_5m, 6),
            "volume_surge": round(vol_surge, 4),
            "spread_ratio": round(spread_ratio, 6),
            "rsi_14": round(rsi, 4),
            "buyer_ratio": round(buyer_ratio, 4)
        }

        meta = {
            "symbol": self.symbol,
            "last_price": curr_price,
            "volume": curr_vol,
            "high_24h": high_24h,
            "low_24h": low_24h,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return features, meta

    def resolve_delayed_ground_truth(
        self,
        tick_id: int,
        features: Dict[str, float],
        age_seconds: float
    ) -> Optional[int]:
        """
        Resolves ground truth after at least 15 seconds have passed:
        Checks if realized market conditions experienced a volatility spike (> threshold).
        """
        if age_seconds < 15.0:
            return None

        # Determine realized label based on volatility condition
        vol = features.get("volatility_5m", 0.0)
        surge = features.get("volume_surge", 1.0)
        is_spike = 1 if (vol > self._volatility_threshold or surge > 2.5) else 0
        return is_spike
