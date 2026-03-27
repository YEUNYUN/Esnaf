"""Tests for technical indicator computation."""

from __future__ import annotations

import random
import time

from src.agent.state import MarketSnapshot
from src.data.indicators import compute_indicators


def _generate_ohlcv(n: int = 100, base_price: float = 50000.0) -> list[list[float]]:
    """Generate fake OHLCV data for testing."""
    candles = []
    price = base_price
    now = time.time() * 1000

    for i in range(n):
        change = random.uniform(-0.02, 0.02)
        o = price
        c = price * (1 + change)
        h = max(o, c) * (1 + random.uniform(0, 0.01))
        low = min(o, c) * (1 - random.uniform(0, 0.01))
        v = random.uniform(100, 1000)
        candles.append([now + i * 900000, o, h, low, c, v])
        price = c

    return candles


class TestIndicatorComputation:
    def test_compute_with_sufficient_data(self) -> None:
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            price=50000.0,
            ohlcv=_generate_ohlcv(100),
        )
        result = compute_indicators(snapshot)

        assert result.rsi is not None
        assert 0 <= result.rsi <= 100
        assert result.macd is not None
        assert result.volatility is not None
        assert result.volatility >= 0

    def test_compute_with_insufficient_data(self) -> None:
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            price=50000.0,
            ohlcv=_generate_ohlcv(10),
        )
        result = compute_indicators(snapshot)
        # Should not crash, indicators remain None
        assert result.rsi is None

    def test_compute_with_empty_data(self) -> None:
        snapshot = MarketSnapshot(symbol="BTC/USDT", price=50000.0)
        result = compute_indicators(snapshot)
        assert result.rsi is None
        assert result.macd is None

    def test_bbands_computed(self) -> None:
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            price=50000.0,
            ohlcv=_generate_ohlcv(100),
        )
        result = compute_indicators(snapshot)
        assert result.bbands_upper is not None
        assert result.bbands_lower is not None
        assert result.bbands_upper > result.bbands_lower

    def test_volume_sma_ratio(self) -> None:
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            price=50000.0,
            ohlcv=_generate_ohlcv(100),
        )
        result = compute_indicators(snapshot)
        assert result.volume_sma_ratio is not None
        assert result.volume_sma_ratio > 0
