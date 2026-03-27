"""Tests for MarketDataClient — async wrapping, freshness, rate-limit retry."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import ccxt
import pytest

from src.config import ExchangeConfig
from src.data.market import MarketDataClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> MarketDataClient:
    cfg = ExchangeConfig(api_key="test", secret="test", testnet=True)
    return MarketDataClient(cfg)


def _fake_ticker(**overrides):
    base = {"last": 50000.0, "quoteVolume": 1e9}
    base.update(overrides)
    return base


def _fake_ohlcv(n: int = 5, age_s: float = 60.0, interval_ms: float = 900_000):
    """Generate OHLCV list whose latest candle is *age_s* seconds old."""
    now_ms = time.time() * 1000
    latest_ts = now_ms - age_s * 1000
    return [
        [latest_ts - (n - 1 - i) * interval_ms, 100, 110, 90, 105, 500]
        for i in range(n)
    ]


def _fake_orderbook():
    return {
        "bids": [[49990.0, 1.0]],
        "asks": [[50010.0, 1.0]],
    }


async def _fake_to_thread(fn, *args, **kwargs):
    """Drop-in replacement for asyncio.to_thread that runs synchronously."""
    return fn(*args, **kwargs)


async def _fake_sleep(_secs):
    """No-op replacement for asyncio.sleep."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncWrapping:
    """Verify that blocking CCXT methods are invoked via asyncio.to_thread."""

    async def test_fetch_snapshot_uses_to_thread(self) -> None:
        client = _make_client()
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = _fake_ticker()
        exchange.fetch_ohlcv.return_value = _fake_ohlcv()
        exchange.fetch_order_book.return_value = _fake_orderbook()
        client._exchange = exchange

        with patch("src.data.market.asyncio.to_thread", new=_fake_to_thread):
            snap = await client.fetch_snapshot("BTC/USDT")

        assert snap.price == 50000.0
        assert snap.symbol == "BTC/USDT"

    async def test_fetch_price_uses_to_thread(self) -> None:
        client = _make_client()
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = _fake_ticker(last=42000.0)
        client._exchange = exchange

        with patch("src.data.market.asyncio.to_thread", new=_fake_to_thread):
            price = await client.fetch_price("ETH/USDT")

        assert price == 42000.0


class TestFreshnessValidation:
    """OHLCV staleness detection."""

    def test_fresh_data_not_stale(self) -> None:
        ohlcv = _fake_ohlcv(age_s=60)
        assert MarketDataClient._check_ohlcv_freshness(ohlcv, "15m") is False

    def test_old_data_is_stale(self) -> None:
        ohlcv = _fake_ohlcv(age_s=3600)  # 1 hour old for 15m candle
        assert MarketDataClient._check_ohlcv_freshness(ohlcv, "15m") is True

    def test_empty_ohlcv_is_stale(self) -> None:
        assert MarketDataClient._check_ohlcv_freshness([], "15m") is True

    async def test_stale_flag_set_on_snapshot(self) -> None:
        """fetch_snapshot should populate the stale flag."""
        client = _make_client()
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = _fake_ticker()
        exchange.fetch_ohlcv.return_value = _fake_ohlcv(age_s=7200)
        exchange.fetch_order_book.return_value = _fake_orderbook()
        client._exchange = exchange

        with patch("src.data.market.asyncio.to_thread", new=_fake_to_thread):
            snap = await client.fetch_snapshot("BTC/USDT")

        assert snap.stale is True

    async def test_fresh_flag_on_snapshot(self) -> None:
        client = _make_client()
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = _fake_ticker()
        exchange.fetch_ohlcv.return_value = _fake_ohlcv(age_s=30)
        exchange.fetch_order_book.return_value = _fake_orderbook()
        client._exchange = exchange

        with patch("src.data.market.asyncio.to_thread", new=_fake_to_thread):
            snap = await client.fetch_snapshot("BTC/USDT")

        assert snap.stale is False


class TestRateLimitRetry:
    """Rate-limit back-off & throttle."""

    async def test_retries_on_rate_limit(self) -> None:
        client = _make_client()
        exchange = MagicMock()

        call_count = 0

        def _fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ccxt.RateLimitExceeded("slow down")
            return _fake_ticker()

        exchange.fetch_ticker.side_effect = _fail_then_succeed
        client._exchange = exchange

        with (
            patch("src.data.market.asyncio.to_thread", new=_fake_to_thread),
            patch("src.data.market.asyncio.sleep", new=_fake_sleep),
        ):
            price = await client.fetch_price("BTC/USDT")

        assert price == 50000.0
        assert call_count == 3

    async def test_raises_after_max_retries(self) -> None:
        client = _make_client()
        exchange = MagicMock()
        exchange.fetch_ticker.side_effect = ccxt.RateLimitExceeded("nope")
        client._exchange = exchange

        with (
            pytest.raises(ccxt.RateLimitExceeded),
            patch("src.data.market.asyncio.to_thread", new=_fake_to_thread),
            patch("src.data.market.asyncio.sleep", new=_fake_sleep),
        ):
            await client.fetch_price("BTC/USDT")

    async def test_throttle_enforces_interval(self) -> None:
        client = _make_client()
        client._last_call_time = time.monotonic()  # just called

        slept: list[float] = []

        async def _record_sleep(secs):
            slept.append(secs)

        with patch("src.data.market.asyncio.sleep", new=_record_sleep):
            await client._throttle()

        assert len(slept) == 1
        assert slept[0] > 0


class TestCloseAndInit:
    def test_close_resets_exchange(self) -> None:
        client = _make_client()
        exchange = MagicMock()
        client._exchange = exchange
        client.close()
        assert client._exchange is None
        exchange.close.assert_called_once()

    def test_lazy_init(self) -> None:
        client = _make_client()
        assert client._exchange is None
        ex = client._get_exchange()
        assert ex is not None
        # Second call returns same instance
        assert client._get_exchange() is ex
