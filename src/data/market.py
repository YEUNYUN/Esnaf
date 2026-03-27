"""Market data fetching via CCXT.

All blocking CCXT calls are wrapped with ``asyncio.to_thread`` so they
never block the event loop.  Rate-limit errors are retried with
exponential back-off, and OHLCV freshness is validated after every fetch.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import time

import ccxt
import structlog

from src.agent.state import MarketSnapshot
from src.config import ExchangeConfig

logger = structlog.get_logger()

# Timeframe → expected candle interval in seconds
_TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds
_MIN_CALL_INTERVAL = 0.25  # seconds between calls per exchange


class MarketDataClient:
    """Fetches market data from exchanges via CCXT.

    Works identically for testnet and live — only the config differs.
    """

    def __init__(self, config: ExchangeConfig) -> None:
        self._config = config
        self._exchange: ccxt.Exchange | None = None
        self._last_call_time: float = 0.0

    def _get_exchange(self) -> ccxt.Exchange:
        """Lazy-initialize the exchange connection."""
        if self._exchange is None:
            exchange_config: dict = {
                "apiKey": self._config.api_key,
                "secret": self._config.secret,
                "enableRateLimit": True,
            }

            if self._config.testnet:
                exchange_config["options"] = {"defaultType": "spot"}

            self._exchange = ccxt.binance(exchange_config)

            if self._config.testnet:
                self._exchange.set_sandbox_mode(True)

            logger.info(
                "exchange_connected",
                exchange="binance",
                testnet=self._config.testnet,
            )

        return self._exchange

    # ------------------------------------------------------------------
    # Rate-limit helpers
    # ------------------------------------------------------------------

    async def _throttle(self) -> None:
        """Enforce minimum interval between exchange calls."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < _MIN_CALL_INTERVAL:
            await asyncio.sleep(_MIN_CALL_INTERVAL - elapsed)

    async def _call(self, fn, *args, **kwargs):
        """Run a blocking CCXT call in a thread with rate-limit retry."""
        for attempt in range(1, _MAX_RETRIES + 1):
            await self._throttle()
            try:
                result = await asyncio.to_thread(fn, *args, **kwargs)
                self._last_call_time = time.monotonic()
                return result
            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "rate_limited",
                    attempt=attempt,
                    delay=delay,
                    error=str(exc),
                )
                if attempt == _MAX_RETRIES:
                    raise
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Data freshness
    # ------------------------------------------------------------------

    @staticmethod
    def _check_ohlcv_freshness(
        ohlcv: list[list[float]],
        timeframe: str,
    ) -> bool:
        """Return *True* if data is stale (latest candle too old).

        A candle is stale when its timestamp is older than 2× the
        expected interval from *now*.
        """
        if not ohlcv:
            return True

        interval = _TIMEFRAME_SECONDS.get(timeframe, 900)
        latest_ts_ms = ohlcv[-1][0]
        now_ms = time.time() * 1000
        age_s = (now_ms - latest_ts_ms) / 1000

        if age_s > interval * 2:
            logger.warning(
                "stale_ohlcv",
                timeframe=timeframe,
                candle_age_s=round(age_s, 1),
                threshold_s=interval * 2,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_snapshot(self, symbol: str, timeframe: str = "15m") -> MarketSnapshot:
        """Fetch a complete market data snapshot for a symbol.

        Gathers price, OHLCV candles, and order book in one call.
        Technical indicators are computed separately (not here).
        """
        exchange = self._get_exchange()

        try:
            ticker = await self._call(exchange.fetch_ticker, symbol)
            ohlcv = await self._call(exchange.fetch_ohlcv, symbol, timeframe, None, 100)
            orderbook = await self._call(exchange.fetch_order_book, symbol, 10)

            best_bid = orderbook["bids"][0][0] if orderbook["bids"] else 0.0
            best_ask = orderbook["asks"][0][0] if orderbook["asks"] else 0.0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
            spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price else 0.0

            stale = self._check_ohlcv_freshness(ohlcv, timeframe)

            snapshot = MarketSnapshot(
                timestamp=datetime.now(UTC),
                symbol=symbol,
                price=float(ticker.get("last", 0)),
                ohlcv=ohlcv,
                volume_24h=float(ticker.get("quoteVolume", 0)),
                bid=best_bid,
                ask=best_ask,
                spread_pct=spread_pct,
                stale=stale,
            )

            logger.debug(
                "market_snapshot",
                symbol=symbol,
                price=snapshot.price,
                spread_pct=f"{spread_pct:.4f}%",
                stale=stale,
            )
            return snapshot

        except ccxt.NetworkError as e:
            logger.error("network_error", symbol=symbol, error=str(e))
            raise
        except ccxt.ExchangeError as e:
            logger.error("exchange_error", symbol=symbol, error=str(e))
            raise

    async def fetch_price(self, symbol: str) -> float:
        """Fetch just the current price. Used for quick checks."""
        exchange = self._get_exchange()
        ticker = await self._call(exchange.fetch_ticker, symbol)
        return float(ticker.get("last", 0))

    def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            self._exchange.close()
            self._exchange = None
