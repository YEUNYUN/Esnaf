"""Market data fetching via CCXT."""

from __future__ import annotations

from datetime import datetime

import ccxt
import structlog

from src.agent.state import MarketSnapshot
from src.config import ExchangeConfig

logger = structlog.get_logger()


class MarketDataClient:
    """Fetches market data from exchanges via CCXT.

    Works identically for testnet and live — only the config differs.
    """

    def __init__(self, config: ExchangeConfig) -> None:
        self._config = config
        self._exchange: ccxt.Exchange | None = None

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

    async def fetch_snapshot(self, symbol: str) -> MarketSnapshot:
        """Fetch a complete market data snapshot for a symbol.

        Gathers price, OHLCV candles, and order book in one call.
        Technical indicators are computed separately (not here).
        """
        exchange = self._get_exchange()

        try:
            # Fetch ticker, OHLCV, and order book
            ticker = exchange.fetch_ticker(symbol)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe="15m", limit=100)
            orderbook = exchange.fetch_order_book(symbol, limit=10)

            best_bid = orderbook["bids"][0][0] if orderbook["bids"] else 0.0
            best_ask = orderbook["asks"][0][0] if orderbook["asks"] else 0.0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
            spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price else 0.0

            snapshot = MarketSnapshot(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                price=float(ticker.get("last", 0)),
                ohlcv=ohlcv,
                volume_24h=float(ticker.get("quoteVolume", 0)),
                bid=best_bid,
                ask=best_ask,
                spread_pct=spread_pct,
            )

            logger.debug(
                "market_snapshot",
                symbol=symbol,
                price=snapshot.price,
                spread_pct=f"{spread_pct:.4f}%",
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
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker.get("last", 0))

    def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            self._exchange.close()
            self._exchange = None
