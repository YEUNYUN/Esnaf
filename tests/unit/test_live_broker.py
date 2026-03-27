"""Tests for the live exchange broker (all CCXT calls mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import ccxt
import pytest

from src.agent.state import Action, TradeProposal
from src.config import ExchangeConfig
from src.execution.live_broker import LiveBroker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proposal(
    action: Action = Action.BUY,
    asset: str = "BTC/USDT",
    size: str = "small",
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 5.0,
) -> TradeProposal:
    return TradeProposal(
        action=action,
        asset=asset,
        confidence=0.8,
        size_suggestion=size,
        reasoning="test",
        timeframe="4h",
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        key_factors=["test"],
        regime_alignment="test",
    )


def _make_order(
    order_id: str = "123",
    side: str = "buy",
    filled: float = 0.006,
    cost: float = 300.0,
    remaining: float = 0.0,
    status: str = "closed",
    fee_cost: float = 0.3,
) -> dict:
    return {
        "id": order_id,
        "symbol": "BTC/USDT",
        "side": side,
        "type": "market",
        "amount": filled + remaining,
        "filled": filled,
        "remaining": remaining,
        "cost": cost,
        "status": status,
        "average": cost / filled if filled else None,
        "price": cost / filled if filled else None,
        "fee": {"cost": fee_cost, "currency": "USDT"},
    }


def _make_balance(free_usdt: float = 10000.0, total_usdt: float = 10000.0) -> dict:
    return {
        "USDT": {"free": free_usdt, "total": total_usdt},
        "BTC": {"free": 0.5, "total": 0.5},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> ExchangeConfig:
    return ExchangeConfig(api_key="test-key", secret="test-secret", testnet=True)


@pytest.fixture
def broker(config: ExchangeConfig) -> LiveBroker:
    with patch("src.execution.live_broker.ccxt.binance") as mock_cls:
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange
        b = LiveBroker(config=config)
        b._exchange = mock_exchange
    return b


# ---------------------------------------------------------------------------
# Testnet activation
# ---------------------------------------------------------------------------

class TestTestnetMode:
    def test_sandbox_mode_enabled(self, config: ExchangeConfig) -> None:
        with patch("src.execution.live_broker.ccxt.binance") as mock_cls:
            mock_exchange = MagicMock()
            mock_cls.return_value = mock_exchange
            LiveBroker(config=config)
            mock_exchange.set_sandbox_mode.assert_called_once_with(True)

    def test_sandbox_mode_disabled_when_not_testnet(self) -> None:
        cfg = ExchangeConfig(api_key="k", secret="s", testnet=False)
        with patch("src.execution.live_broker.ccxt.binance") as mock_cls:
            mock_exchange = MagicMock()
            mock_cls.return_value = mock_exchange
            LiveBroker(config=cfg)
            mock_exchange.set_sandbox_mode.assert_not_called()


# ---------------------------------------------------------------------------
# Successful market buy
# ---------------------------------------------------------------------------

class TestMarketBuy:
    @pytest.mark.asyncio
    async def test_buy_returns_filled(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()

        result = await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        assert result["status"] == "filled"
        assert result["side"] == "long"
        assert result["quantity"] > 0
        assert result["fee"] == 0.3
        assert "order_id" in result

    @pytest.mark.asyncio
    async def test_buy_tracks_position(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()

        await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        assert len(broker._positions) == 1
        pos = broker._positions[0]
        assert pos.side == "long"
        assert pos.symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_buy_calls_create_order(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()

        await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        broker._exchange.create_order.assert_called_once()
        args = broker._exchange.create_order.call_args
        assert args[0][0] == "BTC/USDT"  # symbol
        assert args[0][1] == "market"     # type
        assert args[0][2] == "buy"        # side


# ---------------------------------------------------------------------------
# Successful market sell
# ---------------------------------------------------------------------------

class TestMarketSell:
    @pytest.mark.asyncio
    async def test_sell_returns_filled(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order(side="sell")

        result = await broker.execute(_make_proposal(Action.SELL), current_price=50000.0)

        assert result["status"] == "filled"
        assert result["side"] == "short"

    @pytest.mark.asyncio
    async def test_sell_checks_base_balance(self, broker: LiveBroker) -> None:
        """Sell rejects when base asset balance is too low."""
        balance = _make_balance()
        balance["BTC"] = {"free": 0.0, "total": 0.0}
        broker._exchange.fetch_balance.return_value = balance

        result = await broker.execute(_make_proposal(Action.SELL), current_price=50000.0)

        assert result["status"] == "rejected"
        assert "Insufficient" in result["message"]


# ---------------------------------------------------------------------------
# Insufficient balance handling
# ---------------------------------------------------------------------------

class TestInsufficientBalance:
    @pytest.mark.asyncio
    async def test_buy_rejected_low_balance(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance(free_usdt=0.0, total_usdt=0.0)

        result = await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        assert result["status"] == "rejected"
        assert "Insufficient" in result["message"]

    @pytest.mark.asyncio
    async def test_buy_rejected_exchange_insufficient_funds(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.side_effect = ccxt.InsufficientFunds("not enough")

        result = await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        assert result["status"] == "rejected"
        assert "Insufficient" in result["message"]


# ---------------------------------------------------------------------------
# Portfolio fetching
# ---------------------------------------------------------------------------

class TestPortfolio:
    @pytest.mark.asyncio
    async def test_portfolio_empty(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()

        state = await broker.get_portfolio_state(50000.0)

        assert state.available_capital == 10000.0
        assert len(state.open_positions) == 0

    @pytest.mark.asyncio
    async def test_portfolio_with_position(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()

        await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        state = await broker.get_portfolio_state(50000.0)
        assert len(state.open_positions) == 1
        assert state.daily_trades == 1


# ---------------------------------------------------------------------------
# Stop-loss triggering
# ---------------------------------------------------------------------------

class TestStopLoss:
    @pytest.mark.asyncio
    async def test_stop_loss_triggers_close(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()

        await broker.execute(
            _make_proposal(Action.BUY, stop_loss_pct=2.0),
            current_price=50000.0,
        )
        assert len(broker._positions) == 1

        # Price drops to trigger stop-loss
        close_order = _make_order(order_id="456", side="sell")
        broker._exchange.create_order.return_value = close_order

        closed = await broker.update_prices({"BTC/USDT": 40000.0})

        assert len(closed) == 1
        assert closed[0]["status"] == "closed"
        assert len(broker._positions) == 0

    @pytest.mark.asyncio
    async def test_take_profit_triggers_close(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()

        await broker.execute(
            _make_proposal(Action.BUY, take_profit_pct=5.0),
            current_price=50000.0,
        )

        close_order = _make_order(order_id="789", side="sell")
        broker._exchange.create_order.return_value = close_order

        closed = await broker.update_prices({"BTC/USDT": 60000.0})

        assert len(closed) == 1
        assert closed[0]["status"] == "closed"

    @pytest.mark.asyncio
    async def test_no_stop_trigger_in_range(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()

        await broker.execute(
            _make_proposal(Action.BUY, stop_loss_pct=2.0, take_profit_pct=5.0),
            current_price=50000.0,
        )

        closed = await broker.update_prices({"BTC/USDT": 50500.0})

        assert len(closed) == 0
        assert len(broker._positions) == 1


# ---------------------------------------------------------------------------
# Hold action
# ---------------------------------------------------------------------------

class TestHold:
    @pytest.mark.asyncio
    async def test_hold_returns_hold(self, broker: LiveBroker) -> None:
        result = await broker.execute(_make_proposal(Action.HOLD), current_price=50000.0)
        assert result["status"] == "hold"


# ---------------------------------------------------------------------------
# Network / exchange errors
# ---------------------------------------------------------------------------

class TestErrors:
    @pytest.mark.asyncio
    async def test_network_error_raises(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.side_effect = ccxt.NetworkError("timeout")

        with pytest.raises(RuntimeError, match="Network error"):
            await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

    @pytest.mark.asyncio
    async def test_exchange_unavailable_raises(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.side_effect = ccxt.ExchangeNotAvailable("maintenance")

        with pytest.raises(RuntimeError, match="Exchange unavailable"):
            await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

    @pytest.mark.asyncio
    async def test_fetch_balance_network_error(self, broker: LiveBroker) -> None:
        broker._exchange.fetch_balance.side_effect = ccxt.NetworkError("offline")

        with pytest.raises(RuntimeError, match="Network error"):
            await broker.get_portfolio_state(50000.0)


# ---------------------------------------------------------------------------
# Close with no positions
# ---------------------------------------------------------------------------

class TestCloseNoPositions:
    @pytest.mark.asyncio
    async def test_close_long_no_positions(self, broker: LiveBroker) -> None:
        result = await broker.execute(
            _make_proposal(Action.CLOSE_LONG), current_price=50000.0
        )
        assert result["status"] == "no_positions"

    @pytest.mark.asyncio
    async def test_close_short_no_positions(self, broker: LiveBroker) -> None:
        result = await broker.execute(
            _make_proposal(Action.CLOSE_SHORT), current_price=50000.0
        )
        assert result["status"] == "no_positions"
