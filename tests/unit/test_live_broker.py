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


# ---------------------------------------------------------------------------
# Exchange-side protective orders
# ---------------------------------------------------------------------------

class TestProtectiveOrders:
    """Tests for OCO / stop-loss-limit protective order placement."""

    @pytest.mark.asyncio
    async def test_protective_orders_called_after_long_fill(
        self, broker: LiveBroker,
    ) -> None:
        """_place_protective_orders is invoked after a successful long fill."""
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()
        broker._exchange.private_post_order_oco.return_value = {
            "orderListId": "999",
            "orderReports": [
                {"orderId": "oco-sl"},
                {"orderId": "oco-tp"},
            ],
        }

        result = await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        assert result["status"] == "filled"
        broker._exchange.private_post_order_oco.assert_called_once()
        pos = broker._positions[0]
        assert pos.stop_order_ids == ["oco-sl", "oco-tp"]

    @pytest.mark.asyncio
    async def test_protective_orders_called_after_short_fill(
        self, broker: LiveBroker,
    ) -> None:
        """_place_protective_orders is invoked after a successful short fill."""
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order(side="sell")
        broker._exchange.private_post_order_oco.return_value = {
            "orderListId": "888",
            "orderReports": [
                {"orderId": "oco-sl-short"},
                {"orderId": "oco-tp-short"},
            ],
        }

        result = await broker.execute(
            _make_proposal(Action.SELL), current_price=50000.0,
        )

        assert result["status"] == "filled"
        broker._exchange.private_post_order_oco.assert_called_once()
        call_params = broker._exchange.private_post_order_oco.call_args[0][0]
        assert call_params["side"] == "BUY"  # opposite side for short protection
        pos = broker._positions[0]
        assert pos.stop_order_ids == ["oco-sl-short", "oco-tp-short"]

    @pytest.mark.asyncio
    async def test_oco_fallback_to_stop_loss_limit(
        self, broker: LiveBroker,
    ) -> None:
        """When OCO fails, falls back to a stop-loss-limit order."""
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.side_effect = [
            _make_order(),  # market buy succeeds
            {"id": "sl-only-123", "status": "open"},  # stop-loss-limit
        ]
        broker._exchange.private_post_order_oco.side_effect = ccxt.InvalidOrder(
            "OCO not allowed"
        )

        result = await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        assert result["status"] == "filled"
        # OCO tried and failed
        broker._exchange.private_post_order_oco.assert_called_once()
        # create_order called twice: market buy + stop-loss-limit fallback
        assert broker._exchange.create_order.call_count == 2
        second_call = broker._exchange.create_order.call_args_list[1]
        assert second_call[0][1] == "STOP_LOSS_LIMIT"
        pos = broker._positions[0]
        assert pos.stop_order_ids == ["sl-only-123"]

    @pytest.mark.asyncio
    async def test_all_protective_orders_fail_software_fallback(
        self, broker: LiveBroker,
    ) -> None:
        """When both OCO and stop-loss-limit fail, software stops remain."""
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.side_effect = [
            _make_order(),  # market buy succeeds
            ccxt.ExchangeError("stop loss rejected"),  # SL limit fails
        ]
        broker._exchange.private_post_order_oco.side_effect = ccxt.ExchangeError(
            "OCO rejected"
        )

        result = await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        assert result["status"] == "filled"
        pos = broker._positions[0]
        assert pos.stop_order_ids == []
        # Software-side stops still present
        assert pos.stop_loss is not None
        assert pos.take_profit is not None


class TestCancelProtectiveOrders:
    """Tests for canceling protective orders on position close."""

    @pytest.mark.asyncio
    async def test_cancel_protective_orders_on_close(
        self, broker: LiveBroker,
    ) -> None:
        """Protective orders are canceled before a position is closed."""
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()
        broker._exchange.private_post_order_oco.return_value = {
            "orderReports": [{"orderId": "oco-1"}, {"orderId": "oco-2"}],
        }

        await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)
        assert broker._positions[0].stop_order_ids == ["oco-1", "oco-2"]

        # Now close the position
        close_order = _make_order(order_id="close-456", side="sell")
        broker._exchange.create_order.return_value = close_order
        # Avoid OCO placement during close (no new protective orders)
        broker._exchange.cancel_order.return_value = {}

        await broker.execute(
            _make_proposal(Action.CLOSE_LONG), current_price=51000.0,
        )

        assert len(broker._positions) == 0
        assert broker._exchange.cancel_order.call_count == 2
        cancel_calls = [c[0] for c in broker._exchange.cancel_order.call_args_list]
        assert ("oco-1", "BTC/USDT") in cancel_calls
        assert ("oco-2", "BTC/USDT") in cancel_calls

    @pytest.mark.asyncio
    async def test_cancel_handles_order_already_filled(
        self, broker: LiveBroker,
    ) -> None:
        """Canceling an already-filled order doesn't raise."""
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()
        broker._exchange.private_post_order_oco.return_value = {
            "orderReports": [{"orderId": "gone-1"}, {"orderId": "gone-2"}],
        }

        await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        # Simulate one already filled, one not found
        broker._exchange.cancel_order.side_effect = [
            ccxt.OrderNotFound("already filled"),
            ccxt.InvalidOrder("already canceled"),
        ]
        close_order = _make_order(order_id="close-789", side="sell")
        broker._exchange.create_order.return_value = close_order

        # Should not raise
        result = await broker.execute(
            _make_proposal(Action.CLOSE_LONG), current_price=51000.0,
        )
        assert result["status"] == "closed"


class TestSyncExchangeOrders:
    """Tests for _sync_exchange_orders detecting filled protective orders."""

    @pytest.mark.asyncio
    async def test_sync_detects_filled_stop_loss(
        self, broker: LiveBroker,
    ) -> None:
        """A filled stop-loss on exchange removes the local position."""
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()
        broker._exchange.private_post_order_oco.return_value = {
            "orderReports": [{"orderId": "sl-100"}, {"orderId": "tp-101"}],
        }

        await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)
        assert len(broker._positions) == 1

        # Simulate the stop-loss being filled on exchange
        broker._exchange.fetch_order.return_value = {
            "id": "sl-100",
            "status": "closed",
            "filled": 0.006,
            "average": 48000.0,
            "price": 48000.0,
            "fee": {"cost": 0.3, "currency": "USDT"},
        }

        state = await broker.get_portfolio_state(48000.0)

        # Position should be removed after sync
        assert len(broker._positions) == 0
        assert len(state.open_positions) == 0

    @pytest.mark.asyncio
    async def test_sync_ignores_open_orders(
        self, broker: LiveBroker,
    ) -> None:
        """Open (unfilled) protective orders don't affect positions."""
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()
        broker._exchange.private_post_order_oco.return_value = {
            "orderReports": [{"orderId": "sl-200"}],
        }

        await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        # Order is still open on exchange
        broker._exchange.fetch_order.return_value = {
            "id": "sl-200",
            "status": "open",
            "filled": 0,
        }

        state = await broker.get_portfolio_state(50000.0)

        assert len(broker._positions) == 1
        assert len(state.open_positions) == 1

    @pytest.mark.asyncio
    async def test_sync_handles_fetch_order_error(
        self, broker: LiveBroker,
    ) -> None:
        """Exchange errors during sync don't crash the system."""
        broker._exchange.fetch_balance.return_value = _make_balance()
        broker._exchange.create_order.return_value = _make_order()
        broker._exchange.private_post_order_oco.return_value = {
            "orderReports": [{"orderId": "err-300"}],
        }

        await broker.execute(_make_proposal(Action.BUY), current_price=50000.0)

        broker._exchange.fetch_order.side_effect = ccxt.ExchangeError("gone")

        # Should not raise — position stays
        await broker.get_portfolio_state(50000.0)
        assert len(broker._positions) == 1
