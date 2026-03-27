"""Tests for the paper trading broker."""

from __future__ import annotations

import pytest

from src.agent.state import Action, TradeProposal
from src.execution.paper_broker import PaperBroker


@pytest.fixture
def broker() -> PaperBroker:
    return PaperBroker(initial_capital=10000.0, fee_rate=0.001)


@pytest.fixture
def buy_proposal() -> TradeProposal:
    return TradeProposal(
        action=Action.BUY,
        asset="BTC/USDT",
        confidence=0.8,
        size_suggestion="small",
        reasoning="Test buy",
        timeframe="4h",
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
        key_factors=["test"],
        regime_alignment="test",
    )


class TestPaperBrokerBasics:
    @pytest.mark.asyncio
    async def test_initial_state(self, broker: PaperBroker) -> None:
        state = await broker.get_portfolio_state(50000.0)
        assert state.total_value == 10000.0
        assert state.available_capital == 10000.0
        assert len(state.open_positions) == 0
        assert state.daily_trades == 0

    @pytest.mark.asyncio
    async def test_buy_opens_position(
        self, broker: PaperBroker, buy_proposal: TradeProposal
    ) -> None:
        result = await broker.execute(buy_proposal, current_price=50000.0)
        assert result["status"] == "filled"
        assert result["side"] == "long"
        assert result["price"] == 50000.0
        assert result["quantity"] > 0
        assert result["fee"] > 0

        state = await broker.get_portfolio_state(50000.0)
        assert len(state.open_positions) == 1
        assert state.daily_trades == 1

    @pytest.mark.asyncio
    async def test_hold_does_nothing(self, broker: PaperBroker) -> None:
        hold = TradeProposal(
            action=Action.HOLD,
            asset="BTC/USDT",
            confidence=0.5,
            size_suggestion="small",
            reasoning="No signal",
            timeframe="15m",
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            key_factors=[],
            regime_alignment="N/A",
        )
        result = await broker.execute(hold, current_price=50000.0)
        assert result["status"] == "hold"

        state = await broker.get_portfolio_state(50000.0)
        assert len(state.open_positions) == 0

    @pytest.mark.asyncio
    async def test_close_long_realizes_profit(
        self, broker: PaperBroker, buy_proposal: TradeProposal
    ) -> None:
        # Open at 50000
        await broker.execute(buy_proposal, current_price=50000.0)

        # Close at 55000 (10% up)
        close = TradeProposal(
            action=Action.CLOSE_LONG,
            asset="BTC/USDT",
            confidence=0.8,
            size_suggestion="small",
            reasoning="Take profit",
            timeframe="4h",
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            key_factors=["profit target"],
            regime_alignment="test",
        )
        result = await broker.execute(close, current_price=55000.0)
        assert result["status"] == "closed"
        assert result["pnl"] > 0

        state = await broker.get_portfolio_state(55000.0)
        assert len(state.open_positions) == 0
        assert state.total_value > 10000.0  # Profitable

    @pytest.mark.asyncio
    async def test_close_long_realizes_loss(
        self, broker: PaperBroker, buy_proposal: TradeProposal
    ) -> None:
        # Open at 50000
        await broker.execute(buy_proposal, current_price=50000.0)

        # Close at 45000 (10% down)
        close = TradeProposal(
            action=Action.CLOSE_LONG,
            asset="BTC/USDT",
            confidence=0.8,
            size_suggestion="small",
            reasoning="Stop loss",
            timeframe="4h",
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            key_factors=["stop loss"],
            regime_alignment="test",
        )
        result = await broker.execute(close, current_price=45000.0)
        assert result["status"] == "closed"
        assert result["pnl"] < 0

    @pytest.mark.asyncio
    async def test_fee_deduction(
        self, broker: PaperBroker, buy_proposal: TradeProposal
    ) -> None:
        result = await broker.execute(buy_proposal, current_price=50000.0)
        # 3% of 10000 = 300 position, 0.1% fee = 0.30
        assert result["fee"] == pytest.approx(0.30, abs=0.01)

    @pytest.mark.asyncio
    async def test_unrealized_pnl_updates(
        self, broker: PaperBroker, buy_proposal: TradeProposal
    ) -> None:
        await broker.execute(buy_proposal, current_price=50000.0)

        # Price goes up — unrealized profit (within stop/TP range)
        state_up = await broker.get_portfolio_state(51000.0)
        position = state_up.open_positions[0]
        assert position["unrealized_pnl"] > 0

        # Price goes down slightly — unrealized loss (but NOT past stop-loss)
        state_down = await broker.get_portfolio_state(49500.0)
        position = state_down.open_positions[0]
        assert position["unrealized_pnl"] < 0


class TestPaperBrokerShort:
    @pytest.mark.asyncio
    async def test_short_opens(self, broker: PaperBroker) -> None:
        sell = TradeProposal(
            action=Action.SELL,
            asset="BTC/USDT",
            confidence=0.8,
            size_suggestion="small",
            reasoning="Short signal",
            timeframe="4h",
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            key_factors=["bearish"],
            regime_alignment="With trend",
        )
        result = await broker.execute(sell, current_price=50000.0)
        assert result["status"] == "filled"
        assert result["side"] == "short"

    @pytest.mark.asyncio
    async def test_close_no_positions(self, broker: PaperBroker) -> None:
        close = TradeProposal(
            action=Action.CLOSE_LONG,
            asset="BTC/USDT",
            confidence=0.8,
            size_suggestion="small",
            reasoning="Close",
            timeframe="4h",
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            key_factors=[],
            regime_alignment="test",
        )
        result = await broker.execute(close, current_price=50000.0)
        assert result["status"] == "no_positions"
