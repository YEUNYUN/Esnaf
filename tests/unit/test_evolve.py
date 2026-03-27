"""Tests for the weekly evolution node."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.nodes.evolve import evolve_node


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_recent_trades = AsyncMock(return_value=[
        {"action": "BUY", "symbol": "BTC/USDT", "pnl": 25.0, "regime": "bull_trend", "strategy": "trend_follow"},
        {"action": "BUY", "symbol": "ETH/USDT", "pnl": -10.0, "regime": "ranging", "strategy": "mean_revert"},
        {"action": "SELL", "symbol": "BTC/USDT", "pnl": 15.0, "regime": "bull_trend", "strategy": "trend_follow"},
        {"action": "BUY", "symbol": "SOL/USDT", "pnl": -5.0, "regime": "high_volatility", "strategy": "defensive"},
        {"action": "BUY", "symbol": "BTC/USDT", "pnl": 30.0, "regime": "bull_trend", "strategy": "trend_follow"},
    ])
    db.get_recent_memories = AsyncMock(return_value=[
        {"content": "Trend-following works in bull regimes"},
        {"content": "Mean-revert underperforms in ranging markets"},
    ])
    db.add_memory = AsyncMock()
    return db


@pytest.fixture
def risk_config():
    return {
        "max_capital": 10000,
        "min_confidence": 0.65,
        "max_per_trade": 50,
        "max_daily_drawdown": 50,
        "cooldown_minutes": 15,
    }


class TestEvolveNode:
    """Test the weekly evolution node."""

    @pytest.mark.asyncio
    async def test_skips_with_few_trades(self, risk_config):
        mock_llm = MagicMock()
        db = MagicMock()
        db.get_recent_trades = AsyncMock(return_value=[
            {"action": "BUY", "pnl": 10.0},
            {"action": "SELL", "pnl": -5.0},
        ])

        result = await evolve_node(
            llm_client=mock_llm, model="test-model", db=db, risk_config=risk_config
        )
        assert result["evolution"]["status"] == "skipped"
        assert "Only 2 trades" in result["evolution"]["reason"]

    @pytest.mark.asyncio
    async def test_produces_evolution_proposal(self, mock_db, risk_config):
        mock_llm = MagicMock()
        mock_llm.complete_json = AsyncMock(return_value={
            "week_summary": "Good week, trend following in bull regime working well.",
            "regime_insights": [
                {"regime": "bull_trend", "performance": "good", "observation": "3 wins"},
            ],
            "strategy_insights": [
                {"strategy": "trend_follow", "performance": "good", "observation": "profitable"},
            ],
            "proposed_adjustments": [
                {
                    "parameter": "min_confidence",
                    "current_value": "0.65",
                    "proposed_value": "0.6",
                    "reasoning": "Bull trend success suggests we can lower threshold",
                    "confidence": 0.7,
                    "risk_level": "medium",
                },
            ],
            "keep_doing": ["Trend following in bull markets"],
            "stop_doing": ["Mean reversion in ranging markets"],
        })

        result = await evolve_node(
            llm_client=mock_llm, model="test-model", db=mock_db, risk_config=risk_config
        )

        assert result["evolution"]["status"] == "complete"
        assert "week_summary" in result["evolution"]
        assert len(result["evolution"]["proposed_adjustments"]) == 1
        assert result["evolution"]["proposed_adjustments"][0]["parameter"] == "min_confidence"

        # Should store evolution in memory
        mock_db.add_memory.assert_called_once()
        call_kwargs = mock_db.add_memory.call_args[1]
        assert call_kwargs["entry_type"] == "evolution"

    @pytest.mark.asyncio
    async def test_handles_llm_error(self, mock_db, risk_config):
        mock_llm = MagicMock()
        mock_llm.complete_json = AsyncMock(side_effect=Exception("LLM timeout"))

        result = await evolve_node(
            llm_client=mock_llm, model="test-model", db=mock_db, risk_config=risk_config
        )
        assert result["evolution"]["status"] == "error"
        assert "LLM timeout" in result["evolution"]["error"]

    @pytest.mark.asyncio
    async def test_groups_trades_by_regime(self, mock_db, risk_config):
        """Verify prompt includes regime breakdown."""
        mock_llm = MagicMock()
        mock_llm.complete_json = AsyncMock(return_value={
            "week_summary": "Test",
            "regime_insights": [],
            "strategy_insights": [],
            "proposed_adjustments": [],
            "keep_doing": [],
            "stop_doing": [],
        })

        await evolve_node(
            llm_client=mock_llm, model="test-model", db=mock_db, risk_config=risk_config
        )

        # Check the prompt sent to LLM
        call_args = mock_llm.complete_json.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "bull_trend" in user_msg
        assert "ranging" in user_msg
        assert "trend_follow" in user_msg
