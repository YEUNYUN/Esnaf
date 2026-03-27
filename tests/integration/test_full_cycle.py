"""Integration test — full graph cycle with mocked LLM and market data."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.graph import build_graph
from src.agent.state import AgentState, MarketSnapshot
from src.config import Settings
from src.execution.paper_broker import PaperBroker
from src.llm.client import LLMClient
from src.risk.engine import RiskEngine
from src.storage.database import Database


def _mock_llm_response(content: str):
    """Create a mock LiteLLM response."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = MagicMock(total_tokens=100)
    return resp


REGIME_RESPONSE = json.dumps({
    "regime": "bull_trend",
    "regime_confidence": 0.85,
    "regime_reasoning": "Strong uptrend with rising volume and bullish RSI.",
    "active_narratives": ["BTC ETF inflows"],
    "narrative_stage": "mainstream",
    "recommended_strategy": "trend_follow",
    "strategy_reasoning": "Follow the established uptrend with momentum.",
})

ANALYSIS_RESPONSE = json.dumps({
    "action": "BUY",
    "asset": "BTC/USDT",
    "confidence": 0.78,
    "size_suggestion": "small",
    "reasoning": "RSI showing momentum continuation after pullback to support.",
    "timeframe": "15m",
    "stop_loss_pct": 2.5,
    "take_profit_pct": 5.0,
    "key_factors": ["RSI momentum", "volume spike", "support hold"],
    "regime_alignment": "Buying in bull trend aligned with trend_follow strategy.",
})

REFLECTION_RESPONSE = json.dumps({
    "trades_reviewed": 1,
    "regime_accuracy": "Regime call appears correct — trend continuing.",
    "patterns_noticed": ["Volume increases preceding breakouts"],
    "lessons_learned": ["Wait for volume confirmation before entries"],
})


@pytest.mark.asyncio
async def test_full_cycle_buy_approved(tmp_path):
    """End-to-end: gather → classify (bull) → analyze (BUY) → validate (pass) → execute → reflect."""
    settings = Settings()

    # Mock market client
    market_client = MagicMock()
    snapshot = MarketSnapshot(
        symbol="BTC/USDT",
        price=67500.0,
        volume_24h=1_500_000_000,
        rsi=58.0,
        macd=150.0,
        macd_signal=120.0,
        bbands_upper=69000.0,
        bbands_lower=66000.0,
        volatility=0.025,
        volume_sma_ratio=1.3,
    )
    market_client.fetch_snapshot = AsyncMock(return_value=snapshot)

    broker = PaperBroker(initial_capital=10000.0, fee_rate=0.001)
    risk_engine = RiskEngine(settings.risk)

    # Mock database
    db = MagicMock(spec=Database)
    db.log_decision = AsyncMock()
    db.log_regime = AsyncMock()
    db.log_trade = AsyncMock()
    db.add_memory = AsyncMock()

    # Mock LLM — returns regime, analysis, and reflection in sequence
    call_count = 0

    async def mock_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        messages = kwargs.get("messages", [])
        system_msg = messages[0].get("content", "").lower() if messages else ""

        if "regime classifier" in system_msg:
            return _mock_llm_response(REGIME_RESPONSE)
        if "trading analyst" in system_msg:
            return _mock_llm_response(ANALYSIS_RESPONSE)
        return _mock_llm_response(REFLECTION_RESPONSE)

    llm_client = LLMClient(settings.llm)

    with patch("litellm.acompletion", side_effect=mock_completion):
        graph = build_graph(
            settings=settings,
            market_client=market_client,
            broker=broker,
            llm_client=llm_client,
            risk_engine=risk_engine,
            db=db,
        )

        state: AgentState = {}
        result = await graph.ainvoke(state)

    # Verify the full pipeline executed
    assert result.get("cycle_id") is not None
    assert result.get("regime") is not None
    assert result["regime"].regime.value == "bull_trend"
    assert result["regime"].regime_confidence == 0.85

    assert result.get("proposal") is not None
    assert result["proposal"].action.value == "BUY"
    assert result["proposal"].confidence == 0.78

    assert result.get("validation_passed") is True
    assert result.get("execution_result", {}).get("status") == "filled"

    # Verify DB was called
    assert db.log_decision.called
    assert db.log_regime.called
    assert db.log_trade.called

    # Verify broker has a position
    portfolio = broker.get_portfolio_state(67500.0)
    assert len(portfolio.open_positions) == 1


@pytest.mark.asyncio
async def test_full_cycle_sit_out():
    """When regime is unknown, agent should SIT_OUT without analyzing."""
    settings = Settings()

    market_client = MagicMock()
    snapshot = MarketSnapshot(
        symbol="BTC/USDT",
        price=67500.0,
        volume_24h=500_000_000,
        rsi=50.0,
        volatility=0.01,
        volume_sma_ratio=0.8,
    )
    market_client.fetch_snapshot = AsyncMock(return_value=snapshot)

    broker = PaperBroker(initial_capital=10000.0, fee_rate=0.001)
    risk_engine = RiskEngine(settings.risk)

    db = MagicMock(spec=Database)
    db.log_decision = AsyncMock()
    db.log_regime = AsyncMock()
    db.log_trade = AsyncMock()
    db.add_memory = AsyncMock()

    sit_out_response = json.dumps({
        "regime": "unknown",
        "regime_confidence": 0.3,
        "regime_reasoning": "Mixed signals, no clear direction.",
        "active_narratives": [],
        "narrative_stage": "early",
        "recommended_strategy": "sit_out",
        "strategy_reasoning": "Too uncertain to trade.",
    })

    async def mock_completion(**kwargs):
        messages = kwargs.get("messages", [])
        system_msg = messages[0].get("content", "").lower() if messages else ""
        if "regime classifier" in system_msg:
            return _mock_llm_response(sit_out_response)
        return _mock_llm_response(REFLECTION_RESPONSE)

    llm_client = LLMClient(settings.llm)

    with patch("litellm.acompletion", side_effect=mock_completion):
        graph = build_graph(
            settings=settings,
            market_client=market_client,
            broker=broker,
            llm_client=llm_client,
            risk_engine=risk_engine,
            db=db,
        )

        result = await graph.ainvoke({})

    assert result["regime"].recommended_strategy.value == "sit_out"
    # Should not have executed any trade
    assert not db.log_trade.called
    portfolio = broker.get_portfolio_state(67500.0)
    assert len(portfolio.open_positions) == 0
