"""Gather node — collects all market data for the current cycle.

This is the first node in the graph. It fetches price data, computes
technical indicators, and assembles the complete data snapshot.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog

from src.agent.state import AgentState, MarketSnapshot, PortfolioState, SentimentData
from src.data.indicators import compute_indicators
from src.data.market import MarketDataClient
from src.execution.paper_broker import PaperBroker

logger = structlog.get_logger()


async def gather_node(
    state: AgentState,
    *,
    market_client: MarketDataClient,
    broker: PaperBroker,
    symbol: str,
) -> dict:
    """Gather all market data for this analysis cycle.

    Returns a partial state dict that LangGraph merges into the full state.
    """
    cycle_id = f"cycle-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    try:
        # Fetch market data
        snapshot = await market_client.fetch_snapshot(symbol)

        # Compute technical indicators locally (not by LLM)
        snapshot = compute_indicators(snapshot)

        # Get current portfolio state
        portfolio = broker.get_portfolio_state(snapshot.price)

        # Get sentiment (placeholder — will be wired in Phase 2)
        sentiment = SentimentData()

        logger.info(
            "gather_complete",
            cycle_id=cycle_id,
            price=snapshot.price,
            rsi=snapshot.rsi,
            volatility=snapshot.volatility,
        )

        return {
            "cycle_id": cycle_id,
            "cycle_timestamp": datetime.now(UTC).isoformat(),
            "market": snapshot,
            "sentiment": sentiment,
            "portfolio": portfolio,
            "error": None,
        }

    except Exception as e:
        logger.error("gather_failed", error=str(e))
        return {
            "cycle_id": cycle_id,
            "cycle_timestamp": datetime.now(UTC).isoformat(),
            "market": MarketSnapshot(),
            "sentiment": SentimentData(),
            "portfolio": PortfolioState(),
            "error": f"Gather failed: {e}",
        }
