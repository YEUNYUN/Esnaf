"""Gather node — collects all market data for the current cycle.

This is the first node in the graph. It fetches price data, computes
technical indicators, and assembles the complete data snapshot.
"""

from __future__ import annotations

from datetime import UTC, datetime
import uuid

import structlog

from src.agent.state import (
    AgentState,
    MarketSnapshot,
    PortfolioState,
    Regime,
    RegimeClassification,
    SentimentData,
    Strategy,
)
from src.data.indicators import compute_indicators
from src.data.market import MarketDataClient
from src.data.sentiment import SentimentPipeline
from src.execution.paper_broker import PaperBroker

logger = structlog.get_logger()


async def gather_node(
    state: AgentState,
    *,
    market_client: MarketDataClient,
    broker: PaperBroker,
    sentiment_pipeline: SentimentPipeline | None = None,
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

        # Block trading on stale market data
        if snapshot.stale:
            logger.warning(
                "stale_market_data",
                cycle_id=cycle_id,
                symbol=symbol,
                price=snapshot.price,
            )
            portfolio = await broker.get_portfolio_state(snapshot.price)
            return {
                "cycle_id": cycle_id,
                "cycle_timestamp": datetime.now(UTC).isoformat(),
                "market": snapshot,
                "sentiment": SentimentData(),
                "portfolio": portfolio,
                "regime": RegimeClassification(
                    regime=Regime.UNKNOWN,
                    regime_confidence=0.0,
                    regime_reasoning="Stale market data detected, sitting out",
                    recommended_strategy=Strategy.SIT_OUT,
                    strategy_reasoning="Stale market data detected, sitting out",
                ),
                "error": None,
            }

        # Get current portfolio state
        portfolio = await broker.get_portfolio_state(snapshot.price)

        # Get sentiment (from real APIs if pipeline is wired)
        if sentiment_pipeline:
            sentiment = await sentiment_pipeline.fetch()
        else:
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
