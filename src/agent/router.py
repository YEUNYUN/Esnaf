"""Model selection router — deterministic, no LLM involvement."""

from __future__ import annotations

import structlog

from src.agent.state import MarketSnapshot, PortfolioState
from src.config import LLMConfig

logger = structlog.get_logger()


def select_model(
    config: LLMConfig,
    market: MarketSnapshot | None = None,
    portfolio: PortfolioState | None = None,
) -> str:
    """Select the LLM model based on current market conditions.

    This is a DETERMINISTIC function. The LLM does not choose its own model.

    Rules:
    - Calm market, no positions, no signals → cheap model (Gemini Flash)
    - Active market OR open positions OR notable signals → primary model (Claude via Copilot)
    - Both fail → fallback (Ollama local)
    """
    if market is None:
        return config.cheap_model

    volatility = market.volatility or 0.0
    has_positions = bool(portfolio and portfolio.open_positions)

    # Check for notable signals
    has_signal = False
    if market.rsi is not None and (market.rsi < 30 or market.rsi > 70):
        has_signal = True
    if market.volume_sma_ratio is not None and market.volume_sma_ratio > 2.0:
        has_signal = True

    if volatility < config.calm_volatility_threshold and not has_positions and not has_signal:
        logger.debug("model_selected", model=config.cheap_model, reason="calm_market")
        return config.cheap_model

    logger.debug(
        "model_selected",
        model=config.primary_model,
        reason="active_market",
        volatility=f"{volatility:.4f}",
        has_positions=has_positions,
        has_signal=has_signal,
    )
    return config.primary_model
