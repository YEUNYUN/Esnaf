"""Classify node — regime classification (THE CORE INNOVATION).

This node runs BEFORE any trade decision. It classifies the current
market regime and selects the appropriate strategy. This is what
separates Esnaf from every other LLM trading bot.
"""

from __future__ import annotations

import structlog

from src.agent.state import (
    AgentState,
    NarrativeStage,
    Regime,
    RegimeClassification,
    Strategy,
)
from src.llm.client import LLMClient

logger = structlog.get_logger()

REGIME_CLASSIFICATION_PROMPT = """You are a crypto market regime classifier. Your ONLY job is to classify
the current market regime and recommend a strategy. Be conservative — when in doubt, classify as UNKNOWN
and recommend SIT_OUT.

## Current Market Data
- Symbol: {symbol}
- Price: ${price:,.2f}
- 24h Volume: ${volume:,.0f}

## Technical Indicators
- RSI (14): {rsi}
- MACD: {macd} (Signal: {macd_signal})
- Bollinger Bands: Upper ${bb_upper:,.2f} / Lower ${bb_lower:,.2f}
- Volatility (ATR%): {volatility}
- Volume vs 20-SMA: {vol_ratio}x

## Sentiment
- Fear & Greed Index: {fear_greed}
- News Sentiment: {news_sentiment}

## Portfolio State
- Total Value: ${total_value:,.2f}
- Open Positions: {open_positions}
- Daily P&L: {daily_pnl_pct:.2f}%

## Recent Trade History
{recent_trades}

## Agent Memory (Lessons Learned)
{memory}

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{
  "regime": "bull_trend | bear_trend | ranging | high_volatility | regime_shift | unknown",
  "regime_confidence": 0.0-1.0,
  "regime_reasoning": "3-4 sentences explaining your classification",
  "active_narratives": ["narrative1", "narrative2"],
  "narrative_stage": "early | mainstream | exhausted",
  "recommended_strategy": "trend_follow | mean_revert | defensive | sit_out",
  "strategy_reasoning": "Why this strategy fits the current regime"
}}"""


async def classify_node(
    state: AgentState,
    *,
    llm_client: LLMClient,
    model: str,
) -> dict:
    """Classify the current market regime using LLM reasoning.

    This is the step that 99% of bots skip. We classify FIRST, then decide.
    """
    market = state.get("market")
    portfolio = state.get("portfolio")
    sentiment = state.get("sentiment")

    if not market or market.price == 0:
        logger.warning("classify_skipped", reason="no_market_data")
        return {"regime": None}

    # Format recent trades
    recent_trades = state.get("recent_trades", [])
    trades_text = "No recent trades" if not recent_trades else "\n".join(
        f"- {t.get('action', '?')} {t.get('symbol', '?')} at ${t.get('price', 0):,.2f} "
        f"({t.get('reasoning', 'no reason')})"
        for t in recent_trades[:5]
    )

    # Format memory
    memory_entries = state.get("memory", [])
    memory_text = "No lessons yet" if not memory_entries else "\n".join(
        f"- {m}" for m in memory_entries[:10]
    )

    prompt = REGIME_CLASSIFICATION_PROMPT.format(
        symbol=market.symbol,
        price=market.price,
        volume=market.volume_24h,
        rsi=f"{market.rsi:.1f}" if market.rsi is not None else "N/A",
        macd=f"{market.macd:.2f}" if market.macd is not None else "N/A",
        macd_signal=f"{market.macd_signal:.2f}" if market.macd_signal is not None else "N/A",
        bb_upper=market.bbands_upper or 0,
        bb_lower=market.bbands_lower or 0,
        volatility=f"{market.volatility:.4f}" if market.volatility is not None else "N/A",
        vol_ratio=f"{market.volume_sma_ratio:.2f}" if market.volume_sma_ratio is not None else "N/A",
        fear_greed=sentiment.fear_greed_index if sentiment else "N/A",
        news_sentiment=sentiment.news_sentiment if sentiment else "N/A",
        total_value=portfolio.total_value if portfolio else 0,
        open_positions=len(portfolio.open_positions) if portfolio else 0,
        daily_pnl_pct=portfolio.daily_pnl_pct if portfolio else 0,
        recent_trades=trades_text,
        memory=memory_text,
    )

    try:
        messages = [
            {"role": "system", "content": "You are a precise market regime classifier. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ]

        result = await llm_client.complete_json(messages, model=model)

        # Parse into our schema
        regime = RegimeClassification(
            regime=Regime(result.get("regime", "unknown")),
            regime_confidence=float(result.get("regime_confidence", 0.0)),
            regime_reasoning=result.get("regime_reasoning", ""),
            active_narratives=result.get("active_narratives", []),
            narrative_stage=NarrativeStage(result.get("narrative_stage", "early")),
            recommended_strategy=Strategy(result.get("recommended_strategy", "sit_out")),
            strategy_reasoning=result.get("strategy_reasoning", ""),
        )

        logger.info(
            "regime_classified",
            regime=regime.regime.value,
            confidence=regime.regime_confidence,
            strategy=regime.recommended_strategy.value,
            model=model,
        )

        return {"regime": regime}

    except Exception as e:
        logger.error("classify_failed", error=str(e))
        # On failure, default to sit_out (conservative)
        return {
            "regime": RegimeClassification(
                regime=Regime.UNKNOWN,
                regime_confidence=0.0,
                regime_reasoning=f"Classification failed: {e}",
                recommended_strategy=Strategy.SIT_OUT,
                strategy_reasoning="Error fallback — sitting out",
            )
        }
