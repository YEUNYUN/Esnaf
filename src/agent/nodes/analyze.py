"""Analyze node — trade decision within the classified regime.

Only runs if the regime classifier recommends a strategy other than SIT_OUT.
The LLM makes a specific trade decision WITHIN the context of the classified regime.
"""

from __future__ import annotations

import structlog

from src.agent.sanitize import sanitize_prompt_input
from src.agent.state import (
    Action,
    AgentState,
    Strategy,
    TradeProposal,
)
from src.llm.client import LLMClient

logger = structlog.get_logger()

TRADE_ANALYSIS_PROMPT = """You are a crypto trading analyst. The market regime has been classified.
Your job is to decide whether to trade, and if so, what action to take.

## Current Regime
- Regime: {regime}
- Confidence: {regime_confidence:.0%}
- Reasoning: {regime_reasoning}
- Active Strategy: {strategy}
- Active Narratives: {narratives}

## Market Data
- Symbol: {symbol}
- Price: ${price:,.2f}
- RSI (14): {rsi}
- MACD: {macd} (Signal: {macd_signal})
- Bollinger Bands: Upper ${bb_upper:,.2f} / Lower ${bb_lower:,.2f}
- Volatility (ATR%): {volatility}
- Volume vs 20-SMA: {vol_ratio}x

## Portfolio
- Total Value: ${total_value:,.2f}
- Available Capital: ${available_capital:,.2f}
- Open Positions: {open_positions}
- Daily P&L: {daily_pnl_pct:.2f}%

## Strategy Guidelines
{strategy_guidelines}

## CRITICAL RULES
- You MUST align your decision with the classified regime
- Counter-trend trades need EXTREMELY high conviction
- HOLD is always a valid option — don't force trades
- Be specific about stop-loss and take-profit levels

Respond with ONLY a JSON object:
{{
  "action": "BUY | SELL | HOLD | CLOSE_LONG | CLOSE_SHORT",
  "asset": "{symbol}",
  "confidence": 0.0-1.0,
  "size_suggestion": "small | medium | large",
  "reasoning": "2-3 sentences explaining your decision",
  "timeframe": "15m | 1h | 4h | 1d",
  "stop_loss_pct": 2.0,
  "take_profit_pct": 5.0,
  "key_factors": ["factor1", "factor2"],
  "regime_alignment": "How this trade aligns with the classified regime"
}}"""

STRATEGY_GUIDELINES = {
    Strategy.TREND_FOLLOW: """TREND FOLLOWING in {regime}:
- Look for pullbacks in the trend direction to enter
- Use momentum indicators (RSI, MACD) to confirm trend continuation
- Set wider stops to avoid getting shaken out by noise
- Take partial profits at resistance/support levels
- BUY in bull_trend, SELL/SHORT in bear_trend""",

    Strategy.MEAN_REVERT: """MEAN REVERSION in ranging market:
- Buy near the bottom of the range (support)
- Sell near the top of the range (resistance)
- Use Bollinger Bands to identify overbought/oversold
- Keep tight stops — if range breaks, exit immediately
- Smaller position sizes (ranging markets are unpredictable)""",

    Strategy.DEFENSIVE: """DEFENSIVE in adverse conditions:
- Priority: protect capital, minimize exposure
- Close profitable positions — take profits while available
- Tighten stop-losses on open positions
- Avoid new entries unless extremely high conviction
- Cash is a position""",

    Strategy.SIT_OUT: """SIT OUT — regime is unclear or risky:
- HOLD is the only valid action
- Do NOT enter new positions
- Monitor existing positions for stop-loss/take-profit triggers
- Wait for regime clarity before trading""",
}


async def analyze_node(
    state: AgentState,
    *,
    llm_client: LLMClient,
    model: str,
) -> dict:
    """Analyze and propose a trade within the classified regime."""
    regime = state.get("regime")
    market = state.get("market")
    portfolio = state.get("portfolio")

    # If regime says sit out, skip analysis
    if not regime or regime.recommended_strategy == Strategy.SIT_OUT:
        logger.info("analyze_skipped", reason="sit_out_strategy")
        return {
            "proposal": TradeProposal(
                action=Action.HOLD,
                asset=market.symbol if market else "BTC/USDT",
                confidence=0.0,
                size_suggestion="small",
                reasoning="Sitting out — regime unclear or strategy is SIT_OUT",
                timeframe="15m",
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                key_factors=["sit_out"],
                regime_alignment="Following sit_out directive",
            )
        }

    # Get strategy-specific guidelines
    guidelines = STRATEGY_GUIDELINES.get(
        regime.recommended_strategy,
        "No specific guidelines — use best judgment",
    ).format(regime=regime.regime.value)

    prompt = TRADE_ANALYSIS_PROMPT.format(
        regime=regime.regime.value,
        regime_confidence=regime.regime_confidence,
        regime_reasoning=sanitize_prompt_input(regime.regime_reasoning, max_length=500),
        strategy=regime.recommended_strategy.value,
        narratives=", ".join(
            sanitize_prompt_input(n, max_length=50) for n in regime.active_narratives
        ) or "None detected",
        symbol=market.symbol if market else "BTC/USDT",
        price=market.price if market else 0,
        rsi=f"{market.rsi:.1f}" if market and market.rsi is not None else "N/A",
        macd=f"{market.macd:.2f}" if market and market.macd is not None else "N/A",
        macd_signal=f"{market.macd_signal:.2f}" if market and market.macd_signal is not None else "N/A",
        bb_upper=market.bbands_upper or 0 if market else 0,
        bb_lower=market.bbands_lower or 0 if market else 0,
        volatility=f"{market.volatility:.4f}" if market and market.volatility is not None else "N/A",
        vol_ratio=f"{market.volume_sma_ratio:.2f}" if market and market.volume_sma_ratio is not None else "N/A",
        total_value=portfolio.total_value if portfolio else 0,
        available_capital=portfolio.available_capital if portfolio else 0,
        open_positions=len(portfolio.open_positions) if portfolio else 0,
        daily_pnl_pct=portfolio.daily_pnl_pct if portfolio else 0,
        strategy_guidelines=guidelines,
    )

    try:
        messages = [
            {"role": "system", "content": "You are a precise trading analyst. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ]

        result = await llm_client.complete_json(messages, model=model)

        try:
            action = Action(result.get("action", "HOLD"))
        except ValueError:
            action = Action.HOLD

        proposal = TradeProposal(
            action=action,
            asset=result.get("asset", market.symbol if market else "BTC/USDT"),
            confidence=float(result.get("confidence", 0.0)),
            size_suggestion=result.get("size_suggestion", "small"),
            reasoning=result.get("reasoning", ""),
            timeframe=result.get("timeframe", "15m"),
            stop_loss_pct=float(result.get("stop_loss_pct", 2.0)),
            take_profit_pct=float(result.get("take_profit_pct", 5.0)),
            key_factors=result.get("key_factors", []),
            regime_alignment=result.get("regime_alignment", ""),
        )

        logger.info(
            "trade_analyzed",
            action=proposal.action.value,
            confidence=proposal.confidence,
            reasoning=proposal.reasoning[:100],
        )

        return {"proposal": proposal}

    except Exception as e:
        logger.error("analyze_failed", error=str(e))
        return {
            "proposal": TradeProposal(
                action=Action.HOLD,
                asset=market.symbol if market else "BTC/USDT",
                confidence=0.0,
                size_suggestion="small",
                reasoning=f"Analysis failed: {e}",
                timeframe="15m",
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                key_factors=["error"],
                regime_alignment="Error fallback",
            )
        }
