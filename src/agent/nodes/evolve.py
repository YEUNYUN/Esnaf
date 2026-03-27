"""Evolve node — weekly strategy evolution based on performance review.

The agent reviews its week, identifies patterns in wins/losses by regime and
strategy, and proposes adjustments. These proposals are LOGGED for human review,
NOT auto-applied. The human decides which adjustments to make.

This is the "learning loop" that prevents the agent from being static.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from src.llm.client import LLMClient
from src.storage.database import Database

logger = structlog.get_logger()

EVOLUTION_PROMPT = """Review this week's trading performance and propose strategy adjustments.

## Performance Summary
- Total Trades: {total_trades}
- Winning: {winning_trades} ({win_rate:.1%})
- Losing: {losing_trades}
- Net P&L: ${net_pnl:,.2f} ({pnl_pct:+.2f}%)
- Max Drawdown: {max_drawdown:.2f}%

## Trades by Regime
{trades_by_regime}

## Trades by Strategy
{trades_by_strategy}

## Recent Lessons (from daily reflections)
{recent_lessons}

## Current Risk Parameters
- Min confidence threshold: {min_confidence}
- Max per-trade: ${max_per_trade}
- Max daily drawdown: ${max_daily_drawdown}
- Trade cooldown: {cooldown_minutes} minutes

Analyze the patterns. What's working? What's failing? Why?

Respond with ONLY a JSON object:
{{
  "week_summary": "2-3 sentence overall assessment",
  "regime_insights": [
    {{
      "regime": "regime_name",
      "performance": "good | neutral | poor",
      "observation": "what we noticed"
    }}
  ],
  "strategy_insights": [
    {{
      "strategy": "strategy_name",
      "performance": "good | neutral | poor",
      "observation": "what we noticed"
    }}
  ],
  "proposed_adjustments": [
    {{
      "parameter": "what to change",
      "current_value": "current setting",
      "proposed_value": "new setting",
      "reasoning": "why this change",
      "confidence": 0.0 to 1.0,
      "risk_level": "low | medium | high"
    }}
  ],
  "keep_doing": ["things that are working well"],
  "stop_doing": ["things that are hurting performance"]
}}"""


async def evolve_node(
    *,
    llm_client: LLMClient,
    model: str,
    db: Database,
    risk_config: dict,
) -> dict:
    """Run weekly evolution analysis and propose strategy adjustments.

    Returns a dict with the evolution proposal (for human review).
    """
    # Gather this week's trades
    trades = await db.get_recent_trades(limit=100)

    if len(trades) < 5:
        logger.info("evolve_skip", reason="too few trades for meaningful analysis", count=len(trades))
        return {
            "evolution": {
                "status": "skipped",
                "reason": f"Only {len(trades)} trades — need at least 5 for evolution analysis",
            }
        }

    # Compute stats
    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in trades)

    # Group by regime
    by_regime: dict[str, list[dict]] = {}
    for t in trades:
        regime = t.get("regime", "unknown")
        by_regime.setdefault(regime, []).append(t)

    regime_text = ""
    for regime, regime_trades in by_regime.items():
        wins = sum(1 for t in regime_trades if t.get("pnl", 0) > 0)
        pnl = sum(t.get("pnl", 0) for t in regime_trades)
        regime_text += f"- {regime}: {len(regime_trades)} trades, {wins} wins, P&L ${pnl:,.2f}\n"

    # Group by strategy
    by_strategy: dict[str, list[dict]] = {}
    for t in trades:
        strategy = t.get("strategy", "unknown")
        by_strategy.setdefault(strategy, []).append(t)

    strategy_text = ""
    for strategy, strat_trades in by_strategy.items():
        wins = sum(1 for t in strat_trades if t.get("pnl", 0) > 0)
        pnl = sum(t.get("pnl", 0) for t in strat_trades)
        strategy_text += f"- {strategy}: {len(strat_trades)} trades, {wins} wins, P&L ${pnl:,.2f}\n"

    # Gather recent lessons from memory
    memories = await db.get_recent_memories(entry_type="lesson", limit=20)
    lessons_text = "\n".join(f"- {m['content']}" for m in memories) if memories else "No lessons recorded yet."

    # Compute max drawdown from trade sequence
    equity_curve = [0.0]
    for t in trades:
        equity_curve.append(equity_curve[-1] + t.get("pnl", 0))
    peak = max(equity_curve) if equity_curve else 0
    trough = min(equity_curve) if equity_curve else 0
    max_drawdown = ((peak - trough) / peak * 100) if peak > 0 else 0

    initial_capital = risk_config.get("max_capital", 10000)
    pnl_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0

    prompt = EVOLUTION_PROMPT.format(
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=len(winning) / len(trades) if trades else 0,
        net_pnl=total_pnl,
        pnl_pct=pnl_pct,
        max_drawdown=max_drawdown,
        trades_by_regime=regime_text or "No regime data available.",
        trades_by_strategy=strategy_text or "No strategy data available.",
        recent_lessons=lessons_text,
        min_confidence=risk_config.get("min_confidence", 0.65),
        max_per_trade=risk_config.get("max_per_trade", 50),
        max_daily_drawdown=risk_config.get("max_daily_drawdown", 50),
        cooldown_minutes=risk_config.get("cooldown_minutes", 15),
    )

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading strategy analyst reviewing weekly performance. "
                    "Be data-driven and honest. If the strategy is losing money, say so clearly. "
                    "Propose specific, measurable adjustments. Output ONLY valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        result = await llm_client.complete_json(messages, model=model)

        # Store the evolution proposal in memory
        await db.add_memory(
            entry_type="evolution",
            content=result.get("week_summary", ""),
            metadata={
                "timestamp": datetime.now(UTC).isoformat(),
                "proposed_adjustments": result.get("proposed_adjustments", []),
                "keep_doing": result.get("keep_doing", []),
                "stop_doing": result.get("stop_doing", []),
            },
        )

        logger.info(
            "evolution_complete",
            adjustments=len(result.get("proposed_adjustments", [])),
            summary=result.get("week_summary", "")[:100],
        )

        return {
            "evolution": {
                "status": "complete",
                **result,
            }
        }

    except Exception as e:
        logger.error("evolution_failed", error=str(e))
        return {
            "evolution": {
                "status": "error",
                "error": str(e),
            }
        }
