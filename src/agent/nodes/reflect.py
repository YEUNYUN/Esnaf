"""Reflect node — post-trade analysis that runs after execution.

Reviews recent trades and extracts lessons for the agent's memory.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from src.agent.state import AgentState, ReflectionEntry
from src.llm.client import LLMClient
from src.storage.database import Database

logger = structlog.get_logger()

REFLECTION_PROMPT = """Review the most recent trading cycle and extract lessons.

## This Cycle
- Regime: {regime} (confidence: {regime_confidence:.0%})
- Action: {action}
- Validation: {validation} ({validation_reason})
- Execution: {execution_status}

## Recent Trade Performance
{recent_trades}

## Current Portfolio
- Total Value: ${total_value:,.2f}
- Daily P&L: {daily_pnl_pct:.2f}%

Provide a brief reflection. What patterns do you notice? What should we remember?
Respond with ONLY a JSON object:
{{
  "trades_reviewed": {trade_count},
  "regime_accuracy": "Was the regime classification likely correct? Brief assessment.",
  "patterns_noticed": ["pattern1", "pattern2"],
  "lessons_learned": ["lesson1", "lesson2"]
}}"""


async def reflect_node(
    state: AgentState,
    *,
    llm_client: LLMClient,
    model: str,
    db: Database,
) -> dict:
    """Reflect on the current cycle and extract lessons."""
    regime = state.get("regime")
    proposal = state.get("proposal")
    portfolio = state.get("portfolio")
    execution_result = state.get("execution_result", {})

    recent_trades = await db.get_recent_trades(limit=5)
    trades_text = "No recent trades" if not recent_trades else "\n".join(
        f"- {t['action']} {t['symbol']} at ${t['price']:,.2f} — {t.get('reasoning', 'N/A')[:80]}"
        for t in recent_trades
    )

    prompt = REFLECTION_PROMPT.format(
        regime=regime.regime.value if regime else "unknown",
        regime_confidence=regime.regime_confidence if regime else 0,
        action=proposal.action.value if proposal else "NONE",
        validation=("PASSED" if state.get("validation_passed") else "REJECTED"),
        validation_reason=state.get("validation_reason", "N/A"),
        execution_status=execution_result.get("status", "N/A"),
        recent_trades=trades_text,
        total_value=portfolio.total_value if portfolio else 0,
        daily_pnl_pct=portfolio.daily_pnl_pct if portfolio else 0,
        trade_count=len(recent_trades),
    )

    try:
        messages = [
            {"role": "system", "content": "You are a trading performance analyst. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ]

        # Use cheap model for reflection (not critical reasoning)
        result = await llm_client.complete_json(messages, model=model)

        reflection = ReflectionEntry(
            timestamp=datetime.now(UTC),
            trades_reviewed=result.get("trades_reviewed", 0),
            regime_accuracy=result.get("regime_accuracy", ""),
            patterns_noticed=result.get("patterns_noticed", []),
            lessons_learned=result.get("lessons_learned", []),
        )

        # Store lessons in memory
        for lesson in reflection.lessons_learned:
            await db.add_memory(
                entry_type="lesson",
                content=lesson,
                metadata={
                    "cycle_id": state.get("cycle_id"),
                    "regime": regime.regime.value if regime else "unknown",
                },
            )

        logger.info(
            "reflection_complete",
            lessons=len(reflection.lessons_learned),
            patterns=len(reflection.patterns_noticed),
        )

        return {"reflection": reflection}

    except Exception as e:
        logger.error("reflect_failed", error=str(e))
        return {"reflection": None}
