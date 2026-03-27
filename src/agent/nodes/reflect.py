"""Reflect node — post-trade analysis that runs after execution.

Reviews recent trades and extracts lessons for the agent's memory.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from src.agent.sanitize import sanitize_prompt_input
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

REJECTED_REFLECTION_PROMPT = """The risk engine rejected a proposed trade this cycle.

## This Cycle
- Regime: {regime} (confidence: {regime_confidence:.0%})
- Proposed Action: {action}
- Rejection Reason: {validation_reason}

Was the proposal reasonable given market conditions? Was the risk engine right to block it?
Respond with ONLY a JSON object:
{{
  "trades_reviewed": 0,
  "regime_accuracy": "Brief assessment of regime classification.",
  "patterns_noticed": ["pattern1"],
  "lessons_learned": ["lesson1"]
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
    validation_passed = state.get("validation_passed", False)
    execution_result = state.get("execution_result") or {}

    trade_executed = bool(
        execution_result and execution_result.get("status") == "filled"
    )

    # ── SIT_OUT: no trade proposed → minimal note, no LLM call ──────────
    if not proposal:
        regime_label = regime.regime.value if regime else "unknown"
        note = f"Sat out — regime {regime_label}, no trade proposed"
        logger.info("reflection_skipped", reason="sit_out")
        return {
            "reflection": ReflectionEntry(
                timestamp=datetime.now(UTC),
                trades_reviewed=0,
                regime_accuracy="",
                patterns_noticed=[],
                lessons_learned=[note],
            )
        }

    # ── Build prompt based on trade outcome ─────────────────────────────
    recent_trades = await db.get_recent_trades(limit=5)
    trades_text = "No recent trades" if not recent_trades else "\n".join(
        f"- {t['action']} {t['symbol']} at ${t['price']:,.2f} — "
        f"{sanitize_prompt_input(t.get('reasoning', 'N/A'), max_length=80)}"
        for t in recent_trades
    )

    if not validation_passed:
        # Risk engine rejected → shorter, focused prompt
        prompt = REJECTED_REFLECTION_PROMPT.format(
            regime=regime.regime.value if regime else "unknown",
            regime_confidence=regime.regime_confidence if regime else 0,
            action=proposal.action.value,
            validation_reason=sanitize_prompt_input(
                state.get("validation_reason", "N/A"), max_length=300
            ),
        )
        reflection_type = "rejection"
    else:
        # Trade executed (or hold) → full reflection
        prompt = REFLECTION_PROMPT.format(
            regime=regime.regime.value if regime else "unknown",
            regime_confidence=regime.regime_confidence if regime else 0,
            action=proposal.action.value,
            validation="PASSED",
            validation_reason=sanitize_prompt_input(
                state.get("validation_reason", "N/A"), max_length=300
            ),
            execution_status=execution_result.get("status", "N/A"),
            recent_trades=trades_text,
            total_value=portfolio.total_value if portfolio else 0,
            daily_pnl_pct=portfolio.daily_pnl_pct if portfolio else 0,
            trade_count=len(recent_trades),
        )
        reflection_type = "execution" if trade_executed else "hold"

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
            reflection_type=reflection_type,
            lessons=len(reflection.lessons_learned),
            patterns=len(reflection.patterns_noticed),
        )

        return {"reflection": reflection}

    except Exception as e:
        logger.error("reflect_failed", error=str(e))
        return {"reflection": None}
