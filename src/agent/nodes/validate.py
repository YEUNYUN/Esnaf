"""Validate node — deterministic risk engine check.

This node is pure Python. No LLM involvement. It enforces hard risk limits
that the LLM cannot bypass, modify, or influence.
"""

from __future__ import annotations

import structlog

from src.agent.state import AgentState
from src.risk.engine import RiskEngine

logger = structlog.get_logger()


async def validate_node(
    state: AgentState,
    *,
    risk_engine: RiskEngine,
) -> dict:
    """Run all risk checks against the proposed trade."""
    proposal = state.get("proposal")
    portfolio = state.get("portfolio")
    regime = state.get("regime")

    if proposal is None:
        return {
            "validation_passed": False,
            "validation_reason": "No proposal to validate",
        }

    result = risk_engine.validate(proposal, portfolio, regime)

    logger.info(
        "validation_result",
        passed=result.passed,
        reason=result.reason,
        action=proposal.action.value if proposal else "none",
    )

    return {
        "validation_passed": result.passed,
        "validation_reason": result.reason,
    }
