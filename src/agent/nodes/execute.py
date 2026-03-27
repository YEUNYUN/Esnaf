"""Execute node — places orders via the broker (paper or live).

The agent code doesn't know whether it's paper trading or live.
Only the broker instance differs.
"""

from __future__ import annotations

import structlog

from src.agent.state import Action, AgentState
from src.execution.paper_broker import PaperBroker
from src.risk.engine import RiskEngine
from src.storage.database import Database

logger = structlog.get_logger()


async def execute_node(
    state: AgentState,
    *,
    broker: PaperBroker,
    risk_engine: RiskEngine,
    db: Database,
) -> dict:
    """Execute the validated trade proposal."""
    proposal = state.get("proposal")
    validation_passed = state.get("validation_passed", False)
    market = state.get("market")
    regime = state.get("regime")
    cycle_id = state.get("cycle_id", "unknown")

    # Log the decision regardless of outcome
    await db.log_decision(
        cycle_id=cycle_id,
        regime=regime.regime.value if regime else None,
        regime_confidence=regime.regime_confidence if regime else None,
        regime_reasoning=regime.regime_reasoning if regime else None,
        action=proposal.action.value if proposal else "NONE",
        confidence=proposal.confidence if proposal else None,
        reasoning=proposal.reasoning if proposal else None,
        validation_passed=validation_passed,
        validation_reason=state.get("validation_reason", ""),
        model_used=state.get("selected_model"),
    )

    # Log regime classification
    if regime:
        await db.log_regime(
            cycle_id=cycle_id,
            regime=regime.regime.value,
            confidence=regime.regime_confidence,
            reasoning=regime.regime_reasoning,
            strategy=regime.recommended_strategy.value,
        )

    if not validation_passed or not proposal:
        logger.info("execution_skipped", reason=state.get("validation_reason", "validation failed"))
        return {"execution_result": {"status": "skipped", "reason": state.get("validation_reason")}}

    if proposal.action == Action.HOLD:
        return {"execution_result": {"status": "hold", "message": "No action taken"}}

    # Execute via broker
    current_price = market.price if market else 0.0
    result = broker.execute(proposal, current_price)

    if result.get("status") == "filled":
        risk_engine.record_trade()

        await db.log_trade(
            cycle_id=cycle_id,
            symbol=proposal.asset,
            action=proposal.action.value,
            side=result.get("side", "unknown"),
            price=result.get("price", 0.0),
            quantity=result.get("quantity", 0.0),
            value=result.get("value", 0.0),
            fee=result.get("fee", 0.0),
            stop_loss=current_price * (1 - proposal.stop_loss_pct / 100) if proposal.stop_loss_pct else None,
            take_profit=current_price * (1 + proposal.take_profit_pct / 100) if proposal.take_profit_pct else None,
            regime=regime.regime.value if regime else None,
            strategy=regime.recommended_strategy.value if regime else None,
            confidence=proposal.confidence,
            reasoning=proposal.reasoning,
        )

        logger.info(
            "trade_executed",
            action=proposal.action.value,
            price=result.get("price"),
            value=result.get("value"),
            fee=result.get("fee"),
        )

    return {"execution_result": result}
