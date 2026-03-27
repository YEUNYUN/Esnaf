"""Deterministic risk engine.

This is the guardian. Every trade proposal from the LLM must pass through
this engine. The LLM cannot modify, bypass, or influence these checks.

Design principle: better to miss a good trade than to take a bad one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from src.agent.state import (
    Action,
    PortfolioState,
    Regime,
    RegimeClassification,
    TradeProposal,
)
from src.config import RiskConfig

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result of risk validation."""

    passed: bool
    reason: str
    proposal: TradeProposal | None = None


@dataclass
class RiskEngine:
    """Deterministic risk engine — all hard limits enforced here.

    The LLM proposes, the risk engine disposes. No exceptions.
    """

    config: RiskConfig
    _recent_trade_timestamps: list[float] = field(default_factory=list)
    _kill_switch_active: bool = False

    def validate(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioState,
        regime: RegimeClassification | None,
    ) -> ValidationResult:
        """Run all risk checks against a trade proposal.

        Returns ValidationResult with passed=False and reason if any check fails.
        Checks are ordered from most critical to least critical.
        """
        # Kill switch — if total drawdown exceeds threshold, ALL trading stops
        if self._kill_switch_active:
            return ValidationResult(
                passed=False,
                reason="KILL SWITCH ACTIVE — trading halted due to excessive drawdown. "
                "Manual reset required.",
            )

        if abs(portfolio.daily_pnl_pct) >= self.config.kill_switch_drawdown_pct:
            self._kill_switch_active = True
            logger.critical(
                "kill_switch_triggered",
                daily_pnl_pct=portfolio.daily_pnl_pct,
                threshold=self.config.kill_switch_drawdown_pct,
            )
            return ValidationResult(
                passed=False,
                reason=f"KILL SWITCH TRIGGERED — daily drawdown {portfolio.daily_pnl_pct:.1f}% "
                f"exceeds limit {self.config.kill_switch_drawdown_pct}%",
            )

        # Daily drawdown limit
        if abs(portfolio.daily_pnl_pct) >= self.config.max_daily_drawdown_pct:
            return ValidationResult(
                passed=False,
                reason=f"Daily drawdown {portfolio.daily_pnl_pct:.1f}% "
                f"hit limit {self.config.max_daily_drawdown_pct}%",
            )

        # HOLD actions always pass (no risk)
        if proposal.action == Action.HOLD:
            return ValidationResult(passed=True, reason="HOLD — no action needed", proposal=proposal)

        # Regime confidence check — sit out when unsure
        if regime is None or regime.regime_confidence < self.config.min_regime_confidence:
            confidence = regime.regime_confidence if regime else 0.0
            return ValidationResult(
                passed=False,
                reason=f"Regime confidence {confidence:.2f} below minimum "
                f"{self.config.min_regime_confidence}. Sitting out.",
            )

        # Trade confidence check
        if proposal.confidence < self.config.min_confidence_threshold:
            return ValidationResult(
                passed=False,
                reason=f"Trade confidence {proposal.confidence:.2f} below "
                f"threshold {self.config.min_confidence_threshold}",
            )

        # Counter-trend check — higher bar for trading against the regime
        if self._is_counter_trend(proposal, regime) and proposal.confidence < self.config.counter_trend_confidence:
                return ValidationResult(
                    passed=False,
                    reason=f"Counter-trend trade ({proposal.action.value} in "
                    f"{regime.regime.value}) needs confidence >= "
                    f"{self.config.counter_trend_confidence}, got {proposal.confidence:.2f}",
                )

        # Position size check
        position_value = self._estimate_position_value(proposal, portfolio)
        max_position_value = portfolio.total_value * (self.config.max_position_pct / 100)
        if position_value > max_position_value:
            return ValidationResult(
                passed=False,
                reason=f"Position value ${position_value:.2f} exceeds max "
                f"${max_position_value:.2f} ({self.config.max_position_pct}% of portfolio)",
            )

        # Total exposure check
        current_exposure = sum(
            abs(p.get("value", 0)) for p in portfolio.open_positions
        )
        max_exposure = portfolio.total_value * (self.config.max_total_exposure_pct / 100)
        if current_exposure + position_value > max_exposure:
            return ValidationResult(
                passed=False,
                reason=f"Total exposure ${current_exposure + position_value:.2f} "
                f"would exceed limit ${max_exposure:.2f}",
            )

        # Daily trade count
        if portfolio.daily_trades >= self.config.max_trades_per_day:
            return ValidationResult(
                passed=False,
                reason=f"Daily trade limit reached ({self.config.max_trades_per_day})",
            )

        # Cooldown check
        if self._in_cooldown():
            return ValidationResult(
                passed=False,
                reason=f"Trade cooldown active — wait {self.config.trade_cooldown_minutes}min "
                f"between trades",
            )

        # Available capital check
        if proposal.action == Action.BUY and position_value > portfolio.available_capital:
            return ValidationResult(
                passed=False,
                reason=f"Insufficient capital: need ${position_value:.2f}, "
                f"have ${portfolio.available_capital:.2f}",
            )

        # All checks passed
        logger.info(
            "trade_validated",
            action=proposal.action.value,
            asset=proposal.asset,
            confidence=proposal.confidence,
            regime=regime.regime.value,
        )
        return ValidationResult(passed=True, reason="All risk checks passed", proposal=proposal)

    def record_trade(self) -> None:
        """Record that a trade was executed (for cooldown tracking)."""
        self._recent_trade_timestamps.append(time.time())
        # Keep only last 24h of timestamps
        cutoff = time.time() - 86400
        self._recent_trade_timestamps = [
            t for t in self._recent_trade_timestamps if t > cutoff
        ]

    def reset_kill_switch(self) -> None:
        """Manually reset the kill switch. Requires human decision."""
        logger.warning("kill_switch_reset", by="manual")
        self._kill_switch_active = False

    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch_active

    def _is_counter_trend(
        self, proposal: TradeProposal, regime: RegimeClassification
    ) -> bool:
        """Check if a trade goes against the classified regime."""
        if regime.regime == Regime.BEAR_TREND and proposal.action == Action.BUY:
            return True
        return bool(regime.regime == Regime.BULL_TREND and proposal.action == Action.SELL)

    def _estimate_position_value(
        self, proposal: TradeProposal, portfolio: PortfolioState
    ) -> float:
        """Estimate the dollar value of a proposed position."""
        size_multipliers = {"small": 0.03, "medium": 0.07, "large": 0.10}
        multiplier = size_multipliers.get(proposal.size_suggestion, 0.03)
        return portfolio.total_value * multiplier

    def _in_cooldown(self) -> bool:
        """Check if we're in the cooldown period after a recent trade."""
        if not self._recent_trade_timestamps:
            return False
        last_trade = max(self._recent_trade_timestamps)
        cooldown_seconds = self.config.trade_cooldown_minutes * 60
        return (time.time() - last_trade) < cooldown_seconds
