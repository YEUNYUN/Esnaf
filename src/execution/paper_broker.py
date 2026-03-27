"""Paper trading broker — simulated execution against live prices.

Same interface as the live broker. The agent code doesn't know
whether it's paper trading or live. Only the config differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import uuid

import structlog

from src.agent.state import Action, PortfolioState, TradeProposal

logger = structlog.get_logger()


@dataclass
class PaperPosition:
    """A simulated open position."""

    id: str
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    value: float
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: str = ""
    unrealized_pnl: float = 0.0


@dataclass
class PaperBroker:
    """Simulated broker for paper trading.

    Tracks positions, capital, and P&L as if executing real trades.
    Uses live prices but simulated fills with realistic fee model.
    """

    initial_capital: float = 10000.0
    fee_rate: float = 0.001  # 0.1% per trade (Binance spot maker fee)

    # Internal state
    _capital: float = 0.0
    _positions: list[PaperPosition] = field(default_factory=list)
    _trade_history: list[dict] = field(default_factory=list)
    _daily_start_value: float = 0.0
    _daily_trades: int = 0
    _total_pnl: float = 0.0

    def __post_init__(self) -> None:
        self._capital = self.initial_capital
        self._daily_start_value = self.initial_capital

    def execute(
        self,
        proposal: TradeProposal,
        current_price: float,
    ) -> dict:
        """Execute a paper trade. Returns execution result dict."""

        if proposal.action == Action.HOLD:
            return {"status": "hold", "message": "No action taken"}

        # Calculate position size
        size_multipliers = {"small": 0.03, "medium": 0.07, "large": 0.10}
        multiplier = size_multipliers.get(proposal.size_suggestion, 0.03)

        if proposal.action == Action.BUY:
            return self._open_long(proposal, current_price, multiplier)
        elif proposal.action == Action.SELL:
            return self._open_short(proposal, current_price, multiplier)
        elif proposal.action == Action.CLOSE_LONG:
            return self._close_positions("long", current_price)
        elif proposal.action == Action.CLOSE_SHORT:
            return self._close_positions("short", current_price)

        return {"status": "error", "message": f"Unknown action: {proposal.action}"}

    def get_portfolio_state(self, current_price: float) -> PortfolioState:
        """Get current portfolio state with unrealized P&L."""
        # Check stop-loss and take-profit first (may close positions)
        self._check_stops(current_price)

        # Update unrealized P&L for all remaining positions
        for pos in self._positions:
            if pos.side == "long":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.quantity

        positions_value = sum(p.unrealized_pnl for p in self._positions)
        total_value = self._capital + sum(p.value for p in self._positions) + positions_value

        daily_pnl = total_value - self._daily_start_value
        daily_pnl_pct = (daily_pnl / self._daily_start_value * 100) if self._daily_start_value else 0.0

        return PortfolioState(
            total_value=total_value,
            available_capital=self._capital,
            open_positions=[
                {
                    "id": p.id,
                    "symbol": p.symbol,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "value": p.value,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for p in self._positions
            ],
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            total_pnl=self._total_pnl + positions_value,
            daily_trades=self._daily_trades,
        )

    def reset_daily_stats(self, current_price: float) -> None:
        """Reset daily P&L tracking. Call at start of each trading day."""
        portfolio = self.get_portfolio_state(current_price)
        self._daily_start_value = portfolio.total_value
        self._daily_trades = 0

    def _open_long(
        self, proposal: TradeProposal, price: float, size_mult: float
    ) -> dict:
        """Open a long position."""
        value = (self._capital + sum(p.value for p in self._positions)) * size_mult
        fee = value * self.fee_rate
        quantity = (value - fee) / price

        if value + fee > self._capital:
            return {"status": "rejected", "message": "Insufficient capital"}

        self._capital -= value
        position = PaperPosition(
            id=str(uuid.uuid4())[:8],
            symbol=proposal.asset,
            side="long",
            entry_price=price,
            quantity=quantity,
            value=value - fee,
            stop_loss=price * (1 - proposal.stop_loss_pct / 100),
            take_profit=price * (1 + proposal.take_profit_pct / 100),
            opened_at=datetime.now(UTC).isoformat(),
        )
        self._positions.append(position)
        self._daily_trades += 1

        result = {
            "status": "filled",
            "position_id": position.id,
            "side": "long",
            "price": price,
            "quantity": quantity,
            "value": value - fee,
            "fee": fee,
        }
        self._trade_history.append(result)

        logger.info("paper_trade_opened", **result)
        return result

    def _open_short(
        self, proposal: TradeProposal, price: float, size_mult: float
    ) -> dict:
        """Open a short position (simulated)."""
        value = (self._capital + sum(p.value for p in self._positions)) * size_mult
        fee = value * self.fee_rate

        if value + fee > self._capital:
            return {"status": "rejected", "message": "Insufficient capital"}

        self._capital -= fee  # Only fee deducted for shorts (margin)
        position = PaperPosition(
            id=str(uuid.uuid4())[:8],
            symbol=proposal.asset,
            side="short",
            entry_price=price,
            quantity=value / price,
            value=value,
            stop_loss=price * (1 + proposal.stop_loss_pct / 100),
            take_profit=price * (1 - proposal.take_profit_pct / 100),
            opened_at=datetime.now(UTC).isoformat(),
        )
        self._positions.append(position)
        self._daily_trades += 1

        result = {
            "status": "filled",
            "position_id": position.id,
            "side": "short",
            "price": price,
            "quantity": position.quantity,
            "value": value,
            "fee": fee,
        }
        self._trade_history.append(result)

        logger.info("paper_trade_opened", **result)
        return result

    def _close_positions(self, side: str, current_price: float) -> dict:
        """Close all positions of a given side."""
        to_close = [p for p in self._positions if p.side == side]
        if not to_close:
            return {"status": "no_positions", "message": f"No {side} positions to close"}

        total_pnl = 0.0
        total_fees = 0.0
        for pos in to_close:
            if side == "long":
                pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - current_price) * pos.quantity

            close_fee = abs(pnl + pos.value) * self.fee_rate
            net_pnl = pnl - close_fee
            total_pnl += net_pnl
            total_fees += close_fee

            self._capital += pos.value + net_pnl
            self._total_pnl += net_pnl
            self._positions.remove(pos)

        self._daily_trades += 1

        result = {
            "status": "closed",
            "side": side,
            "positions_closed": len(to_close),
            "pnl": total_pnl,
            "fees": total_fees,
            "price": current_price,
        }
        self._trade_history.append(result)

        logger.info("paper_positions_closed", **result)
        return result

    def _check_stops(self, current_price: float) -> None:
        """Check and trigger stop-loss / take-profit for all positions."""
        to_close: list[tuple[str, str]] = []  # (side, reason) pairs

        for pos in self._positions:
            if pos.side == "long":
                if pos.stop_loss and current_price <= pos.stop_loss:
                    to_close.append((pos.side, "stop_loss"))
                elif pos.take_profit and current_price >= pos.take_profit:
                    to_close.append((pos.side, "take_profit"))
            else:
                if pos.stop_loss and current_price >= pos.stop_loss:
                    to_close.append((pos.side, "stop_loss"))
                elif pos.take_profit and current_price <= pos.take_profit:
                    to_close.append((pos.side, "take_profit"))

        # Close after iteration to avoid modifying list during iteration
        closed_sides: set[str] = set()
        for side, reason in to_close:
            if side not in closed_sides:
                logger.info("paper_stop_triggered", reason=reason, price=current_price)
                self._close_positions(side, current_price)
                closed_sides.add(side)
