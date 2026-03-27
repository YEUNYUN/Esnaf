"""Live broker — real/testnet exchange execution via CCXT.

Same public interface as PaperBroker so agent code is broker-agnostic.
All CCXT calls are synchronous and wrapped with asyncio.to_thread().
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

import ccxt
import structlog

from src.agent.state import Action, PortfolioState, TradeProposal
from src.config import ExchangeConfig

logger = structlog.get_logger()


@dataclass
class TrackedOrder:
    """An order placed on the exchange, kept for reconciliation."""

    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    amount: float
    price: float | None
    status: str  # "open", "closed", "canceled", "expired"
    filled: float = 0.0
    remaining: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    timestamp: str = ""


@dataclass
class LivePosition:
    """A tracked open position derived from exchange fills."""

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
    order_id: str = ""
    stop_order_ids: list[str] = field(default_factory=list)


@dataclass
class LiveBroker:
    """Live exchange broker via CCXT.

    Mirrors PaperBroker's public interface: execute(), get_portfolio_state(),
    reset_daily_stats(), and update_prices().
    """

    config: ExchangeConfig = field(default_factory=ExchangeConfig)
    fee_rate: float = 0.001

    # Internal state
    _exchange: ccxt.Exchange | None = field(default=None, repr=False)
    _positions: list[LivePosition] = field(default_factory=list)
    _tracked_orders: list[TrackedOrder] = field(default_factory=list)
    _trade_history: list[dict] = field(default_factory=list)
    _daily_start_value: float = 0.0
    _daily_trades: int = 0
    _total_pnl: float = 0.0

    def __post_init__(self) -> None:
        self._exchange = self._create_exchange()

    def _create_exchange(self) -> ccxt.Exchange:
        """Instantiate and configure the CCXT exchange."""
        exchange = ccxt.binance(
            {
                "apiKey": self.config.api_key,
                "secret": self.config.secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        if self.config.testnet:
            exchange.set_sandbox_mode(True)
            logger.info("live_broker_sandbox_enabled")
        logger.info(
            "live_broker_initialized",
            exchange=exchange.id,
            testnet=self.config.testnet,
        )
        return exchange

    # ------------------------------------------------------------------
    # Public interface (mirrors PaperBroker)
    # ------------------------------------------------------------------

    async def execute(
        self,
        proposal: TradeProposal,
        current_price: float,
    ) -> dict:
        """Execute a trade on the exchange. Returns execution result dict."""
        if proposal.action == Action.HOLD:
            return {"status": "hold", "message": "No action taken"}

        size_multipliers = {"small": 0.03, "medium": 0.07, "large": 0.10}
        multiplier = size_multipliers.get(proposal.size_suggestion, 0.03)

        if proposal.action == Action.BUY:
            return await self._open_long(proposal, current_price, multiplier)
        elif proposal.action == Action.SELL:
            return await self._open_short(proposal, current_price, multiplier)
        elif proposal.action == Action.CLOSE_LONG:
            return await self._close_positions("long", current_price, proposal.asset)
        elif proposal.action == Action.CLOSE_SHORT:
            return await self._close_positions("short", current_price, proposal.asset)

        return {"status": "error", "message": f"Unknown action: {proposal.action}"}

    async def get_portfolio_state(self, current_price: float) -> PortfolioState:
        """Fetch balance from exchange and merge with local position tracking."""
        # Reconcile with exchange — detect protective orders filled externally
        await self._sync_exchange_orders()

        await self._check_stops(current_price)

        for pos in self._positions:
            if pos.side == "long":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.quantity

        balance = await self._fetch_balance()
        free_capital = balance.get("USDT", {}).get("free", 0.0)
        total_balance = balance.get("USDT", {}).get("total", 0.0)

        positions_value = sum(p.value for p in self._positions)
        unrealized = sum(p.unrealized_pnl for p in self._positions)
        total_value = total_balance + positions_value + unrealized

        daily_pnl = total_value - self._daily_start_value if self._daily_start_value else 0.0
        daily_pnl_pct = (
            (daily_pnl / self._daily_start_value * 100) if self._daily_start_value else 0.0
        )

        return PortfolioState(
            total_value=total_value,
            available_capital=free_capital,
            open_positions=[
                {
                    "id": p.id,
                    "symbol": p.symbol,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "value": p.value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "order_id": p.order_id,
                }
                for p in self._positions
            ],
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            total_pnl=self._total_pnl + unrealized,
            daily_trades=self._daily_trades,
        )

    async def reset_daily_stats(self, current_price: float) -> None:
        """Reset daily P&L tracking. Call at start of each trading day."""
        portfolio = await self.get_portfolio_state(current_price)
        self._daily_start_value = portfolio.total_value
        self._daily_trades = 0

    async def update_prices(self, prices: dict) -> list[dict]:
        """Check stop-loss/take-profit triggers. Returns list of closed position results."""
        closed: list[dict] = []
        for symbol, price in prices.items():
            results = await self._check_stops_for_symbol(symbol, price)
            closed.extend(results)
        return closed

    # ------------------------------------------------------------------
    # Private — order execution
    # ------------------------------------------------------------------

    async def _open_long(
        self, proposal: TradeProposal, price: float, size_mult: float
    ) -> dict:
        """Place a market buy order on the exchange."""
        balance = await self._fetch_balance()
        free_usdt = balance.get("USDT", {}).get("free", 0.0)
        total_assets = free_usdt + sum(p.value for p in self._positions)

        value = total_assets * size_mult
        if value > free_usdt or value <= 0:
            logger.warning("live_insufficient_balance", required=value, available=free_usdt)
            return {"status": "rejected", "message": "Insufficient balance"}

        quantity = value / price

        try:
            order = await self._create_market_order(proposal.asset, "buy", quantity)
        except ccxt.InsufficientFunds:
            logger.warning("live_insufficient_funds_exchange", symbol=proposal.asset)
            return {"status": "rejected", "message": "Insufficient balance on exchange"}
        except ccxt.ExchangeNotAvailable as exc:
            raise RuntimeError(f"Exchange unavailable (maintenance?): {exc}") from exc
        except ccxt.NetworkError as exc:
            raise RuntimeError(f"Network error placing order: {exc}") from exc

        tracked = self._track_order(order)
        filled_qty = tracked.filled
        filled_cost = tracked.cost
        fee = tracked.fee

        if filled_qty == 0:
            logger.error("live_order_not_filled", order_id=tracked.order_id)
            return {"status": "rejected", "message": "Order not filled"}

        if tracked.remaining > 0:
            logger.warning(
                "live_partial_fill",
                order_id=tracked.order_id,
                filled=filled_qty,
                remaining=tracked.remaining,
            )

        entry_price = filled_cost / filled_qty if filled_qty else price
        position = LivePosition(
            id=tracked.order_id,
            symbol=proposal.asset,
            side="long",
            entry_price=entry_price,
            quantity=filled_qty,
            value=filled_cost,
            stop_loss=entry_price * (1 - proposal.stop_loss_pct / 100),
            take_profit=entry_price * (1 + proposal.take_profit_pct / 100),
            opened_at=datetime.now(UTC).isoformat(),
            order_id=tracked.order_id,
        )
        self._positions.append(position)
        self._daily_trades += 1

        # Place exchange-side protective orders (OCO or stop-loss-limit)
        await self._place_protective_orders(position)

        result = {
            "status": "filled",
            "position_id": position.id,
            "order_id": tracked.order_id,
            "side": "long",
            "price": entry_price,
            "quantity": filled_qty,
            "value": filled_cost,
            "fee": fee,
        }
        self._trade_history.append(result)
        logger.info("live_trade_opened", **result)
        return result

    async def _open_short(
        self, proposal: TradeProposal, price: float, size_mult: float
    ) -> dict:
        """Place a market sell order (spot sell of base asset)."""
        balance = await self._fetch_balance()
        free_usdt = balance.get("USDT", {}).get("free", 0.0)
        total_assets = free_usdt + sum(p.value for p in self._positions)

        value = total_assets * size_mult
        quantity = value / price

        # For spot short, we sell the base asset we hold
        base = proposal.asset.split("/")[0] if "/" in proposal.asset else proposal.asset
        base_free = balance.get(base, {}).get("free", 0.0)

        if quantity > base_free:
            logger.warning(
                "live_insufficient_base_balance",
                required=quantity,
                available=base_free,
                base=base,
            )
            return {"status": "rejected", "message": f"Insufficient {base} balance"}

        try:
            order = await self._create_market_order(proposal.asset, "sell", quantity)
        except ccxt.InsufficientFunds:
            logger.warning("live_insufficient_funds_exchange", symbol=proposal.asset)
            return {"status": "rejected", "message": "Insufficient balance on exchange"}
        except ccxt.ExchangeNotAvailable as exc:
            raise RuntimeError(f"Exchange unavailable (maintenance?): {exc}") from exc
        except ccxt.NetworkError as exc:
            raise RuntimeError(f"Network error placing order: {exc}") from exc

        tracked = self._track_order(order)
        filled_qty = tracked.filled
        filled_cost = tracked.cost
        fee = tracked.fee

        if filled_qty == 0:
            logger.error("live_order_not_filled", order_id=tracked.order_id)
            return {"status": "rejected", "message": "Order not filled"}

        if tracked.remaining > 0:
            logger.warning(
                "live_partial_fill",
                order_id=tracked.order_id,
                filled=filled_qty,
                remaining=tracked.remaining,
            )

        entry_price = filled_cost / filled_qty if filled_qty else price
        position = LivePosition(
            id=tracked.order_id,
            symbol=proposal.asset,
            side="short",
            entry_price=entry_price,
            quantity=filled_qty,
            value=filled_cost,
            stop_loss=entry_price * (1 + proposal.stop_loss_pct / 100),
            take_profit=entry_price * (1 - proposal.take_profit_pct / 100),
            opened_at=datetime.now(UTC).isoformat(),
            order_id=tracked.order_id,
        )
        self._positions.append(position)
        self._daily_trades += 1

        # Place exchange-side protective orders (OCO or stop-loss-limit)
        await self._place_protective_orders(position)

        result = {
            "status": "filled",
            "position_id": position.id,
            "order_id": tracked.order_id,
            "side": "short",
            "price": entry_price,
            "quantity": filled_qty,
            "value": filled_cost,
            "fee": fee,
        }
        self._trade_history.append(result)
        logger.info("live_trade_opened", **result)
        return result

    async def _close_positions(
        self, side: str, current_price: float, symbol: str
    ) -> dict:
        """Close all positions of a given side by placing opposite market orders."""
        to_close = [p for p in self._positions if p.side == side and p.symbol == symbol]
        if not to_close:
            return {"status": "no_positions", "message": f"No {side} positions to close"}

        total_pnl = 0.0
        total_fees = 0.0
        closed_count = 0

        for pos in to_close:
            # Cancel exchange-side protective orders before closing
            await self._cancel_protective_orders(pos)

            close_side = "sell" if side == "long" else "buy"
            try:
                order = await self._create_market_order(pos.symbol, close_side, pos.quantity)
            except ccxt.ExchangeNotAvailable as exc:
                raise RuntimeError(f"Exchange unavailable: {exc}") from exc
            except ccxt.NetworkError as exc:
                raise RuntimeError(f"Network error closing position: {exc}") from exc

            tracked = self._track_order(order)
            fill_price = tracked.cost / tracked.filled if tracked.filled else current_price

            if side == "long":
                pnl = (fill_price - pos.entry_price) * tracked.filled
            else:
                pnl = (pos.entry_price - fill_price) * tracked.filled

            net_pnl = pnl - tracked.fee
            total_pnl += net_pnl
            total_fees += tracked.fee
            self._total_pnl += net_pnl
            self._positions.remove(pos)
            closed_count += 1

        self._daily_trades += 1

        result = {
            "status": "closed",
            "side": side,
            "positions_closed": closed_count,
            "pnl": total_pnl,
            "fees": total_fees,
            "price": current_price,
        }
        self._trade_history.append(result)
        logger.info("live_positions_closed", **result)
        return result

    # ------------------------------------------------------------------
    # Private — stop-loss / take-profit
    # ------------------------------------------------------------------

    async def _check_stops(self, current_price: float) -> None:
        """Check and trigger stops for all positions (single-price mode).

        NOTE: This is a software-side fallback. Primary protection is via
        exchange-side OCO / stop-loss-limit orders placed in
        _place_protective_orders(). This method catches anything that
        slips through (e.g., if exchange orders failed to place).
        """
        triggers: list[tuple[str, str, str]] = []  # (side, reason, symbol)

        for pos in self._positions:
            if pos.side == "long":
                if pos.stop_loss and current_price <= pos.stop_loss:
                    triggers.append((pos.side, "stop_loss", pos.symbol))
                elif pos.take_profit and current_price >= pos.take_profit:
                    triggers.append((pos.side, "take_profit", pos.symbol))
            else:
                if pos.stop_loss and current_price >= pos.stop_loss:
                    triggers.append((pos.side, "stop_loss", pos.symbol))
                elif pos.take_profit and current_price <= pos.take_profit:
                    triggers.append((pos.side, "take_profit", pos.symbol))

        closed: set[tuple[str, str]] = set()
        for side, reason, symbol in triggers:
            key = (side, symbol)
            if key not in closed:
                logger.info("live_stop_triggered", reason=reason, price=current_price)
                await self._close_positions(side, current_price, symbol)
                closed.add(key)

    async def _check_stops_for_symbol(self, symbol: str, price: float) -> list[dict]:
        """Check stops for a specific symbol. Returns closed position results."""
        results: list[dict] = []
        triggers: list[tuple[str, str]] = []

        for pos in self._positions:
            if pos.symbol != symbol:
                continue
            if pos.side == "long":
                if pos.stop_loss and price <= pos.stop_loss:
                    triggers.append(("long", "stop_loss"))
                elif pos.take_profit and price >= pos.take_profit:
                    triggers.append(("long", "take_profit"))
            else:
                if pos.stop_loss and price >= pos.stop_loss:
                    triggers.append(("short", "stop_loss"))
                elif pos.take_profit and price <= pos.take_profit:
                    triggers.append(("short", "take_profit"))

        closed_sides: set[str] = set()
        for side, reason in triggers:
            if side not in closed_sides:
                logger.info("live_stop_triggered", reason=reason, symbol=symbol, price=price)
                result = await self._close_positions(side, price, symbol)
                results.append(result)
                closed_sides.add(side)
        return results

    # ------------------------------------------------------------------
    # Private — exchange-side protective orders (OCO / stop-loss-limit)
    # ------------------------------------------------------------------

    async def _place_protective_orders(self, position: LivePosition) -> list[str]:
        """Place exchange-side protective orders (OCO or stop-loss-limit fallback).

        Strategy:
        1. Try OCO order (combines TP limit + SL stop-limit, one cancels other)
        2. If OCO fails, try stop-loss-limit only (more critical than TP)
        3. If that also fails, log warning — software-side _check_stops() is fallback
        Returns list of order IDs placed on exchange.
        """
        if not position.stop_loss or not position.take_profit:
            logger.info(
                "protective_orders_skipped_no_levels",
                position_id=position.id,
            )
            return []

        assert self._exchange is not None
        symbol_raw = position.symbol.replace("/", "")
        order_ids: list[str] = []

        # Determine order direction: long positions need sell-side protection,
        # short positions need buy-side protection.
        if position.side == "long":
            oco_side = "SELL"
            # SL limit price slightly below trigger for slippage buffer
            sl_limit_price = position.stop_loss * 0.995
        else:
            oco_side = "BUY"
            # SL limit price slightly above trigger for slippage buffer
            sl_limit_price = position.stop_loss * 1.005

        # --- Attempt 1: OCO order ---
        try:
            oco_result = await asyncio.to_thread(
                self._exchange.private_post_order_oco,
                {
                    "symbol": symbol_raw,
                    "side": oco_side,
                    "quantity": position.quantity,
                    "price": str(position.take_profit),
                    "stopPrice": str(position.stop_loss),
                    "stopLimitPrice": str(sl_limit_price),
                    "stopLimitTimeInForce": "GTC",
                },
            )
            for o in oco_result.get("orderReports", []):
                oid = str(o.get("orderId", ""))
                if oid:
                    order_ids.append(oid)
            if not order_ids:
                oco_list_id = str(oco_result.get("orderListId", ""))
                if oco_list_id:
                    order_ids.append(oco_list_id)
            logger.info(
                "protective_oco_placed",
                position_id=position.id,
                symbol=position.symbol,
                order_ids=order_ids,
            )
            position.stop_order_ids = order_ids
            return order_ids
        except (ccxt.InvalidOrder, ccxt.ExchangeError) as exc:
            logger.warning(
                "protective_oco_failed",
                position_id=position.id,
                error=str(exc),
            )

        # --- Attempt 2: Stop-loss-limit only ---
        try:
            sl_order = await asyncio.to_thread(
                self._exchange.create_order,
                position.symbol,
                "STOP_LOSS_LIMIT",
                oco_side.lower(),
                position.quantity,
                sl_limit_price,
                {"stopPrice": position.stop_loss, "timeInForce": "GTC"},
            )
            oid = str(sl_order.get("id", ""))
            if oid:
                order_ids.append(oid)
            logger.info(
                "protective_stop_loss_placed",
                position_id=position.id,
                symbol=position.symbol,
                order_id=oid,
            )
            position.stop_order_ids = order_ids
            return order_ids
        except (ccxt.InvalidOrder, ccxt.ExchangeError) as exc:
            logger.warning(
                "protective_stop_loss_failed_software_fallback",
                position_id=position.id,
                error=str(exc),
            )

        # --- Fallback: software-side stops remain active via _check_stops() ---
        return []

    async def _cancel_protective_orders(self, position: LivePosition) -> None:
        """Cancel all exchange-side protective orders for a position.

        Gracefully handles orders that have already been filled or canceled.
        """
        if not position.stop_order_ids:
            return

        assert self._exchange is not None
        for oid in position.stop_order_ids:
            try:
                await asyncio.to_thread(
                    self._exchange.cancel_order, oid, position.symbol,
                )
                logger.info(
                    "protective_order_canceled",
                    position_id=position.id,
                    order_id=oid,
                )
            except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                logger.info(
                    "protective_order_already_gone",
                    position_id=position.id,
                    order_id=oid,
                )
            except ccxt.ExchangeError as exc:
                logger.warning(
                    "protective_order_cancel_error",
                    position_id=position.id,
                    order_id=oid,
                    error=str(exc),
                )
        position.stop_order_ids = []

    async def _sync_exchange_orders(self) -> None:
        """Reconcile local state with exchange — detect filled protective orders.

        Called at the start of each cycle. If a protective stop-loss or
        take-profit was filled on the exchange, update local tracking.
        """
        assert self._exchange is not None
        positions_to_remove: list[LivePosition] = []

        for pos in self._positions:
            if not pos.stop_order_ids:
                continue
            for oid in pos.stop_order_ids:
                try:
                    order = await asyncio.to_thread(
                        self._exchange.fetch_order, oid, pos.symbol,
                    )
                except (ccxt.OrderNotFound, ccxt.ExchangeError) as exc:
                    logger.warning(
                        "sync_order_fetch_error",
                        order_id=oid,
                        error=str(exc),
                    )
                    continue

                if order.get("status") == "closed" and order.get("filled", 0) > 0:
                    fill_price = (
                        order.get("average")
                        or order.get("price")
                        or pos.entry_price
                    )
                    fee_info = order.get("fee") or {}
                    fee_cost = float(fee_info.get("cost", 0.0) or 0.0)

                    if pos.side == "long":
                        pnl = (fill_price - pos.entry_price) * order["filled"]
                    else:
                        pnl = (pos.entry_price - fill_price) * order["filled"]

                    net_pnl = pnl - fee_cost
                    self._total_pnl += net_pnl

                    logger.info(
                        "protective_order_filled",
                        position_id=pos.id,
                        order_id=oid,
                        fill_price=fill_price,
                        pnl=net_pnl,
                    )
                    positions_to_remove.append(pos)
                    break  # position handled, no need to check other orders

        for pos in positions_to_remove:
            if pos in self._positions:
                self._positions.remove(pos)

    # ------------------------------------------------------------------
    # Private — CCXT wrappers (all via asyncio.to_thread)
    # ------------------------------------------------------------------

    async def _create_market_order(
        self, symbol: str, side: str, amount: float
    ) -> dict:
        """Place a market order via CCXT, wrapped in asyncio.to_thread."""
        assert self._exchange is not None
        logger.info("live_placing_order", symbol=symbol, side=side, amount=amount)
        order = await asyncio.to_thread(
            self._exchange.create_order, symbol, "market", side, amount
        )
        logger.info("live_order_placed", order_id=order.get("id"), status=order.get("status"))
        return order

    async def _fetch_balance(self) -> dict:
        """Fetch account balance via CCXT."""
        assert self._exchange is not None
        try:
            return await asyncio.to_thread(self._exchange.fetch_balance)
        except ccxt.ExchangeNotAvailable as exc:
            raise RuntimeError(f"Exchange unavailable: {exc}") from exc
        except ccxt.NetworkError as exc:
            raise RuntimeError(f"Network error fetching balance: {exc}") from exc

    async def _fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Fetch open orders for reconciliation."""
        assert self._exchange is not None
        try:
            return await asyncio.to_thread(self._exchange.fetch_open_orders, symbol)
        except ccxt.NetworkError as exc:
            raise RuntimeError(f"Network error fetching open orders: {exc}") from exc

    async def _fetch_my_trades(self, symbol: str | None = None) -> list[dict]:
        """Fetch recent trades for reconciliation."""
        assert self._exchange is not None
        try:
            return await asyncio.to_thread(self._exchange.fetch_my_trades, symbol)
        except ccxt.NetworkError as exc:
            raise RuntimeError(f"Network error fetching trades: {exc}") from exc

    # ------------------------------------------------------------------
    # Private — order tracking
    # ------------------------------------------------------------------

    def _track_order(self, order: dict) -> TrackedOrder:
        """Parse a CCXT order response into a TrackedOrder for bookkeeping."""
        fee_info = order.get("fee") or {}
        fee_cost = fee_info.get("cost", 0.0) or 0.0

        tracked = TrackedOrder(
            order_id=str(order.get("id", "")),
            symbol=order.get("symbol", ""),
            side=order.get("side", ""),
            amount=order.get("amount", 0.0) or 0.0,
            price=order.get("average") or order.get("price"),
            status=order.get("status", "unknown"),
            filled=order.get("filled", 0.0) or 0.0,
            remaining=order.get("remaining", 0.0) or 0.0,
            cost=order.get("cost", 0.0) or 0.0,
            fee=float(fee_cost),
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._tracked_orders.append(tracked)

        logger.info(
            "live_order_tracked",
            order_id=tracked.order_id,
            status=tracked.status,
            filled=tracked.filled,
            remaining=tracked.remaining,
        )
        return tracked
