"""Backtesting framework — validate strategies against historical data.

Uses vectorized backtesting via pandas for speed. Compares agent strategies
against benchmarks: buy-and-hold, simple MA crossover, and random.

This is NOT a realistic simulation — it's a rapid validation tool to catch
obviously bad strategies before paper trading. Real validation requires
paper trading with live data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    buy_hold_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_pnl: float
    trades: list[dict] = field(default_factory=list)

    @property
    def beats_buy_hold(self) -> bool:
        return self.total_return_pct > self.buy_hold_return_pct


class Backtester:
    """Vectorized backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
        slippage_pct: float = 0.05,
    ) -> None:
        self._capital = initial_capital
        self._fee_rate = fee_rate
        self._slippage_pct = slippage_pct / 100

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        strategy_name: str = "agent",
    ) -> BacktestResult:
        """Run a backtest with the given signals.

        Args:
            df: OHLCV DataFrame with 'close' column
            signals: 1 (long), -1 (short), 0 (no position)
            strategy_name: Label for this strategy
        """
        assert len(df) == len(signals), "Signals must align with price data"
        assert "close" in df.columns, "DataFrame must have 'close' column"

        df = df.copy()
        df["signal"] = signals.values

        # Apply slippage
        df["exec_price"] = df["close"] * (1 + self._slippage_pct * np.sign(df["signal"]))

        # Position tracking
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["position_change"] = df["position"].diff().fillna(df["position"])

        # Returns
        df["market_return"] = df["close"].pct_change().fillna(0)
        df["strategy_return"] = df["position"].shift(1).fillna(0) * df["market_return"]

        # Fees on position changes
        df["fee"] = abs(df["position_change"]) * self._fee_rate
        df["strategy_return_net"] = df["strategy_return"] - df["fee"]

        # Equity curves
        df["equity"] = self._capital * (1 + df["strategy_return_net"]).cumprod()
        df["buy_hold_equity"] = self._capital * (1 + df["market_return"]).cumprod()

        # Extract trades
        trades = self._extract_trades(df)

        # Metrics
        final_value = df["equity"].iloc[-1]
        total_return = (final_value / self._capital - 1) * 100
        buy_hold_return = (df["buy_hold_equity"].iloc[-1] / self._capital - 1) * 100

        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(winning) / len(trades) if trades else 0
        avg_pnl = sum(t["pnl"] for t in trades) / len(trades) if trades else 0

        gross_profit = sum(t["pnl"] for t in winning) if winning else 0
        gross_loss = abs(sum(t["pnl"] for t in losing)) if losing else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        peak = df["equity"].expanding().max()
        drawdown = (df["equity"] - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100

        # Sharpe ratio (annualized, crypto trades 365 days)
        daily_returns = df["strategy_return_net"]
        sharpe = (
            daily_returns.mean() / daily_returns.std() * np.sqrt(365)
            if daily_returns.std() > 0
            else 0.0
        )

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=str(df.index[0]),
            end_date=str(df.index[-1]),
            initial_capital=self._capital,
            final_value=final_value,
            total_return_pct=total_return,
            buy_hold_return_pct=buy_hold_return,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_pnl,
            trades=trades,
        )

    def _extract_trades(self, df: pd.DataFrame) -> list[dict]:
        """Extract individual trades from position series."""
        trades = []
        in_trade = False
        entry_idx = 0
        entry_price = 0.0
        side = ""

        for i in range(len(df)):
            pos = df["position"].iloc[i]
            prev_pos = df["position"].iloc[i - 1] if i > 0 else 0

            if pos != 0 and prev_pos == 0:
                in_trade = True
                entry_idx = i
                entry_price = df["exec_price"].iloc[i]
                side = "long" if pos > 0 else "short"

            elif pos == 0 and prev_pos != 0 and in_trade:
                exit_price = df["exec_price"].iloc[i]
                if side == "long":
                    pnl_pct = (exit_price / entry_price - 1) * 100
                else:
                    pnl_pct = (entry_price / exit_price - 1) * 100

                pnl = self._capital * 0.1 * (pnl_pct / 100)

                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })
                in_trade = False

        return trades


def generate_benchmark_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Generate benchmark strategy signals for comparison."""
    n = len(df)

    buy_hold = pd.Series(1, index=df.index)

    ma_fast = df["close"].rolling(10).mean()
    ma_slow = df["close"].rolling(30).mean()
    ma_signal = pd.Series(0, index=df.index)
    ma_signal[ma_fast > ma_slow] = 1
    ma_signal[ma_fast < ma_slow] = -1
    ma_signal.iloc[:30] = 0

    rng = np.random.default_rng(42)
    random_raw = rng.choice([-1, 0, 0, 0, 1], size=n)
    random_signal = pd.Series(random_raw, index=df.index)

    return {
        "buy_hold": buy_hold,
        "ma_crossover": ma_signal,
        "random": random_signal,
    }
