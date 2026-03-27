"""Tests for the backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import Backtester, BacktestResult, generate_benchmark_signals


def _make_price_data(n: int = 200, start_price: float = 50000.0, trend: float = 0.0005) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    returns = rng.normal(trend, 0.02, n)
    prices = start_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "close": prices,
        "open": prices * (1 + rng.normal(0, 0.001, n)),
        "high": prices * (1 + abs(rng.normal(0.005, 0.003, n))),
        "low": prices * (1 - abs(rng.normal(0.005, 0.003, n))),
        "volume": rng.uniform(100, 10000, n),
    })
    return df


class TestBacktester:
    """Test the backtesting engine."""

    def test_buy_hold_returns_market_return(self):
        """Buy-and-hold signal should approximate market return."""
        df = _make_price_data(100)
        signals = pd.Series(1, index=df.index)  # Always long

        bt = Backtester(initial_capital=10000, fee_rate=0, slippage_pct=0)
        result = bt.run(df, signals, "buy_hold")

        # Should be very close to buy-and-hold benchmark
        assert abs(result.total_return_pct - result.buy_hold_return_pct) < 1.0

    def test_no_trades_hold_signal(self):
        """All-zero signals should produce ~0 return (minus noise)."""
        df = _make_price_data(100)
        signals = pd.Series(0, index=df.index)

        bt = Backtester(initial_capital=10000)
        result = bt.run(df, signals, "no_trade")

        assert result.total_trades == 0
        assert abs(result.total_return_pct) < 1.0  # Near zero

    def test_fees_reduce_returns(self):
        """Higher fees should produce lower returns."""
        df = _make_price_data(100)
        # Alternate between long and flat to generate many trades
        signals = pd.Series([1 if i % 10 < 5 else 0 for i in range(100)], index=df.index)

        bt_no_fee = Backtester(initial_capital=10000, fee_rate=0, slippage_pct=0)
        bt_high_fee = Backtester(initial_capital=10000, fee_rate=0.01, slippage_pct=0)

        result_no_fee = bt_no_fee.run(df, signals, "no_fee")
        result_high_fee = bt_high_fee.run(df, signals, "high_fee")

        assert result_no_fee.final_value > result_high_fee.final_value

    def test_result_metrics_valid(self):
        """Check that all metrics are reasonable."""
        df = _make_price_data(200)
        signals = pd.Series([1 if i % 20 < 10 else 0 for i in range(200)], index=df.index)

        bt = Backtester(initial_capital=10000)
        result = bt.run(df, signals, "test")

        assert result.initial_capital == 10000
        assert result.final_value > 0
        assert 0 <= result.win_rate <= 1
        assert result.max_drawdown_pct >= 0
        assert result.total_trades >= 0
        assert isinstance(result.sharpe_ratio, float)

    def test_beats_buy_hold_property(self):
        """beats_buy_hold should correctly compare returns."""
        result = BacktestResult(
            strategy_name="test",
            start_date="2025-01-01",
            end_date="2025-06-01",
            initial_capital=10000,
            final_value=12000,
            total_return_pct=20.0,
            buy_hold_return_pct=15.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            max_drawdown_pct=5.0,
            sharpe_ratio=1.5,
            profit_factor=2.0,
            avg_trade_pnl=200,
        )
        assert result.beats_buy_hold is True


class TestBenchmarkSignals:
    """Test benchmark signal generation."""

    def test_generates_all_benchmarks(self):
        df = _make_price_data(100)
        benchmarks = generate_benchmark_signals(df)

        assert "buy_hold" in benchmarks
        assert "ma_crossover" in benchmarks
        assert "random" in benchmarks

    def test_buy_hold_all_ones(self):
        df = _make_price_data(50)
        benchmarks = generate_benchmark_signals(df)
        assert (benchmarks["buy_hold"] == 1).all()

    def test_ma_crossover_warmup(self):
        """MA crossover should be 0 during warmup period."""
        df = _make_price_data(100)
        benchmarks = generate_benchmark_signals(df)
        assert (benchmarks["ma_crossover"].iloc[:30] == 0).all()

    def test_signals_correct_length(self):
        df = _make_price_data(150)
        benchmarks = generate_benchmark_signals(df)
        for name, sig in benchmarks.items():
            assert len(sig) == 150, f"{name} has wrong length"
