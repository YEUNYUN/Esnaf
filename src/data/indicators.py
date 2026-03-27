"""Technical indicator computation.

All indicators are computed locally in plain Python (via pandas-ta).
The LLM does NOT compute indicators — it receives them pre-computed.
This ensures indicators are deterministic, free, and fast.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta
import structlog

from src.agent.state import MarketSnapshot

logger = structlog.get_logger()


def compute_indicators(snapshot: MarketSnapshot) -> MarketSnapshot:
    """Compute technical indicators from OHLCV data and attach to snapshot.

    Modifies the snapshot in place and returns it.
    Requires at least 30 candles for reliable indicator values.
    """
    if not snapshot.ohlcv or len(snapshot.ohlcv) < 30:
        logger.warning("insufficient_candles", count=len(snapshot.ohlcv) if snapshot.ohlcv else 0)
        return snapshot

    # Convert OHLCV to DataFrame
    df = pd.DataFrame(
        snapshot.ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    # RSI (14-period)
    rsi = ta.rsi(df["close"], length=14)
    if rsi is not None and not rsi.empty:
        snapshot.rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

    # MACD (12, 26, 9)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        macd_col = [c for c in macd.columns if "MACD_" in c and "MACDs" not in c and "MACDh" not in c]
        signal_col = [c for c in macd.columns if "MACDs_" in c]
        if macd_col:
            val = macd[macd_col[0]].iloc[-1]
            snapshot.macd = float(val) if not pd.isna(val) else None
        if signal_col:
            val = macd[signal_col[0]].iloc[-1]
            snapshot.macd_signal = float(val) if not pd.isna(val) else None

    # Bollinger Bands (20, 2)
    bbands = ta.bbands(df["close"], length=20, std=2)
    if bbands is not None and not bbands.empty:
        upper_col = [c for c in bbands.columns if "BBU_" in c]
        lower_col = [c for c in bbands.columns if "BBL_" in c]
        if upper_col:
            val = bbands[upper_col[0]].iloc[-1]
            snapshot.bbands_upper = float(val) if not pd.isna(val) else None
        if lower_col:
            val = bbands[lower_col[0]].iloc[-1]
            snapshot.bbands_lower = float(val) if not pd.isna(val) else None

    # Volatility (ATR-based, 14-period, as % of price)
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    if atr is not None and not atr.empty:
        atr_val = atr.iloc[-1]
        if not pd.isna(atr_val) and snapshot.price > 0:
            snapshot.volatility = float(atr_val / snapshot.price)

    # Volume SMA ratio (current volume vs 20-period SMA)
    vol_sma = ta.sma(df["volume"], length=20)
    if vol_sma is not None and not vol_sma.empty:
        sma_val = vol_sma.iloc[-1]
        current_vol = df["volume"].iloc[-1]
        if not pd.isna(sma_val) and sma_val > 0:
            snapshot.volume_sma_ratio = float(current_vol / sma_val)

    logger.debug(
        "indicators_computed",
        rsi=snapshot.rsi,
        macd=snapshot.macd,
        volatility=snapshot.volatility,
    )
    return snapshot
