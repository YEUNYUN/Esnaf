"""Root conftest — ensures src/ is importable in all environments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add project root to sys.path so 'from src.xxx import ...' works
# regardless of how pip install is configured
sys.path.insert(0, str(Path(__file__).parent))

# Env vars that .env might set, polluting test defaults
_ENV_KEYS = [
    "BINANCE_API_KEY", "BINANCE_SECRET", "BINANCE_TESTNET",
    "GEMINI_API_KEY", "PRIMARY_MODEL", "CHEAP_MODEL", "FALLBACK_MODEL",
    "COPILOT_API_BASE", "PAPER_TRADING", "TRADING_PAIR",
    "CYCLE_INTERVAL", "ANALYSIS_INTERVAL", "LOG_LEVEL",
    "MAX_CAPITAL", "INITIAL_CAPITAL", "TEMPERATURE",
    "CALM_VOLATILITY_THRESHOLD", "CRYPTOPANIC_API_KEY",
]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent .env values from leaking into tests."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
