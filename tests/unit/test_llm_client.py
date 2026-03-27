"""Tests for LLM client and model routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.state import MarketSnapshot, PortfolioState
from src.config import LLMConfig
from src.llm.client import LLMClient, LLMRouter


class TestLLMRouter:
    """Test deterministic model routing logic."""

    def setup_method(self):
        self.config = LLMConfig()

    def test_calm_market_uses_cheap_model(self):
        router = LLMRouter(self.config)
        market = MarketSnapshot(volatility=0.01, rsi=50.0, volume_sma_ratio=1.0)
        portfolio = PortfolioState(open_positions=[])
        assert router.select_model(market, portfolio) == self.config.cheap_model

    def test_high_volatility_uses_primary_model(self):
        router = LLMRouter(self.config)
        market = MarketSnapshot(volatility=0.05, rsi=50.0, volume_sma_ratio=1.0)
        portfolio = PortfolioState(open_positions=[])
        assert router.select_model(market, portfolio) == self.config.primary_model

    def test_open_positions_uses_primary_model(self):
        router = LLMRouter(self.config)
        market = MarketSnapshot(volatility=0.01, rsi=50.0, volume_sma_ratio=1.0)
        portfolio = PortfolioState(open_positions=[{"symbol": "BTC/USDT"}])
        assert router.select_model(market, portfolio) == self.config.primary_model

    def test_extreme_rsi_uses_primary_model(self):
        """RSI < 30 or > 70 is a signal → use primary model."""
        router = LLMRouter(self.config)
        market = MarketSnapshot(volatility=0.01, rsi=25.0, volume_sma_ratio=1.0)
        portfolio = PortfolioState(open_positions=[])
        assert router.select_model(market, portfolio) == self.config.primary_model

    def test_high_volume_uses_primary_model(self):
        """Volume > 2x SMA is a signal."""
        router = LLMRouter(self.config)
        market = MarketSnapshot(volatility=0.01, rsi=50.0, volume_sma_ratio=2.5)
        portfolio = PortfolioState(open_positions=[])
        assert router.select_model(market, portfolio) == self.config.primary_model

    def test_no_market_data_uses_cheap_model(self):
        router = LLMRouter(self.config)
        assert router.select_model(None, None) == self.config.cheap_model


class TestLLMClient:
    """Test LLM client with mocked LiteLLM responses."""

    def setup_method(self):
        self.config = LLMConfig()
        self.client = LLMClient(self.config)

    @pytest.mark.asyncio
    async def test_complete_returns_text(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello world"))]
        mock_response.usage = MagicMock(total_tokens=10)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await self.client.complete([{"role": "user", "content": "Hi"}])
            assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_complete_json_parses_direct_json(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"action": "BUY", "confidence": 0.8}'))]
        mock_response.usage = MagicMock(total_tokens=20)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await self.client.complete_json([{"role": "user", "content": "analyze"}])
            assert result["action"] == "BUY"
            assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_complete_json_extracts_from_markdown(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(
            content='Here is my analysis:\n```json\n{"action": "HOLD", "confidence": 0.5}\n```\nDone.'
        ))]
        mock_response.usage = MagicMock(total_tokens=30)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await self.client.complete_json([{"role": "user", "content": "analyze"}])
            assert result["action"] == "HOLD"

    @pytest.mark.asyncio
    async def test_complete_json_fails_on_invalid(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="I can't provide JSON right now."))]
        mock_response.usage = MagicMock(total_tokens=10)

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response), \
             pytest.raises(ValueError, match="Could not parse JSON"):
            await self.client.complete_json([{"role": "user", "content": "analyze"}])

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        """When primary model fails, should try fallback."""
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("API error")
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="fallback response"))]
            resp.usage = MagicMock(total_tokens=5)
            return resp

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await self.client.complete(
                [{"role": "user", "content": "Hi"}],
                model=self.config.primary_model,
            )
            assert result == "fallback response"
            assert call_count == 2
