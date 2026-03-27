"""LLM integration via LiteLLM with hybrid model routing.

Routing logic (deterministic, NOT decided by the LLM):
- Calm markets (low volatility, no positions, no signals) → Gemini Flash (cheap/free)
- Active markets (high volatility, open positions, signals) → Claude via Copilot proxy
- Both unavailable → Ollama local fallback
"""

from __future__ import annotations

import json
from typing import Any

import litellm
import structlog

from src.agent.state import MarketSnapshot, PortfolioState
from src.config import LLMConfig

logger = structlog.get_logger()

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True


class LLMRouter:
    """Hybrid model router — selects the cheapest model that's good enough."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

        # Configure LiteLLM for Copilot proxy
        if config.copilot_api_base:
            litellm.api_base = None  # Don't set globally; pass per-call

    def select_model(
        self,
        market: MarketSnapshot | None = None,
        portfolio: PortfolioState | None = None,
    ) -> str:
        """Deterministic model selection based on market conditions.

        The LLM does NOT choose which model to use. This is pure Python logic.
        """
        # Default to cheap model
        if market is None:
            return self._config.cheap_model

        volatility = market.volatility or 0.0
        has_positions = bool(portfolio and portfolio.open_positions)
        has_signal = self._has_signal(market)

        if volatility < self._config.calm_volatility_threshold and not has_positions and not has_signal:
            model = self._config.cheap_model
            logger.debug("model_routed", model=model, reason="calm_market")
        else:
            model = self._config.primary_model
            logger.debug(
                "model_routed",
                model=model,
                reason="active_market",
                volatility=volatility,
                has_positions=has_positions,
            )

        return model

    def _has_signal(self, market: MarketSnapshot) -> bool:
        """Quick check if there's any notable signal worth deep analysis."""
        if market.rsi is not None and (market.rsi < 30 or market.rsi > 70):
            return True
        return bool(market.volume_sma_ratio is not None and market.volume_sma_ratio > 2.0)


class LLMClient:
    """Unified LLM client wrapping LiteLLM for structured output."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self.router = LLMRouter(config)

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a completion request to the selected model.

        Args:
            messages: Chat messages in OpenAI format
            model: Override model selection (if None, uses router default)
            response_format: JSON schema for structured output

        Returns:
            The LLM's response text
        """
        model = model or self._config.primary_model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self._config.temperature,
        }

        # Copilot proxy needs api_base override
        if "openai/" in model and self._config.copilot_api_base:
            kwargs["api_base"] = self._config.copilot_api_base
            kwargs["api_key"] = "copilot-proxy"  # Proxy handles auth

        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content or ""
            logger.debug(
                "llm_response",
                model=model,
                tokens=response.usage.total_tokens if response.usage else 0,
            )
            return content

        except Exception as e:
            logger.error("llm_error", model=model, error=str(e))
            # Try fallback model
            if model != self._config.fallback_model:
                logger.info("llm_fallback", from_model=model, to_model=self._config.fallback_model)
                kwargs["model"] = self._config.fallback_model
                kwargs.pop("api_base", None)
                kwargs.pop("api_key", None)
                response = await litellm.acompletion(**kwargs)
                return response.choices[0].message.content or ""
            raise

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> dict:
        """Send a completion request and parse the response as JSON.

        Falls back to extracting JSON from markdown code blocks if needed.
        """
        content = await self.complete(messages, model=model)

        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```json" in content:
            start = content.index("```json") + 7
            end = content.index("```", start)
            return json.loads(content[start:end].strip())

        if "```" in content:
            start = content.index("```") + 3
            end = content.index("```", start)
            return json.loads(content[start:end].strip())

        raise ValueError(f"Could not parse JSON from LLM response: {content[:200]}")
