"""Prompt input sanitization — defense against prompt injection.

All user-controlled or LLM-generated text that gets re-injected into
prompts MUST pass through sanitize_prompt_input() first.
"""

from __future__ import annotations

import re

# Patterns that look like prompt injection attempts
_INJECTION_PATTERNS = re.compile(
    r"^(Ignore|System:|You are|IMPORTANT:|Override)",
    re.IGNORECASE | re.MULTILINE,
)

# Control characters except newline (\x0a)
_CONTROL_CHARS = re.compile(r"[\x00-\x09\x0b-\x1f]")

# Consecutive newlines (3+)
_MULTI_NEWLINES = re.compile(r"\n{2,}")


def sanitize_prompt_input(text: str, max_length: int = 200) -> str:
    """Sanitize text before injecting it into an LLM prompt.

    1. Strip control characters (\\x00-\\x1f except \\n)
    2. Truncate to *max_length*
    3. Remove lines that look like prompt injection
    4. Collapse multiple newlines into one
    5. Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""

    # 1. Strip control characters
    text = _CONTROL_CHARS.sub("", text)

    # 2. Truncate
    if len(text) > max_length:
        text = text[:max_length]

    # 3. Remove injection-like lines
    text = _INJECTION_PATTERNS.sub("", text)

    # 4. Collapse multiple newlines
    text = _MULTI_NEWLINES.sub("\n", text)

    # 5. Strip whitespace
    return text.strip()
