"""Unified multi-provider LLM dispatch.

Routes calls to Google Gemini, OpenAI, or Anthropic based on the model name prefix:
    "gemini-*" → Google GenAI
    "gpt-*"    → OpenAI
    "claude-*" → Anthropic

All providers expose the same interface via llm_call().
Token usage is tracked globally and can be retrieved with get_token_usage().
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


# ── Token tracking ────────────────────────────────────────────────────────────


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0
    per_call: list[dict] = field(default_factory=list)

    def record(self, model: str, input_t: int, output_t: int) -> None:
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.total_calls += 1
        self.per_call.append({
            "model": model,
            "input_tokens": input_t,
            "output_tokens": output_t,
        })

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> dict:
        return {
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
        }

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_calls = 0
        self.per_call = []


_token_usage = TokenUsage()


def get_token_usage() -> TokenUsage:
    """Return the global token usage tracker."""
    return _token_usage


def reset_token_usage() -> None:
    """Reset the global token usage tracker."""
    _token_usage.reset()


# ── Lazy-initialised clients ──────────────────────────────────────────────────

_gemini_client = None
_openai_client = None
_anthropic_client = None


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client()
        log.info("Gemini client initialised")
    return _gemini_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
        log.info("OpenAI client initialised")
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic()
        log.info("Anthropic client initialised")
    return _anthropic_client


# ── Provider dispatch ─────────────────────────────────────────────────────────


def _call_gemini(
    prompt: str,
    model: str,
    system_instruction: str | None,
    temperature: float,
) -> str:
    from google.genai import types

    client = _get_gemini()
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    usage = getattr(response, "usage_metadata", None)
    if usage:
        _token_usage.record(
            model,
            getattr(usage, "prompt_token_count", 0),
            getattr(usage, "candidates_token_count", 0),
        )

    return response.text


def _call_openai(
    prompt: str,
    model: str,
    system_instruction: str | None,
    temperature: float,
    reasoning_effort: str | None = None,
) -> str:
    client = _get_openai()
    input_messages = []
    if system_instruction:
        input_messages.append({"role": "system", "content": system_instruction})
    input_messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model, "input": input_messages}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    try:
        response = client.responses.create(**kwargs)
    except Exception as e:
        if reasoning_effort and "reasoning" in str(e).lower():
            log.info("Model %s: effort=%s not supported, retrying without", model, reasoning_effort)
            kwargs.pop("reasoning", None)
            response = client.responses.create(**kwargs)
        else:
            raise

    usage = getattr(response, "usage", None)
    if usage:
        _token_usage.record(
            model,
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0),
        )

    return response.output_text


def _call_anthropic(
    prompt: str,
    model: str,
    system_instruction: str | None,
    temperature: float,
) -> str:
    client = _get_anthropic()
    kwargs = {
        "model": model,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_instruction:
        kwargs["system"] = system_instruction

    response = client.messages.create(**kwargs)

    usage = getattr(response, "usage", None)
    if usage:
        _token_usage.record(
            model,
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0),
        )

    return response.content[0].text


# ── Public interface ──────────────────────────────────────────────────────────


_REASONING_EFFORT: dict[str, str] = {
    "gpt-5.3": "medium",
    "gpt-5.3-chat-latest": "medium",
    "gpt-5.4": None,
    "gpt-5.4-mini": None,
}


def llm_call(
    prompt: str,
    model: str,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    reasoning_effort: str | None = None,
) -> str:
    """Make an LLM call, routing to the correct provider based on model prefix.

    Token usage is automatically tracked and accessible via get_token_usage().
    """
    if model.startswith("gemini"):
        return _call_gemini(prompt, model, system_instruction, temperature)
    elif model.startswith("gpt"):
        effort = reasoning_effort or _REASONING_EFFORT.get(model)
        return _call_openai(prompt, model, system_instruction, temperature, effort)
    elif model.startswith("claude"):
        return _call_anthropic(prompt, model, system_instruction, temperature)
    else:
        raise ValueError(
            f"Unknown model prefix: {model!r}. "
            "Model name must start with 'gemini-', 'gpt-', or 'claude-'."
        )
