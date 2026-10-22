"""Thin LLM adapter so the rest of the code never imports a provider SDK.

The planner needs exactly one capability: "answer by calling this function with
JSON arguments that match this schema". Both OpenAI and Anthropic support that
natively, so each adapter is ~20 lines.

Pick the provider with env vars (no code change to switch):
    LLM_PROVIDER=openai | anthropic        (default: whichever API key is set)
    OPENAI_API_KEY / OPENAI_MODEL          (default model: gpt-4o-mini)
    ANTHROPIC_API_KEY / ANTHROPIC_MODEL    (default model: claude-sonnet-5-5)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional, Protocol

log = logging.getLogger("auto_job_apply")


class LLM(Protocol):
    async def call_tool(self, system: str, user: str, tool_name: str,
                        tool_description: str, schema: dict) -> dict:
        """Force the model to call `tool_name`; return the parsed arguments ({} on failure)."""


class OpenAILLM:
    def __init__(self, client=None, model: Optional[str] = None):
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()  # reads OPENAI_API_KEY
        self.client = client
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def call_tool(self, system, user, tool_name, tool_description, schema) -> dict:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=[{"type": "function",
                    "function": {"name": tool_name, "description": tool_description, "parameters": schema}}],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        calls = resp.choices[0].message.tool_calls or []
        for call in calls:
            if call.function.name == tool_name:
                try:
                    return json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    log.warning("OpenAI returned malformed tool arguments")
                    return {}
        return {}


class AnthropicLLM:
    def __init__(self, client=None, model: Optional[str] = None):
        if client is None:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        self.client = client
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

    async def call_tool(self, system, user, tool_name, tool_description, schema) -> dict:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            temperature=0,
            system=system,
            tools=[{"name": tool_name, "description": tool_description, "input_schema": schema}],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use" and block.name == tool_name:
                return dict(block.input)
        return {}


def get_llm(provider: Optional[str] = None) -> Optional[LLM]:
    """Build the configured adapter, or None if no key/SDK is available."""
    provider = (provider or os.getenv("LLM_PROVIDER", "")).lower().strip()
    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            log.warning("No LLM configured: set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            return None
    try:
        if provider == "openai":
            return OpenAILLM()
        if provider == "anthropic":
            return AnthropicLLM()
        log.warning("Unknown LLM_PROVIDER=%r (use 'openai' or 'anthropic')", provider)
    except Exception as exc:  # missing SDK or key
        log.warning("Could not start %s client: %s", provider, exc)
    return None