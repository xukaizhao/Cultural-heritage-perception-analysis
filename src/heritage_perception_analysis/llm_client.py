"""OpenAI-compatible chat completion client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI


def normalize_base_url(base_url: str | None) -> str | None:
    """Normalize OpenAI-compatible base URLs to the API root."""

    if not base_url:
        return None
    cleaned = base_url.rstrip("/")
    suffix = "/chat/completions"
    if cleaned.endswith(suffix):
        return cleaned[: -len(suffix)]
    return cleaned


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for text and vision models."""

    api_key: str
    base_url: str | None
    vision_api_key: str
    vision_base_url: str | None
    text_model: str
    vision_model: str
    temperature: float = 0.0
    json_mode: bool = False

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build configuration from environment variables."""

        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set LLM_API_KEY or OPENAI_API_KEY before running the pipeline."
            )

        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return cls(
            api_key=api_key,
            base_url=normalize_base_url(
                os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            ),
            vision_api_key=os.getenv("VISION_API_KEY") or api_key,
            vision_base_url=normalize_base_url(
                os.getenv("VISION_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
            ),
            text_model=os.getenv("TEXT_MODEL", model),
            vision_model=os.getenv("VISION_MODEL", os.getenv("TEXT_MODEL", model)),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            json_mode=os.getenv("LLM_JSON_MODE", "false").lower() in {"1", "true", "yes"},
        )


class LLMClient:
    """Small wrapper around chat completions with token accounting."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self.text_client = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        self.vision_client = AsyncOpenAI(
            api_key=self.config.vision_api_key,
            base_url=self.config.vision_base_url,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        *,
        json_mode: bool | None = None,
        use_vision_client: bool = False,
    ) -> tuple[str, dict[str, int]]:
        """Return response text and token usage for a chat completion."""

        use_json_mode = self.config.json_mode if json_mode is None else json_mode
        kwargs: dict[str, Any] = {
            "model": model or self.config.text_model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        client = self.vision_client if use_vision_client else self.text_client
        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = response.usage
        token_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        }
        return content, token_usage
