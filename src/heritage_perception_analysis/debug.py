"""Debug helpers for printing model inputs and outputs."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


def _summarize_data_uri(data_uri: str) -> str:
    """Return a compact placeholder for a data URI."""

    prefix = data_uri.split(",", 1)[0]
    return f"<{prefix}; {len(data_uri)} characters>"


def sanitize_messages(messages: list[dict[str, Any]], include_image_data_uri: bool = False) -> list[dict[str, Any]]:
    """Return messages safe for terminal printing."""

    if include_image_data_uri:
        return deepcopy(messages)

    sanitized = deepcopy(messages)
    for message in sanitized:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url.startswith("data:image/"):
                    image_url["url"] = _summarize_data_uri(url)
    return sanitized


@dataclass
class ModelIODebugPrinter:
    """Printer for model inputs and outputs grouped by pipeline step."""

    enabled: bool = False
    include_image_data_uri: bool = False

    def print_step(self, step_name: str, messages: list[dict[str, Any]], response_text: str) -> None:
        """Print messages and response text for one model call."""

        if not self.enabled:
            return

        printable_messages = sanitize_messages(
            messages,
            include_image_data_uri=self.include_image_data_uri,
        )
        print(f"\n===== MODEL INPUT: {step_name} =====")
        print(json.dumps(printable_messages, ensure_ascii=False, indent=2))
        print(f"===== MODEL OUTPUT: {step_name} =====")
        print(response_text)
        print(f"===== END MODEL I/O: {step_name} =====\n")
