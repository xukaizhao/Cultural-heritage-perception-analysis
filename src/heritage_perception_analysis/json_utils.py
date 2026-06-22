"""Utilities for robust JSON parsing from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    """Remove common Markdown JSON fences from a response."""

    return _FENCE_RE.sub("", text.strip()).strip()


def extract_json_text(text: str) -> str:
    """Extract the outermost JSON object or array text from a response."""

    cleaned = strip_code_fences(text)
    if not cleaned:
        raise ValueError("Empty LLM response.")

    if cleaned[0] in "[{":
        return cleaned

    starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
    if not starts:
        raise ValueError("No JSON object or array found in LLM response.")

    start = min(starts)
    open_char = cleaned[start]
    close_char = "}" if open_char == "{" else "]"
    end = cleaned.rfind(close_char)
    if end < start:
        raise ValueError("Malformed JSON fragment in LLM response.")

    return cleaned[start : end + 1]


def parse_json_response(text: str) -> Any:
    """Parse a JSON object or array from a possibly fenced LLM response."""

    return json.loads(extract_json_text(text))


def compact_json(data: Any) -> str:
    """Serialize data as compact, deterministic JSON."""

    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pretty_json(data: Any) -> str:
    """Serialize data as readable JSON."""

    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def coerce_score(value: Any) -> int | None:
    """Return an integer score in [1, 10], or None for missing or invalid values."""

    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().upper()
        if cleaned in {"", "N/A", "NA", "NULL", "NONE"}:
            return None
        value = cleaned
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if score < 1 or score > 10:
        return None
    return score


def as_string_list(value: Any) -> list[str]:
    """Coerce a JSON value to a clean list of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
