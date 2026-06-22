"""Text review analysis pipeline for cultural value scoring."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .debug import ModelIODebugPrinter
from .dimensions import DIMENSION_KEYS
from .evidence import enforce_verbatim_evidence
from .json_utils import as_string_list, coerce_score, compact_json, parse_json_response
from .llm_client import LLMClient
from .memory import RollingContextMemory
from .prompts import (
    PREDEFINED_TEXT_EXAMPLES,
    build_preliminary_extraction_prompt,
    build_verification_scoring_prompt,
)


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


@dataclass
class TextAnalysisResult:
    """Structured output for one review."""

    review: str
    preliminary: dict[str, Any]
    verified: dict[str, Any]
    token_usage: dict[str, int]
    processing_time_seconds: float
    error: str | None = None

    @property
    def identified_dimensions(self) -> list[str]:
        return [
            key
            for key in DIMENSION_KEYS
            if self.verified.get(key, {}).get("evidence")
        ]

    def to_row(self) -> dict[str, Any]:
        """Convert the result to a flat CSV row."""

        row: dict[str, Any] = {
            "review_text": self.review,
            "identified_dimensions": compact_json(self.identified_dimensions),
            "preliminary_json": compact_json(self.preliminary),
            "verified_json": compact_json(self.verified),
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "error": self.error,
        }
        for key in DIMENSION_KEYS:
            item = self.verified.get(key, {})
            row[f"{key}_score"] = item.get("score")
            row[f"verified_{key}_evidence"] = compact_json(item.get("evidence", []))
            row[f"{key}_status"] = item.get("status", "N/A")
        row.update(self.token_usage)
        return row


def read_csv_with_fallback(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a CSV file using common encodings."""

    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - depends on file encoding
            last_error = exc
    raise RuntimeError(f"Could not read CSV file: {path}") from last_error


def empty_verified_payload() -> dict[str, dict[str, Any]]:
    """Return an empty verified payload for all dimensions."""

    return {
        key: {
            "evidence": [],
            "score": None,
            "status": "N/A",
            "removed_nonverbatim": [],
            "removed_irrelevant": [],
        }
        for key in DIMENSION_KEYS
    }


def _find_dimension_item(data: dict[str, Any], key: str) -> Any:
    """Find a dimension payload with tolerant key matching."""

    if key in data:
        return data[key]
    normalized_key = key.replace("_", "").lower()
    for candidate_key, candidate_value in data.items():
        normalized_candidate = str(candidate_key).replace("_", "").replace(" ", "").lower()
        if normalized_candidate in {normalized_key, f"{normalized_key}value"}:
            return candidate_value
    return None


def normalize_preliminary_payload(data: Any) -> dict[str, dict[str, Any]]:
    """Normalize preliminary extraction output to the canonical schema."""

    normalized = {
        key: {"rationale": "", "evidence": []}
        for key in DIMENSION_KEYS
    }
    if not isinstance(data, dict):
        return normalized

    for key in DIMENSION_KEYS:
        item = _find_dimension_item(data, key)
        if isinstance(item, dict):
            evidence = item.get("evidence", item.get("sentences", []))
            normalized[key] = {
                "rationale": str(item.get("rationale", item.get("reasoning", ""))).strip(),
                "evidence": as_string_list(evidence),
            }
        elif isinstance(item, list):
            normalized[key] = {"rationale": "", "evidence": as_string_list(item)}
        elif isinstance(item, str) and item.strip():
            normalized[key] = {"rationale": "", "evidence": [item.strip()]}
    return normalized


def normalize_verified_payload(review: str, data: Any) -> dict[str, dict[str, Any]]:
    """Normalize verification output and enforce verbatim evidence."""

    normalized = empty_verified_payload()
    if not isinstance(data, dict):
        return normalized

    for key in DIMENSION_KEYS:
        item = _find_dimension_item(data, key)
        if not isinstance(item, dict):
            continue

        evidence = as_string_list(item.get("evidence", item.get("verified_evidence", [])))
        kept, removed_by_code = enforce_verbatim_evidence(review, evidence)
        removed_nonverbatim = as_string_list(item.get("removed_nonverbatim", []))
        removed_nonverbatim.extend(x for x in removed_by_code if x not in removed_nonverbatim)
        score = coerce_score(item.get("score"))
        status = str(item.get("status", "valid" if kept else "N/A")).strip() or "N/A"

        if not kept:
            score = None
            status = "N/A"
        elif score is None:
            status = "valid"

        normalized[key] = {
            "evidence": kept,
            "score": score,
            "status": status,
            "removed_nonverbatim": removed_nonverbatim,
            "removed_irrelevant": as_string_list(item.get("removed_irrelevant", [])),
        }
    return normalized


class TextCulturalValuePipeline:
    """Two-agent review pipeline with rolling contextual memory."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory_size: int = 3,
        print_model_io: bool = False,
    ):
        self.llm = llm_client or LLMClient()
        self.memory = RollingContextMemory(
            max_examples=memory_size,
            seed_examples=PREDEFINED_TEXT_EXAMPLES,
        )
        self.debug_printer = ModelIODebugPrinter(enabled=print_model_io)

    async def preliminary_extract(self, review: str) -> tuple[dict[str, Any], dict[str, int]]:
        """Run the preliminary extraction agent."""

        prompt = build_preliminary_extraction_prompt(review, self.memory.examples())
        messages = [{"role": "user", "content": prompt}]
        response_text, token_usage = await self.llm.complete(
            messages,
            model=self.llm.config.text_model,
        )
        self.debug_printer.print_step("text.preliminary_extraction", messages, response_text)
        return normalize_preliminary_payload(parse_json_response(response_text)), {
            "preliminary_prompt_tokens": token_usage.get("prompt_tokens", 0),
            "preliminary_completion_tokens": token_usage.get("completion_tokens", 0),
            "preliminary_total_tokens": token_usage.get("total_tokens", 0),
        }

    async def verify_and_score(
        self,
        review: str,
        preliminary: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Run the verification and scoring agent."""

        prompt = build_verification_scoring_prompt(review, preliminary, self.memory.examples())
        messages = [{"role": "user", "content": prompt}]
        response_text, token_usage = await self.llm.complete(
            messages,
            model=self.llm.config.text_model,
        )
        self.debug_printer.print_step("text.verification_scoring", messages, response_text)
        return normalize_verified_payload(review, parse_json_response(response_text)), {
            "verification_prompt_tokens": token_usage.get("prompt_tokens", 0),
            "verification_completion_tokens": token_usage.get("completion_tokens", 0),
            "verification_total_tokens": token_usage.get("total_tokens", 0),
        }

    async def analyze_review(self, review: str) -> TextAnalysisResult:
        """Analyze one review and update memory after a successful result."""

        start_time = time.time()
        preliminary: dict[str, Any] = normalize_preliminary_payload({})
        verified: dict[str, Any] = empty_verified_payload()
        token_usage = {
            "preliminary_prompt_tokens": 0,
            "preliminary_completion_tokens": 0,
            "preliminary_total_tokens": 0,
            "verification_prompt_tokens": 0,
            "verification_completion_tokens": 0,
            "verification_total_tokens": 0,
            "total_tokens": 0,
        }
        error: str | None = None

        try:
            preliminary, preliminary_tokens = await self.preliminary_extract(review)
            token_usage.update(preliminary_tokens)
            verified, verification_tokens = await self.verify_and_score(review, preliminary)
            token_usage.update(verification_tokens)
            token_usage["total_tokens"] = (
                token_usage["preliminary_total_tokens"]
                + token_usage["verification_total_tokens"]
            )
        except Exception as exc:
            error = str(exc)

        duration = time.time() - start_time
        result = TextAnalysisResult(
            review=review,
            preliminary=preliminary,
            verified=verified,
            token_usage=token_usage,
            processing_time_seconds=duration,
            error=error,
        )

        if not error and result.identified_dimensions:
            self.memory.add(review, verified)

        return result


async def analyze_text_csv(
    input_csv: str | os.PathLike[str],
    output_csv: str | os.PathLike[str],
    *,
    text_column: str = "comments",
    limit: int = 0,
    resume: bool = True,
    drop_duplicates: bool = True,
    print_model_io: bool = False,
) -> None:
    """Analyze reviews from a CSV file and write an incremental CSV output."""

    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_csv_with_fallback(input_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Input CSV must contain a '{text_column}' column. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_column]).copy()
    df[text_column] = df[text_column].astype(str)
    if drop_duplicates:
        df = df.drop_duplicates(subset=[text_column])
    if limit > 0:
        df = df.head(limit)

    processed_reviews: set[str] = set()
    if resume and output_path.exists():
        existing = read_csv_with_fallback(output_path)
        if "review_text" in existing.columns:
            if "error" in existing.columns:
                successful_existing = existing[
                    existing["error"].isna()
                    | (existing["error"].astype(str).str.strip() == "")
                ]
            else:
                successful_existing = existing
            processed_reviews = set(successful_existing["review_text"].dropna().astype(str))

    pipeline = TextCulturalValuePipeline(print_model_io=print_model_io)
    rows_written = output_path.exists() and output_path.stat().st_size > 0

    for review in tqdm(df[text_column].tolist(), desc="Analyzing reviews", unit="review"):
        if resume and review in processed_reviews:
            continue
        result = await pipeline.analyze_review(review)
        row = result.to_row()
        pd.DataFrame([row]).to_csv(
            output_path,
            mode="a" if rows_written else "w",
            header=not rows_written,
            index=False,
            encoding="utf-8-sig",
        )
        rows_written = True


def run_text_csv(**kwargs: Any) -> None:
    """Synchronous wrapper for the text CSV pipeline."""

    asyncio.run(analyze_text_csv(**kwargs))
