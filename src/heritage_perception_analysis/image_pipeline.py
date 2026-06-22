"""Image visual-quality assessment pipeline."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import time
from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .debug import ModelIODebugPrinter
from .json_utils import coerce_score, compact_json, parse_json_response
from .llm_client import LLMClient
from .memory import MemoryExample, RollingContextMemory
from .prompts import build_image_system_instruction
from .text_pipeline import read_csv_with_fallback


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IGNORED_IMAGE_DIR_NAMES = {"image_memory"}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_MEMORY_DIR = REPO_ROOT / "examples" / "image_memory"
DEFAULT_IMAGE_MEMORY_OUTPUTS: dict[str, dict[str, Any]] = {
    "1.jpg": {
        "visual_analysis": (
            "A grand palace complex with a broad ceremonial plaza, golden roofs, "
            "white stone balustrades, and a vivid blue sky."
        ),
        "score": 10,
        "reason": (
            "The monumental symmetry, bright colors, and iconic palace setting create "
            "a highly distinctive and visually powerful heritage scene."
        ),
    },
    "2.jpg": {
        "visual_analysis": (
            "A section of the Great Wall climbs across green mountain ridges with "
            "watchtowers and stone paths clearly visible."
        ),
        "score": 10,
        "reason": (
            "The dramatic wall line, mountain setting, and strong historic identity "
            "make the image visually compelling and strongly attractive for a visit."
        ),
    },
    "3.jpg": {
        "visual_analysis": (
            "The Temple of Heaven appears under a clear sky with its circular hall, "
            "white terrace, and a large crowd of visitors."
        ),
        "score": 10,
        "reason": (
            "The landmark architecture and open sky are visually memorable."
        ),
    },
}


@dataclass
class ImageAnalysisResult:
    """Structured output for one image."""

    image_path: str
    attraction: str
    visual_analysis: str
    score: int | None
    reason: str
    raw_json: dict[str, Any]
    token_usage: dict[str, int]
    processing_time_seconds: float
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Convert the result to a flat CSV row."""

        return {
            "image_path": self.image_path,
            "attraction": self.attraction,
            "visual_analysis": self.visual_analysis,
            "score": self.score,
            "reason": self.reason,
            "raw_json": compact_json(self.raw_json),
            "prompt_tokens": self.token_usage.get("prompt_tokens", 0),
            "completion_tokens": self.token_usage.get("completion_tokens", 0),
            "total_tokens": self.token_usage.get("total_tokens", 0),
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "error": self.error,
        }


def image_to_data_uri(image_path: str | os.PathLike[str]) -> str:
    """Encode an image as a data URI for multimodal chat APIs."""

    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    with path.open("rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def normalize_image_response(data: Any) -> dict[str, Any]:
    """Normalize the model response for one image."""

    if not isinstance(data, dict):
        return {"visual_analysis": "", "score": None, "reason": ""}
    return {
        "visual_analysis": str(data.get("visual_analysis", data.get("analysis", ""))).strip(),
        "score": coerce_score(data.get("score")),
        "reason": str(data.get("reason", "")).strip(),
    }


def discover_images(root: str | os.PathLike[str], recursive: bool = False) -> list[Path]:
    """Discover images either recursively or within root and first-level folders."""

    root_path = Path(root)
    if recursive:
        candidates = (
            path
            for path in root_path.rglob("*")
            if not any(part in IGNORED_IMAGE_DIR_NAMES for part in path.relative_to(root_path).parts)
        )
    else:
        candidates = [
            path
            for path in root_path.iterdir()
            if not (path.is_dir() and path.name in IGNORED_IMAGE_DIR_NAMES)
        ]
        for child in root_path.iterdir():
            if child.name in IGNORED_IMAGE_DIR_NAMES:
                continue
            if child.is_dir():
                candidates.extend(child.iterdir())

    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def attraction_name_for_image(root: Path, image_path: Path) -> str:
    """Infer an attraction name from the image folder."""

    parent = image_path.parent
    if parent == root:
        name = root.name
    else:
        name = parent.name
    return name.removeprefix("result_")


def load_default_image_memory_examples(
    image_dir: Path = DEFAULT_IMAGE_MEMORY_DIR,
) -> list[MemoryExample]:
    """Load default image examples used to initialize contextual memory."""

    examples: list[MemoryExample] = []
    for filename, verified_output in DEFAULT_IMAGE_MEMORY_OUTPUTS.items():
        image_path = image_dir / filename
        if not image_path.exists():
            continue
        examples.append(
            MemoryExample(
                input_payload=image_to_data_uri(image_path),
                verified_output=verified_output,
                label=filename,
            )
        )
    return examples


class ImageVisualQualityPipeline:
    """Visual quality assessment with rolling contextual memory."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory_size: int = 3,
        print_model_io: bool = False,
        include_image_data_uri: bool = False,
        seed_examples: Iterable[MemoryExample] | None = None,
    ):
        self.llm = llm_client or LLMClient()
        self.memory = RollingContextMemory(
            max_examples=memory_size,
            seed_examples=seed_examples if seed_examples is not None else load_default_image_memory_examples(),
        )
        self.debug_printer = ModelIODebugPrinter(
            enabled=print_model_io,
            include_image_data_uri=include_image_data_uri,
        )

    def _build_messages(self, image_data_uri: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_image_system_instruction(),
            }
        ]

        for example in self.memory.examples():
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Contextual memory example."},
                        {
                            "type": "image_url",
                            "image_url": {"url": str(example.input_payload)},
                        },
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": compact_json(example.verified_output),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Target image."},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        )
        return messages

    async def analyze_image(
        self,
        image_path: str | os.PathLike[str],
        *,
        attraction: str,
        relative_path: str,
    ) -> ImageAnalysisResult:
        """Analyze one image and update memory after successful scoring."""

        start_time = time.time()
        raw_json: dict[str, Any] = {}
        normalized = {"visual_analysis": "", "score": None, "reason": ""}
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        error: str | None = None
        image_data_uri = ""

        try:
            image_data_uri = image_to_data_uri(image_path)
            messages = self._build_messages(image_data_uri)
            response_text, token_usage = await self.llm.complete(
                messages,
                model=self.llm.config.vision_model,
                use_vision_client=True,
            )
            self.debug_printer.print_step(
                "image.visual_quality_assessment",
                messages,
                response_text,
            )
            raw_json = parse_json_response(response_text)
            normalized = normalize_image_response(raw_json)
        except Exception as exc:
            error = str(exc)

        duration = time.time() - start_time
        result = ImageAnalysisResult(
            image_path=relative_path,
            attraction=attraction,
            visual_analysis=normalized["visual_analysis"],
            score=normalized["score"],
            reason=normalized["reason"],
            raw_json=raw_json,
            token_usage=token_usage,
            processing_time_seconds=duration,
            error=error,
        )

        if not error and result.score is not None and image_data_uri:
            self.memory.add(
                image_data_uri,
                {
                    "visual_analysis": result.visual_analysis,
                    "score": result.score,
                    "reason": result.reason,
                },
                label=relative_path,
            )

        return result


async def analyze_image_root(
    image_root: str | os.PathLike[str],
    output_csv: str | os.PathLike[str],
    *,
    recursive: bool = False,
    limit_per_folder: int = 1000,
    resume: bool = True,
    print_model_io: bool = False,
    include_image_data_uri: bool = False,
) -> None:
    """Analyze images under a root folder and write incremental CSV output."""

    root_path = Path(image_root)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = discover_images(root_path, recursive=recursive)
    if limit_per_folder > 0:
        counts: dict[str, int] = {}
        limited: list[Path] = []
        for image in images:
            attraction = attraction_name_for_image(root_path, image)
            counts.setdefault(attraction, 0)
            if counts[attraction] < limit_per_folder:
                limited.append(image)
                counts[attraction] += 1
        images = limited

    processed_paths: set[str] = set()
    if resume and output_path.exists():
        existing = read_csv_with_fallback(output_path)
        if "image_path" in existing.columns:
            if "error" in existing.columns:
                successful_existing = existing[
                    existing["error"].isna()
                    | (existing["error"].astype(str).str.strip() == "")
                ]
            else:
                successful_existing = existing
            processed_paths = set(successful_existing["image_path"].dropna().astype(str))

    pipeline = ImageVisualQualityPipeline(
        print_model_io=print_model_io,
        include_image_data_uri=include_image_data_uri,
    )
    rows_written = output_path.exists() and output_path.stat().st_size > 0

    for image in tqdm(images, desc="Analyzing images", unit="image"):
        relative_path = str(image.relative_to(root_path))
        if resume and relative_path in processed_paths:
            continue

        result = await pipeline.analyze_image(
            image,
            attraction=attraction_name_for_image(root_path, image),
            relative_path=relative_path,
        )
        pd.DataFrame([result.to_row()]).to_csv(
            output_path,
            mode="a" if rows_written else "w",
            header=not rows_written,
            index=False,
            encoding="utf-8-sig",
        )
        rows_written = True


def run_image_root(**kwargs: Any) -> None:
    """Synchronous wrapper for the image pipeline."""

    asyncio.run(analyze_image_root(**kwargs))
