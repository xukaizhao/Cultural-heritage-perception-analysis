"""Prompt builders for text and image analysis."""

from __future__ import annotations

from typing import Any

from .dimensions import DIMENSION_KEYS, dimension_definition_block
from .json_utils import pretty_json
from .memory import MemoryExample


TEXT_OUTPUT_SCHEMA = {
    key: {
        "rationale": "Brief public rationale. Do not include private chain-of-thought.",
        "evidence": ["Exact sentence or phrase copied verbatim from the review."],
    }
    for key in DIMENSION_KEYS
}

VERIFIED_OUTPUT_SCHEMA = {
    key: {
        "evidence": ["Verified exact sentence or phrase from the original review."],
        "score": "Integer from 1 to 10, or null when no valid evidence remains.",
        "status": "valid or N/A",
        "removed_nonverbatim": ["Evidence removed because it was not copied verbatim."],
        "removed_irrelevant": ["Evidence removed because it did not fit the definition."],
    }
    for key in DIMENSION_KEYS
}


PREDEFINED_TEXT_EXAMPLES: tuple[MemoryExample, ...] = (
    MemoryExample(
        input_payload=(
            "The Imperial Palace, the Forbidden City, over 700,000 square meters, "
            "represents the highest architectural standard of the Ming and Qing "
            "dynasties. Only shock, reverence, solemnity, and amazement. A "
            "must-visit place when in Beijing!"
        ),
        verified_output={
            "social": {
                "evidence": ["Only shock, reverence, solemnity, and amazement"],
                "score": 10,
                "status": "valid",
            },
            "economic": {
                "evidence": ["must-visit place when in Beijing!"],
                "score": 10,
                "status": "valid",
            },
            "political": {
                "evidence": [],
                "score": None,
                "status": "N/A",
            },
            "historic": {
                "evidence": [
                    "represents the highest architectural standard of the Ming and Qing dynasties"
                ],
                "score": 10,
                "status": "valid",
            },
            "aesthetical": {
                "evidence": ["Only shock, reverence, solemnity, and amazement"],
                "score": 10,
                "status": "valid",
            },
            "scientific": {
                "evidence": [
                    "represents the highest architectural standard of the Ming and Qing dynasties"
                ],
                "score": 10,
                "status": "valid",
            },
            "age": {
                "evidence": [],
                "score": None,
                "status": "N/A",
            },
            "ecological": {
                "evidence": [],
                "score": None,
                "status": "N/A",
            },
        },
        label="text_seed_1",
    ),
    MemoryExample(
        input_payload=(
            "Its unique architectural style, pavilions, terraces, lattice windows, "
            "plants, bonsai, and famous calligraphies and paintings, all come "
            "together in one masterpiece for the world to see. From garden to The "
            "Treatise on Superfluous Things, I'm deeply moved by the lofty "
            "achievements of the ancients. Living in this peaceful era allows us "
            "to experience and feel this superb art of garden design firsthand. "
            "Proud to be born in China!"
        ),
        verified_output={
            "social": {
                "evidence": ["Proud to be born in China!"],
                "score": 10,
                "status": "valid",
            },
            "economic": {
                "evidence": [],
                "score": None,
                "status": "N/A",
            },
            "political": {
                "evidence": [
                    "Living in this peaceful era allows us to experience and feel this superb art of garden design firsthand."
                ],
                "score": 8,
                "status": "valid",
            },
            "historic": {
                "evidence": [
                    "From garden to The Treatise on Superfluous Things, I'm deeply moved by the lofty achievements of the ancients."
                ],
                "score": 10,
                "status": "valid",
            },
            "aesthetical": {
                "evidence": [
                    "Its unique architectural style, pavilions, terraces, lattice windows, plants, bonsai, and famous calligraphies and paintings, all come together in one masterpiece for the world to see."
                ],
                "score": 10,
                "status": "valid",
            },
            "scientific": {
                "evidence": [
                    "I'm deeply moved by the lofty achievements of the ancients",
                    "feel this superb art of garden design firsthand",
                ],
                "score": 10,
                "status": "valid",
            },
            "age": {
                "evidence": ["I'm deeply moved by the lofty achievements of the ancients"],
                "score": 10,
                "status": "valid",
            },
            "ecological": {
                "evidence": [
                    "plants, bonsai, and famous calligraphies and paintings, all come together in one masterpiece for the world to see"
                ],
                "score": 8,
                "status": "valid",
            },
        },
        label="text_seed_2",
    ),
    MemoryExample(
        input_payload=(
            "There were far too many people, and the service couldn't keep up. "
            "They really should control the number of visitors. The experience "
            "was terrible; I couldn't feel any sense of beauty. I came with high "
            "expectations but left completely disappointed."
        ),
        verified_output={
            "social": {
                "evidence": [
                    "I came with high expectations but left completely disappointed."
                ],
                "score": 1,
                "status": "valid",
            },
            "economic": {
                "evidence": [
                    "There were far too many people, and the service couldn't keep up.",
                    "They really should control the number of visitors.",
                ],
                "score": 1,
                "status": "valid",
            },
            "political": {
                "evidence": [],
                "score": None,
                "status": "N/A",
            },
            "historic": {
                "evidence": [],
                "score": None,
                "status": "N/A",
            },
            "aesthetical": {
                "evidence": ["I couldn't feel any sense of beauty"],
                "score": 1,
                "status": "valid",
            },
            "scientific": {
                "evidence": [],
                "score": None,
                "status": "N/A",
            },
            "age": {
                "evidence": [],
                "score": None,
                "status": "N/A",
            },
            "ecological": {
                "evidence": [],
                "score": None,
                "status": "N/A",
            },
        },
        label="text_seed_3",
    ),
)


def format_text_examples(examples: list[MemoryExample] | tuple[MemoryExample, ...]) -> str:
    """Render text examples as compact prompt material."""

    if not examples:
        return "None."
    blocks = []
    for index, example in enumerate(examples, start=1):
        label = f" ({example.label})" if example.label else ""
        blocks.append(
            "\n".join(
                [
                    f"Example {index}{label}",
                    "Review:",
                    str(example.input_payload),
                    "Verified JSON:",
                    pretty_json(example.verified_output),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_preliminary_extraction_prompt(review: str, memory_examples: list[MemoryExample]) -> str:
    """Build the preliminary extraction prompt for one review."""

    return f"""
Role definition: You are a cultural value analysis expert.

Input data:
Target review:
{review}

Rolling contextual memory:
{format_text_examples(memory_examples)}

Definitions of 8 cultural values:
{dimension_definition_block()}

Execution task:
For each dimension, reason internally and extract only original sentence(s) or
phrases from the target review that primarily support that dimension. Evidence
must be copied verbatim from the target review. Do not invent text. Do not score
the dimensions in this step.

Output specification:
Return one JSON object with exactly these top-level keys:
{", ".join(DIMENSION_KEYS)}

Each value must follow this schema:
{pretty_json(TEXT_OUTPUT_SCHEMA)}

Use an empty evidence list when the review does not support a dimension.
Return JSON only.
""".strip()


def build_verification_scoring_prompt(
    review: str,
    preliminary_json: dict[str, Any],
    memory_examples: list[MemoryExample],
) -> str:
    """Build the verification and scoring prompt for one review."""

    return f"""
Role definition: You are a cultural evidence auditor and evaluator.

Input data:
Target review:
{review}

Rolling contextual memory:
{format_text_examples(memory_examples)}

Preliminary extraction JSON:
{pretty_json(preliminary_json)}

Definitions of 8 cultural values:
{dimension_definition_block()}

Execution tasks:
1. Hallucination check: verify that every extracted evidence item appears
   verbatim in the target review. Remove evidence that does not appear exactly.
2. Relevance validation: remove evidence that does not strictly fit the
   dimension definition.
3. Perception scoring: assign a dimension-level perception score from 1 to 10
   based only on verified evidence, where 1 is very negative, 5 is neutral, and
   10 is very positive. If no valid evidence remains, set score to null and
   status to "N/A".

Output specification:
Return one JSON object with exactly these top-level keys:
{", ".join(DIMENSION_KEYS)}

Each value must follow this schema:
{pretty_json(VERIFIED_OUTPUT_SCHEMA)}

Return JSON only.
""".strip()


def build_image_system_instruction() -> str:
    """Return the image scoring system instruction."""

    return """
You are a prospective tourist preparing to visit the heritage attraction shown
in each target image.

Assess the visual quality of the attraction on a scale from 1 to 10 based on
first impressions. Generate a concise reason that balances visual appeal and
anticipated tourist experience.

Reference scoring:
1-3: Unattractive, messy, poorly preserved, or lacking cultural distinctiveness.
4-6: Standard heritage attraction, acceptable but not visually striking.
7-8: Visually appealing heritage attraction with clear cultural or experiential
     interest.
9-10: Highly distinctive, visually impressive, and strongly motivating for a
      tourist visit.

Output specification:
Return JSON only using this schema:
{
  "visual_analysis": "Concise description of what is visible and relevant.",
  "score": 1,
  "reason": "Concise scoring reason."
}
""".strip()
