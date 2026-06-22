"""Evidence validation helpers."""

from __future__ import annotations


def enforce_verbatim_evidence(review: str, evidence: list[str]) -> tuple[list[str], list[str]]:
    """Keep evidence items that appear verbatim in the review."""

    kept: list[str] = []
    removed: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        cleaned = item.strip().strip('"').strip("'").strip()
        if not cleaned:
            continue
        if cleaned in review:
            if cleaned not in seen:
                kept.append(cleaned)
                seen.add(cleaned)
        else:
            removed.append(cleaned)
    return kept, removed
