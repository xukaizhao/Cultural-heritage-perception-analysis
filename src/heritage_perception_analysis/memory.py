"""Rolling contextual memory for few-shot calibration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable


@dataclass(frozen=True)
class MemoryExample:
    """One successful input-output pair retained for contextual memory."""

    input_payload: Any
    verified_output: dict[str, Any]
    label: str | None = None


class RollingContextMemory:
    """Keep the most recent successful examples in insertion order."""

    def __init__(self, max_examples: int = 3, seed_examples: Iterable[MemoryExample] | None = None):
        if max_examples < 1:
            raise ValueError("max_examples must be at least 1.")
        self.max_examples = max_examples
        self._examples: Deque[MemoryExample] = deque(maxlen=max_examples)
        if seed_examples:
            for example in seed_examples:
                self.add(example.input_payload, example.verified_output, example.label)

    def add(self, input_payload: Any, verified_output: dict[str, Any], label: str | None = None) -> None:
        """Append a successful input-output pair."""

        self._examples.append(
            MemoryExample(
                input_payload=input_payload,
                verified_output=verified_output,
                label=label,
            )
        )

    def examples(self) -> list[MemoryExample]:
        """Return examples from oldest to newest."""

        return list(self._examples)

    def __len__(self) -> int:
        return len(self._examples)
