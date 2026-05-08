"""Extract the predicted A/B/C/D letter from a model's raw output text.

Tries a sequence of regex patterns ordered from most specific to most general.
For each pattern, the LAST match in the text is taken — when a chain-of-thought
response discusses several options before committing to one, the last mention
of an option letter is the actual answer rather than an intermediate step.
With a single match the rule is a no-op (last == first), so robustness comes
from pattern coverage rather than the last-match rule itself.

Returns ``None`` if no pattern matches; the caller treats ``None`` as a parse
failure (counted as wrong in accuracy, separately reported as
``parse_success_rate`` for visibility).

Patterns (in order):
1. ``(?:final\\s+)?[Aa]nswer(?:\\s+is)?:\\s*\\(?([A-D])\\)?`` — "Answer: B",
   "Final Answer: B", "Answer is B".
2. ``correct\\s+(?:option|choice|answer)\\s+is\\s*\\(?([A-D])\\)?``.
3. ``\\\\boxed\\{([A-D])\\}`` — LaTeX-style boxed answer, common in math-tuned models.
4. ``\\b([A-D])\\s*$`` — bare letter at end of text, last-resort fallback.
"""
from __future__ import annotations

import re
from typing import Optional


PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:final\s+)?[Aa]nswer(?:\s+is)?:\s*\(?([A-D])\)?"),
    re.compile(r"correct\s+(?:option|choice|answer)\s+is\s*\(?([A-D])\)?"),
    re.compile(r"\\boxed\{([A-D])\}"),
    re.compile(r"\b([A-D])\s*$"),
)


def parse_letter(text: str) -> Optional[str]:
    """Return the predicted letter ('A'..'D'), or ``None`` if no pattern matches.

    Tries the patterns in declaration order. For each, ``re.findall`` collects
    all matches and the LAST one wins (see module docstring).
    """
    if not text:
        return None
    for pattern in PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    return None
