"""Evidence spans by sentence occlusion.

For each sentence/line, score the text with that piece removed. A large drop in the margin z means the piece
drives the score. This is cheaper and simpler than Integrated Gradients; it is NOT the handbook's method, and the
model card says so. Spans are character offsets into the ORIGINAL text; `text` is cut to 300 characters.
"""
from __future__ import annotations

import re
from typing import Protocol, Sequence

import numpy as np

MAX_EVIDENCE_CHARS = 300
_PIECE = re.compile(r"[^\n.!?]+[.!?]*")


class Scorer(Protocol):
    def score(self, texts: Sequence[str]) -> np.ndarray: ...


def split_pieces(text: str, max_parts: int = 20) -> list[tuple[int, int]]:
    """(start, end) offsets of non-empty sentence/line pieces, at most `max_parts` of them."""
    out = []
    for m in _PIECE.finditer(text):
        if m.group().strip():
            out.append((m.start(), m.end()))
        if len(out) >= max_parts:
            break
    return out


def top_spans(text: str, scorer: Scorer, z_full: float, k: int = 3, max_parts: int = 20) -> list[dict]:
    parts = split_pieces(text, max_parts)
    if len(parts) < 2:
        return [{"kind": "span", "span": [0, len(text)], "text": text[:MAX_EVIDENCE_CHARS], "drop_effect_z": None}]
    without = [text[:s] + text[e:] for s, e in parts]
    z_without = np.asarray(scorer.score(without), dtype=float)
    effects = z_full - z_without                       # positive = this piece pushes the score up
    order = np.argsort(-effects)[:k]
    return [
        {
            "kind": "span",
            "span": [parts[i][0], parts[i][1]],
            "text": text[parts[i][0]:parts[i][1]].strip()[:MAX_EVIDENCE_CHARS],
            "drop_effect_z": round(float(effects[i]), 3),
        }
        for i in order
        if effects[i] > 0
    ]
