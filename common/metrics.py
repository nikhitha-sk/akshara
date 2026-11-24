"""Evaluation metrics used across all four tools (Section 11 of the guide)."""

from __future__ import annotations

from collections.abc import Sequence

from common.text import normalize, split_words


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences (strings or lists of tokens)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Character error rate over a corpus: total char edits / total reference chars."""
    edits = sum(edit_distance(normalize(r), normalize(h)) for r, h in zip(refs, hyps, strict=True))
    total = sum(len(normalize(r)) for r in refs)
    return edits / max(total, 1)


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Word error rate over a corpus: total word edits / total reference words."""
    edits = 0
    total = 0
    for r, h in zip(refs, hyps, strict=True):
        rw, hw = split_words(normalize(r)), split_words(normalize(h))
        edits += edit_distance(rw, hw)
        total += len(rw)
    return edits / max(total, 1)


def topk_accuracy(refs: Sequence[str], candidates: Sequence[Sequence[str]], k: int) -> float:
    """Share of items whose correct answer appears in the first k candidates."""
    hits = sum(normalize(r) in [normalize(c) for c in cands[:k]] for r, cands in zip(refs, candidates, strict=True))
    return hits / max(len(refs), 1)


def spell_prf(
    sources: Sequence[str], targets: Sequence[str], predictions: Sequence[str], beta: float = 0.5
) -> dict[str, float]:
    """Word-level precision / recall / F-beta for a spell-checker.

    Each item is (wrong sentence, correct sentence, model's output). Sentences must
    keep the same number of words (true for our synthetic and real test sets).
      * a "change" is any word where prediction != source
      * a correct change is where prediction == target and target != source
    """
    tp = fp = fn = 0
    for src, tgt, pred in zip(sources, targets, predictions, strict=True):
        s, t, p = (split_words(normalize(x)) for x in (src, tgt, pred))
        if not (len(s) == len(t) == len(p)):
            raise ValueError(f"word count mismatch: {src!r} / {tgt!r} / {pred!r}")
        for sw, tw, pw in zip(s, t, p, strict=True):
            needs_fix = sw != tw
            changed = pw != sw
            if changed and pw == tw:
                tp += 1
            elif changed:
                fp += 1
                if needs_fix:
                    fn += 1
            elif needs_fix:
                fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    b2 = beta * beta
    f = (1 + b2) * precision * recall / max(b2 * precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, f"f{beta}": f, "tp": tp, "fp": fp, "fn": fn}
