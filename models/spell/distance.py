"""Kannada-aware edit distance.

Plain edit distance treats every change the same. Real Kannada typos are not like
that: ಲ/ಳ mix-ups, short/long vowel signs and a missing ್ are far more common than
random letters. So:
  * we compare AKSHARAS (letter-blocks), not raw Unicode characters, and
  * swapping two "confusable" characters costs much less than 1.
"""

from __future__ import annotations

from functools import lru_cache

from common.text import split_aksharas

CONFUSABLE_GROUPS = [
    "ಿೀ", "ುೂ", "ೆೇ", "ೊೋ", "ಅಆ", "ಇಈ", "ಉಊ", "ಎಏ", "ಒಓ",
    "ಲಳ", "ನಣ", "ಶಷಸ", "ತಟ", "ದಡ", "ಕಖ", "ಗಘ", "ಚಛ", "ಜಝ", "ತಥ", "ದಧ", "ಟಠ", "ಡಢ",
    "ಪಫ", "ಬಭ", "ರಱ", "ಂನ", "ಂಮ",
]  # fmt: skip
CONFUSABLE: set[frozenset[str]] = {
    frozenset((a, b)) for g in CONFUSABLE_GROUPS for a in g for b in g if a != b
}
CHEAP_SUB = 0.3
# Characters that are often dropped or added by mistake (missing ಾ, missing ್, extra ಂ ...)
CHEAP_INDEL = {"ಾ": 0.4, "್": 0.4, "ಂ": 0.5, "ೕ": 0.3, "‌": 0.1, "‍": 0.1}


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return CHEAP_SUB if frozenset((a, b)) in CONFUSABLE else 1.0


def _indel_cost(c: str) -> float:
    return CHEAP_INDEL.get(c, 1.0)


@lru_cache(maxsize=200_000)
def char_distance(a: str, b: str) -> float:
    """Weighted Levenshtein over characters."""
    prev = [0.0]
    for ch in b:
        prev.append(prev[-1] + _indel_cost(ch))
    for ca in a:
        cur = [prev[0] + _indel_cost(ca)]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + _indel_cost(ca),
                    cur[j - 1] + _indel_cost(cb),
                    prev[j - 1] + _sub_cost(ca, cb),
                )
            )
        prev = cur
    return prev[-1]


def akshara_distance(a: str, b: str) -> float:
    """Weighted Levenshtein over aksharas. Replacing one akshara costs at most 1."""
    xa, xb = split_aksharas(a), split_aksharas(b)
    prev = [float(j) for j in range(len(xb) + 1)]
    for i, pa in enumerate(xa, 1):
        cur = [float(i)]
        for j, pb in enumerate(xb, 1):
            sub = min(1.0, char_distance(pa, pb))
            # dropping a whole akshara that is only a sign (rare) still costs 1
            cur.append(min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + sub))
        prev = cur
    return prev[-1]
