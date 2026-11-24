"""Rule-based transliteration between casual English letters and Kannada script.

Two directions:
  * roman_to_kannada(): "naanu" -> ["ನಾನು", ...]. Generates many possible spellings
    (because "t" can be ತ or ಟ, "e" can be ೆ or ೇ, ...) with a small beam search, then
    re-ranks them with a word-frequency list. This is the Phase 1 BASELINE that the
    neural model has to beat.
  * kannada_to_roman(): "ಉಲ್ಲರ್" -> ["ullar", "ulla r"...]. Used for the "reverse trick"
    (Dataset 1, Tulu method A): turn Tulu words into realistic English-letter spellings.
"""

from __future__ import annotations

import heapq
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from common.text import CONSONANTS, VIRAMA, normalize

# ---------------------------------------------------------------------------
# Kannada -> roman tables (first option = most common casual spelling)
# ---------------------------------------------------------------------------
CONS_ROMAN: dict[str, list[str]] = {
    "ಕ": ["k"], "ಖ": ["kh", "k"], "ಗ": ["g"], "ಘ": ["gh", "g"], "ಙ": ["ng"],
    "ಚ": ["ch", "c"], "ಛ": ["chh", "ch"], "ಜ": ["j"], "ಝ": ["jh", "j"], "ಞ": ["nj", "n"],
    "ಟ": ["t"], "ಠ": ["th", "t"], "ಡ": ["d"], "ಢ": ["dh", "d"], "ಣ": ["n"],
    "ತ": ["t", "th"], "ಥ": ["th"], "ದ": ["d", "dh"], "ಧ": ["dh", "d"], "ನ": ["n"],
    "ಪ": ["p"], "ಫ": ["ph", "f"], "ಬ": ["b"], "ಭ": ["bh", "b"], "ಮ": ["m"],
    "ಯ": ["y"], "ರ": ["r"], "ಱ": ["r"], "ಲ": ["l"], "ವ": ["v", "w"],
    "ಶ": ["sh", "s"], "ಷ": ["sh", "s"], "ಸ": ["s"], "ಹ": ["h"], "ಳ": ["l"],
    "ೞ": ["zh", "l"], "ೝ": ["n"],
}  # fmt: skip

VOWEL_ROMAN: dict[str, list[str]] = {
    # independent vowels
    "ಅ": ["a"], "ಆ": ["aa", "a"], "ಇ": ["i"], "ಈ": ["ee", "i", "ii"], "ಉ": ["u"],
    "ಊ": ["oo", "u", "uu"], "ಋ": ["ru", "ri"], "ೠ": ["ru"], "ಎ": ["e"], "ಏ": ["e", "ae", "ee"],
    "ಐ": ["ai"], "ಒ": ["o"], "ಓ": ["o", "oo"], "ಔ": ["au", "ou"],
    # vowel signs
    "ಾ": ["aa", "a"], "ಿ": ["i"], "ೀ": ["ee", "i", "ii"], "ು": ["u"], "ೂ": ["oo", "u", "uu"],
    "ೃ": ["ru", "ri"], "ೆ": ["e"], "ೇ": ["e", "ae", "ee"], "ೈ": ["ai"], "ೊ": ["o"],
    "ೋ": ["o", "oo"], "ೌ": ["au", "ou"],
}  # fmt: skip

LABIALS = set("ಪಫಬಭಮ")
VELARS = set("ಕಖಗಘ")


def _roman_options(word: str) -> list[list[str]]:
    """For each output position, the list of possible roman spellings."""
    word = normalize(word)
    opts: list[list[str]] = []
    i = 0
    while i < len(word):
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch in CONSONANTS:
            opts.append(CONS_ROMAN.get(ch, [ch]))
            if nxt == "಼":  # nukta: ಜ಼ -> z, ಫ಼ -> f
                opts[-1] = {"ಜ": ["z"], "ಫ": ["f"]}.get(ch, opts[-1])
                i += 1
                nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt == VIRAMA:
                i += 1  # dead consonant: no vowel
            elif nxt in VOWEL_ROMAN:
                opts.append(VOWEL_ROMAN[nxt])
                i += 1
            else:
                opts.append(["a"])  # inherent vowel
        elif ch in VOWEL_ROMAN:
            opts.append(VOWEL_ROMAN[ch])
        elif ch == "ಂ":
            if nxt in LABIALS:
                opts.append(["m"])
            elif nxt in VELARS:
                opts.append(["n", "ng"])
            else:
                opts.append(["n", "m"] if not nxt else ["n"])
        elif ch == "ಃ":
            opts.append(["h", ""])
        elif ch in (VIRAMA, "಼"):
            pass
        else:
            opts.append([ch])
        i += 1
    return opts


def kannada_to_roman(word: str, max_variants: int = 6) -> list[str]:
    """Return realistic English-letter spellings of a Kannada-script word.

    The first item is the most common casual spelling. The rest change one
    ambiguous letter at a time (aa->a, sh->s, ...), plus a "lazy" spelling that
    uses the shortest option everywhere, which is how many people type on phones.
    """
    opts = _roman_options(word)
    canonical = "".join(o[0] for o in opts)
    out = [canonical]
    lazy = "".join(min(o, key=len) for o in opts)
    if lazy not in out:
        out.append(lazy)
    for idx, o in enumerate(opts):
        for alt in o[1:]:
            v = "".join(alt if j == idx else oo[0] for j, oo in enumerate(opts))
            if v not in out:
                out.append(v)
            if len(out) >= max_variants:
                return out
    return out


# ---------------------------------------------------------------------------
# roman -> Kannada candidate generator
# ---------------------------------------------------------------------------
# (independent vowel, vowel sign, cost). Sign "" = inherent 'a'.
VOWEL_UNITS: dict[str, list[tuple[str, str, float]]] = {
    "a": [("ಅ", "", 0.0), ("ಆ", "ಾ", 1.0)],
    "aa": [("ಆ", "ಾ", 0.0)],
    "i": [("ಇ", "ಿ", 0.0), ("ಈ", "ೀ", 0.8)],
    "ii": [("ಈ", "ೀ", 0.0)],
    "ee": [("ಈ", "ೀ", 0.0), ("ಏ", "ೇ", 0.9)],
    "u": [("ಉ", "ು", 0.0), ("ಊ", "ೂ", 0.8)],
    "uu": [("ಊ", "ೂ", 0.0)],
    "oo": [("ಊ", "ೂ", 0.2), ("ಓ", "ೋ", 0.5)],
    "e": [("ಎ", "ೆ", 0.0), ("ಏ", "ೇ", 0.6)],
    "ae": [("ಏ", "ೇ", 0.0)],
    "ai": [("ಐ", "ೈ", 0.0)],
    "ei": [("ಐ", "ೈ", 0.3)],
    "o": [("ಒ", "ೊ", 0.0), ("ಓ", "ೋ", 0.6)],
    "au": [("ಔ", "ೌ", 0.0)],
    "ou": [("ಔ", "ೌ", 0.2)],
    "ru": [("ಋ", "ೃ", 1.2)],
    "ri": [("ಋ", "ೃ", 1.4)],
}

# roman -> [(Kannada consonant or cluster, cost)]
CONS_UNITS: dict[str, list[tuple[str, float]]] = {
    "k": [("ಕ", 0.0)], "kh": [("ಖ", 0.0)], "q": [("ಕ", 0.2)],
    "g": [("ಗ", 0.0)], "gh": [("ಘ", 0.0)],
    "c": [("ಕ", 0.5), ("ಚ", 0.5)], "ch": [("ಚ", 0.0), ("ಛ", 1.0)], "chh": [("ಛ", 0.0)],
    "j": [("ಜ", 0.0)], "jh": [("ಝ", 0.0)], "z": [("ಜ", 0.5), ("ಝ", 0.8)],
    "t": [("ತ", 0.0), ("ಟ", 0.4)], "th": [("ತ", 0.2), ("ಥ", 0.6), ("ಟ", 0.8), ("ಠ", 1.0)],
    "d": [("ದ", 0.0), ("ಡ", 0.4)], "dh": [("ಧ", 0.2), ("ದ", 0.6), ("ಢ", 1.0), ("ಡ", 1.0)],
    "n": [("ನ", 0.0), ("ಣ", 0.5)],
    "p": [("ಪ", 0.0)], "ph": [("ಫ", 0.0)], "f": [("ಫ", 0.0)],
    "b": [("ಬ", 0.0)], "bh": [("ಭ", 0.0)], "m": [("ಮ", 0.0)],
    "y": [("ಯ", 0.0)], "r": [("ರ", 0.0)], "l": [("ಲ", 0.0), ("ಳ", 0.4)],
    "v": [("ವ", 0.0)], "w": [("ವ", 0.0)],
    "sh": [("ಶ", 0.0), ("ಷ", 0.4)], "s": [("ಸ", 0.0), ("ಶ", 0.8), ("ಷ", 1.0)],
    "h": [("ಹ", 0.0)],
    "ksh": [("ಕ್ಷ", 0.0)], "x": [("ಕ್ಷ", 0.3), ("ಕ್ಸ", 0.5)],
    "jn": [("ಜ್ಞ", 0.3)], "gn": [("ಜ್ಞ", 0.6)],
}  # fmt: skip

_UNITS_BY_LEN = sorted({*VOWEL_UNITS, *CONS_UNITS}, key=len, reverse=True)
_MAX_UNIT = max(len(u) for u in _UNITS_BY_LEN)
_ROMAN_WORD_RE = re.compile(r"[a-z]+")
INDEPENDENT_VOWEL_MIDWORD_COST = 1.5
FINAL_VIRAMA_COST = 0.3
FINAL_INHERENT_A_COST = 0.5
ANUSVARA_COST = 0.2


@dataclass(order=True)
class _State:
    cost: float
    out: str
    pending: bool  # last output is a consonant still waiting for its vowel


def _expand(roman: str, beam: int) -> list[tuple[str, float]]:
    """Beam search over every way to split `roman` into units."""
    states: dict[int, list[_State]] = {0: [_State(0.0, "", False)]}
    n = len(roman)
    for pos in range(n):
        here = heapq.nsmallest(beam, states.pop(pos, []))
        for st in here:
            for size in range(1, _MAX_UNIT + 1):
                unit = roman[pos : pos + size]
                if len(unit) < size:
                    break
                nxt_pos = pos + size
                bucket = states.setdefault(nxt_pos, [])
                for indep, sign, c in VOWEL_UNITS.get(unit, []):
                    if st.pending:
                        bucket.append(_State(st.cost + c, st.out + sign, False))
                    else:
                        extra = INDEPENDENT_VOWEL_MIDWORD_COST if st.out else 0.0
                        bucket.append(_State(st.cost + c + extra, st.out + indep, False))
                for kn, c in CONS_UNITS.get(unit, []):
                    joiner = VIRAMA if st.pending else ""
                    bucket.append(_State(st.cost + c, st.out + joiner + kn, True))
                # n/m after a vowel and before a consonant (or m at word end) -> anusvara ಂ
                if unit in ("n", "m") and st.out and not st.pending:
                    following = roman[nxt_pos : nxt_pos + 1]
                    if (following and following not in "aeiou") or (unit == "m" and not following):
                        bucket.append(_State(st.cost + ANUSVARA_COST, st.out + "ಂ", False))
    finals = []
    for st in states.get(n, []):
        if st.pending:
            finals.append((st.out + VIRAMA, st.cost + FINAL_VIRAMA_COST))
            finals.append((st.out, st.cost + FINAL_INHERENT_A_COST))
        else:
            finals.append((st.out, st.cost))
    best: dict[str, float] = {}
    for out, c in finals:
        if out and c < best.get(out, math.inf):
            best[out] = c
    return sorted(best.items(), key=lambda x: x[1])


class RuleTransliterator:
    """roman -> Kannada script, re-ranked with a word-frequency list (the baseline)."""

    def __init__(self, wordfreq: Mapping[str, int] | None = None, beam: int = 64):
        self.wordfreq = dict(wordfreq or {})
        self.beam = beam
        self._max_log = math.log10(max(self.wordfreq.values(), default=1) + 1)

    def _score(self, word: str, cost: float) -> float:
        freq = self.wordfreq.get(word, 0)
        if not freq:
            return cost
        # Known words get a big bonus; frequent words a bit more.
        return cost - 3.0 - 2.0 * math.log10(freq + 1) / max(self._max_log, 1e-9)

    def word_candidates(self, roman_word: str, k: int = 5) -> list[str]:
        rw = roman_word.lower()
        if not _ROMAN_WORD_RE.fullmatch(rw):
            return [roman_word]
        cands = _expand(rw, self.beam)
        ranked = sorted(cands, key=lambda x: self._score(x[0], x[1]))
        return [w for w, _ in ranked[:k]]

    def transliterate(self, text: str, k: int = 3) -> dict:
        """Convert a whole sentence. Non-letters (spaces, digits, punctuation) pass through."""
        output_parts: list[str] = []
        alternatives: list[list[str]] = []
        pos = 0
        for m in re.finditer(r"[A-Za-z]+", text):
            output_parts.append(text[pos : m.start()])
            cands = self.word_candidates(m.group(), k=k)
            output_parts.append(cands[0])
            alternatives.append(cands)
            pos = m.end()
        output_parts.append(text[pos:])
        return {"output": "".join(output_parts), "alternatives": alternatives}
