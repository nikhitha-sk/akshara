"""Level 1 spell-checker: dictionary lookup + Kannada-aware ranking (the Phase 2 baseline).

1. Detect: a Kannada word not in the word-frequency list is suspicious.
2. Candidates: SymSpell finds dictionary words within 2 character edits (very fast).
3. Rank: Kannada-aware akshara distance first, word frequency second.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from common.text import kannada_ratio, normalize, tokenize
from models.spell.distance import akshara_distance

try:
    from symspellpy import SymSpell, Verbosity
except ImportError:  # pragma: no cover
    SymSpell = None


def load_wordfreq(path: str | Path, min_count: int = 1) -> dict[str, int]:
    """Read a `word<TAB>count` file."""
    freq: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) >= min_count:
                freq[normalize(parts[0])] = int(parts[1])
    return freq


class SpellChecker:
    def __init__(
        self,
        wordfreq: Mapping[str, int],
        max_edit: int = 2,
        prefix_length: int = 8,
        index_top_n: int | None = 150_000,
    ):
        """`index_top_n`: only the most common N words are offered as suggestions
        (keeps memory small); every word in `wordfreq` still counts as "known"."""
        self.freq = dict(wordfreq)
        self.max_edit = max_edit
        self._sym = None
        if SymSpell is not None:
            self._sym = SymSpell(max_dictionary_edit_distance=max_edit, prefix_length=prefix_length)
            ranked = sorted(self.freq.items(), key=lambda x: -x[1])[:index_top_n]
            for w, c in ranked:
                self._sym.create_dictionary_entry(w, c)

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> SpellChecker:
        return cls(load_wordfreq(path), **kw)

    def is_known(self, word: str) -> bool:
        return normalize(word) in self.freq

    def _raw_candidates(self, word: str) -> list[str]:
        if self._sym is not None:
            return [
                s.term
                for s in self._sym.lookup(word, Verbosity.ALL, max_edit_distance=self.max_edit)
            ]
        # slow fallback without symspellpy: scan words of similar length
        return [w for w in self.freq if abs(len(w) - len(word)) <= self.max_edit]

    def suggest(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        """Return up to k (suggestion, distance) pairs, best first."""
        word = normalize(word)
        scored = []
        for cand in self._raw_candidates(word):
            if cand == word:
                continue
            d = akshara_distance(word, cand)
            # frequency as a tie-breaker: common words win small distance ties
            prior = math.log10(self.freq.get(cand, 1) + 1) / 10
            scored.append((cand, d, d - prior))
        scored.sort(key=lambda x: x[2])
        return [(c, round(d, 3)) for c, d, _ in scored[:k]]

    def check(self, text: str, k: int = 5) -> list[dict]:
        """Find unknown Kannada words in text, with suggestions and character offsets."""
        errors = []
        for tok in tokenize(text):
            if not tok.is_word or kannada_ratio(tok.text) < 0.5 or len(tok.text) < 2:
                continue
            if self.is_known(tok.text):
                continue
            errors.append(
                {
                    "word": tok.text,
                    "start": tok.start,
                    "end": tok.end,
                    "suggestions": [s for s, _ in self.suggest(tok.text, k=k)],
                }
            )
        return errors

    def correct(self, text: str, max_distance: float = 1.0) -> str:
        """Replace each unknown word with its best suggestion if it is close enough.

        Precision matters more than recall: we only fix words when the best
        suggestion is within `max_distance` (Kannada-aware) and clearly better
        than the runner-up.
        """
        out = []
        for tok in tokenize(text):
            if tok.is_word and kannada_ratio(tok.text) >= 0.5 and not self.is_known(tok.text):
                sugg = self.suggest(tok.text, k=2)
                if sugg and sugg[0][1] <= max_distance:
                    if len(sugg) == 1 or sugg[1][1] - sugg[0][1] >= 0.1 or sugg[0][1] <= 0.4:
                        out.append(sugg[0][0])
                        continue
            out.append(tok.text)
        return "".join(out)
