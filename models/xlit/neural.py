"""Run the trained transliteration model (CPU is fine: the model is small).

    from models.xlit.neural import NeuralTransliterator
    xl = NeuralTransliterator("models/xlit/checkpoints/xlit.pt", wordfreqs={"kn": {...}, "tcy": {...}})
    xl.transliterate("encha ullar?", lang="tcy")

Same output shape as RuleTransliterator.transliterate(), so the API can swap them.

Ranking: candidates = the model's beam + the rule engine's guesses that are real words (the beam
sometimes misses short common words like "ooru" -> ಊರು). Every candidate gets the model's
log-prob plus a word-list bonus that grows with how common the word is:
    score = model_logprob + bonus * log10(1 + count)
so a frequent word (ಏನು, 1,119×) beats a rare typo the corpus happens to contain (ಎನು, 9×).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

import torch

from models.xlit.model import PAD, Vocab, XlitTransformer, beam_search, score_targets
from models.xlit.rules import RuleTransliterator

MAX_WORD = 30
CHUNK = 128  # words per beam-search batch (keeps CPU memory small)
RULE_EXTRA = 8  # rule candidates considered per word (only those in the word list are kept)
_WORD_RE = re.compile(r"[A-Za-z]+")

Scored = list[tuple[str, float, int]]  # (candidate, model log-prob, word-list count)


class NeuralTransliterator:
    def __init__(
        self,
        checkpoint: str | Path,
        wordfreqs: Mapping[str, Mapping[str, int]] | None = None,
        beam: int = 5,
        bonus: float = 0.5,
        use_rules: bool = True,
    ):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.sv, self.tv = Vocab(ck["src_vocab"]), Vocab(ck["tgt_vocab"])
        self.model = XlitTransformer(**ck["config"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.info = {k: ck.get(k) for k in ("stage", "epoch", "val_top1")}
        self.wordfreqs = {k: dict(v) for k, v in (wordfreqs or {}).items()}
        self.beam, self.bonus, self.use_rules = beam, bonus, use_rules
        self._cache: dict[tuple[str, str], Scored] = {}
        self._rules = {lang: RuleTransliterator(wf) for lang, wf in self.wordfreqs.items()}

    def _src(self, words: list[str], lang: str) -> torch.Tensor:
        enc = [self.sv.encode(w, lang=lang) for w in words]
        src = torch.full((len(enc), max(map(len, enc))), PAD, dtype=torch.long)
        for i, ids in enumerate(enc):
            src[i, : len(ids)] = torch.tensor(ids)
        return src

    def _batch(self, words: list[str], lang: str) -> list[Scored]:
        src = self._src(words, lang)
        wf = self.wordfreqs.get(lang, {})
        found: list[dict[str, float]] = []
        for hyps in beam_search(self.model, src, beam=self.beam):
            cands: dict[str, float] = {}
            for ids, score in hyps:
                w = self.tv.decode(ids)
                if w and w not in cands:
                    cands[w] = score
            found.append(cands)
        if self.use_rules and lang in self._rules:  # add known words the beam missed
            extra = []
            for i, w in enumerate(words):
                for c in self._rules[lang].word_candidates(w, k=RULE_EXTRA):
                    if c in wf and c not in found[i] and all(ch in self.tv.stoi for ch in c):
                        extra.append((i, c))
            for j in range(0, len(extra), CHUNK * 2):
                part = extra[j : j + CHUNK * 2]
                rows = src[[i for i, _ in part]]
                tgts = [self.tv.encode(c) for _, c in part]
                for (i, c), s in zip(part, score_targets(self.model, rows, tgts), strict=True):
                    found[i][c] = s
        return [[(c, s, wf.get(c, 0)) for c, s in cands.items()] for cands in found]

    def scored(self, words: list[str], lang: str) -> list[Scored]:
        """Unranked candidates with their parts of the score (cached)."""
        words = [w.lower() for w in words]
        todo = sorted({w for w in words if (w, lang) not in self._cache and len(w) <= MAX_WORD})
        if len(self._cache) + len(todo) > 50_000:
            self._cache.clear()
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i : i + CHUNK]
            for w, c in zip(chunk, self._batch(chunk, lang), strict=True):
                self._cache[(w, lang)] = c
        return [self._cache.get((w, lang), []) for w in words]

    def rank(self, cands: Scored, bonus: float | None = None) -> list[str]:
        b = self.bonus if bonus is None else bonus
        return [c for c, *_ in sorted(cands, key=lambda x: -(x[1] + b * math.log10(1 + x[2])))]

    def candidates(self, words: list[str], lang: str, k: int = 5) -> list[list[str]]:
        res = []
        for w, cands in zip(words, self.scored(words, lang), strict=True):
            if cands:
                res.append(self.rank(cands)[:k])
            elif len(w) > MAX_WORD:  # very long "word": fall back to rules
                rules = self._rules.get(lang) or RuleTransliterator()
                res.append(rules.word_candidates(w.lower(), k=k))
            else:
                res.append([w])
        return res

    def transliterate(self, text: str, lang: str = "kn", k: int = 3) -> dict:
        matches = list(_WORD_RE.finditer(text))
        cands = self.candidates([m.group() for m in matches], lang, k=k) if matches else []
        parts, pos = [], 0
        for m, c in zip(matches, cands, strict=True):
            parts += [text[pos : m.start()], c[0]]
            pos = m.end()
        parts.append(text[pos:])
        return {"output": "".join(parts), "alternatives": cands}
