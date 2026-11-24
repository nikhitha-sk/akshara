"""Run the trained transliteration model (CPU is fine: the model is small).

    from models.xlit.neural import NeuralTransliterator
    xl = NeuralTransliterator("models/xlit/checkpoints/xlit.pt", wordfreqs={"kn": {...}, "tcy": {...}})
    xl.transliterate("encha ullar?", lang="tcy")

Same output shape as RuleTransliterator.transliterate(), so the API can swap them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import torch

from models.xlit.model import PAD, Vocab, XlitTransformer, beam_search
from models.xlit.rules import RuleTransliterator

MAX_WORD = 30
CHUNK = 128  # words per beam-search batch (keeps CPU memory small)
_WORD_RE = re.compile(r"[A-Za-z]+")


class NeuralTransliterator:
    def __init__(
        self,
        checkpoint: str | Path,
        wordfreqs: Mapping[str, Mapping[str, int]] | None = None,
        beam: int = 5,
        bonus: float = 1.5,
    ):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.sv, self.tv = Vocab(ck["src_vocab"]), Vocab(ck["tgt_vocab"])
        self.model = XlitTransformer(**ck["config"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.info = {k: ck.get(k) for k in ("stage", "epoch", "val_top1")}
        self.wordfreqs = {k: dict(v) for k, v in (wordfreqs or {}).items()}
        self.beam, self.bonus = beam, bonus
        self._cache: dict[tuple[str, str], list[str]] = {}
        self._rules = {lang: RuleTransliterator(wf) for lang, wf in self.wordfreqs.items()}

    def _batch(self, words: list[str], lang: str) -> list[list[str]]:
        enc = [self.sv.encode(w, lang=lang) for w in words]
        src = torch.full((len(enc), max(map(len, enc))), PAD, dtype=torch.long)
        for i, ids in enumerate(enc):
            src[i, : len(ids)] = torch.tensor(ids)
        wf = self.wordfreqs.get(lang, {})
        out = []
        for hyps in beam_search(self.model, src, beam=self.beam):
            cands: dict[str, float] = {}
            for ids, score in hyps:
                w = self.tv.decode(ids)
                if w and w not in cands:
                    cands[w] = score + (self.bonus if w in wf else 0.0)
            out.append(sorted(cands, key=lambda c: -cands[c]))
        return out

    def candidates(self, words: list[str], lang: str, k: int = 5) -> list[list[str]]:
        words = [w.lower() for w in words]
        todo = sorted({w for w in words if (w, lang) not in self._cache and len(w) <= MAX_WORD})
        if len(self._cache) + len(todo) > 50_000:
            self._cache.clear()
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i : i + CHUNK]
            for w, c in zip(chunk, self._batch(chunk, lang), strict=True):
                self._cache[(w, lang)] = c or [w]
        res = []
        for w in words:
            if (w, lang) in self._cache:
                res.append(self._cache[(w, lang)][:k])
            else:  # very long "word": fall back to rules
                rules = self._rules.get(lang) or RuleTransliterator()
                res.append(rules.word_candidates(w, k=k))
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
