"""Context spell-checker (Phase 2, level 2): dictionary finds suspects, a model picks the fix.

    from models.spell.neural import NeuralSpellChecker
    sc = NeuralSpellChecker("models/spell/checkpoints/spell.pt", {"kn": kn_freq, "tcy": tcy_freq})
    sc.correct("ನಮಸ್ಕರ, ನಾನು ಬೆಂಗಲೂರು", lang="kn")
    sc.check(text, lang="kn")          # same shape as SpellChecker.check()

How it decides (guide, Phase 2 step 4):
  1. Suspects: words not in the dictionary (+ words the model would change: "real-word" errors,
     e.g. ಲ->ಳ that happens to make another real word).
  2. Candidates: SymSpell suggestions (Kannada-aware distance) + the model's own word, known words only.
  3. The model (char seq2seq trained on wrong->right windows) scores each candidate *sentence*:
     delta = log P(sentence with fix | input) - log P(sentence unchanged | input).
  4. Fix an unknown word if delta > margin_unknown. Change a word the dictionary knows only if
     delta > margin_known (much higher) and it is a 1-letter-ish edit. Precision first.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

from common.text import kannada_ratio, normalize, tokenize
from models.spell.checker import SpellChecker
from models.spell.distance import akshara_distance
from models.xlit.model import PAD, Vocab, XlitTransformer, greedy_decode, score_targets

MAX_CHARS = 96  # model window (training windows are <= 96 characters)
DEFAULT_MARGINS = {"unknown": 0.0, "known": 4.0}


def _norm(text: str) -> str:
    # keep zero-width marks so character offsets match the caller's text (the API underlines in place)
    return normalize(text, strip_zero_width=False)


@dataclass
class Suspect:
    start: int
    end: int
    word: str
    known: bool
    cands: list[tuple[str, float]] = field(default_factory=list)  # (candidate, delta), best first


def _words(text: str):
    return [
        t for t in tokenize(text) if t.is_word and len(t.text) >= 2 and kannada_ratio(t.text) >= 0.5
    ]


def _chunks(text: str, max_chars: int = MAX_CHARS) -> list[tuple[int, int]]:
    """(start, end) spans of whole words, each at most max_chars long."""
    spans, start, last_end = [], None, None
    for tok in tokenize(text):
        if not tok.is_word:
            continue
        if start is None:
            start = tok.start
        elif tok.end - start > max_chars:
            spans.append((start, last_end))
            start = tok.start
        last_end = tok.end
    if start is not None:
        spans.append((start, last_end))
    return spans


class NeuralSpellChecker:
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        wordfreqs: Mapping[str, Mapping[str, int]] | None = None,
        *,
        model: XlitTransformer | None = None,
        vocab: Vocab | None = None,
        checkers: Mapping[str, SpellChecker] | None = None,
        margins: Mapping[str, Mapping[str, float]] | None = None,
        device: str = "cpu",
        k: int = 5,
        batch: int = 64,
    ):
        if checkpoint is not None:
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            vocab = Vocab(ck["vocab"])
            model = XlitTransformer(**ck["config"])
            model.load_state_dict(ck["state_dict"])
            margins = margins or ck.get("margins")
        self.model = model.to(device).eval()
        self.v = vocab
        self.device, self.k, self.batch = device, k, batch
        self.checkers = dict(
            checkers or {lang: SpellChecker(wf) for lang, wf in (wordfreqs or {}).items()}
        )
        self.margins = {lang: dict(DEFAULT_MARGINS) for lang in ("kn", "tcy")}
        for lang, m in (margins or {}).items():
            self.margins[lang].update(m)

    # ---- model helpers -------------------------------------------------------------------
    def _src(self, texts: list[str], lang: str) -> torch.Tensor:
        enc = [self.v.encode(t, lang=lang) for t in texts]
        src = torch.full((len(enc), max(map(len, enc))), PAD, dtype=torch.long)
        for i, ids in enumerate(enc):
            src[i, : len(ids)] = torch.tensor(ids)
        return src.to(self.device)

    def _greedy(self, src: torch.Tensor) -> list[str]:
        return [
            self.v.decode(ids)
            for ids in greedy_decode(self.model, src, max_len=MAX_CHARS + 4).tolist()
        ]

    def _score(self, src: torch.Tensor, rows: list[int], texts: list[str]) -> list[float]:
        out = []
        for i in range(0, len(texts), self.batch * 4):
            r, t = rows[i : i + self.batch * 4], texts[i : i + self.batch * 4]
            out += score_targets(self.model, src[r], [self.v.encode(x) for x in t], alpha=0.0)
        return out

    # ---- core ------------------------------------------------------------------------------
    @torch.no_grad()
    def propose(
        self, texts: list[str], lang: str, use_model_words: bool = True
    ) -> list[list[Suspect]]:
        """For each text (<= MAX_CHARS), the suspect words with candidates scored by the model."""
        sc = self.checkers[lang]
        result: list[list[Suspect]] = []
        for b in range(0, len(texts), self.batch):
            chunk = [_norm(t) for t in texts[b : b + self.batch]]
            src = self._src(chunk, lang)
            greedy = self._greedy(src) if use_model_words else [""] * len(chunk)
            rows, variants, owners = (
                [],
                [],
                [],
            )  # owners: (text idx, suspect idx, cand) or (text idx,)
            per_text = []
            for ti, (text, g) in enumerate(zip(chunk, greedy, strict=True)):
                toks, gws = _words(text), [t.text for t in _words(g)]
                aligned = len(gws) == len(toks)
                sus = []
                for j, tok in enumerate(toks):
                    known = sc.is_known(tok.text)
                    cands = [] if known else [c for c, _ in sc.suggest(tok.text, k=self.k)]
                    mw = gws[j] if aligned else None
                    if mw and mw != tok.text and sc.is_known(mw) and mw not in cands:
                        if not known or akshara_distance(tok.text, mw) <= 1.0:
                            cands.append(mw)
                    if not known or cands:
                        sus.append(
                            Suspect(tok.start, tok.end, tok.text, known, [(c, 0.0) for c in cands])
                        )
                per_text.append(sus)
                if any(s.cands for s in sus):
                    rows.append(ti), variants.append(text), owners.append((ti,))
                    for si, s in enumerate(sus):
                        for c, _ in s.cands:
                            rows.append(ti)
                            variants.append(text[: s.start] + c + text[s.end :])
                            owners.append((ti, si, c))
            scores = self._score(src, rows, variants) if variants else []
            base = {o[0]: sc_ for o, sc_ in zip(owners, scores, strict=True) if len(o) == 1}
            deltas: dict[tuple[int, int], list[tuple[str, float]]] = {}
            for o, sc_ in zip(owners, scores, strict=True):
                if len(o) == 3:
                    deltas.setdefault((o[0], o[1]), []).append((o[2], sc_ - base[o[0]]))
            for ti, sus in enumerate(per_text):
                for si, s in enumerate(sus):
                    s.cands = sorted(deltas.get((ti, si), []), key=lambda x: -x[1])
            result += per_text
        return result

    def decide(self, text: str, suspects: list[Suspect], lang: str, margins=None) -> str:
        m = margins or self.margins[lang]
        out = _norm(text)
        for s in sorted(suspects, key=lambda s: -s.start):
            if not s.cands:
                continue
            best, delta = s.cands[0]
            ok = delta > m["unknown"] if not s.known else (
                delta > m["known"] and akshara_distance(s.word, best) <= 1.0)  # fmt: skip
            if ok:
                out = out[: s.start] + best + out[s.end :]
        return out

    # ---- public API (any length of text) ---------------------------------------------------
    def _spans(self, text: str, lang: str) -> list[tuple[int, list[Suspect]]]:
        text = _norm(text)
        spans = [(a, b) for a, b in _chunks(text) if b - a <= MAX_CHARS]
        props = self.propose([text[a:b] for a, b in spans], lang) if spans else []
        return [(a, p) for (a, _), p in zip(spans, props, strict=True)]

    def correct(self, text: str, lang: str = "kn") -> str:
        text = _norm(text)
        out, pos = [], 0
        for a, sus in self._spans(text, lang):
            end = max((s.end for s in sus), default=0)
            seg = text[a : a + end] if sus else ""
            if sus:
                out += [text[pos:a], self.decide(seg, sus, lang)]
                pos = a + end
        out.append(text[pos:])
        return "".join(out)

    def check(self, text: str, lang: str = "kn", k: int = 5) -> list[dict]:
        """Unknown words (always) + known words the model is confident are wrong."""
        m = self.margins[lang]
        errors = []
        for a, sus in self._spans(text, lang):
            for s in sus:
                good = [c for c, d in s.cands if s.known is False or d > m["known"]]
                if s.known and not good:
                    continue
                errors.append({"word": s.word, "start": a + s.start, "end": a + s.end,
                               "suggestions": good[:k], "real_word": s.known})  # fmt: skip
        return sorted(errors, key=lambda e: e["start"])
