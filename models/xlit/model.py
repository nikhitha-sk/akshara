"""Character-level Transformer for transliteration (letters in -> letters out).

Shared by training (models/xlit/train.py) and the live API (models/xlit/neural.py).
Source: a language tag (<kn>/<tcy>) + English letters. Target: Kannada-script characters.
"""

from __future__ import annotations

import math

import torch
from torch import nn

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
LANG_TOKENS = ["<kn>", "<tcy>"]


class Vocab:
    def __init__(self, symbols: list[str]):
        self.itos = list(symbols)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def build(cls, chars, extra=()) -> Vocab:
        return cls(SPECIALS + list(extra) + sorted(set(chars) - set(SPECIALS)))

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, lang: str | None = None, bos: bool = False) -> list[int]:
        ids = [BOS] if bos else []
        if lang:
            ids.append(self.stoi[f"<{lang}>"])
        ids += [self.stoi.get(c, UNK) for c in text]
        return ids + [EOS]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i >= len(SPECIALS) and not self.itos[i].startswith("<"):
                out.append(self.itos[i])
        return "".join(out)


class XlitTransformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 256,
        nhead: int = 4,
        layers: int = 4,
        ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 48,
    ):
        super().__init__()
        self.config = dict(src_vocab=src_vocab, tgt_vocab=tgt_vocab, d_model=d_model, nhead=nhead,
                           layers=layers, ff=ff, dropout=dropout, max_len=max_len)  # fmt: skip
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec, layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, tgt_vocab)

    def _embed(self, emb: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        return emb(x) * math.sqrt(self.d_model) + self.pos(pos)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._embed(self.src_emb, src), src_key_padding_mask=src == PAD)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_pad: torch.Tensor, tgt_pad: bool = True):
        L = tgt.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt.device), 1)
        h = self.decoder(
            self._embed(self.tgt_emb, tgt),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=(tgt == PAD) if tgt_pad else None,
            memory_key_padding_mask=src_pad,
        )
        return self.out(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.decode(tgt_in, self.encode(src), src == PAD)


@torch.no_grad()
def greedy_decode(model: XlitTransformer, src: torch.Tensor, max_len: int = 40) -> torch.Tensor:
    memory, pad = model.encode(src), src == PAD
    ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        logits = model.decode(ys, memory, pad, tgt_pad=False)[:, -1]
        logits[:, [PAD, BOS]] = float("-inf")
        nxt = logits.argmax(-1)
        nxt[done] = PAD
        ys = torch.cat([ys, nxt[:, None]], 1)
        done |= nxt == EOS
        if done.all():
            break
    return ys[:, 1:]


@torch.no_grad()
def score_targets(
    model: XlitTransformer, src: torch.Tensor, targets: list[list[int]], alpha: float = 0.7
) -> list[float]:
    """Length-normalized log-prob of given target ids (each ending in EOS), same scale as beam_search.

    src: (N, S), one row per target. Used to score extra candidates (e.g. from the rule engine).
    """
    L = max(map(len, targets))
    tgt_in = torch.full((len(targets), L), PAD, dtype=torch.long, device=src.device)
    gold = torch.full((len(targets), L), PAD, dtype=torch.long, device=src.device)
    for i, t in enumerate(targets):
        tgt_in[i, 0] = BOS
        tgt_in[i, 1 : len(t)] = torch.tensor(t[:-1])
        gold[i, : len(t)] = torch.tensor(t)
    logp = model(src, tgt_in).float().log_softmax(-1)
    tok = logp.gather(-1, gold[..., None]).squeeze(-1).masked_fill(gold == PAD, 0.0)
    return [float(s) / (len(t) ** alpha) for s, t in zip(tok.sum(1).tolist(), targets, strict=True)]


@torch.no_grad()
def beam_search(
    model: XlitTransformer, src: torch.Tensor, beam: int = 5, max_len: int = 40, alpha: float = 0.7
) -> list[list[tuple[list[int], float]]]:
    """Batched beam search. Returns, for each input, `beam` (token ids, length-normalized log-prob)."""
    B, dev = src.size(0), src.device
    memory = model.encode(src).repeat_interleave(beam, 0)
    pad = (src == PAD).repeat_interleave(beam, 0)
    seqs = torch.full((B * beam, 1), BOS, dtype=torch.long, device=dev)
    scores = torch.full((B, beam), float("-inf"), device=dev)
    scores[:, 0] = 0.0
    scores = scores.view(-1)
    finished = torch.zeros(B * beam, dtype=torch.bool, device=dev)
    for _ in range(max_len):
        logp = model.decode(seqs, memory, pad, tgt_pad=False)[:, -1].float().log_softmax(-1)
        V = logp.size(-1)
        logp[:, [PAD, BOS]] = float("-inf")  # never real output tokens
        logp[finished] = float("-inf")
        logp[finished, PAD] = 0.0  # finished beams just carry their score forward
        cand = (scores[:, None] + logp).view(B, beam * V)
        top, idx = cand.topk(beam, dim=1)
        src_beam = (torch.arange(B, device=dev)[:, None] * beam + idx // V).view(-1)
        tok = (idx % V).view(-1)
        seqs = torch.cat([seqs[src_beam], tok[:, None]], 1)
        finished = finished[src_beam] | (tok == EOS)
        scores = top.view(-1)
        if finished.all():
            break
    out = []
    seqs, scores = seqs[:, 1:].tolist(), scores.tolist()
    for b in range(B):
        hyps = []
        for j in range(b * beam, (b + 1) * beam):
            ids = seqs[j]
            length = ids.index(EOS) + 1 if EOS in ids else len(ids)
            hyps.append((ids[:length], scores[j] / (length**alpha)))
        hyps.sort(key=lambda h: -h[1])
        out.append(hyps)
    return out
