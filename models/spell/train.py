"""Train the context spell model (Phase 2): wrong text window -> right text window.

    # quick local smoke test (CPU, a few minutes)
    python -m models.spell.train --data data/spell/kaggle --out runs/spell_smoke --max-train 3000 \
        --epochs 1 --eval-limit 100

    # real run (Kaggle GPU) -- see notebooks/kaggle_spell/
    python -m models.spell.train --data /kaggle/input/akshara-spell-data --out /kaggle/working/spell

One model for Kannada and Tulu (a <kn>/<tcy> tag in front), trained on both at once (Tulu windows
are already doubled in the data). Same small char Transformer as transliteration.
After training: tune the two decision margins on VALID for the best F0.5 (precision first), then
score TEST against the dictionary baseline (SymSpell + Kannada-aware distance).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import torch

from common.metrics import spell_prf
from models.spell.checker import SpellChecker
from models.spell.neural import MAX_CHARS, NeuralSpellChecker
from models.xlit.model import LANG_TOKENS, PAD, Vocab, XlitTransformer, greedy_decode
from models.xlit.train import Trainer, log, read_wordfreq, trim

LANGS = ("kn", "tcy")
MAX_LEN = MAX_CHARS + 2  # + lang tag/BOS + EOS
UNK_GRID = [-1.0, 0.0, 1.0, 2.0, 3.0, 5.0]
KNOWN_GRID = [2.0, 3.0, 4.0, 6.0, 8.0, float("inf")]


def read_pairs(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    pairs = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            p = line.rstrip("\r\n").split("\t")
            if len(p) == 2 and len(p[0]) <= MAX_CHARS and len(p[1]) <= MAX_CHARS:
                pairs.append((p[0], p[1]))
                if limit and len(pairs) >= limit:
                    break
    return pairs


def encode(pairs, lang, v: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.full((len(pairs), MAX_LEN), PAD, dtype=torch.int16)
    tgt = torch.full((len(pairs), MAX_LEN + 1), PAD, dtype=torch.int16)
    for i, (w, r) in enumerate(pairs):
        s, t = v.encode(w, lang=lang), v.encode(r, bos=True)
        src[i, : len(s)] = torch.tensor(s, dtype=torch.int16)
        tgt[i, : len(t)] = torch.tensor(t, dtype=torch.int16)
    return src, tgt


@torch.no_grad()
def greedy_exact(model, pairs, lang, v, device, bs=256) -> float:
    """Share of windows the model rewrites exactly right (a quick learning signal)."""
    model.eval()
    hits = 0
    for i in range(0, len(pairs), bs):
        src, _ = encode(pairs[i : i + bs], lang, v)
        pred = greedy_decode(model, trim(src).to(device), max_len=MAX_LEN + 2).tolist()
        hits += sum(v.decode(p) == r for p, (_, r) in zip(pred, pairs[i : i + bs], strict=True))
    return hits / max(len(pairs), 1)


def save(model, v, path, extra=None) -> None:
    torch.save({"state_dict": model.state_dict(), "config": model.config, "vocab": v.itos, **(extra or {})}, path)


def prf(pairs, preds) -> dict:
    r = spell_prf([w for w, _ in pairs], [c for _, c in pairs], preds)
    return {k: round(x, 4) if isinstance(x, float) else x for k, x in r.items()}


def tune_and_test(nsc: NeuralSpellChecker, valid, test, lang) -> dict:
    t0 = time.time()
    vp = nsc.propose([w for w, _ in valid], lang)
    best = None
    for mu, mk in itertools.product(UNK_GRID, KNOWN_GRID):
        m = {"unknown": mu, "known": mk}
        r = prf(valid, [nsc.decide(w, s, lang, m) for (w, _), s in zip(valid, vp, strict=True)])
        if best is None or r["f0.5"] > best[1]["f0.5"]:
            best = (m, r)
    margins, val_res = best
    nsc.margins[lang] = dict(margins)
    tp = nsc.propose([w for w, _ in test], lang)
    ms = (time.time() - t0) * 1000 / max(len(valid) + len(test), 1)
    res = {"margins": margins, "valid": val_res,
           "test": prf(test, [nsc.decide(w, s, lang) for (w, _), s in zip(test, tp, strict=True)]),
           "test_no_real_word_fixes": prf(test, [nsc.decide(w, s, lang, {**margins, "known": math.inf})
                                                 for (w, _), s in zip(test, tp, strict=True)]),
           "ms_per_window_gpu": round(ms, 1)}  # fmt: skip
    examples = []
    for (w, r), s in zip(test, tp, strict=True):
        p = nsc.decide(w, s, lang)
        if w != r or p != w:
            examples.append({"wrong": w, "right": r, "pred": p})
        if len(examples) >= 20:
            break
    res["examples"] = examples
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=None, help="limit train pairs per language")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--hours", type=float, default=1.5, help="training time budget")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    deadline = time.time() + a.hours * 3600
    log(f"device={device} {torch.cuda.get_device_name(0) if amp else ''}")

    d = {}
    for lang in LANGS:
        d[(lang, "train")] = read_pairs(a.data / f"{lang}_train.tsv", a.max_train)
        for split in ("valid", "test"):
            d[(lang, split)] = read_pairs(a.data / f"{lang}_{split}.tsv", a.eval_limit)
    for k, x in d.items():
        log(f"{k}: {len(x)} pairs")
    v = Vocab.build({c for x in d.values() for p in x for s in p for c in s}, extra=LANG_TOKENS)
    log(f"vocab: {len(v)}")

    parts = [encode(d[(lang, "train")], lang, v) for lang in LANGS]
    src, tgt = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    if amp:
        src, tgt = src.to(device), tgt.to(device)
    model = XlitTransformer(len(v), len(v), d_model=a.d_model, layers=a.layers, ff=4 * a.d_model,
                            max_len=MAX_LEN + 8).to(device)  # fmt: skip
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  train windows: {len(src)}")

    steps = a.epochs * math.ceil(len(src) / a.batch)
    tr = Trainer(model, device, a.lr, steps, warmup=min(2000, steps // 10 + 1), amp=amp)
    val = {lang: d[(lang, "valid")][:500] for lang in LANGS}
    ckpt, best = a.out / "spell.pt", -1.0
    for ep in range(1, a.epochs + 1):
        perm, losses, t0 = torch.randperm(len(src), device=src.device), [], time.time()
        for i in range(0, len(perm), a.batch):
            idx = perm[i : i + a.batch]
            losses.append(tr.step(src[idx], tgt[idx]))
            if len(losses) % 1000 == 0:
                log(f"  step {len(losses)}: loss {sum(losses[-1000:]) / 1000:.4f}")
            if time.time() > deadline:
                break
        acc = {lang: round(greedy_exact(model, val[lang], lang, v, device), 4) for lang in LANGS}
        score = sum(acc.values()) / len(acc)
        log(f"epoch {ep}: loss {sum(losses) / len(losses):.4f}  val exact {acc}  ({time.time() - t0:.0f}s)")
        if score > best:
            best = score
            save(model, v, ckpt, {"epoch": ep, "val_exact": acc})
        if time.time() > deadline:
            log("time budget reached")
            break

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    checkers = {lang: SpellChecker(read_wordfreq(a.data / f"{lang}_wordfreq.tsv")) for lang in LANGS}
    nsc = NeuralSpellChecker(model=model, vocab=v, checkers=checkers, device=device, batch=128)
    metrics = {"config": {**vars(a), "data": str(a.data), "out": str(a.out),
                          "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}}  # fmt: skip
    for lang in LANGS:
        test = d[(lang, "test")]
        base = prf(test, [checkers[lang].correct(w) for w, _ in test])
        res = tune_and_test(nsc, d[(lang, "valid")], test, lang)
        metrics[lang] = {"n_test": len(test), "dictionary_baseline": base, "context_model": res}
        log(f"{lang}: baseline {base}  |  model {res['test']}  margins {res['margins']}")
    save(model, v, ckpt, {"epoch": ck["epoch"], "val_exact": ck["val_exact"], "margins": nsc.margins})
    (a.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved {ckpt} and metrics.json")


if __name__ == "__main__":
    main()
