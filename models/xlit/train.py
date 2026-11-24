"""Train the transliteration model: Kannada first, then Tulu (transfer learning).

    # quick local smoke test (CPU, a few minutes)
    python -m models.xlit.train --data data/xlit/kaggle --out runs/xlit_smoke --max-train 3000 \
        --kn-epochs 1 --tcy-epochs 1 --eval-limit 100

    # real run (Kaggle GPU) -- see notebooks/kaggle_xlit/
    python -m models.xlit.train --data /kaggle/input/akshara-xlit-data --out /kaggle/working/xlit

Stage 1: learn the Kannada script from ~2.9M Aksharantar pairs.
Stage 2: continue on Tulu pairs with ~20% Kannada mixed in, so it doesn't forget Kannada.
Final:   beam search (top-5) + word-list re-rank, scored on every test set we have.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

from common.metrics import cer, topk_accuracy
from models.xlit.model import (
    LANG_TOKENS,
    PAD,
    Vocab,
    XlitTransformer,
    beam_search,
    greedy_decode,
)

MAX_SRC, MAX_TGT = 33, 33  # 30 chars + lang tag/BOS + EOS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_tsv(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    pairs = []
    if not path.exists():
        return pairs
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
                if limit and len(pairs) >= limit:
                    break
    return pairs


def read_wordfreq(path: Path) -> dict[str, int]:
    out = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            w, _, c = line.rstrip("\n").partition("\t")
            if c.isdigit():
                out[w] = int(c)
    return out


def encode_pairs(pairs, lang, sv: Vocab, tv: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.full((len(pairs), MAX_SRC), PAD, dtype=torch.int16)
    tgt = torch.full((len(pairs), MAX_TGT + 1), PAD, dtype=torch.int16)
    for i, (r, n) in enumerate(pairs):
        s = sv.encode(r, lang=lang)[:MAX_SRC]
        t = tv.encode(n, bos=True)[: MAX_TGT + 1]
        src[i, : len(s)] = torch.tensor(s, dtype=torch.int16)
        tgt[i, : len(t)] = torch.tensor(t, dtype=torch.int16)
    return src, tgt


def trim(x: torch.Tensor) -> torch.Tensor:
    """Drop all-padding columns at the end of a batch."""
    n = int((x != PAD).sum(1).max())
    return x[:, : max(n, 1)].long()


class Trainer:
    def __init__(self, model, device, lr, total_steps, warmup, amp):
        self.model, self.device = model, device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0)))),
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
        self.amp = amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def step(self, src, tgt) -> float:
        self.model.train()
        src, tgt = trim(src).to(self.device), trim(tgt).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            logits = self.model(src, tgt[:, :-1])
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)).float(), tgt[:, 1:].reshape(-1))
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        return float(loss.detach())


def greedy_accuracy(model, src, tgt, tv: Vocab, device, bs=1024) -> float:
    model.eval()
    hits = 0
    for i in range(0, len(src), bs):
        s = trim(src[i : i + bs]).to(device)
        pred = greedy_decode(model, s).cpu()
        for p, t in zip(pred, tgt[i : i + bs], strict=True):
            hits += tv.decode(p) == tv.decode(t[1:])
    return hits / max(len(src), 1)


def evaluate(model, pairs, lang, sv, tv, device, wordfreq, beam=5, bs=256, bonus=1.5) -> dict:
    """Top-1 / top-5 / CER with plain beam search and with word-list re-ranking."""
    model.eval()
    refs = [n for _, n in pairs]
    plain, reranked = [], []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i : i + bs]
        src = torch.full((len(chunk), MAX_SRC), PAD, dtype=torch.long)
        for j, (r, _) in enumerate(chunk):
            s = sv.encode(r, lang=lang)[:MAX_SRC]
            src[j, : len(s)] = torch.tensor(s)
        for hyps in beam_search(model, trim(src).to(device), beam=beam):
            cands = []
            for ids, score in hyps:
                w = tv.decode(ids)
                if w and w not in [c for c, _ in cands]:
                    cands.append((w, score))
            plain.append([c for c, _ in cands] or [""])
            rr = sorted(cands, key=lambda x: -(x[1] + (bonus if x[0] in wordfreq else 0.0)))
            reranked.append([c for c, _ in rr] or [""])
    res = {"n": len(pairs)}
    for name, cands in (("beam", plain), ("beam+wordlist", reranked)):
        res[name] = {
            "top1": round(topk_accuracy(refs, cands, 1), 4),
            "top5": round(topk_accuracy(refs, cands, 5), 4),
            "cer": round(cer(refs, [c[0] for c in cands]), 4),
        }
    res["examples"] = [
        {"roman": r, "gold": n, "pred": c[:3]} for (r, n), c in list(zip(pairs, reranked, strict=True))[:15]
    ]
    return res


def save(model, sv, tv, path: Path, extra: dict | None = None) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config, "src_vocab": sv.itos, "tgt_vocab": tv.itos, **(extra or {})},
        path,
    )


def run_epochs(trainer, src, tgt, epochs, batch, val, tv, device, ckpt, sv, deadline, tag, patience=None):
    best, bad = -1.0, 0
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(src))
        losses = []
        t0 = time.time()
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            losses.append(trainer.step(src[idx], tgt[idx]))
            if time.time() > deadline:
                break
        acc = greedy_accuracy(trainer.model, *val, tv, device)
        log(f"{tag} epoch {ep}: loss {sum(losses) / len(losses):.4f}  val top1(greedy) {acc:.4f}  ({time.time() - t0:.0f}s)")
        if acc > best:
            best, bad = acc, 0
            save(trainer.model, sv, tv, ckpt, {"stage": tag, "epoch": ep, "val_top1": acc})
        else:
            bad += 1
            if patience and bad >= patience:
                log(f"{tag}: early stop")
                break
        if time.time() > deadline:
            log("time budget reached")
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kn-epochs", type=int, default=8)
    ap.add_argument("--tcy-epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--tcy-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--tcy-lr", type=float, default=2e-4)
    ap.add_argument("--kn-mix", type=float, default=0.2, help="share of Kannada in stage-2 batches")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=None, help="limit Kannada train pairs (smoke tests)")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--hours", type=float, default=8.0, help="training time budget")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    deadline = time.time() + a.hours * 3600
    log(f"device={device} {torch.cuda.get_device_name(0) if amp else ''}")

    d = {f"{lang}_{s}": read_tsv(a.data / f"{lang}_{s}.tsv", a.max_train if (lang, s) == ("kn", "train") else None)
         for lang in ("kn", "tcy") for s in ("train", "valid", "test")}  # fmt: skip
    human = read_tsv(a.data / "human_tcy_test.tsv")
    for k, v in d.items():
        log(f"{k}: {len(v)} pairs")

    all_pairs = [p for v in d.values() for p in v] + human
    sv = Vocab.build({c for r, _ in all_pairs for c in r}, extra=LANG_TOKENS)
    tv = Vocab.build({c for _, n in all_pairs for c in n})
    log(f"vocab: src {len(sv)}  tgt {len(tv)}")

    enc = {k: encode_pairs(v, k.split("_")[0], sv, tv) for k, v in d.items()}
    if amp:  # keep the (int16) data on the GPU: fast batching
        enc = {k: (s.to(device), t.to(device)) for k, (s, t) in enc.items()}

    model = XlitTransformer(len(sv), len(tv), d_model=a.d_model, layers=a.layers, ff=4 * a.d_model).to(device)
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    wf = {lang: read_wordfreq(a.data / f"{lang}_wordfreq.tsv") for lang in ("kn", "tcy")}
    lim = a.eval_limit
    val_lim = lambda k: tuple(x[: lim or 2000] for x in enc[k])  # noqa: E731

    # ---- Stage 1: Kannada ----
    kn_src, kn_tgt = enc["kn_train"]
    steps = a.kn_epochs * math.ceil(len(kn_src) / a.batch)
    tr = Trainer(model, device, a.lr, steps, warmup=min(2000, steps // 10 + 1), amp=amp)
    ckpt1 = a.out / "xlit_kn.pt"
    run_epochs(tr, kn_src, kn_tgt, a.kn_epochs, a.batch, val_lim("kn_valid"), tv, device, ckpt1, sv, deadline, "kn")
    model.load_state_dict(torch.load(ckpt1, map_location=device)["state_dict"])

    metrics = {"stage1_kannada_only": {
        "kn_test": evaluate(model, d["kn_test"][:lim], "kn", sv, tv, device, wf["kn"]),
        "tcy_test_zero_shot": evaluate(model, d["tcy_test"][:lim], "tcy", sv, tv, device, wf["tcy"]),
    }}  # fmt: skip
    log(f"stage 1 done: {json.dumps({k: v['beam+wordlist'] for k, v in metrics['stage1_kannada_only'].items()})}")

    # ---- Stage 2: Tulu (+ some Kannada so it doesn't forget) ----
    t_src, t_tgt = enc["tcy_train"]
    n_kn = int(len(t_src) * a.kn_mix / (1 - a.kn_mix))
    pick = torch.randperm(len(kn_src), device=kn_src.device)[:n_kn]
    mix_src = torch.cat([t_src, kn_src[pick]])
    mix_tgt = torch.cat([t_tgt, kn_tgt[pick]])
    steps = a.tcy_epochs * math.ceil(len(mix_src) / a.tcy_batch)
    tr = Trainer(model, device, a.tcy_lr, steps, warmup=100, amp=amp)
    ckpt2 = a.out / "xlit.pt"
    run_epochs(tr, mix_src, mix_tgt, a.tcy_epochs, a.tcy_batch, val_lim("tcy_valid"), tv, device, ckpt2, sv,
               deadline + 3600, "tcy", patience=6)  # fmt: skip
    model.load_state_dict(torch.load(ckpt2, map_location=device)["state_dict"])

    metrics["final"] = {
        "kn_test": evaluate(model, d["kn_test"][:lim], "kn", sv, tv, device, wf["kn"]),
        "tcy_test_machine_made": evaluate(model, d["tcy_test"][:lim], "tcy", sv, tv, device, wf["tcy"]),
    }
    if human:
        metrics["final"]["tcy_test_HUMAN"] = evaluate(model, human, "tcy", sv, tv, device, wf["tcy"])
    metrics["config"] = {**vars(a), "data": str(a.data), "out": str(a.out), "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    (a.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log("final: " + json.dumps({k: v["beam+wordlist"] for k, v in metrics["final"].items()}))
    log(f"saved {ckpt2} and metrics.json")


if __name__ == "__main__":
    main()
