"""Compare transliteration models on exactly the same test words.

    python -m scripts.compare_xlit --ckpt models/xlit/checkpoints/xlit.pt \
        --indicxlit runs/indicxlit   # folder with indicxlit_{kn,tcy}_test.tsv from the Kaggle job

Rows scored = the words IndicXlit finished (or the whole test file if no IndicXlit output).
Appends one line per (model, test set) to reports/baselines.jsonl and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from common.metrics import cer, topk_accuracy
from common.paths import REPORTS, XLIT
from common.text import normalize
from models.spell.checker import load_wordfreq
from models.xlit.rules import RuleTransliterator

KAGGLE = XLIT / "kaggle"
TESTS = {"kn_test": "kn", "tcy_test": "tcy", "casual_check_kn": "kn"}
CASUAL = REPORTS / "casual_check_kn.tsv"  # 62 hand-written everyday words (ooru, oota, enu...)


def read_rows(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\r\n").split("\t") for line in f if line.strip()]


def score(refs: list[str], cands: list[list[str]], ms: float) -> dict:
    # NFC + drop invisible marks (IndicXlit adds ZWNJ after a word-final virama) so only real
    # spelling differences count as errors.
    refs = [normalize(r) for r in refs]
    cands = [[normalize(x) for x in c] or [""] for c in cands]
    return {
        "top1": round(topk_accuracy(refs, cands, 1), 4),
        "top5": round(topk_accuracy(refs, cands, 5), 4),
        "cer": round(cer(refs, [c[0] for c in cands]), 4),
        "ms_per_word": round(ms, 2),
    }


def timed(fn, words: list[str]) -> tuple[list[list[str]], float]:
    t0 = time.perf_counter()
    out = fn(words)
    return out, (time.perf_counter() - t0) * 1000 / max(len(words), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models/xlit/checkpoints/xlit.pt"))
    ap.add_argument("--ckpt-kn", type=Path, default=Path("models/xlit/checkpoints/xlit_kn.pt"),
                    help="Kannada-only checkpoint used for kn_test (falls back to --ckpt)")  # fmt: skip
    ap.add_argument("--indicxlit", type=Path, default=Path("runs/indicxlit"))
    ap.add_argument("--data", type=Path, default=KAGGLE)
    ap.add_argument("--tag", default="", help="model version shown in the table, e.g. v2")
    a = ap.parse_args()

    freqs = {lang: load_wordfreq(a.data / f"{lang}_wordfreq.tsv") for lang in ("kn", "tcy")}
    neural = {}
    if a.ckpt.exists():
        from models.xlit.neural import NeuralTransliterator

        neural["tcy"] = NeuralTransliterator(a.ckpt, freqs)
        neural["kn"] = (
            NeuralTransliterator(a.ckpt_kn, freqs) if a.ckpt_kn.exists() else neural["tcy"]
        )

    results = []
    for test, lang in TESTS.items():
        ix_file = a.indicxlit / f"indicxlit_{test}.tsv"
        gold = CASUAL if test == "casual_check_kn" else a.data / f"{test}.tsv"
        if ix_file.exists():
            ix_rows = [r for r in read_rows(ix_file) if len(r) >= 2]
            rows = [(r[0], r[1]) for r in ix_rows]
        elif gold.exists():
            ix_rows, rows = [], [tuple(r[:2]) for r in read_rows(gold)]
        else:
            continue
        romans, refs = [r for r, _ in rows], [n for _, n in rows]
        models = {}
        if ix_rows:
            models["AI4Bharat IndicXlit (kn)"] = ([r[2:7] for r in ix_rows], float("nan"))
        rules = RuleTransliterator(freqs[lang])
        models["rules+wordlist"] = timed(
            lambda ws, x=rules: [x.word_candidates(w, k=5) for w in ws], romans
        )
        if lang in neural:
            models[f"char-transformer+wordlist (ours{' ' + a.tag if a.tag else ''})"] = timed(
                lambda ws, lg=lang: neural[lg].candidates(ws, lg, k=5), romans
            )
        for name, (cands, ms) in models.items():
            res = {"tool": "xlit", "model": name, "lang": lang, "test_file": test, "n": len(rows),
                   **score(refs, cands, ms), "date": date.today().isoformat()}  # fmt: skip
            if ms != ms:  # NaN: timing not measured here (ran on Kaggle)
                res.pop("ms_per_word")
            results.append(res)

    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "baselines.jsonl", "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("| Test | Model | n | Top-1 | Top-5 | CER | ms/word |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['test_file']} | {r['model']} | {r['n']:,} | {r['top1']:.1%} | {r['top5']:.1%} "
              f"| {r['cer']:.1%} | {r.get('ms_per_word', '—')} |")  # fmt: skip


if __name__ == "__main__":
    main()
