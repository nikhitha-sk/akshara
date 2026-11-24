"""Make synthetic (wrong -> right) spell-checker pairs (Dataset 2).

    python -m scripts.make_synthetic_spell --lang tcy --n 50000
    python -m scripts.make_synthetic_spell --lang kn  --n 200000

Writes data/spell/<lang>_synthetic.jsonl rows: {"wrong": ..., "right": ..., "lang": ...}
These are TRAIN data only. The test set must be real mistakes typed by people.
"""

from __future__ import annotations

import argparse
import json
import random

from common.paths import CORPUS, SPELL
from models.spell.noise import corrupt_sentence


def main(lang: str, n: int, rate: float, seed: int) -> None:
    rng = random.Random(seed)
    with open(CORPUS / f"{lang}_sentences.txt", encoding="utf-8") as f:
        sentences = [s.strip() for s in f if 3 <= len(s.split()) <= 30]
    rng.shuffle(sentences)
    SPELL.mkdir(parents=True, exist_ok=True)
    out = SPELL / f"{lang}_synthetic.jsonl"
    made = 0
    with open(out, "w", encoding="utf-8") as w:
        for s in sentences[:n]:
            wrong = corrupt_sentence(s, rng, rate=rate)
            if wrong == s:
                continue
            w.write(json.dumps({"lang": lang, "wrong": wrong, "right": s}, ensure_ascii=False) + "\n")
            made += 1
    print(f"{lang}: {made} synthetic pairs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["kn", "tcy"], required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    main(a.lang, a.n, a.rate, a.seed)
