"""Pack spell-checker training data into small TSV files for Kaggle / Colab (Phase 2).

    python -m scripts.prepare_spell_data
    cd data/spell/kaggle && kaggle datasets create -p .        # first time (private)

Output folder data/spell/kaggle/:
    {kn,tcy}_{train,valid,test}.tsv   wrong<TAB>right text windows (<= 96 characters, whole words)
    {kn,tcy}_wordfreq.tsv             dictionaries (the checker only fixes towards known words)

Sentences come from the cleaned Wikipedia corpus and are split into train/valid/test BY SENTENCE.
Typos are made by models/spell/noise.py (vowel-sign swaps, ಲ/ಳ, virama, ...). 20% of windows are
left clean, so the model learns to copy correct text and the test also counts false alarms.
These synthetic tests only say "the model learned the typo patterns". The honest exam is the
real-mistakes test set typed by volunteers.
"""

from __future__ import annotations

import json
import random

from common.paths import CORPUS, SPELL
from common.text import kannada_ratio, split_words
from models.spell.noise import corrupt_sentence

OUT = SPELL / "kaggle"
MAX_CHARS = 96
CLEAN_SHARE = 0.2
SIZES = {"kn": {"train": 500_000, "eval": 2000}, "tcy": {"train": None, "eval": 2000}}
TCY_COPIES = 2  # Tulu has few sentences: noise each training window twice (different typos)


def windows(sentence: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Cut a sentence into chunks of whole words, each at most max_chars long."""
    out, cur = [], ""
    for w in sentence.split():
        if len(w) > max_chars:
            continue
        if cur and len(cur) + 1 + len(w) > max_chars:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    return [x for x in out if len(split_words(x, kannada_only=True)) >= 2 and kannada_ratio(x) >= 0.8]


def make_pairs(sents: list[str], rng: random.Random, copies: int = 1) -> list[tuple[str, str]]:
    pairs = []
    for s in sents:
        for win in windows(s):
            for _ in range(copies):
                wrong = win if rng.random() < CLEAN_SHARE else corrupt_sentence(win, rng, rate=0.15)
                pairs.append((wrong, win))
    rng.shuffle(pairs)
    return pairs


def _write(name: str, pairs: list[tuple[str, str]]) -> None:
    with open(OUT / name, "w", encoding="utf-8") as f:
        for w, r in pairs:
            f.write(f"{w}\t{r}\n")
    print(f"{name}: {len(pairs)} pairs")


def main(seed: int = 13) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for lang, size in SIZES.items():
        rng = random.Random(f"{seed}-{lang}")
        with open(CORPUS / f"{lang}_sentences.txt", encoding="utf-8") as f:
            sents = sorted({s.strip().replace("\t", " ") for s in f if 3 <= len(s.split()) <= 60})
        rng.shuffle(sents)
        n = size["eval"]
        test, valid, train = sents[:n], sents[n : 2 * n], sents[2 * n :]
        if size["train"]:
            train = train[: size["train"]]
        _write(f"{lang}_test.tsv", make_pairs(test, rng))
        _write(f"{lang}_valid.tsv", make_pairs(valid, rng))
        _write(f"{lang}_train.tsv", make_pairs(train, rng, copies=TCY_COPIES if lang == "tcy" else 1))
        with open(CORPUS / f"{lang}_wordfreq.tsv", encoding="utf-8") as src, open(
            OUT / f"{lang}_wordfreq.tsv", "w", encoding="utf-8"
        ) as dst:
            for i, line in enumerate(src):
                if lang == "kn" and i >= 200_000:
                    break
                dst.write(line)

    meta = {
        "title": "akshara-spell-data",
        "id": "nikhithakotian/akshara-spell-data",
        "licenses": [{"name": "CC-BY-SA-4.0"}],
        "subtitle": "Kannada + Tulu wrong->right text windows (synthetic typos) + dictionaries",
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if (SPELL / "README.md").exists():
        (OUT / "README.md").write_text((SPELL / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"ready: {OUT}")


if __name__ == "__main__":
    main()
