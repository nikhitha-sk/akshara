"""Pack transliteration data into small TSV files for training on Kaggle / Colab.

    python -m scripts.prepare_xlit_data
    kaggle datasets create -p data/xlit/kaggle      # first time (private)
    kaggle datasets version -p data/xlit/kaggle -m "update"   # later

Output folder data/xlit/kaggle/:
    kn_{train,valid,test}.tsv     Aksharantar Kannada (roman<TAB>native)
    kn_casual.tsv                 extra TRAIN pairs: casual WhatsApp-style spellings of common Kannada
                                  words (ooru -> ಊರು), made with the reverse trick. Aksharantar
                                  rarely spells ಊ as "oo" or ಈ as "ee", but real users do.
    tcy_{train,valid,test}.tsv    Tulu reverse-trick pairs, split BY WORD (no word in two splits)
    {kn,tcy}_wordfreq.tsv         word lists for re-ranking
    human_tcy_test.tsv            (only if volunteers have typed pairs) the honest Tulu test

The Tulu valid/test here are machine-made, so they measure "did the model learn Tulu
spelling patterns", not real-user accuracy. The human test set is the real exam.
"""

from __future__ import annotations

import json
import random
import shutil

from common.paths import CORPUS, ROOT, VOLUNTEER, XLIT
from common.text import kannada_ratio, normalize, split_words
from models.xlit.rules import kannada_to_roman

OUT = XLIT / "kaggle"
MAX_LEN = 30
CASUAL_WORDS = 150_000  # most frequent Kannada words used for kn_casual.tsv
CASUAL_VARIANTS = 4


def _ok(roman: str, native: str) -> bool:
    return (
        0 < len(roman) <= MAX_LEN
        and 0 < len(native) <= MAX_LEN
        and roman.isascii()
        and roman.isalpha()
    )


def _write(name: str, pairs: list[tuple[str, str]]) -> None:
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r, n in pairs:
            f.write(f"{r}\t{n}\n")
    print(f"{name}: {len(pairs)} pairs")


def main(seed: int = 13) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    held_out: set[str] = set()
    for split in ("train", "valid", "test"):
        pairs, seen = [], set()
        with open(XLIT / f"kn_{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                p = (row["roman"], normalize(row["native"]))
                if _ok(*p) and p not in seen:
                    seen.add(p)
                    pairs.append(p)
        _write(f"kn_{split}.tsv", pairs)
        if split != "train":
            held_out.update(n for _, n in pairs)

    # Casual spellings for common words; words in kn valid/test and in the hand-made casual check
    # (reports/casual_check_kn.tsv) are left out, so those scores stay honest.
    check = ROOT / "reports" / "casual_check_kn.tsv"
    if check.exists():
        rows = [line.rstrip("\r\n").split("\t") for line in open(check, encoding="utf-8")]
        held_out.update(r[1] for r in rows if len(r) == 2)
    casual = []
    with open(CORPUS / "kn_wordfreq.tsv", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= CASUAL_WORDS:
                break
            w = line.split("\t")[0]
            if w in held_out or kannada_ratio(w) < 1.0:
                continue
            casual += [
                (r, w) for r in kannada_to_roman(w, max_variants=CASUAL_VARIANTS) if _ok(r, w)
            ]
    random.Random(seed + 1).shuffle(casual)  # own RNG: keeps the Tulu split unchanged
    _write("kn_casual.tsv", casual)

    by_word: dict[str, list[str]] = {}
    with open(XLIT / "tcy_reverse_pairs.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if _ok(row["roman"], row["native"]):
                by_word.setdefault(row["native"], []).append(row["roman"])
    words = sorted(by_word)
    rng.shuffle(words)
    n_eval = min(500, len(words) // 10)
    splits = {
        "test": words[:n_eval],
        "valid": words[n_eval : 2 * n_eval],
        "train": words[2 * n_eval :],
    }
    for split, ws in splits.items():
        _write(f"tcy_{split}.tsv", [(r, w) for w in ws for r in by_word[w]])

    # Human-typed Tulu pairs from the volunteer app (word-aligned sentences only)
    human = []
    typed = VOLUNTEER / "typed.jsonl"
    if typed.exists():
        for line in open(typed, encoding="utf-8"):
            row = json.loads(line)
            rw, nw = (
                split_words(row.get("roman", "").lower()),
                split_words(normalize(row["tulu"]), kannada_only=True),
            )
            if rw and len(rw) == len(nw):
                human += [(r, n) for r, n in zip(rw, nw, strict=True) if _ok(r, n)]
    if human:
        _write("human_tcy_test.tsv", human)

    for lang, top in (("kn", 200_000), ("tcy", None)):
        with (
            open(CORPUS / f"{lang}_wordfreq.tsv", encoding="utf-8") as src,
            open(OUT / f"{lang}_wordfreq.tsv", "w", encoding="utf-8") as dst,
        ):
            for i, line in enumerate(src):
                if top and i >= top:
                    break
                dst.write(line)

    meta = {
        "title": "akshara-xlit-data",
        "id": "nikhithakotian/akshara-xlit-data",
        "licenses": [{"name": "CC-BY-SA-4.0"}],
        "subtitle": "Kannada (Aksharantar) + Tulu romanized-to-Kannada-script word pairs",
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.copy(XLIT / "README.md", OUT / "README.md")
    print(f"ready: {OUT}")


if __name__ == "__main__":
    main()
