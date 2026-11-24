"""Count every word in the corpus -> word-frequency list (the dictionary).

    python -m scripts.build_wordfreq --lang tcy --min-count 2
    python -m scripts.build_wordfreq --lang kn  --min-count 3

Writes data/corpus/<lang>_wordfreq.tsv  (word<TAB>count, most common first).
"""

from __future__ import annotations

import argparse
from collections import Counter

from common.paths import CORPUS
from common.text import split_aksharas, split_words


def main(lang: str, min_count: int) -> None:
    src = CORPUS / f"{lang}_sentences.txt"
    out = CORPUS / f"{lang}_wordfreq.tsv"
    counts: Counter[str] = Counter()
    with open(src, encoding="utf-8") as f:
        for line in f:
            counts.update(split_words(line, kannada_only=True))
    kept = 0
    with open(out, "w", encoding="utf-8") as w:
        for word, c in counts.most_common():
            if c < min_count:
                break
            # a lone vowel sign / virama is not a word
            if not split_aksharas(word) or word[0] in "ಾಿೀುೂೆೇೈೊೋೌ್ಂಃ":
                continue
            w.write(f"{word}\t{c}\n")
            kept += 1
    print(f"{lang}: {len(counts)} distinct words, kept {kept} with count >= {min_count} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["kn", "tcy"], required=True)
    ap.add_argument("--min-count", type=int, default=3)
    a = ap.parse_args()
    main(a.lang, a.min_count)
