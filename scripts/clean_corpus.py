"""Clean raw text into one-sentence-per-line corpus files (Section 7.4, step 4).

    python -m scripts.clean_corpus --lang tcy --inputs data/corpus/interim/tcy_wiki_plain.txt
    python -m scripts.clean_corpus --lang kn  --inputs data/corpus/interim/kn_wiki_plain.txt

Steps: NFC-normalize -> split sentences -> drop lines that are mostly English/numbers
-> drop very short/long lines -> remove duplicates. Writes data/corpus/<lang>_sentences.txt.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tqdm import tqdm

from common.paths import CORPUS
from common.text import kannada_ratio, normalize, split_sentences, split_words

_BRACKETS_RE = re.compile(r"\(\s*\)|\[\s*\]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_line(line: str) -> list[str]:
    line = normalize(line)
    line = _URL_RE.sub("", line)
    line = _BRACKETS_RE.sub("", line)
    out = []
    for s in split_sentences(line):
        s = re.sub(r"\s+", " ", s).strip(" -–—,;:")
        words = split_words(s)
        if not (3 <= len(words) <= 60):
            continue
        if kannada_ratio(s) < 0.85:
            continue
        digits = sum(c.isdigit() for c in s)
        if digits / max(len(s), 1) > 0.2:
            continue
        out.append(s)
    return out


def main(lang: str, inputs: list[Path], output: Path | None) -> None:
    output = output or CORPUS / f"{lang}_sentences.txt"
    seen: set[int] = set()
    kept = total = 0
    with open(output, "w", encoding="utf-8") as w:
        for path in inputs:
            with open(path, encoding="utf-8") as f:
                for line in tqdm(f, desc=path.name, unit=" lines", mininterval=2):
                    for s in clean_line(line):
                        total += 1
                        h = hash(s)
                        if h in seen:
                            continue
                        seen.add(h)
                        w.write(s + "\n")
                        kept += 1
    print(f"{lang}: kept {kept} unique sentences (of {total}) -> {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["kn", "tcy"], required=True)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    main(a.lang, a.inputs, a.output)
