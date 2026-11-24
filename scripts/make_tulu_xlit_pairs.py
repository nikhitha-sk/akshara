"""Tulu transliteration pairs with the "reverse trick" (Dataset 1, Tulu method A).

    python -m scripts.make_tulu_xlit_pairs --top 8000

Takes the most common Tulu words from data/corpus/tcy_wordfreq.tsv, makes 2-3 realistic
English-letter spellings for each with the rule-based converter, and writes:
  * data/xlit/tcy_reverse_pairs.jsonl  (training pairs, marked needs_review)
  * data/xlit/tcy_review_sheet.csv     (open in Excel/Sheets and fix spellings by hand)

IMPORTANT: script-made pairs go to TRAIN only. The Tulu TEST set (500 pairs) must be
typed by real people (recorder_app "type" mode collects these).
"""

from __future__ import annotations

import argparse
import csv
import json

from common.paths import CORPUS, XLIT
from models.spell.checker import load_wordfreq
from models.xlit.rules import kannada_to_roman


def main(top: int, variants: int) -> None:
    freq = load_wordfreq(CORPUS / "tcy_wordfreq.tsv")
    words = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1]) if len(w) >= 2][:top]
    XLIT.mkdir(parents=True, exist_ok=True)
    n = 0
    with (
        open(XLIT / "tcy_reverse_pairs.jsonl", "w", encoding="utf-8") as w,
        open(XLIT / "tcy_review_sheet.csv", "w", encoding="utf-8-sig", newline="") as c,
    ):
        sheet = csv.writer(c)
        sheet.writerow(["native", "count", "spelling_1", "spelling_2", "spelling_3", "ok? (y/n)", "fixed spellings (comma separated)"])
        for word in words:
            spellings = kannada_to_roman(word, max_variants=variants)
            for s in spellings:
                rec = {"lang": "tcy", "roman": s, "native": word, "source": "reverse_rule", "needs_review": True}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            sheet.writerow([word, freq[word], *(spellings + ["", "", ""])[:3], "", ""])
    print(f"{len(words)} Tulu words -> {n} pairs -> {XLIT / 'tcy_reverse_pairs.jsonl'}")
    print(f"review sheet -> {XLIT / 'tcy_review_sheet.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8000)
    ap.add_argument("--variants", type=int, default=3)
    a = ap.parse_args()
    main(a.top, a.variants)
