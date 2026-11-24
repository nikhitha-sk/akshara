"""Score the rule-based baselines so the neural models have a number to beat.

    python -m scripts.eval_baselines xlit --lang kn --pairs data/xlit/kn_test.jsonl --limit 2000
    python -m scripts.eval_baselines spell --lang tcy --pairs data/spell/tcy_synthetic.jsonl --limit 1000

Results are appended to reports/baselines.jsonl and printed.
NOTE: scores on synthetic data are only a sanity check. The honest numbers come from
the human-made test sets (Tulu xlit test, real-mistakes spell test).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from common.metrics import cer, spell_prf, topk_accuracy
from common.paths import CORPUS, REPORTS
from models.spell.checker import SpellChecker, load_wordfreq
from models.xlit.rules import RuleTransliterator


def _read(path: Path, limit: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def eval_xlit(lang: str, pairs: Path, limit: int, use_dict: bool) -> dict:
    rows = _read(pairs, limit)
    freq = load_wordfreq(CORPUS / f"{lang}_wordfreq.tsv") if use_dict else {}
    xl = RuleTransliterator(freq)
    t0 = time.perf_counter()
    cands = [xl.word_candidates(r["roman"], k=5) for r in rows]
    ms = (time.perf_counter() - t0) * 1000 / max(len(rows), 1)
    refs = [r["native"] for r in rows]
    return {
        "tool": "xlit", "model": f"rules{'+wordlist' if use_dict else ''}", "lang": lang,
        "test_file": str(pairs), "n": len(rows),
        "top1": round(topk_accuracy(refs, cands, 1), 4),
        "top5": round(topk_accuracy(refs, cands, 5), 4),
        "cer": round(cer(refs, [c[0] for c in cands]), 4),
        "ms_per_word": round(ms, 2),
    }  # fmt: skip


def eval_spell(lang: str, pairs: Path, limit: int) -> dict:
    rows = _read(pairs, limit)
    sc = SpellChecker.from_file(CORPUS / f"{lang}_wordfreq.tsv")
    t0 = time.perf_counter()
    preds = [sc.correct(r["wrong"]) for r in rows]
    ms = (time.perf_counter() - t0) * 1000 / max(len(rows), 1)
    res = spell_prf([r["wrong"] for r in rows], [r["right"] for r in rows], preds)
    return {
        "tool": "spell", "model": "symspell+akshara-distance", "lang": lang,
        "test_file": str(pairs), "n": len(rows), **{k: round(v, 4) for k, v in res.items()},
        "ms_per_sentence": round(ms, 2),
    }  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["xlit", "spell"])
    ap.add_argument("--lang", choices=["kn", "tcy"], required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--no-dict", action="store_true", help="xlit: rules only, no word-list re-rank")
    a = ap.parse_args()
    if a.tool == "xlit":
        result = eval_xlit(a.lang, a.pairs, a.limit, not a.no_dict)
    else:
        result = eval_spell(a.lang, a.pairs, a.limit)
    result["date"] = date.today().isoformat()
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "baselines.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
