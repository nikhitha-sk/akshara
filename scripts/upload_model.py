"""Upload a trained model to the Hugging Face Hub (private by default).

    python -m scripts.upload_model xlit --ckpt models/xlit/checkpoints/xlit.pt \
        --metrics runs/xlit_v2/xlit/metrics.json --tag v2 --repo Stardiaries/akshara-xlit

Needs `huggingface-cli login` (a write token). The model card's score table comes from the
`scripts.compare_xlit --tag vN` lines in reports/baselines.jsonl; xlit_kn.pt next to --ckpt is
uploaded too.
Download later with:  hf_hub_download("Stardiaries/akshara-xlit", "xlit.pt")
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from common.paths import REPORTS

CARD = """---
language: [kn, tcy]
license: mit
tags: [transliteration, kannada, tulu, character-level, akshara]
---

# Akshara transliteration (English letters → Kannada script) for Kannada and Tulu

A small character-level Transformer (encoder–decoder, {params}) that turns casual romanized typing
("encha ullar") into Kannada script ("ಎಂಚ ಉಲ್ಲರ್"). Tulu is written in Kannada script.

* **Training:** Kannada first (AI4Bharat Aksharantar, ~2.9M word pairs, + 434k casual spellings
  such as ooru -> ಊರು), then fine-tuned on Tulu
  word pairs made from Tulu Wikipedia words (with 20% Kannada mixed in so it does not forget).
* **Decoding:** beam search (5) + the rule engine's real-word guesses, ranked by
  model log-prob + 0.5 * log10(1 + word count). `xlit_kn.pt` (Kannada-only stage) is best for Kannada.
* **Code:** https://github.com/nikhitha-sk/akshara (`models/xlit/`).

## Scores (same words for every model; AI4Bharat IndicXlit gets 80.1% on Kannada, 68.1% on Tulu)

| Test set | Words | Top-1 | Top-5 | CER |
|---|---|---|---|---|
{rows}

**Honest note:** the Tulu test words are machine-made (Kannada script → roman by rules), so the Tulu
score is optimistic. The real target (top-1 ≥ 70% on a human-typed Tulu test set) needs volunteer data.

## Use

```python
from huggingface_hub import hf_hub_download
from models.xlit.neural import NeuralTransliterator   # from the GitHub repo
xl = NeuralTransliterator(hf_hub_download("{repo}", "xlit.pt"))
xl.transliterate("encha ullar", lang="tcy")["output"]
```
"""


def _score_rows(tag: str) -> str:
    """Model-card table from the fair comparison (reports/baselines.jsonl, scripts.compare_xlit)."""
    best: dict[str, dict] = {}
    for line in open(REPORTS / "baselines.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("tool") == "xlit" and r.get("model", "").endswith(f"(ours {tag})"):
            best[r["test_file"]] = r  # last line wins
    names = {"kn_test": "Kannada (Aksharantar test)", "tcy_test": "Tulu (machine-made test)",
             "casual_check_kn": "Everyday casual Kannada (62 words)"}  # fmt: skip
    return "\n".join(
        f"| {names.get(t, t)} | {r['n']:,} | {r['top1']:.1%} | {r['top5']:.1%} | {r['cer']:.1%} |"
        for t, r in best.items()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["xlit"])
    ap.add_argument("--ckpt", type=Path, required=True, help="Tulu (+Kannada) checkpoint xlit.pt")
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="version used in compare_xlit, e.g. v2")
    ap.add_argument("--repo", default="Stardiaries/akshara-xlit")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi

    m = json.loads(a.metrics.read_text(encoding="utf-8"))
    rows = _score_rows(a.tag)
    if not rows:
        raise SystemExit(f"no compare_xlit results for tag {a.tag!r}; run scripts.compare_xlit first")
    params = f"{m.get('config', {}).get('params_M', '?')}M parameters"
    api = HfApi()
    api.create_repo(a.repo, private=not a.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(CARD.format(rows=rows, params=params, repo=a.repo), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=a.repo)
    api.upload_file(path_or_fileobj=str(a.metrics), path_in_repo="metrics.json", repo_id=a.repo)
    api.upload_file(path_or_fileobj=str(a.ckpt), path_in_repo="xlit.pt", repo_id=a.repo,
                    commit_message=f"xlit {a.tag}")  # fmt: skip
    kn = a.ckpt.with_name("xlit_kn.pt")
    if kn.exists():
        api.upload_file(path_or_fileobj=str(kn), path_in_repo="xlit_kn.pt", repo_id=a.repo,
                        commit_message=f"xlit_kn {a.tag}")  # fmt: skip
    print(f"uploaded to https://huggingface.co/{a.repo} ({'public' if a.public else 'private'})")


if __name__ == "__main__":
    main()
