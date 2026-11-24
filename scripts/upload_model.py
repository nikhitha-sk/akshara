"""Upload a trained model to the Hugging Face Hub (private by default).

    python -m scripts.upload_model xlit --ckpt models/xlit/checkpoints/xlit.pt \
        --metrics runs/xlit_kaggle/xlit/metrics.json --repo Stardiaries/akshara-xlit

Needs `huggingface-cli login` (a write token). The model card is generated from metrics.json.
Download later with:  hf_hub_download("Stardiaries/akshara-xlit", "xlit.pt")
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

CARD = """---
language: [kn, tcy]
license: mit
tags: [transliteration, kannada, tulu, character-level, akshara]
---

# Akshara transliteration (English letters → Kannada script) for Kannada and Tulu

A small character-level Transformer (encoder–decoder, {params}) that turns casual romanized typing
("encha ullar") into Kannada script ("ಎಂಚ ಉಲ್ಲರ್"). Tulu is written in Kannada script.

* **Training:** Kannada first (AI4Bharat Aksharantar, ~2.9M word pairs), then fine-tuned on Tulu
  word pairs made from Tulu Wikipedia words (with 20% Kannada mixed in so it does not forget).
* **Decoding:** beam search (5), then answers found in a word list are moved up.
* **Code:** https://github.com/nikhitha-sk/akshara (`models/xlit/`).

## Scores (beam + word list)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["xlit"])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--repo", default="Stardiaries/akshara-xlit")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi

    m = json.loads(a.metrics.read_text(encoding="utf-8"))
    final = m.get("final", {})
    rows = "\n".join(
        f"| {name} | {r['n']:,} | {r['beam+wordlist']['top1']:.1%} | {r['beam+wordlist']['top5']:.1%} "
        f"| {r['beam+wordlist']['cer']:.1%} |"
        for name, r in final.items()
    )
    params = f"{m.get('config', {}).get('params_M', '?')}M parameters"
    api = HfApi()
    api.create_repo(a.repo, private=not a.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(CARD.format(rows=rows, params=params, repo=a.repo), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=a.repo)
    api.upload_file(path_or_fileobj=str(a.metrics), path_in_repo="metrics.json", repo_id=a.repo)
    api.upload_file(path_or_fileobj=str(a.ckpt), path_in_repo="xlit.pt", repo_id=a.repo)
    print(f"uploaded to https://huggingface.co/{a.repo} ({'public' if a.public else 'private'})")


if __name__ == "__main__":
    main()
