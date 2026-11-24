"""Download AI4Bharat Aksharantar Kannada transliteration pairs (Dataset 1, Kannada part).

    python -m scripts.download_aksharantar

Writes data/xlit/kn_{train,valid,test}.jsonl with rows:
    {"lang": "kn", "roman": "namaskara", "native": "ನಮಸ್ಕಾರ", "source": "aksharantar"}

License: Aksharantar is CC BY 4.0 (check the dataset page before publishing).
"""

from __future__ import annotations

import json
import zipfile

from huggingface_hub import hf_hub_download

from common.paths import RAW, XLIT
from common.text import normalize


def main() -> None:
    zpath = hf_hub_download(
        "ai4bharat/Aksharantar", "kan.zip", repo_type="dataset", local_dir=RAW / "aksharantar"
    )
    XLIT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            split = next((s for s in ("train", "valid", "test") if s in name), None)
            if split is None:
                continue
            out = XLIT / f"kn_{split}.jsonl"
            n = 0
            with z.open(name) as f, open(out, "w", encoding="utf-8") as w:
                for line in f:
                    row = json.loads(line)
                    roman = row.get("english word") or row.get("english_word")
                    native = row.get("native word") or row.get("native_word")
                    if not roman or not native:
                        continue
                    rec = {"lang": "kn", "roman": roman.strip().lower(), "native": normalize(native), "source": "aksharantar"}
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            print(f"{name} -> {out} ({n} pairs)")


if __name__ == "__main__":
    main()
