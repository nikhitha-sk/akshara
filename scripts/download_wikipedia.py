"""Download the latest Kannada / Tulu Wikipedia dump into data/raw/wikipedia/.

    python -m scripts.download_wikipedia --lang tcy
    python -m scripts.download_wikipedia --lang kn

License: Wikipedia text is CC BY-SA. Give credit when you publish anything built from it.
"""

from __future__ import annotations

import argparse

import requests
from tqdm import tqdm

from common.paths import RAW

URL = "https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"
HEADERS = {"User-Agent": "akshara-corpus-builder/0.1 (research project)"}


def download(lang: str) -> None:
    url = URL.format(lang=lang)
    out = RAW / "wikipedia" / f"{lang}wiki-latest-pages-articles.xml.bz2"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = out.stat().st_size if out.exists() else 0
    head = requests.head(url, headers=HEADERS, timeout=30, allow_redirects=True)
    total = int(head.headers.get("content-length", 0))
    if total and done == total:
        print(f"already downloaded: {out}")
        return
    headers = dict(HEADERS)
    if done:
        headers["Range"] = f"bytes={done}-"
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if done and r.status_code == 206 else "wb"
        with open(out, mode) as f, tqdm(total=total, initial=done if mode == "ab" else 0, unit="B", unit_scale=True, desc=lang) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["kn", "tcy"], required=True)
    download(ap.parse_args().lang)
