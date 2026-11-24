"""Turn a Wikipedia XML dump into plain text (one paragraph per line).

    python -m scripts.extract_wikipedia --lang tcy

Reads data/raw/wikipedia/<lang>wiki-latest-pages-articles.xml.bz2 (never modified)
and writes data/corpus/interim/<lang>_wiki_plain.txt.
"""

from __future__ import annotations

import argparse
import bz2
import re
import xml.etree.ElementTree as ET

import mwparserfromhell
from tqdm import tqdm

from common.paths import INTERIM, RAW

_DROP_SECTIONS = re.compile(r"\n==\s*(ಉಲ್ಲೇಖ|ಉಲ್ಲೇಖಗಳು|ಬಾಹ್ಯ ಸಂಪರ್ಕ|ಹೊರಗಿನ ಕೊಂಡಿಗಳು|ಇದನ್ನೂ ನೋಡಿ|References|External links).*", re.S)
_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.S)
_REF_RE = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S)
_FILE_LINK_RE = re.compile(r"\[\[(?:File|Image|ಚಿತ್ರ|ಸಂಚಿಕೆ|ಫೈಲ್|Category|ವರ್ಗ)\s*:[^\]]*(?:\[\[[^\]]*\]\][^\]]*)*\]\]", re.I)


def wikitext_to_plain(wikitext: str) -> list[str]:
    wikitext = _DROP_SECTIONS.sub("", wikitext)
    wikitext = _TABLE_RE.sub("", wikitext)
    wikitext = _REF_RE.sub("", wikitext)
    wikitext = _FILE_LINK_RE.sub("", wikitext)
    try:
        text = mwparserfromhell.parse(wikitext).strip_code(normalize=True, collapse=True)
    except Exception:
        return []
    paras = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith(("=", "*", "#", "|", "!", "{", "}")):
            continue
        paras.append(line)
    return paras


def extract(lang: str) -> None:
    src = RAW / "wikipedia" / f"{lang}wiki-latest-pages-articles.xml.bz2"
    out = INTERIM / f"{lang}_wiki_plain.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_pages = n_lines = 0
    with bz2.open(src, "rb") as f, open(out, "w", encoding="utf-8") as w:
        ns = None
        title = text = None
        page_ns = redirect = None
        for event, el in tqdm(ET.iterparse(f, events=("start", "end")), desc=lang, unit=" el", mininterval=2):
            tag = el.tag
            if ns is None and tag.startswith("{"):
                ns = tag[: tag.index("}") + 1]
            local = tag[len(ns or "") :]
            if event == "start":
                if local == "page":
                    title = text = page_ns = None
                    redirect = False
                continue
            if local == "title":
                title = el.text
            elif local == "ns":
                page_ns = el.text
            elif local == "redirect":
                redirect = True
            elif local == "text":
                text = el.text or ""
            elif local == "page":
                if page_ns == "0" and not redirect and text and title:
                    for para in wikitext_to_plain(text):
                        w.write(para + "\n")
                        n_lines += 1
                    n_pages += 1
                el.clear()
    print(f"{lang}: {n_pages} articles -> {n_lines} paragraphs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["kn", "tcy"], required=True)
    extract(ap.parse_args().lang)
