"""Shared text helpers for Kannada and Tulu (Tulu is written in Kannada script).

Used by every tool: normalize(), split_words(), split_aksharas(), tokenize().
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Kannada script character classes (Unicode block U+0C80-U+0CFF)
# ---------------------------------------------------------------------------
VIRAMA = "್"  # ್  (halant)
ZWJ = "‍"
ZWNJ = "‌"
NUKTA = "಼"

INDEPENDENT_VOWELS = set(
    "ಅಆಇಈಉಊಋಌಎಏಐಒಓಔೠೡ"
)
CONSONANTS = {chr(c) for c in range(0x0C95, 0x0CBA)} | {"ೝ", "ೞ"}
CONSONANTS -= {"಩", "಴"}  # unassigned code points
VOWEL_SIGNS = {chr(c) for c in range(0x0CBE, 0x0CCD)} | {"ೕ", "ೖ", "ೢ", "ೣ"}
VOWEL_SIGNS -= {"೅", "೉"}  # unassigned
MODIFIERS = {"ಀ", "ಁ", "ಂ", "ಃ", "ೳ"}  # candrabindu, anusvara ಂ, visarga ಃ
DIGITS = {chr(c) for c in range(0x0CE6, 0x0CF0)}

# Marks that never start a new akshara: they attach to the one before.
COMBINING = VOWEL_SIGNS | MODIFIERS | {VIRAMA, NUKTA, ZWJ, ZWNJ}

KANNADA_RANGE = "ಀ-೿"
_KN_LETTER = f"[{KANNADA_RANGE}{ZWJ}{ZWNJ}]"
_WORD_RE = re.compile(rf"(?:{_KN_LETTER}|[A-Za-z0-9])+(?:['\-](?:{_KN_LETTER}|[A-Za-z0-9])+)*")
_KN_WORD_RE = re.compile(rf"{_KN_LETTER}+")
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿­]")
_SPACE_RE = re.compile(r"[ \t  -   　]+")


def is_kannada_char(ch: str) -> bool:
    return "ಀ" <= ch <= "೿"


def normalize(text: str, strip_zero_width: bool = True) -> str:
    """NFC-normalize text and remove invisible characters.

    The same Kannada letter can be stored in more than one way (e.g. ೀ can be
    one code point or ಿ + ೕ). NFC makes them identical. Zero-width joiners are
    removed by default because they make identical-looking words compare unequal.
    """
    text = unicodedata.normalize("NFC", text)
    if strip_zero_width:
        text = _ZERO_WIDTH_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def split_words(text: str, kannada_only: bool = False) -> list[str]:
    """Split text into words (keeps Kannada combining marks inside the word)."""
    pattern = _KN_WORD_RE if kannada_only else _WORD_RE
    return pattern.findall(text)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    is_word: bool


def tokenize(text: str) -> list[Token]:
    """Split text into word and non-word tokens, keeping character offsets.

    Joining all token texts gives back the original string, which is what the
    spell-checker needs to underline words in place.
    """
    tokens: list[Token] = []
    pos = 0
    for m in _WORD_RE.finditer(text):
        if m.start() > pos:
            tokens.append(Token(text[pos : m.start()], pos, m.start(), False))
        tokens.append(Token(m.group(), m.start(), m.end(), True))
        pos = m.end()
    if pos < len(text):
        tokens.append(Token(text[pos:], pos, len(text), False))
    return tokens


def split_aksharas(word: str) -> list[str]:
    """Split Kannada text into aksharas (letter-blocks), e.g. ತ್ತೇ, ಕ್ಷ, ಮಾ.

    An akshara is: (consonant (+ virama + consonant)*  |  independent vowel)
    followed by any vowel signs / anusvara / visarga. A consonant that follows a
    virama joins the previous akshara (that is how conjuncts / ottakshara work).
    """
    aksharas: list[str] = []
    cur = ""
    for ch in word:
        if not cur:
            cur = ch
            continue
        prev = cur[-1]
        joins_conjunct = ch in CONSONANTS and (
            prev == VIRAMA or (prev in (ZWJ, ZWNJ) and len(cur) > 1 and cur[-2] == VIRAMA)
        )
        if ch in COMBINING or joins_conjunct:
            cur += ch
        else:
            aksharas.append(cur)
            cur = ch
    if cur:
        aksharas.append(cur)
    return aksharas


def kannada_ratio(text: str) -> float:
    """Share of letters (ignoring spaces, digits and punctuation) that are Kannada script."""
    letters = [c for c in text if c.isalpha() or unicodedata.category(c).startswith("M")]
    if not letters:
        return 0.0
    return sum(is_kannada_char(c) for c in letters) / len(letters)


_SENT_END_RE = re.compile(r"(?<=[.!?।॥])\s+")


def split_sentences(text: str) -> list[str]:
    """Very simple sentence splitter: split after . ! ? । ॥ followed by space."""
    return [s.strip() for s in _SENT_END_RE.split(text) if s.strip()]
