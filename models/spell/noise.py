"""Break correct Kannada/Tulu text on purpose, copying real typing mistakes.

Used to make synthetic (wrong -> right) training pairs for the spell model
(Dataset 2 in the guide). Synthetic pairs go into TRAIN only, never TEST.
"""

from __future__ import annotations

import random

from common.text import CONSONANTS, VIRAMA, split_aksharas, split_words

VOWEL_SIGN_SWAPS = {"ಿ": "ೀ", "ೀ": "ಿ", "ು": "ೂ", "ೂ": "ು", "ೆ": "ೇ", "ೇ": "ೆ", "ೊ": "ೋ", "ೋ": "ೊ"}
LETTER_SWAPS = {
    "ಲ": ["ಳ"], "ಳ": ["ಲ"], "ನ": ["ಣ"], "ಣ": ["ನ"],
    "ಶ": ["ಷ", "ಸ"], "ಷ": ["ಶ", "ಸ"], "ಸ": ["ಶ", "ಷ"],
    "ಅ": ["ಆ"], "ಆ": ["ಅ"], "ಇ": ["ಈ"], "ಈ": ["ಇ"], "ಉ": ["ಊ"], "ಊ": ["ಉ"],
    "ಎ": ["ಏ"], "ಏ": ["ಎ"], "ಒ": ["ಓ"], "ಓ": ["ಒ"],
    "ತ": ["ಟ", "ಥ"], "ಟ": ["ತ"], "ದ": ["ಡ", "ಧ"], "ಡ": ["ದ"], "ಬ": ["ಭ"], "ಭ": ["ಬ"],
}  # fmt: skip


def _vowel_sign_swap(w: str, rng: random.Random) -> str | None:
    idx = [i for i, c in enumerate(w) if c in VOWEL_SIGN_SWAPS]
    if not idx:
        return None
    i = rng.choice(idx)
    return w[:i] + VOWEL_SIGN_SWAPS[w[i]] + w[i + 1 :]


def _drop_long_vowel(w: str, rng: random.Random) -> str | None:
    idx = [i for i, c in enumerate(w) if c == "ಾ"]
    if not idx:
        return None
    i = rng.choice(idx)
    return w[:i] + w[i + 1 :]


def _letter_swap(w: str, rng: random.Random) -> str | None:
    idx = [i for i, c in enumerate(w) if c in LETTER_SWAPS]
    if not idx:
        return None
    i = rng.choice(idx)
    return w[:i] + rng.choice(LETTER_SWAPS[w[i]]) + w[i + 1 :]


def _virama_error(w: str, rng: random.Random) -> str | None:
    # drop a virama (breaks a conjunct: ತ್ತ -> ತತ) ...
    idx = [i for i, c in enumerate(w) if c == VIRAMA]
    if idx and rng.random() < 0.6:
        i = rng.choice(idx)
        return w[:i] + w[i + 1 :]
    # ... or add one after a bare consonant (ತತ -> ತ್ತ)
    idx = [
        i
        for i, c in enumerate(w[:-1])
        if c in CONSONANTS and w[i + 1] in CONSONANTS
    ]
    if not idx:
        return None
    i = rng.choice(idx)
    return w[: i + 1] + VIRAMA + w[i + 1 :]


def _akshara_typo(w: str, rng: random.Random) -> str | None:
    aks = split_aksharas(w)
    if len(aks) < 2:
        return None
    op = rng.choice(["delete", "double", "swap"])
    i = rng.randrange(len(aks) - (1 if op == "swap" else 0))
    if op == "delete":
        aks.pop(i)
    elif op == "double":
        aks.insert(i, aks[i])
    else:
        aks[i], aks[i + 1] = aks[i + 1], aks[i]
    return "".join(aks)


# (function, weight) -- realistic mistakes are more likely than random typos
NOISE_FUNCS = [
    (_vowel_sign_swap, 0.30),
    (_drop_long_vowel, 0.15),
    (_letter_swap, 0.25),
    (_virama_error, 0.15),
    (_akshara_typo, 0.15),
]


def corrupt_word(word: str, rng: random.Random) -> str:
    """Return a misspelled version of word (or the word itself if nothing applies)."""
    funcs = [f for f, _ in NOISE_FUNCS]
    weights = [w for _, w in NOISE_FUNCS]
    for _ in range(5):
        f = rng.choices(funcs, weights=weights)[0]
        out = f(word, rng)
        if out and out != word:
            return out
    return word


def corrupt_sentence(sentence: str, rng: random.Random, rate: float = 0.15) -> str:
    """Break about `rate` of the Kannada words in a sentence (at least one)."""
    words = split_words(sentence, kannada_only=True)
    if not words:
        return sentence
    n = max(1, round(len(words) * rate))
    targets = set(rng.sample(range(len(words)), min(n, len(words))))
    out, pos, wi = [], 0, 0
    for w in words:
        start = sentence.index(w, pos)
        out.append(sentence[pos:start])
        out.append(corrupt_word(w, rng) if wi in targets else w)
        pos = start + len(w)
        wi += 1
    out.append(sentence[pos:])
    return "".join(out)
