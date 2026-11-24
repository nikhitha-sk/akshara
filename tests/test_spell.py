import random

from models.spell.checker import SpellChecker
from models.spell.distance import akshara_distance
from models.spell.noise import corrupt_sentence, corrupt_word

WORDS = {"ನಮಸ್ಕಾರ": 300, "ನಮಸ್ಕಾರಗಳು": 50, "ಬೆಂಗಳೂರು": 200, "ನಾನು": 500, "ಶಾಲೆಗೆ": 80}


def test_akshara_distance_prefers_realistic_mistakes():
    # missing long vowel sign is cheaper than a random wrong letter
    assert akshara_distance("ನಮಸ್ಕರ", "ನಮಸ್ಕಾರ") < 0.5
    assert akshara_distance("ಬೆಂಗಲೂರು", "ಬೆಂಗಳೂರು") < 0.5
    assert akshara_distance("ನಮಕ್ಕಾರ", "ನಮಸ್ಕಾರ") >= 0.9


def test_checker_finds_and_fixes():
    sc = SpellChecker(WORDS)
    errs = sc.check("ನಮಸ್ಕರ, ನಾನು ಬೆಂಗಲೂರು")
    assert [e["word"] for e in errs] == ["ನಮಸ್ಕರ", "ಬೆಂಗಲೂರು"]
    assert errs[0]["suggestions"][0] == "ನಮಸ್ಕಾರ"
    assert errs[0]["start"] == 0 and errs[0]["end"] == len("ನಮಸ್ಕರ")
    assert sc.correct("ನಮಸ್ಕರ, ನಾನು ಬೆಂಗಲೂರು") == "ನಮಸ್ಕಾರ, ನಾನು ಬೆಂಗಳೂರು"


def test_noise_changes_text():
    rng = random.Random(0)
    assert corrupt_word("ನಮಸ್ಕಾರ", rng) != "ನಮಸ್ಕಾರ"
    s = "ನಾನು ಶಾಲೆಗೆ ಹೋಗುತ್ತೇನೆ"
    noisy = corrupt_sentence(s, rng, rate=0.15)
    assert noisy != s
    assert len(noisy.split()) == len(s.split())
