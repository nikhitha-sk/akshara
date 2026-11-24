from common.metrics import cer, edit_distance, spell_prf, topk_accuracy, wer
from common.text import kannada_ratio, normalize, split_aksharas, split_words, tokenize


def test_normalize_nfc_and_zero_width():
    # ೀ written as ಿ + ೕ (decomposed) must equal the single code point ೀ
    decomposed = "ಕೀ"
    assert normalize(decomposed) == "ಕೀ"
    assert normalize("ನಮ‌ಸ್ಕಾರ") == "ನಮಸ್ಕಾರ"
    assert normalize("  ಹಲೋ    ಜಗತ್ತು ") == "ಹಲೋ ಜಗತ್ತು"


def test_split_aksharas():
    assert split_aksharas("ನಮಸ್ಕಾರ") == ["ನ", "ಮ", "ಸ್ಕಾ", "ರ"]
    assert split_aksharas("ಹೋಗುತ್ತೇನೆ") == ["ಹೋ", "ಗು", "ತ್ತೇ", "ನೆ"]
    assert split_aksharas("ಉಲ್ಲರ್") == ["ಉ", "ಲ್ಲ", "ರ್"]
    assert split_aksharas("ಕ್ಷ") == ["ಕ್ಷ"]
    assert split_aksharas("ಬೆಂಗಳೂರು") == ["ಬೆಂ", "ಗ", "ಳೂ", "ರು"]


def test_split_words_and_tokenize():
    text = "ನಾನು ಶಾಲೆಗೆ, hello 2024!"
    assert split_words(text) == ["ನಾನು", "ಶಾಲೆಗೆ", "hello", "2024"]
    assert split_words(text, kannada_only=True) == ["ನಾನು", "ಶಾಲೆಗೆ"]
    toks = tokenize(text)
    assert "".join(t.text for t in toks) == text
    assert [t.text for t in toks if t.is_word][1] == "ಶಾಲೆಗೆ"


def test_kannada_ratio():
    assert kannada_ratio("ನಮಸ್ಕಾರ") == 1.0
    assert kannada_ratio("hello") == 0.0
    assert 0.3 < kannada_ratio("ನಮಸ್ಕಾರ hello") < 0.8


def test_metrics():
    assert edit_distance("kitten", "sitting") == 3
    assert cer(["abcd"], ["abcf"]) == 0.25
    assert wer(["ನಾನು ಶಾಲೆಗೆ ಹೋಗುತ್ತೇನೆ"], ["ನಾನು ಶಾಲೆಗೆ ಹೋಗುತ್ತೆನೆ"]) == 1 / 3
    assert topk_accuracy(["a", "b"], [["x", "a"], ["b"]], k=1) == 0.5
    assert topk_accuracy(["a", "b"], [["x", "a"], ["b"]], k=2) == 1.0
    r = spell_prf(["ನಮಸ್ಕರ ಗೆಳೆಯ"], ["ನಮಸ್ಕಾರ ಗೆಳೆಯ"], ["ನಮಸ್ಕಾರ ಗೆಳೆಯ"])
    assert r["precision"] == 1.0 and r["recall"] == 1.0
