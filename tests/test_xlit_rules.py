from models.xlit.rules import RuleTransliterator, kannada_to_roman

WORDS = {"ನಾನು": 500, "ಶಾಲೆಗೆ": 80, "ಹೋಗುತ್ತೇನೆ": 40, "ನಮಸ್ಕಾರ": 300, "ಉಲ್ಲರ್": 20, "ಎಂಚ": 30}


def test_kannada_to_roman():
    assert kannada_to_roman("ನಮಸ್ಕಾರ")[0] == "namaskaara"
    assert "namaskara" in kannada_to_roman("ನಮಸ್ಕಾರ")
    assert kannada_to_roman("ಉಲ್ಲರ್")[0] == "ullar"
    assert kannada_to_roman("ಎಂಚ")[0] == "encha"
    assert kannada_to_roman("ಶಾಲೆಗೆ")[0] == "shaalege"


def test_roman_to_kannada_without_dictionary():
    xl = RuleTransliterator()
    assert xl.word_candidates("ullar")[0] == "ಉಲ್ಲರ್"
    assert "ನಾನು" in xl.word_candidates("naanu")
    assert "ನಮಸ್ಕಾರ" in xl.word_candidates("namaskaara")


def test_roman_to_kannada_with_dictionary_rerank():
    xl = RuleTransliterator(WORDS)
    res = xl.transliterate("naanu shaalege hoguttene")
    assert res["output"] == "ನಾನು ಶಾಲೆಗೆ ಹೋಗುತ್ತೇನೆ"
    assert xl.word_candidates("namaskara")[0] == "ನಮಸ್ಕಾರ"
    assert xl.transliterate("encha ullar?")["output"] == "ಎಂಚ ಉಲ್ಲರ್?"
