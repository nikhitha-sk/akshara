from models.spell.neural import Suspect, _chunks


def test_chunks_cover_whole_words_within_limit():
    text = " ".join(["ಕನ್ನಡ"] * 40)  # 5 chars per word
    spans = _chunks(text, max_chars=30)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for a, b in spans:
        assert b - a <= 30
        assert text[a:b].split() == ["ಕನ್ನಡ"] * len(text[a:b].split())


def test_decide_respects_margins(monkeypatch):
    from models.spell.neural import NeuralSpellChecker

    nsc = NeuralSpellChecker.__new__(NeuralSpellChecker)  # no model needed for decide()
    nsc.margins = {"kn": {"unknown": 0.0, "known": 4.0}}
    text = "ನಮಸ್ಕರ ಗೆಳೆಯ"
    unknown = Suspect(0, 6, "ನಮಸ್ಕರ", False, [("ನಮಸ್ಕಾರ", 2.0)])
    known = Suspect(7, 12, "ಗೆಳೆಯ", True, [("ಗೆಲೆಯ", 3.0)])  # below the 'known' margin
    assert nsc.decide(text, [unknown, known], "kn") == "ನಮಸ್ಕಾರ ಗೆಳೆಯ"
    weak = Suspect(0, 6, "ನಮಸ್ಕರ", False, [("ನಮಸ್ಕಾರ", -0.5)])
    assert nsc.decide(text, [weak], "kn") == text
