# Baseline scores (v0.1 — Nov 2026)

Every phase must **beat its baseline**. Raw results: `reports/baselines.jsonl`
(re-run with `python -m scripts.eval_baselines ...`).

## Transliteration (Kannada, Aksharantar test, 2,000 words)

| Model | Top-1 | Top-5 | CER | ms/word |
|---|---|---|---|---|
| Rules only | 11.8% | 32.6% | 23.8% | 1.5 |
| Rules + Kannada word list re-rank | **39.0%** | **49.8%** | **18.0%** | 1.6 |
| AI4Bharat IndicXlit | todo | todo | todo | |
| Char-Transformer / ByT5 (ours) | Phase 1 | | | |

Target for the trained model: Tulu top-1 ≥ 70%, top-5 ≥ 90% on the **human-made** Tulu test set
(not built yet — needs volunteers). Scoring Tulu on the reverse-trick pairs would be meaningless,
because those pairs were made by the same rules.

## Spell-checker (synthetic test, 1,000 sentences, ~15% words broken)

| Lang | Model | Precision | Recall | F0.5 | ms/sentence |
|---|---|---|---|---|---|
| Tulu | SymSpell + akshara-aware distance | 72.9% | 50.9% | 0.671 | 1.8 |
| Kannada | SymSpell + akshara-aware distance | 77.6% | 42.5% | 0.666 | 1.4 |
| — | ByT5-small (ours) | Phase 2 | | | |

Target: precision ≥ 85% on **real** mistakes (volunteer "type fast" task).

### Why the baselines miss (first error analysis)
* **Transliteration:** casual romanization is very ambiguous (t → ತ/ಟ, e → ೆ/ೇ, l → ಲ/ಳ). The word list
  fixes many cases, but rare words and inflected forms not in Wikipedia fall back to raw rules.
* **Spell:** a broken word that becomes *another real word* (e.g. ಲ→ಳ giving a valid word) is invisible to a
  dictionary — that is exactly what the context model (ByT5) is for. The source corpus also has typos,
  which become "known words".
* **Formal vs. spoken:** the dictionary comes from Wikipedia, so correct colloquial forms get "fixed"
  wrongly — e.g. spoken Kannada ಇದ್ದೀನಿ → ಇದ್ದೀತು. Volunteer sentences and the feedback flywheel are the fix.
