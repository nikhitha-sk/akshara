# Scores (updated Nov 2026)

Every phase must **beat its baseline**. Raw results: `reports/baselines.jsonl`
(re-run with `python -m scripts.eval_baselines ...`).

## Transliteration (Phase 1) — model v2, Nov 2026

All models scored on **exactly the same words**, text NFC-normalized and invisible marks removed
(`python -m scripts.compare_xlit`). Top-1 = first answer exactly right.

| Test set | Words | Model | Top-1 | Top-5 | CER |
|---|---|---|---|---|---|
| Kannada (Aksharantar test) | 11,380 | Rules + word list | 48.7% | 58.2% | 18.2% |
| | | AI4Bharat IndicXlit (Kannada) | **80.1%** | **89.6%** | **3.6%** |
| | | **Ours v2** (char-Transformer, 7.4M) | 76.7% | 89.5% | 4.3% |
| Tulu (machine-made test) | 1,170 | Rules + word list | 80.8% | 91.5% | 5.8% |
| | | AI4Bharat IndicXlit (Kannada) | 68.1% | 84.0% | 8.5% |
| | | **Ours v2** | **90.4%** | **99.4%** | **1.9%** |
| Everyday casual Kannada (hand-written) | 62 | Rules + word list | 88.7% | 93.5% | 4.7% |
| | | **Ours v2** | **95.2%** | 96.8% | 1.4% |

Speed: ~30–70 ms per word on a laptop CPU (no GPU needed). v1 → v2 history: Kannada 75.2% → 76.7%,
Tulu 87.8% → 90.4%, casual 91.9% → 95.2% (raw lines in `baselines.jsonl`).

**How the model is built:** Kannada first (2.9M Aksharantar pairs + 434k casual spellings such as
`ooru → ಊರು`), then fine-tuned on 15.5k Tulu pairs with 20% Kannada mixed in. Beam search (5),
plus the rule engine's real-word guesses, ranked by `model log-prob + 0.5 · log10(1 + word count)`.
Kannada is served by the Kannada-only checkpoint (it is ~1.5 points better on Kannada).

### Honest reading
* **Kannada:** IndicXlit is still ~3 points ahead on its own benchmark (it was trained on the same
  Aksharantar data with a bigger model). Our model is close and much smaller.
* **Tulu:** we beat IndicXlit by 22 points, **but this test is machine-made**: its romanizations come
  from the same rules that made our Tulu training pairs (even the plain rules beat IndicXlit here).
  So the checklist item *"beats IndicXlit on the Tulu test set"* is only provisionally done. The real
  exam is the **human-typed Tulu test set** from the volunteer app (target: top-1 ≥ 70%, top-5 ≥ 90%).
* **Casual typing:** Aksharantar almost never spells ಊ as "oo" (34 of 2.9M words), while WhatsApp users
  do. Adding casual spellings (v2) fixed `ooru` → ಊರು and `eenu` → ಏನು. The 62-word check list was written
  by the developer, so treat it as a sanity check, not a benchmark.
* **Still wrong (v2):** `kelsa` → ಕೆಲ್ಸಾ (want ಕೆಲಸ: spoken form vs written form), `yeshtu` → ಯಷ್ಟು
  (want ಎಷ್ಟು), and `oota` → ಓಟ (running) instead of ಊಟ (meal) — one word at a time cannot pick the
  meaning; that needs sentence context (Phase 2 / feedback data).

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
