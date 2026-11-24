# Dataset card — akshara-xlit v0

| Field | Value |
|---|---|
| What | romanized (English letters) ↔ Kannada-script word pairs, Kannada and Tulu |
| Format | JSONL rows `{"lang", "roman", "native", "source", "needs_review"?}` |
| Kannada | AI4Bharat Aksharantar (`kn_train/valid/test.jsonl`) — CC BY 4.0 |
| Tulu (train) | `tcy_reverse_pairs.jsonl`: top 8,000 Tulu Wikipedia words → rule-based romanization, 2–3 spellings each. **Machine-made, needs human review.** |
| Tulu (test) | TODO: 500 pairs typed by people (volunteer app). Never script-made. |
| License | Kannada: CC BY 4.0 (Aksharantar). Tulu: CC BY-SA 4.0 (derived from Wikipedia). |
| Known gaps | reverse-trick spellings follow one convention; real users vary more. Wikipedia Tulu is formal, not chat Tulu. |
| Made by | Nikhitha S Kotian, Nov 2026 |

Rebuild: `python -m scripts.download_aksharantar` and `python -m scripts.make_tulu_xlit_pairs`.
