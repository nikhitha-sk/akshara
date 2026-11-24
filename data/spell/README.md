# Dataset card — akshara-spell v0

| Field | Value |
|---|---|
| What | (wrong sentence → right sentence) pairs for spell correction |
| Format | JSONL rows `{"lang", "wrong", "right"}` |
| Train | `<lang>_synthetic.jsonl`: corpus sentences with ~15% of words broken on purpose (vowel-sign swaps, ಲ/ಳ ನ/ಣ ಶ/ಷ/ಸ swaps, missing ಾ / ್, akshara typos) |
| Test | TODO: real mistakes from volunteers typing fast (volunteer app task 3), corrected by hand |
| Dictionary | `data/corpus/<lang>_wordfreq.tsv` |
| License | CC BY-SA 4.0 (derived from Wikipedia) |
| Known gaps | synthetic noise ≠ real typing habits; source corpus itself contains some typos |
| Made by | Nikhitha S Kotian, Nov 2026 |

Rebuild: `python -m scripts.make_synthetic_spell --lang tcy --n 50000`.
