# Dataset card — akshara-corpus v0

| Field | Value |
|---|---|
| What | clean sentences (one per line) + word-frequency lists, Kannada and Tulu |
| Files | `kn_sentences.txt`, `tcy_sentences.txt`, `kn_wordfreq.tsv`, `tcy_wordfreq.tsv` |
| Sources | Kannada & Tulu Wikipedia dumps (Nov 2026). Planned: IndicCorp/Sangraha slice (kn), FLORES Tulu, volunteer sentences (tcy) |
| Cleaning | NFC, zero-width chars removed, URLs removed, sentence split, ≥85% Kannada-script letters, 3–60 words, de-duplicated |
| Word list | Tulu: words seen ≥2 times. Kannada: ≥3 times |
| License | CC BY-SA 4.0 (Wikipedia) — credit Wikipedia contributors |
| Known gaps | Wikipedia is formal/encyclopedic; Tulu Wikipedia includes some Kannada lines and typos |
| Made by | Nikhitha S Kotian, Nov 2026 |
