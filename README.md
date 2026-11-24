# ಅಕ್ಷರ Akshara — AI Language Toolkit for Kannada & Tulu

One web app with four AI tools for languages big tech mostly ignores:

| Tool | What it does | Status |
|---|---|---|
| **Transliteration** | `naanu shaalege hoguttene` → `ನಾನು ಶಾಲೆಗೆ ಹೋಗುತ್ತೇನೆ` | ✅ char-Transformer v2 (Kannada 76.7% · Tulu 90.4% top-1) · rules fallback |
| **Spell-checker** | `ನಮಸ್ಕರ` → `ನಮಸ್ಕಾರ` | ✅ SymSpell + Kannada-aware distance baseline · 🔜 ByT5 context model |
| **OCR** | photo of a printed page → editable text | 🔜 Phase 3 (synthetic images + CRNN-CTC) |
| **Speech-to-text** | spoken Tulu → `ಎಂಚ ಉಲ್ಲರ್?` | 🔜 Phase 4 (Whisper fine-tuned on Tulu) |

Tulu (~2.5M speakers) is written in Kannada script, so everything built for Kannada is reused for Tulu.
The full job is done here: **collect data → clean → label → train → test → ship**.

## Quick start

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# Phase 0: build the shared corpus (Tulu takes ~1 min, Kannada ~5 min)
python -m scripts.download_wikipedia --lang tcy
python -m scripts.extract_wikipedia  --lang tcy
python -m scripts.clean_corpus       --lang tcy --inputs data/corpus/interim/tcy_wiki_plain.txt
python -m scripts.build_wordfreq     --lang tcy --min-count 2
# (repeat with --lang kn and --min-count 3)

# Datasets
python -m scripts.download_aksharantar                 # Kannada xlit pairs
python -m scripts.make_tulu_xlit_pairs --top 8000      # Tulu xlit pairs (reverse trick)
python -m scripts.make_synthetic_spell --lang tcy      # synthetic typo pairs

# Run
uvicorn api.main:app --reload              # demo page at http://127.0.0.1:8000, docs at /docs
streamlit run recorder_app/app.py          # volunteer app (collect Tulu text + speech)
pytest                                     # tests
```

## Data so far

| Dataset | Size | Source |
|---|---|---|
| Kannada corpus | 1.6M sentences · 360k-word dictionary | Kannada Wikipedia (CC BY-SA) |
| Tulu corpus | 122k sentences · 75k-word dictionary | Tulu Wikipedia (CC BY-SA) |
| Kannada xlit pairs | Aksharantar | AI4Bharat (CC BY 4.0) |
| Tulu xlit pairs (train) | 18k pairs from 8k words | reverse trick, needs review |
| Tulu spell pairs (train) | 49k synthetic | error-maker script |

See [`docs/DATA_PLAN.md`](docs/DATA_PLAN.md) for what is automatic and what needs volunteers.

## Scores

| Transliteration (top-1) | Rules | AI4Bharat IndicXlit | **Ours v2** |
|---|---|---|---|
| Kannada, 11,380 words | 48.7% | **80.1%** | 76.7% |
| Tulu, 1,170 words (machine-made test) | 80.8% | 68.1% | **90.4%** |

The Tulu test is machine-made, so the real check is a human-typed Tulu test (volunteers).
Details, spell-checker scores and error analysis: [`reports/BASELINES.md`](reports/BASELINES.md).

## Training the transliteration model

```bash
python -m scripts.prepare_xlit_data                     # pack data/xlit/kaggle/
cd data/xlit/kaggle && kaggle datasets version -p . -m "update"
cd notebooks/kaggle_xlit && kaggle kernels push -p . --accelerator NvidiaTeslaT4   # ~1.5 h on a T4
kaggle kernels output nikhithakotian/akshara-xlit-train -p runs/xlit
cp runs/xlit/xlit/xlit*.pt models/xlit/checkpoints/     # the API picks them up on restart
python -m scripts.compare_xlit --tag vN                 # score vs rules and IndicXlit
```

## Repo layout

```
common/        normalize, split_words, split_aksharas, metrics (CER/WER/top-k/P-R-F0.5)
data/          raw/ (never edited) · corpus/ · xlit/ spell/ ocr/ stt/ + dataset cards
scripts/       download_*, extract_*, clean_*, build_*, make_synthetic_*, eval_baselines
recorder_app/  Streamlit app volunteers use to type / record Tulu
models/        xlit/ (rules, char-Transformer, train.py) · spell/ · ocr/ · stt/
api/           FastAPI: /xlit /spell /ocr /stt /feedback /health /history
notebooks/     Colab / Kaggle training notebooks
reports/       evaluation results
docs/          data plan, consent form, labeling style guide
```

## The data flywheel

Every "Was this right? → No, fix it" click saves a *(model output, user fix)* pair in the
`feedback` table. Those pairs become new training data for the next model version.

## Roadmap

| Phase | Weeks | Deliverable |
|---|---|---|
| 0 · Setup & corpus | 1–2 | ✅ repo, corpus, word lists, text helpers, volunteer app |
| 1 · Transliteration | 3–5 | ✅ char model v2 (Kannada → Tulu transfer), IndicXlit compared · 🔜 human Tulu test set |
| 2 · Spell-checker | 6–7 | ✅ baseline · 🔜 ByT5, real-mistake test set |
| 3 · OCR | 8–10 | synthetic line images, CRNN-CTC, beat Tesseract `kan` |
| 4 · Speech-to-text | 11–14 | Tulu clips, Whisper fine-tune, WER-vs-hours chart |
| 5 · App & launch | 15–16 | Next.js frontend, Postgres, Docker, HF Spaces, eval report |

---
Built by **Nikhitha S Kotian**. Wikipedia-derived data is CC BY-SA; credit to Wikipedia contributors and AI4Bharat.
