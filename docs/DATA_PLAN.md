# Data plan — who does what

In low-resource AI, **the data is the project**. This file tracks every dataset, where it
comes from, and whether a script can get it or a human has to.

## ✅ Automatic (scripts download/build these — no human needed)

| Dataset | Script | Status |
|---|---|---|
| Tulu Wikipedia corpus + word list | `download_wikipedia` → `extract_wikipedia` → `clean_corpus` → `build_wordfreq` | done: 121k sentences, 74k words |
| Kannada Wikipedia corpus + word list | same, `--lang kn` | done |
| Kannada transliteration pairs (Aksharantar) | `download_aksharantar` | done |
| Tulu transliteration TRAIN pairs (reverse trick) | `make_tulu_xlit_pairs` | done: 8k words → 18k pairs (needs review) |
| Synthetic spell pairs (Tulu + Kannada) | `make_synthetic_spell` | done for Tulu |
| Synthetic OCR line images | Phase 3 (`make_synthetic_ocr`) | todo |
| Kannada speech (FLEURS, OpenSLR-79) | Phase 4 | todo — open download |

## 🔑 Needs an account / sign-up (Nikhitha does once, then scripts take over)

| What | Why | Where |
|---|---|---|
| Hugging Face account + access token (write) | store models & private datasets, volunteer app backup | huggingface.co → Settings → Access Tokens |
| Kaggle account (phone-verified) | ~30 free GPU hours/week for training | kaggle.com |
| Weights & Biases (optional) | track training runs | wandb.ai |
| Accept terms for gated datasets | IndicVoices / Kathbath / Vaani need you to click "agree" on their pages with your HF login | each dataset page on huggingface.co |

## 🙋 Only humans can make these (the valuable part)

| Dataset | Size goal | How |
|---|---|---|
| **Tulu sentences from people** | 2,000+ | volunteer app task 1 |
| **Tulu transliteration TEST set** | 500 pairs | volunteer app task 1 (the "WhatsApp spelling" box) |
| **Review of reverse-trick pairs** | ~8,000 rows, skim | open `data/xlit/tcy_review_sheet.csv`, mark wrong spellings |
| **Real spelling mistakes TEST set** | 50 sentences × 10 people | volunteer app task 3 |
| **Tulu speech clips** | 5–10 hours, 20+ speakers | volunteer app task 2 (1 hr of a volunteer ≈ 300–400 clips) |
| **Real OCR test lines** | 500–1,000 lines | photograph old Kannada/Tulu book pages, type the text (Label Studio) |

## Golden rules
1. Consent first (see `docs/CONSENT_FORM.md`). The volunteer app asks before anything is saved.
2. `data/raw/` is never edited. Cleaning writes new files.
3. Everything is NFC-normalized (`common.text.normalize`).
4. Synthetic data → TRAIN only. Real, human-made data → TEST.
5. Every dataset gets a card (`data/*/README.md`).
