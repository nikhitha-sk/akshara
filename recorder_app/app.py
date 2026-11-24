"""Akshara volunteer app: collect Tulu text and speech from native speakers.

    streamlit run recorder_app/app.py

Three tasks (each fills a gap in the guide's datasets):
  1. Translate  -> Tulu sentences (corpus) + how people type them in English letters
                   (the human-made transliteration TEST set)
  2. Record     -> Tulu speech clips for speech-to-text (Dataset 4)
  3. Type fast  -> real spelling mistakes for the spell-checker TEST set (Dataset 2)

Everything is saved under data/raw/volunteers/ (never committed to git). If the
environment variables HF_TOKEN and AKSHARA_HF_DATASET (e.g. "nikhitha-sk/akshara-volunteers",
a PRIVATE dataset repo) are set, the folder is also pushed to Hugging Face every few
minutes, so nothing is lost when the app runs on a free cloud host.
"""

from __future__ import annotations

import json
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CORPUS, VOLUNTEER  # noqa: E402
from common.text import kannada_ratio, normalize  # noqa: E402

HERE = Path(__file__).parent
CLIPS = VOLUNTEER / "clips"
CLIPS.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def _scheduler():
    token, repo = os.getenv("HF_TOKEN"), os.getenv("AKSHARA_HF_DATASET")
    if not (token and repo):
        return None
    from huggingface_hub import CommitScheduler

    return CommitScheduler(
        repo_id=repo, repo_type="dataset", folder_path=VOLUNTEER, every=5, token=token, private=True
    )


def _append(name: str, row: dict) -> None:
    row["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sched = _scheduler()
    lock = sched.lock if sched else None
    if lock:
        lock.acquire()
    try:
        with open(VOLUNTEER / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if lock:
            lock.release()


@st.cache_data
def english_prompts() -> list[str]:
    return [line.strip() for line in open(HERE / "prompts_en.txt", encoding="utf-8") if line.strip()]


@st.cache_data
def tulu_sentences() -> list[str]:
    """Short, clean Tulu sentences to read aloud / retype."""
    path = CORPUS / "tcy_sentences.txt"
    if not path.exists():
        return ["ಎಂಚ ಉಲ್ಲರ್?", "ಈರೆನ ಪುದರ್ ದಾದ?", "ಯಾನ್ ಇನಿ ಬರ್ಪೆ."]
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if 4 <= len(s.split()) <= 12 and kannada_ratio(s) > 0.97 and not any(c.isdigit() for c in s):
                out.append(s)
    random.Random(7).shuffle(out)
    return out[:5000]


# ---------------------------------------------------------------------------
st.set_page_config(page_title="Akshara · Help build Tulu AI", page_icon="ಅ", layout="centered")
st.title("ಅಕ್ಷರ · Akshara")
st.caption("Help build free AI tools for Tulu: typing, spell-check and voice typing.")

ss = st.session_state
ss.setdefault("speaker", None)
ss.setdefault("count", 0)

# --- Step 1: consent --------------------------------------------------------
if ss.speaker is None:
    st.subheader("Before you start")
    st.markdown(
        "Your typed sentences and voice recordings will be used to train open AI models for "
        "**Kannada and Tulu**. Your name is **never** shared. You can ask for your data to be "
        "deleted at any time using the contributor ID shown after you join."
    )
    with st.form("consent"):
        c1 = st.checkbox("My recordings / typed sentences may be used to train AI models for Kannada and Tulu.")
        c2 = st.checkbox("They may be shared publicly in an open dataset (my name will NOT be shared).")
        age = st.selectbox("Age group", ["", "under 18", "18-25", "26-40", "41-60", "60+"])
        gender = st.selectbox("Gender (optional)", ["prefer not to say", "female", "male", "other"])
        town = st.text_input("Town / area where you learned Tulu (e.g. Udupi, Mangaluru, Puttur)")
        st.caption("Under 18? Please ask a parent or guardian before you take part.")
        if st.form_submit_button("I agree, let's start", type="primary"):
            if not c1:
                st.error("Please tick the first box to continue.")
            elif not age:
                st.error("Please choose your age group.")
            else:
                ss.speaker = "spk_" + uuid.uuid4().hex[:8]
                _append(
                    "participants.jsonl",
                    {"speaker": ss.speaker, "consent_train": c1, "consent_public": c2,
                     "age": age, "gender": gender, "town": town.strip()},
                )  # fmt: skip
                st.rerun()
    st.stop()

st.success(f"Your contributor ID: **{ss.speaker}** · contributions this session: **{ss.count}**")
task = st.radio(
    "Choose a task",
    ["1 · Translate to Tulu", "2 · Read aloud (record)", "3 · Type fast"],
    horizontal=True,
)


def _next(key: str, pool: list[str]) -> str:
    if key not in ss:
        ss[key] = random.choice(pool)
    return ss[key]


# --- Task 1: translate ------------------------------------------------------
if task.startswith("1"):
    prompt = _next("en_prompt", english_prompts())
    st.markdown(f"### “{prompt}”")
    st.write("How would **you** say this in Tulu?")
    with st.form("translate", clear_on_submit=True):
        tulu = st.text_input("Tulu in Kannada script (ಕನ್ನಡ ಲಿಪಿ)", placeholder="ಎಂಚ ಉಲ್ಲರ್?")
        roman = st.text_input("Same sentence the way you type it on WhatsApp (English letters)", placeholder="encha ullar?")
        a, b = st.columns(2)
        save = a.form_submit_button("Save & next", type="primary")
        skip = b.form_submit_button("Skip")
    if save:
        if not tulu.strip() or kannada_ratio(tulu) < 0.5:
            st.warning("Please type the Tulu sentence in Kannada script.")
        else:
            _append("typed.jsonl", {"speaker": ss.speaker, "prompt_en": prompt, "tulu": normalize(tulu), "roman": roman.strip()})
            ss.count += 1
            del ss["en_prompt"]
            st.rerun()
    if skip:
        del ss["en_prompt"]
        st.rerun()

# --- Task 2: record ---------------------------------------------------------
elif task.startswith("2"):
    sent = _next("read_sent", tulu_sentences())
    st.markdown(f"### {sent}")
    st.caption("Read the sentence naturally, in your own accent. A quiet room works best.")
    audio = st.audio_input("Press the mic, read, then press stop", key=f"rec_{ss.count}")
    a, b = st.columns(2)
    if a.button("Save & next", type="primary", disabled=audio is None):
        clip = CLIPS / f"tcy_{ss.speaker}_{uuid.uuid4().hex[:6]}.wav"
        sched = _scheduler()
        if sched:
            with sched.lock:
                clip.write_bytes(audio.getvalue())
        else:
            clip.write_bytes(audio.getvalue())
        _append("recordings.jsonl", {"speaker": ss.speaker, "audio": f"clips/{clip.name}", "text": sent})
        ss.count += 1
        del ss["read_sent"]
        st.rerun()
    if b.button("Skip (hard to read)"):
        del ss["read_sent"]
        st.rerun()

# --- Task 3: type fast ------------------------------------------------------
else:
    sent = _next("fast_sent", tulu_sentences())
    st.markdown(f"### {sent}")
    st.write("Type this sentence **quickly**, the way you normally would. **Don't fix your mistakes!** "
             "Your natural typos teach the spell-checker what real mistakes look like.")
    with st.form("fast", clear_on_submit=True):
        typed = st.text_input("Type here (Kannada script)")
        keyboard = st.selectbox("What are you typing on?", ["phone", "laptop / computer"])
        a, b = st.columns(2)
        save = a.form_submit_button("Save & next", type="primary")
        skip = b.form_submit_button("Skip")
    if save and typed.strip():
        _append("typed_fast.jsonl", {"speaker": ss.speaker, "right": sent, "typed": typed, "device": keyboard})
        ss.count += 1
        del ss["fast_sent"]
        st.rerun()
    if skip:
        del ss["fast_sent"]
        st.rerun()

st.divider()
st.caption("Thank you for helping Tulu go digital. ಸೊಲ್ಮೆಲು! 🙏  ·  To delete your data, send your contributor ID to the project owner.")

