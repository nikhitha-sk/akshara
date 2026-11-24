"""Akshara API: one FastAPI app holding all four tools (Section 06 / 10 of the guide).

    uvicorn api.main:app --reload
    open http://127.0.0.1:8000  (demo page)   ·   http://127.0.0.1:8000/docs  (API docs)

/xlit uses the trained neural model when a checkpoint is found (AKSHARA_XLIT_CKPT or
models/xlit/checkpoints/xlit.pt), otherwise the rule baseline. /spell runs the dictionary
baseline. /ocr and /stt arrive in Phases 3 and 4.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from api import db
from common.paths import CORPUS
from common.text import normalize
from models.spell.checker import SpellChecker, load_wordfreq
from models.xlit.rules import RuleTransliterator

XLIT_CKPT = Path(os.getenv("AKSHARA_XLIT_CKPT", Path(__file__).parents[1] / "models/xlit/checkpoints/xlit.pt"))

Lang = Literal["kn", "tcy"]
VERSIONS = {"xlit": "rules-v0.1", "spell": "symspell-akshara-v0.1"}
RATE_LIMIT = 30  # requests per minute per IP
STATIC = Path(__file__).parent / "static"

state: dict = {}


def _load_models() -> None:
    state["xlit"], state["spell"], state["words"] = {}, {}, {}
    freqs = {}
    for lang in ("kn", "tcy"):
        path = CORPUS / f"{lang}_wordfreq.tsv"
        freqs[lang] = load_wordfreq(path) if path.exists() else {}
        state["words"][lang] = len(freqs[lang])
        state["xlit"][lang] = RuleTransliterator(freqs[lang])
        state["spell"][lang] = SpellChecker(freqs[lang])
    state["xlit_neural"] = None
    if XLIT_CKPT.exists():
        from models.xlit.neural import NeuralTransliterator  # needs torch

        state["xlit_neural"] = NeuralTransliterator(XLIT_CKPT, freqs)
        VERSIONS["xlit"] = f"char-transformer ({XLIT_CKPT.name})"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()  # load once at start-up, then answer requests fast
    state["db"] = db.connect()
    yield
    state["db"].close()


app = FastAPI(title="Akshara API", version="0.1.0", lifespan=lifespan)
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.method == "POST":
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return JSONResponse({"detail": "Too many requests, slow down."}, status_code=429)
        q.append(now)
    return await call_next(request)


class XlitIn(BaseModel):
    text: str = Field(..., max_length=2000, examples=["naanu shaalege hoguttene"])
    lang: Lang = "kn"
    k: int = Field(3, ge=1, le=10)
    engine: Literal["auto", "neural", "rules"] = "auto"


class SpellIn(BaseModel):
    text: str = Field(..., max_length=5000, examples=["ನಮಸ್ಕರ, ನಾನು ಬೆಂಗಲೂರು"])
    lang: Lang = "kn"


class FeedbackIn(BaseModel):
    tool: Literal["xlit", "spell", "ocr", "stt"]
    input_ref: str | None = None
    model_output: str = Field(..., max_length=10000)
    user_fix: str = Field(..., max_length=10000)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "models": VERSIONS, "dictionary_words": state.get("words", {})}


@app.post("/xlit")
def xlit(body: XlitIn):
    neural = state["xlit_neural"]
    if body.engine == "neural" and neural is None:
        raise HTTPException(503, "No trained transliteration model loaded.")
    if neural is not None and body.engine != "rules":
        res, version = neural.transliterate(body.text, lang=body.lang, k=body.k), VERSIONS["xlit"]
    else:
        res, version = state["xlit"][body.lang].transliterate(body.text, k=body.k), "rules-v0.1"
    res["engine"] = version
    res["request_id"] = db.log_request(state["db"], "xlit", body.lang, body.text, res, version)
    return res


@app.post("/spell")
def spell(body: SpellIn):
    text = normalize(body.text, strip_zero_width=False)
    checker = state["spell"][body.lang]
    res = {"text": text, "errors": checker.check(text), "corrected": checker.correct(text)}
    res["request_id"] = db.log_request(state["db"], "spell", body.lang, text, res, VERSIONS["spell"])
    return res


@app.post("/feedback")
def feedback(body: FeedbackIn):
    """Save a (wrong output, right output) pair. These become free training data."""
    if normalize(body.model_output) == normalize(body.user_fix):
        return {"ok": True, "saved": False}
    db.add_feedback(state["db"], body.tool, body.input_ref, body.model_output, normalize(body.user_fix))
    return {"ok": True, "saved": True}


@app.get("/history")
def get_history(limit: int = 10):
    return db.history(state["db"], min(limit, 50))


@app.post("/ocr")
def ocr():
    raise HTTPException(501, "OCR arrives in Phase 3 (weeks 8-10).")


@app.post("/stt")
def stt():
    raise HTTPException(501, "Speech-to-text arrives in Phase 4 (weeks 11-14).")
