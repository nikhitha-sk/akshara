from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CORPUS = DATA / "corpus"
INTERIM = CORPUS / "interim"
XLIT = DATA / "xlit"
SPELL = DATA / "spell"
REPORTS = ROOT / "reports"
VOLUNTEER = RAW / "volunteers"

LANGS = ("kn", "tcy")  # Kannada, Tulu (ISO 639-3 code for Tulu is "tcy")
