import os
import tempfile

os.environ["AKSHARA_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


def test_api_end_to_end():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/xlit", json={"text": "ullar", "lang": "tcy"}).json()
        assert r["output"] == "ಉಲ್ಲರ್"
        r = c.post("/spell", json={"text": "ನಮಸ್ಕಾರ", "lang": "kn"}).json()
        assert "errors" in r and "corrected" in r
        r = c.post("/feedback", json={"tool": "xlit", "model_output": "ಉಲ್ಲರ", "user_fix": "ಉಲ್ಲರ್"}).json()
        assert r == {"ok": True, "saved": True}
        assert len(c.get("/history").json()) == 2
        assert c.post("/ocr").status_code == 501
