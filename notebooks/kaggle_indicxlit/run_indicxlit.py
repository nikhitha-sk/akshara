"""Kaggle CPU job: run AI4Bharat IndicXlit on our test words (the baseline to beat).

IndicXlit needs fairseq 0.12.2, which only ships wheels for Python <= 3.8, so this job builds a
small Python 3.8 env with uv. It writes predictions (top-5 per word); scoring is done locally with
    python -m scripts.score_xlit_predictions runs/indicxlit/indicxlit_*.tsv

Push:  cd notebooks/kaggle_indicxlit && kaggle kernels push -p .
"""

import glob
import os
import subprocess

ENV = "/tmp/ix"
OUT = "/kaggle/working"
sh = lambda c: subprocess.run(c, shell=True, check=True)  # noqa: E731

sh("pip install -q uv")
sh(f"uv venv -p 3.8 --seed {ENV}")
pip = f"{ENV}/bin/python -m pip install -q"
sh(f"{pip} 'pip<24.1'")  # newer pip rejects omegaconf 2.0.x metadata
sh(f"{pip} torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu")
sh(f"{pip} 'numpy<1.24' fairseq==0.12.2")
sh(f"{pip} --no-deps ai4bharat-transliteration==1.1.3")
sh(f"{pip} pydload ujson indic-nlp-library tqdm")
# urduhack (Urdu-only normalizer) drags in TensorFlow; Kannada never uses it, so stub it out.
site = subprocess.check_output(
    [f"{ENV}/bin/python", "-c", "import site; print(site.getsitepackages()[0])"], text=True
).strip()
os.makedirs(f"{site}/urduhack", exist_ok=True)
open(f"{site}/urduhack/__init__.py", "w").write("def normalize(text):\n    return text\n")

data = os.path.dirname(glob.glob("/kaggle/input/**/kn_test.tsv", recursive=True)[0])
job = r"""
import sys, time
from ai4bharat.transliteration import XlitEngine
data, out = sys.argv[1], sys.argv[2]
e = XlitEngine("kn", beam_width=10, rescore=True, src_script_type="en")
budget = time.time() + 3.5 * 3600
for name, limit in [("tcy_test", None), ("kn_test", None)]:
    rows = [l.rstrip("\n").split("\t") for l in open(f"{data}/{name}.tsv", encoding="utf-8")][:limit]
    t0 = time.time()
    with open(f"{out}/indicxlit_{name}.tsv", "w", encoding="utf-8") as f:
        for i, (roman, native) in enumerate(rows):
            if time.time() > budget:
                print("time budget reached", flush=True); break
            try:
                c = e.translit_word(roman, lang_code="kn", topk=5)
            except Exception as ex:
                print("fail", roman, ex, flush=True); c = []
            f.write("\t".join([roman, native] + list(c)) + "\n")
            if i % 500 == 0:
                print(name, i, len(rows), f"{time.time()-t0:.0f}s", roman, c[:3], flush=True)
    print(name, "done", f"{time.time()-t0:.0f}s", flush=True)
"""
open("/tmp/ix_job.py", "w").write(job)
sh(f"cd /tmp && {ENV}/bin/python /tmp/ix_job.py {data} {OUT}")
