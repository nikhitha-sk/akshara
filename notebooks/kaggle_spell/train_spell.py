"""Kaggle GPU job: train the Akshara context spell model (Phase 2).

Push from the repo root:
    cd notebooks/kaggle_spell && kaggle kernels push -p . --accelerator NvidiaTeslaT4 -t 9000
Get results:
    kaggle kernels output nikhithakotian/akshara-spell-train -p runs/spell_v1
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/nikhitha-sk/akshara"
CODE = "/tmp/akshara"

subprocess.run(["git", "clone", "--depth", "1", REPO, CODE], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "symspellpy"], check=True)
hits = glob.glob("/kaggle/input/**/kn_train.tsv", recursive=True)
hits = [h for h in hits if "spell" in h] or hits
assert hits, "dataset akshara-spell-data is not attached"
data = os.path.dirname(hits[0])
print("data:", data, os.listdir(data), flush=True)
subprocess.run(["nvidia-smi"], check=False)

cmd = [
    sys.executable, "-m", "models.spell.train",
    "--data", data,
    "--out", "/kaggle/working/spell",
    "--epochs", "4",
    "--hours", "1.4",  # free GPU quota is only 6 h/week; evaluation adds ~10 min
]  # fmt: skip
subprocess.run(cmd, check=True, cwd=CODE)
