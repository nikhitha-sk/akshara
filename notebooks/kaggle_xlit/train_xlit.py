"""Kaggle GPU job: train the Akshara transliteration model.

Push from the repo root:
    cd notebooks/kaggle_xlit && kaggle kernels push -p . --accelerator NvidiaTeslaT4
Get results:
    kaggle kernels status nikhithakotian/akshara-xlit-train
    kaggle kernels output nikhithakotian/akshara-xlit-train -p runs/xlit_kaggle
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/nikhitha-sk/akshara"
CODE = "/tmp/akshara"

subprocess.run(["git", "clone", "--depth", "1", REPO, CODE], check=True)
hits = glob.glob("/kaggle/input/**/kn_train.tsv", recursive=True)
assert hits, "dataset akshara-xlit-data is not attached"
data = os.path.dirname(hits[0])
print("data:", data, os.listdir(data), flush=True)
subprocess.run(["nvidia-smi"], check=False)

cmd = [
    sys.executable, "-m", "models.xlit.train",
    "--data", data,
    "--out", "/kaggle/working/xlit",
    "--kn-epochs", "8",
    "--tcy-epochs", "40",
    "--hours", "2.5",  # free GPU quota is only 6 h/week
]  # fmt: skip
subprocess.run(cmd, check=True, cwd=CODE)
