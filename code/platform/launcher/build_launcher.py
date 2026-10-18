"""Writes launcher/HIT401_Platform_Colab.ipynb (run on any machine: python3 launcher/build_launcher.py)."""
import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}  # noqa: E731
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}  # noqa: E731

cells = [
    md("""
# Wildlife Image Authenticity Analysis Platform: Colab launcher
HIT401 research prototype (Prototype v2). It runs two **frozen** research detectors (the original OmniAID v2 and the
project's wildlife-adapted OmniAID v2) on one or many wildlife images. Nothing is trained or re-tuned.

**Before you start:** *Runtime -> Change runtime type -> T4 GPU*, then run the cells from top to bottom.

You need two things:
1. **The platform package**: `HIT401_Wildlife_Authenticity_Platform.zip`.
2. **The project's adapted parameter file**: `adapted_epoch_1.pt` (6.4 MB). It is *not* inside the ZIP (see
   `models/README_MODEL_SETUP.md`); the project team provides it to you privately.

**Easiest way:** before running anything, open the **Files** panel (folder icon on the left) → **Upload to session
storage** and upload both files. The cells below find them automatically in `/content/`. (If you skip this, cells 2 and 4
show an upload button instead.)

The official base checkpoint (3.24 GB) is downloaded automatically from Hugging Face; no account or token is needed.
"""),
    md("## 1. Check the GPU"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; assert torch.cuda.is_available(), "No GPU: choose Runtime -> Change runtime type -> T4 GPU, then run again."
print("GPU:", torch.cuda.get_device_name(0))
"""),
    md("""
## 2. Get the platform package
If you uploaded `HIT401_Wildlife_Authenticity_Platform.zip` with the Files panel, nothing to do. Otherwise leave
`PACKAGE_PATH` empty and an upload button appears. If the ZIP or an unzipped platform folder is in your Google Drive, put its path here instead (this needs the
Drive to be mounted: run `from google.colab import drive; drive.mount('/content/drive')` first).
"""),
    code("""
PACKAGE_PATH = ""   # e.g. "/content/drive/MyDrive/HIT401_Wildlife_Authenticity_Platform.zip" or a folder path; empty = upload

import os, shutil, zipfile
APP_DIR = "/content/HIT401_Wildlife_Authenticity_Platform"
def _install(src):
    if os.path.isdir(src):
        shutil.copytree(src, APP_DIR, dirs_exist_ok=True, ignore=shutil.ignore_patterns("_omniaid_repo", "_run", "__pycache__", "*.pt"))
    else:
        with zipfile.ZipFile(src) as z:
            z.extractall("/content")
UPLOADED_ZIP = "/content/HIT401_Wildlife_Authenticity_Platform.zip"   # uploaded via the Files panel
if not os.path.exists(f"{APP_DIR}/app.py"):
    if PACKAGE_PATH:
        _install(PACKAGE_PATH)
    elif os.path.exists(UPLOADED_ZIP):
        _install(UPLOADED_ZIP)
    else:
        from google.colab import files
        up = files.upload()
        if not up:
            raise SystemExit("No file uploaded. Upload HIT401_Wildlife_Authenticity_Platform.zip and run this cell again.")
        _install(next(iter(up)))
assert os.path.exists(f"{APP_DIR}/app.py"), "Platform files not found"
print("Platform files ready in", APP_DIR)
"""),
    md("## 3. Install the pinned packages (about 1-2 minutes)"),
    code("""
!pip install -q -r "$APP_DIR/requirements.txt"
"""),
    md("""
## 4. Provide the adapted parameter file `adapted_epoch_1.pt`
If you uploaded `adapted_epoch_1.pt` with the Files panel, nothing to do. Otherwise leave `ADAPTED_PATH` empty for an
upload button, or give its path in your mounted Google Drive.
The file is checked against its frozen SHA-256 (`733ab455...`); a wrong or modified file is rejected.
"""),
    code("""
ADAPTED_PATH = ""   # e.g. "/content/drive/MyDrive/adapted_epoch_1.pt"; empty = upload

import hashlib
target = f"{APP_DIR}/models/adapted_epoch_1.pt"
UPLOADED_PT = "/content/adapted_epoch_1.pt"   # uploaded via the Files panel
if not os.path.exists(target):
    if ADAPTED_PATH:
        shutil.copy(ADAPTED_PATH, target)
    elif os.path.exists(UPLOADED_PT):
        shutil.copy(UPLOADED_PT, target)
    else:
        from google.colab import files
        up = files.upload()
        if not up:
            raise SystemExit("No file uploaded. Upload adapted_epoch_1.pt and run this cell again.")
        shutil.move(next(iter(up)), target)
sha = hashlib.sha256(open(target, "rb").read()).hexdigest()
assert sha == "733ab455a3d1f677451c9368846eb44e36236d492571d7960d075da641a3cd66", f"Wrong adapted file (SHA-256 {sha})"
print("adapted_epoch_1.pt OK:", sha)
"""),
    md("""
## 5. Start the platform
First start: clones the official OmniAID code (pinned commit), downloads the 3.24 GB base checkpoint, checks all
SHA-256 hashes and re-scores the 12 bundled verification images against their saved research scores (about 5-10
minutes). It will not start if anything differs. Then open the printed **App URL** in a new browser tab.
If Colab asks whether this notebook may use a secret such as `HF_TOKEN`, you may decline: no token is needed.
"""),
    code("""
import os
from google.colab.output import eval_js
os.environ["HIT401_ROOT_URL"] = eval_js("google.colab.kernel.proxyPort(7860)", timeout_sec=60).rstrip("/")  # needs this tab to be responsive
os.environ["HIT401_APP_DIR"] = APP_DIR
%cd $APP_DIR
print("App URL (open in a new tab once the app has started):", os.environ["HIT401_ROOT_URL"])
%run app.py
"""),
    md("""
## 6. Optional: record verification evidence
Re-checks model integrity, re-scores the bundled verification images, tests a mixed batch with invalid files and writes
`verification/results/v2_verification.json` and `v2_verification_summary.md` inside the platform folder.
"""),
    code("""
%run -i $APP_DIR/verification/verify_v2.py
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parent / "HIT401_Platform_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
