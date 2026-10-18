"""Paths and frozen constants for the Wildlife Image Authenticity Analysis Platform (Prototype v2).

Everything that locates a file can be overridden with an environment variable, so the platform does not depend on
any personal Google Drive layout. The frozen research constants below are copied from the project evidence
(frozen_selection.json, EXP4_EVALUATED.json, Prototype v1 omniaid_models.py) and must never be edited.
"""
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def _path(env, default):
    return Path(os.environ.get(env) or default).expanduser().resolve()


# ---- locations (portable; override with environment variables) ----
MODELS_DIR = _path("HIT401_MODELS_DIR", APP_DIR / "models")
# frozen Exp 3 parameter file (NOT bundled, see models/README_MODEL_SETUP.md)
ADAPTED_FILE = _path("HIT401_ADAPTED_FILE", MODELS_DIR / "adapted_epoch_1.pt")
# architecture config used by Exp 3, Exp 4 and Prototype v1 (project file, bundled)
ARCH_CONFIG = _path("HIT401_ARCH_CONFIG", MODELS_DIR / "config_omniaid_v2_recovered.json")
# optional: a local copy of the official base checkpoint; otherwise it is downloaded from Hugging Face
BASE_CKPT_FILE = os.environ.get("HIT401_BASE_CKPT") or None
# clone of the official OmniAID repository at the pinned commit (created on first start)
REPO_DIR = _path("HIT401_OMNIAID_REPO", APP_DIR / "_omniaid_repo")
# scratch directory (only passed as the unused --output_dir argument, as in v1)
RUN_DIR = _path("HIT401_RUN_DIR", APP_DIR / "_run")
VERIFY_IMAGES_DIR = _path("HIT401_VERIFY_IMAGES", APP_DIR / "verification_images")
EXPECTED_SCORES = VERIFY_IMAGES_DIR / "expected_scores.csv"

# ---- frozen research constants (identical to Prototype v1 / Experiments 1, 3 and 4) ----
OMNIAID_REPO_URL = "https://github.com/yunncheng/OmniAID.git"
OMNIAID_COMMIT = "40749406fbcd8893c11a160edf4a72a2d4dc7056"
HF_REPO, HF_REVISION, HF_FILENAME = "Yunncheng/OmniAID", "279cae7398ac6636f46fc4668f755f11210b36bf", "ckpt/checkpoint_omniaid_v2.pth"
BASE_CKPT_SHA256 = "919338928cab88cef1e6e4b85777749c74bdff6a1796fcd7e43a91422327f906"
ADAPTED_SHA256 = "733ab455a3d1f677451c9368846eb44e36236d492571d7960d075da641a3cd66"
ARCH_CONFIG_SHA256 = "213ae1d675f490812c5b6cba202d3894978c7b7cd1bd67c4c9a3a2a1f5d6ff4a"
T_ORIGINAL = 0.35996127128601074   # Exp 1, Youden's J on VALIDATION
T_ADAPTED = 0.5848991870880127     # Exp 3 (epoch 2 = adapted_epoch_1.pt), Youden's J on VALIDATION
SELECTED_EPOCH = 2

# start-up self-check tolerance: the same value Prototype v1 and Experiment 4 used
SELF_CHECK_TOLERANCE = 1e-4

# upload handling (platform policy, applied BEFORE the unchanged model pathway)
# the official OmniAID dataset only scans these extensions (data/datasets.py at the pinned commit)
MODEL_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
MAX_FILE_MB = float(os.environ.get("HIT401_MAX_FILE_MB", "50"))
MAX_BATCH = int(os.environ.get("HIT401_MAX_BATCH", "200"))
