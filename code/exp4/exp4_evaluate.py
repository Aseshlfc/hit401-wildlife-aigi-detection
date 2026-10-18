"""Experiment 4 - score the FROZEN Exp 4 fake set with BOTH frozen detectors. Run ONCE, in a FRESH Colab GPU runtime.

  python3 exp4_evaluate.py
Environment reproduces Experiment 3 exactly (same repo commit, checkpoint revision + SHA-256, pinned packages,
same model construction, same scoring function: official GenerativeImageDataset eval transform, fp32, eval mode).
No training, no threshold selection, no model selection. Thresholds are the frozen Exp 1 / Exp 3 values.
Integrity check (does NOT touch the Dataset v1 TEST split): both models must reproduce their saved VALIDATION scores.
Writes evaluation/exp4_predictions.csv and the lock evaluation/EXP4_EVALUATED.json.
"""
import subprocess, sys, os, json, hashlib, datetime, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from exp4_common import *

LOCK = EVAL / "EXP4_EVALUATED.json"
assert not LOCK.exists(), f"Experiment 4 was already evaluated: {json.load(open(LOCK))}. Do not re-run."
FROZEN4 = json.load(open(META / "EXP4_DATASET_FROZEN.json"))
man4 = __import__("pandas").read_csv(META / "exp4_frozen_manifest.csv")
assert hashlib.sha256((META / "exp4_frozen_manifest.csv").read_bytes()).hexdigest() == FROZEN4["frozen_manifest_sha256"]
N4 = FROZEN4["n_images"]; assert len(man4) == N4

# ---------------- environment identical to Experiment 3 ----------------
pk = ["transformers==4.57.3", "huggingface_hub==0.36.0", "timm==1.0.10", "tensorboardx==2.6.4", "thop==0.1.1.post2209072238"]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pk], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
OMNIAID_REPO_URL = "https://github.com/yunncheng/OmniAID.git"
OMNIAID_COMMIT = "40749406fbcd8893c11a160edf4a72a2d4dc7056"
HF_REPO, HF_REVISION, HF_FILENAME = "Yunncheng/OmniAID", "279cae7398ac6636f46fc4668f755f11210b36bf", "ckpt/checkpoint_omniaid_v2.pth"
BASE_CKPT_SHA256 = "919338928cab88cef1e6e4b85777749c74bdff6a1796fcd7e43a91422327f906"
ADAPT = DRIVE / "metadata" / "model_adaptation"
FROZEN3 = json.load(open(ADAPT / "frozen_selection.json"))
T_ORIG = 0.35996127128601074            # Exp 1 frozen threshold (validation Youden)
T_ADAPT = FROZEN3["adapted_threshold"]  # 0.5848991870880127 (Exp 3 frozen)
assert abs(T_ADAPT - 0.5848991870880127) < 1e-15 and FROZEN3["selected_epoch"] == 2
REPO = Path("/content/OmniAID_exp4")
if not REPO.exists(): subprocess.run(["git", "clone", "-q", OMNIAID_REPO_URL, str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "-q", OMNIAID_COMMIT], check=True)
assert subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == OMNIAID_COMMIT
assert subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip() == ""

import torch, torchvision, transformers, timm, numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
BASE_CKPT = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME, revision=HF_REVISION)
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    return h.hexdigest()
assert sha256_file(BASE_CKPT) == BASE_CKPT_SHA256
SELECTED_FILE = Path(FROZEN3["adapted_file"])
assert sha256_file(SELECTED_FILE) == FROZEN3["adapted_file_sha256"] == "733ab455a3d1f677451c9368846eb44e36236d492571d7960d075da641a3cd66"

os.chdir(REPO); sys.path.insert(0, str(REPO))
import main_finetune
from data.datasets import GenerativeImageDataset
from torch.utils.data import DataLoader

# identical argument set to Exp 3 Stage 7b (training fields are irrelevant for inference but kept identical)
ARCH_CFG = ADAPT / "config_omniaid_v2_recovered.json"
DUMMY = Path("/content/exp4_dummy"); (DUMMY / "0_real").mkdir(parents=True, exist_ok=True)
args = main_finetune.get_args_parser().parse_args([
    "--model", "OmniAID", "--img_size", "336", "--batch_size", "4", "--update_freq", "8", "--epochs", "5", "--blr", "0.0002",
    "--opt", "adamw", "--weight_decay", "0.0", "--min_lr", "1e-06", "--warmup_epochs", "0", "--smoothing", "0.1",
    "--use_amp", "True", "--seed", "0", "--num_workers", "2", "--data_path", str(DUMMY), "--eval_data_path", str(DUMMY),
    "--output_dir", "/content/exp4_run", "--moe_config_path", str(ARCH_CFG), "--is_hybrid", "True",
    "--training_mode", "standard", "--pretrained_checkpoint", BASE_CKPT, "--save_ckpt", "False", "--dist_eval", "False"])
for k, v in json.load(open(ARCH_CFG)).items():
    if getattr(args, k, None) is None: setattr(args, k, v)
device = main_finetune.setup_for_training(args)

@torch.no_grad()
def score_folder(net, root, batch_size=16):   # verbatim Exp 3 Stage 7c scoring
    ds = GenerativeImageDataset(root=str(root), is_train=False, resolution=args.img_size, model_name=args.model)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    net.eval(); rows = []
    for x, y, paths, _ in dl:
        out = net(x.to(device)); logits = out["cls"].float().cpu(); prob = out["prob"].float().cpu()
        for pth, yy, pr, lg in zip(paths, y.tolist(), prob.tolist(), logits.tolist()):
            rows.append({"filename": os.path.basename(pth), "y_true": yy, "score": pr, "logit_real": lg[0], "logit_fake": lg[1]})
    return pd.DataFrame(rows)

def build_original():
    m, m_wo, _, _ = main_finetune.create_model_and_ema(args, device)
    base = torch.load(BASE_CKPT, map_location="cpu", weights_only=False, mmap=True)
    m_wo.load_state_dict(base["model"], strict=True); del base; m_wo.eval(); return m_wo

def build_adapted(adapted_path):  # verbatim logic of Exp 3 build_adapted_model
    a = torch.load(adapted_path, map_location="cpu"); assert a["base_checkpoint_sha256"] == BASE_CKPT_SHA256
    m_wo = build_original(); sd = m_wo.state_dict()
    assert all(k in sd and sd[k].shape == v.shape for k, v in a["adapted_state"].items())
    res = m_wo.load_state_dict(a["adapted_state"], strict=False)
    assert not res.unexpected_keys and set(sd) - set(res.missing_keys) == set(a["adapted_state"])
    m_wo.eval(); return m_wo

# views (symlinks, nothing copied): VALIDATION for the integrity check; frozen Exp 4 set for scoring
manifest = pd.read_csv(DRIVE / "metadata" / "split_manifest.csv"); manifest["filename"] = manifest.local_path.map(os.path.basename)
val_m = manifest[manifest.split == "val"]
VAL_VIEW = Path("/content/exp4_val_view")
for row in val_m.itertuples():
    link = VAL_VIEW / ("0_real" if row.label == "real" else "1_fake") / row.filename; link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists(): os.symlink(DRIVE / "final" / "validation" / row.label / row.filename, link)
fname2id = dict(zip(val_m.filename, val_m.image_id))
E4_VIEW = Path("/content/exp4_eval_view/1_fake"); E4_VIEW.mkdir(parents=True, exist_ok=True)
for rr in man4.itertuples():
    link = E4_VIEW / rr.filename
    if not link.exists(): os.symlink(FINAL_DIR / rr.filename, link)
assert len(list(E4_VIEW.iterdir())) == N4
for rr in man4.itertuples():   # frozen images unchanged since freeze
    assert hashlib.sha256((FINAL_DIR / rr.filename).read_bytes()).hexdigest() == rr.sha256_jpeg

integrity = {}; results = {}
for name, builder, ref, thr in [("original", build_original, "validation_predictions_epoch0_unadapted.csv", T_ORIG),
                                ("adapted", lambda: build_adapted(SELECTED_FILE), "validation_predictions.csv", T_ADAPT)]:
    t0 = time.time(); net = builder()
    v = score_folder(net, VAL_VIEW); v["image_id"] = v.filename.map(fname2id)
    rec = pd.read_csv(ADAPT / ref); c = v.merge(rec[["image_id", "score"]], on="image_id", suffixes=("_now", "_saved"))
    md = float((c.score_now - c.score_saved).abs().max()); flips = int(((c.score_now >= thr) != (c.score_saved >= thr)).sum())
    integrity[name] = {"validation_n": len(c), "max_abs_score_diff_vs_saved": md, "decision_flips_at_threshold": flips, "reference": ref}
    print(f"[{name}] validation reproduction: n={len(c)} max|diff|={md:.2e} flips={flips}", flush=True)
    assert len(c) == 693 and md < 1e-4 and flips == 0, f"{name}: validation scores not reproduced - STOP"
    s = score_folder(net, E4_VIEW.parent); assert len(s) == N4 and (s.y_true == 1).all()
    s = s.rename(columns={"score": f"score_{name}", "logit_real": f"logit_real_{name}", "logit_fake": f"logit_fake_{name}"}).drop(columns="y_true")
    s[f"pred_{name}"] = np.where(s[f"score_{name}"] >= thr, "fake", "real")
    results[name] = s; print(f"[{name}] scored {N4} in {time.time() - t0:.0f}s (incl. build + validation)", flush=True)
    del net; torch.cuda.empty_cache()

pred = man4[["image_id", "filename", "category", "style", "n", "seed"]].merge(results["original"], on="filename").merge(results["adapted"], on="filename")
assert len(pred) == N4
EVAL.mkdir(parents=True, exist_ok=True)
pred.to_csv(EVAL / "exp4_predictions.csv", index=False)
env = {"torch": torch.__version__, "torchvision": torchvision.__version__, "transformers": transformers.__version__, "timm": timm.__version__,
       "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0], "omniaid_commit": OMNIAID_COMMIT,
       "base_checkpoint": f"{HF_REPO}@{HF_REVISION}:{HF_FILENAME}", "base_checkpoint_sha256": BASE_CKPT_SHA256,
       "adapted_file": str(SELECTED_FILE), "adapted_file_sha256": FROZEN3["adapted_file_sha256"]}
json.dump({"evaluated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "n": N4,
           "threshold_original": T_ORIG, "threshold_adapted": T_ADAPT, "frozen_manifest_sha256": FROZEN4["frozen_manifest_sha256"],
           "predictions_sha256": hashlib.sha256((EVAL / "exp4_predictions.csv").read_bytes()).hexdigest(),
           "validation_integrity": integrity, "environment": env}, open(LOCK, "w"), indent=1)
print("EXP4 scored once. Lock written:", LOCK)
