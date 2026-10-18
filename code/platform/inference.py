"""Load the two FROZEN detectors used in this project (inference only; nothing is trained or re-selected here).

  original : OmniAID v2 official checkpoint, frozen Exp 1 threshold 0.35996127128601074 (validation Youden)
  adapted  : the same checkpoint + the frozen Exp 3 parameter file adapted_epoch_1.pt (epoch 2),
             frozen Exp 3 threshold 0.5848991870880127 (validation Youden)

Model construction, preprocessing and scoring are the same code path as Experiment 4 (exp4_evaluate.py):
official OmniAID repository at a pinned commit, official GenerativeImageDataset eval transform, fp32, eval mode.

Prototype v2: this file is a copy of the VERIFIED Prototype v1 omniaid_models.py. Only file LOCATIONS changed (they now
come from config.py instead of the project Google Drive), and the start-up self-check uses the bundled verification
images instead of Drive validation folders. Model construction, the argument set, preprocessing, score_files,
thresholds and the decision rule are unchanged. See verification/LOADER_CHANGES.md for the line-by-line diff.
Needs a CUDA GPU (the pinned OmniAID argument parser defaults to --device cuda, as in every project run).
"""
import copy, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

from config import (OMNIAID_REPO_URL, OMNIAID_COMMIT, HF_REPO, HF_REVISION, HF_FILENAME, BASE_CKPT_SHA256, ADAPTED_SHA256,
                    ARCH_CONFIG_SHA256, T_ORIGINAL, T_ADAPTED, SELECTED_EPOCH, ADAPTED_FILE, ARCH_CONFIG, BASE_CKPT_FILE,
                    REPO_DIR as REPO, RUN_DIR, EXPECTED_SCORES, VERIFY_IMAGES_DIR, SELF_CHECK_TOLERANCE)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class Detectors:
    """Holds both frozen models and scores single images with the official preprocessing."""

    def __init__(self, verify_hashes=True):
        if not REPO.exists():
            subprocess.run(["git", "clone", "-q", OMNIAID_REPO_URL, str(REPO)], check=True)
        subprocess.run(["git", "-C", str(REPO), "checkout", "-q", OMNIAID_COMMIT], check=True)
        # v2: confirm the checkout really is the pinned, unmodified commit (the same check exp4_evaluate.py made)
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        assert head == OMNIAID_COMMIT and not dirty, f"OmniAID repository is not the clean pinned commit ({head})"
        import torch
        from huggingface_hub import hf_hub_download
        self.torch = torch
        base_ckpt = BASE_CKPT_FILE or hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME, revision=HF_REVISION)
        # v2: the frozen Exp 3 values come from config.py (copied from frozen_selection.json), the file from a portable path
        adapted_file = Path(ADAPTED_FILE)
        if not adapted_file.is_file():
            raise FileNotFoundError(f"Adapted parameter file not found: {adapted_file}. See models/README_MODEL_SETUP.md")
        assert abs(0.5848991870880127 - T_ADAPTED) < 1e-15 and SELECTED_EPOCH == 2
        # v2: the two small project files are always hashed (about 1 s); only the 3.24 GB base hash can be skipped, as in v1
        assert sha256_file(adapted_file) == ADAPTED_SHA256, "adapted parameter file hash mismatch"
        assert sha256_file(ARCH_CONFIG) == ARCH_CONFIG_SHA256, "architecture config hash mismatch"
        self.base_hash_verified = False
        if verify_hashes:  # about 1 minute; proves the exact frozen weights are loaded
            assert sha256_file(base_ckpt) == BASE_CKPT_SHA256, "base checkpoint hash mismatch"
            self.base_hash_verified = True

        os.chdir(REPO); sys.path.insert(0, str(REPO))
        import main_finetune
        from data.datasets import GenerativeImageDataset
        self.GenerativeImageDataset = GenerativeImageDataset
        cfg = ARCH_CONFIG
        dummy = Path(tempfile.mkdtemp()) / "0_real"; dummy.mkdir(parents=True)
        # same argument set as Exp 3 / Exp 4 (training-only fields are unused at inference)
        args = main_finetune.get_args_parser().parse_args([
            "--model", "OmniAID", "--img_size", "336", "--batch_size", "4", "--update_freq", "8", "--epochs", "5",
            "--blr", "0.0002", "--opt", "adamw", "--weight_decay", "0.0", "--min_lr", "1e-06", "--warmup_epochs", "0",
            "--smoothing", "0.1", "--use_amp", "True", "--seed", "0", "--num_workers", "0",
            "--data_path", str(dummy.parent), "--eval_data_path", str(dummy.parent), "--output_dir", str(RUN_DIR),
            "--moe_config_path", str(cfg), "--is_hybrid", "True", "--training_mode", "standard",
            "--pretrained_checkpoint", base_ckpt, "--save_ckpt", "False", "--dist_eval", "False"])
        for k, v in json.load(open(cfg)).items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)
        self.args = args
        self.device = main_finetune.setup_for_training(args)

        # original model
        _, original, _, _ = main_finetune.create_model_and_ema(args, self.device)
        base = torch.load(base_ckpt, map_location="cpu", weights_only=False, mmap=True)
        original.load_state_dict(base["model"], strict=True); del base
        original.eval()
        # adapted model = copy of the original + frozen Exp 3 parameters (same logic as build_adapted_model)
        adapted = copy.deepcopy(original)
        a = torch.load(adapted_file, map_location="cpu")
        assert a["base_checkpoint_sha256"] == BASE_CKPT_SHA256
        sd = adapted.state_dict()
        assert all(k in sd and sd[k].shape == v.shape for k, v in a["adapted_state"].items())
        res = adapted.load_state_dict(a["adapted_state"], strict=False)
        assert not res.unexpected_keys and set(sd) - set(res.missing_keys) == set(a["adapted_state"])
        adapted.eval()
        self.models = {"original": (original, T_ORIGINAL), "adapted": (adapted, T_ADAPTED)}

    def score_files(self, paths):
        """Return {path: {"original": score, "adapted": score}}. Files are linked, never re-encoded."""
        torch = self.torch
        from torch.utils.data import DataLoader
        view = Path(tempfile.mkdtemp()) / "0_real"; view.mkdir(parents=True)
        names = {}
        for i, p in enumerate(paths):
            link = view / f"{i:04d}_{Path(p).name}"
            os.symlink(os.path.abspath(p), link); names[link.name] = p
        ds = self.GenerativeImageDataset(root=str(view.parent), is_train=False, resolution=self.args.img_size, model_name=self.args.model)
        out = {p: {} for p in paths}
        with torch.no_grad():
            for name, (net, _) in self.models.items():
                for x, _, fpaths, _ in DataLoader(ds, batch_size=8, shuffle=False, num_workers=0):
                    prob = net(x.to(self.device))["prob"].float().cpu().tolist()
                    for fp, pr in zip(fpaths, prob):
                        out[names[os.path.basename(fp)]][name] = float(pr)
        return out

    def self_check(self):
        """v2: re-score the bundled verification images (Dataset v1 VALIDATION and frozen Exp 4 images) and compare with
        their saved experiment scores (verification_images/expected_scores.csv). Same 1e-4 tolerance as v1."""
        import pandas as pd
        exp = pd.read_csv(EXPECTED_SCORES)
        files = [str(VERIFY_IMAGES_DIR / f) for f in exp.filename]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            return {"images": 0, "max_abs_diff": float("nan"), "passed": False, "error": f"missing verification images: {missing}"}
        got = self.score_files(files)
        diffs, flips, per_image = [], 0, []
        for r, f in zip(exp.itertuples(), files):
            do, da = abs(got[f]["original"] - r.expected_original_score), abs(got[f]["adapted"] - r.expected_adapted_score)
            diffs += [do, da]
            flips += int((got[f]["original"] >= T_ORIGINAL) != (r.expected_original_score >= T_ORIGINAL))
            flips += int((got[f]["adapted"] >= T_ADAPTED) != (r.expected_adapted_score >= T_ADAPTED))
            per_image.append({"filename": r.filename, "original_score": got[f]["original"], "expected_original_score": r.expected_original_score,
                              "adapted_score": got[f]["adapted"], "expected_adapted_score": r.expected_adapted_score, "max_abs_diff": max(do, da)})
        md = max(diffs)
        return {"images": len(files), "max_abs_diff": md, "decision_flips": flips,
                "passed": md < SELF_CHECK_TOLERANCE and flips == 0, "per_image": per_image}
