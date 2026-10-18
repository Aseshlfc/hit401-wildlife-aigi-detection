"""Load the two FROZEN detectors used in this project (inference only; nothing is trained or re-selected here).

  original : OmniAID v2 official checkpoint, frozen Exp 1 threshold 0.35996127128601074 (validation Youden)
  adapted  : the same checkpoint + the frozen Exp 3 parameter file adapted_epoch_1.pt (epoch 2),
             frozen Exp 3 threshold 0.5848991870880127 (validation Youden)

Model construction, preprocessing and scoring are the same code path as Experiment 4 (exp4_evaluate.py):
official OmniAID repository at a pinned commit, official GenerativeImageDataset eval transform, fp32, eval mode.
Runs in Google Colab with a GPU and the project Google Drive mounted.
"""
import copy, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

DRIVE = Path(os.environ.get("HIT401_DRIVE", "/content/drive/MyDrive/HIT401_Wildlife_Deepfake"))
ADAPT = DRIVE / "metadata" / "model_adaptation"
OMNIAID_REPO_URL = "https://github.com/yunncheng/OmniAID.git"
OMNIAID_COMMIT = "40749406fbcd8893c11a160edf4a72a2d4dc7056"
HF_REPO, HF_REVISION, HF_FILENAME = "Yunncheng/OmniAID", "279cae7398ac6636f46fc4668f755f11210b36bf", "ckpt/checkpoint_omniaid_v2.pth"
BASE_CKPT_SHA256 = "919338928cab88cef1e6e4b85777749c74bdff6a1796fcd7e43a91422327f906"
ADAPTED_SHA256 = "733ab455a3d1f677451c9368846eb44e36236d492571d7960d075da641a3cd66"
T_ORIGINAL = 0.35996127128601074
T_ADAPTED = 0.5848991870880127
REPO = Path("/content/OmniAID_prototype")


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
        import torch
        from huggingface_hub import hf_hub_download
        self.torch = torch
        base_ckpt = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME, revision=HF_REVISION)
        frozen = json.load(open(ADAPT / "frozen_selection.json"))
        adapted_file = Path(frozen["adapted_file"])
        assert abs(frozen["adapted_threshold"] - T_ADAPTED) < 1e-15 and frozen["selected_epoch"] == 2
        if verify_hashes:  # about 1 minute; proves the exact frozen weights are loaded
            assert sha256_file(base_ckpt) == BASE_CKPT_SHA256, "base checkpoint hash mismatch"
            assert sha256_file(adapted_file) == ADAPTED_SHA256, "adapted parameter file hash mismatch"

        os.chdir(REPO); sys.path.insert(0, str(REPO))
        import main_finetune
        from data.datasets import GenerativeImageDataset
        self.GenerativeImageDataset = GenerativeImageDataset
        cfg = ADAPT / "config_omniaid_v2_recovered.json"
        dummy = Path(tempfile.mkdtemp()) / "0_real"; dummy.mkdir(parents=True)
        # same argument set as Exp 3 / Exp 4 (training-only fields are unused at inference)
        args = main_finetune.get_args_parser().parse_args([
            "--model", "OmniAID", "--img_size", "336", "--batch_size", "4", "--update_freq", "8", "--epochs", "5",
            "--blr", "0.0002", "--opt", "adamw", "--weight_decay", "0.0", "--min_lr", "1e-06", "--warmup_epochs", "0",
            "--smoothing", "0.1", "--use_amp", "True", "--seed", "0", "--num_workers", "0",
            "--data_path", str(dummy.parent), "--eval_data_path", str(dummy.parent), "--output_dir", "/content/prototype_run",
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

    def self_check(self, n_per_class=2):
        """Re-score a few VALIDATION images and compare with the saved Exp 1 / Exp 3 validation scores."""
        import pandas as pd
        man = pd.read_csv(DRIVE / "metadata" / "split_manifest.csv")
        man["filename"] = man.local_path.map(os.path.basename)
        val = man[man.split == "val"]
        pick = pd.concat([val[val.label == "real"].head(n_per_class), val[val.label == "fake"].head(n_per_class)])
        files = [str(DRIVE / "final" / "validation" / r.label / r.filename) for r in pick.itertuples()]
        got = self.score_files(files)
        saved_o = pd.read_csv(ADAPT / "validation_predictions_epoch0_unadapted.csv").set_index("image_id").score
        saved_a = pd.read_csv(ADAPT / "validation_predictions.csv").set_index("image_id").score
        diffs = []
        for r, f in zip(pick.itertuples(), files):
            diffs.append(abs(got[f]["original"] - saved_o[r.image_id])); diffs.append(abs(got[f]["adapted"] - saved_a[r.image_id]))
        return {"images": len(files), "max_abs_diff": max(diffs), "passed": max(diffs) < 1e-4}
