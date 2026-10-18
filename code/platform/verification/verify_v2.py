"""Prototype v2 SOFTWARE verification (not an experiment). Run in the SAME Colab kernel after the app has started:

    %run -i verification/verify_v2.py            (DET = the loaded Detectors object created by app.py)

Uses only the already-loaded frozen detectors. Nothing is trained, no threshold is chosen, no TEST image is scored.
Checks: A model integrity, B known-image score reproduction (bundled images, Prototype v1 examples, optional Drive
validation images), C batch-size consistency, D mixed batch incl. edge files + CSV + metadata, E review logic, F timing.
Writes verification/results/v2_verification.json and v2_verification_summary.md (or HIT401_VERIFY_OUT).
If the project Drive is mounted (HIT401_DRIVE), a few extra Drive-only validation images are also compared.
"""
import datetime, hashlib, json, os, platform, sys, tempfile, time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.environ["HIT401_APP_DIR"]) / "verification"
APP = HERE.parent
sys.path.insert(0, str(APP))
import config as C  # noqa: E402
from batch_processing import analyse_files  # noqa: E402
from exports import summarise, write_csv  # noqa: E402
from status import REVIEW, decide, evaluate_scores  # noqa: E402
from inference import sha256_file  # noqa: E402

OUT = Path(os.environ.get("HIT401_VERIFY_OUT") or HERE / "results"); OUT.mkdir(parents=True, exist_ok=True)
DRIVE = Path(os.environ.get("HIT401_DRIVE", "/content/drive/MyDrive/HIT401_Wildlife_Deepfake"))
TOL = C.SELF_CHECK_TOLERANCE
det = globals().get("DET")
assert det is not None, "Run the app first in this kernel (%run app.py) so that DET exists."
R = {"run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
     "nature": "software verification of Prototype v2; not a research experiment; no TEST image scored",
     "environment": {"gpu": torch.cuda.get_device_name(0), "python": platform.python_version(), "torch": torch.__version__}}
for m in ("transformers", "timm", "gradio", "huggingface_hub", "PIL"):
    try:
        R["environment"][m] = __import__(m).__version__
    except Exception:
        pass
checks = {}

# ---------------- A. model integrity ----------------
from huggingface_hub import hf_hub_download  # noqa: E402
base = C.BASE_CKPT_FILE or hf_hub_download(repo_id=C.HF_REPO, filename=C.HF_FILENAME, revision=C.HF_REVISION)
hb, ha, hc = sha256_file(base), sha256_file(C.ADAPTED_FILE), sha256_file(C.ARCH_CONFIG)
import subprocess  # noqa: E402
head = subprocess.run(["git", "-C", str(C.REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
o_net, t_o = det.models["original"]; a_net, t_a = det.models["adapted"]
so, sa = o_net.state_dict(), a_net.state_dict()
changed = [k for k in so if not torch.equal(so[k], sa[k])]
adapted_keys = set(torch.load(C.ADAPTED_FILE, map_location="cpu")["adapted_state"])
A = {"base_checkpoint_sha256": hb, "base_matches": hb == C.BASE_CKPT_SHA256,
     "adapted_sha256": ha, "adapted_matches": ha == C.ADAPTED_SHA256,
     "arch_config_sha256": hc, "arch_config_matches": hc == C.ARCH_CONFIG_SHA256,
     "omniaid_commit": head, "commit_matches": head == C.OMNIAID_COMMIT,
     "original_threshold": t_o, "adapted_threshold": t_a,
     "thresholds_match_frozen": t_o == 0.35996127128601074 and t_a == 0.5848991870880127,
     "original_eval_mode": not o_net.training, "adapted_eval_mode": not a_net.training,
     "total_parameters": int(sum(p.numel() for p in o_net.parameters())),
     "tensors_differing": len(changed), "adapted_state_tensors": len(adapted_keys),
     "differing_tensors_all_in_adapted_state": set(changed) <= adapted_keys,
     "v1_reference": {"total_parameters": 507797575, "tensors_differing": 1734}}
A["passed"] = all([A["base_matches"], A["adapted_matches"], A["arch_config_matches"], A["commit_matches"],
                   A["thresholds_match_frozen"], A["original_eval_mode"], A["adapted_eval_mode"],
                   A["total_parameters"] == 507797575, A["tensors_differing"] == 1734, A["differing_tensors_all_in_adapted_state"]])
R["A_model_integrity"] = A; checks["A"] = A["passed"]
del so, sa

# ---------------- B. known-image score reproduction ----------------
exp = pd.read_csv(C.EXPECTED_SCORES)
files = [str(C.VERIFY_IMAGES_DIR / f) for f in exp.filename]
t0 = time.time(); got = det.score_files(files); t_batch = time.time() - t0
rows = []
for r, f in zip(exp.itertuples(), files):
    g = got[f]
    rows.append({"file": r.filename, "set": r.research_set, "known_origin": r.known_origin,
                 "v2_original_score": g["original"], "saved_original_score": r.expected_original_score,
                 "abs_diff_original": abs(g["original"] - r.expected_original_score),
                 "v2_adapted_score": g["adapted"], "saved_adapted_score": r.expected_adapted_score,
                 "abs_diff_adapted": abs(g["adapted"] - r.expected_adapted_score),
                 "v2_original_decision": decide(g["original"], t_o), "saved_original_decision": r.expected_original_decision,
                 "v2_adapted_decision": decide(g["adapted"], t_a), "saved_adapted_decision": r.expected_adapted_decision,
                 "v2_status": evaluate_scores(g["original"], g["adapted"])["platform_status"],
                 "expected_status": r.expected_platform_status})
v1 = json.load(open(HERE / "reference_v1" / "prototype_verification.json"))["sample_images"] \
    if (HERE / "reference_v1" / "prototype_verification.json").exists() else []
v1map = {x["file"]: x for x in v1}
for row in rows:  # also compare with the Prototype v1 run (6 d.p. as recorded there)
    if row["file"] in v1map:
        row["v1_original_score_6dp"] = v1map[row["file"]]["original_score"]; row["v1_adapted_score_6dp"] = v1map[row["file"]]["adapted_score"]
        row["abs_diff_vs_v1_6dp"] = max(abs(round(row["v2_original_score"], 6) - row["v1_original_score_6dp"]),
                                        abs(round(row["v2_adapted_score"], 6) - row["v1_adapted_score_6dp"]))

drive_rows = []
if (DRIVE / "metadata" / "split_manifest.csv").exists():  # optional extra checks with the project Drive
    man = pd.read_csv(DRIVE / "metadata" / "split_manifest.csv"); man["filename"] = man.local_path.map(os.path.basename)
    val = man[man.split == "val"]
    v1_selfcheck = pd.concat([val[val.label == "real"].head(2), val[val.label == "fake"].head(2)])  # v1's 4 self-check images
    extra = list(v1_selfcheck.itertuples()) + list(man[man.filename.isin(["inat_opossum_486213_614582.jpg",
                                                                         "sdxl_opossum_camera_night_ir_504061.jpg"])].itertuples())
    vo = pd.read_csv(DRIVE / "metadata/model_adaptation/validation_predictions_epoch0_unadapted.csv").set_index("image_id").score
    va = pd.read_csv(DRIVE / "metadata/model_adaptation/validation_predictions.csv").set_index("image_id").score
    paths = [str(DRIVE / "final" / "validation" / e.label / e.filename) for e in extra]
    g2 = det.score_files(paths)
    for e, p in zip(extra, paths):
        razor = e.filename == "sdxl_opossum_camera_night_ir_504061.jpg"
        drive_rows.append({"file": e.filename, "role": "razor-edge case (score reproduction only; decision NOT verified)" if razor
                           else ("Prototype v1 example" if "614582" in e.filename else "Prototype v1 start-up self-check image"),
                           "v2_original_score": g2[p]["original"], "saved_original_score": float(vo[e.image_id]),
                           "v2_adapted_score": g2[p]["adapted"], "saved_adapted_score": float(va[e.image_id]),
                           "max_abs_diff": max(abs(g2[p]["original"] - vo[e.image_id]), abs(g2[p]["adapted"] - va[e.image_id])),
                           "decision_checked": not razor,
                           "decisions_match": (not razor) and decide(g2[p]["original"], t_o) == decide(vo[e.image_id], t_o)
                           and decide(g2[p]["adapted"], t_a) == decide(va[e.image_id], t_a)})
mx = max(max(r["abs_diff_original"], r["abs_diff_adapted"]) for r in rows)
dec_ok = all(r["v2_original_decision"] == r["saved_original_decision"] and r["v2_adapted_decision"] == r["saved_adapted_decision"]
             and r["v2_status"] == r["expected_status"] for r in rows)
dmx = max([r["max_abs_diff"] for r in drive_rows], default=0.0)
ddec = all(r["decisions_match"] for r in drive_rows if r["decision_checked"])
R["B_known_image_reproduction"] = {"bundled_images": rows, "bundled_max_abs_diff": mx, "bundled_decisions_and_status_match": dec_ok,
                                   "drive_images": drive_rows, "drive_max_abs_diff": dmx if drive_rows else None,
                                   "drive_decisions_match": ddec if drive_rows else None, "tolerance": TOL}
checks["B"] = mx < TOL and dec_ok and (not drive_rows or (dmx < TOL and ddec))

# ---------------- C. batch-size consistency (the app scores in chunks) ----------------
single = {f: det.score_files([f])[f] for f in files}
cmx = max(max(abs(single[f]["original"] - got[f]["original"]), abs(single[f]["adapted"] - got[f]["adapted"])) for f in files)
R["C_batch_consistency"] = {"images": len(files), "max_abs_diff_batch_vs_single": cmx, "tolerance": TOL}
checks["C"] = cmx < TOL

# ---------------- D. mixed batch through the platform pathway ----------------
from PIL import Image  # noqa: E402
tmp = Path(tempfile.mkdtemp(prefix="v2verify_"))
Image.new("RGB", (320, 240), (90, 120, 80)).save(tmp / "generated_no_exif.jpg")
Image.new("RGBA", (200, 200), (10, 80, 40, 128)).save(tmp / "generated_rgba.png")
(tmp / "notes.txt").write_text("not an image")
(tmp / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0 broken data")
full = (C.VERIFY_IMAGES_DIR / exp.filename[0]).read_bytes(); (tmp / "truncated.jpg").write_bytes(full[: len(full) // 2])
batch = [(f, Path(f).name) for f in files] + [(str(tmp / n), n) for n in
                                              ("generated_no_exif.jpg", "generated_rgba.png", "notes.txt", "corrupt.jpg", "truncated.jpg")]
t0 = time.time(); recs, elapsed = analyse_files(det, batch)
csvp = write_csv(recs, OUT / "sample_batch_results.csv")
csv_rows = pd.read_csv(csvp)
by = {r["filename"]: r for r in recs}
expected_fail = {"notes.txt": "UNSUPPORTED", "corrupt.jpg": "FAILED", "truncated.jpg": "FAILED"}
D = {"submitted": len(batch), "records": len(recs), "csv_rows": len(csv_rows),
     "summary": summarise(recs, elapsed),
     "edge_statuses": {n: by[n]["processing_status"] for n in ("generated_no_exif.jpg", "generated_rgba.png", *expected_fail)},
     "bundled_status_match": all(by[r.filename]["platform_status"] == r.expected_platform_status for r in exp.itertuples()),
     "bundled_scores_match_batch1": max(abs(by[r.filename]["original_score"] - r.expected_original_score) for r in exp.itertuples()) < TOL,
     "metadata_example": {k: by["generated_no_exif.jpg"][k] for k in ("format", "dimensions", "colour_mode", "exif_present", "gps_metadata_present")},
     "csv_has_no_paths": str(tmp) not in open(csvp).read() and str(C.VERIFY_IMAGES_DIR) not in open(csvp).read()}
D["passed"] = (D["records"] == D["submitted"] == D["csv_rows"] == 17 and D["bundled_status_match"] and D["bundled_scores_match_batch1"]
               and all(D["edge_statuses"][n] == s for n, s in expected_fail.items())
               and D["edge_statuses"]["generated_no_exif.jpg"] == "ANALYSED" and D["edge_statuses"]["generated_rgba.png"] == "ANALYSED"
               and D["summary"]["analysed"] == 14 and D["summary"]["failed_or_unsupported"] == 3 and D["csv_has_no_paths"]
               and D["metadata_example"]["exif_present"] is False)
R["D_mixed_batch"] = D; checks["D"] = D["passed"]
R["D_note"] = "The two generated test images are plain synthetic colour fields; their scores/statuses are software checks only and carry no meaning."

# ---------------- E. review-suggested logic on real saved disagreements ----------------
dis = [r for r in rows if r["expected_status"] == REVIEW]
R["E_review_logic"] = {"real_disagreement_examples": [{k: r[k] for k in ("file", "v2_original_decision", "v2_adapted_decision", "v2_status")} for r in dis],
                       "passed": len(dis) == 3 and all(r["v2_status"] == REVIEW for r in dis)}
checks["E"] = R["E_review_logic"]["passed"]

# ---------------- F. application timing (performance only, not a research result) ----------------
R["F_timing_application_only"] = {"bundled_12_images_one_call_seconds": round(t_batch, 2), "mixed_batch_17_files_seconds": round(elapsed, 2),
                                  "seconds_per_analysed_image_mixed_batch": round(elapsed / 14, 3)}
R["checks"] = checks; R["all_passed"] = all(checks.values())
json.dump(R, open(OUT / "v2_verification.json", "w"), indent=1, default=str)

md = [f"# Prototype v2 verification ({R['run_utc']})", "", "Software verification only; not a research experiment. No TEST image was scored.", "",
      f"Environment: {R['environment']}", "", "| Check | Result |", "|---|---|"]
names = {"A": "Model integrity (hashes, commit, thresholds, eval mode, 507,797,575 params, 1,734 adapted tensors)",
         "B": f"Known-image score reproduction (max diff bundled {mx:.2e}; Drive {dmx:.2e}; tolerance {TOL:g})",
         "C": f"Batch vs single-image scoring (max diff {cmx:.2e})", "D": "Mixed batch of 17 files: counts, statuses, CSV rows, metadata, edge files",
         "E": "Review Suggested on the 3 real saved disagreement examples"}
md += [f"| {k}. {names[k]} | {'PASS' if v else 'FAIL'} |" for k, v in checks.items()]
md += ["", "## Bundled images: v2 vs saved research scores", "", "| File | v2 original | saved | v2 adapted | saved | max diff | status |", "|---|---|---|---|---|---|---|"]
md += [f"| {r['file']} | {r['v2_original_score']:.6f} | {r['saved_original_score']:.6f} | {r['v2_adapted_score']:.6f} | {r['saved_adapted_score']:.6f} | "
       f"{max(r['abs_diff_original'], r['abs_diff_adapted']):.1e} | {r['v2_status']} |" for r in rows]
if drive_rows:
    md += ["", "## Drive-only validation images", "", "| File | role | max diff | decisions match |", "|---|---|---|---|"]
    md += [f"| {r['file']} | {r['role']} | {r['max_abs_diff']:.1e} | {r['decisions_match'] if r['decision_checked'] else 'not checked (razor edge)'} |" for r in drive_rows]
(OUT / "v2_verification_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md)); print("\nALL PASSED" if R["all_passed"] else "\nSOME CHECKS FAILED", "->", OUT)
