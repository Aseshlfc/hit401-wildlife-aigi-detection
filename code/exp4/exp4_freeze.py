"""Experiment 4 - freeze the QC-accepted image set BEFORE any detector sees it.

  python3 exp4_freeze.py
Checks: every generated image has a final decision (no NEEDS_USER left); every cell has exactly its quota of ACCEPT;
accepted images are, per cell, exactly the accepted seeds in predefined order (no gaps skipped). Copies accepted
images to final/1_fake/, writes exp4_frozen_manifest.csv (with SHA-256) and EXP4_DATASET_FROZEN.json (lock).
"""
import datetime, hashlib, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from exp4_common import *
import pandas as pd

LOCK = META / "EXP4_DATASET_FROZEN.json"
assert not LOCK.exists(), f"Experiment 4 dataset already frozen: {json.load(open(LOCK))}"
assert not (EVAL / "EXP4_EVALUATED.json").exists()
man = pd.read_csv(META / "generation_manifest.csv"); qc = pd.read_csv(META / "qc_decisions.csv")
assert man.image_id.is_unique and qc.image_id.is_unique
assert set(man.image_id) == set(qc.image_id), "some images have no QC decision"
assert not (qc.qc_decision == "NEEDS_USER").any(), "unresolved NEEDS_USER decisions"
assert set(qc.qc_decision) <= {"ACCEPT", "REJECT"}
m = man.merge(qc, on="image_id")
capped = {}   # user decision 2026-09-26: freeze at quota-or-cap; cells that hit the 5x cap keep what they have
for ai, (cat, _) in enumerate(ANIMALS):
    for si, sty in enumerate(STYLES):
        c = m[(m.category == cat) & (m["style"] == sty)].sort_values("n")
        assert list(c.n) == list(range(len(c))), f"{cat}/{sty}: seed sequence has gaps {list(c.n)}"
        acc = c[c.qc_decision == "ACCEPT"]
        cap = 5 * QUOTA[sty]
        if len(acc) == QUOTA[sty]:
            assert (c.n.max() in set(acc.n)), f"{cat}/{sty}: generated beyond quota (last seed not accepted)"
        else:
            assert len(acc) < QUOTA[sty] and len(c) == cap, f"{cat}/{sty}: {len(acc)} accepted, {len(c)} attempts - short but not at cap {cap}"
            capped[f"{cat}/{sty}"] = {"accepted": int(len(acc)), "quota": QUOTA[sty], "attempts": int(len(c)), "cap": cap}
final = m[m.qc_decision == "ACCEPT"].sort_values(["category", "style", "n"]).reset_index(drop=True)
assert len(final) == 336 - sum(v["quota"] - v["accepted"] for v in capped.values())
FINAL_DIR.mkdir(parents=True, exist_ok=True)
assert not any(FINAL_DIR.iterdir()), "final/1_fake is not empty"
for r in final.itertuples():
    src = CAND_DIR / r.filename; dst = FINAL_DIR / r.filename
    shutil.copy2(src, dst)
    assert hashlib.sha256(dst.read_bytes()).hexdigest() == r.sha256_jpeg == hashlib.sha256(src.read_bytes()).hexdigest()
final.to_csv(META / "exp4_frozen_manifest.csv", index=False)
counts = final.groupby("style").size().to_dict()
lock = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "n_images": len(final), "per_style": counts, "per_species": final.groupby("category").size().to_dict(),
        "generated_total": len(man), "rejected_total": int((m.qc_decision == "REJECT").sum()),
        "frozen_manifest_sha256": hashlib.sha256((META / "exp4_frozen_manifest.csv").read_bytes()).hexdigest(),
        "qc_decisions_sha256": hashlib.sha256((META / "qc_decisions.csv").read_bytes()).hexdigest(),
        "generation_manifest_sha256": hashlib.sha256((META / "generation_manifest.csv").read_bytes()).hexdigest(),
        "generator": MODEL_ID, "generator_revision": MODEL_REVISION,
        "planned_n": 336, "capped_cells": capped,
        "cap_rule": "5 x quota attempts per species x style cell (user-set before replacements); user decision 2026-09-26: freeze at 326 with capped cells as-is",
        "note": "Frozen before any detector saw these images. No image may be replaced/removed based on detector results."}
json.dump(lock, open(LOCK, "w"), indent=1)
print(json.dumps(lock, indent=1))
