"""Batch analysis: validate every upload, score the valid ones with BOTH frozen detectors, build one record per file.

Scoring goes through the unchanged Prototype v1 pathway Detectors.score_files (models loaded once, reused for every
image). Files are scored in chunks only so that progress can be shown. Experiment evidence (batch sizes 1, 8 and 16)
reproduced saved scores within about 1e-6, and v2 verification re-checks this on the bundled images.
No file is ever silently skipped: every submitted file gets a record with a processing_status.
"""
import time
import warnings
from pathlib import Path

from metadata import extract_metadata, validate_image
from status import evaluate_scores

CHUNK = 16


def analyse_files(detectors, files, progress=None):
    """files: list of (path, display_name). Returns (records, elapsed_seconds)."""
    t0 = time.time()
    records, valid = [], []
    for i, (path, name) in enumerate(files):
        md = extract_metadata(path, name)
        ok, pstatus, note = validate_image(path)
        rec = {**md, "path": str(path), "processing_status": pstatus, "processing_note": note,
               "original_score": None, "original_threshold": None, "original_decision": "",
               "adapted_score": None, "adapted_threshold": None, "adapted_decision": "",
               "models_agree": None, "platform_status": "NOT ANALYSED"}
        records.append(rec)
        if ok:
            valid.append(i)

    for c in range(0, len(valid), CHUNK):
        idx = valid[c:c + CHUNK]
        paths = [records[i]["path"] for i in idx]
        if progress:
            progress((c + len(idx)) / max(len(valid), 1), desc=f"Scoring images {c + 1}-{c + len(idx)} of {len(valid)}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                scores = detectors.score_files(paths)
            except Exception as e:  # keep the app alive; report the whole chunk as failed
                scores, err = {}, f"Scoring error ({type(e).__name__}: {str(e)[:120]})"
            else:
                err = None
        # the official dataset replaces an unreadable image with a random other one and only warns;
        # any such warning invalidates the affected file (validation should make this impossible)
        bad = {w_path for w in caught for w_path in paths if "Error loading image" in str(w.message) and Path(w_path).name in str(w.message)}
        for i in idx:
            r = records[i]
            s = scores.get(r["path"], {})
            if err or r["path"] in bad or not {"original", "adapted"} <= set(s):
                r.update(processing_status="FAILED", processing_note=err or "The model could not read this image.")
                continue
            r.update(evaluate_scores(s["original"], s["adapted"]))
    return records, time.time() - t0
