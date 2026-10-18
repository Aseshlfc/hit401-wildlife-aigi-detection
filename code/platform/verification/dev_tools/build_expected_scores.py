"""Build verification_images/expected_scores.csv from the SAVED research predictions (no model is run).

Sources (project evidence, read-only):
  Dataset v1 VALIDATION, original model : Verification_Evidence/model_adaptation/validation_predictions_epoch0_unadapted.csv (Exp 1/3 reference)
  Dataset v1 VALIDATION, adapted model  : Verification_Evidence/model_adaptation/validation_predictions.csv (Exp 3 selected epoch)
  Exp 4 frozen FLUX set, both models    : Verification_Evidence/model_heldout_generator/evaluation/exp4_predictions.csv
Run on the development Mac:  python3 verification/dev_tools/build_expected_scores.py
"""
import hashlib, os, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from config import T_ORIGINAL, T_ADAPTED  # noqa: E402

EV = Path(os.environ.get("HIT401_EVIDENCE", Path.home() / "Desktop/Deepfake/Verification_Evidence"))
IMG = HERE / "verification_images"

ROLE = {  # why each image is in the set (chosen from saved predictions; no TEST image)
    "cct_opossum_5976a4fe-23d2-11e8-a6a3-ec086b02610b.jpg": "authentic camera-trap; Prototype v1 example",
    "cct_raccoon_59c4c055-23d2-11e8-a6a3-ec086b02610b.jpg": "authentic camera-trap (night IR)",
    "inat_bobcat_1628425_1997958.jpg": "authentic citizen-science photo (CC0)",
    "inat_bobcat_3277526_3789805.jpg": "authentic citizen-science trail-camera photo (CC0)",
    "inat_skunk_4310670_5178747.jpg": "detector disagreement on an authentic image (CC0): original AI-GENERATED, adapted REAL",
    "sdxl_opossum_community_500004.jpg": "SDXL synthetic; Prototype v1 example",
    "sdxl_opossum_camera_night_ir_504002.jpg": "SDXL synthetic night IR; Prototype v1 example",
    "sdxl_bobcat_camera_night_ir_544080.jpg": "detector disagreement on an SDXL image: original REAL, adapted AI-GENERATED",
    "flux_bobcat_community_640016.jpg": "FLUX.1-schnell synthetic (held out from project adaptation); Prototype v1 example",
    "flux_coyote_camera_night_ir_634000.jpg": "FLUX.1-schnell synthetic night IR",
    "flux_bobcat_community_640037.jpg": "detector disagreement on a FLUX image: original REAL, adapted AI-GENERATED",
    "flux_squirrel_camera_day_652005.jpg": "KNOWN LIMITATION: FLUX synthetic image that BOTH detectors classify as REAL (one of 15 in Exp 4)",
}


def decision(s, t):
    return "AI-GENERATED" if s >= t else "REAL"


def main():
    vo = pd.read_csv(EV / "model_adaptation/validation_predictions_epoch0_unadapted.csv").set_index("filename")
    va = pd.read_csv(EV / "model_adaptation/validation_predictions.csv").set_index("filename")
    e4 = pd.read_csv(EV / "model_heldout_generator/evaluation/exp4_predictions.csv").set_index("filename")
    rows = []
    for fn, role in ROLE.items():
        p = IMG / fn
        if fn in e4.index:
            so, sa, src, truth = e4.loc[fn, "score_original"], e4.loc[fn, "score_adapted"], "Exp 4 frozen FLUX.1-schnell set", "AI-generated"
            ref = "exp4_predictions.csv (score_original, score_adapted)"
        else:
            so, sa = vo.loc[fn, "score"], va.loc[fn, "score"]
            src = f"Dataset v1 VALIDATION, {va.loc[fn, 'source']}"
            truth = "authentic" if va.loc[fn, "label"] == "real" else "AI-generated"
            ref = "validation_predictions_epoch0_unadapted.csv (original); validation_predictions.csv (adapted)"
        do, da = decision(so, T_ORIGINAL), decision(sa, T_ADAPTED)
        status = {("REAL", "REAL"): "LIKELY AUTHENTIC", ("AI-GENERATED", "AI-GENERATED"): "LIKELY AI-GENERATED"}.get((do, da), "REVIEW SUGGESTED")
        rows.append({"filename": fn, "known_origin": truth, "research_set": src,
                     "expected_original_score": repr(float(so)), "original_threshold": repr(T_ORIGINAL), "expected_original_decision": do,
                     "expected_adapted_score": repr(float(sa)), "adapted_threshold": repr(T_ADAPTED), "expected_adapted_decision": da,
                     "expected_platform_status": status,
                     "min_margin_to_threshold": round(min(abs(so - T_ORIGINAL), abs(sa - T_ADAPTED)), 6),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "score_source": ref, "role": role})
    out = pd.DataFrame(rows)
    out.to_csv(IMG / "expected_scores.csv", index=False)
    print(out[["filename", "expected_original_decision", "expected_adapted_decision", "expected_platform_status", "min_margin_to_threshold"]].to_string())


if __name__ == "__main__":
    main()
