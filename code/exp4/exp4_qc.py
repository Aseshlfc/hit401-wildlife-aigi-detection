"""Experiment 4 - quality control (no detector involved at any point).

  python3 exp4_qc.py auto                       # automated checks + contact sheets for images without a QC decision
  python3 exp4_qc.py record --reviewed <ids.txt> --rejects <rejects.json> --reviewer <name>
      ids.txt     : image_ids visually reviewed in this round (one per line)
      rejects.json: {"image_id": {"decision": "REJECT"|"NEEDS_USER", "rule": "R1".."R8", "reason": "..."}}
      every reviewed id not in rejects.json is recorded as ACCEPT.

Fixed visual QC rules (same criteria as Dataset v1 SDXL review, decided before generation):
  R1 picture frame / border / mat / letterbox around the scene
  R2 invented text, watermark, logo, caption or timestamp overlay
  R3 fake user-interface / on-screen elements
  R4 malformed anatomy (wrong number of limbs/heads/tails, distorted face or body)
  R5 two animals merged, doubled head, duplicated body parts
  R6 wrong or unidentifiable target species (incl. no animal visible)
  R7 style mismatch (camera_night_ir must look like a night infrared/monochrome trail-camera frame;
     camera_day like a daytime trail-camera frame; community like a handheld/consumer photo), or cartoon/CGI/painting look
  R8 broken scene geometry, split-panel or collage composition, severe rendering artifact
NOT rejection reasons (they occur in real wildlife photos): darkness, blur, noise, odd framing, small or partly hidden animal.
Automated checks (A1-A4): decodes as RGB 768x768; not blank/NaN (pixel std >= 2); SHA-256 unique; near-duplicate
(pHash Hamming <= 4 within the same species) -> flagged for visual attention (not auto-rejected).
Genuinely ambiguous cases -> NEEDS_USER (decided by the user, then recorded).
"""
import argparse, datetime, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from exp4_common import *
import pandas as pd, numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("auto")
r = sub.add_parser("record"); r.add_argument("--reviewed", required=True); r.add_argument("--rejects", required=True)
r.add_argument("--reviewer", required=True)
rs = sub.add_parser("resolve"); rs.add_argument("--decisions", required=True)   # user decisions for NEEDS_USER images
co = sub.add_parser("correct"); co.add_argument("--corrections", required=True)  # audited correction of a recorded decision (before freeze only)
a = ap.parse_args()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
QC = META / "qc_decisions.csv"; AUTO = META / "qc_auto_checks.csv"
man = pd.read_csv(META / "generation_manifest.csv")
qc = pd.read_csv(QC) if QC.exists() else pd.DataFrame(columns=["image_id", "qc_decision", "rule", "reason", "auto_flags", "reviewer", "decided_utc", "review_round"])

if a.cmd == "auto":
    import imagehash
    rows = []
    for rr in man.itertuples():
        p = CAND_DIR / rr.filename; flags = []
        try:
            im = Image.open(p); im.load(); ok_mode = im.mode == "RGB"; size = im.size
            arr = np.asarray(im.convert("RGB"))
            if size != (768, 768) or not ok_mode: flags.append("A1_size_or_mode")
            if arr.std() < 2: flags.append("A2_blank")
            ph = str(imagehash.phash(im.convert("RGB")))
        except Exception as e:
            flags.append(f"A1_decode_error:{e}"); ph = ""
        rows.append({"image_id": rr.image_id, "category": rr.category, "style": rr.style, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "phash": ph, "flags": flags})
    from collections import Counter
    shac = Counter(r["sha256"] for r in rows)
    for r in rows:
        if shac[r["sha256"]] > 1: r["flags"].append("A3_exact_duplicate")
    for cat in sorted({r["category"] for r in rows}):
        g = [r for r in rows if r["category"] == cat and r["phash"]]
        hs = [imagehash.hex_to_hash(r["phash"]) for r in g]
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                if hs[x] - hs[y] <= 4:
                    g[x]["flags"].append(f"A4_near_dup:{g[y]['image_id']}"); g[y]["flags"].append(f"A4_near_dup:{g[x]['image_id']}")
    for r in rows: r["flags"] = ";".join(r["flags"])
    df = pd.DataFrame(rows)
    df.to_csv(AUTO, index=False)
    print("auto checks:", len(df), "images;", int((df["flags"] != "").sum()), "flagged")
    # contact sheets: one per species x style, only images without a decision yet
    todo = man[~man.image_id.isin(qc.image_id)].sort_values(["category", "style", "n"])
    SHEETS.mkdir(parents=True, exist_ok=True)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception: font = ImageFont.load_default()
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); made = []
    for (cat, sty), g in todo.groupby(["category", "style"], sort=False):
        ids = list(g.image_id); cols = 4; T = 360; H = 26
        rows_n = (len(ids) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * T, rows_n * (T + H)), "white"); d = ImageDraw.Draw(sheet)
        for k, iid in enumerate(ids):
            im = Image.open(CAND_DIR / f"{iid}.jpg").convert("RGB").resize((T, T))
            x, y = (k % cols) * T, (k // cols) * (T + H)
            sheet.paste(im, (x, y + H)); d.text((x + 4, y + 3), iid.replace("flux_", ""), fill="black", font=font)
        out = SHEETS / f"sheet_{tag}_{cat}_{sty}.jpg"; sheet.save(out, quality=90); made.append(out.name)
        with open(SHEETS / f"sheet_{tag}_{cat}_{sty}.txt", "w") as fh: fh.write("\n".join(ids) + "\n")
    print("contact sheets:", len(made)); print("\n".join(made))

elif a.cmd == "correct":
    # {"image_id": {"decision": ..., "rule": ..., "why_corrected": ...}}; every change is logged in qc_corrections.csv
    assert not (META / "EXP4_DATASET_FROZEN.json").exists(), "dataset frozen - no corrections allowed"
    corr = json.load(open(a.corrections)); rows = []
    for iid, d in corr.items():
        idx = qc.index[qc.image_id == iid]; assert len(idx) == 1, iid
        assert d["decision"] in ("ACCEPT", "REJECT", "NEEDS_USER") and d.get("why_corrected")
        i = idx[0]; t = now()
        rows.append({"image_id": iid, "old_decision": qc.at[i, "qc_decision"], "old_rule": qc.at[i, "rule"], "old_reason": qc.at[i, "reason"],
                     "new_decision": d["decision"], "new_rule": d.get("rule", ""), "new_reason": d.get("reason", ""),
                     "why_corrected": d["why_corrected"], "corrected_utc": t})
        qc.at[i, "qc_decision"] = d["decision"]; qc.at[i, "rule"] = d.get("rule", ""); qc.at[i, "reason"] = d.get("reason", "")
        qc.at[i, "decided_utc"] = t
    qc.to_csv(QC, index=False)
    L = META / "qc_corrections.csv"; pd.DataFrame(rows).to_csv(L, mode="a", header=not L.exists(), index=False)
    print("corrected:", [(r["image_id"], r["old_decision"], "->", r["new_decision"]) for r in rows])

elif a.cmd == "resolve":
    # {"image_id": {"decision": "ACCEPT"|"REJECT", "rule": "...", "note": "..."}} - decisions made by the USER
    dec = json.load(open(a.decisions)); log_rows = []
    for iid, d in dec.items():
        idx = qc.index[qc.image_id == iid]
        assert len(idx) == 1 and qc.at[idx[0], "qc_decision"] == "NEEDS_USER", f"{iid} is not a pending NEEDS_USER image"
        assert d["decision"] in ("ACCEPT", "REJECT")
        old = qc.at[idx[0], "reason"]; t = now()
        log_rows.append({"image_id": iid, "previous_decision": "NEEDS_USER", "previous_reason": old,
                         "user_decision": d["decision"], "rule": d.get("rule", ""), "note": d.get("note", ""), "resolved_utc": t})
        qc.at[idx[0], "qc_decision"] = d["decision"]; qc.at[idx[0], "rule"] = d.get("rule", "")
        qc.at[idx[0], "reason"] = f"USER DECISION: {d.get('note', '')} (was NEEDS_USER: {old})"
        qc.at[idx[0], "reviewer"] = "user"; qc.at[idx[0], "decided_utc"] = t
    qc.to_csv(QC, index=False)
    L = META / "qc_user_resolutions.csv"
    pd.DataFrame(log_rows).to_csv(L, mode="a", header=not L.exists(), index=False)
    print(f"resolved {len(log_rows)} NEEDS_USER images:", {r['image_id']: r['user_decision'] for r in log_rows})
    print("pending NEEDS_USER now:", int((qc.qc_decision == "NEEDS_USER").sum()))

else:
    reviewed = [l.strip() for l in open(a.reviewed) if l.strip()]
    rej = json.load(open(a.rejects))
    assert set(rej) <= set(reviewed), "rejected ids must be in the reviewed list"
    assert set(reviewed) <= set(man.image_id), "unknown image id"
    already = set(qc.image_id) & set(reviewed); assert not already, f"already decided: {sorted(already)[:5]}"
    auto = pd.read_csv(AUTO).set_index("image_id") if AUTO.exists() else None
    rnd = int(qc.review_round.max()) + 1 if len(qc) else 1
    new = []
    for iid in reviewed:
        d = rej.get(iid, {"decision": "ACCEPT", "rule": "", "reason": ""})
        flags = auto.at[iid, "flags"] if auto is not None and iid in auto.index else ""
        flags = "" if isinstance(flags, float) else flags
        if any(f.startswith(("A1", "A2", "A3")) for f in flags.split(";") if f) and d["decision"] == "ACCEPT":
            raise SystemExit(f"{iid} failed an automated check ({flags}); it cannot be ACCEPTed")
        new.append({"image_id": iid, "qc_decision": d["decision"], "rule": d.get("rule", ""), "reason": d.get("reason", ""),
                    "auto_flags": flags, "reviewer": a.reviewer, "decided_utc": now(), "review_round": rnd})
    qc = pd.concat([qc, pd.DataFrame(new)], ignore_index=True); qc.to_csv(QC, index=False)
    m = man.merge(qc[["image_id", "qc_decision"]], on="image_id", how="left")
    summ = m.groupby(["category", "style"]).qc_decision.apply(lambda s: (s == "ACCEPT").sum()).unstack()
    print(f"round {rnd}: recorded {len(new)} ({sum(x['qc_decision'] != 'ACCEPT' for x in new)} not accepted)")
    print("ACCEPTED per cell:\n", summ, "\nquota:", QUOTA)
