"""Experiment 4 - analysis of the locked predictions (no model is run here).

PRIMARY (RQ4, fake-only set): fake recall / sensitivity, missed fakes, FNR, score distributions, by style, by species,
original-vs-adapted per-image paired comparison (exact McNemar on discordant pairs), Wilson 95% CIs.
SUPPLEMENTARY (clearly labelled, not the RQ4 result): composite of the 334 frozen Dataset v1 TEST real images
(scores already saved in Exp 1 / Exp 3 - the TEST split is NOT re-run) + the frozen Exp 4 fakes.
"""
import json, sys, datetime, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from exp4_common import *
import numpy as np, pandas as pd
from scipy.stats import binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

lock = json.load(open(EVAL / "EXP4_EVALUATED.json"))
P = pd.read_csv(EVAL / "exp4_predictions.csv")
assert hashlib.sha256((EVAL / "exp4_predictions.csv").read_bytes()).hexdigest() == lock["predictions_sha256"]
T = {"original": lock["threshold_original"], "adapted": lock["threshold_adapted"]}
PLOTS = EVAL / "plots"; PLOTS.mkdir(exist_ok=True)

def wilson(k, n, z=1.959964):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))

def recall_table(df, by):
    rows = []
    groups = [("ALL", df)] if by is None else list(df.groupby(by))
    for g, d in groups:
        row = {"group": g if not isinstance(g, tuple) else "/".join(g), "n": len(d)}
        for m in ("original", "adapted"):
            k = int((d[f"pred_{m}"] == "fake").sum()); n = len(d)
            row.update({f"{m}_detected": k, f"{m}_missed": n - k, f"{m}_fake_recall": round(k / n, 4),
                        f"{m}_fnr": round(1 - k / n, 4), f"{m}_recall_ci95": wilson(k, n),
                        f"{m}_mean_score": round(float(d[f"score_{m}"].mean()), 4), f"{m}_median_score": round(float(d[f"score_{m}"].median()), 4)})
        row["recall_change_pts"] = round(100 * (row["adapted_fake_recall"] - row["original_fake_recall"]), 2)
        rows.append(row)
    return pd.DataFrame(rows)

overall = recall_table(P, None); by_style = recall_table(P, "style"); by_species = recall_table(P, "category")
by_cell = recall_table(P, ["category", "style"])
for name, t in [("overall", overall), ("by_style", by_style), ("by_species", by_species), ("by_species_style", by_cell)]:
    t.to_csv(EVAL / f"exp4_recall_{name}.csv", index=False)

# paired per-image comparison
o = P.pred_original == "fake"; a = P.pred_adapted == "fake"
P["paired_outcome"] = np.select([o & a, ~o & ~a, o & ~a, ~o & a],
                                ["both_detected", "both_missed", "missed_only_by_adapted", "missed_only_by_original"])
b = int((o & ~a).sum()); c = int((~o & a).sum())
mcn = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
P["score_change"] = P.score_adapted - P.score_original
P.to_csv(EVAL / "exp4_paired_comparison.csv", index=False)
P[P.paired_outcome != "both_detected"].to_csv(EVAL / "exp4_missed_fakes.csv", index=False)

# context: the same detectors on the SDXL (Dataset v1) TEST fakes - saved results, not re-run
base = pd.read_csv(DRIVE / "metadata" / "model_baseline" / "baseline_test_predictions.csv")
ad3 = pd.read_csv(DRIVE / "metadata" / "model_adaptation" / "test_predictions.csv")
sdxl_ctx = {"original_sdxl_test_fake_recall": round(float((base[base.true_label == "fake"].predicted_label == "fake").mean()), 4),
            "adapted_sdxl_test_fake_recall": round(float((ad3[ad3.true_label == "fake"].predicted_label == "fake").mean()), 4),
            "n_sdxl_test_fakes": int((base.true_label == "fake").sum())}

primary = {"n_fake_images": len(P), "generator": f"{MODEL_ID}@{MODEL_REVISION}", "thresholds": T,
           "overall": overall.iloc[0].to_dict(), "paired": {"both_detected": int((o & a).sum()), "both_missed": int((~o & ~a).sum()),
           "missed_only_by_adapted": b, "missed_only_by_original": c, "exact_mcnemar_p_two_sided": round(float(mcn), 6)},
           "score_change_adapted_minus_original": {"mean": round(float(P.score_change.mean()), 4), "median": round(float(P.score_change.median()), 4)},
           "context_sdxl_dataset_v1_test": sdxl_ctx}
json.dump(primary, open(EVAL / "exp4_primary_results.json", "w"), indent=1, default=str)

# SUPPLEMENTARY composite: 334 frozen TEST reals (saved scores) + frozen Exp 4 fakes
supp = {"label": "SUPPLEMENTARY composite (NOT the primary RQ4 result): 334 Dataset v1 TEST real images (saved scores; "
                 "original-model real scores come from the Exp 1 scorer, adapted from Exp 3) + " + str(len(P)) + " Exp 4 FLUX fakes"}
for m, reals in (("original", base[base.true_label == "real"].model_score.values), ("adapted", ad3[ad3.true_label == "real"].model_score.values)):
    s = np.r_[reals, P[f"score_{m}"].values]; y = np.r_[np.zeros(len(reals)), np.ones(len(P))]; pr = (s >= T[m]).astype(int)
    tn = int(((pr == 0) & (y == 0)).sum()); fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum()); tp = int(((pr == 1) & (y == 1)).sum())
    supp[m] = {"n_real": len(reals), "n_fake": len(P), "TN": tn, "FP": fp, "FN": fn, "TP": tp, "accuracy": round((tp + tn) / len(y), 4),
               "balanced_accuracy": round((tp / (tp + fn) + tn / (tn + fp)) / 2, 4), "precision": round(tp / (tp + fp), 4) if tp + fp else None,
               "roc_auc": round(float(roc_auc_score(y, s)), 4), "average_precision": round(float(average_precision_score(y, s)), 4)}
json.dump(supp, open(EVAL / "exp4_supplementary_composite.json", "w"), indent=1)

# plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for i, m in enumerate(("original", "adapted")):
    ax[i].hist(P[f"score_{m}"], bins=30, range=(0, 1), color="C1", alpha=.8)
    ax[i].axvline(T[m], color="k", ls="--", label=f"frozen threshold {T[m]:.4f}")
    ax[i].set_title(f"{m} OmniAID v2 - FLUX fakes (n={len(P)})"); ax[i].set_xlabel("score (higher = more fake-like)"); ax[i].legend()
ax[0].set_ylabel("count"); plt.tight_layout(); plt.savefig(PLOTS / "exp4_score_distributions.png", dpi=150); plt.close()
for name, t in (("style", by_style), ("species", by_species)):
    x = np.arange(len(t)); plt.figure(figsize=(max(6, len(t) * 1.1), 4))
    plt.bar(x - .2, t.original_fake_recall, .4, label="original"); plt.bar(x + .2, t.adapted_fake_recall, .4, label="adapted (Exp 3)")
    plt.xticks(x, t.group, rotation=20); plt.ylim(0, 1.05); plt.ylabel("fake recall"); plt.title(f"Exp 4 FLUX fake recall by {name}")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / f"exp4_recall_by_{name}.png", dpi=150); plt.close()
plt.figure(figsize=(5, 5))
for sty, g in P.groupby("style"): plt.scatter(g.score_original, g.score_adapted, s=12, label=sty, alpha=.7)
plt.axvline(T["original"], color="k", ls="--", lw=.8); plt.axhline(T["adapted"], color="k", ls=":", lw=.8)
plt.xlabel("original score"); plt.ylabel("adapted score"); plt.title("Per-image scores, FLUX fakes"); plt.legend()
plt.xlim(0, 1); plt.ylim(0, 1); plt.tight_layout(); plt.savefig(PLOTS / "exp4_paired_scores.png", dpi=150); plt.close()

summary = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "primary": primary,
           "by_style": by_style.to_dict("records"), "by_species": by_species.to_dict("records"), "supplementary": supp}
json.dump(summary, open(EVAL / "experiment4_summary.json", "w"), indent=1, default=str)
print(json.dumps(primary, indent=1, default=str)); print(by_style[["group", "n", "original_fake_recall", "adapted_fake_recall", "original_missed", "adapted_missed"]])
print(by_species[["group", "n", "original_fake_recall", "adapted_fake_recall", "original_missed", "adapted_missed"]]); print(json.dumps(supp, indent=1))
