"""Supporting statistics computed ONLY from already-frozen prediction files (no model is run, no decision is changed).
Post-hoc analysis of locked results: Wilson 95% CIs, exact paired McNemar (binomial on discordant pairs), absolute differences.
Inputs (local evidence copies, byte-identical to Drive):
  Verification_Evidence/model_baseline/baseline_test_predictions.csv      (Exp 1, threshold 0.35996127128601074)
  Verification_Evidence/model_adaptation/test_predictions.csv              (Exp 3, threshold 0.5848991870880127)
  Verification_Evidence/model_heldout_generator/evaluation/exp4_predictions.csv (Exp 4)
"""
import json, os, hashlib
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score
E = os.environ.get('HIT401_EVIDENCE', os.path.expanduser('~/Desktop/Deepfake/Verification_Evidence'))  # evidence root (see docs/reproducibility.md)
OUT = os.path.dirname(os.path.abspath(__file__))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 2), round(100 * (c + h), 2)

def mcnemar(b, c):
    return float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0

rows = []
def add(exp, group, metric, k, n, note=''):
    lo, hi = wilson(k, n); rows.append(dict(experiment=exp, group=group, metric=metric, k=k, n=n, value_pct=round(100 * k / n, 2), ci95_low=lo, ci95_high=hi, note=note))

inputs = {}
# ---------------- Dataset v1 TEST: Exp 1 (original) vs Exp 3 (adapted) ----------------
fb = f'{E}/model_baseline/baseline_test_predictions.csv'; fa = f'{E}/model_adaptation/test_predictions.csv'
inputs['exp1'] = (fb, sha(fb)); inputs['exp3'] = (fa, sha(fa))
b = pd.read_csv(fb); a = pd.read_csv(fa)
m = b[['image_id', 'true_label', 'source', 'category', 'style', 'model_score', 'predicted_label']].merge(
    a[['image_id', 'model_score', 'predicted_label']], on='image_id', suffixes=('_orig', '_adapt'))
assert len(m) == 664 and m.image_id.is_unique
assert ((b.model_score >= 0.35996127128601074) == (b.predicted_label == 'fake')).all()
assert ((a.model_score >= 0.5848991870880127) == (a.predicted_label == 'fake')).all()
m['correct_orig'] = m.predicted_label_orig == m.true_label; m['correct_adapt'] = m.predicted_label_adapt == m.true_label
for tag, col in (('Exp1 original', 'orig'), ('Exp3 adapted', 'adapt')):
    c = m[f'correct_{col}']
    add(tag, 'ALL (n=664)', 'accuracy', int(c.sum()), 664)
    add(tag, 'fake (n=330)', 'fake recall', int(c[m.true_label == 'fake'].sum()), 330)
    add(tag, 'real (n=334)', 'real recall', int(c[m.true_label == 'real'].sum()), 334)
    for s in ('Caltech', 'iNaturalist'):
        g = m[m.source == s]; add(tag, f'real {s} (n={len(g)})', 'real recall', int(g[f'correct_{col}'].sum()), len(g))
    for s in ('camera_day', 'camera_night_ir', 'community'):
        g = m[m['style'] == s]; add(tag, f'fake {s} (n={len(g)})', 'fake recall', int(g[f'correct_{col}'].sum()), len(g))
    for s in sorted(m.category.unique()):
        g = m[m.category == s]; add(tag, f'category {s} (n={len(g)})', 'accuracy', int(g[f'correct_{col}'].sum()), len(g))
y = (m.true_label == 'fake').astype(int)
auc = {k: round(float(roc_auc_score(y, m[f'model_score_{k}'])), 6) for k in ('orig', 'adapt')}
ap = {k: round(float(average_precision_score(y, m[f'model_score_{k}'])), 6) for k in ('orig', 'adapt')}
b3 = int((m.correct_orig & ~m.correct_adapt).sum()); c3 = int((~m.correct_orig & m.correct_adapt).sum())
bb = int((m.correct_orig & m.correct_adapt).sum()); ww = int((~m.correct_orig & ~m.correct_adapt).sum())
r = m[m.true_label == 'real']; br = int((r.correct_orig & ~r.correct_adapt).sum()); cr = int((~r.correct_orig & r.correct_adapt).sum())
exp3 = dict(n=664, both_correct=bb, both_wrong=ww, correct_only_original=b3, correct_only_adapted=c3,
            exact_mcnemar_p_all=mcnemar(b3, c3), real_only_new_errors=br, real_only_corrected=cr, exact_mcnemar_p_real=mcnemar(br, cr),
            accuracy_diff_pts=round(100 * (m.correct_adapt.mean() - m.correct_orig.mean()), 2),
            real_recall_diff_pts=round(100 * (r.correct_adapt.mean() - r.correct_orig.mean()), 2),
            inat_real_recall_diff_pts=round(100 * (r[r.source == 'iNaturalist'].correct_adapt.mean() - r[r.source == 'iNaturalist'].correct_orig.mean()), 2),
            roc_auc=auc, average_precision=ap, remaining_errors=list(m[~m.correct_adapt].image_id))
# ---------------- Exp 4 FLUX (fake-only) ----------------
f4 = f'{E}/model_heldout_generator/evaluation/exp4_predictions.csv'; inputs['exp4'] = (f4, sha(f4))
p4 = pd.read_csv(f4); assert len(p4) == 326
o = p4.pred_original == 'fake'; d = p4.pred_adapted == 'fake'
for tag, s in (('Exp4 original', o), ('Exp4 adapted', d)):
    add(tag, 'FLUX fake ALL (n=326)', 'fake recall', int(s.sum()), 326)
    for st in ('camera_day', 'camera_night_ir', 'community'):
        g = p4['style'] == st; add(tag, f'FLUX {st} (n={int(g.sum())})', 'fake recall', int(s[g].sum()), int(g.sum()))
    for sp in sorted(p4.category.unique()):
        g = p4.category == sp; add(tag, f'FLUX {sp} (n={int(g.sum())})', 'fake recall', int(s[g].sum()), int(g.sum()))
b4 = int((o & ~d).sum()); c4 = int((~o & d).sum())
exp4 = dict(n=326, both_detected=int((o & d).sum()), both_missed=int((~o & ~d).sum()), missed_only_by_adapted=b4, missed_only_by_original=c4,
            exact_mcnemar_p=mcnemar(b4, c4), recall_diff_pts=round(100 * (d.mean() - o.mean()), 2))
# SDXL vs FLUX fake recall (unpaired: different images) per model
sd = {}
for tag, col, fl in (('original', 'orig', o), ('adapted', 'adapt', d)):
    k1 = int(m[m.true_label == 'fake'][f'correct_{col}'].sum()); k2 = int(fl.sum())
    sd[tag] = dict(sdxl=f'{k1}/330', flux=f'{k2}/326', sdxl_ci=wilson(k1, 330), flux_ci=wilson(k2, 326), diff_pts=round(100 * (k2 / 326 - k1 / 330), 2))
res = dict(note='Post-hoc analysis of frozen predictions only. No model run, no threshold or selection change.',
           inputs={k: {'file': v[0].replace(os.path.expanduser('~'), '~'), 'sha256': v[1]} for k, v in inputs.items()},
           dataset_v1_test_exp1_vs_exp3=exp3, exp4_flux_original_vs_adapted=exp4, sdxl_vs_flux_fake_recall=sd)
pd.DataFrame(rows).to_csv(f'{OUT}/wilson_confidence_intervals.csv', index=False)
json.dump(res, open(f'{OUT}/paired_tests_and_summary.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
