"""Re-derive the headline numbers of the final report from the frozen files in this repository.

Added in October 2026 when the repository was prepared. It does not run any model and does not change any result:
it only reads the frozen manifests and per-image prediction files in this repository and prints the values that
appear in the report (Sections IV and V), so that a reviewer can check them without the project Google Drive.

Usage (from the repository root):  python code/analysis/verify_from_repository.py
"""
import json
from math import comb, sqrt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEL = json.loads((ROOT / 'locks/frozen_selection.json').read_text())
T_ORIG = 0.35996127128601074          # Exp 1 validation Youden threshold (results/exp1_baseline_and_exp2_subgroups/environment_record.json)
T_ADAPT = SEL['adapted_threshold']    # Exp 3 validation Youden threshold (locks/frozen_selection.json)


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f'{100 * (c - h):.2f}-{100 * (c + h):.2f}'


def mcnemar_exact(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def pct(k, n):
    return f'{k}/{n} = {100 * k / n:.2f}% (95% CI {wilson(k, n)})'


m = pd.read_csv(ROOT / 'manifests/dataset_v1/split_manifest.csv')
print('Dataset v1:', len(m), 'images |', m.label.value_counts().to_dict(), '|', m.source.value_counts().to_dict())
print('  splits:', m.split.value_counts().to_dict())
loc = m[m.source == 'Caltech'].groupby('location')['split'].nunique()
print(f'  Caltech camera locations: {loc.size}; locations appearing in more than one split: {(loc > 1).sum()}')

b = pd.read_csv(ROOT / 'results/exp1_baseline_and_exp2_subgroups/baseline_test_predictions.csv')
a = pd.read_csv(ROOT / 'results/exp3_adaptation/test_predictions.csv')
for name, df, t in (('Exp 1 original', b, T_ORIG), ('Exp 3 adapted', a, T_ADAPT)):
    y = df.true_label == 'fake'; p = df.model_score >= t
    tp, tn, fp, fn = (y & p).sum(), (~y & ~p).sum(), (~y & p).sum(), (y & ~p).sum()
    print(f'\n{name} (threshold {t}): TN {tn} FP {fp} FN {fn} TP {tp}')
    print('  accuracy   ', pct(tp + tn, len(df)))
    print('  fake recall', pct(tp, tp + fn))
    print('  real recall', pct(tn, tn + fp))
    r = df[~y]
    for s, g in r.groupby('source'):
        print(f'  real recall {s:12s}', pct((g.model_score < t).sum(), len(g)))
    for s, g in df[y].groupby('style'):
        print(f'  fake recall {s:16s}', pct((g.model_score >= t).sum(), len(g)))

mm = b[['image_id', 'true_label', 'model_score']].merge(a[['image_id', 'model_score']], on='image_id', suffixes=('_o', '_a'))
y = mm.true_label == 'fake'
co = (mm.model_score_o >= T_ORIG) == y; ca = (mm.model_score_a >= T_ADAPT) == y
print(f'\nExp 3 paired: corrected {(~co & ca).sum()}, new errors {(co & ~ca).sum()}, wrong under both {(~co & ~ca).sum()}, '
      f'exact McNemar p = {mcnemar_exact((~co & ca).sum(), (co & ~ca).sum()):.2e}')

e = pd.read_csv(ROOT / 'results/exp4_heldout_generator/exp4_predictions.csv')
do, da = e.score_original >= T_ORIG, e.score_adapted >= T_ADAPT
print(f'\nExp 4 FLUX.1-schnell (n = {len(e)}):')
print('  original fake recall', pct(do.sum(), len(e)))
print('  adapted  fake recall', pct(da.sum(), len(e)))
print(f'  both detected {(do & da).sum()}, both missed {(~do & ~da).sum()}, missed only by original {(~do & da).sum()}, '
      f'missed only by adapted {(do & ~da).sum()}, exact McNemar p = {mcnemar_exact((~do & da).sum(), (do & ~da).sum()):.2f}')
for s, g in e.groupby('style'):
    print(f'  {s:16s} original', pct((g.score_original >= T_ORIG).sum(), len(g)), '| adapted', pct((g.score_adapted >= T_ADAPT).sum(), len(g)))
