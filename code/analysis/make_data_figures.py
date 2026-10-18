"""Data figures for the final report, generated ONLY from frozen evidence files (no model run).
Grayscale-safe: hatching/markers distinguish series; no titles inside images (captions are in the report)."""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
H = os.path.expanduser('~'); E = os.environ.get('HIT401_EVIDENCE', f'{H}/Desktop/Deepfake/Verification_Evidence')  # evidence root (see docs/reproducibility.md)
OUT = os.path.dirname(os.path.abspath(__file__)); W = f'{H}/Desktop/Deepfake/FINAL_CAPSTONE_WORKSPACE'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'font.size': 8.5,
                     'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300,
                     'legend.frameon': False, 'hatch.linewidth': 0.6})
C_O, C_A = '#9a9a9a', '#2b2b2b'   # original = light grey + '//' hatch, adapted = dark
T_O, T_A = 0.35996127128601074, 0.5848991870880127
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', pad_inches=0.03); plt.close(fig); print('saved', name)
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
b = pd.read_csv(f'{E}/model_baseline/baseline_test_predictions.csv'); a = pd.read_csv(f'{E}/model_adaptation/test_predictions.csv')
m = b.merge(a[['image_id', 'model_score', 'predicted_label']], on='image_id', suffixes=('_o', '_a'))
p4 = pd.read_csv(f'{E}/model_heldout_generator/evaluation/exp4_predictions.csv')
man = pd.read_csv(f'{E}/dataset/split_manifest.csv')
# ---------- Fig: dataset composition ----------
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw={'width_ratios': [1, 1.15]})
splits = ['train', 'val', 'test']; lab = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
parts = [('Caltech (real)', lambda d: (d.source == 'Caltech'), 'white', '....'), ('iNaturalist (real)', lambda d: (d.source == 'iNaturalist'), '#d9d9d9', ''),
         ('SDXL community', lambda d: (d['style'] == 'community'), '#8c8c8c', '//'), ('SDXL camera day', lambda d: (d['style'] == 'camera_day'), '#5a5a5a', ''),
         ('SDXL camera night/IR', lambda d: (d['style'] == 'camera_night_ir'), '#1f1f1f', '')]
left = np.zeros(3)
for name, f, col, h in parts:
    v = np.array([int(f(man[man.split == s]).sum()) for s in splits])
    ax[0].barh([lab[s] for s in splits], v, left=left, color=col, hatch=h, edgecolor='black', lw=0.5, label=name); left += v
for i, s in enumerate(splits): ax[0].text(left[i] + 40, i, f'{int(left[i]):,}', va='center', fontsize=8)
ax[0].invert_yaxis(); ax[0].set_xlabel('Images'); ax[0].set_xlim(0, 3900); ax[0].legend(fontsize=7, loc='lower right', ncol=1)
cats = sorted(man.category.unique()); x = np.arange(len(cats))
rv = [int(((man.category == c) & (man.label == 'real')).sum()) for c in cats]; fv = [int(((man.category == c) & (man.label == 'fake')).sum()) for c in cats]
ax[1].bar(x - 0.2, rv, 0.4, color='white', edgecolor='black', lw=0.5, hatch='....', label='Real'); ax[1].bar(x + 0.2, fv, 0.4, color='#5a5a5a', edgecolor='black', lw=0.5, label='Synthetic (SDXL)')
ax[1].set_xticks(x); ax[1].set_xticklabels([c.capitalize() for c in cats], rotation=30); ax[1].set_ylabel('Images'); ax[1].set_ylim(0, 420); ax[1].legend(fontsize=7, ncol=2, loc='upper center')
fig.tight_layout(); save(fig, 'fig_dataset_composition')
# ---------- Fig: confusion matrices before/after ----------
fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.2))
for i, (col, t) in enumerate((('predicted_label_o', 'Original OmniAID v2 (Exp. 1)'), ('predicted_label_a', 'Adapted OmniAID v2 (Exp. 3)'))):
    cm = np.array([[((m.true_label == r) & (m[col] == p)).sum() for p in ('real', 'fake')] for r in ('real', 'fake')])
    ax[i].imshow(cm, cmap='Greys', vmin=0, vmax=600)
    for r in range(2):
        for c in range(2): ax[i].text(c, r, f'{cm[r, c]}', ha='center', va='center', fontsize=11, color='white' if cm[r, c] > 300 else 'black', fontweight='bold')
    ax[i].set_xticks([0, 1]); ax[i].set_xticklabels(['Real', 'AI-generated']); ax[i].set_yticks([0, 1]); ax[i].set_yticklabels(['Real', 'AI-generated'])
    ax[i].set_xlabel('Predicted'); ax[i].set_ylabel('True' if i == 0 else ''); ax[i].set_title(t, fontsize=8.5)
    for s in ax[i].spines.values(): s.set_visible(True)
fig.tight_layout(); save(fig, 'fig_confusion_before_after')
# ---------- Fig: subgroup accuracy before/after with Wilson CI ----------
groups = [('Caltech\n(real, 158)', m.source == 'Caltech'), ('iNaturalist\n(real, 176)', m.source == 'iNaturalist'),
          ('Community\n(SDXL, 168)', m['style'] == 'community'), ('Camera day\n(SDXL, 87)', m['style'] == 'camera_day'), ('Night/IR\n(SDXL, 75)', m['style'] == 'camera_night_ir')]
fig, ax = plt.subplots(figsize=(6.4, 2.4)); x = np.arange(len(groups))
for j, (suf, col, h, lb) in enumerate((('o', C_O, '////', 'Original (Exp. 1)'), ('a', C_A, '', 'Adapted (Exp. 3)'))):
    vals, lo, hi = [], [], []
    for g, msk in groups:
        d = m[msk]; k = int((d[f'predicted_label_{suf}'] == d.true_label).sum()); n = len(d); l, u = wilson(k, n)
        vals.append(100 * k / n); lo.append(max(0, 100 * (k / n - l))); hi.append(max(0, 100 * (u - k / n)))
    ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=col, hatch=h, edgecolor='black', lw=0.5, label=lb, yerr=[lo, hi], capsize=2, error_kw={'lw': 0.6})
    for xi, v in zip(x + (j - 0.5) * 0.38, vals): ax.text(xi, 82.3, f'{v:.1f}', ha='center', fontsize=6.5, rotation=90, color='black', bbox=dict(fc='white', ec='none', pad=0.6))
ax.set_xticks(x); ax.set_xticklabels([g for g, _ in groups]); ax.set_ylabel('Correct (%)'); ax.set_ylim(80, 101.5); ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
fig.tight_layout(); save(fig, 'fig_subgroups_before_after')
# ---------- Fig: category accuracy before/after ----------
fig, ax = plt.subplots(figsize=(6.4, 2.1)); x = np.arange(len(cats))
for j, (suf, col, h, lb) in enumerate((('o', C_O, '////', 'Original (Exp. 1)'), ('a', C_A, '', 'Adapted (Exp. 3)'))):
    vals = [100 * (m[m.category == c][f'predicted_label_{suf}'] == m[m.category == c].true_label).mean() for c in cats]
    ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=col, hatch=h, edgecolor='black', lw=0.5, label=lb)
ax.set_xticks(x); ax.set_xticklabels([f'{c.capitalize()}\n(n={int((m.category == c).sum())})' for c in cats]); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(90, 101); ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
fig.tight_layout(); save(fig, 'fig_category_before_after')
# ---------- Fig: score distributions Dataset v1 TEST ----------
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3), sharey=True); bins = np.linspace(0, 1, 41)
for i, (suf, T, t) in enumerate((('o', T_O, 'Original OmniAID v2'), ('a', T_A, 'Adapted OmniAID v2'))):
    ax[i].hist(m[m.true_label == 'real'][f'model_score_{suf}'], bins, color='white', edgecolor='black', lw=0.6, hatch='....', label='Real (n=334)')
    ax[i].hist(m[m.true_label == 'fake'][f'model_score_{suf}'], bins, color='#4d4d4d', edgecolor='black', lw=0.4, alpha=0.85, label='SDXL (n=330)')
    ax[i].axvline(T, color='black', ls='--', lw=1); ax[i].text(T + 0.015, 0.55, f'threshold\n{T:.4f}', transform=ax[i].get_xaxis_transform(), fontsize=7)
    ax[i].set_yscale('log'); ax[i].set_xlabel('Model score (higher = more AI-like)'); ax[i].set_title(t, fontsize=8.5)
ax[0].set_ylabel('Images (log scale)'); ax[0].legend(fontsize=7, loc='upper right')
fig.tight_layout(); save(fig, 'fig_scores_datasetv1')
# ---------- Fig: Exp 3 validation history ----------
hst = pd.read_csv(f'{E}/model_adaptation/training_history.csv')
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.1))
ax[0].plot(hst.epoch, 100 * hst.val_balanced_accuracy, 'k-o', ms=4, label='Balanced accuracy'); ax[0].plot(hst.epoch, 100 * hst.val_real_recall, 'k--s', ms=4, mfc='white', label='Real recall'); ax[0].plot(hst.epoch, 100 * hst.val_fake_recall, 'k:^', ms=4, mfc='#999', label='Fake recall')
ax[0].axvline(2, color='#777', lw=0.8); ax[0].text(2.08, 98.35, 'selected\nepoch 2', fontsize=7); ax[0].set_xlabel('Epoch (0 = unadapted)'); ax[0].set_ylabel('Validation (%)'); ax[0].set_ylim(98, 100.25); ax[0].legend(fontsize=7, loc='lower right')
ax[1].plot(hst.epoch, hst.val_threshold_youden, 'k-o', ms=4); ax[1].axvline(2, color='#777', lw=0.8)
for e, t in zip(hst.epoch, hst.val_threshold_youden): ax[1].text(e, t + 0.012, f'{t:.3f}', ha='center', fontsize=6.5)
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Youden threshold (validation)'); ax[1].set_ylim(0.3, 0.7)
fig.tight_layout(); save(fig, 'fig_exp3_validation_history')
# ---------- Fig: Exp 4 recall by style & species with CIs ----------
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={'width_ratios': [1, 1.9]}, sharey=True)
for axi, key, order in ((ax[0], 'style', ['camera_night_ir', 'community', 'camera_day']), (ax[1], 'category', sorted(p4.category.unique()))):
    x = np.arange(len(order))
    for j, (suf, col, h, lb) in enumerate((('original', C_O, '////', 'Original'), ('adapted', C_A, '', 'Adapted'))):
        v, lo, hi = [], [], []
        for g in order:
            d = p4[p4[key] == g]; k = int((d[f'pred_{suf}'] == 'fake').sum()); n = len(d); l, u = wilson(k, n)
            v.append(100 * k / n); lo.append(max(0, 100 * (k / n - l))); hi.append(max(0, 100 * (u - k / n)))
        axi.bar(x + (j - 0.5) * 0.38, v, 0.38, color=col, hatch=h, edgecolor='black', lw=0.5, label=lb, yerr=[lo, hi], capsize=2, error_kw={'lw': 0.6})
    nm = {'camera_night_ir': 'Night/IR', 'community': 'Community', 'camera_day': 'Camera day'}
    axi.set_xticks(x); axi.set_xticklabels([f"{nm.get(g, g.capitalize())}\n(n={int((p4[key] == g).sum())})" for g in order], fontsize=7.5)
ax[0].set_ylabel('FLUX fake recall (%)'); ax[0].set_ylim(60, 102); ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
fig.tight_layout(); save(fig, 'fig_exp4_recall_style_species')
# ---------- Fig: Exp 4 score distributions + paired scatter ----------
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={'width_ratios': [1.3, 1]})
bins = np.linspace(0, 1, 41)
ax[0].hist(p4.score_original, bins, color='white', edgecolor='black', lw=0.6, hatch='////', label='Original')
ax[0].hist(p4.score_adapted, bins, color='#3a3a3a', edgecolor='black', lw=0.4, alpha=0.7, label='Adapted')
ax[0].axvline(T_O, color='black', ls=':', lw=1); ax[0].axvline(T_A, color='black', ls='--', lw=1)
ax[0].text(T_O - 0.02, 0.93, 'orig. thr.', transform=ax[0].get_xaxis_transform(), fontsize=6.5, ha='right'); ax[0].text(T_A + 0.015, 0.93, 'adapted thr.', transform=ax[0].get_xaxis_transform(), fontsize=6.5)
ax[0].set_xlabel('Model score on FLUX fakes (n=326)'); ax[0].set_ylabel('Images'); ax[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 0.85))
mk = {'community': ('o', 'white'), 'camera_day': ('s', '#777777'), 'camera_night_ir': ('^', 'black')}
for s, (mm, fc) in mk.items():
    d = p4[p4['style'] == s]; ax[1].scatter(d.score_original, d.score_adapted, s=9, marker=mm, facecolors=fc, edgecolors='black', lw=0.4, label={'community': 'Community', 'camera_day': 'Camera day', 'camera_night_ir': 'Night/IR'}[s])
ax[1].axvline(T_O, color='black', ls=':', lw=0.8); ax[1].axhline(T_A, color='black', ls='--', lw=0.8); ax[1].plot([0, 1], [0, 1], color='#aaa', lw=0.6)
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].set_xlabel('Original score'); ax[1].set_ylabel('Adapted score'); ax[1].legend(fontsize=6.5, loc='lower right', handletextpad=0.2)
fig.tight_layout(); save(fig, 'fig_exp4_scores_paired')
# ---------- Fig: cross-experiment fake recall SDXL vs FLUX ----------
fig, ax = plt.subplots(figsize=(4.4, 2.5)); x = np.arange(2)
for j, (col, h, lb, ks) in enumerate(((C_O, '////', 'Original OmniAID v2', [(329, 330), (303, 326)]), (C_A, '', 'Adapted OmniAID v2', [(330, 330), (301, 326)]))):
    v = [100 * k / n for k, n in ks]; ci = [wilson(k, n) for k, n in ks]
    ax.bar(x + (j - 0.5) * 0.38, v, 0.38, color=col, hatch=h, edgecolor='black', lw=0.5, label=lb, yerr=[[max(0, vv - 100 * c[0]) for vv, c in zip(v, ci)], [max(0, 100 * c[1] - vv) for vv, c in zip(v, ci)]], capsize=2, error_kw={'lw': 0.6})
    for xi, vv in zip(x + (j - 0.5) * 0.38, v): ax.text(xi, 81, f'{vv:.1f}%', ha='center', rotation=90, fontsize=7, color='black', bbox=dict(fc='white', ec='none', pad=0.6))
ax.set_xticks(x); ax.set_xticklabels(['SDXL fakes\n(Dataset v1 TEST, n=330)', 'FLUX.1-schnell fakes\n(Exp. 4, n=326)']); ax.set_ylabel('Fake recall (%)'); ax.set_ylim(80, 101.5); ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
fig.tight_layout(); save(fig, 'fig_sdxl_vs_flux')
# ---------- Fig: FLUX QC acceptance by species x style ----------
gm = pd.read_csv(f'{E}/model_heldout_generator/metadata/generation_manifest.csv').merge(pd.read_csv(f'{E}/model_heldout_generator/metadata/qc_decisions.csv'), on='image_id')
fig, ax = plt.subplots(figsize=(6.4, 2.2)); sp = sorted(gm.category.unique()); x = np.arange(len(sp))
for j, (s, col, h, lb) in enumerate((('community', 'white', '....', 'Community'), ('camera_night_ir', '#1f1f1f', '', 'Night/IR'), ('camera_day', '#8c8c8c', '//', 'Camera day'))):
    v = [100 * (gm[(gm.category == c) & (gm['style'] == s)].qc_decision == 'ACCEPT').mean() for c in sp]
    ax.bar(x + (j - 1) * 0.27, v, 0.27, color=col, hatch=h, edgecolor='black', lw=0.5, label=lb)
ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in sp]); ax.set_ylabel('QC acceptance (%)'); ax.set_ylim(0, 105); ax.legend(ncol=3, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(); save(fig, 'fig_exp4_qc_acceptance')
