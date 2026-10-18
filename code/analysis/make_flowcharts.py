"""Flowcharts for the final report and supervisor pack. Grayscale; status encoded by border style + symbol:
   solid border + check = complete; dashed + arrow = next; dotted + circle = optional/future."""
import os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': ['Times New Roman', 'DejaVu Serif'], 'savefig.dpi': 300})
ST = {'done': ('-', 1.0, ''), 'next': ('--', 1.2, '→ '), 'opt': (':', 1.2, '○ '), 'none': ('-', 0.9, '')}
class F:
    def __init__(s, w, h): s.fig, s.ax = plt.subplots(figsize=(w, h)); s.ax.set_xlim(0, w); s.ax.set_ylim(0, h); s.ax.axis('off'); s.w, s.h = w, h
    def box(s, x, y, w, h, t, st='none', fc='white', fs=7.5, bold=False):
        ls, lw, pre = ST[st]
        s.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec='black', ls=ls, lw=lw))
        s.ax.text(x, y, pre + t, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', wrap=True); return (x, y, w, h)
    def dia(s, x, y, w, h, t, fs=7):
        s.ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], fc='#f0f0f0', ec='black', lw=0.9)); s.ax.text(x, y, t, ha='center', va='center', fontsize=fs); return (x, y, w, h)
    def data(s, x, y, w, h, t, fs=7.5):   # cylinder-like data store
        s.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.25', fc='#e6e6e6', ec='black', lw=1.0)); s.ax.text(x, y, t, ha='center', va='center', fontsize=fs); return (x, y, w, h)
    def band(s, x, y, w, h, t):
        s.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#fafafa', ec='#9a9a9a', lw=0.7, ls='-')); s.ax.text(x + 0.08, y + h - 0.12, t, ha='left', va='top', fontsize=7, style='italic', color='#333')
    def arr(s, a, b, lab='', side='v', ls='-', rad=0.0, fs=6.5):
        (x1, y1, w1, h1), (x2, y2, w2, h2) = a, b
        if side == 'v': p1, p2 = (x1, y1 - h1 / 2 if y2 < y1 else y1 + h1 / 2), (x2, y2 + h2 / 2 if y2 < y1 else y2 - h2 / 2)
        else: p1, p2 = (x1 + w1 / 2 if x2 > x1 else x1 - w1 / 2, y1), (x2 - w2 / 2 if x2 > x1 else x2 + w2 / 2, y2)
        s.ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=8, lw=0.8, color='black', ls=ls, connectionstyle=f'arc3,rad={rad}'))
        if lab: s.ax.text((p1[0] + p2[0]) / 2 + 0.05, (p1[1] + p2[1]) / 2, lab, fontsize=fs, ha='left', va='center', style='italic')
    def legend(s, x, y):
        s.ax.set_ylim(y - 0.15, s.h)
        for i, (k, t) in enumerate((('done', 'complete'), ('next', 'next'), ('opt', 'optional / future'))):
            ls, lw, pre = ST[k]; s.ax.add_patch(FancyBboxPatch((x + i * 1.55, y), 0.35, 0.18, boxstyle='round,pad=0.01', fc='white', ec='black', ls=ls, lw=lw)); s.ax.text(x + i * 1.55 + 0.45, y + 0.09, pre + t, fontsize=6.5, va='center')
    def save(s, n): s.fig.savefig(f'{OUT}/{n}.png', bbox_inches='tight', pad_inches=0.04); plt.close(s.fig); print('saved', n)

# 1. Overall research methodology / experimental progression
f = F(6.6, 8.3); X = 3.3; bw, bh = 3.3, 0.34; Y = [8.0 - i * 0.5 for i in range(16)]
steps = [('Research framing: synthetic wildlife and ecological image authenticity', 'done'), ('Critical literature review (two falsification search rounds)', 'done'),
         ('Research gap confirmed; aim and RQ1–RQ4 fixed', 'done'), ('Dataset design: Caltech + iNaturalist real, SDXL in three capture styles', 'done'),
         ('Automated + visual quality control; leakage-controlled split', 'done'), ('Dataset v1 frozen (4,676 images: 3,319 / 693 / 664)', 'done'),
         ('Model comparison → OmniAID v2 selected', 'done'), ('Exp. 1: off-the-shelf baseline (RQ1)', 'done'), ('Exp. 2: source / style / category analysis (RQ2)', 'done'),
         ('Exp. 3: parameter-efficient fine-tuning of OmniAID v2', 'done'), ('Exp. 3: frozen selection + one-time TEST (RQ3)', 'done'),
         ('Exp. 4: FLUX.1-schnell generation (held out from project adaptation)', 'done'), ('Exp. 4: pre-registered QC + dataset freeze (326 images)', 'done'),
         ('Exp. 4: one-time evaluation of both frozen detectors (RQ4)', 'done'), ('Final comparative synthesis and report', 'done'), ('Research prototype', 'next')]
bs = []
for (t, st), y in zip(steps, Y): bs.append(f.box(X, y, 5.2, bh, t, st, fc='#f2f2f2' if 'Exp.' in t else 'white', fs=7.3))
for a, b in zip(bs, bs[1:]): f.arr(a, b)
ob = f.box(X, Y[-1] - 0.5, 5.2, bh, 'Robustness testing (JPEG recompression / resizing)', 'opt'); f.arr(bs[-1], ob, ls=':')
f.legend(1.2, -0.75); f.save('flow_01_methodology_progression')

# 2. Dataset construction and QC
f = F(6.8, 5.6)
c = f.data(1.2, 5.2, 1.9, 0.5, 'Caltech Camera Traps\n1,190 candidates (LILA BC)'); i = f.data(3.4, 5.2, 1.9, 0.5, 'iNaturalist API\n1,190 candidates (CC0/CC-BY)'); s = f.data(5.6, 5.2, 1.9, 0.5, 'SDXL base 1.0\n2,345 generated (3 styles)')
m = f.box(3.4, 4.35, 5.8, 0.42, 'Manifest verification (every file ↔ record; no duplicate IDs)', 'done')
for a in (c, i, s): f.arr(a, m)
q = f.box(3.4, 3.65, 5.8, 0.48, 'Automated QC: decode check, minimum size, SHA-256 exact duplicates,\nperceptual-hash near duplicates (Hamming ≤ 4), darkest 5% flagged', 'done')
f.arr(m, q)
r = f.box(1.6, 2.8, 2.8, 0.52, 'Real images: flagged items reviewed\n(dark frames; near-duplicate sample)', 'done'); v = f.box(5.1, 2.8, 3.1, 0.52, 'SDXL: full visual review on 56 contact sheets\n2,139 reviewed; 206 blocked by Drive client', 'done')
f.arr(q, r); f.arr(q, v)
d = f.dia(3.4, 1.85, 2.8, 0.7, 'Decision per image')
f.arr(r, d); f.arr(v, d)
opts = [('KEEP', 0.7), ('GROUP_SAME_SPLIT\n(Caltech near-dups)', 2.05), ('NOT_REVIEWED\n(TRAIN only)', 3.4), ('NEEDS_MANUAL_\nREVIEW (removed)', 4.75), ('EXCLUDE\n(removed)', 6.1)]
ob = [f.box(x, 0.95, 1.2, 0.46, t, 'none', fs=6.5, fc='#e0e0e0' if 'removed' in t else 'white') for t, x in opts]
for o in ob: f.arr(d, o)
f.box(3.4, 0.25, 6.2, 0.34, 'Dataset v1 frozen: 2,365 real (1,190 Caltech + 1,175 iNaturalist) + 2,311 SDXL = 4,676', 'done', fc='#f2f2f2', bold=True)
f.save('flow_02_dataset_qc')

# 3. Leakage-controlled split
f = F(6.6, 4.4)
d = f.data(3.3, 4.0, 3.2, 0.45, 'Dataset v1 (4,676 images)')
rules = [(1.15, 'Caltech\nwhole camera locations\n(20) kept in one split'), (3.3, 'iNaturalist\nexact + near-duplicate\ngroups kept together'), (5.45, 'SDXL\n206 NOT_REVIEWED → TRAIN\nval/test = reviewed KEEP')]
rb = [f.box(x, 3.0, 1.95, 0.72, t, 'done', fs=7) for x, t in rules]
for r in rb: f.arr(d, r)
a = f.box(3.3, 2.05, 5.8, 0.4, 'Greedy whole-group assignment toward 70 / 15 / 15', 'done')
for r in rb: f.arr(r, a)
k = f.dia(3.3, 1.25, 3.4, 0.6, '10 integrity checks pass?')
f.arr(a, k)
f.ax.text(5.1, 1.35, 'no → revise', fontsize=6.5, style='italic')
tb = [f.data(x, 0.35, 1.8, 0.45, t, fs=7) for x, t in ((1.3, 'TRAIN 3,319\n1,668 real / 1,651 fake'), (3.3, 'VALIDATION 693\n363 / 330'), (5.3, 'TEST 664\n334 / 330'))]
for t in tb: f.arr(k, t)
f.ax.text(5.1, 1.05, 'yes; physical copy\nverified 4,676/4,676', fontsize=6.5, style='italic')
f.save('flow_03_split')

# 4. Split roles / experiment design
f = F(6.8, 4.2)
tr = f.data(1.0, 3.6, 1.6, 0.42, 'TRAIN 3,319'); va = f.data(1.0, 2.35, 1.6, 0.42, 'VALIDATION 693'); te = f.data(1.0, 1.1, 1.6, 0.42, 'TEST 664 (frozen)')
fl = f.data(1.0, 0.25, 1.6, 0.36, 'FLUX set 326 (frozen)')
e1 = f.box(3.7, 3.6, 3.0, 0.5, 'Exp. 3 adaptation: update 1.45 M parameters\n(experts, router, head)', 'done', fs=7)
e2 = f.box(3.7, 2.35, 3.0, 0.62, 'Decisions only: Youden threshold (Exp. 1)\nepoch selection + threshold (Exp. 3)\nreproduction checks (Exp. 4)', 'done', fs=7)
e3 = f.box(3.7, 1.1, 3.0, 0.5, 'One-time measurement: Exp. 1 and Exp. 3\n(TEST_EVALUATED.json lock)', 'done', fs=7)
e4 = f.box(3.7, 0.25, 3.0, 0.4, 'Exp. 4: both frozen detectors, once', 'done', fs=7)
for a, b in ((tr, e1), (va, e2), (te, e3), (fl, e4)): f.arr(a, b, side='h')
f.box(6.05, 1.75, 1.3, 2.6, 'No test-set\ntuning:\nthresholds and\ncheckpoints\nfrozen before\nany TEST or\nFLUX scoring', 'none', fc='#f2f2f2', fs=7)
f.save('flow_04_split_roles')

# 5. Exp 3 workflow
f = F(6.6, 5.2); X = 3.3
b1 = f.box(X, 4.85, 5.4, 0.4, 'Official OmniAID repo @ 40749406 + v2 checkpoint (SHA-256 91933892…)', 'done', fs=7)
b2 = f.dia(X, 4.15, 4.2, 0.62, 'Can official OmniAID_LoRA load the v2 checkpoint?', fs=6.8)
f.arr(b1, b2); f.ax.text(5.55, 4.2, 'no: separate\narchitecture', fontsize=6.5, style='italic')
b3 = f.box(X, 3.35, 5.8, 0.52, 'Parameter-efficient fine-tuning of OmniAID v2 (official main_finetune.py route):\nstrict load 2,804/2,804 keys; train rank-1 experts + router + head = 1,445,959 of 507,797,575 (0.285%)', 'done', fs=6.8)
f.arr(b2, b3)
b4 = f.box(X, 2.55, 5.8, 0.48, 'Batch 8 → CUDA out-of-memory; pre-planned fallback batch 4 × accumulation 8 (effective 32)\nAdamW lr 2.5e-5, cosine, label smoothing 0.1, AMP, seed 0, 5 epochs', 'done', fs=6.8)
f.arr(b3, b4)
b5 = f.box(X, 1.8, 5.8, 0.42, 'Validation after every epoch (693 images) → rule: max balanced accuracy → fake recall → earliest', 'done', fs=6.8)
f.arr(b4, b5)
b6 = f.box(X, 1.1, 5.8, 0.44, 'Epoch 2 frozen (adapted_epoch_1.pt, SHA 733ab455…); Youden threshold 0.584899;\nreload check max |Δ| = 1.1e-16', 'done', fs=6.8)
f.arr(b5, b6)
b7 = f.box(X, 0.35, 5.8, 0.4, 'One-time TEST (664) → TEST_EVALUATED.json lock → comparison with Exp. 1', 'done', fs=7, fc='#f2f2f2')
f.arr(b6, b7); f.save('flow_05_exp3')

# 6. Exp 4 workflow
f = F(6.8, 5.6); X = 2.9
a1 = f.box(X, 5.25, 5.4, 0.4, 'FLUX.1-schnell @ 741f7c3c (gated; held out from project adaptation)', 'done', fs=7)
a2 = f.box(X, 4.6, 5.4, 0.46, 'Pilot: 21 images (7 species × 3 styles), no detector; NF4 4-bit on T4;\n6.6 GiB VRAM, ~20.6 s/image, bit-identical seeds', 'done', fs=6.8)
a3 = f.box(X, 3.9, 5.4, 0.46, 'Predefined seeds: 7 × (24 community + 12 camera day + 12 night/IR) = 336;\nDataset v1 prompt logic; 4 steps, guidance 0, 768 × 768, JPEG 95', 'done', fs=6.8)
a4 = f.box(X, 3.2, 5.4, 0.42, 'Fixed QC rules R1–R8 + A1–A4; ambiguous → user decision; no detector used', 'done', fs=7)
a5 = f.dia(X, 2.45, 3.4, 0.62, 'Cell at quota?', fs=7)
a6 = f.box(6.0, 2.45, 1.5, 0.62, 'Next unused\nseed (cap:\n5 × quota)', 'done', fs=6.8)
for p, q in ((a1, a2), (a2, a3), (a3, a4), (a4, a5)): f.arr(p, q)
f.arr(a5, a6, side='h'); f.ax.text(4.75, 2.55, 'no', fontsize=6.5, style='italic'); f.ax.add_patch(FancyArrowPatch((6.0, 2.76), (5.6, 3.2), arrowstyle='-|>', mutation_scale=8, lw=0.8, connectionstyle='arc3,rad=0.3'))
a7 = f.box(X, 1.6, 5.4, 0.46, 'Freeze at 326 (653 generated; bobcat / opossum camera day capped at 60 attempts)\nEXP4_DATASET_FROZEN.json', 'done', fs=6.8)
f.arr(a5, a7, 'yes / cap')
a8 = f.box(X, 0.9, 5.4, 0.42, 'Fresh runtime: Exp. 3 environment; validation reproduced (max |Δ| 1.1e-16)', 'done', fs=7)
a9 = f.box(X, 0.25, 5.4, 0.4, 'Score once: original (thr 0.3600) and adapted (thr 0.5849) → EXP4_EVALUATED.json', 'done', fs=7, fc='#f2f2f2')
f.arr(a7, a8); f.arr(a8, a9); f.save('flow_06_exp4')

# 7. Prototype pipeline
f = F(6.6, 2.6)
u = f.box(0.75, 1.9, 1.3, 0.5, 'User uploads\nJPG / PNG', 'next', fs=7)
p = f.box(2.35, 1.9, 1.5, 0.5, 'Official v2 preprocessing\n336 × 336, CLIP norm.', 'next', fs=6.8)
o = f.box(4.3, 2.3, 1.9, 0.42, 'Original OmniAID v2\nthreshold 0.3600', 'next', fs=6.8)
a = f.box(4.3, 1.5, 1.9, 0.42, 'Wildlife-adapted v2\nthreshold 0.5849', 'next', fs=6.8)
r = f.box(3.3, 0.45, 6.2, 0.5, 'Output: preview, REAL / AI-GENERATED per model, model score, threshold, model version,\nresearch-prototype disclaimer (score is not a calibrated confidence)', 'next', fs=6.8)
f.arr(u, p, side='h'); f.arr(p, o, side='h'); f.arr(p, a, side='h'); f.arr(a, r)
f.save('flow_07_prototype')
