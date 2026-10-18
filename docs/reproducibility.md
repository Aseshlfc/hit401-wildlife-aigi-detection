# Reproducibility

This document explains how the results in the final report were produced and which files a reviewer needs. The
values below come from the environment records, configuration files and lock files in this repository.

## 1. What is in this repository and what is not

| Category | Status |
|---|---|
| Code we wrote (notebooks, `code/exp4`, `code/analysis`, `code/platform`) | Included |
| Manifests, QC logs, split, licence/attribution records | Included (`manifests/`) |
| Frozen per-image predictions, metrics, training history, environment records, logs | Included (`results/`) |
| Freeze and lock files | Included (`locks/`) |
| Figures used in the report | Included (`figures/`) |
| OmniAID source code | External: cloned from GitHub at the pinned commit when the code runs |
| OmniAID v2 base checkpoint (3.24 GB) and CLIP weights | External: downloaded from Hugging Face and checked by SHA-256 |
| Adapted parameter file `adapted_epoch_1.pt` (about 6 MB) | Not redistributed (upstream weight terms not explicit); obtain from the team or reproduce with Exp 3 |
| Dataset v1 images (4,676) and FLUX images (653 generated, 326 frozen) | Stored on the project Google Drive (`HIT401_Wildlife_Deepfake/final/` and `exp4_heldout_generator/`), not published; identified by the manifests |
| A few verification images used by the platform self-check | Included in `code/platform/verification_images/` with attribution and licence text |

## 2. Environment

All GPU work ran in **Google Colab on one NVIDIA Tesla T4** (about 14.6 GiB usable memory).

| Use | Recorded versions (source) |
|---|---|
| Detection, adaptation, evaluation, platform | Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0, transformers 4.57.3, timm 1.0.10, huggingface_hub 0.36.0, CUDA 12.8 (`results/exp1_baseline_and_exp2_subgroups/environment_record.json`, `results/exp3_adaptation/environment_record.txt`); Gradio 5.50.0 for the platform |
| FLUX generation (separate runtime) | diffusers 0.40.0, accelerate 1.15.0, bitsandbytes 0.50.2, transformers 5.16.1 (generation environment records on the project Drive; summarised in the report) |
| Statistics and figures (local, no GPU) | pandas, NumPy, SciPy, scikit-learn, Matplotlib |

`requirements.txt` lists these packages.

## 3. Detector

- **Code:** https://github.com/yunncheng/OmniAID at commit `40749406fbcd8893c11a160edf4a72a2d4dc7056`, used unmodified.
- **Checkpoint:** Hugging Face `Yunncheng/OmniAID`, revision `279cae7398ac6636f46fc4668f755f11210b36bf`, file
  `ckpt/checkpoint_omniaid_v2.pth`, SHA-256 `919338928cab88cef1e6e4b85777749c74bdff6a1796fcd7e43a91422327f906`.
- **Backbone:** CLIP ViT-L/14 at 336 px (`openai/clip-vit-large-patch14-336`); six rank-1 low-rank experts, top-2
  routing, router hidden size 256 (read from the arguments stored in the checkpoint: `configs/v2_checkpoint_training_args.json`,
  `configs/config_omniaid_v2_recovered.json`).
- **Preprocessing:** RGB, resize to 336 × 336, CLIP mean/std normalisation; no augmentation.
- **Decision rule:** predicted AI-generated if model score ≥ threshold. The score is not calibrated and is not a probability.

## 4. Dataset v1

- **Real images:** Caltech Camera Traps from the LILA BC public bucket and iNaturalist through its API (CC0 and CC-BY
  only), 170 candidates per species per source. Records: `manifests/dataset_v1/caltech_candidate_selection.csv`,
  `inaturalist_candidate_manifest.csv` (observation ID, photo ID, URL, licence, attribution).
- **SDXL:** `stabilityai/stable-diffusion-xl-base-1.0`, fp16, 25 steps, guidance 6.5, 768 × 768, JPEG. 165 community +
  85 camera day + 85 camera night/IR per species (2,345 generated). Prompt, negative prompt and seed for every image:
  `manifests/sdxl/sdxl_full_generation_manifest.csv`. Pilots: `sdxl_pilot_manifest.csv`, `sdxl_pilot_v2_manifest.csv`.
- **Quality control:** full decode, minimum size, SHA-256 exact duplicates, perceptual-hash near duplicates (Hamming
  distance ≤ 4 within species), review of the darkest real images, visual review of SDXL images.
  Records: `authentic_image_qc_log.csv`, `caltech_qc_full.csv`, `inaturalist_qc_full.csv`, the duplicate files,
  `manifests/sdxl/sdxl_technical_qc_full.csv`, `sdxl_visual_qc_log.csv`, `dataset_freeze_summary.txt`.
- **Split rules (`split_manifest.csv`):**
  - whole Caltech camera locations in one split (20 locations; none shared);
  - iNaturalist exact and near-duplicate groups kept in one split;
  - the 206 SDXL images that could not be visually reviewed are in train only;
  - groups assigned greedily towards 70/15/15; ten integrity checks passed before the freeze.
- **Result:** 4,676 images; train 3,319, validation 693, test 664 (334 real, 330 synthetic). Frozen 23 September 2026.

## 5. Experiments

### Experiment 1 (notebook `01_Dataset_Preparation.ipynb`, final sections)
Score validation; choose the threshold that maximises Youden's J (TPR − FPR) on **validation only**:
`0.35996127128601074`; freeze; score the 664 test images once.
Output: `results/exp1_baseline_and_exp2_subgroups/baseline_test_predictions.csv`.

### Experiment 2
No model run. Subgroups of the Exp 1 predictions by source, style and species, and the list of errors:
`subgroup_source.csv`, `subgroup_style.csv`, `subgroup_category.csv`, `baseline_errors.csv`; before/after versions in
`results/exp3_adaptation/before_after_subgroups.csv`.

### Experiment 3 (notebook `Experiment_3_..._EXECUTED.ipynb`)
- Official route: `main_finetune.py --model OmniAID --pretrained_checkpoint`, `training_mode=standard`; strict load
  2,804/2,804 tensors. `OmniAID_LoRA` was not used because it cannot load the v2 checkpoint.
- Trainable: low-rank experts 1,180,224 + router 263,685 + output head 2,050 = **1,445,959 of 507,797,575 (0.285%)**.
- Training (`configs/adaptation_config.json`): seed 0; 5 epochs; batch 4 × gradient accumulation 8 = 32 (batch 8 ran
  out of memory in a probe); AdamW, weight decay 0; learning rate 2.5e-5 (2e-4 × 32 / 256); half-cycle cosine to 1e-6,
  no warm-up; label smoothing 0.1 plus the official orthogonality and router-balance terms; mixed precision.
  The loader drops the incomplete last batch: 829 batches of 4 per epoch, i.e. 103 complete optimiser steps.
- Selection rule (fixed in advance, validation only): highest balanced accuracy at each epoch's own Youden threshold;
  differences < 0.001 broken by higher fake recall, then earlier epoch. Validation reached 1.0 at epochs 2–5, so
  **epoch 2** was selected. File `adapted_epoch_1.pt` (zero-indexed name), SHA-256
  `733ab455a3d1f677451c9368846eb44e36236d492571d7960d075da641a3cd66`; threshold `0.5848991870880127`.
- Frozen at 2026-09-24 12:39:55 UTC (`locks/frozen_selection.json`); test scored once at 13:19:20 UTC
  (`locks/TEST_EVALUATED.json`); the notebook stops if this lock exists.
- Outputs: `results/exp3_adaptation/` (training history, validation predictions per epoch, test predictions, metrics).

### Experiment 4 (`code/exp4/`, runner notebook `Experiment_4_HeldOut_Generator_FLUX.ipynb`)
- FLUX.1-schnell revision `741f7c3ce8b383c54771c7003378a50191e9efe9` (gated; needs a Hugging Face read token in the
  `HF_TOKEN` Colab secret). Transformer and T5 encoder in 4-bit NF4 with CPU offload; 4 steps, guidance 0,
  768 × 768, JPEG quality 95.
- 336 predefined seeds (7 species × [24 community + 12 camera day + 12 night/IR]) with the Dataset v1 prompt logic
  (`code/exp4/dataset_v1_prompt_logic.json`).
- QC: automated checks plus visual rules R1–R8 (`code/exp4/QC_OPERATIONAL_NOTES.md`); a rejected image is replaced by
  the next unused seed, with a cap of 5× each cell's quota. No detector was run during QC. First-pass visual decisions
  were made with an AI assistant applying the written rules; ambiguous cases were decided by a team member
  (`manifests/exp4_flux/qc_decisions.csv`, `qc_user_resolutions.csv`).
- 653 generated, 326 accepted; frozen in `locks/EXP4_DATASET_FROZEN.json`. Both frozen detectors first re-scored the 693
  validation images (max difference 1.1e-16), then scored the 326 images once (`locks/EXP4_EVALUATED.json`).
- Outputs: `results/exp4_heldout_generator/`. FLUX is held out from **project adaptation** only; Flux.1-family images are
  part of OmniAID's own training data, so this is not a zero-shot test.

Order of commands: `python exp4_generate.py pilot|main|replace`, `python exp4_qc.py auto|record|resolve`,
`python exp4_freeze.py`, `python exp4_evaluate.py` (stops if `EXP4_EVALUATED.json` exists), `python exp4_analyze.py`.
`exp4_common.py` expects the project Drive at `/content/drive/MyDrive/HIT401_Wildlife_Deepfake`.

## 6. Statistics and figures

- `code/analysis/verify_from_repository.py` re-derives the headline numbers from this repository (no GPU).
- `code/analysis/compute_statistics.py` computes Wilson 95% intervals and exact two-sided McNemar tests (binomial test
  on the discordant pairs) from the frozen predictions; its outputs are `results/statistics/wilson_confidence_intervals.csv`
  and `paired_tests_and_summary.json`.
- `code/analysis/make_data_figures.py` and `make_flowcharts.py` produced the figures in `figures/`.
- These two scripts were run against the project's evidence folder, whose layout differs from this repository
  (`model_baseline/`, `model_adaptation/`, `model_heldout_generator/evaluation/`, `model_heldout_generator/metadata/`).
  Set `HIT401_EVIDENCE` to a folder with that layout to re-run them. The evidence files themselves are the same files
  that are in `results/` and `manifests/exp4_flux/`.
- None of these statistics was used to make any decision; they were computed after the experiments were locked.

## 7. Changes made when preparing this repository

The repository was assembled by copying files from the project workspace in October 2026. The only changes were:

- notebooks: saved progress-bar widget state was removed, and two display-only outputs showing a single dataset image
  were removed from `01_Dataset_Preparation.ipynb` (cells 25 and 96); no code cell and no text output was changed;
- `code/analysis`: the evidence folder can be set with `HIT401_EVIDENCE` (same default as before), and one unused line
  containing a personal Google Drive path was deleted from `make_data_figures.py`;
- `code/analysis/build_traceability.py` was left out because it checked an older draft of the report;
- `verify_from_repository.py`, this documentation, `LICENSING.md` and `CITATION.cff` were added.

No experiment was re-run and no result file was edited. `SHA256SUMS.txt` lists the hash of every file.
