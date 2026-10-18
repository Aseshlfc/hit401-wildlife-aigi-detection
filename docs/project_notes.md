# Project notes

A short factual summary of how the project developed, taken from the project records (file timestamps, run logs, lock
files and the final report). Dates are 2026.

| When | What happened | Record |
|---|---|---|
| August | Project proposal: train a wildlife-specific real/fake classifier (ResNet-50 or EfficientNet-B4), compare with a general-purpose detector, 87% accuracy target | Proposal |
| September (interim) | Interim report: 1,190 Caltech + 1,190 iNaturalist candidates in seven species, SDXL pilots, planned CNNDetection baseline | Interim report |
| After the interim | Supervisor feedback asked for an aim built on a research gap, clearer gaps from the literature, comparison tables and real experimental results. Re-reading Akbas et al. (2025), which already reached about 99.6% on mostly domestic animals with transfer-learned CNNs, showed that training another classifier would repeat an existing type of experiment. The project moved to evaluating and adapting a released general-purpose detector. | Final report, Section I |
| 21–22 Sep | Full SDXL generation (2,345 images); logged literature search to test the research gap; OmniAID paper read in full | SDXL manifest backups; `literature_search_log.csv` |
| 22–23 Sep | Quality control and split. The Google Drive desktop client returned partly downloaded files and later throttled visual review, so 206 SDXL images were left unreviewed and restricted to training. Location-disjoint split, ten integrity checks, physical copy inside Colab (4,676/4,676). | `dataset_freeze_summary.txt`, `split_manifest.csv` |
| 23 Sep | **Dataset v1 frozen.** OmniAID v2 selected after comparing candidate detectors. | `dataset_freeze_summary.txt` |
| 22 Sep 16:14 UTC (23 Sep ACST) | **Experiment 1** baseline run; first attempt failed on a checkpoint path, fixed without code changes | `environment_record.json` |
| 24 Sep | Pre-training checks found that `OmniAID_LoRA` (the route named in the experiment plan) is a separate architecture that cannot load the v2 checkpoint, so the official v2 fine-tuning route was used instead | Exp 3 notebook (terminology note); `checkpoint_compatibility.json` (strict load of v2) |
| 24 Sep | A batch-size-8 memory probe ran out of T4 memory before training; batch 4 × accumulation 8 used | `adaptation_config.json` |
| 24 Sep 09:02–12:13 UTC | **Experiment 3** training (5 epochs); epoch 2 selected on validation and frozen at 12:39:55 UTC; test scored once at 13:19:20 UTC | `training_log.txt`, `locks/` |
| 24–25 Sep | **Experiment 4** protocol fixed; FLUX.1-schnell pilot, main generation and replacement rounds (653 images), twelve QC rounds | `manifests/exp4_flux/` |
| 25 Sep (UTC) | FLUX set frozen at 326 images after two camera-day cells reached the attempt cap; both detectors evaluated once | `EXP4_DATASET_FROZEN.json`, `EXP4_EVALUATED.json` |
| 26 Sep | Post-hoc statistics (Wilson intervals, exact McNemar); single-image Prototype v1 verified on Colab | `results/statistics/`, `code/platform/verification/reference_v1/` |
| 29 Sep | Wildlife Image Authenticity Analysis Platform (Prototype v2) verified on a Colab T4 against saved research predictions, and the package tested on a fresh runtime without Drive | `code/platform/verification/` |
| October | Final report written and revised; this repository prepared | Final report |

## Decisions worth knowing about

- **Validation-only decisions.** Thresholds (Youden's J), the adaptation epoch and the FLUX acceptance rules were fixed
  before the relevant test data were scored, and lock files stop the test sets being scored twice.
- **Model score, not confidence.** OmniAID scores were not calibrated, so the report and the platform show the raw
  model score and each detector's threshold instead of a confidence percentage.
- **FLUX wording.** FLUX.1-schnell was held out from project adaptation. OmniAID's Mirage-Train data include Flux.1-family
  images, so Experiment 4 is not a test on a generator the detector has never seen.
- **No threshold change to hide errors.** The single remaining Exp 3 error (a real iNaturalist rabbit just above the
  adapted threshold) was left as it is.
- **Weights not published.** The adapted parameter file is kept private because the redistribution terms of the
  upstream OmniAID and CLIP weights are not explicit.
