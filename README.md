# Detecting AI-Generated Wildlife Images

**Evaluating generalisation and domain adaptation of AI-generated image detectors in ecological wildlife imagery**

HIT401 Capstone Project, Charles Darwin University, Group 24 (2026). Supervisor: Charles Yeo.
Group members: Ashesh Luitel, Mahezabeen Momtarin Arpi, Christian Jay Guarino Borja, Shahriar Khandakar Seam.

This repository holds the code, data manifests, frozen per-image predictions and records behind our final report
(Appendix C). It is meant to let a reviewer check where the numbers in the report come from. It is not a ready-made
wildlife "fake detector", and the project does not claim to solve AI-generated wildlife detection.

---

## What we studied and why

Text-to-image models such as SDXL and FLUX can now produce realistic photographs of wild animals. If fake photographs
enter citizen-science platforms or ecological datasets, they can create false records of where animals live. Some
detectors already report very high accuracy, including on animal images, but they are mostly tested on general web
images. Real wildlife images are very different from each other: camera-trap frames are taken by fixed cameras, often at
night in infrared, while citizen-science photographs are taken with many different phones and cameras.

We therefore asked whether a released general-purpose detector works **consistently** on different kinds of real
wildlife images, and whether lightweight adaptation to wildlife images changes that, including on a second generator.

| RQ | Research question | Experiment |
|---|---|---|
| RQ1 | How accurately does a released general-purpose AI-generated-image detector, used without adaptation, distinguish real from AI-generated wildlife images? | Exp 1: off-the-shelf baseline |
| RQ2 | How does the detector's performance vary by real image source, synthetic style and wildlife species? | Exp 2: subgroup and error analysis of the Exp 1 predictions |
| RQ3 | Does parameter-efficient wildlife-domain adaptation reduce the detector's errors on the same test images while preserving detection of SDXL synthetic images? | Exp 3: parameter-efficient adaptation, before/after on the same frozen test set |
| RQ4 | After adaptation, does the detector still detect synthetic wildlife images from FLUX.1-schnell, a generator held out from project adaptation, as well as the original detector? | Exp 4: FLUX.1-schnell evaluation of both frozen detectors |

## What we did

**Dataset v1 (frozen 23 September 2026).** 4,676 images of seven North American mammals (bobcat, coyote, opossum,
rabbit, raccoon, skunk, squirrel), chosen because they occur in both real sources:

- 1,190 camera-trap images from **Caltech Camera Traps** (LILA BC, CDLA-Permissive);
- 1,175 citizen-science photographs from **iNaturalist** (CC0 and CC-BY only, licence and attribution recorded per image);
- 2,311 **SDXL** images in three styles that mirror the real sources: community photo, daytime camera trap and night/infrared camera trap.

Splits: train 3,319, validation 693, test 664 (334 real, 330 synthetic). To stop leakage, whole Caltech camera
locations were kept in one split (20 locations, none shared between splits), duplicate and near-duplicate iNaturalist
images were kept together, and 206 SDXL images that could not be visually reviewed (a Google Drive client problem) were
restricted to training, so every synthetic image in validation and test was visually reviewed.

**Detector.** OmniAID v2 (Guo et al., ICML 2026), a CLIP ViT-L/14 detector with semantic experts (including an Animal
expert) and a universal artifact expert. We chose it because it is designed to generalise across generators and image
content, its official checkpoint ran on our Colab T4, and its own results left the animal/ecological question open.

**Experiments.** Every threshold, epoch and image-acceptance decision was made on validation data or before scoring, and
test sets were scored once and then locked (`locks/`).

1. **Exp 1:** OmniAID v2 as released; threshold chosen on validation only with Youden's J and frozen before testing.
2. **Exp 2:** no new model run; the Exp 1 predictions broken down by real source, synthetic style and species.
3. **Exp 3:** parameter-efficient fine-tuning with the official OmniAID v2 training route, updating only the rank-1
   low-rank experts, router and output head (1,445,959 of 507,797,575 parameters, 0.285%). Epoch chosen on validation;
   new validation threshold; test scored once.
4. **Exp 4:** 326 FLUX.1-schnell images, generated with predefined seeds and the same prompt logic, accepted by fixed QC
   rules and frozen before either detector saw them; both frozen detectors scored once.

**Platform.** The *Wildlife Image Authenticity Analysis Platform* (`code/platform/`) runs both frozen detectors on one
image or a batch and shows each model score, threshold and decision. If the two detectors disagree, the image is marked
**Review Suggested**. It is a software artefact for screening, not a fifth experiment, and its labels are not proof.

## Results

All numbers below were re-derived from the files in this repository with
`python code/analysis/verify_from_repository.py` (October 2026). Model scores are raw detector outputs, not
probabilities or confidence values.

**Exp 1 baseline** (`results/exp1_baseline_and_exp2_subgroups/baseline_test_predictions.csv`, threshold 0.35996127)

| Measure | Value |
|---|---|
| Accuracy | 645/664 = **97.14%** (95% CI 95.57–98.16) |
| Fake (SDXL) recall | 329/330 = 99.70% |
| Real recall | 316/334 = 94.61% |
| Errors | 18 false positives, 1 false negative |

**Exp 2 subgroups** (same file; `subgroup_*.csv`): all 18 false positives were iNaturalist photographs. Real recall was
100% for Caltech (158/158) and **89.77%** for iNaturalist (158/176). Synthetic detection was 99.40–100% across the three
SDXL styles. This is a source-dependent difference; the experiment did not isolate its cause.

**Exp 3 adaptation** (`results/exp3_adaptation/`, threshold 0.5848991870880127, selected epoch 2 = file `adapted_epoch_1.pt`)

| Measure | Original → adapted |
|---|---|
| Accuracy | 97.14% → **99.85%** (663/664) |
| iNaturalist real recall | 89.77% → **99.43%** |
| SDXL fake recall | 99.70% → **100%** |
| Paired errors | 18 of the 19 baseline errors corrected, 0 new errors, 1 remaining (a real iNaturalist rabbit) |
| Exact McNemar test | p = 7.6 × 10⁻⁶ |

**Exp 4 FLUX.1-schnell** (`results/exp4_heldout_generator/`, 326 images, synthetic only)

| Measure | Original | Adapted |
|---|---|---|
| Fake recall | 303/326 = **92.94%** | 301/326 = **92.33%** |
| Paired comparison | both detected 293, both missed 15, missed only by original 8, only by adapted 10 | exact McNemar p = 0.81 |

Adaptation did not improve detection of FLUX images, and both detectors caught about seven percentage points fewer FLUX
than SDXL images. Wilson intervals and McNemar tests for all subgroups are in `results/statistics/`.

### What these results do and do not show

- In our data, the overall accuracy of the baseline hid uneven performance on real images from different sources.
- Adaptation fixed almost all of those errors **within** Dataset v1 (same sources, same generator); it does not show
  better behaviour on other camera-trap networks or citizen-science platforms.
- **FLUX.1-schnell was held out from project adaptation.** It is not a generator that OmniAID has never seen:
  OmniAID's Mirage-Train data include Flux.1-family images, and the training settings stored in the released v2
  checkpoint indicate Mirage-Train was used at least in its router stage. Exp 4 therefore tests whether **our**
  wildlife adaptation changed detection when the generator changed from SDXL to FLUX, compared with the original
  detector. It is not a zero-shot generalisation result.
- Seven species, two real sources, two generators and one adaptation run (seed 0) limit how far these results generalise.

## Repository structure

| Folder | Contents |
|---|---|
| `notebooks/` | `01_Dataset_Preparation.ipynb` (download, QC, SDXL generation, split, Exp 1); `Experiment_3_..._EXECUTED.ipynb` (executed Exp 3); `Experiment_4_HeldOut_Generator_FLUX.ipynb` (Exp 4 runner) |
| `code/exp4/` | FLUX generation, QC, freeze, evaluation and analysis scripts, Dataset v1 prompt logic, QC operational notes |
| `code/analysis/` | `compute_statistics.py` (Wilson intervals, exact McNemar), `make_data_figures.py`, `make_flowcharts.py`, `verify_from_repository.py` |
| `code/platform/` | Wildlife Image Authenticity Analysis Platform: app, inference, status rules, metadata, batch/export, 31 unit tests, Colab launcher, verification records, bundled verification images with attribution |
| `configs/` | Exp 3 adaptation config, recovered OmniAID v2 architecture config, v2 checkpoint training arguments, checkpoint compatibility record |
| `locks/` | `frozen_selection.json`, `TEST_EVALUATED.json`, `EXP4_DATASET_FROZEN.json`, `EXP4_EVALUATED.json` |
| `manifests/dataset_v1/` | `split_manifest.csv` (every image: source, species, style, camera location, QC decision, split), Caltech and iNaturalist candidate, QC and duplicate records, iNaturalist licence and attribution, freeze summary |
| `manifests/sdxl/` | SDXL generation manifest (prompt, negative prompt, seed, settings for all 2,345 generated images), technical QC, pilot manifests |
| `manifests/exp4_flux/` | FLUX generation manifest, automated QC, QC decisions, team decisions on ambiguous cases, corrections, frozen manifest |
| `results/` | Per-image predictions, metrics, subgroup tables, training history, environment records and logs for Exp 1–4; `statistics/` |
| `figures/` | Figures used in the report, generated from the evidence files or drawn as flowcharts |
| `docs/` | `reproducibility.md`, `project_notes.md` |

## Running and reproducing

See `docs/reproducibility.md` for the full procedure. In short:

- **Check the reported numbers (no GPU):** `pip install pandas` then `python code/analysis/verify_from_repository.py`.
- **Experiments 1, 3 and 4** ran in Google Colab on a Tesla T4 with the project Google Drive mounted. The notebooks and
  scripts need the images (not in this repository) and the external model files listed below. Exp 3 and Exp 4 refuse to
  re-score their test sets while the lock files exist, by design.
- **Platform:** open `code/platform/launcher/HIT401_Platform_Colab.ipynb` in Colab; see `code/platform/README.md`.

`requirements.txt` lists the package versions recorded in our runs.

## External data and models (not redistributed here)

| Item | Where to get it |
|---|---|
| OmniAID code | https://github.com/yunncheng/OmniAID, commit `40749406fbcd8893c11a160edf4a72a2d4dc7056` |
| OmniAID v2 checkpoint | Hugging Face `Yunncheng/OmniAID`, revision `279cae7398ac6636f46fc4668f755f11210b36bf`, file `ckpt/checkpoint_omniaid_v2.pth`, SHA-256 `919338928cab88cef1e6e4b85777749c74bdff6a1796fcd7e43a91422327f906` |
| CLIP ViT-L/14-336 | Hugging Face `openai/clip-vit-large-patch14-336` (loaded by the OmniAID code) |
| Caltech Camera Traps | LILA BC, https://lila.science/datasets/caltech-camera-traps (CDLA-Permissive); image IDs and locations in `split_manifest.csv` |
| iNaturalist photographs | iNaturalist API; observation ID, photo ID, original URL, licence and attribution for each image in `manifests/dataset_v1/inaturalist_candidate_manifest.csv` |
| SDXL | Hugging Face `stabilityai/stable-diffusion-xl-base-1.0` (CreativeML Open RAIL++-M); prompts and seeds in `manifests/sdxl/` |
| FLUX.1-schnell | Hugging Face `black-forest-labs/FLUX.1-schnell`, revision `741f7c3ce8b383c54771c7003378a50191e9efe9` (Apache-2.0, gated); prompts and seeds in `manifests/exp4_flux/` |
| Adapted parameter file | `adapted_epoch_1.pt`, SHA-256 `733ab455a3d1f677451c9368846eb44e36236d492571d7960d075da641a3cd66`. **Not published**, because the redistribution terms of the upstream OmniAID and CLIP weights are not explicit. Available from the project team for assessment, or reproducible with Exp 3. |
| Image collections | The 4,676 Dataset v1 images and the 653 FLUX images (326 frozen) are stored on the project Google Drive and are not published here. The manifests identify every image, and the synthetic images can be regenerated from their seeds. |

## Problems we met and how we handled them

- **Google Drive client returned partly downloaded files.** Some SDXL files looked corrupt; a retry showed all 2,345
  were valid. The same client later throttled visual review, so 206 unreviewed SDXL images were restricted to training,
  and the physical split copy was done inside Colab and checked file by file.
- **The planned LoRA route did not fit our checkpoint.** The official `OmniAID_LoRA` variant is a separate architecture
  that cannot load the v2 checkpoint, so we used OmniAID v2's own parameter-efficient fine-tuning route, which loaded
  the checkpoint strictly (2,804/2,804 tensors) and kept the adapted model directly comparable with the baseline.
- **T4 memory.** A batch-size-8 probe ran out of GPU memory before training; we used batch 4 with gradient accumulation
  over 8 batches (same effective batch of 32 and same learning rate).
- **FLUX camera-day prompts.** FLUX often turned "daytime trail-camera photo" into a flash-lit night scene or added
  timestamp text. We kept the prompts, accepted only clearly daylit images, and froze the set at 326 when two cells
  reached the 5× attempt cap instead of changing the rules after seeing failures.

## How the work was done, and AI assistance

The research questions, scope, experimental design, dataset rules, validation/test protections, decisions on ambiguous
QC cases, the interpretation of the results and the conclusions were decided and reviewed by the group. All results come
from executing the project code; the per-image outputs are in `results/`.

AI tools were used to assist the group, as declared in Appendix B of the final report. Claude (Anthropic), used through
Claude Code, helped with writing and debugging code and notebooks, managing Colab sessions, checking and summarising
outputs, checking references, and drafting documentation that I reviewed. It also made the
first-pass visual QC decisions on the FLUX images by applying the group's written rules R1–R8 (606 of 653 decisions,
logged with the reviewer `claude-code-visual` in `manifests/exp4_flux/qc_decisions.csv`); the 47 ambiguous cases were
decided by a team member (`qc_user_resolutions.csv`). No detector was used during QC, and no threshold, epoch or model
was chosen by an AI tool on test data. This README and the files in `docs/` were also prepared with AI assistance from
the project records and reviewed by me.

## Licensing

See `LICENSING.md`. In short: third-party data and models keep their own licences and are not redistributed except for
a few bundled verification images whose licences allow it (with attribution in `code/platform/verification_images/`).
The group has not yet chosen an open-source licence for its own code, so no `LICENSE` file is included.

## Citation and acknowledgements

If you refer to this work, please cite the final report (see `CITATION.cff`). This project builds directly on:

- OmniAID: Guo, Y. et al. (2026) *OmniAID: decoupling semantics and artifacts for universal AI-generated image detection in the wild*, ICML 2026 — https://github.com/yunncheng/OmniAID
- CLIP: Radford, A. et al. (2021) *Learning transferable visual models from natural language supervision*, ICML 2021
- Caltech Camera Traps: Beery, S., Van Horn, G. and Perona, P. (2018) *Recognition in terra incognita*, ECCV 2018 — via LILA BC
- iNaturalist: https://www.inaturalist.org, and Van Horn, G. et al. (2018) *The iNaturalist species classification and detection dataset*, CVPR 2018
- SDXL: Podell, D. et al. (2023) *SDXL: improving latent diffusion models for high-resolution image synthesis*
- FLUX.1-schnell: Black Forest Labs (2024), https://huggingface.co/black-forest-labs/FLUX.1-schnell
